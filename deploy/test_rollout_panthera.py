"""Hardware-free regression tests for the rollout's command boundary.

Run: conda run -n panthera python -m unittest discover -s deploy -p 'test_*.py'
"""
import queue
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rollout_panthera import (Cameras, FrameRecorder, H264VideoWriter, JointMapping,
                             RobotControl, guard_target, run)


def arguments():
    return SimpleNamespace(joint_signs=[1, -1, 1, 1, -1, 1],
                           joint_offsets=[0, .2, 0, 0, .1, 0],
                           gripper_closed=0., gripper_open=1.6,
                           max_joint_step=.04, command_velocity=.3, control_hz=30,
                           max_target_jump=.75, max_tracking_error=.35,
                           max_gripper_step=.05, feedback_timeout_s=.05, motor_timeout_ms=500,
                           mode='dry-run')


class MappingAndLimits(unittest.TestCase):
    def setUp(self):
        self.args = arguments()
        self.lower, self.upper = np.full(6, -2.), np.full(6, 2.)

    def guard(self, goal, previous=None, measured=None):
        return guard_target(goal, np.zeros(7) if previous is None else previous,
                            np.zeros(7) if measured is None else measured,
                            self.args, self.lower, self.upper)

    def test_mapping_roundtrip_with_offsets_and_reversed_joints(self):
        mapping = JointMapping(self.args)
        raw = np.array([.1, .3, .4, -.2, .6, .1, .8])
        np.testing.assert_allclose(mapping.to_sdk(mapping.to_policy(raw)), raw)
        self.assertAlmostEqual(mapping.to_policy(raw)[-1], .5)

    def test_velocity_and_gripper_increment_limits(self):
        sent = self.guard(np.array([.2] * 6 + [1.6]))
        np.testing.assert_allclose(sent[:6], .01)
        self.assertAlmostEqual(sent[-1], .08)

    def test_rejects_nonfinite_wrong_width_joint_limits_and_large_jump(self):
        for goal in ([0] * 6, [np.nan] * 7, [np.inf] * 7,
                     [2.1] + [0] * 6, [.8] + [0] * 6):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                self.guard(goal)

    def test_rejects_stalled_robot_and_next_step_excess(self):
        previous = np.array([.36] + [0] * 6)
        with self.assertRaisesRegex(ValueError, 'not following'):
            self.guard(previous, previous=previous)
        previous[0] = .349
        with self.assertRaisesRegex(ValueError, 'Next command'):
            self.guard(np.array([.5] + [0] * 6), previous=previous)

    def test_calibration_validation(self):
        self.args.joint_signs[0] = 0
        with self.assertRaises(ValueError):
            JointMapping(self.args)


class FeedbackAndThread(unittest.TestCase):
    def controller(self):
        controller = RobotControl.__new__(RobotControl)
        controller.args = arguments()
        controller.mapping = JointMapping(controller.args)
        controller.robot = Mock()
        controller.robot.motor_count = 6
        controller.robot.joint_limits = dict(lower=np.full(6, -2.), upper=np.full(6, 2.))
        states = [SimpleNamespace(time=1., fault=0, position=0.) for _ in range(7)]
        controller.robot.get_current_state.return_value = states[:6]
        controller.robot.get_current_state_gripper.return_value = states[6]
        controller.last_stamps = np.zeros(7)
        controller.last_updates = np.full(7, time.monotonic())
        return controller, states

    def test_stale_feedback_is_detected_per_motor(self):
        controller, states = self.controller()
        np.testing.assert_allclose(controller.feedback(), np.zeros(7))
        controller.last_updates[3] -= 1
        for index, state in enumerate(states):
            if index != 3:
                state.time += 1
        with self.assertRaises(TimeoutError):
            controller.feedback()

    def test_fault_and_uninitialized_feedback(self):
        controller, states = self.controller()
        states[0].fault = 1
        with self.assertRaisesRegex(RuntimeError, 'fault'):
            controller.feedback()
        states[0].fault, states[0].time = 0, 0
        self.assertIsNone(controller.feedback())

    def test_dry_run_never_sends_policy_targets(self):
        controller, _ = self.controller()
        controller.stop_event, controller.ready = threading.Event(), threading.Event()
        controller.lock = threading.Lock()
        controller.last_target, controller.error = None, None
        controller.commands = queue.Queue()
        controller.commands.put((np.array([.1] * 6 + [.5]), threading.Event()))
        with patch.object(controller.stop_event, 'wait', side_effect=lambda _: controller.stop_event.set()):
            controller._run()
        self.assertIsNone(controller.error)
        controller.robot.Joint_Pos_Vel.assert_not_called()
        controller.robot.gripper_control.assert_not_called()
        controller.robot.set_timeout.assert_not_called()
        controller.robot.set_stop.assert_not_called()

    def test_control_failure_stops_robot(self):
        controller, states = self.controller()
        controller.args.mode = 'rollout'
        controller.stop_event, controller.ready = threading.Event(), threading.Event()
        controller.error = None
        states[2].fault = 3
        controller._run()
        self.assertIsInstance(controller.error, RuntimeError)
        controller.robot.Joint_Pos_Vel.assert_not_called()
        controller.robot.set_stop.assert_called_once()

    def test_rollout_velocity_tracks_sent_targets_and_is_zero_while_holding(self):
        controller, _ = self.controller()
        controller.args.mode = 'rollout'
        controller.args.max_torque = [21, 36, 36, 21, 10, 10]
        controller.args.gripper_velocity = controller.args.gripper_torque = .5
        controller.stop_event, controller.ready = threading.Event(), threading.Event()
        controller.lock = threading.Lock()
        controller.last_target, controller.error = None, None
        controller.commands = queue.Queue()
        goal = np.array([.003, -.004, 0., .2, -.2, .001, 0.])
        acknowledgements = [threading.Event(), threading.Event()]

        def next_tick(_):
            tick = controller.robot.Joint_Pos_Vel.call_count
            if tick == 1:  # Initial measured-pose hold, then a mixed-direction move.
                controller.commands.put((controller.mapping.to_policy(goal), acknowledgements[0]))
            elif tick == 3:  # Tick 3 repeats the target despite nonzero encoder error.
                controller.commands.put((controller.mapping.to_policy(np.zeros(7)), acknowledgements[1]))
            elif tick == 4:
                controller.stop_event.set()

        with patch.object(controller.stop_event, 'wait', side_effect=next_tick):
            controller._run()

        self.assertIsNone(controller.error)
        calls = controller.robot.Joint_Pos_Vel.call_args_list
        self.assertEqual(len(calls), 4)
        sent = [.003, -.004, 0., .01, -.01, .001]
        velocity = np.array([.09, -.12, 0., .3, -.3, .03])
        for call, position, speed in zip(calls, [np.zeros(6), sent, sent, np.zeros(6)],
                                        [np.zeros(6), velocity, np.zeros(6), -velocity]):
            np.testing.assert_allclose(call.args[0], position, atol=1e-12)
            np.testing.assert_allclose(call.args[1], speed, atol=1e-12)
        self.assertTrue(all(event.is_set() for event in acknowledgements))
        controller.robot.set_stop.assert_called_once()


def camera_args(**overrides):
    values = dict(head_serial='head', wrist_serial='wrist', camera_width=8,
                  camera_height=6, camera_fps=30, camera_timeout_ms=100,
                  camera_warmup_frames=2, recording_queue_size=128,
                  recording_flush_timeout_s=5, recording_video_crf=23)
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeFrame:
    def __init__(self, number):
        self.number = number

    def get_frame_number(self):
        return self.number

    def get_timestamp(self):
        return self.number * 1000 / 30

    def get_frame_timestamp_domain(self):
        return 'hardware_clock'

    def get_data(self):
        return np.tile(np.array([self.number % 256, 17, 33], dtype=np.uint8), (6, 8, 1))


class FakeRealSense:
    stream = SimpleNamespace(color=1)
    format = SimpleNamespace(rgb8=1)

    def __init__(self, fail_start=None, gap=False):
        self.pipelines = []
        self.fail_start, self.gap = fail_start, gap

    def config(self):
        config = SimpleNamespace(serial=None, enable_stream=lambda *a: None)
        config.enable_device = lambda serial: setattr(config, 'serial', serial)
        return config

    def pipeline(self):
        sdk = self
        class Pipeline:
            def __init__(self):
                self.stopped = threading.Event()
                self.number = self.stop_calls = 0
                self.started = False

            def start(self, config):
                if config.serial == sdk.fail_start:
                    raise RuntimeError('injected camera start failure')
                self.started = True

            def wait_for_frames(self, timeout):
                if self.stopped.wait(0.005):
                    raise RuntimeError('pipeline stopped')
                self.number += 1
                if sdk.gap and self.number == 3:
                    self.number += 1
                frame = FakeFrame(self.number)
                return SimpleNamespace(get_color_frame=lambda: frame)

            def stop(self):
                self.stop_calls += 1
                self.stopped.set()
                if not self.started:
                    raise RuntimeError('pipeline never started')
        pipeline = Pipeline()
        self.pipelines.append(pipeline)
        return pipeline


class CameraRecording(unittest.TestCase):
    def wait_until(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('Timed out waiting for capture')
            time.sleep(0.005)

    def metadata(self, number):
        return dict(sequence=number, frame_number=number + 1,
                    missing_before=0, file='video.mp4', video_frame_index=number,
                    video_timestamp_s=number / 30)

    def test_continuous_capture_h264_frame_alignment_and_rgb_inputs(self):
        import cv2
        rs = FakeRealSense()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, pyrealsense2=rs):
            cameras = Cameras(camera_args(), directory)
            try:
                self.wait_until(lambda: min(cameras.counts.values()) >= 8)
                images, selected = cameras.read(with_metadata=True)
                for key in images:
                    np.testing.assert_array_equal(images[key], FakeFrame(selected[key]['frame_number']).get_data())
                selected_counts = cameras.counts.copy()
                # No calls to read(): acquisition still advances during inference/actions.
                self.wait_until(lambda: all(cameras.counts[k] >= selected_counts[k] + 8
                                            for k in selected_counts))
            finally:
                cameras.close()
            cameras.close()
            self.assertTrue(all(not thread.is_alive() for thread in cameras.threads))
            self.assertFalse(cameras.recorder.thread.is_alive())
            summary = json.loads((Path(directory) / 'camera_frames/summary.json').read_text())
            self.assertTrue(summary['complete'])
            for key in FrameRecorder.keys:
                folder = Path(directory) / 'camera_frames' / key
                rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), summary['cameras'][key]['received'])
                self.assertFalse(list(folder.glob('*.png')))
                probe = json.loads(subprocess.check_output([
                    'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name,pix_fmt,nb_read_frames,r_frame_rate',
                    '-of', 'json', str(folder / 'video.mp4')], text=True))['streams'][0]
                self.assertEqual(probe['codec_name'], 'h264')
                self.assertEqual(probe['pix_fmt'], 'yuv420p')
                self.assertEqual(probe['r_frame_rate'], '30/1')
                self.assertEqual(int(probe['nb_read_frames']), len(rows))
                self.assertEqual([r['video_frame_index'] for r in rows], list(range(len(rows))))
                capture = cv2.VideoCapture(str(folder / 'video.mp4'))
                self.assertEqual([row['frame_number'] for row in rows], list(range(1, len(rows) + 1)))
                for row in rows:
                    ok, bgr = capture.read()
                    self.assertTrue(ok)
                    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    np.testing.assert_allclose(image, FakeFrame(row['frame_number']).get_data(), atol=6)
                    self.assertEqual(row['file'], 'video.mp4')
                    self.assertEqual(row['video_timestamp_s'], row['video_frame_index'] / 30)
                    self.assertEqual(row['device_timestamp_ms'], row['frame_number'] * 1000 / 30)
                    self.assertEqual(row['timestamp_domain'], 'hardware_clock')
                    self.assertGreater(row['host_monotonic_ns'], 0)
                    self.assertGreater(row['host_time_ns'], 0)
                self.assertFalse(capture.read()[0])
                capture.release()
            self.assertTrue(all(p.stop_calls == 1 for p in rs.pipelines))

    def test_disabled_recording_still_reads_current_frames(self):
        with patch.dict(sys.modules, pyrealsense2=FakeRealSense()):
            cameras = Cameras(camera_args())
            try:
                self.assertIsNone(cameras.recorder)
                self.assertEqual(set(cameras.read()), set(FrameRecorder.keys))
            finally:
                cameras.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                cameras.read()

    def test_frame_gaps_are_recorded_as_incomplete(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, pyrealsense2=FakeRealSense(gap=True)):
            cameras = Cameras(camera_args(), directory)
            self.wait_until(lambda: min(cameras.counts.values()) >= 5)
            cameras.close()
            summary = json.loads((Path(directory) / 'camera_frames/summary.json').read_text())
            self.assertFalse(summary['complete'])
            for count in summary['cameras'].values():
                self.assertEqual(count['missing_frame_numbers'], 1)
                self.assertEqual(count['received'], count['saved'])

    def test_partial_start_failure_releases_first_camera_and_recorder(self):
        rs = FakeRealSense(fail_start='wrist')
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, pyrealsense2=rs):
            with self.assertRaisesRegex(RuntimeError, 'start failure'):
                Cameras(camera_args(), directory)
            self.assertTrue(all(p.stopped.is_set() for p in rs.pipelines))
            self.assertFalse(any(t.name.startswith('capture-') or t.name == 'camera-recorder'
                                 for t in threading.enumerate()))

    def test_close_drains_every_queued_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FrameRecorder(directory, camera_args())
            for number in range(50):
                recorder.enqueue('cam_high', FakeFrame(number).get_data(), self.metadata(number))
            recorder.close()
            recorder.close()
            summary = json.loads((Path(directory) / 'camera_frames/summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['cameras']['cam_high']['saved'], 50)

    def test_full_queue_fails_explicitly(self):
        release = threading.Event()
        with tempfile.TemporaryDirectory() as directory, patch.object(FrameRecorder, '_write', lambda _: release.wait(2)):
            recorder = FrameRecorder(directory, camera_args(recording_queue_size=1))
            try:
                recorder.enqueue('cam_high', FakeFrame(1).get_data(), self.metadata(1))
                with self.assertRaisesRegex(RuntimeError, 'queue is full'):
                    recorder.enqueue('cam_high', FakeFrame(2).get_data(), self.metadata(2))
            finally:
                release.set()
                with self.assertRaisesRegex(RuntimeError, 'queue is full'):
                    recorder.close()
            summary = json.loads((Path(directory) / 'camera_frames/summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['cameras']['cam_high']['received'], 2)

    def test_disk_write_failure_propagates_to_health_and_summary(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                H264VideoWriter, 'write', side_effect=OSError('Failed to write video')):
            recorder = FrameRecorder(directory, camera_args())
            recorder.enqueue('cam_high', FakeFrame(1).get_data(), self.metadata(0))
            self.wait_until(lambda: recorder.error is not None)
            with self.assertRaisesRegex(RuntimeError, 'Failed to write'):
                recorder.health()
            with self.assertRaisesRegex(RuntimeError, 'Failed to write'):
                recorder.close()
            summary = json.loads((Path(directory) / 'camera_frames/summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['cameras']['cam_high']['saved'], 0)

    def test_encoder_failure_during_finalization_marks_recording_incomplete(self):
        original_close = H264VideoWriter.close
        def fail_close(writer):
            original_close(writer)
            raise OSError('injected finalization failure')
        with tempfile.TemporaryDirectory() as directory, patch.object(H264VideoWriter, 'close', fail_close):
            recorder = FrameRecorder(directory, camera_args())
            recorder.enqueue('cam_high', FakeFrame(1).get_data(), self.metadata(0))
            with self.assertRaisesRegex(RuntimeError, 'finalization failure'):
                recorder.close()
            summary = json.loads((Path(directory) / 'camera_frames/summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertIsNotNone(recorder.writers['cam_high'].proc.poll())

    def test_flush_timeout_kills_encoder_and_marks_recording_incomplete(self):
        entered, release = threading.Event(), threading.Event()
        original_abort = H264VideoWriter.abort
        def stall(writer, rgb):
            entered.set()
            release.wait(2)
        def abort(writer):
            original_abort(writer)
            release.set()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(H264VideoWriter, 'write', stall), \
                patch.object(H264VideoWriter, 'abort', abort):
            recorder = FrameRecorder(directory, camera_args(recording_flush_timeout_s=.1))
            recorder.enqueue('cam_high', FakeFrame(1).get_data(), self.metadata(0))
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(RuntimeError, 'flush timed out'):
                recorder.close()
            self.assertFalse(recorder.thread.is_alive())
            self.assertIsNotNone(recorder.writers['cam_high'].proc.poll())
            summary = json.loads((Path(directory) / 'camera_frames/summary.json').read_text())
            self.assertFalse(summary['complete'])

    def test_missing_ffmpeg_fails_before_recording_starts(self):
        with tempfile.TemporaryDirectory() as directory, patch('rollout_panthera.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'requires ffmpeg'):
                FrameRecorder(directory, camera_args())

    def test_run_records_during_inference_and_flushes_on_interrupt_or_step_limit(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt), tempfile.TemporaryDirectory() as directory:
                args = camera_args(log_root=directory, mode='dry-run', max_steps=2,
                                   execute_steps=2, save_observations=1,
                                   feedback_timeout_s=.5, load_timeout_s=1,
                                   state_source='previous-target')
                robot, policy = Mock(), Mock()
                robot.snapshot.return_value = (np.zeros(7), np.zeros(7))
                policy.receive.return_value = {'ready': {}}
                def predict(obs, health):
                    for _ in range(10):
                        health()
                        time.sleep(.01)
                    if interrupt:
                        raise KeyboardInterrupt('injected Ctrl+C')
                    return dict(actions=np.zeros((50, 7)).tolist(), elapsed_s=.1)
                policy.predict.side_effect = predict
                with patch.dict(sys.modules, pyrealsense2=FakeRealSense()), \
                        patch('rollout_panthera.preflight'), \
                        patch('rollout_panthera.PolicyProcess', return_value=policy), \
                        patch('rollout_panthera.RobotControl', return_value=robot):
                    if interrupt:
                        with self.assertRaises(KeyboardInterrupt):
                            run(args)
                    else:
                        run(args)
                robot.close.assert_called_once()
                policy.close.assert_called_once()
                folder, = Path(directory).iterdir()
                summary = json.loads((folder / 'camera_frames/summary.json').read_text())
                self.assertTrue(summary['complete'])
                self.assertTrue(all(c['saved'] >= 10 for c in summary['cameras'].values()))
                self.assertTrue((folder / 'observation/00000.npz').is_file())
                if not interrupt:
                    row = json.loads((folder / 'rollout.jsonl').read_text())
                    for key, meta in row['camera_frames'].items():
                        self.assertTrue((folder / 'camera_frames' / key / meta['file']).exists())
                        rows = [json.loads(line) for line in
                                (folder / 'camera_frames' / key / 'frames.jsonl').read_text().splitlines()]
                        self.assertEqual(rows[meta['video_frame_index']], meta)
                        with np.load(folder / 'observation/00000.npz') as data:
                            np.testing.assert_array_equal(data[key], FakeFrame(meta['frame_number']).get_data())


if __name__ == '__main__':
    unittest.main()
