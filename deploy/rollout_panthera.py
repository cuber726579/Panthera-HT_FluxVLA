#!/usr/bin/env python3
"""Panthera SDK rollout with a persistent FluxVLA subprocess. See the .sh file.

The robot process needs only the existing panthera environment. Observations
travel through a private temporary NPZ file; responses use a dedicated stdout
pipe. No ROS, network listener, or extra IPC dependency is required.
"""

import argparse
import contextlib
import copy
import glob
import json
import os
from pathlib import Path
import queue
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np


def vector(value, size, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected {size} finite numbers, got {value}')
    return value


class JointMapping:
    def __init__(self, args):
        self.signs = vector(args.joint_signs, 6, 'joint signs')
        self.offsets = vector(args.joint_offsets, 6, 'joint offsets')
        if not np.isin(self.signs, [-1, 1]).all():
            raise ValueError('Joint signs must be -1 or +1')
        self.closed, self.open = args.gripper_closed, args.gripper_open
        if not np.isfinite([self.closed, self.open]).all() or self.closed == self.open:
            raise ValueError('Gripper endpoints must be finite and distinct')

    def to_policy(self, raw):
        raw = vector(raw, 7, 'SDK state')
        return np.r_[self.signs * (raw[:6] - self.offsets),
                     np.clip((raw[6] - self.closed) / (self.open - self.closed), 0, 1)]

    def to_sdk(self, action):
        action = vector(action, 7, 'policy action')
        return np.r_[self.signs * action[:6] + self.offsets,
                     self.closed + np.clip(action[6], 0, 1) * (self.open - self.closed)]


def guard_target(desired, previous, measured, args, lower, upper):
    """Validate before sending, then bound target increments in SDK coordinates."""
    desired = vector(desired, 7, 'target')
    previous = vector(previous, 7, 'previous target')
    measured = vector(measured, 7, 'feedback')
    if np.any(desired[:6] < lower) or np.any(desired[:6] > upper):
        raise ValueError(f'Policy target outside SDK joint limits: {desired[:6]}')
    if np.max(np.abs(desired[:6] - previous[:6])) > args.max_target_jump:
        raise ValueError(f'Policy target jump exceeds {args.max_target_jump} rad')
    if np.max(np.abs(previous[:6] - measured[:6])) > args.max_tracking_error:
        raise ValueError('Robot is not following the previous command')
    limit = min(args.max_joint_step, args.command_velocity / args.control_hz)
    result = previous.copy()
    result[:6] += np.clip(desired[:6] - previous[:6], -limit, limit)
    if np.max(np.abs(result[:6] - measured[:6])) > args.max_tracking_error:
        raise ValueError('Next command would exceed tracking-error limit')
    grip_limit = args.max_gripper_step * abs(args.gripper_open - args.gripper_closed)
    result[6] += np.clip(desired[6] - previous[6], -grip_limit, grip_limit)
    return result


class Policy:
    """Lives entirely in the fluxvla interpreter, never opens hardware."""
    def __init__(self, args, load_model):
        import torch
        from mmengine import Config
        from safetensors import safe_open
        from fluxvla.engines import build_dataset_from_cfg, build_transform_from_cfg
        from fluxvla.engines.utils.torch_utils import configure_inference_attention_defaults

        configure_inference_attention_defaults()
        if not torch.cuda.is_available():
            raise RuntimeError('FluxVLA policy requires a CUDA device')
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        self.args, self.torch = args, torch
        cfg = Config.fromfile(args.model_config)
        data_cfg = copy.deepcopy(cfg.inference.dataset)
        data_cfg.update(norm_stats=args.statistics, model_path=args.tokenizer_root)
        for transform in data_cfg['transforms']:
            if transform['type'] == 'ProcessPrompts':
                transform['model_path'] = args.tokenizer_root
        self.dataset = build_dataset_from_cfg(data_cfg)
        denorm_cfg = copy.deepcopy(cfg.inference.denormalize_action)
        denorm_cfg['norm_stats'] = args.statistics
        self.denormalize = build_transform_from_cfg(denorm_cfg)
        with safe_open(args.checkpoint, framework='pt', device='cpu') as f:
            tensor_count = len(list(f.keys()))
        self.info = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                         checkpoint=args.checkpoint, tensor_count=tensor_count,
                         param_dtype=args.param_dtype, loaded=False)
        if load_model:
            from accelerate import init_empty_weights
            from safetensors.torch import load_file
            from fluxvla.engines import build_vla_from_cfg
            model_cfg = copy.deepcopy(cfg.inference_model)
            model_cfg.update(pretrained_name_or_path=None, num_steps=args.denoise_steps)
            # Skip allocating/initializing weights that the checkpoint replaces.
            # Keep non-persistent buffers (e.g. RoPE frequencies) materialized.
            with init_empty_weights(include_buffers=False):
                self.model = build_vla_from_cfg(model_cfg)
            # assign preserves the checkpoint's mixed FP32/BF16 tensors.
            # Calling .bfloat16() globally would break OpenPI FP32 flow layers.
            weights = load_file(args.checkpoint, device='cuda:0')
            self.model.load_state_dict(weights, strict=True, assign=True)
            del weights
            if args.param_dtype == 'fp32':
                self.model.float()
            # Parameters are already on CUDA; move the remaining CPU buffers.
            self.model.eval().requires_grad_(False).cuda()
            self.info['loaded'] = True
            self.info['parameters'] = sum(p.numel() for p in self.model.parameters())

    def predict(self, path):
        with np.load(path, allow_pickle=False) as data:
            state = vector(data['qpos'], 7, 'observation state').astype(np.float32)
            obs = {key: data[key] for key in ('cam_high', 'cam_left_wrist')}
        obs.update(qpos=state, task_description=self.args.instruction)
        start = time.monotonic()
        batch = self.dataset(obs)
        if not self.info['loaded']:
            return dict(shapes={k: list(v.shape) for k, v in batch.items()})
        torch = self.torch
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = self.model.predict_action(**batch)
        actions = self.denormalize(dict(action=raw.float().cpu().numpy(), state=state))
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f'Invalid model actions: {actions.shape}')
        return dict(actions=actions.tolist(), elapsed_s=time.monotonic() - start,
                    peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20)


def worker(config_path):
    # Reserve the original pipe and route even native-library stdout to stderr.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with open(config_path) as f:
        args = argparse.Namespace(**json.load(f))
    try:
        policy = Policy(args, load_model=args.mode != 'check')
        print(json.dumps(dict(ready=policy.info)), file=output)
        for line in sys.stdin:
            request = json.loads(line)
            print(json.dumps(policy.predict(request['path']), allow_nan=False), file=output)
    except Exception as exc:
        print(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')), file=output)
        raise


class PolicyProcess:
    def __init__(self, args, directory):
        self.args = args
        self.directory = Path(directory)
        config = self.directory / 'worker.json'
        config.write_text(json.dumps(vars(args)))
        self.proc = subprocess.Popen(
            [args.policy_python, '-u', str(Path(__file__).resolve()), '--worker', str(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)

    def receive(self, timeout, health=lambda: None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            health()
            if self.selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'Policy process exited: {self.proc.poll()}')
                result = json.loads(line)
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result
        raise TimeoutError(f'Policy response timed out after {timeout}s')

    def predict(self, obs, health=lambda: None):
        path = self.directory / 'observation.npz'
        np.savez(path, **obs)
        self.proc.stdin.write(json.dumps(dict(path=str(path))) + '\n')
        self.proc.stdin.flush()
        return self.receive(self.args.policy_timeout_s, health)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.selector.close()


class H264VideoWriter:
    """One FFmpeg process per camera; RGB input, H.264/yuv420p MP4 output."""
    def __init__(self, directory, args):
        self.args = args
        self.finalized = False
        self.error_path = Path(directory) / 'encoder.log'
        self.error_log = self.error_path.open('wb')
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
            '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{args.camera_width}x{args.camera_height}',
            '-framerate', str(args.camera_fps), '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
            '-crf', str(args.recording_video_crf), '-pix_fmt', 'yuv420p',
            '-threads', '2', '-filter_threads', '1', '-g', str(args.camera_fps),
            # Fragmented MP4 avoids keeping the entire index until shutdown.
            '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
            str(Path(directory) / 'video.mp4'),
        ]
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL, stderr=self.error_log,
                                         bufsize=0, start_new_session=True)
        except BaseException:
            self.error_log.close()
            raise
        os.set_blocking(self.proc.stdin.fileno(), False)

    def write(self, rgb):
        data = memoryview(np.ascontiguousarray(rgb)).cast('B')
        deadline = time.monotonic() + self.args.recording_flush_timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdin, selectors.EVENT_WRITE)
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('H.264 encoder input timed out')
                if self.proc.poll() is not None:
                    raise RuntimeError(f'H.264 encoder exited; see {self.error_path}')
                if not selector.select(min(remaining, .1)):
                    continue
                try:
                    written = os.write(self.proc.stdin.fileno(), data)
                except BlockingIOError:
                    continue
                if written == 0:
                    raise OSError('H.264 encoder accepted zero bytes')
                data = data[written:]

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()

    def close(self):
        try:
            self.proc.stdin.close()
            try:
                code = self.proc.wait(timeout=self.args.recording_flush_timeout_s)
            except subprocess.TimeoutExpired:
                self.abort()
                self.proc.wait(timeout=5)
                raise TimeoutError('H.264 encoder finalization timed out')
            if code != 0:
                raise RuntimeError(f'H.264 encoder exited with status {code}; see {self.error_path}')
            self.finalized = True
        finally:
            self.error_log.close()


class FrameRecorder:
    """Bounded H.264 recording with one timestamp row per submitted video frame."""
    keys = ('cam_high', 'cam_left_wrist')

    def __init__(self, directory, args):
        if args.camera_width % 2 or args.camera_height % 2:
            raise ValueError('H.264 yuv420p recording requires even camera width and height')
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('H.264 recording requires ffmpeg with the libx264 encoder')
        encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                  capture_output=True, text=True, timeout=10, check=True)
        if 'libx264 ' not in encoders.stdout:
            raise RuntimeError('FFmpeg does not provide the libx264 encoder')
        self.args = args
        self.directory = Path(directory) / 'camera_frames'
        for key in self.keys:
            (self.directory / key).mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=args.recording_queue_size)
        self.closing, self.aborted = threading.Event(), threading.Event()
        self.lock, self.close_lock = threading.Lock(), threading.Lock()
        self.error = None
        self.closed = False
        self.writers = {}
        self.counts = {key: dict(received=0, saved=0, missing_frame_numbers=0)
                       for key in self.keys}
        self.thread = threading.Thread(target=self._write, name='camera-recorder', daemon=True)
        self.thread.start()

    def health(self):
        if self.error is not None:
            raise RuntimeError(f'Camera recording failed: {self.error}') from self.error

    def enqueue(self, key, rgb, metadata):
        self.health()
        if self.closing.is_set():
            raise RuntimeError('Camera recorder is closing')
        if rgb.dtype != np.uint8 or rgb.shape != (self.args.camera_height, self.args.camera_width, 3):
            raise ValueError('Video frame must be RGB uint8 with the configured camera dimensions')
        with self.lock:
            self.counts[key]['received'] += 1
            self.counts[key]['missing_frame_numbers'] += metadata['missing_before']
        try:
            self.queue.put_nowait((key, rgb, metadata.copy()))
        except queue.Full:
            self.error = RuntimeError('Recording queue is full; disk/video encoder cannot keep up. '
                                      'Recording is incomplete; stopping rollout.')
            self.health()

    def _write(self):
        try:
            with contextlib.ExitStack() as stack:
                logs = {key: stack.enter_context(
                    (self.directory / key / 'frames.jsonl').open('w', buffering=1))
                    for key in self.keys}
                while not self.aborted.is_set() and (not self.closing.is_set() or not self.queue.empty()):
                    try:
                        key, rgb, metadata = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    with self.lock:
                        if self.aborted.is_set():
                            break
                        if key not in self.writers:
                            self.writers[key] = H264VideoWriter(self.directory / key, self.args)
                        writer = self.writers[key]
                        index = self.counts[key]['saved']
                    if metadata['video_frame_index'] != index:
                        raise ValueError(f'Nonconsecutive video frame index for {key}: {metadata}')
                    writer.write(rgb)
                    logs[key].write(json.dumps(metadata, allow_nan=False) + '\n')
                    with self.lock:
                        self.counts[key]['saved'] += 1
        except Exception as exc:
            if self.error is None:
                self.error = exc
        finally:
            # Always send EOF and reap every encoder, even after a write failure.
            for writer in self.writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    if self.error is None:
                        self.error = exc

    def close(self):
        # Called after acquisition threads stop, so the queue cannot grow.
        with self.close_lock:
            if self.closed:
                self.health()
                return
            self.closing.set()
            self.thread.join(timeout=self.args.recording_flush_timeout_s)
            if self.thread.is_alive():
                self.error = TimeoutError('Camera recording flush timed out; video is incomplete')
                self.aborted.set()
                with self.lock:
                    for writer in self.writers.values():
                        writer.abort()  # Unblocks a full input pipe or stalled finalization.
                self.thread.join(timeout=5)
            with self.lock:
                counts = copy.deepcopy(self.counts)
            complete = self.error is None and not self.thread.is_alive() and all(
                value['received'] == value['saved'] and value['missing_frame_numbers'] == 0
                and (value['saved'] == 0 or self.writers[key].finalized)
                for key, value in counts.items())
            summary = dict(complete=complete, cameras=counts, codec='h264', pixel_format='yuv420p',
                           fps=self.args.camera_fps, crf=self.args.recording_video_crf,
                           timing='Constant FPS video; use frames.jsonl for actual capture timestamps',
                           error=None if self.error is None else str(self.error))
            (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2))
            while not self.queue.empty():
                self.queue.get_nowait()
            self.closed = True
            print(json.dumps(dict(event='camera_recording_closed', **summary)), flush=True)
            self.health()


class Cameras:
    def __init__(self, args, record_dir=None):
        import pyrealsense2 as rs
        self.args, self.pipelines = args, {}
        self.started = set()
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.close_lock = threading.Lock()
        self.latest, self.threads, self.counts = {}, [], {}
        self.error = None
        self.closed = False
        self.recorder = FrameRecorder(record_dir, args) if record_dir is not None else None
        try:
            for key, serial in [('cam_high', args.head_serial),
                                ('cam_left_wrist', args.wrist_serial)]:
                pipeline, config = rs.pipeline(), rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, args.camera_width, args.camera_height,
                                     rs.format.rgb8, args.camera_fps)
                self.pipelines[key] = pipeline
                pipeline.start(config)
                self.started.add(key)
                thread = threading.Thread(target=self._capture, args=(key, serial, pipeline),
                                          name=f'capture-{key}', daemon=True)
                self.threads.append(thread)
                thread.start()
            deadline = (time.monotonic() + args.camera_timeout_ms / 1000
                        + 2 * args.camera_warmup_frames / args.camera_fps)
            with self.condition:
                while any(self.counts.get(key, 0) < args.camera_warmup_frames
                          for key in self.pipelines):
                    self.health()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Camera warmup timed out')
                    self.condition.wait(min(0.05, remaining))
        except BaseException as exc:
            if self.recorder is not None and self.recorder.error is None:
                self.recorder.error = exc
            try:
                self.close()
            except Exception as cleanup_error:
                raise exc from cleanup_error
            raise

    def _capture(self, key, serial, pipeline):
        previous, sequence = None, 0
        try:
            while not self.stop.is_set():
                frame = pipeline.wait_for_frames(self.args.camera_timeout_ms).get_color_frame()
                if self.stop.is_set():
                    break
                if not frame:
                    raise RuntimeError(f'Missing RGB frame: {key}')
                number = frame.get_frame_number()
                if number == previous:
                    continue  # Never record the same SDK frame twice.
                if previous is not None and number < previous:
                    raise RuntimeError(f'Camera frame counter went backwards: {key}')
                missing = 0 if previous is None else number - previous - 1
                rgb = np.asanyarray(frame.get_data()).copy()
                metadata = dict(sequence=sequence, frame_number=number, serial=serial,
                                device_timestamp_ms=frame.get_timestamp(),
                                timestamp_domain=str(frame.get_frame_timestamp_domain()),
                                host_monotonic_ns=time.monotonic_ns(),
                                host_time_ns=time.time_ns(), missing_before=missing,
                                file='video.mp4', video_frame_index=sequence,
                                video_timestamp_s=sequence / self.args.camera_fps)
                if self.recorder is not None:
                    self.recorder.enqueue(key, rgb, metadata)
                if missing:
                    print(json.dumps(dict(event='camera_frame_gap', camera=key,
                                          frame_number=number, missing=missing)), flush=True)
                with self.condition:
                    self.latest[key] = (rgb, metadata)
                    self.counts[key] = sequence + 1
                    self.condition.notify_all()
                previous, sequence = number, sequence + 1
        except Exception as exc:
            if not self.stop.is_set():
                self.error = exc
                if self.recorder is not None and self.recorder.error is None:
                    self.recorder.error = exc
                self.stop.set()
                with self.condition:
                    self.condition.notify_all()

    def health(self):
        if self.error is not None:
            raise RuntimeError(f'Camera acquisition failed: {self.error}') from self.error
        if self.recorder is not None:
            self.recorder.health()

    def read(self, with_metadata=False):
        self.health()
        with self.condition:
            if self.stop.is_set():
                raise RuntimeError('Cameras have been closed')
            images, frames = {}, {}
            for key in self.pipelines:
                if key not in self.latest:
                    raise RuntimeError(f'No camera frame: {key}')
                rgb, metadata = self.latest[key]
                age = (time.monotonic_ns() - metadata['host_monotonic_ns']) / 1e9
                if age > self.args.camera_timeout_ms / 1000:
                    raise TimeoutError(f'Stale camera frame: {key}, age={age:.3f}s')
                images[key], frames[key] = rgb.copy(), metadata.copy()
        return (images, frames) if with_metadata else images

    def close(self):
        with self.close_lock:
            if self.closed:
                return
            self.stop.set()
            errors = []
            for key, pipeline in list(self.pipelines.items()):
                try:
                    pipeline.stop()  # Unblocks pending wait_for_frames.
                except Exception as exc:
                    if key in self.started:
                        errors.append(f'{key}: {exc}')
                    else:
                        del self.pipelines[key]  # Failed start: release native reference.
                else:
                    self.started.discard(key)
                    del self.pipelines[key]
            for thread in self.threads:
                thread.join(timeout=self.args.camera_timeout_ms / 1000 + 1)
            if any(thread.is_alive() for thread in self.threads):
                errors.append('Camera acquisition thread did not stop')
            else:
                self.latest.clear()
                if self.recorder is not None:
                    try:
                        self.recorder.close()
                    except Exception as exc:
                        errors.append(str(exc))
            if errors:
                raise RuntimeError('Camera cleanup failed: ' + '; '.join(errors))
            self.closed = True


class RobotControl:
    """One thread owns SDK access; holds last target during policy inference."""
    def __init__(self, args, mapping):
        from Panthera_lib import Panthera
        self.args, self.mapping = args, mapping
        self.robot = Panthera(args.robot_config)
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.lock, self.commands = threading.Lock(), queue.Queue(maxsize=1)
        self.error = None
        self.measured = self.last_target = None
        self.last_stamps = np.zeros(7)
        self.last_updates = np.full(7, time.monotonic())
        self.thread = threading.Thread(target=self._run, name='panthera-control', daemon=True)
        self.thread.start()

    def health(self):
        if self.error is not None:
            raise RuntimeError(f'Robot control failed: {self.error}') from self.error
        if not self.thread.is_alive():
            raise RuntimeError('Robot control thread stopped')

    def feedback(self):
        self.robot.send_get_motor_state_cmd()
        self.robot.motor_send_cmd()
        states = self.robot.get_current_state() + [self.robot.get_current_state_gripper()]
        if len(states) != 7:
            raise RuntimeError(f'Expected 7 motors, got {len(states)}')
        now = time.monotonic()
        stamps = np.array([s.time for s in states])
        updated = (stamps != self.last_stamps) & (stamps > 0)
        self.last_updates[updated] = now
        self.last_stamps = stamps
        if np.any(now - self.last_updates > self.args.feedback_timeout_s):
            raise TimeoutError('Motor feedback stopped updating')
        if any(s.fault != 0 for s in states):
            raise RuntimeError(f'Motor fault codes: {[s.fault for s in states]}')
        if np.any(stamps <= 0):
            return None
        return vector([s.position for s in states], 7, 'motor feedback')

    def _run(self):
        active = self.args.mode == 'rollout'
        try:
            if self.robot.motor_count != 6:
                raise RuntimeError('Only single 6-DOF Panthera + gripper is supported')
            lower, upper = self.robot.joint_limits['lower'], self.robot.joint_limits['upper']
            if active:
                self.robot.set_timeout(self.args.motor_timeout_ms)
                self.robot.motor_send_cmd()
            while not self.stop_event.is_set():
                start = time.monotonic()
                measured = self.feedback()
                if measured is None:
                    self.stop_event.wait(1 / self.args.control_hz)
                    continue
                if np.any(measured[:6] < lower) or np.any(measured[:6] > upper):
                    raise ValueError('Measured joints outside configured SDK limits')
                if self.last_target is None:
                    self.last_target = measured.copy()
                pending = None
                try:
                    pending = self.commands.get_nowait()
                except queue.Empty:
                    pass
                target = self.last_target.copy()
                if pending is not None:
                    target = guard_target(self.mapping.to_sdk(pending[0]), target, measured,
                                          self.args, lower, upper)
                if active:
                    if np.max(np.abs(target[:6] - measured[:6])) > self.args.max_tracking_error:
                        raise ValueError('Tracking error exceeds configured limit')
                    if not self.robot.Joint_Pos_Vel(target[:6], [self.args.command_velocity] * 6,
                                                    self.args.max_torque, iswait=False):
                        raise RuntimeError('SDK rejected joint command')
                    if not self.robot.gripper_control(target[6], self.args.gripper_velocity,
                                                       self.args.gripper_torque):
                        raise RuntimeError('SDK rejected gripper command')
                with self.lock:
                    self.measured, self.last_target = measured, target
                self.ready.set()
                if pending is not None:
                    pending[1].set()
                self.stop_event.wait(max(0, 1 / self.args.control_hz - (time.monotonic() - start)))
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            if active:
                with contextlib.suppress(Exception):
                    self.robot.set_stop()

    def snapshot(self):
        self.health()
        with self.lock:
            if self.measured is None:
                raise RuntimeError('Robot feedback not ready')
            return self.mapping.to_policy(self.measured), self.mapping.to_policy(self.last_target)

    def send(self, action):
        done = threading.Event()
        self.commands.put_nowait((vector(action, 7, 'action'), done))
        deadline = time.monotonic() + self.args.feedback_timeout_s
        while not done.wait(0.01):
            self.health()
            if time.monotonic() > deadline:
                raise TimeoutError('Control thread did not acknowledge command')
        return self.snapshot()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('SDK thread did not stop; motor watchdog remains active')
        # The SDK destructor also applies its configured brake/stop policy.
        self.robot = None


def sdk_serial_ports(motor_config):
    """Mirror this SDK's USB VID/PID filter without opening serial devices."""
    valid_vendors = {0xFAE1} if motor_config['board_special_flag'] else {0xCAF1, 0xCAE1}
    ports = []
    for name in sorted(glob.glob(motor_config['Serial_Type'] + '*')):
        sysfs = Path('/sys/class/tty') / Path(name).resolve().name / 'device'
        for parent in sysfs.resolve().parents:
            vendor, product = parent / 'idVendor', parent / 'idProduct'
            if vendor.is_file() and product.is_file():
                if int(vendor.read_text().strip(), 16) in valid_vendors and int(product.read_text().strip(), 16) == 0xFFFF:
                    ports.append(name)
                break
    return ports


def preflight(args):
    import yaml
    sys.path.insert(0, str(Path(args.sdk_root) / 'panthera_python/scripts'))
    from Panthera_lib import Panthera  # noqa: F401 - import only, no device access
    import pyrealsense2  # noqa: F401

    for path in (args.policy_python, args.model_config, args.checkpoint, args.statistics,
                 args.robot_config, str(Path(args.tokenizer_root) / 'tokenizer/tokenizer.model')):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    with open(args.robot_config) as f:
        robot = yaml.safe_load(f)['robot']
    motor_path = (Path(args.robot_config).parent / robot['param_file']).resolve()
    with motor_path.open() as f:
        motor_config = yaml.safe_load(f)['robot']
    if args.mode in ('dry-run', 'rollout'):
        # The vendor constructor indexes its port list without a bounds check.
        # Refuse absent/busy serial ports before allowing it to open hardware.
        ports = sdk_serial_ports(motor_config)
        serial_ids = [port['serial_id']
                      for board in motor_config['CANboard'].values()
                      for port in board['CANport'].values()]
        if not serial_ids or min(serial_ids) < 1 or max(serial_ids) > len(ports):
            raise RuntimeError(f'SDK serial_id {serial_ids} cannot select from detected boards: {ports}')
        if shutil.which('fuser') is None:
            raise RuntimeError('Install psmisc (fuser) to check serial ownership before rollout')
        owners = subprocess.run(['fuser', *ports], capture_output=True, text=True, timeout=5)
        if owners.returncode == 0 or owners.stdout.strip():
            raise RuntimeError('SDK serial ports are in use; stop the owning controller first. '
                               f'{owners.stderr.strip()} {owners.stdout.strip()}')
        if owners.returncode != 1 or owners.stderr.strip():
            raise RuntimeError(f'Cannot establish serial ownership: {owners.stderr}')
    torque = vector(args.max_torque, 6, 'max torque')
    if np.any(torque <= 0) or np.any(torque > np.asarray(robot['max_torque'])):
        raise ValueError('Torque must be positive and <= SDK configuration')
    if args.command_velocity > min(robot['velocity_limits']):
        raise ValueError('Command velocity exceeds SDK configuration')
    lo, hi = robot['gripper_limits']['lower'], robot['gripper_limits']['upper']
    if not all(lo <= v <= hi for v in (args.gripper_open, args.gripper_closed)):
        raise ValueError('Gripper calibration outside SDK limits')
    if args.head_serial == args.wrist_serial:
        raise ValueError('The two cameras need different serial numbers')
    with open(args.statistics) as f:
        stats = json.load(f)['private']
    for key in ('proprio', 'action'):
        for quantile in ('q01', 'q99'):
            vector(stats[key][quantile], 7, f'{key}.{quantile}')
        if np.any(np.asarray(stats[key]['q99']) < np.asarray(stats[key]['q01'])):
            raise ValueError(f'{key}: q99 must be >= q01')
    return JointMapping(args)


def run(args):
    mapping = preflight(args)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_root) / f'{stamp}_{args.mode}_{os.getpid()}'
    log_dir.mkdir(parents=True)
    (log_dir / 'parameters.json').write_text(json.dumps(vars(args), indent=2))
    print(f'Mode={args.mode}; robot Python={sys.executable}; logs={log_dir}', flush=True)
    with tempfile.TemporaryDirectory(prefix='panthera-policy-') as temp, contextlib.ExitStack() as stack:
        policy = PolicyProcess(args, temp)
        stack.callback(policy.close)
        ready = policy.receive(args.load_timeout_s)
        print(json.dumps(ready, ensure_ascii=False), flush=True)
        (log_dir / 'policy.json').write_text(json.dumps(ready, indent=2))
        log = stack.enter_context((log_dir / 'rollout.jsonl').open('w', buffering=1))
        if args.mode in ('check', 'smoke'):
            obs = dict(qpos=np.zeros(7, dtype=np.float32),
                       cam_high=np.zeros((args.camera_height, args.camera_width, 3), np.uint8),
                       cam_left_wrist=np.zeros((args.camera_height, args.camera_width, 3), np.uint8))
            for index in range(1 if args.mode == 'check' else args.smoke_chunks):
                result = policy.predict(obs)
                log.write(json.dumps(dict(chunk=index, **result), allow_nan=False) + '\n')
                summary = {k: v for k, v in result.items() if k != 'actions'}
                if 'actions' in result:
                    summary['action_shape'] = list(np.asarray(result['actions']).shape)
                print(json.dumps(summary), flush=True)
            print(f'{args.mode} passed; no robot or camera opened.', flush=True)
            return

        observation_dir = log_dir / 'observation'
        if args.save_observations:
            observation_dir.mkdir()
        cameras = Cameras(args, log_dir if args.save_observations else None)
        stack.callback(cameras.close)
        robot = RobotControl(args, mapping)
        stack.callback(robot.close)
        robot.ready.wait(args.feedback_timeout_s + 1)
        measured, previous = robot.snapshot()
        def health():
            robot.health()
            cameras.health()
        steps, chunk = 0, 0
        while steps < args.max_steps:
            obs, camera_frames = cameras.read(with_metadata=True)
            measured, previous = robot.snapshot()
            state = previous if args.state_source == 'previous-target' else measured
            obs['qpos'] = state.astype(np.float32)
            if args.save_observations:
                np.savez(observation_dir / f'{chunk:05d}.npz', **obs)
            result = policy.predict(obs, health)
            actions = np.asarray(result['actions'], dtype=np.float64)
            if actions.shape != (50, 7) or not np.isfinite(actions).all():
                raise ValueError('Policy IPC returned invalid actions')
            log.write(json.dumps(dict(event='prediction', chunk=chunk, state=state.tolist(),
                                      measured=measured.tolist(), camera_frames=camera_frames,
                                      **result), allow_nan=False) + '\n')
            count = min(args.execute_steps, args.max_steps - steps)
            if args.mode == 'rollout':
                for action in actions[:count]:
                    health()
                    measured, command = robot.send(action)
                    log.write(json.dumps(dict(event='command', step=steps,
                                              requested=action.tolist(), sent=command.tolist(),
                                              measured=measured.tolist()), allow_nan=False) + '\n')
                    steps += 1
            else:
                # Read-only preview must not pretend that predicted commands happened.
                steps += count
            print(f'chunk={chunk} steps={steps} inference={result["elapsed_s"]:.3f}s '
                  f'first_action={np.round(actions[0], 4).tolist()}', flush=True)
            chunk += 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'smoke', 'dry-run', 'rollout'], required=True)
    parser.add_argument('--param-dtype', choices=['checkpoint', 'fp32'], required=True)
    parser.add_argument('--state-source', choices=['previous-target', 'measured'], required=True)
    for name in ('policy-python', 'sdk-root', 'robot-config', 'model-config', 'checkpoint',
                 'statistics', 'tokenizer-root', 'instruction', 'head-serial', 'wrist-serial', 'log-root'):
        parser.add_argument('--' + name, required=True)
    for name in ('seed', 'denoise-steps', 'camera-width', 'camera-height', 'camera-fps',
                 'camera-timeout-ms', 'camera-warmup-frames', 'execute-steps', 'max-steps',
                 'smoke-chunks', 'motor-timeout-ms', 'save-observations', 'recording-queue-size',
                 'recording-video-crf'):
        parser.add_argument('--' + name, required=True, type=int)
    for name in ('control-hz', 'policy-timeout-s', 'load-timeout-s', 'feedback-timeout-s',
                 'command-velocity', 'max-joint-step', 'max-target-jump', 'max-tracking-error',
                 'gripper-closed', 'gripper-open', 'gripper-velocity', 'gripper-torque', 'max-gripper-step',
                 'recording-flush-timeout-s'):
        parser.add_argument('--' + name, required=True, type=float)
    for name in ('max-torque', 'joint-signs', 'joint-offsets'):
        parser.add_argument('--' + name, required=True, type=float, nargs=6)
    args = parser.parse_args()
    exceptions = {'seed', 'save_observations', 'gripper_closed', 'gripper_open',
                  'recording_video_crf'}
    for key, value in vars(args).items():
        if isinstance(value, (int, float)) and key not in exceptions:
            if not np.isfinite(value) or value <= 0:
                parser.error(f'{key} must be positive and finite')
    if not 1 <= args.execute_steps <= 50:
        parser.error('execute-steps must be in [1, 50]')
    if args.save_observations not in (0, 1):
        parser.error('save-observations must be 0 or 1')
    if not 0 <= args.recording_video_crf <= 51:
        parser.error('recording-video-crf must be in [0, 51]')
    if not 0 < args.motor_timeout_ms <= 32760:
        parser.error('motor-timeout-ms must be in [1, 32760]')
    if args.motor_timeout_ms <= 2000 / args.control_hz:
        parser.error('motor-timeout-ms must allow at least two control periods')
    return args


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            run(parse_args())
        except KeyboardInterrupt:
            print('Interrupted; robot control and policy process cleaned up.', file=sys.stderr)
            sys.exit(130)
