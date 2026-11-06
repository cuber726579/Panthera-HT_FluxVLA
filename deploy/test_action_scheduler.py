"""Offline RTC scheduling tests; no SDK import or device access."""
import io
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from action_scheduler import ActionBuffer, DelayBudget, relative_rtc_actions
from rollout_panthera import JointMapping, PolicyProcess, RobotControl, run_async
from test_rollout_panthera import arguments


def trajectory(start=0., length=50):
    result = np.zeros((length, 7))
    result[:, 0] = start + np.arange(length) * .001
    result[:, 6] = .5
    return result


class BufferTests(unittest.TestCase):
    def seeded(self, limit=200):
        b = ActionBuffer(limit)
        actions = trajectory()
        self.assertTrue(b.install(b.reserve(0), actions, actions))
        return b

    def test_preserves_committed_prefix_and_skips_new_prefix(self):
        b = self.seeded()
        b.advance(b.peek()[1])
        r = b.reserve(5)
        committed = b.effective[:5].copy()
        for _ in range(2):
            b.advance(b.peek()[1])
        new = trajectory(.2)
        self.assertTrue(b.install(r, new[5:], new[5:]))
        np.testing.assert_array_equal(b.effective[:3], committed[2:])
        np.testing.assert_array_equal(b.effective[3:], new[5:])
        self.assertEqual(b.cursor, 3)

    def test_late_and_duplicate_results_cannot_rewind(self):
        b = self.seeded()
        r = b.reserve(2)
        for _ in range(3):
            b.advance(b.peek()[1])
        before = b.effective.copy()
        self.assertFalse(b.install(r, trajectory(), trajectory()))
        np.testing.assert_array_equal(b.effective, before)
        r = b.reserve(2)
        self.assertTrue(b.install(r, trajectory(), trajectory()))
        self.assertFalse(b.install(r, trajectory(), trajectory()))

    def test_boundary_arrival_is_valid_and_step_limit_is_exact(self):
        b = self.seeded(12)
        r = b.reserve(3)
        for _ in range(3):
            b.advance(b.peek()[1])
        self.assertTrue(b.install(r, trajectory(), trajectory()))
        self.assertEqual(b.remaining, 9)
        for _ in range(9):
            b.advance(b.peek()[1])
        self.assertIsNone(b.peek())
        self.assertEqual(b.cursor, 12)

    def test_wrong_execution_invalid_prefix_and_bad_output_rejected(self):
        b = self.seeded()
        with self.assertRaisesRegex(RuntimeError, 'committed'):
            b.advance(np.ones(7))
        with self.assertRaises(RuntimeError):
            b.reserve(0)
        with self.assertRaises(ValueError):
            b.reserve(50)
        with self.assertRaises(ValueError):
            b.install(b.reserve(2), np.zeros((2, 6)), np.zeros((2, 6)))

    def test_rebases_six_joints_but_not_gripper(self):
        absolute = trajectory(.5)
        state = np.array([.2, -.3, .1, .4, -.2, .3, .9])
        relative = relative_rtc_actions(absolute, state, [True]*6 + [False])
        np.testing.assert_allclose(relative[:, :6] + state[:6], absolute[:, :6], atol=1e-7)
        np.testing.assert_array_equal(relative[:, 6], absolute[:, 6])

    def test_delay_budget_uses_tail_and_rejects_impossible_horizon(self):
        budget = DelayBudget(30, initial_s=.2, margin_steps=2)
        self.assertEqual(budget.steps(), 8)
        budget.observe(.4)
        self.assertEqual(budget.steps(), 14)
        budget.observe(2.)
        with self.assertRaises(RuntimeError):
            budget.steps()
        self.assertEqual(DelayBudget(30, fixed_steps=7).steps(), 7)


def fake_controller(max_steps=100, overrides=None):
    """Run the actual SDK-owning loop against an ideal-feedback fake."""
    args = arguments()
    args.mode = 'rollout'
    args.control_hz = 100
    args.max_steps = max_steps
    args.gripper_velocity, args.gripper_torque = .5, .5
    args.max_torque = [1.] * 6
    args.inference_mode, args.rtc_method = 'async', 'guidance'
    args.rtc_delay_steps, args.rtc_initial_delay_s = 10, .03
    args.rtc_margin_steps = 2
    args.execute_steps = 15
    args.state_source, args.save_observations = 'previous-target', 0
    if overrides:
        vars(args).update(overrides)
    c = RobotControl.__new__(RobotControl)
    c.args, c.mapping = args, JointMapping(args)
    c.robot = Mock()
    c.robot.motor_count = 6
    c.robot.joint_limits = dict(lower=np.full(6, -3.), upper=np.full(6, 3.))
    c.lock, c.commands = threading.Lock(), queue.Queue(maxsize=1)
    c.action_buffer, c.async_events = ActionBuffer(max_steps), queue.Queue(maxsize=4096)
    c.stop_event, c.ready = threading.Event(), threading.Event()
    c.home_abort, c.control_done = threading.Event(), threading.Event()
    c.error = c.return_error = None
    c.measured = c.last_target = None
    c.feedback = lambda: (c.last_target.copy() if c.last_target is not None else c.mapping.to_sdk(np.zeros(7)))
    c.thread = threading.Thread(target=c._run)
    c.thread.start()
    if not c.ready.wait(1):
        raise AssertionError('Fake controller did not start')
    return c


class FakePolicy:
    def __init__(self, controller, latency=.02):
        self.c, self.latency = controller, latency
        self.pending, self.requests, self.progress = None, [], []

    def submit(self, obs, rtc=None):
        if self.pending is not None:
            raise AssertionError('Overlapping inference')
        cursor = self.c.async_snapshot()['cursor']
        self.requests.append(rtc)
        self.pending = (time.monotonic(), cursor, obs['qpos'].copy(), rtc)
        return len(self.requests) - 1

    def poll(self, health):
        health()
        began, cursor, state, rtc = self.pending
        if time.monotonic() - began < self.latency:
            return None
        self.progress.append(self.c.async_snapshot()['cursor'] - cursor)
        result = trajectory(state[0])
        if rtc is not None:
            n = rtc['prefix_len']
            result[:n] = rtc['actions'][:n]
        self.pending = None
        return dict(actions=result.tolist(), elapsed_s=self.latency,
                    rtc_method='guidance' if rtc else 'none', request_id=len(self.requests)-1)


class AsyncIntegrationTests(unittest.TestCase):
    def drive(self, controller, policy):
        cameras = Mock()
        cameras.read.return_value = (dict(cam_high=np.zeros((2, 2, 3)),
                                          cam_left_wrist=np.zeros((2, 2, 3))), {})
        log = io.StringIO()
        try:
            with tempfile.TemporaryDirectory() as directory:
                run_async(controller.args, controller, cameras, policy, log, Path(directory))
        finally:
            controller.stop_event.set()
            controller.thread.join(2)
            self.assertFalse(controller.thread.is_alive())
        return [json.loads(line) for line in log.getvalue().splitlines()]

    def test_control_continues_during_inference_and_ends_at_step_limit(self):
        c = fake_controller()
        policy = FakePolicy(c, latency=.035)
        rows = self.drive(c, policy)
        self.assertIsNone(c.error)
        cmds = [r for r in rows if r['event'] == 'command']
        self.assertEqual([r['step'] for r in cmds], list(range(c.args.max_steps)))
        self.assertGreaterEqual(len(policy.requests), 3)
        self.assertTrue(any(n >= 2 for n in policy.progress[1:]))
        self.assertTrue(all(r is not None for r in policy.requests[1:]))
        self.assertTrue(all(r['accepted'] for r in rows if r['event'] == 'prediction'))

    def test_late_result_is_discarded_without_replaying_actions(self):
        c = fake_controller(110)
        c.args.rtc_delay_steps = 3
        policy = FakePolicy(c, latency=.08)
        original_submit = policy.submit
        def submit(obs, rtc=None):
            # Only the second request misses its reservation; later ones recover.
            policy.latency = .08 if len(policy.requests) < 2 else .005
            return original_submit(obs, rtc)
        policy.submit = submit
        rows = self.drive(c, policy)
        predictions = [r for r in rows if r['event'] == 'prediction']
        self.assertTrue(any(not r['accepted'] for r in predictions))
        cmds = [r for r in rows if r['event'] == 'command']
        self.assertEqual([r['step'] for r in cmds], list(range(c.args.max_steps)))

    def test_exhaustion_holds_then_exits_instead_of_repeating(self):
        c = fake_controller(100)
        policy = FakePolicy(c, latency=.65)
        with self.assertRaisesRegex(RuntimeError, 'buffer exhausted'):
            self.drive(c, policy)
        self.assertEqual(c.action_buffer.cursor, 50)
        self.assertIsNone(c.error)

    def test_async_without_rtc_and_interrupt_do_not_leave_execution_running(self):
        c = fake_controller(55)
        c.args.rtc_method = 'none'
        policy = FakePolicy(c)
        self.drive(c, policy)
        self.assertTrue(all(r is None for r in policy.requests))
        c = fake_controller(100)
        policy = FakePolicy(c)
        original_poll = policy.poll
        def interrupted(health):
            if len(policy.requests) >= 2:
                raise KeyboardInterrupt('test interrupt during inference')
            return original_poll(health)
        policy.poll = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.drive(c, policy)
        self.assertTrue(c.control_done.is_set())
        self.assertLess(c.action_buffer.cursor, 100)

    def test_safety_fault_stops_before_consuming_unsafe_target(self):
        c = fake_controller()
        c.robot.Joint_Pos_Vel.return_value = False
        try:
            c.thread.join(1)
            self.assertIsNotNone(c.error)
            self.assertEqual(c.action_buffer.cursor, 0)
        finally:
            c.stop_event.set()
            c.thread.join(1)


class PolicyIPCTests(unittest.TestCase):
    def client(self, directory):
        p = PolicyProcess.__new__(PolicyProcess)
        p.args = SimpleNamespace(policy_timeout_s=1.)
        p.directory = Path(directory)
        p.pending, p.next_request_id = None, 0
        p.proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None)
        p.selector = Mock()
        p.selector.select.return_value = []
        return p

    def test_single_request_unique_files_and_response_id(self):
        with tempfile.TemporaryDirectory() as directory:
            p = self.client(directory)
            rtc = dict(actions=trajectory(), prefix_len=8)
            first = p.submit(dict(qpos=np.zeros(7)), rtc)
            path = p.pending['path']
            with np.load(path) as data:
                np.testing.assert_array_equal(data['rtc_actions'], rtc['actions'])
            self.assertIsNone(p.poll())
            with self.assertRaisesRegex(RuntimeError, 'one policy'):
                p.submit({})
            p.proc.stdout = io.StringIO(json.dumps(dict(request_id=first, actions=[]))+'\n')
            p.selector.select.return_value = [True]
            self.assertEqual(p.poll()['request_id'], first)
            self.assertFalse(path.exists())
            second = p.submit(dict(qpos=np.ones(7)))
            self.assertNotEqual(path, p.pending['path'])
            p.proc.stdout = io.StringIO(json.dumps(dict(request_id=first))+'\n')
            with self.assertRaisesRegex(RuntimeError, 'ID mismatch'):
                p.poll()
            self.assertNotEqual(first, second)

    def test_timeout_worker_error_and_eof_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            p = self.client(directory)
            p.submit({})
            p.pending['began'] -= 2
            with self.assertRaises(TimeoutError):
                p.poll()
            p.pending['began'] = time.monotonic()
            p.selector.select.return_value = [True]
            p.proc.stdout = io.StringIO('{"error":"injected worker failure"}\n')
            with self.assertRaisesRegex(RuntimeError, 'worker failure'):
                p.poll()
            p.proc.stdout = io.StringIO('')
            with self.assertRaisesRegex(RuntimeError, 'process exited'):
                p.poll()


if __name__ == '__main__':
    unittest.main()
