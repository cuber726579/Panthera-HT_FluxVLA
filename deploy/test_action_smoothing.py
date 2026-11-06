"""Acceleration/RTC integration tests. All SDK calls are mocks."""
import contextlib
import io
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from action_scheduler import ActionBuffer, acceleration_step
from launch_panthera import build_command
from rollout_panthera import (JointMapping, RobotControl, parse_args, project_target,
                             validate_committed_target)
import test_action_scheduler as async_tests
from test_rollout_panthera import arguments


def controller():
    c = RobotControl.__new__(RobotControl)
    c.args = arguments()
    c.args.action_smoothing, c.args.command_acceleration = 'acceleration', 1.5
    c.args.max_steps, c.args.inference_mode, c.args.mode = 200, 'async', 'rollout'
    c.args.gripper_velocity = c.args.gripper_torque = .5
    c.args.max_torque = [1.] * 6
    c.mapping, c.lock = JointMapping(c.args), threading.Lock()
    c.robot = Mock()
    c.robot.joint_limits = dict(lower=np.full(6, -2.), upper=np.full(6, 2.))
    c.health = lambda: None
    c.action_buffer = ActionBuffer(200)
    c.last_target, c.last_command_velocity = np.zeros(7), np.zeros(6)
    c.measured = np.zeros(7)
    c.feedback = lambda: c.last_target.copy()
    c.home_abort = threading.Event()
    return c


def consume(c):
    requested, effective, velocity = c.action_buffer.peek()
    target = c.mapping.to_sdk(effective)
    validate_committed_target(target, velocity, c.last_target, c.last_command_velocity,
                              c.last_target, c.args, **c.robot.joint_limits)
    c.action_buffer.advance(effective)
    c.last_target, c.last_command_velocity = target, velocity
    c.measured = target.copy()


class Projection(unittest.TestCase):
    def step(self, target, q, v, **kw):
        options = dict(dt=1/30, max_velocity=.3, acceleration=1.5,
                       lower=np.full(6, -1.), upper=np.full(6, 1.))
        options.update(kw)
        return acceleration_step(np.full(6, target), q, v, **options)

    def test_reversal_and_constant_target_settle_without_limit_overshoot(self):
        q, v = np.zeros(6), np.zeros(6)
        for target in [1.] * 140 + [-1.] * 260 + [.1] * 180:
            new_q, new_v = self.step(target, q, v)
            self.assertLessEqual(np.max(abs(new_v-v)), .05 + 1e-10)
            self.assertLessEqual(np.max(abs(new_v)), .3 + 1e-10)
            self.assertLessEqual(np.max(abs(new_q)), 1. + 1e-10)
            np.testing.assert_allclose(new_q-q, new_v/30, atol=1e-12)
            q, v = new_q, new_v
        np.testing.assert_allclose(q, .1, atol=1e-9)
        np.testing.assert_allclose(v, 0, atol=1e-9)

    def test_retarget_inside_braking_distance_and_hard_limit_infeasibility(self):
        q, v = np.zeros(6), np.full(6, .3)
        new_q, new_v = self.step(0., q, v)
        np.testing.assert_allclose(new_v, .25)
        self.assertTrue(np.all(new_q > q))  # Cannot stop instantaneously at reference.
        with self.assertRaisesRegex(ValueError, 'feasible braking'):
            self.step(.99, np.full(6, .999), v)

    def test_invalid_arguments_and_raw_prediction_guards(self):
        for kw in [dict(acceleration=0), dict(dt=np.nan), dict(max_velocity=-1)]:
            with self.assertRaises(ValueError):
                self.step(0, np.zeros(6), np.zeros(6), **kw)
        c = controller()
        for target in [np.full(7, np.nan), np.r_[2.1, np.zeros(6)], np.r_[.8, np.zeros(6)]]:
            with self.assertRaises(ValueError):
                project_target(target, c.last_target, c.last_command_velocity,
                               c.args, **c.robot.joint_limits)


class CommitIntegration(unittest.TestCase):
    def test_early_one_remaining_and_boundary_arrivals_keep_position_and_velocity(self):
        for advance in (0, 3, 4):
            with self.subTest(advance=advance):
                c = controller()
                plan = np.tile(c.mapping.to_policy(np.r_[np.full(6, .3), .8]), (50, 1))
                self.assertTrue(c.install_prediction(c.action_buffer.reserve(0), plan))
                for _ in range(8):
                    consume(c)
                reservation = c.action_buffer.reserve(4)
                old_q, old_v = c.action_buffer.effective[:4].copy(), c.action_buffer.velocities[:4].copy()
                for _ in range(advance):
                    consume(c)
                live_q, live_v = c.last_target.copy(), c.last_command_velocity.copy()
                new = np.tile(c.mapping.to_policy(np.r_[np.full(6, -.3), .8]), (50, 1))
                self.assertTrue(c.install_prediction(reservation, new))
                keep = 4 - advance
                np.testing.assert_array_equal(c.action_buffer.effective[:keep], old_q[advance:])
                np.testing.assert_array_equal(c.action_buffer.velocities[:keep], old_v[advance:])
                np.testing.assert_array_equal(c.last_target, live_q)
                np.testing.assert_array_equal(c.last_command_velocity, live_v)
                for _ in range(20):
                    consume(c)
                # Request-time RTC uses exactly the projected positions.
                np.testing.assert_array_equal(c.async_snapshot(4)['old_actions'], c.action_buffer.effective)

    def test_late_and_failed_predictions_do_not_mutate_buffer_or_live_state(self):
        c = controller()
        plan = np.tile(c.mapping.to_policy(np.r_[np.full(6, .2), .8]), (50, 1))
        c.install_prediction(c.action_buffer.reserve(0), plan)
        late = c.action_buffer.reserve(2)
        for _ in range(3):
            consume(c)
        q, v = c.action_buffer.effective.copy(), c.action_buffer.velocities.copy()
        self.assertFalse(c.install_prediction(late, plan))
        bad = plan.copy()
        bad[20, 0] = 10.
        with self.assertRaises(ValueError):
            c.install_prediction(c.action_buffer.reserve(3), bad)
        np.testing.assert_array_equal(c.action_buffer.effective, q)
        np.testing.assert_array_equal(c.action_buffer.velocities, v)

    def test_live_feedback_and_tampered_velocity_rejected(self):
        c = controller()
        target, v = project_target(np.r_[np.full(6, .2), 0], c.last_target,
                                    c.last_command_velocity, c.args, **c.robot.joint_limits)
        with self.assertRaisesRegex(RuntimeError, 'committed position'):
            validate_committed_target(target, -v, c.last_target, c.last_command_velocity,
                                      c.last_target, c.args, **c.robot.joint_limits)
        measured = c.last_target.copy()
        measured[0] = .5
        with self.assertRaisesRegex(ValueError, 'not following'):
            validate_committed_target(target, v, c.last_target, c.last_command_velocity,
                                      measured, c.args, **c.robot.joint_limits)

    def test_real_control_loop_with_ideal_feedback_and_normal_stop(self):
        c = async_tests.fake_controller(80, dict(action_smoothing='acceleration', command_acceleration=1.5))
        rows = async_tests.AsyncIntegrationTests().drive(c, async_tests.FakePolicy(c, latency=.025))
        self.assertIsNone(c.error)
        self.assertIsNone(c.return_error)
        commands = [r for r in rows if r['event']=='command']
        self.assertEqual(len(commands), 80)
        speeds = np.array([call.args[1] for call in c.robot.Joint_Pos_Vel.call_args_list])
        self.assertLessEqual(np.max(abs(np.diff(speeds, axis=0))), .015 + 1e-9)
        np.testing.assert_allclose(speeds[-1], 0, atol=1e-10)

    def test_fault_skips_braking_and_homing(self):
        c = async_tests.fake_controller(80, dict(action_smoothing='acceleration', command_acceleration=1.5))
        c.robot.Joint_Pos_Vel.return_value = False
        with self.assertRaises(RuntimeError):
            async_tests.AsyncIntegrationTests().drive(c, async_tests.FakePolicy(c))
        self.assertIsNotNone(c.error)
        c.robot.set_stop.assert_called_once()

    def test_exhaustion_and_interrupt_brake_without_replaying_policy(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                c = async_tests.fake_controller(100, dict(action_smoothing='acceleration',
                                                          command_acceleration=1.5))
                policy = async_tests.FakePolicy(c, latency=.025 if interrupted else .65)
                poll = policy.poll
                if interrupted:
                    def interrupt(health):
                        if len(policy.requests) >= 2:
                            raise KeyboardInterrupt('offline interrupt')
                        return poll(health)
                    policy.poll = interrupt
                with self.assertRaises(KeyboardInterrupt if interrupted else RuntimeError):
                    async_tests.AsyncIntegrationTests().drive(c, policy)
                self.assertIsNone(c.error)
                self.assertIsNone(c.return_error)
                speeds = np.array([x.args[1] for x in c.robot.Joint_Pos_Vel.call_args_list])
                self.assertLessEqual(np.max(abs(np.diff(speeds, axis=0))), .015 + 1e-9)
                np.testing.assert_allclose(speeds[-1], 0, atol=1e-10)
                if not interrupted:
                    self.assertEqual(c.action_buffer.cursor, 50)


class LifecycleAndConfig(unittest.TestCase):
    def run_transition(self, c, fn):
        now = [0.]
        def wait(seconds):
            now[0] += max(seconds, .001)
        with patch('rollout_panthera.time.monotonic', side_effect=lambda: now[0]), \
                patch.object(c.home_abort, 'wait', side_effect=wait):
            fn()

    def test_braking_then_homing_preserves_acceleration_and_gripper(self):
        c = controller()
        c.last_target = np.r_[np.full(6, .15), .8]
        c.last_command_velocity = np.array([.3, -.3, .1, -.1, 0., .2])
        initial_q, initial_v = c.last_target.copy(), c.last_command_velocity.copy()
        self.run_transition(c, c._return_to_zero)
        calls = c.robot.Joint_Pos_Vel.call_args_list
        q = np.vstack([initial_q[:6]] + [x.args[0] for x in calls])
        v = np.vstack([initial_v] + [x.args[1] for x in calls])
        np.testing.assert_allclose(np.diff(q, axis=0)*30, v[1:], atol=1e-9)
        self.assertLessEqual(np.max(abs(np.diff(v, axis=0))), .05+1e-9)
        np.testing.assert_allclose(q[-1], 0, atol=1e-9)
        np.testing.assert_allclose(v[-1], 0, atol=1e-9)
        for call in c.robot.gripper_control.call_args_list:
            self.assertEqual(call.args, (.8, 0., .5))

    def test_braking_fault_cancel_and_homing_stall_fail_closed(self):
        c = controller()
        c.last_command_velocity[:] = .3
        c.feedback = Mock(side_effect=TimeoutError('stale'))
        with self.assertRaises(TimeoutError):
            self.run_transition(c, c._brake_to_hold)
        c.robot.Joint_Pos_Vel.assert_not_called()
        c = controller()
        c.home_abort.set()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            c._return_to_zero()
        c.robot.Joint_Pos_Vel.assert_not_called()
        c = controller()
        c.last_target[:6] = .1
        c.args.return_zero_timeout_s = 1.
        c.feedback = lambda: np.r_[np.full(6, .1), 0.]
        with self.assertRaises(TimeoutError):
            self.run_transition(c, c._return_to_zero)

    def test_yaml_cli_and_invalid_options_without_launching(self):
        config = Path(__file__).parent/'config/openpi_panthera_15000_async.yaml'
        command, _ = build_command(config, environ={})
        argv = command[1:]
        with patch('sys.argv', argv):
            parsed = parse_args()
        self.assertEqual(parsed.action_smoothing, 'acceleration')
        self.assertEqual(parsed.command_acceleration, 1.5)
        for option, value in [('--command-acceleration', 'nan'), ('--command-acceleration', '0'),
                              ('--inference-mode', 'sync')]:
            invalid = argv.copy()
            invalid[invalid.index(option)+1] = value
            with patch('sys.argv', invalid), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()


if __name__ == '__main__':
    unittest.main()
