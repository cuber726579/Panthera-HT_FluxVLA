"""Hardware-free regression tests for the rollout's command boundary.

Run: conda run -n panthera python -m unittest discover -s deploy -p 'test_*.py'
"""
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rollout_panthera import JointMapping, RobotControl, guard_target


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


if __name__ == '__main__':
    unittest.main()
