#!/usr/bin/env python3
"""Replay saved predictions through the real buffer installer, without hardware.

No SDK, camera or model is instantiated. Predictions and their installation
versions are frozen from the log; this is not a counterfactual closed-loop run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import yaml

from action_scheduler import ActionBuffer, Reservation
from rollout_panthera import (JointMapping, RobotControl, project_target,
                             validate_committed_target)


def replay(run_dir, acceleration, mode):
    params = json.loads((run_dir / 'parameters.json').read_text())
    args = SimpleNamespace(**params)
    args.action_smoothing, args.command_acceleration = mode, acceleration
    limits = yaml.safe_load(Path(args.robot_config).read_text())['robot']['joint_limits']
    limits = {k: np.asarray(v, dtype=float) for k, v in limits.items()}
    rows = [json.loads(line) for line in (run_dir / 'rollout.jsonl').read_text().splitlines()]
    predictions = [r for r in rows if r['event'] == 'prediction' and r['accepted']]
    commands = [r for r in rows if r['event'] == 'command']
    if not commands or not predictions:
        raise ValueError(f'No replayable actions in {run_dir}')
    # __init__ opens hardware: intentionally bypass it and provide data only.
    c = RobotControl.__new__(RobotControl)
    c.args, c.mapping, c.lock = args, JointMapping(args), threading.Lock()
    c.robot = SimpleNamespace(joint_limits=limits)
    c.health = lambda: None
    c.action_buffer = ActionBuffer(args.max_steps)
    c.last_target = c.mapping.to_sdk(predictions[0]['state'])
    initial = c.last_target.copy()
    c.last_command_velocity = np.zeros(6)
    sent, velocities, raw, times = [], [], [], []
    checked_prefix_steps = 0
    for command in commands:
        b = c.action_buffer
        if command['step'] != b.cursor:
            raise ValueError('Non-contiguous command log')
        # plan_version records which predictions were installed at each tick.
        # JSONL event ordering alone is insufficient: commands are queue-drained.
        while b.version < command['plan_version']:
            prediction = predictions[b.version]
            reservation = Reservation(prediction['source_version'], prediction['origin'],
                                      prediction['boundary'] - prediction['origin'])
            keep = reservation.boundary - b.cursor
            if keep < 0:
                raise ValueError('Recorded prediction cannot be installed at this cursor')
            before, before_v = b.effective[:keep].copy(), b.velocities[:keep].copy()
            start = time.perf_counter()
            accepted = c.install_prediction(reservation, prediction['actions'])
            times.append(time.perf_counter() - start)
            if not accepted:
                raise AssertionError('Recorded accepted prediction was rejected')
            np.testing.assert_array_equal(b.effective[:keep], before)
            np.testing.assert_array_equal(b.velocities[:keep], before_v)
            checked_prefix_steps += keep
        requested, effective, velocity = b.peek()
        np.testing.assert_allclose(requested, command['requested'], atol=1e-12, rtol=0)
        target = c.mapping.to_sdk(effective)
        # Ideal feedback validates the command path. Historical feedback belongs
        # to the old trajectory and is reported separately, not a plant model.
        validate_committed_target(target, velocity, c.last_target, c.last_command_velocity,
                                  c.last_target, args, limits['lower'], limits['upper'])
        b.advance(c.mapping.to_policy(target))
        c.last_target, c.last_command_velocity = target, velocity.copy()
        sent.append(effective)
        velocities.append(velocity)
        raw.append(requested)
    # Exercise the normal-stop projection from the final nonzero command speed.
    brake = []
    if mode == 'acceleration':
        for _ in range(int(np.ceil(args.command_velocity * args.control_hz / acceleration)) + 2):
            if np.max(abs(c.last_command_velocity)) < 1e-10:
                break
            target, velocity = project_target(c.last_target, c.last_target, c.last_command_velocity,
                                               args, limits['lower'], limits['upper'])
            validate_committed_target(target, velocity, c.last_target, c.last_command_velocity,
                                      c.last_target, args, limits['lower'], limits['upper'])
            c.last_target, c.last_command_velocity = target, velocity
            brake.append(dict(position_sdk=target.tolist(), velocity_sdk=velocity.tolist()))
        assert np.max(abs(c.last_command_velocity)) < 1e-10
    sent, velocities, raw = map(np.asarray, (sent, velocities, raw))
    original = np.array([r['sent'] for r in commands])
    measured = np.array([r['measured'] for r in commands])
    if mode == 'none':
        np.testing.assert_allclose(sent, original, atol=1e-9, rtol=0)
    accelerations = np.diff(np.vstack((np.zeros(6), velocities)), axis=0) * args.control_hz
    reversal = ((velocities[1:] * velocities[:-1] < 0)
                & (abs(velocities[1:]) > .05) & (abs(velocities[:-1]) > .05))
    boundary = np.array([r['step'] in {p['boundary'] for p in predictions[1:]} for r in commands])
    deviation = abs(sent[:, :6] - original[:, :6])
    historical_error = abs(sent[:, :6] - measured[:, :6])
    metrics = dict(mode=mode, acceleration_limit=acceleration, commands=len(commands),
                   installed_predictions=c.action_buffer.version, checked_prefix_steps=checked_prefix_steps,
                   max_velocity=float(abs(velocities).max()), max_acceleration=float(abs(accelerations).max()),
                   acceleration_abs_p95=float(np.percentile(abs(accelerations), 95)),
                   abrupt_reversals_per_joint=reversal.sum(axis=0).tolist(),
                   max_boundary_delta_v=float(abs(accelerations[boundary]).max() / args.control_hz),
                   deviation_from_original_rad=dict(mean=float(deviation.mean()), p95=float(np.percentile(deviation, 95)),
                                                    maximum=float(deviation.max())),
                   historical_feedback_difference_max=float(historical_error.max()),
                   historical_tracking_threshold_exceeding_ticks=int(np.sum(np.max(historical_error, axis=1) > args.max_tracking_error)),
                   install_time_ms=dict(p95=float(np.percentile(times, 95) * 1000), maximum=float(max(times) * 1000)),
                   normal_stop_braking_ticks=len(brake))
    arrays = dict(sent=sent, original=original, measured=measured, requested=raw,
                  velocity_sdk=velocities, acceleration_sdk=accelerations,
                  initial_sdk=initial, boundary=boundary, control_hz=args.control_hz)
    return metrics, arrays, brake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acceleration', type=float, default=1.5)
    args = parser.parse_args()
    if not np.isfinite(args.acceleration) or args.acceleration <= 0:
        parser.error('acceleration must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(validation='Fixed logged predictions; ideal feedback; no SDK/cameras/model opened. '
                             'Historical feedback differences are not new tracking measurements.', runs=[])
    for run in args.runs:
        entry = dict(run=str(run.resolve()), input_sha256={})
        params = json.loads((run / 'parameters.json').read_text())
        for name, path in [('parameters.json', run / 'parameters.json'),
                           ('rollout.jsonl', run / 'rollout.jsonl'),
                           ('robot_config', Path(params['robot_config']))]:
            entry['input_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for mode in ('none', 'acceleration'):
            metrics, arrays, brake = replay(run, args.acceleration, mode)
            entry[mode] = metrics
            np.savez_compressed(args.output / f'{run.name}_{mode}.npz', **arrays)
            (args.output / f'{run.name}_{mode}_brake.json').write_text(json.dumps(brake, indent=2))
        result['runs'].append(entry)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
