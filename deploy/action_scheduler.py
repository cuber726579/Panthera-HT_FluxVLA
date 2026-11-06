"""Hardware-free action scheduling. Caller serializes access with the SDK lock.

Indices count issued policy actions, not encoder arrivals. A reservation fixes
the old prefix; a response may replace only the suffix before that boundary.
"""
from dataclasses import dataclass
import math

import numpy as np


def braking_speed(distance, dt, acceleration):
    """Largest next speed whose discrete braking tail fits in distance.

    The tail includes this tick's displacement: dt * sum(max(v-i*a*dt, 0)).
    This is deliberately discrete, rather than sqrt(2*a*d), which can cross
    a limit by one tick. All calculations are in SDK joint coordinates.
    """
    distance = np.maximum(distance, 0.)
    dv = acceleration * dt
    n = np.maximum(1., np.ceil((np.sqrt(1. + 8. * distance / (dv * dt)) - 1.) / 2.))
    return (distance / dt + dv * n * (n - 1.) / 2.) / n


def acceleration_step(desired, previous, velocity, *, dt, max_velocity,
                      acceleration, lower, upper):
    """Project one six-joint reference with braking and acceleration bounds.

    Retargeting inside the stopping distance may overshoot the *reference*;
    hard joint limits must remain reachable under braking, or we fail closed.
    Limits concern nominal command ticks, not measured motor acceleration.
    """
    desired, previous, velocity, lower, upper = [
        np.asarray(x, dtype=np.float64) for x in (desired, previous, velocity, lower, upper)]
    if any(x.shape != (6,) or not np.isfinite(x).all()
           for x in (desired, previous, velocity, lower, upper)):
        raise ValueError('Acceleration projection requires six finite joint values')
    if (not np.isfinite([dt, max_velocity, acceleration]).all()
            or min(dt, max_velocity, acceleration) <= 0 or np.any(lower >= upper)):
        raise ValueError('Invalid acceleration projection limits')
    eps = 1e-10
    if (np.any(previous < lower - eps) or np.any(previous > upper + eps)
            or np.any(desired < lower) or np.any(desired > upper)
            or np.any(np.abs(velocity) > max_velocity + eps)):
        raise ValueError('Acceleration projection starts outside limits')
    dv = acceleration * dt
    low = np.maximum.reduce((np.full(6, -max_velocity), velocity - dv,
                             -braking_speed(previous - lower, dt, acceleration)))
    high = np.minimum.reduce((np.full(6, max_velocity), velocity + dv,
                              braking_speed(upper - previous, dt, acceleration)))
    if np.any(low > high + eps):
        raise ValueError('No feasible braking trajectory before joint limit')
    error = desired - previous
    wanted = np.sign(error) * np.minimum(max_velocity, braking_speed(abs(error), dt, acceleration))
    projected_velocity = np.clip(wanted, low, np.maximum(low, high))
    projected = previous + projected_velocity * dt
    if np.any(projected < lower - eps) or np.any(projected > upper + eps):
        raise ValueError('Projected command outside joint limits')
    return projected, projected_velocity


@dataclass(frozen=True)
class Reservation:
    version: int
    origin: int
    prefix_len: int

    @property
    def boundary(self):
        return self.origin + self.prefix_len


class ActionBuffer:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.cursor = 0
        self.version = 0
        self.requested = np.empty((0, 7))
        self.effective = np.empty((0, 7))
        # SDK-coordinate velocities; effective positions are policy-coordinate.
        self.velocities = np.empty((0, 6))
        self.started = False

    @property
    def remaining(self):
        return len(self.requested)

    def reserve(self, prefix_len):
        if not 0 <= prefix_len < 50:
            raise ValueError('RTC prefix length must be in [0, 49]')
        if self.started and (prefix_len == 0 or self.remaining <= prefix_len):
            raise RuntimeError('Insufficient action buffer for inference delay')
        return Reservation(self.version, self.cursor, prefix_len)

    def accepts(self, reservation):
        return (reservation.version == self.version
                and self.cursor <= reservation.boundary)

    def install(self, reservation, requested, effective, velocities=None):
        if not self.accepts(reservation):
            return False
        requested, effective = np.asarray(requested), np.asarray(effective)
        if (requested.shape != effective.shape or requested.ndim != 2
                or requested.shape[1] != 7 or not len(requested)
                or not np.isfinite(requested).all() or not np.isfinite(effective).all()):
            raise ValueError('Invalid action buffer')
        # Optional for legacy buffer-only callers; controller always supplies it.
        velocities = np.zeros((len(requested), 6)) if velocities is None else np.asarray(velocities)
        if velocities.shape != (len(requested), 6) or not np.isfinite(velocities).all():
            raise ValueError('Invalid buffered velocities')
        keep = reservation.boundary - self.cursor
        if keep > self.remaining:
            return False
        capacity = self.max_steps - self.cursor
        self.requested = np.concatenate((self.requested[:keep], requested))[:capacity].copy()
        self.effective = np.concatenate((self.effective[:keep], effective))[:capacity].copy()
        self.velocities = np.concatenate((self.velocities[:keep], velocities))[:capacity].copy()
        self.version += 1
        self.started = True
        return True

    def peek(self):
        if not self.remaining:
            return None
        return self.requested[0].copy(), self.effective[0].copy(), self.velocities[0].copy()

    def advance(self, actual):
        if not self.remaining:
            raise RuntimeError('Cannot consume an empty action buffer')
        if not np.allclose(actual, self.effective[0], atol=1e-9, rtol=0):
            raise RuntimeError('Executed target differs from committed RTC trajectory')
        self.requested = self.requested[1:]
        self.effective = self.effective[1:]
        self.velocities = self.velocities[1:]
        self.cursor += 1


class DelayBudget:
    def __init__(self, hz, initial_s=.3, margin_steps=2, fixed_steps=0):
        self.hz, self.initial_s = hz, initial_s
        self.margin_steps, self.fixed_steps = margin_steps, fixed_steps
        self.samples = []

    def observe(self, elapsed_s):
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('Invalid inference latency')
        self.samples = (self.samples + [elapsed_s])[-20:]

    def steps(self):
        if self.fixed_steps:
            return self.fixed_steps
        latency = max(self.initial_s, float(np.percentile(self.samples, 95))) if self.samples else self.initial_s
        steps = math.ceil(latency * self.hz) + self.margin_steps
        if steps >= 49:
            raise RuntimeError('Inference delay leaves no usable action horizon')
        return max(1, steps)


def relative_rtc_actions(absolute, state, mask):
    """Rebase old absolute targets onto this observation, before normalization."""
    absolute = np.asarray(absolute, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if (absolute.ndim != 2 or absolute.shape[1] != len(mask)
            or state.shape != mask.shape or not np.isfinite(absolute).all()
            or not np.isfinite(state).all()):
        raise ValueError('Invalid RTC targets/state')
    return absolute - np.where(mask, state, 0)[None, :]
