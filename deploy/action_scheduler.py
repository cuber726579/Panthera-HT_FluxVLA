"""Hardware-free action scheduling. Caller serializes access with the SDK lock.

Indices count issued policy actions, not encoder arrivals. A reservation fixes
the old prefix; a response may replace only the suffix before that boundary.
"""
from dataclasses import dataclass
import math

import numpy as np


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

    def install(self, reservation, requested, effective):
        if not self.accepts(reservation):
            return False
        requested, effective = np.asarray(requested), np.asarray(effective)
        if (requested.shape != effective.shape or requested.ndim != 2
                or requested.shape[1] != 7 or not len(requested)
                or not np.isfinite(requested).all() or not np.isfinite(effective).all()):
            raise ValueError('Invalid action buffer')
        keep = reservation.boundary - self.cursor
        if keep > self.remaining:
            return False
        capacity = self.max_steps - self.cursor
        self.requested = np.concatenate((self.requested[:keep], requested))[:capacity].copy()
        self.effective = np.concatenate((self.effective[:keep], effective))[:capacity].copy()
        self.version += 1
        self.started = True
        return True

    def peek(self):
        if not self.remaining:
            return None
        return self.requested[0].copy(), self.effective[0].copy()

    def advance(self, actual):
        if not self.remaining:
            raise RuntimeError('Cannot consume an empty action buffer')
        if not np.allclose(actual, self.effective[0], atol=1e-9, rtol=0):
            raise RuntimeError('Executed target differs from committed RTC trajectory')
        self.requested = self.requested[1:]
        self.effective = self.effective[1:]
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
