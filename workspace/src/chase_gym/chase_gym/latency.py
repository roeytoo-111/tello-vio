"""The delay line: true geometry in, stale measurements out.

Implements guide 16.1 lines 4-5: the TRUE projection enters a queue stamped
with sim time; the observation samples the queue at (t - Delta_t), where the
per-episode base delay is drawn from the MEASURED 150-350 ms distribution [V]
and each frame adds jitter of tens of ms. At Delta_t = 0.1 s control periods
that is a delay of 1.5-3.5 steps -- exactly what the k = 4 action history
compensates (guide 4.2).

Time here is *sim* time supplied by the caller, which is what lets the same
class serve Tier B's lockstep latency_shim unchanged
(sim_training_architecture.md 3.3).
"""
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

from . import constants as C


class DelayQueue:
    """Ordered (t, payload) store; sample(t) returns the newest payload with
    timestamp <= t -- the newest frame that could physically have arrived."""

    def __init__(self, max_age_s: float = 2.0):
        self._q = deque()
        self.max_age_s = float(max_age_s)

    def clear(self) -> None:
        self._q.clear()

    def push(self, t: float, payload: Any) -> None:
        if self._q and t < self._q[-1][0]:
            raise ValueError(
                f'non-monotonic push: {t} after {self._q[-1][0]}')
        self._q.append((t, payload))
        cutoff = t - self.max_age_s
        while self._q and self._q[0][0] < cutoff:
            self._q.popleft()

    # Comparison slack, far below any control period. When the latency is
    # an exact multiple of dt, `t - latency` lands ON a stamp and the
    # outcome of a bare <= rides accumulated float error -- and the error
    # GROWS with sim time, so a backend whose clock starts later (e.g.
    # after a reset settle hold) silently flips from inclusive to
    # exclusive and the observation freezes at its pre-rolled value
    # (measured on GzChaseEnv, 2026-09-21).
    _EPS_S = 1e-9

    def sample(self, t: float) -> Optional[tuple]:
        """(t_meas, payload) of the newest entry with t_meas <= t, else None
        (the pipe is still empty at episode start unless pre-rolled).
        Scans from the newest end: the answer sits ~2-4 entries in at the
        measured delays, vs ~20 from the old end."""
        for t_meas, payload in reversed(self._q):
            if t_meas <= t + self._EPS_S:
                return (t_meas, payload)
        return None


class LatencyModel:
    """Per-episode base + per-frame jitter, or an empirical sample file.

    `enabled=False` collapses the delay to zero -- the smoke-test world of
    guide 28.2 ("static target, no latency, no noise").
    """

    def __init__(self, enabled: bool = True,
                 base_range_s=C.LATENCY_RANGE_S,
                 jitter_std_s: float = 0.020,
                 samples: Optional[Sequence[float]] = None):
        self.enabled = bool(enabled)
        self.base_range_s = (float(base_range_s[0]), float(base_range_s[1]))
        self.jitter_std_s = float(jitter_std_s)
        self._samples = np.asarray(samples, dtype=float) if samples is not None else None
        self._base = 0.0

    @classmethod
    def from_samples(cls, samples, jitter_std_s: float = 0.0):
        """Empirical latency histogram (e.g. t_receive - t_capture from the
        chase_detector CSV logs) replaces the uniform draw."""
        return cls(enabled=True, jitter_std_s=jitter_std_s, samples=samples)

    def reset(self, rng: np.random.Generator) -> float:
        """Draw the episode's base delay; returns it (for the info dict)."""
        if not self.enabled:
            self._base = 0.0
        elif self._samples is not None:
            self._base = float(rng.choice(self._samples))
        else:
            self._base = float(rng.uniform(*self.base_range_s))
        return self._base

    @property
    def base_s(self) -> float:
        return self._base

    def delay(self, rng: np.random.Generator) -> float:
        """This frame's total delay: base + fresh jitter, floored at 0."""
        if not self.enabled:
            return 0.0
        return max(0.0, self._base + rng.normal(0.0, self.jitter_std_s))
