"""Detector corruption: what the oracle's perfect boxes must suffer.

Three measured effects (guide 16.1 line 6; offline_training_recipe.md
stage 0), each parameterised by the Phase-0 statistics file when it exists:

  * centre jitter  ~ N(0, sigma_px) on (u, v), and sigma_px/2 on width;
  * per-frame dropout with probability RISING as the box shrinks -- the
    source paper's own named weakness is small targets [P App. A];
  * BURST loss: a two-state Markov chain (ok <-> burst) modelling the
    video-stream stalls that killed the CTS deployment
    (offline_training_recipe.md 6.1) -- per-frame dropout alone cannot
    produce multi-frame gaps at the measured rate.

Defaults are placeholders with the documented shape [D-impl 5,
sim_implementation_map.md] and are REPLACED by fitting the chase_detector
CSV logs (csv_log.py schema) once real captures exist. `enabled=False` is
the clean-world smoke test.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml


@dataclass
class CorruptionConfig:
    enabled: bool = True
    sigma_px: float = 2.0
    # Dropout probability vs box width: linear ramp between the two anchor
    # widths, clamped outside -- monotone in 1/size like the measured curve.
    dropout_p_large: float = 0.02
    dropout_p_small: float = 0.30
    width_large_px: float = 90.0
    width_small_px: float = 15.0
    # Burst loss: P(enter) per step while ok; mean burst length in steps.
    burst_enter_p: float = 0.005
    burst_mean_len: float = 5.0

    @classmethod
    def from_file(cls, path: str) -> 'CorruptionConfig':
        """Load a fitted corruption-model file (YAML mapping of the fields
        above). Unknown keys are rejected: a typo in a measurement file must
        fail loudly, not silently default."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown corruption keys {sorted(unknown)} in {path}')
        return cls(**raw)


@dataclass
class Measurement:
    """One synthetic detection, in the DroneDetection frame convention."""
    u: float
    v: float
    w_px: float


class CorruptionModel:
    def __init__(self, cfg: Optional[CorruptionConfig] = None):
        self.cfg = cfg or CorruptionConfig()
        self._in_burst = False

    def reset(self) -> None:
        self._in_burst = False

    def dropout_probability(self, w_px: float) -> float:
        c = self.cfg
        if w_px >= c.width_large_px:
            return c.dropout_p_large
        if w_px <= c.width_small_px:
            return c.dropout_p_small
        # Linear in width between the anchors.
        f = (c.width_large_px - w_px) / (c.width_large_px - c.width_small_px)
        return c.dropout_p_large + f * (c.dropout_p_small - c.dropout_p_large)

    def apply(self, meas: Optional[Measurement], rng: np.random.Generator
              ) -> Optional[Measurement]:
        """True measurement in -> corrupted measurement or None (missed).

        Called once per control step, in step order, so the burst chain
        advances exactly once per step even when the target is out of frame
        (a stream stall does not care what the camera sees).
        """
        c = self.cfg
        if not c.enabled:
            return meas

        # Advance the burst chain first.
        if self._in_burst:
            if rng.random() < 1.0 / max(c.burst_mean_len, 1.0):
                self._in_burst = False
        elif rng.random() < c.burst_enter_p:
            self._in_burst = True

        if meas is None or self._in_burst:
            return None
        if rng.random() < self.dropout_probability(meas.w_px):
            return None
        return Measurement(
            u=meas.u + rng.normal(0.0, c.sigma_px),
            v=meas.v + rng.normal(0.0, c.sigma_px),
            w_px=max(1.0, meas.w_px + rng.normal(0.0, c.sigma_px / 2.0)),
        )
