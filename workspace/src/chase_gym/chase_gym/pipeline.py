"""The shared measurement pipe and reset-draw contract.

Both tiers' envs (ChaseEnv and chase_sim_gz.GzChaseEnv) consume this ONE
implementation of the truth -> delay -> corruption path and the reset draw
prefix. The point is structural, not aesthetic: the A<->B replay gate and
every cross-tier checkpoint move rely on "one seed, one geometry, one
noise stream", and that contract broke the moment it lived in two textual
copies (it did -- caught twice during review). Now it cannot drift.

DRAW ORDER CONTRACT (enforced by draw_reset_placement + MeasurementPipe
and unit-tested cross-tier in chase_sim_gz/test):
    1. placement: z_cam, u0, v0            (3 uniforms)
    2. family choice                        (1 choice)
    3. MeasurementPipe.reset                (latency base draw, if enabled)
    4. target generator construction        (its parameter draws)
    5. tier-specific draws LAST (e.g. Tier A's T_lag jitter)
"""
from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np

from . import constants as C
from .corruption import CorruptionConfig, CorruptionModel, Measurement
from .latency import DelayQueue, LatencyModel


def resolve_corruption_cfg(corruption_cfg, enabled_gate: bool
                           ) -> CorruptionConfig:
    """A FRESH CorruptionConfig from a dataclass or a plain mapping (YAML /
    dot-override path), with the env-level enable gate applied. Never
    mutates the caller's object: one fitted config is routinely shared
    across several env instances."""
    if isinstance(corruption_cfg, dict):
        unknown = set(corruption_cfg) - set(CorruptionConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown corruption keys {sorted(unknown)}')
        cfg = CorruptionConfig(**corruption_cfg)
    elif isinstance(corruption_cfg, CorruptionConfig):
        cfg = replace(corruption_cfg)
    else:
        raise TypeError(
            f'corruption_cfg must be CorruptionConfig or dict, '
            f'got {type(corruption_cfg).__name__}')
    cfg.enabled = cfg.enabled and enabled_gate
    return cfg


def draw_reset_placement(rng: np.random.Generator, reset_range: Tuple[float, float],
                         frame_margin: float, families: Sequence[str]
                         ) -> Tuple[float, float, float, str]:
    """The shared reset draw prefix: (z_cam, u0, v0, family)."""
    z_cam = rng.uniform(*reset_range)
    u0 = rng.uniform(C.FRAME_W * frame_margin, C.FRAME_W * (1.0 - frame_margin))
    v0 = rng.uniform(C.FRAME_H * frame_margin, C.FRAME_H * (1.0 - frame_margin))
    family = str(rng.choice(list(families)))
    return z_cam, u0, v0, family


def body_offset_from_frame_draw(z_cam: float, u0: float, v0: float
                                ) -> np.ndarray:
    """Back-project the drawn box centre to a body-frame offset (yaw 0)."""
    x_cam = (u0 - C.CX) * z_cam / C.FX
    y_cam = (v0 - C.CY) * z_cam / C.FY
    return np.array([z_cam, -x_cam, -y_cam])


class MeasurementPipe:
    """truth in -> stale, corrupted detection out, on caller-supplied sim
    time. Owns the DelayQueue + LatencyModel + CorruptionModel triple both
    envs previously wired by hand."""

    def __init__(self, *, latency_enabled: bool, latency_range_s,
                 latency_jitter_std_s: float, corruption_cfg,
                 corruption_enabled: bool):
        self.latency = LatencyModel(enabled=latency_enabled,
                                    base_range_s=latency_range_s,
                                    jitter_std_s=latency_jitter_std_s)
        self.corruption = CorruptionModel(
            resolve_corruption_cfg(corruption_cfg, corruption_enabled))
        self.queue = DelayQueue()

    @classmethod
    def from_env_cfg(cls, cfg) -> 'MeasurementPipe':
        return cls(latency_enabled=cfg.latency,
                   latency_range_s=cfg.latency_range_s,
                   latency_jitter_std_s=cfg.latency_jitter_std_s,
                   corruption_cfg=cfg.corruption_cfg,
                   corruption_enabled=cfg.corruption)

    @property
    def latency_base_s(self) -> float:
        return self.latency.base_s

    def reset(self, rng: np.random.Generator) -> None:
        self.latency.reset(rng)
        self.corruption.reset()
        self.queue.clear()

    def preroll(self, t0: float, dt: float, meas: Optional[Measurement],
                max_latency_s: float, jitter_std_s: float) -> None:
        """Fill the pipe with a short static pre-episode history so the
        first observations sample real (if stale) frames, not an empty
        line -- the pair hovered before ENGAGE."""
        if meas is None:
            return
        n_pre = int(np.ceil((max_latency_s + 4 * jitter_std_s) / dt)) + 1
        for i in range(n_pre, 0, -1):
            self.queue.push(t0 - i * dt, meas)

    def push(self, t: float, meas: Measurement) -> None:
        self.queue.push(t, meas)

    def sample(self, t: float, rng: np.random.Generator
               ) -> Optional[Measurement]:
        delayed = self.queue.sample(t - self.latency.delay(rng))
        payload = delayed[1] if delayed is not None else None
        return self.corruption.apply(payload, rng)
