"""THE observation assembler -- one module, every tier, and flight.

Implements rl_training_guide.md 4.3 exactly. This module is imported by the
Tier-A env, the Tier-B GzChaseEnv, the Tier-C evaluation loop and (later)
the chase_state deployment node; that single import is what makes
train/deploy skew structurally impossible (guide 21).

Layout (latency_aware mode, dimension 6 + 2k, k = 4 -> 14):

  0-1  ex, ey        (u - 480)/480, (v - 360)/360, clipped to [-1, 1]
  2-3  ex_prev, ey_prev
  4    visible       1.0 if a detection arrived THIS step, else 0.0
  5    staleness     time since last detection / loss_timeout, clipped [0, 1]
  6..  a_{t-1} .. a_{t-k}   the last k actions SENT (post-clamp), newest first

Corruption rules (guide 4.3): on a miss the errors are ZEROED and visible=0
-- never held (holding trains ghost-chasing); the history stores what was
sent, not what the actor proposed.

The faithful arm bypasses this module entirely: its observation is the raw
box-centre pixels of the original code [C drone_sim_env.py:24-33], produced
by FaithfulPointMassEnv itself.
"""
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import constants as C
from .corruption import Measurement


@dataclass(frozen=True)
class ObservationSpec:
    """Everything an inference node needs to reproduce the observation --
    the checkpoint carries this and refuses to run without a match
    (guide 19)."""
    mode: str = 'latency_aware'          # or 'raw_pixels' (faithful arm)
    k: int = C.K_ACTION_HISTORY
    action_dim: int = 2
    cx: float = C.CX
    cy: float = C.CY
    half_w: float = C.CX
    half_h: float = C.CY
    loss_timeout_s: float = 1.0          # [D-impl 6] staleness normaliser
    control_dt_s: float = C.DT

    @property
    def dim(self) -> int:
        if self.mode == 'raw_pixels':
            return 2
        return 6 + self.action_dim * self.k

    def to_dict(self) -> dict:
        d = asdict(self)
        d['dim'] = self.dim
        return d

    def spec_hash(self) -> str:
        """Stable hash over the canonical JSON -- the refusal handle."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class ObservationAssembler:
    def __init__(self, spec: Optional[ObservationSpec] = None):
        self.spec = spec or ObservationSpec()
        if self.spec.mode not in ('latency_aware', 'raw_pixels'):
            raise ValueError(f'unknown observation mode {self.spec.mode!r}')
        self.reset()

    def reset(self) -> None:
        s = self.spec
        self._ex_prev = 0.0
        self._ey_prev = 0.0
        self._since_detection_s = s.loss_timeout_s   # starts fully stale
        # Preallocated ring, newest at _head: assemble() runs once per
        # control step on the hot path -- no per-step allocation.
        self._history = np.zeros((s.k, s.action_dim), dtype=np.float32)
        self._head = 0

    def record_action(self, action_sent: np.ndarray) -> None:
        """Store what was actually SENT (post-clamp), newest first."""
        a = np.clip(np.asarray(action_sent, dtype=np.float32).reshape(-1),
                    -1.0, 1.0)
        if a.shape[0] != self.spec.action_dim:
            raise ValueError(
                f'action dim {a.shape[0]} != spec {self.spec.action_dim}')
        self._head = (self._head - 1) % self.spec.k
        self._history[self._head] = a

    def assemble(self, meas: Optional[Measurement], dt_s: float) -> np.ndarray:
        """One observation from this step's (possibly missing) measurement.

        `meas` is the delayed+corrupted detection the pipe delivered this
        control step, or None on a miss. `dt_s` is the real step period (in
        sim: the control period; in flight: the measured tick).
        """
        s = self.spec
        if s.mode == 'raw_pixels':
            if meas is None:
                # The original's training world has no misses, and its
                # deployment loop zeroes commands instead of invoking the
                # actor [C main.py:236-248]. Feeding the frame centre makes
                # the actor's correction ~zero -- the closest observation-
                # side approximation when arm F is evaluated in this env.
                return np.array([C.CX, C.CY], dtype=np.float32)
            return np.array([meas.u, meas.v], dtype=np.float32)

        if meas is not None:
            ex = float(np.clip((meas.u - s.cx) / s.half_w, -1.0, 1.0))
            ey = float(np.clip((meas.v - s.cy) / s.half_h, -1.0, 1.0))
            visible = 1.0
            self._since_detection_s = 0.0
        else:
            ex = 0.0
            ey = 0.0
            visible = 0.0
            self._since_detection_s += dt_s
        staleness = min(self._since_detection_s / s.loss_timeout_s, 1.0)

        obs = np.empty(s.dim, dtype=np.float32)
        obs[0] = ex
        obs[1] = ey
        obs[2] = self._ex_prev
        obs[3] = self._ey_prev
        obs[4] = visible
        obs[5] = staleness
        hist = obs[6:].reshape(s.k, s.action_dim)
        n_tail = s.k - self._head
        hist[:n_tail] = self._history[self._head:]
        hist[n_tail:] = self._history[:self._head]
        self._ex_prev = ex
        self._ey_prev = ey

        assert obs.shape == (s.dim,)
        if not np.all(np.isfinite(obs)):
            # A non-finite value must never reach the network
            # (drl_ros2_reference_analysis.md 3.4).
            raise FloatingPointError(f'non-finite observation: {obs}')
        return obs
