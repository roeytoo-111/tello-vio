"""The reward module -- truth in, scalar out, every tier the same.

Two tracking variants plus the INTERCEPT extension, exactly as specified:

* `repaired` (guide 16.2): the original code's threshold-100 core with the
  one-term continuity repair and the /100 scaling (provably policy-invariant,
  Ng et al. 1999 via guide 22), plus the smoothness and terminal-loss terms.
  Anchors, unit-tested: r(0)=1.0, r(50)=0.5, r(100)=0.0, r(300)=-0.5,
  r(600)=-1.25.

* `faithful` (rl_block_diagram.md 5): the published code's reward verbatim --
  threshold 100, unscaled, DISCONTINUOUS (0 just inside, -25 just outside).
  reward = 100 - dist   if dist <= 100
         = -0.25 * dist if dist  > 100

* INTERCEPT adds (sim_training_architecture.md 1): potential-based range
  shaping F = gamma*Phi(s') - Phi(s) with Phi = -lambda*range -- the Ng-form
  that provably cannot move the optimum -- and a terminal capture bonus B
  that CAN, by design: capture is the task.

The reward is computed from the TRUE, CURRENT geometry while the observation
comes through the delay-and-noise pipe (guide 16.1 line 8): acting on old
data is punished by the true state of the world.
"""
from dataclasses import dataclass

import numpy as np

from . import constants as C


def track_repaired(dist_px: float) -> float:
    """Continuous, scaled tracking core. In (0, 1] inside the disc."""
    if dist_px <= C.REWARD_THRESHOLD_PX:
        return (C.REWARD_THRESHOLD_PX - dist_px) / 100.0
    return -0.25 * (dist_px - C.REWARD_THRESHOLD_PX) / 100.0


def track_faithful(dist_px: float) -> float:
    """The published code's branch, cliff included [C drone_sim_env.py:70]."""
    if dist_px <= C.REWARD_THRESHOLD_PX:
        return C.REWARD_THRESHOLD_PX - dist_px
    return -0.25 * dist_px


@dataclass
class RewardConfig:
    version: str = 'repaired'        # 'repaired' | 'faithful'
    w_smooth: float = 0.05           # guide 16.2 initial value
    w_loss: float = 5.0
    # INTERCEPT terms (ignored for FOLLOW):
    intercept: bool = False
    gamma: float = 0.99              # must equal the trainer's gamma
    shaping_lambda: float = 0.1
    capture_bonus: float = 10.0


class RewardComputer:
    def __init__(self, cfg: RewardConfig):
        if cfg.version not in ('repaired', 'faithful'):
            raise ValueError(f'unknown reward version {cfg.version!r}')
        self.cfg = cfg

    def compute(self, *, dist_px: float, action: np.ndarray,
                prev_action: np.ndarray, lost: bool, captured: bool = False,
                range_m: float = 0.0, prev_range_m: float = 0.0) -> float:
        """One step's reward from TRUE geometry.

        dist_px:  true projected centre error (even if the detector dropped
                  the frame -- reward is training-time information).
        lost:     this step terminates with the target out of frame.
        captured: this step terminates with an INTERCEPT capture.
        """
        cfg = self.cfg
        if cfg.version == 'faithful':
            # Verbatim original: no smoothness, no loss term, no shaping.
            return track_faithful(dist_px)

        r = track_repaired(dist_px)
        diff = np.asarray(action, dtype=float) - np.asarray(prev_action, dtype=float)
        r -= cfg.w_smooth * float(diff @ diff)
        if lost:
            r -= cfg.w_loss
        if cfg.intercept:
            # Potential-based shaping, Ng-form; on terminal steps the
            # convention Phi(terminal) = 0 keeps the telescoping sum
            # policy-independent.
            phi_next = 0.0 if (lost or captured) else -cfg.shaping_lambda * range_m
            phi_prev = -cfg.shaping_lambda * prev_range_m
            r += cfg.gamma * phi_next - phi_prev
            if captured:
                r += cfg.capture_bonus
        return r
