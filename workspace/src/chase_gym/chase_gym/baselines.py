"""Null baselines: the P-controller and proportional navigation.

The P-controller is the denominator of every RL claim (rl_specification.md
10: "does the learned policy beat a P controller?") and the demonstration
source for buffer prefill (guide 17). PN is the classical interception law
the INTERCEPT variant must beat (sim_training_architecture.md 1): if the
learned policy does not beat PN under latency, ship PN.

Both expose get_action(obs) -> action on the SAME latency-aware observation
vector the policy sees -- the controller-swap-as-parameter pattern
(drl_ros2_reference_analysis.md 3.1). They read only observation entries,
never env internals: a baseline with privileged state would be a rigged
comparison.
"""
import numpy as np

from . import constants as C
from .observation import ObservationSpec


class PController:
    """Proportional centring on (ex, ey).

    Signs derive from the tested camera geometry (kinematics.py):
      +yaw moves the box right (+ex)  -> a_yaw = -k * ex
      +up  moves the box down  (+ey)  -> a_up  = -k * ey
    On a miss (visible = 0) it holds zero -- never chases a ghost.
    """

    def __init__(self, spec: ObservationSpec = None, k_v: float = 1.2,
                 k_h: float = 1.2, rate_limit: float = 0.5):
        self.spec = spec or ObservationSpec()
        if self.spec.mode != 'latency_aware':
            raise ValueError('PController reads the latency-aware layout')
        self.k_v = float(k_v)
        self.k_h = float(k_h)
        self.rate_limit = float(rate_limit)
        self._prev = np.zeros(2, dtype=np.float32)

    def reset(self):
        self._prev = np.zeros(2, dtype=np.float32)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ex, ey, visible = float(obs[0]), float(obs[1]), float(obs[4])
        if visible < 0.5:
            raw = np.zeros(2, dtype=np.float32)
        else:
            # Action layout [a_v, a_h] = [linear.z, angular.z].
            raw = np.array([-self.k_v * ey, -self.k_h * ex],
                           dtype=np.float32)
        raw = np.clip(raw, -1.0, 1.0)
        # Rate limit per step: the spec's anti-judder recommendation
        # (rl_specification.md 4) applied to the baseline too, so the
        # comparison is against its best self.
        delta = np.clip(raw - self._prev, -self.rate_limit, self.rate_limit)
        out = np.clip(self._prev + delta, -1.0, 1.0)
        self._prev = out
        return out


class PNController:
    """Proportional navigation in the yaw plane, PD pursuit vertically.

    Classical PN: commanded turn rate = N * lambda_dot, with lambda the
    INERTIAL line-of-sight angle (CCW+). The camera measures only the body
    bearing az = psi - lambda (right+, small-angle: ex * half_w / fx), so
    lambda_dot must be reconstructed:

        lambda_dot = psi_dot - az_dot

    az_dot comes from the one-step error difference already in the
    observation (entries 2-3), and psi_dot from the last action SENT
    (entry 7 = a_h(t-1), the newest history slot) times omega_max -- both
    observation-only quantities, so the baseline holds no privileged state.
    Sign check (unit-tested): a target drifting right with the follower
    still gives az_dot > 0, lambda_dot < 0, turn command clockwise (a_h < 0)
    -- toward the target's motion, as PN must.

    The vertical axis is a translation, not a rotation, so PN's turn-rate
    form does not apply; a PD pursuit on elevation (with the same
    finite-difference lead term) is the standard companion law.
    A small centring P-term keeps the target in frame between manoeuvres.
    """

    def __init__(self, spec: ObservationSpec = None, nav_gain: float = 3.0,
                 k_center: float = 0.4, k_v: float = 1.0, k_v_lead: float = 3.0,
                 omega_max: float = 1.5):
        self.spec = spec or ObservationSpec()
        if self.spec.mode != 'latency_aware':
            raise ValueError('PNController reads the latency-aware layout')
        self.nav_gain = float(nav_gain)
        self.k_center = float(k_center)
        self.k_v = float(k_v)
        self.k_v_lead = float(k_v_lead)
        self.omega_max = float(omega_max)
        self._prev_visible = False

    def reset(self):
        self._prev_visible = False

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ex, ey = float(obs[0]), float(obs[1])
        ex_prev, ey_prev = float(obs[2]), float(obs[3])
        visible = float(obs[4])
        a_h_prev = float(obs[7])         # newest history entry, [a_v, a_h]
        if visible < 0.5:
            self._prev_visible = False
            return np.zeros(2, dtype=np.float32)
        # Rate terms only when the PREVIOUS frame was also visible: the
        # assembler zeroes errors on a miss (anti-ghosting), so the first
        # frame after a dropout would difference a real bearing against an
        # artificial zero -- a fabricated LOS rate that saturates the turn
        # command in the wrong direction exactly when PN must be at its
        # best. Reacquisition frames fly on the centring terms alone.
        use_rates = self._prev_visible
        self._prev_visible = True
        dt = self.spec.control_dt_s
        if use_rates:
            # Bearing rate, rad/s, right-positive (small-angle conversion).
            az_dot = (ex - ex_prev) * self.spec.half_w / C.FX / dt
            psi_dot = a_h_prev * self.omega_max
            lambda_dot = psi_dot - az_dot
            yaw_rate_cmd = self.nav_gain * lambda_dot     # rad/s, CCW+
            lead_v = -self.k_v_lead * (ey - ey_prev)
        else:
            yaw_rate_cmd = 0.0
            lead_v = 0.0
        a_h = yaw_rate_cmd / self.omega_max - self.k_center * ex
        a_v = -self.k_v * ey + lead_v
        return np.clip(np.array([a_v, a_h], dtype=np.float32), -1.0, 1.0)
