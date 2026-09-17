"""Follower kinematics and the world -> image projection for Tier A.

The model is deliberately point-kinematic (rl_training_guide.md section 16):
the policy only ever sees the image-plane projection, so simulating the
projection is sufficient -- and validatable. Rigid-body effects (attitude
tilt, yaw-translation coupling) are Tier B's job.

Frames and signs, derived once and unit-tested (test_kinematics.py), because
three simulators + one aircraft = four chances for a silent sign flip
(sim_training_architecture.md section 4.4):

  world:  z up, yaw psi is CCW about z (REP-103).
  body:   x forward, y left, z up.
  camera: x right = -y_body, y down = -z_body, z forward = +x_body
          (the Tello's fixed forward mount, no tilt in Tier A).

Consequences the tests assert:
  +yaw (CCW)      -> a target ahead moves RIGHT in the image (+u).
  +linear.z (up)  -> a target ahead moves DOWN in the image (+v).
  +linear.x (fwd) -> range shrinks -> the box grows.

The follower's achieved velocity follows the commanded velocity through a
first-order lag with time constant T_lag -- measured in Phase 1, randomised
+/-30% per episode until then (guide 16.1 line 2). Yaw rate uses the same
lag: the whole velocity loop sits behind the same link and firmware.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C


@dataclass
class FollowerState:
    """World-frame pose plus the lagged velocity state."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    # Achieved body-frame velocities: [forward, up] m/s and yaw rate rad/s.
    v_fwd: float = 0.0
    v_up: float = 0.0
    yaw_rate: float = 0.0


class FollowerKinematics:
    def __init__(self, t_lag: float):
        if t_lag <= 0:
            raise ValueError(f'T_lag must be positive, got {t_lag}')
        self.t_lag = float(t_lag)
        self.state = FollowerState()

    def reset(self, pos=(0.0, 0.0, 1.0), yaw: float = 0.0) -> None:
        self.state = FollowerState(pos=np.asarray(pos, dtype=float).copy(),
                                   yaw=float(yaw))

    def step(self, v_fwd_cmd: float, v_up_cmd: float, yaw_rate_cmd: float,
             dt: float) -> None:
        """First-order lag toward the commands, then integrate the pose."""
        s = self.state
        alpha = dt / self.t_lag
        # Explicit Euler on v' = (cmd - v)/T_lag; alpha < 1 for dt=0.1,
        # T_lag >= 0.14 s -- guard anyway so a tiny T_lag cannot overshoot.
        alpha = min(alpha, 1.0)
        s.v_fwd += alpha * (v_fwd_cmd - s.v_fwd)
        s.v_up += alpha * (v_up_cmd - s.v_up)
        s.yaw_rate += alpha * (yaw_rate_cmd - s.yaw_rate)

        s.pos[0] += math.cos(s.yaw) * s.v_fwd * dt
        s.pos[1] += math.sin(s.yaw) * s.v_fwd * dt
        s.pos[2] += s.v_up * dt
        s.yaw += s.yaw_rate * dt


def world_to_camera(target_pos: np.ndarray, follower: FollowerState) -> np.ndarray:
    """Target world position -> camera-frame [x right, y down, z forward]."""
    rel = np.asarray(target_pos, dtype=float) - follower.pos
    c, s = math.cos(follower.yaw), math.sin(follower.yaw)
    # Body frame: rotate world rel by -yaw about z.
    bx = c * rel[0] + s * rel[1]        # forward
    by = -s * rel[0] + c * rel[1]       # left
    bz = rel[2]                         # up
    return np.array([-by, -bz, bx])


def project(cam: np.ndarray, ref_width_m: float = C.TELLO_BODY_W):
    """Camera-frame point -> (u, v, w_px, in_frame).

    Pinhole with OUR calibration (constants.py). w_px = fx * W / z is the
    similar-triangles box width the oracle publishes and the standoff rule
    inverts (chase_detector/geometry.py uses the same identity). The FOV test
    is the box CENTRE inside the frame -- the same criterion the Tier-B
    oracle applies (sim_training_architecture.md 3.3 step 4).
    """
    x, y, z = float(cam[0]), float(cam[1]), float(cam[2])
    if z <= 1e-3:
        return 0.0, 0.0, 0.0, False
    u = C.CX + C.FX * x / z
    v = C.CY + C.FY * y / z
    w_px = C.FX * ref_width_m / z
    in_frame = (0.0 <= u < C.FRAME_W) and (0.0 <= v < C.FRAME_H)
    return u, v, w_px, in_frame
