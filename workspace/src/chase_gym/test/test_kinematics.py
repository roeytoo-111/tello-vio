"""Sign conventions and projection -- tested once, recorded, never assumed
(sim_training_architecture.md 3.2 item 4: the standing sign-test rule)."""
import math

import numpy as np
import pytest

from chase_gym import constants as C
from chase_gym.kinematics import (FollowerKinematics, FollowerState, project,
                                  world_to_camera)


def _uvw(target, follower_state, ref=C.TELLO_BODY_W):
    return project(world_to_camera(np.asarray(target, float), follower_state),
                   ref)


def test_target_ahead_projects_to_centre():
    st = FollowerState(pos=np.array([0.0, 0.0, 1.0]))
    u, v, w, in_frame = _uvw([2.0, 0.0, 1.0], st)
    assert in_frame
    assert u == pytest.approx(C.CX)
    assert v == pytest.approx(C.CY)
    # Oracle box-width table [V]: body 0.098 m at 2 m -> ~45 px.
    assert w == pytest.approx(C.FX * C.TELLO_BODY_W / 2.0)
    assert w == pytest.approx(45.0, abs=1.0)


def test_positive_yaw_moves_image_right():
    """+yaw (CCW, REP-103) must move a target ahead RIGHT in the image (+u).
    This is the convention the checkpoint action map records."""
    st = FollowerState(pos=np.array([0.0, 0.0, 1.0]), yaw=0.0)
    u0, _, _, _ = _uvw([2.0, 0.0, 1.0], st)
    st.yaw = +0.1
    u1, _, _, _ = _uvw([2.0, 0.0, 1.0], st)
    assert u1 > u0


def test_climb_moves_image_down():
    """+linear.z (up) must move the target DOWN in the image (+v)."""
    st = FollowerState(pos=np.array([0.0, 0.0, 1.0]))
    _, v0, _, _ = _uvw([2.0, 0.0, 1.0], st)
    st.pos[2] += 0.3
    _, v1, _, _ = _uvw([2.0, 0.0, 1.0], st)
    assert v1 > v0


def test_forward_grows_box():
    st = FollowerState(pos=np.array([0.0, 0.0, 1.0]))
    _, _, w0, _ = _uvw([2.0, 0.0, 1.0], st)
    st.pos[0] += 0.5
    _, _, w1, _ = _uvw([2.0, 0.0, 1.0], st)
    assert w1 > w0


def test_target_left_is_negative_ex():
    """World +y is body-left at yaw 0 -> image left (u < cx)."""
    st = FollowerState(pos=np.array([0.0, 0.0, 1.0]))
    u, _, _, _ = _uvw([2.0, 0.5, 1.0], st)
    assert u < C.CX


def test_fov_test_matches_frame_bounds():
    st = FollowerState(pos=np.array([0.0, 0.0, 1.0]))
    # Half the H-FOV (55.1 deg [V]) puts the centre exactly at the edge.
    half = math.radians(C.HFOV_DEG / 2.0)
    just_in = [2.0, -2.0 * math.tan(half) * 0.999, 1.0]
    just_out = [2.0, -2.0 * math.tan(half) * 1.001, 1.0]
    assert _uvw(just_in, st)[3]
    assert not _uvw(just_out, st)[3]


def test_behind_camera_is_not_in_frame():
    st = FollowerState(pos=np.array([0.0, 0.0, 1.0]))
    assert not _uvw([-1.0, 0.0, 1.0], st)[3]


def test_first_order_lag_63_percent_at_t_lag():
    """v(T_lag) = (1 - 1/e) * v_cmd for a step command -- the calibration
    identity the Phase-1 measurement will use."""
    t_lag = 0.25
    fk = FollowerKinematics(t_lag)
    fk.reset()
    dt = 0.001
    for _ in range(int(t_lag / dt)):
        fk.step(1.0, 0.0, 0.0, dt)
    assert fk.state.v_fwd == pytest.approx(1.0 - math.exp(-1.0), abs=0.01)


def test_yaw_integration():
    fk = FollowerKinematics(0.01)     # near-instant lag
    fk.reset()
    for _ in range(100):
        fk.step(0.0, 0.0, 0.5, 0.01)  # 0.5 rad/s for 1 s
    assert fk.state.yaw == pytest.approx(0.5, abs=0.02)


def test_hfov_constant_matches_doc():
    assert C.HFOV_DEG == pytest.approx(55.1, abs=0.1)   # [V] 55.1 deg
    assert C.VFOV_DEG == pytest.approx(42.8, abs=0.2)
    assert C.MAX_DIST_PX == pytest.approx(600.0)
