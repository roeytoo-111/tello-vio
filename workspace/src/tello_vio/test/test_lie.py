"""Correctness tests for the SO(3)/SE(3) utilities.

These are pure NumPy and require neither ROS nor a drone, so they are the first
line of defence against sign/convention regressions.
"""
import numpy as np

from tello_vio import lie


RNG = np.random.default_rng(0xC0FFEE)


def rand_rotvec(scale=1.0):
    v = RNG.normal(size=3)
    v /= np.linalg.norm(v)
    return v * RNG.uniform(0.0, np.pi * scale)


def test_exp_log_roundtrip():
    for _ in range(500):
        phi = rand_rotvec(0.999)
        assert np.allclose(lie.Log(lie.Exp(phi)), phi, atol=1e-9)


def test_exp_log_roundtrip_tiny_and_near_pi():
    for mag in [0.0, 1e-12, 1e-9, 1e-6, 1e-4, np.pi - 1e-6, np.pi - 1e-9]:
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        phi = axis * mag
        R = lie.Exp(phi)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.allclose(lie.Log(R), phi, atol=1e-6)


def test_exp_is_a_rotation():
    for _ in range(200):
        R = lie.Exp(rand_rotvec())
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_right_jacobian_matches_finite_difference():
    """Jr is *defined* by Exp(phi + d) = Exp(phi) Exp(Jr(phi) d)."""
    for _ in range(200):
        phi = rand_rotvec(0.9)
        Jr = lie.right_jacobian(phi)
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1e-7
            lhs = lie.Exp(phi + d)
            rhs = lie.Exp(phi) @ lie.Exp(Jr @ d)
            assert np.allclose(lhs, rhs, atol=1e-9)


def test_right_jacobian_inverse():
    for _ in range(300):
        phi = rand_rotvec(0.95)
        assert np.allclose(lie.right_jacobian(phi) @ lie.right_jacobian_inv(phi), np.eye(3), atol=1e-9)


def test_quat_rot_roundtrip():
    for _ in range(500):
        R = lie.Exp(rand_rotvec(0.99))
        q = lie.rot_to_quat(R)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
        assert np.allclose(lie.quat_to_rot(q), R, atol=1e-10)


def test_quat_mul_matches_matrix_product():
    for _ in range(300):
        Ra, Rb = lie.Exp(rand_rotvec()), lie.Exp(rand_rotvec())
        qa, qb = lie.rot_to_quat(Ra), lie.rot_to_quat(Rb)
        assert np.allclose(lie.quat_to_rot(lie.quat_mul(qa, qb)), Ra @ Rb, atol=1e-10)


def test_quat_exp_log_agree_with_matrix_versions():
    for _ in range(300):
        phi = rand_rotvec(0.99)
        assert np.allclose(lie.quat_to_rot(lie.quat_exp(phi)), lie.Exp(phi), atol=1e-10)
        assert np.allclose(lie.quat_log(lie.rot_to_quat(lie.Exp(phi))), phi, atol=1e-9)


def test_boxplus_boxminus_are_inverses():
    for _ in range(300):
        q = lie.rot_to_quat(lie.Exp(rand_rotvec()))
        d = RNG.normal(size=3) * 0.3
        assert np.allclose(lie.quat_boxminus(lie.quat_boxplus(q, d), q), d, atol=1e-9)


def test_euler_roundtrip():
    for _ in range(500):
        yaw = RNG.uniform(-np.pi, np.pi)
        pitch = RNG.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)
        roll = RNG.uniform(-np.pi, np.pi)
        R = lie.euler_zyx_to_rot(yaw, pitch, roll)
        y2, p2, r2 = lie.rot_to_euler_zyx(R)
        assert np.allclose(lie.euler_zyx_to_rot(y2, p2, r2), R, atol=1e-10)


def test_euler_zyx_axis_semantics():
    """Yaw about +z, pitch about +y, roll about +x, right-handed."""
    assert np.allclose(lie.euler_zyx_to_rot(np.pi / 2, 0, 0) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(lie.euler_zyx_to_rot(0, np.pi / 2, 0) @ [1, 0, 0], [0, 0, -1], atol=1e-12)
    assert np.allclose(lie.euler_zyx_to_rot(0, 0, np.pi / 2) @ [0, 1, 0], [0, 0, 1], atol=1e-12)


def test_project_to_so3_recovers_rotation():
    for _ in range(200):
        R = lie.Exp(rand_rotvec())
        noisy = R + RNG.normal(scale=1e-3, size=(3, 3))
        Rp = lie.project_to_so3(noisy)
        assert np.isclose(np.linalg.det(Rp), 1.0, atol=1e-12)
        assert np.allclose(Rp, R, atol=5e-3)


def test_se3_inverse():
    for _ in range(200):
        T = lie.se3(lie.Exp(rand_rotvec()), RNG.normal(size=3))
        assert np.allclose(lie.se3_inv(T) @ T, np.eye(4), atol=1e-12)


def test_so3_mean_of_perturbations():
    R0 = lie.Exp(rand_rotvec(0.5))
    rots = [R0 @ lie.Exp(RNG.normal(scale=0.02, size=3)) for _ in range(400)]
    assert np.linalg.norm(lie.Log(R0.T @ lie.so3_mean(rots))) < 5e-3
