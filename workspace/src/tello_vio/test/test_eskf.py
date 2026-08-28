"""ESKF tests: measurement Jacobians against numerical differentiation, the
stochastic-cloning covariance, and closed-loop convergence on a simulated flight.

Every ``H`` matrix below is checked by perturbing the *error state* through the
same retraction the filter uses and finite-differencing the residual. That is
the only check that actually catches a wrong convention, because a Jacobian
derived under the left-multiplicative convention still looks structurally
correct -- it just makes the filter converge to the wrong answer.
"""
import copy

import numpy as np

from tello_vio import lie
from tello_vio.eskf import (BA, BG, BP, DIM, ErrorStateKF, EskfConfig, P, PC, TH,
                            THC, V, _tangent_basis)

RNG = np.random.default_rng(2024)


def make_kf(seed=0):
    rng = np.random.default_rng(seed)
    kf = ErrorStateKF(EskfConfig(gating=False))
    kf.initialise(0.0, lie.rot_to_quat(lie.Exp(rng.normal(scale=0.3, size=3))), 105.0)
    kf.p = rng.normal(scale=1.0, size=3)
    kf.v = rng.normal(scale=0.5, size=3)
    kf.ba = rng.normal(scale=0.05, size=3)
    kf.bg = rng.normal(scale=0.01, size=3)
    kf.bp = 105.0 + rng.normal(scale=0.2)
    kf.p_c = kf.p + rng.normal(scale=0.3, size=3)
    kf.q_c = lie.quat_boxplus(kf.q, rng.normal(scale=0.15, size=3))
    return kf


def perturb(kf, dx):
    """Apply the filter's own retraction to a copy, so the FD matches _inject."""
    k = copy.deepcopy(kf)
    k.p = k.p + dx[P]
    k.v = k.v + dx[V]
    k.q = lie.quat_boxplus(k.q, dx[TH])
    k.ba = k.ba + dx[BA]
    k.bg = k.bg + dx[BG]
    k.bp = k.bp + dx[BP][0]
    k.p_c = k.p_c + dx[PC]
    k.q_c = lie.quat_boxplus(k.q_c, dx[THC])
    return k


def numeric_H(kf, residual_fn, out_dim, eps=1e-6):
    """FD of -r(x) w.r.t. the error state, i.e. dh/dx, matching the filter's H."""
    J = np.zeros((out_dim, DIM))
    for i in range(DIM):
        d = np.zeros(DIM)
        d[i] = eps
        rp = residual_fn(perturb(kf, d))
        rm = residual_fn(perturb(kf, -d))
        J[:, i] = -(rp - rm) / (2 * eps)  # r = z - h  =>  dh = -dr
    return J


# --------------------------------------------------------------------------- #

def test_attitude_jacobian():
    """Differentiate the *shipped* helper, not a transcription of it."""
    kf = make_kf(1)
    q_meas = lie.quat_boxplus(kf.q, np.array([0.03, -0.02, 0.05]))

    def res(k):
        return k.attitude_hr(q_meas, 0.03, 0.26)[1]

    H = kf.attitude_hr(q_meas, 0.03, 0.26)[0]
    assert np.allclose(H, numeric_H(kf, res, 3), atol=1e-6)


def test_attitude_update_moves_toward_the_measurement():
    """Guards the sign of H: a wrong sign still converges, just the wrong way."""
    kf = ErrorStateKF(EskfConfig(gating=False))
    kf.initialise(0.0, lie.quat_identity(), 100.0)
    target = lie.quat_boxplus(lie.quat_identity(), np.array([0.10, -0.05, 0.20]))
    before = np.linalg.norm(lie.quat_boxminus(target, kf.q))
    for _ in range(50):
        kf.update_attitude(target, np.deg2rad(2.0), np.deg2rad(2.0))
    after = np.linalg.norm(lie.quat_boxminus(target, kf.q))
    assert after < 0.05 * before, f"attitude moved {before} -> {after}"


def test_body_velocity_jacobian():
    kf = make_kf(2)
    v_meas = kf.R.T @ kf.v + np.array([0.05, -0.03, 0.02])

    def res(k):
        return v_meas - k.R.T @ k.v

    Rw = kf.R
    H = np.zeros((3, DIM))
    H[:, V] = Rw.T
    H[:, TH] = lie.skew(Rw.T @ kf.v)
    assert np.allclose(H, numeric_H(kf, res, 3), atol=1e-6)


def test_barometer_jacobian():
    kf = make_kf(3)
    z = kf.p[2] + kf.bp + 0.1

    def res(k):
        return np.array([z - (k.p[2] + k.bp)])

    H = np.zeros((1, DIM))
    H[0, 2] = 1.0
    H[0, BP.start] = 1.0
    assert np.allclose(H, numeric_H(kf, res, 1), atol=1e-6)


def test_tof_jacobian_includes_tilt_compensation():
    kf = make_kf(4)
    kf.q = lie.euler_zyx_to_quat(0.4, 0.15, -0.2)  # a real tilt, not level
    kf.p[2] = 0.8
    ground = 0.0
    z = 0.85

    def res(k):
        return np.array([z - (k.p[2] - ground) / k.R[2, 2]])

    Rw = kf.R
    c = Rw[2, 2]
    dz = kf.p[2] - ground
    H = np.zeros((1, DIM))
    H[0, 2] = 1.0 / c
    H[0, TH] = -dz / c ** 2 * np.cross([0, 0, 1.0], Rw.T @ np.array([0, 0, 1.0]))
    assert np.allclose(H, numeric_H(kf, res, 1), atol=1e-6)


def test_tof_tilt_compensation_actually_matters():
    """At 20 deg of bank the naive (uncompensated) model is ~6 % wrong."""
    R = lie.euler_zyx_to_rot(0.0, np.deg2rad(20.0), 0.0)
    h = 1.0
    slant = h / R[2, 2]
    assert abs(slant - h) / h > 0.05


def test_visual_relative_jacobian_rotation_and_direction():
    kf = make_kf(5)
    R_BC = lie.euler_zyx_to_rot(-np.pi / 2, 0.0, -np.pi / 2)  # FLU body -> optical
    p_BC = np.array([0.03, 0.0, -0.01])

    R_hat, Rc_hat = kf.R, lie.quat_to_rot(kf.q_c)
    A_hat = Rc_hat.T @ R_hat
    w_hat = Rc_hat.T @ (kf.p - kf.p_c)
    d_hat = R_BC.T @ (w_hat - p_BC + A_hat @ p_BC)
    assert np.linalg.norm(d_hat) > 0.05, "need a real baseline for this test"

    # Synthesise a measurement close to (but not equal to) the prediction.
    R_c1c2 = R_BC.T @ A_hat @ R_BC @ lie.Exp([0.01, -0.008, 0.012])
    t_dir = d_hat / np.linalg.norm(d_hat) + np.array([0.02, -0.015, 0.01])
    t_dir /= np.linalg.norm(t_dir)
    args = (R_c1c2, t_dir, R_BC, p_BC, 0.02, 0.05)

    def res(k):
        return k.visual_relative_hr(*args)[1]

    H = kf.visual_relative_hr(*args)[0]
    assert H.shape == (5, DIM)
    assert np.allclose(H, numeric_H(kf, res, 5), atol=1e-5)


def test_visual_rotation_only_path_when_baseline_vanishes():
    """Pure rotation: the essential matrix's direction is noise, so drop it."""
    kf = make_kf(7)
    kf.p_c = kf.p.copy()  # zero baseline
    R_BC = np.eye(3)
    H, r, Rm, kind, dof = kf.visual_relative_hr(
        lie.Exp([0.01, 0.02, -0.01]), np.array([1.0, 0.0, 0.0]),
        R_BC, np.zeros(3), 0.02, 0.05)
    assert kind == "visual_rot" and dof == 3 and H.shape == (3, DIM)


def test_visual_update_moves_position_toward_the_observed_direction():
    """End-to-end sign check on the bearing residual.

    Note the setup: the clone must be followed by real *propagation* before a
    relative measurement can do anything. Straight after ``clone()`` the live
    state and the clone are perfectly correlated, so the filter knows the
    baseline exactly (it is zero) and correctly ignores any relative
    measurement. Only accumulated process noise makes the baseline uncertain
    and therefore correctable -- which is the stochastic-cloning machinery
    behaving as designed.
    """
    kf = ErrorStateKF(EskfConfig(gating=False))
    kf.initialise(0.0, lie.quat_identity(), 100.0)
    kf.clone()
    # Fly +x for 2 s: this both builds a baseline and grows its uncertainty.
    for _ in range(20):
        kf.propagate(np.array([1.0, 0.0, 9.80665]), np.zeros(3), 0.1)
    base0 = kf.p - kf.p_c
    assert np.linalg.norm(base0) > 1.0

    # Truth is 15 deg off the predicted direction, in +y.
    a = np.deg2rad(15.0)
    truth = np.array([np.cos(a), np.sin(a), 0.0])
    ang0 = np.arccos(np.clip(base0 / np.linalg.norm(base0) @ truth, -1, 1))

    for _ in range(60):
        kf.update_visual_relative(np.eye(3), truth, np.eye(3), np.zeros(3), 0.02, 0.02)

    base1 = kf.p - kf.p_c
    ang1 = np.arccos(np.clip(base1 / np.linalg.norm(base1) @ truth, -1, 1))
    assert ang1 < 0.25 * ang0, f"bearing residual pushed the wrong way: {ang0} -> {ang1}"


def test_relative_measurement_is_ignored_immediately_after_cloning():
    """The clone's zero-uncertainty baseline must not be moved by vision."""
    kf = ErrorStateKF(EskfConfig(gating=False))
    kf.initialise(0.0, lie.quat_identity(), 100.0)
    kf.clone()
    kf.p = np.array([1.0, 0.0, 0.0])  # inconsistent mean, deterministic baseline
    before = kf.p.copy()
    kf.update_visual_relative(np.eye(3), np.array([0.0, 1.0, 0.0]),
                              np.eye(3), np.zeros(3), 0.02, 0.02)
    assert np.allclose(kf.p, before, atol=1e-12)


def test_tangent_basis_is_orthonormal_and_perpendicular():
    for _ in range(500):
        u = RNG.normal(size=3)
        u /= np.linalg.norm(u)
        B = _tangent_basis(u)
        assert np.allclose(B.T @ B, np.eye(2), atol=1e-12)
        assert np.allclose(B.T @ u, 0.0, atol=1e-12)


def test_clone_copies_covariance_and_cross_terms():
    kf = make_kf(6)
    kf.P = np.eye(DIM) * 0.01 + RNG.normal(scale=1e-3, size=(DIM, DIM))
    kf.P = kf.P @ kf.P.T
    kf.clone()
    assert np.allclose(kf.p_c, kf.p)
    assert np.allclose(kf.q_c, kf.q)
    # Right after cloning the clone and the live state are perfectly correlated,
    # so their covariance blocks and cross-block must all coincide.
    assert np.allclose(kf.P[PC, PC], kf.P[P, P], atol=1e-12)
    assert np.allclose(kf.P[THC, THC], kf.P[TH, TH], atol=1e-12)
    assert np.allclose(kf.P[PC, P], kf.P[P, P], atol=1e-12)
    assert np.allclose(kf.P[PC, V], kf.P[P, V], atol=1e-12)
    assert np.allclose(kf.P[THC, BG], kf.P[TH, BG], atol=1e-12)


def test_covariance_stays_psd_over_a_long_run():
    kf = ErrorStateKF(EskfConfig())
    kf.initialise(0.0, lie.quat_identity(), 100.0)
    rng = np.random.default_rng(3)
    for k in range(2000):
        acc = np.array([0.0, 0.0, 9.80665]) + rng.normal(scale=0.1, size=3)
        gyro = rng.normal(scale=0.05, size=3)
        kf.propagate(acc, gyro, 0.1)
        if k % 2 == 0:
            kf.update_body_velocity(rng.normal(scale=0.1, size=3), 0.15)
            kf.update_barometer(100.0 + rng.normal(scale=0.3), 0.3)
        if k % 20 == 0:
            kf.clone()
    eig = np.linalg.eigvalsh(kf.P)
    assert eig.min() > -1e-9, f"covariance lost PSD, min eig {eig.min()}"
    assert np.all(np.isfinite(kf.P))


def test_zupt_and_zaru_drive_bias_estimates_to_truth():
    """The core claim behind stationary-phase calibration."""
    true_ba = np.array([0.20, -0.15, 0.10])
    true_bg = np.array([0.02, -0.03, 0.015])
    kf = ErrorStateKF(EskfConfig())
    kf.initialise(0.0, lie.quat_identity(), 100.0)
    rng = np.random.default_rng(4)
    dt = 0.1
    for _ in range(600):
        acc = np.array([0.0, 0.0, 9.80665]) + true_ba + rng.normal(scale=0.03, size=3)
        gyro = true_bg + rng.normal(scale=0.01, size=3)
        kf.propagate(acc, gyro, dt)
        kf.update_zero_velocity(0.02)
        kf.update_zero_angular_rate(gyro, 0.02)
        kf.update_attitude(lie.quat_identity(), np.deg2rad(2.0), np.deg2rad(15.0))
        kf.update_barometer(100.0 + rng.normal(scale=0.3), 0.3)
    assert np.linalg.norm(kf.bg - true_bg) < 0.01, kf.bg
    # Accel bias in x/y is observable through the attitude constraint; the z
    # component trades off against the gravity magnitude and converges slower.
    assert np.linalg.norm((kf.ba - true_ba)[:2]) < 0.06, kf.ba
    assert np.linalg.norm(kf.v) < 0.05


def test_gating_rejects_a_gross_outlier():
    kf = ErrorStateKF(EskfConfig(gating=True))
    kf.initialise(0.0, lie.quat_identity(), 100.0)
    for _ in range(30):
        kf.propagate(np.array([0.0, 0.0, 9.80665]), np.zeros(3), 0.1)
        kf.update_barometer(100.0, 0.3)
    before = kf.p[2]
    accepted = kf.update_barometer(180.0, 0.3)  # 80 m jump: impossible
    assert not accepted
    assert abs(kf.p[2] - before) < 1e-9
