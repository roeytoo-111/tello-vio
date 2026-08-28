"""Validation of IMU preintegration.

Four independent checks, because each catches a different class of bug:

1. *Exactness* against the discrete propagation model the measurements were
   synthesised from -- catches sign errors and misplaced ``dt``.
2. *Bias Jacobians* against finite differences over a full re-integration --
   catches the update-ordering trap in Forster's appendix B.
3. *Residual Jacobians* against on-manifold numerical differentiation --
   catches wrong retraction / wrong right-Jacobian usage.
4. *Covariance* against Monte Carlo -- catches a wrong noise discretisation,
   which is invisible in the mean and silently breaks filter consistency.
"""
import numpy as np
import pytest

from tello_vio import lie
from tello_vio.preintegration import GRAVITY_ENU, ImuNoiseModel, PreintegratedImu

RNG = np.random.default_rng(7)


def synth_imu(n=200, dt=0.005, seed=1):
    """Synthesise a wiggly trajectory and the *exact* IMU readings for it.

    The trajectory is produced by the same discrete recursion the
    preintegration model assumes, so check 1 can demand machine precision.
    """
    rng = np.random.default_rng(seed)
    R = lie.Exp(rng.normal(scale=0.4, size=3))
    v = rng.normal(scale=0.5, size=3)
    p = rng.normal(scale=1.0, size=3)
    R0, v0, p0 = R.copy(), v.copy(), p.copy()

    accs, gyros = [], []
    for k in range(n):
        t = k * dt
        # Body-frame specific force and rate; amplitudes are drone-like
        # (a few m/s^2, up to ~1 rad/s).
        w = np.array([0.6 * np.sin(1.7 * t), 0.4 * np.cos(2.3 * t), 0.25 + 0.3 * np.sin(0.9 * t)])
        a_world = np.array([1.5 * np.sin(2.1 * t), 1.1 * np.cos(1.3 * t), 0.8 * np.sin(0.7 * t)])
        f_body = R.T @ (a_world - GRAVITY_ENU)  # specific force = a - g
        accs.append(f_body)
        gyros.append(w)

        a_w = R @ f_body + GRAVITY_ENU
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        R = R @ lie.Exp(w * dt)

    return (np.array(accs), np.array(gyros), dt,
            (R0, v0, p0), (R, v, p))


def make_pim(accs, gyros, dt, ba=None, bg=None, noise=None):
    ba = np.zeros(3) if ba is None else ba
    bg = np.zeros(3) if bg is None else bg
    pim = PreintegratedImu(noise or ImuNoiseModel(), ba, bg)
    for a, w in zip(accs, gyros):
        pim.integrate(a, w, dt)
    return pim


# --------------------------------------------------------------------------- #
# 1. exactness
# --------------------------------------------------------------------------- #

def test_predict_reproduces_the_generating_trajectory():
    accs, gyros, dt, (R0, v0, p0), (Rn, vn, pn) = synth_imu()
    pim = make_pim(accs, gyros, dt)
    R, v, p = pim.predict(R0, v0, p0)
    assert np.allclose(R, Rn, atol=1e-11)
    assert np.allclose(v, vn, atol=1e-10)
    assert np.allclose(p, pn, atol=1e-10)
    assert np.isclose(pim.dt, len(accs) * dt)


def test_residual_is_zero_at_the_true_states():
    accs, gyros, dt, (R0, v0, p0), (Rn, vn, pn) = synth_imu()
    pim = make_pim(accs, gyros, dt)
    r = pim.residual(R0, v0, p0, Rn, vn, pn)
    assert np.linalg.norm(r) < 1e-9


def test_residual_is_nonzero_when_a_state_is_wrong():
    accs, gyros, dt, (R0, v0, p0), (Rn, vn, pn) = synth_imu()
    pim = make_pim(accs, gyros, dt)
    r = pim.residual(R0, v0, p0, Rn, vn, pn + np.array([0.1, 0.0, 0.0]))
    assert np.linalg.norm(r) > 0.05


def test_stationary_level_imu_produces_no_motion():
    """A level IMU at rest reads [0,0,+9.81] and must integrate to zero motion."""
    n, dt = 400, 0.01
    accs = np.tile(-GRAVITY_ENU, (n, 1))
    gyros = np.zeros((n, 3))
    pim = make_pim(accs, gyros, dt)
    R, v, p = pim.predict(np.eye(3), np.zeros(3), np.zeros(3))
    assert np.allclose(R, np.eye(3), atol=1e-12)
    assert np.allclose(v, 0.0, atol=1e-12)
    # 400 steps of +-0.5*g*dt^2 cancelling leaves ~1e-12 m of float residue;
    # that is 11 orders of magnitude below anything physically meaningful.
    assert np.allclose(p, 0.0, atol=1e-9)


def test_freefall_imu_reads_zero_and_falls():
    """In free fall the accelerometer reads zero and the body follows gravity."""
    n, dt = 100, 0.01
    pim = make_pim(np.zeros((n, 3)), np.zeros((n, 3)), dt)
    T = n * dt
    _, v, p = pim.predict(np.eye(3), np.zeros(3), np.zeros(3))
    assert np.allclose(v, GRAVITY_ENU * T, atol=1e-12)
    assert np.allclose(p, 0.5 * GRAVITY_ENU * T * T, atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. bias Jacobians
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("axis", range(3))
def test_bias_jacobians_match_full_reintegration(axis):
    accs, gyros, dt, _, _ = synth_imu(n=120, dt=0.005)
    ba0, bg0 = np.full(3, 0.02), np.full(3, 0.003)
    pim = make_pim(accs, gyros, dt, ba0, bg0)

    eps = 1e-6
    for which in ("acc", "gyro"):
        d = np.zeros(3)
        d[axis] = eps
        ba1 = ba0 + (d if which == "acc" else 0.0)
        bg1 = bg0 + (d if which == "gyro" else 0.0)

        # Ground truth: re-preintegrate from scratch at the perturbed bias.
        ref = make_pim(accs, gyros, dt, ba1, bg1)
        # Prediction: first-order extrapolation from the linearisation point.
        dR, dv, dp, _ = pim.bias_corrected_delta(ba1, bg1)

        assert np.linalg.norm(lie.Log(ref.dR.T @ dR)) < 1e-9
        assert np.allclose(dv, ref.dv, atol=1e-8)
        assert np.allclose(dp, ref.dp, atol=1e-8)


def test_bias_correction_beats_ignoring_the_bias():
    """Sanity: the correction must actually reduce error for a realistic bias step."""
    accs, gyros, dt, _, _ = synth_imu(n=200, dt=0.005)
    ba0, bg0 = np.zeros(3), np.zeros(3)
    pim = make_pim(accs, gyros, dt, ba0, bg0)
    ba1, bg1 = np.array([0.05, -0.03, 0.02]), np.array([0.004, 0.002, -0.003])
    ref = make_pim(accs, gyros, dt, ba1, bg1)

    _, dv_c, dp_c, _ = pim.bias_corrected_delta(ba1, bg1)
    err_corrected = np.linalg.norm(dp_c - ref.dp)
    err_ignored = np.linalg.norm(pim.dp - ref.dp)
    assert err_corrected < 0.02 * err_ignored


# --------------------------------------------------------------------------- #
# 3. residual Jacobians
# --------------------------------------------------------------------------- #

def _numeric_jac(f, x_boxplus, dim, eps=1e-6):
    r0 = f(np.zeros(dim))
    J = np.zeros((r0.size, dim))
    for k in range(dim):
        d = np.zeros(dim)
        d[k] = eps
        J[:, k] = (f(d) - f(-d)) / (2 * eps)
    return J


def test_residual_jacobians_match_numerical_differentiation():
    accs, gyros, dt, (R0, v0, p0), (Rn, vn, pn) = synth_imu(n=80, dt=0.005)
    ba, bg = np.array([0.01, -0.02, 0.005]), np.array([0.002, -0.001, 0.003])
    pim = make_pim(accs, gyros, dt, ba, bg)

    # Perturb away from the exact solution so the residual is not at a
    # degenerate point.
    Ri = R0 @ lie.Exp([0.02, -0.01, 0.03])
    vi = v0 + [0.05, -0.02, 0.01]
    pi = p0 + [0.1, 0.05, -0.03]
    Rj = Rn @ lie.Exp([-0.01, 0.02, 0.01])
    vj = vn + [0.02, 0.03, -0.01]
    pj = pn + [-0.05, 0.02, 0.04]
    ba_q, bg_q = ba + 0.004, bg + 0.0005

    J_i, J_j, J_b = pim.jacobians(Ri, vi, pi, Rj, vj, pj, ba_q, bg_q)

    def res_i(d):
        return pim.residual(Ri @ lie.Exp(d[0:3]), vi + d[3:6], pi + d[6:9],
                            Rj, vj, pj, ba_q, bg_q)

    def res_j(d):
        return pim.residual(Ri, vi, pi,
                            Rj @ lie.Exp(d[0:3]), vj + d[3:6], pj + d[6:9], ba_q, bg_q)

    def res_b(d):
        return pim.residual(Ri, vi, pi, Rj, vj, pj, ba_q + d[0:3], bg_q + d[3:6])

    assert np.allclose(J_i, _numeric_jac(res_i, None, 9), atol=2e-5)
    assert np.allclose(J_j, _numeric_jac(res_j, None, 9), atol=2e-5)
    assert np.allclose(J_b, _numeric_jac(res_b, None, 6), atol=2e-4)


# --------------------------------------------------------------------------- #
# 4. covariance
# --------------------------------------------------------------------------- #

def test_covariance_is_psd_and_grows_monotonically():
    accs, gyros, dt, _, _ = synth_imu(n=100, dt=0.01)
    pim = PreintegratedImu(ImuNoiseModel(), np.zeros(3), np.zeros(3))
    prev_trace = -1.0
    for a, w in zip(accs, gyros):
        pim.integrate(a, w, dt)
        assert np.allclose(pim.cov, pim.cov.T, atol=1e-18)
        eig = np.linalg.eigvalsh(pim.cov)
        assert eig.min() > -1e-15
        tr = float(np.trace(pim.cov))
        assert tr >= prev_trace - 1e-15
        prev_trace = tr


def test_covariance_matches_monte_carlo():
    """The analytic 9x9 must match sampling the same noise through the model."""
    n, dt = 60, 0.01
    accs, gyros, _, (R0, v0, p0), _ = synth_imu(n=n, dt=dt)
    noise = ImuNoiseModel(gyro_noise_density=0.02, accel_noise_density=0.05)
    pim = make_pim(accs, gyros, dt, noise=noise)
    R_ref, v_ref, p_ref = pim.dR, pim.dv, pim.dp

    sa = noise.accel_noise_density / np.sqrt(dt)
    sg = noise.gyro_noise_density / np.sqrt(dt)

    rng = np.random.default_rng(11)
    samples = []
    for _ in range(4000):
        pk = PreintegratedImu(noise, np.zeros(3), np.zeros(3))
        for a, w in zip(accs, gyros):
            pk.integrate(a + rng.normal(scale=sa, size=3),
                         w + rng.normal(scale=sg, size=3), dt)
        samples.append(np.concatenate([lie.Log(R_ref.T @ pk.dR),
                                       pk.dv - v_ref, pk.dp - p_ref]))
    emp = np.cov(np.asarray(samples).T)

    # Compare standard deviations; 4000 samples gives ~1.5 % sampling error on
    # a std, so 20 % agreement is a strong test of the discretisation constant
    # (a missing/extra dt would show up as a factor of ~10 here).
    d_ana = np.sqrt(np.diag(pim.cov))
    d_emp = np.sqrt(np.diag(emp))
    rel = np.abs(d_ana - d_emp) / np.maximum(d_emp, 1e-12)
    assert rel.max() < 0.20, f"per-axis relative std error {rel}"


# --------------------------------------------------------------------------- #
# 5. degenerate information matrices
# --------------------------------------------------------------------------- #

def test_information_of_an_empty_increment_is_zero_not_infinite():
    """An edge with no samples must assert *no* constraint, not an infinite one."""
    pim = PreintegratedImu(ImuNoiseModel(), np.zeros(3), np.zeros(3))
    Info = pim.information()
    assert np.allclose(Info, 0.0)
    assert np.all(np.isfinite(Info))


def test_information_is_capped_for_a_rank_deficient_single_sample():
    """One sample leaves the covariance rank 6; the missing 3 must not blow up."""
    pim = PreintegratedImu(ImuNoiseModel(), np.zeros(3), np.zeros(3))
    pim.integrate(np.array([0.0, 0.0, 9.80665]), np.zeros(3), 0.01)
    assert np.linalg.matrix_rank(pim.cov, tol=1e-14) < 9
    Info = pim.information()
    assert np.all(np.isfinite(Info))
    assert np.linalg.eigvalsh(Info).max() <= PreintegratedImu.MAX_INFORMATION * 1.0001


def test_information_matches_plain_inverse_when_well_conditioned():
    """Away from the degeneracies, information() must be an honest inverse.

    Not exactly: the 1e-9 diagonal floor perturbs the smallest eigenvalue
    (~2e-4 here) by ~5e-6 relative. That is the intended price of guaranteed
    invertibility, so the tolerance is set just above it rather than at
    machine precision.
    """
    accs, gyros, dt, _, _ = synth_imu(n=100, dt=0.01)
    pim = make_pim(accs, gyros, dt)
    assert np.allclose(pim.information() @ pim.cov, np.eye(9), atol=1e-4)
