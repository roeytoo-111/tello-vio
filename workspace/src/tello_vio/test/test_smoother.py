"""Fixed-lag smoother tests.

The load-bearing test is :func:`test_marginalisation_preserves_the_solution`:
a sliding window that *deletes* old keyframes instead of marginalising them
still runs, still converges, and still looks plausible on a plot -- it is just
quietly wrong and over-confident. Only the batch-equivalence property catches
that, so it is checked directly.
"""
import numpy as np

from tello_vio import lie
from tello_vio.preintegration import GRAVITY_ENU, ImuNoiseModel, PreintegratedImu
from tello_vio.smoother import (S_P, AttitudeFactor,
                                BiasRandomWalkFactor, BodyVelocityFactor,
                                FixedLagSmoother, HeightFactor, ImuFactor,
                                KeyframeState, PositionFactor,
                                VisualRelativeFactor)

R_BC = lie.euler_zyx_to_rot(-np.pi / 2, 0.0, -np.pi / 2)   # FLU body -> optical
P_BC = np.array([0.03, 0.0, -0.01])


def truth_trajectory(n_kf=5, kf_dt=0.4, imu_dt=0.02, seed=0):
    """A smooth flight plus the exact IMU stream that generates it."""
    steps = int(round(kf_dt / imu_dt))

    R = np.eye(3)
    v = np.array([0.4, 0.1, 0.0])
    p = np.zeros(3)
    kfs = [KeyframeState(0.0, R.copy(), p.copy(), v.copy(), np.zeros(3), np.zeros(3))]
    bursts = []

    ba_true = np.array([0.05, -0.03, 0.02])
    bg_true = np.array([0.004, -0.002, 0.003])
    t = 0.0
    for k in range(n_kf - 1):
        accs, gyros = [], []
        for _ in range(steps):
            w = np.array([0.15 * np.sin(1.3 * t), 0.10 * np.cos(0.9 * t), 0.25])
            a_world = np.array([0.5 * np.sin(0.8 * t), 0.3 * np.cos(1.1 * t),
                                0.2 * np.sin(0.6 * t)])
            f_body = R.T @ (a_world - GRAVITY_ENU)
            accs.append(f_body + ba_true)      # what the sensor would report
            gyros.append(w + bg_true)
            a_w = R @ f_body + GRAVITY_ENU
            p = p + v * imu_dt + 0.5 * a_w * imu_dt ** 2
            v = v + a_w * imu_dt
            R = R @ lie.Exp(w * imu_dt)
            t += imu_dt
        bursts.append((np.array(accs), np.array(gyros), imu_dt))
        kfs.append(KeyframeState(t, R.copy(), p.copy(), v.copy(),
                                 ba_true.copy(), bg_true.copy()))
    for kf in kfs:
        kf.ba = ba_true.copy()
        kf.bg = bg_true.copy()
    return kfs, bursts, ba_true, bg_true


def build_graph(kfs, bursts, ba0, bg0, noise=None, with_visual=True,
                window=10, robust=False):
    """Assemble a fully constrained window around the given truth."""
    noise = noise or ImuNoiseModel()
    sm = FixedLagSmoother(window=window, robust=robust)
    for k, kf in enumerate(kfs):
        sm.add_keyframe(k, kf.copy())

    for k, (accs, gyros, dt) in enumerate(bursts):
        pim = PreintegratedImu(noise, ba0, bg0)
        for a, w in zip(accs, gyros):
            pim.integrate(a, w, dt)
        sm.add_factor(ImuFactor(k, k + 1, pim))
        sm.add_factor(BiasRandomWalkFactor(k, k + 1, pim.bias_random_walk_cov()
                                           + np.eye(6) * 1e-8))

    for k, kf in enumerate(kfs):
        sm.add_factor(AttitudeFactor(k, lie.rot_to_quat(kf.R),
                                     np.deg2rad(2.0), np.deg2rad(15.0)))
        sm.add_factor(BodyVelocityFactor(k, kf.R.T @ kf.v, 0.15))
        sm.add_factor(HeightFactor(k, kf.p[2], 0.30))
    # Anchor the gauge: without an absolute position the graph is invariant to
    # a global translation and the Hessian is rank deficient by three.
    sm.add_factor(PositionFactor(0, kfs[0].p, 0.02))

    if with_visual:
        for k in range(len(kfs) - 1):
            a, b = kfs[k], kfs[k + 1]
            R_c1 = a.R @ R_BC
            R_c2 = b.R @ R_BC
            p_c1 = a.p + a.R @ P_BC
            p_c2 = b.p + b.R @ P_BC
            R_rel = R_c1.T @ R_c2
            t_rel = R_c1.T @ (p_c2 - p_c1)
            n = np.linalg.norm(t_rel)
            sm.add_factor(VisualRelativeFactor(
                k, k + 1, R_rel, t_rel / n if n > 1e-9 else np.zeros(3),
                R_BC, P_BC, rot_std=np.deg2rad(1.5), dir_std=np.deg2rad(8.0),
                use_translation=n > 1e-9))
    return sm


def perturb(sm, rng, scale=1.0):
    for k in sm.keys:
        s = sm.states[k]
        s.R = s.R @ lie.Exp(rng.normal(scale=0.05 * scale, size=3))
        s.p = s.p + rng.normal(scale=0.10 * scale, size=3)
        s.v = s.v + rng.normal(scale=0.10 * scale, size=3)
        s.ba = s.ba + rng.normal(scale=0.02 * scale, size=3)
        s.bg = s.bg + rng.normal(scale=0.002 * scale, size=3)


def max_error(sm, kfs, keys=None):
    keys = keys if keys is not None else sm.keys
    ep = max(np.linalg.norm(sm.states[k].p - kfs[k].p) for k in keys)
    er = max(np.linalg.norm(lie.Log(kfs[k].R.T @ sm.states[k].R)) for k in keys)
    ev = max(np.linalg.norm(sm.states[k].v - kfs[k].v) for k in keys)
    return ep, er, ev


# --------------------------------------------------------------------------- #

def test_residuals_vanish_at_the_true_states():
    kfs, bursts, ba, bg = truth_trajectory()
    sm = build_graph(kfs, bursts, ba, bg)
    _, _, cost = sm._accumulate()
    assert cost < 1e-12, f"cost at truth is {cost}"


def test_converges_to_truth_from_a_perturbed_start():
    kfs, bursts, ba, bg = truth_trajectory(n_kf=5)
    sm = build_graph(kfs, bursts, ba, bg)
    perturb(sm, np.random.default_rng(1))
    ep0, er0, ev0 = max_error(sm, kfs)
    rep = sm.solve(iterations=25)
    ep, er, ev = max_error(sm, kfs)
    assert rep["status"] == "ok", rep
    assert rep["cost"] < 1e-6, rep
    assert ep < 0.01 * max(ep0, 1e-9) or ep < 1e-4, (ep0, ep)
    # 5e-4 rad is 0.03 deg -- the residual is LM's convergence tolerance, not
    # an estimation error: the measurements are noiseless and generated at
    # truth, so the optimum *is* truth and the cost above confirms we reached it.
    assert er < 5e-4 and ev < 2e-3, (er, ev)


def test_recovers_imu_biases_it_was_not_given():
    """Biases start at zero and must be pulled to the true values by the graph."""
    kfs, bursts, ba, bg = truth_trajectory(n_kf=6)
    sm = build_graph(kfs, bursts, np.zeros(3), np.zeros(3))
    for k in sm.keys:
        sm.states[k].ba = np.zeros(3)
        sm.states[k].bg = np.zeros(3)
    sm.solve(iterations=30)
    est_ba = sm.states[sm.keys[-1]].ba
    est_bg = sm.states[sm.keys[-1]].bg
    assert np.linalg.norm(est_bg - bg) < 5e-4, (est_bg, bg)
    assert np.linalg.norm(est_ba - ba) < 0.02, (est_ba, ba)


def test_marginalisation_preserves_the_solution():
    """Marginalise-then-solve must equal solve-the-full-batch. The core property."""
    kfs, bursts, ba, bg = truth_trajectory(n_kf=5)
    sm = build_graph(kfs, bursts, ba, bg)
    rng = np.random.default_rng(2)
    perturb(sm, rng)
    sm.solve(iterations=30)

    keep = sm.keys[1:]
    optimum = {k: sm.states[k].copy() for k in keep}

    # Eliminate the oldest keyframe at the optimum, keeping its information.
    sm.marginalise_oldest()
    assert sm.keys == keep
    assert sm.prior is not None

    # Kick the survivors and re-solve; the prior must pull them back to the
    # exact same optimum the full batch found.
    perturb(sm, np.random.default_rng(3), scale=0.5)
    sm.solve(iterations=30)

    for k in keep:
        assert np.linalg.norm(sm.states[k].p - optimum[k].p) < 2e-3, k
        assert np.linalg.norm(lie.Log(optimum[k].R.T @ sm.states[k].R)) < 2e-3, k
        assert np.linalg.norm(sm.states[k].v - optimum[k].v) < 5e-3, k


def test_deleting_instead_of_marginalising_is_measurably_worse():
    """Quantifies what marginalisation buys, so the machinery earns its keep."""
    kfs, bursts, ba, bg = truth_trajectory(n_kf=5)

    sm = build_graph(kfs, bursts, ba, bg)
    perturb(sm, np.random.default_rng(2))
    sm.solve(iterations=30)
    keep = sm.keys[1:]
    optimum = {k: sm.states[k].copy() for k in keep}

    # Variant A: proper marginalisation.
    sm_a = build_graph(kfs, bursts, ba, bg)
    perturb(sm_a, np.random.default_rng(2))
    sm_a.solve(iterations=30)
    sm_a.marginalise_oldest()
    perturb(sm_a, np.random.default_rng(4), scale=0.5)
    sm_a.solve(iterations=30)

    # Variant B: naive deletion of the keyframe and its factors.
    sm_b = build_graph(kfs, bursts, ba, bg)
    perturb(sm_b, np.random.default_rng(2))
    sm_b.solve(iterations=30)
    drop = sm_b.keys[0]
    sm_b.factors = [f for f in sm_b.factors if drop not in f.keys]
    sm_b.states.pop(drop)
    sm_b._order.pop(0)
    perturb(sm_b, np.random.default_rng(4), scale=0.5)
    sm_b.solve(iterations=30)

    err_a = max(np.linalg.norm(sm_a.states[k].p - optimum[k].p) for k in keep)
    err_b = max(np.linalg.norm(sm_b.states[k].p - optimum[k].p) for k in keep)
    assert err_a < 2e-3, err_a
    assert err_b > 5 * max(err_a, 1e-4), f"marginalised {err_a:.4f} vs deleted {err_b:.4f}"


def test_window_stays_bounded_and_prior_chains():
    kfs, bursts, ba, bg = truth_trajectory(n_kf=9)
    sm = build_graph(kfs, bursts, ba, bg, window=4)
    sm.solve(iterations=15)
    sm.enforce_window()
    assert len(sm.keys) == 4
    sm.enforce_window()
    assert len(sm.keys) == 4
    rep = sm.solve(iterations=10)
    assert rep["status"] == "ok", rep
    assert np.all(np.isfinite(sm.states[sm.keys[-1]].p))


def test_robust_kernel_suppresses_a_visual_outlier():
    kfs, bursts, ba, bg = truth_trajectory(n_kf=5)

    def run(robust):
        sm = build_graph(kfs, bursts, ba, bg, robust=robust)
        # Corrupt one visual factor with a 40 deg rotation blunder -- the kind
        # of thing a mis-associated match set produces after RANSAC.
        for f in sm.factors:
            if isinstance(f, VisualRelativeFactor) and f.i == 1:
                f.R_c1c2 = f.R_c1c2 @ lie.Exp([0.0, 0.0, np.deg2rad(40.0)])
                break
        perturb(sm, np.random.default_rng(5), scale=0.2)
        sm.solve(iterations=30)
        return max_error(sm, kfs)[1]

    err_plain = run(False)
    err_robust = run(True)
    assert err_robust < 0.5 * err_plain, (err_robust, err_plain)


def test_translation_gauge_is_unobservable_without_a_position_anchor():
    """Observability, stated as a rank test on the information matrix.

    A visual-inertial graph measures *relative* geometry plus gravity. Gravity
    pins roll and pitch; nothing pins absolute position, and nothing pins yaw
    about gravity. So the Hessian of a window with no absolute-position
    measurement is rank deficient by exactly 3 (global translation), and by 4 if
    the drone's yaw-bearing measurements are also removed.

    Asserting this explicitly documents *why* PositionFactor exists, and it is
    the check that would catch a future factor accidentally constraining
    something it cannot observe.
    """
    kfs, bursts, ba, bg = truth_trajectory(n_kf=4)

    sm_free = build_graph(kfs, bursts, ba, bg, with_visual=True)
    sm_free.factors = [f for f in sm_free.factors if not isinstance(f, PositionFactor)]
    # HeightFactor observes z, so removing PositionFactor alone leaves 2 free
    # directions (x, y). Remove height too for a clean 3-dimensional null space.
    sm_free.factors = [f for f in sm_free.factors if not isinstance(f, HeightFactor)]
    H_free, _, _ = sm_free._accumulate()
    w_free = np.linalg.eigvalsh(H_free)
    n_null = int(np.sum(w_free < 1e-8 * max(1.0, w_free.max())))
    assert n_null == 3, f"expected a 3-D translation null space, got {n_null}"

    # The null space must be exactly "translate every keyframe identically".
    _, V = np.linalg.eigh(H_free)
    null = V[:, :3]
    for axis in range(3):
        d = np.zeros(len(sm_free.keys) * 15)
        for n in range(len(sm_free.keys)):
            d[n * 15 + S_P.start + axis] = 1.0
        d /= np.linalg.norm(d)
        # d must lie inside the numerical null space.
        assert np.linalg.norm(null @ (null.T @ d) - d) < 1e-6, axis

    # Re-adding the anchor makes the system full rank.
    sm_anchored = build_graph(kfs, bursts, ba, bg, with_visual=True)
    H, _, _ = sm_anchored._accumulate()
    w = np.linalg.eigvalsh(H)
    assert w.min() > 1e-8 * w.max(), f"still rank deficient: {w[:5]}"
    L = np.linalg.cholesky(H)   # must not raise
    assert np.all(np.isfinite(L))
