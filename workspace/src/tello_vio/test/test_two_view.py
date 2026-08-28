"""Two-view geometry tests on synthetic scenes with known ground truth.

Each test exercises a *failure mode that actually happens on a Tello indoors*:
a general room, a bare wall/floor (planar), a yaw-on-the-spot manoeuvre (pure
rotation), and matches polluted by KLT outliers.
"""
import numpy as np
import pytest

from tello_vio import lie
from tello_vio.two_view import (derotated_parallax, estimate_relative_pose,
                                normalise_points, solve_pnp, triangulate)

# Intrinsics of the shipped Tello calibration, downscaled to the 480x360 the
# front-end actually runs at.
K = np.array([[919.424717 * 0.5, 0.0, 459.655779 * 0.5],
              [0.0, 911.926190 * 0.5, 323.551997 * 0.5],
              [0.0, 0.0, 1.0]])
D = np.zeros(5)


def project(X_cam, K, noise=0.0, rng=None):
    z = X_cam[:, 2]
    uv = (X_cam[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    if noise > 0:
        uv = uv + rng.normal(scale=noise, size=uv.shape)
    return uv, z


def make_scene(n=250, planar=False, seed=0, depth=(2.0, 6.0)):
    rng = np.random.default_rng(seed)
    if planar:
        # A wall at z = 4 m: the degenerate case for the essential matrix.
        X = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n),
                             np.full(n, 4.0)])
    else:
        X = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n),
                             rng.uniform(*depth, n)])
    return X, rng


def two_views(X, R, t, rng, noise=0.3):
    """Project a point cloud into camera 1 (identity) and camera 2 (R, t)."""
    uv1, z1 = project(X, K, noise, rng)
    X2 = X @ R.T + t
    uv2, z2 = project(X2, K, noise, rng)
    keep = (z1 > 0.2) & (z2 > 0.2)
    return uv1[keep], uv2[keep], X[keep]


# --------------------------------------------------------------------------- #

def test_general_scene_recovers_rotation_and_translation_direction():
    X, rng = make_scene(300, planar=False, seed=1)
    R_true = lie.Exp([0.02, 0.10, -0.03])
    t_true = np.array([0.35, 0.05, 0.10])
    uv1, uv2, _ = two_views(X, R_true, t_true, rng)

    res = estimate_relative_pose(uv1, uv2, K, D, px_threshold=1.0)
    assert res.ok, res.reason
    assert res.model == "essential", res.planar_score
    assert res.translation_reliable
    assert np.degrees(np.linalg.norm(lie.Log(R_true.T @ res.R))) < 1.0

    # The scene was built with the OpenCV relation x2 = R x1 + t_true, so the
    # camera-1-frame baseline the estimator wants is -R^T t_true.
    b_true = -(R_true.T @ t_true)
    cos = res.t @ (b_true / np.linalg.norm(b_true))
    assert cos > 0.995, f"direction error {np.degrees(np.arccos(cos)):.2f} deg"
    assert np.isclose(np.linalg.norm(res.t), 1.0)


def test_translation_convention_is_camera1_frame_forward_baseline():
    """Pin the convention: moving the camera +x must report a +x baseline.

    Built as an explicit, geometry-free statement of intent, because this is
    the one sign in the pipeline that is both easy to flip and hard to notice.
    """
    X, rng = make_scene(300, planar=False, seed=11)
    # Camera 2 sits 0.4 m to the +x of camera 1, same orientation. World and
    # camera-1 frames coincide, so the expected baseline is exactly +x.
    R_true = np.eye(3)
    p2 = np.array([0.4, 0.0, 0.0])
    t_cv = -R_true @ p2                       # x2 = R x1 + t_cv
    uv1, uv2, _ = two_views(X, R_true, t_cv, rng)

    res = estimate_relative_pose(uv1, uv2, K, D, px_threshold=1.0)
    assert res.ok and res.translation_reliable, res.reason
    assert res.t @ np.array([1.0, 0.0, 0.0]) > 0.99, res.t


def test_planar_scene_selects_homography_and_still_recovers_pose():
    X, rng = make_scene(300, planar=True, seed=2)
    R_true = lie.Exp([0.01, 0.06, 0.0])
    t_true = np.array([0.4, 0.0, 0.05])
    uv1, uv2, _ = two_views(X, R_true, t_true, rng)

    res = estimate_relative_pose(uv1, uv2, K, D, px_threshold=1.0)
    assert res.ok, res.reason
    assert res.model == "homography", f"R_H={res.planar_score:.3f}"
    assert res.planar_score > 0.45
    assert np.degrees(np.linalg.norm(lie.Log(R_true.T @ res.R))) < 2.0


@pytest.mark.parametrize("axis,ang", [(2, 0.15), (1, 0.12), (0, 0.10)])
def test_pure_rotation_keeps_rotation_but_flags_translation(axis, ang):
    """Rotating on the spot: t is meaningless, R is not. Keep R, drop t."""
    X, rng = make_scene(300, planar=False, seed=3)
    v = np.zeros(3)
    v[axis] = ang
    R_true = lie.Exp(v)
    uv1, uv2, _ = two_views(X, R_true, np.zeros(3), rng)

    res = estimate_relative_pose(uv1, uv2, K, D, px_threshold=1.0,
                                 parallax_thresh_deg=1.0)
    assert res.ok, res.reason
    assert np.degrees(np.linalg.norm(lie.Log(R_true.T @ res.R))) < 2.0
    assert not res.translation_reliable, \
        f"parallax {np.degrees(res.parallax_rad):.3f} deg wrongly accepted"


def test_derotated_parallax_separates_rotation_from_translation():
    X, rng = make_scene(200, seed=4)
    # Rotate about y (pitch), not z: a yaw about the optical axis barely moves
    # bearings near the image centre, so it makes a weak demonstration.
    R_rot = lie.Exp([0.0, 0.20, 0.0])
    uv1, uv2, Xk = two_views(X, R_rot, np.zeros(3), rng, noise=0.0)
    p1n, p2n = normalise_points(uv1, K, D), normalise_points(uv2, K, D)
    # Rotation-only: de-rotated parallax collapses to ~0 ...
    assert np.median(derotated_parallax(p1n, p2n, R_rot)) < np.deg2rad(0.05)
    # ... while the naive (un-derotated) angle is large and would fool a gate.
    assert np.median(derotated_parallax(p1n, p2n, np.eye(3))) > np.deg2rad(5.0)

    uv1, uv2, _ = two_views(X, np.eye(3), np.array([0.5, 0, 0]), rng, noise=0.0)
    p1n, p2n = normalise_points(uv1, K, D), normalise_points(uv2, K, D)
    assert np.median(derotated_parallax(p1n, p2n, np.eye(3))) > np.deg2rad(2.0)


def test_outliers_are_rejected():
    X, rng = make_scene(300, seed=5)
    R_true = lie.Exp([0.0, 0.08, 0.0])
    t_true = np.array([0.3, 0.0, 0.0])
    uv1, uv2, _ = two_views(X, R_true, t_true, rng)
    # 25 % of matches are garbage, which is a realistic KLT failure rate over
    # low-texture indoor surfaces.
    n_bad = len(uv1) // 4
    idx = rng.choice(len(uv1), n_bad, replace=False)
    uv2 = uv2.copy()
    uv2[idx] = rng.uniform([0, 0], [480, 360], size=(n_bad, 2))

    res = estimate_relative_pose(uv1, uv2, K, D, px_threshold=1.0)
    assert res.ok, res.reason
    assert np.degrees(np.linalg.norm(lie.Log(R_true.T @ res.R))) < 2.0
    assert res.n_inliers > 0.6 * (len(uv1) - n_bad)
    assert np.count_nonzero(res.inliers[idx]) < 0.15 * n_bad


def test_triangulation_recovers_structure_up_to_scale():
    X, rng = make_scene(200, seed=6)
    R_true = lie.Exp([0.0, 0.05, 0.0])
    t_true = np.array([0.5, 0.0, 0.0])
    uv1, uv2, Xk = two_views(X, R_true, t_true, rng, noise=0.0)
    p1n, p2n = normalise_points(uv1, K, D), normalise_points(uv2, K, D)

    # Triangulate with the *unit* translation: structure comes back scaled by
    # 1/|t_true|, which is exactly the monocular scale ambiguity.
    Xt, good = triangulate(p1n, p2n, R_true, t_true / np.linalg.norm(t_true))
    assert np.count_nonzero(good) > 0.8 * len(Xk)
    s = np.linalg.norm(t_true)
    assert np.allclose(Xt[good] * s, Xk[good], atol=1e-6)


def test_triangulation_rejects_low_parallax_points():
    X, rng = make_scene(200, seed=7, depth=(400.0, 600.0))  # effectively at infinity
    R_true, t_true = np.eye(3), np.array([0.05, 0.0, 0.0])
    uv1, uv2, _ = two_views(X, R_true, t_true, rng, noise=0.3)
    p1n, p2n = normalise_points(uv1, K, D), normalise_points(uv2, K, D)
    _, good = triangulate(p1n, p2n, R_true, t_true / np.linalg.norm(t_true))
    assert np.count_nonzero(good) < 0.05 * len(p1n)


def test_pnp_recovers_camera_pose_against_a_known_map():
    X, rng = make_scene(150, seed=8)
    R_cw = lie.Exp([0.05, -0.10, 0.03])
    t_cw = np.array([0.2, -0.1, 0.05])
    Xc = X @ R_cw.T + t_cw
    uv, z = project(Xc, K, noise=0.4, rng=rng)
    keep = z > 0.3

    ok, R_est, t_est, idx = solve_pnp(X[keep], uv[keep], K, D)
    assert ok
    assert np.degrees(np.linalg.norm(lie.Log(R_cw.T @ R_est))) < 1.0
    assert np.linalg.norm(t_est - t_cw) < 0.03


def test_too_few_matches_fails_cleanly():
    res = estimate_relative_pose(np.zeros((4, 2)), np.zeros((4, 2)), K, D)
    assert not res.ok and "8 matches" in res.reason
