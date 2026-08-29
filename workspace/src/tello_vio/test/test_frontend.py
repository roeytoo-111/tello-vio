"""Front-end tests on a rendered synthetic fly-through.

Renders a real 3-D point cloud as textured blobs (not a warped plane), so the
essential-matrix path is genuinely exercised, then flies a camera through it and
checks that the emitted measurements match the ground-truth motion.
"""
import time

import cv2
import numpy as np
import pytest

from tello_vio import lie
from tello_vio.frontend import FeatureTracker, FrontendConfig, VoFrontend

W, H = 960, 720
K = np.array([[919.424717, 0.0, 459.655779],
              [0.0, 911.926190, 323.551997],
              [0.0, 0.0, 1.0]])
D = np.zeros(5)


def make_cloud(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(-4.0, 4.0, n),
        rng.uniform(-3.0, 3.0, n),
        rng.uniform(2.5, 9.0, n),
    ]), rng


def render(X_world, R_cw, t_cw, rng, bg_noise=3.0):
    """Project the cloud into a camera at (R_cw, t_cw) and draw trackable blobs."""
    img = np.full((H, W), 40, dtype=np.uint8)
    Xc = X_world @ R_cw.T + t_cw
    z = Xc[:, 2]
    ok = z > 0.5
    uv = (Xc[ok, :2] / z[ok, None]) @ K[:2, :2].T + K[:2, 2]
    for (u, v) in uv:
        if -10 < u < W + 10 and -10 < v < H + 10:
            cv2.circle(img, (int(round(u)), int(round(v))), 4, 220, -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    if bg_noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, bg_noise, img.shape), 0, 255)
        img = img.astype(np.uint8)
    return img


# --------------------------------------------------------------------------- #

def test_tracker_survives_small_motion_and_drops_bad_tracks():
    X, rng = make_cloud(400, seed=1)
    img0 = render(X, np.eye(3), np.zeros(3), rng)
    img1 = render(X, np.eye(3), np.array([0.06, 0.0, 0.0]), rng)

    tr = FeatureTracker(FrontendConfig())
    tr.prev_gray = img0
    n = tr.detect(img0)
    assert n > 80, f"only detected {n} features"
    before = len(tr.pts)
    pts, ids, kept = tr.track(img1)
    assert len(pts) > 0.75 * before, f"kept {len(pts)}/{before}"
    assert len(pts) == len(ids)

    # A totally unrelated image must fail the forward-backward test wholesale.
    noise = (rng.random((H, W)) * 255).astype(np.uint8)
    tr2 = FeatureTracker(FrontendConfig())
    tr2.prev_gray = img0
    tr2.detect(img0)
    pts2, _, _ = tr2.track(noise)
    assert len(pts2) < 0.25 * before, f"{len(pts2)} bogus tracks survived"


def test_detection_is_spread_across_the_image_by_the_grid():
    X, rng = make_cloud(900, seed=2)
    img = render(X, np.eye(3), np.zeros(3), rng)
    cfg = FrontendConfig()
    tr = FeatureTracker(cfg)
    tr.detect(cv2.resize(img, (cfg.work_width, int(H * cfg.work_width / W))))
    pts = tr.pts
    assert len(pts) > 80
    h = int(H * cfg.work_width / W)
    occupied = set()
    for x, y in pts:
        occupied.add((int(x // (cfg.work_width / cfg.grid_cols)),
                      int(y // (h / cfg.grid_rows))))
    # A non-bucketed detector routinely fills 3-5 cells; the grid should fill
    # most of them.
    assert len(occupied) >= 0.6 * cfg.grid_cols * cfg.grid_rows, len(occupied)


def test_intrinsics_are_rescaled_with_the_image():
    fe = VoFrontend(K, D, FrontendConfig(work_width=480))
    gray = np.zeros((H, W), np.uint8)
    fe._prepare(gray)
    assert np.isclose(fe.K[0, 0], K[0, 0] * 0.5)
    assert np.isclose(fe.K[1, 1], K[1, 1] * 0.5)
    assert np.isclose(fe.K[0, 2], (K[0, 2] + 0.5) * 0.5 - 0.5)
    # Distortion lives in normalised coordinates and must be left alone.
    assert np.allclose(fe.D, D)


def test_flythrough_measurements_are_accurate_and_honestly_calibrated():
    """Fly forward+right while yawing, over several scenes.

    Two claims are checked, and the second is the important one:

    1. the measurements are *accurate* -- sub-degree rotation on the essential
       path, and a translation bearing that points the right way;
    2. the measurements are *honest* -- the 1-sigma each one reports via
       ``noise_std`` actually brackets its own error. An over-confident visual
       measurement is far more damaging to a Kalman filter than an inaccurate
       one, because the filter has no way to discount it.
    """
    rot_z, dir_z = [], []
    rot_err_essential = []
    n_meas = 0

    for seed in range(6):
        X, rng = make_cloud(700, seed=seed)
        cfg = FrontendConfig(kf_min_parallax_px=8.0, kf_min_interval_s=0.0)
        fe = VoFrontend(K, D, cfg)

        poses = []
        for k in range(45):
            t = k * 0.05
            yaw = 0.25 * t * (1 if seed % 2 else -1)
            R_wc = lie.euler_zyx_to_rot(yaw, 0.10 * np.sin(0.7 * t), 0.05 * np.cos(t))
            p_wc = np.array([0.55 * t, 0.10 * t, 0.15 * np.sin(0.8 * t)])
            R_cw = R_wc.T
            poses.append((R_cw, -R_cw @ p_wc, p_wc))

        prev = 0
        for k, (R_cw, t_cw, p_wc) in enumerate(poses):
            img = render(X, R_cw, t_cw, rng)
            # The rotation prior the real system gets free from the AHRS.
            R_prior = R_cw @ poses[prev][0].T
            m = fe.process(img, stamp=k * 0.05, R_pred_c1c2=R_prior)
            if m is None:
                continue
            Rk_cw, _, pk_wc = poses[prev]
            R_true = R_cw @ Rk_cw.T
            t_true = Rk_cw @ (p_wc - pk_wc)
            prev = k
            n_meas += 1

            s_rot, s_dir = m.noise_std()
            e_rot = float(np.linalg.norm(lie.Log(R_true.T @ m.R_c1c2)))
            rot_z.append(e_rot / s_rot)
            if m.model == "essential":
                rot_err_essential.append(np.degrees(e_rot))

            if m.translation_reliable and np.linalg.norm(t_true) > 1e-6:
                u = t_true / np.linalg.norm(t_true)
                e_dir = float(np.arccos(np.clip(m.t_dir_c1 @ u, -1, 1)))
                dir_z.append(e_dir / s_dir)

    assert n_meas >= 30, f"only {n_meas} measurements produced"
    assert len(dir_z) >= 15, f"only {len(dir_z)} bearing measurements"

    # Accuracy on the well-conditioned path.
    assert np.median(rot_err_essential) < 1.0, np.median(rot_err_essential)

    # Calibration: errors must sit inside the reported sigma, and the reported
    # sigma must not be absurdly loose either (that would make it useless).
    for name, z in (("rotation", np.array(rot_z)), ("bearing", np.array(dir_z))):
        assert np.percentile(z, 90) < 3.0, f"{name} over-confident: p90 z={np.percentile(z,90):.2f}"
        assert np.mean(z > 3.0) < 0.10, f"{name} {np.mean(z>3)*100:.0f}% beyond 3 sigma"
        assert np.median(z) > 0.05, f"{name} absurdly under-confident: median z={np.median(z):.3f}"


def test_rotation_compensated_trigger_beats_raw_displacement():
    """The prior should raise parallax at the keyframe, not just move it around."""
    def run(use_prior):
        X, rng = make_cloud(700, seed=3)
        fe = VoFrontend(K, D, FrontendConfig(kf_min_parallax_px=8.0, kf_min_interval_s=0.0))
        poses = []
        for k in range(40):
            t = k * 0.05
            R_wc = lie.euler_zyx_to_rot(0.25 * t, 0.0, 0.0)
            p_wc = np.array([0.55 * t, 0.10 * t, 0.0])
            R_cw = R_wc.T
            poses.append((R_cw, -R_cw @ p_wc))
        prev, pars = 0, []
        for k, (R_cw, t_cw) in enumerate(poses):
            img = render(X, R_cw, t_cw, rng)
            prior = (R_cw @ poses[prev][0].T) if use_prior else None
            m = fe.process(img, stamp=k * 0.05, R_pred_c1c2=prior)
            if m is not None:
                pars.append(np.degrees(m.parallax_rad))
                prev = k
        return np.median(pars) if pars else 0.0

    with_prior, without = run(True), run(False)
    assert with_prior > 1.6 * without, f"prior {with_prior:.2f} deg vs raw {without:.2f} deg"


def test_hover_produces_rotation_only_measurements():
    """Station-keeping: no baseline, so translation must be flagged unreliable."""
    X, rng = make_cloud(700, seed=4)
    fe = VoFrontend(K, D, FrontendConfig(kf_max_interval_s=0.2, kf_min_interval_s=0.0))
    flagged = 0
    total = 0
    for k in range(30):
        R_cw = lie.euler_zyx_to_rot(0.03 * np.sin(k * 0.4), 0.0, 0.0)
        img = render(X, R_cw, np.zeros(3), rng)
        m = fe.process(img, stamp=k * 0.05)
        if m is not None:
            total += 1
            flagged += int(not m.translation_reliable)
    assert total > 0
    assert flagged >= 0.8 * total, f"{flagged}/{total} correctly flagged"


@pytest.mark.slow
def test_frontend_cost_fits_a_30hz_budget():
    """Sanity on CPU cost. Not a hard perf gate -- CI machines vary wildly."""
    X, rng = make_cloud(700, seed=5)
    fe = VoFrontend(K, D, FrontendConfig())
    imgs = [render(X, lie.euler_zyx_to_rot(0.02 * k, 0, 0),
                   np.array([0.03 * k, 0, 0]), rng) for k in range(25)]
    for im in imgs[:5]:
        fe.process(im, 0.0)
    t0 = time.perf_counter()
    for k, im in enumerate(imgs):
        fe.process(im, 0.05 * k)
    ms = (time.perf_counter() - t0) / len(imgs) * 1e3
    print(f"\nfront-end: {ms:.2f} ms/frame at {fe.cfg.work_width}px wide")
    assert ms < 33.0, f"{ms:.1f} ms/frame exceeds the 30 Hz budget"


# --------------------------------------------------------------------------- #
# Regression: mid-interval re-detection
# --------------------------------------------------------------------------- #

def test_redetection_midway_does_not_break_correspondence():
    """Crashed in flight: ValueError broadcasting (180,2) against (77,2).

    `detect()` tops the track set back up as soon as tracking thins out. The
    old code aligned the keyframe observations to the live ones by array
    position, which stops being true the instant new features are added -- the
    rotation-compensated flow then differenced a 180-point live set against a
    77-point prediction and took the whole estimator down with it.

    Correspondence is now resolved by feature id, so the two sets can differ in
    length freely. Driven hard here: features are forced to die every frame so
    re-detection fires constantly.
    """
    X, rng = make_cloud(700, seed=11)
    cfg = FrontendConfig(min_features=150, max_features=180,
                         kf_min_interval_s=0.0, kf_min_parallax_px=6.0)
    fe = VoFrontend(K, D, cfg)

    prev_R = np.eye(3)
    for k in range(40):
        R_cw = lie.euler_zyx_to_rot(0.05 * k, 0.02 * np.sin(k), 0.0)
        p_wc = np.array([0.05 * k, 0.02 * k, 0.0])
        img = render(X, R_cw, -R_cw @ p_wc, rng)

        # A rotation prior is what exercises the failing path.
        prior = R_cw @ prev_R.T
        prev_R = R_cw

        # Kill half the tracks to guarantee detect() fires mid-interval.
        if fe.tracker.pts.shape[0] > 20 and k % 2 == 0:
            keep = fe.tracker.pts.shape[0] // 2
            fe.tracker.pts = fe.tracker.pts[:keep]
            fe.tracker.ids = fe.tracker.ids[:keep]

        fe.process(img, stamp=k * 0.05, R_pred_c1c2=prior)  # must not raise

        kf_m, cur_m = fe.matched_pairs()
        if kf_m is not None:
            assert kf_m.shape == cur_m.shape, "matched pairs must align"


def test_matched_pairs_align_by_id_not_position():
    """The property the fix rests on."""
    fe = VoFrontend(K, D, FrontendConfig())
    fe.kf_pts = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]], np.float32)
    fe.kf_ids = np.array([10, 11, 12, 13], dtype=np.int64)

    # Tracker lost id 11, kept the rest, and gained two brand-new features.
    fe.tracker.pts = np.array([[0., 9.], [2., 9.], [3., 9.], [7., 7.], [8., 8.]], np.float32)
    fe.tracker.ids = np.array([10, 12, 13, 20, 21], dtype=np.int64)

    kf_m, cur_m = fe.matched_pairs()
    assert len(kf_m) == len(cur_m) == 3
    assert np.allclose(kf_m[:, 0], [0., 2., 3.])   # ids 10, 12, 13
    assert np.allclose(cur_m[:, 1], [9., 9., 9.])  # their current positions


def test_matched_pairs_handles_total_loss():
    fe = VoFrontend(K, D, FrontendConfig())
    fe.kf_pts = np.array([[0., 0.]], np.float32)
    fe.kf_ids = np.array([1], dtype=np.int64)
    fe.tracker.pts = np.zeros((0, 2), np.float32)
    fe.tracker.ids = np.zeros((0,), dtype=np.int64)
    assert fe.matched_pairs() == (None, None)
