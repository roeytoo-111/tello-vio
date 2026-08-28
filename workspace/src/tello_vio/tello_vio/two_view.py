"""Two-view geometry: model selection, relative pose, triangulation, parallax.

This is the geometric core of the visual front-end. Four things here are easy
to get subtly wrong and expensive to debug in flight, so each is handled
explicitly:

**1. Work in normalised coordinates, not pixels.** ``cv2.findEssentialMat`` can
take a focal length and principal point, but that path assumes square pixels
and no distortion. The Tello's stream has neither guaranteed. We push points
through :func:`cv2.undistortPoints` (which applies ``K^-1`` *and* removes
distortion) and then estimate ``E`` with ``K = I``. The RANSAC threshold must be
converted to the same units -- forgetting that leaves the threshold ~900x too
large and RANSAC accepts everything.

**2. Planar scenes break the essential matrix.** Indoor flying means floors,
walls and ceilings: exactly the degenerate configuration where ``E`` is not
uniquely determined and ``recoverPose`` returns confident nonsense. We score a
homography against the fundamental matrix (Torr's approach, as used by
ORB-SLAM) and switch models when the scene is plane-dominated.

**3. Low parallax breaks translation.** With little baseline the translation
direction is unobservable -- ``t`` is essentially a random unit vector. We
measure parallax *after* de-rotating the bearings, which is the only measure
that separates "the drone rotated" from "the drone translated", and report it
so the filter can fall back to a rotation-only update.

**4. Translation from a monocular pair has no scale.** ``t`` is always returned
as a unit vector and is documented as a *direction*. Nothing downstream is
allowed to treat it as metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TwoViewResult:
    """Outcome of a two-view relative-pose estimation.

    Conventions -- read these before using ``R`` or ``t``:

    * ``R`` maps **camera-1 coordinates into camera-2**: ``x2 = R x1 + t_cv``.
      This is OpenCV's convention and is what ``recoverPose`` returns.
    * ``t`` is **NOT** OpenCV's ``t``. It is the baseline **from camera 1 to
      camera 2, expressed in camera 1**, as a unit vector -- which is the
      quantity a pose estimator actually wants. OpenCV's ``t_cv`` is the
      opposite: the vector from camera 2 to camera 1, expressed in camera 2.
      The two differ by ``t = -R^T t_cv``, i.e. a negation *and* a frame
      change, so mistaking one for the other yields a direction roughly 180
      degrees wrong -- close enough to plausible that it survives a casual
      plot and quietly drives the filter backwards.
    * ``t`` is a unit **direction**, never a displacement: the magnitude of
      monocular translation is unobservable.

    ``parallax_rad`` is the median de-rotated bearing angle and is the number
    to gate on before believing ``t`` at all.
    """

    ok: bool
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inliers: np.ndarray | None = None
    n_inliers: int = 0
    parallax_rad: float = 0.0
    model: str = "none"          # 'essential' | 'homography' | 'none'
    planar_score: float = 0.0    # R_H in [0,1]; high means plane-dominated
    translation_reliable: bool = False
    reason: str = ""


def normalise_points(pts: np.ndarray, K: np.ndarray, D: np.ndarray | None) -> np.ndarray:
    """Pixels -> ideal normalised image coordinates (distortion removed).

    Returns an ``(N, 2)`` array on the ``z = 1`` plane, i.e. ``K^-1`` applied
    and lens distortion undone. Everything downstream assumes this.
    """
    pts = np.ascontiguousarray(np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2))
    D = np.zeros(5) if D is None else np.asarray(D, dtype=np.float64).reshape(1, -1)
    out = cv2.undistortPoints(pts, np.asarray(K, dtype=np.float64), D)
    return out.reshape(-1, 2)


def bearings(pts_norm: np.ndarray) -> np.ndarray:
    """Normalised image points -> unit bearing vectors in the camera frame."""
    p = np.asarray(pts_norm, dtype=np.float64).reshape(-1, 2)
    f = np.hstack([p, np.ones((p.shape[0], 1))])
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def derotated_parallax(p1n: np.ndarray, p2n: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Per-correspondence parallax angle (rad) with the rotation taken out.

    The raw angle between two bearings conflates rotation with translation: a
    drone spinning on the spot produces large bearing changes and zero
    parallax. Rotating view-2's bearings back into view-1's frame removes that,
    leaving only the angle the *baseline* induces. This is the quantity that
    decides whether triangulation and the translation direction mean anything.
    """
    f1 = bearings(p1n)
    # R maps camera-1 coordinates into camera-2 (OpenCV's convention), so the
    # inverse rotation brings view-2's rays back into view-1's orientation.
    # With row-vector arrays, `v @ R` is `R^T v` -- writing `v @ R.T` here
    # applies R instead of R^T and *doubles* the apparent parallax of a pure
    # rotation, defeating the very gate this function exists to feed.
    f2 = bearings(p2n) @ R
    c = np.clip(np.einsum("ij,ij->i", f1, f2), -1.0, 1.0)
    return np.arccos(c)


def _score_homography(H, p1, p2, sigma):
    """Symmetric transfer error score for a homography (Torr / ORB-SLAM style)."""
    # `e` below is already divided by sigma^2, so the threshold is the raw
    # chi-square quantile -- multiplying it by sigma^2 again (a tempting
    # symmetry) makes it ~1e5 times too small and every point an outlier.
    th = 5.991                          # chi2(2, 0.95)
    Hinv = np.linalg.inv(H)
    score = 0.0
    inl = np.ones(len(p1), dtype=bool)
    for (A, B, M) in ((p1, p2, H), (p2, p1, Hinv)):
        Ah = np.hstack([A, np.ones((len(A), 1))])
        proj = Ah @ M.T
        w = proj[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        proj = proj[:, :2] / w
        e = np.sum((proj - B) ** 2, axis=1) / (sigma * sigma)
        ok = e < th
        inl &= ok
        score += np.sum(np.where(ok, th - e, 0.0))
    return float(score), inl


def _score_fundamental(F, p1, p2, sigma):
    """Symmetric epipolar-distance score for a fundamental matrix."""
    th = 3.841                          # chi2(1, 0.95): epipolar distance is 1-D
    th_score = 5.991                    # scored on the homography's scale so the
                                        # two models are directly comparable
    p1h = np.hstack([p1, np.ones((len(p1), 1))])
    p2h = np.hstack([p2, np.ones((len(p2), 1))])
    score = 0.0
    inl = np.ones(len(p1), dtype=bool)
    for (A, B, M) in ((p1h, p2h, F), (p2h, p1h, F.T)):
        l = A @ M.T                     # epipolar lines in the other image
        num = np.einsum("ij,ij->i", l, B) ** 2
        den = l[:, 0] ** 2 + l[:, 1] ** 2
        den = np.where(den < 1e-20, 1e-20, den)
        e = (num / den) / (sigma * sigma)
        ok = e < th
        inl &= ok
        score += np.sum(np.where(ok, th_score - e, 0.0))
    return float(score), inl


def estimate_relative_pose(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None = None,
    px_threshold: float = 1.0,
    min_inliers: int = 15,
    parallax_thresh_deg: float = 1.0,
    planar_ratio: float = 0.45,
    prob: float = 0.999,
) -> TwoViewResult:
    """Recover ``(R, t_dir)`` taking camera 1 to camera 2 from 2-D matches.

    Parameters
    ----------
    pts1, pts2:
        Matched pixel coordinates, ``(N, 2)`` each, same ordering.
    px_threshold:
        RANSAC inlier threshold **in pixels**; converted internally to
        normalised units using the focal length from ``K``.
    parallax_thresh_deg:
        Below this median de-rotated parallax the translation direction is
        marked unreliable (``translation_reliable = False``) and the caller
        should fall back to a rotation-only update.

    Returns a :class:`TwoViewResult` -- see its docstring for the exact
    meaning of ``R`` and ``t``. In short: ``R`` follows OpenCV
    (``x2 = R x1 + t_cv``) but ``t`` is converted to the camera-1-frame,
    camera-1-to-camera-2 baseline direction that pose estimators expect.
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    if len(pts1) < 8 or len(pts1) != len(pts2):
        return TwoViewResult(ok=False, reason=f"need >=8 matches, got {len(pts1)}")

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    p1n = normalise_points(pts1, K, D)
    p2n = normalise_points(pts2, K, D)

    # A pixel threshold means different things in normalised coordinates
    # depending on focal length; convert once, here.
    f_mean = 0.5 * (K[0, 0] + K[1, 1])
    thr_n = px_threshold / f_mean

    # ---- model selection: plane or general 3-D structure? -----------------
    H, _ = cv2.findHomography(p1n, p2n, cv2.RANSAC, thr_n, maxIters=2000, confidence=prob)
    F, _ = cv2.findFundamentalMat(p1n, p2n, cv2.FM_RANSAC, thr_n, prob, 2000)
    if F is not None and F.shape[0] > 3:      # OpenCV can return stacked solutions
        F = F[:3, :]

    s_h, s_f = 0.0, 0.0
    if H is not None:
        s_h, _ = _score_homography(H, p1n, p2n, thr_n)
    if F is not None:
        s_f, _ = _score_fundamental(F, p1n, p2n, thr_n)
    r_h = s_h / (s_h + s_f) if (s_h + s_f) > 0 else 0.0

    use_homography = H is not None and r_h > planar_ratio

    # ---- recover (R, t) ----------------------------------------------------
    if use_homography:
        n_sol, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, np.eye(3))
        best = _select_homography_solution(Rs, Ts, Ns, p1n, p2n)
        if best is None:
            return TwoViewResult(ok=False, model="homography", planar_score=r_h,
                                 reason="no admissible homography decomposition")
        R, t = best
        # The inlier set is what agrees with the *homography*, not what
        # survives triangulation: under pure rotation nothing triangulates
        # (zero baseline) yet every match is a perfectly good inlier, and
        # conflating the two throws the whole measurement away exactly when
        # the rotation estimate is at its most reliable.
        _, inl = _score_homography(H, p1n, p2n, thr_n)
        model = "homography"
    else:
        E, mask_e = cv2.findEssentialMat(
            p1n, p2n, np.eye(3), method=cv2.RANSAC, prob=prob, threshold=thr_n
        )
        if E is None or E.shape[0] < 3:
            return TwoViewResult(ok=False, model="essential", planar_score=r_h,
                                 reason="findEssentialMat failed")
        # RANSAC can return several stacked candidate E matrices; take the first.
        E = E[:3, :3]
        n_good, R, t, mask_p = cv2.recoverPose(E, p1n, p2n, np.eye(3), mask=mask_e)
        inl = (mask_p.ravel() > 0)
        if np.count_nonzero(inl) < min_inliers and mask_e is not None:
            # recoverPose's mask additionally enforces cheirality, which is
            # degenerate at zero baseline. Fall back to the epipolar inliers so
            # a rotation-only measurement can still be produced.
            inl = (mask_e.ravel() > 0)
        t = t.reshape(3)
        model = "essential"

    n_in = int(np.count_nonzero(inl))
    if n_in < min_inliers:
        return TwoViewResult(ok=False, model=model, planar_score=r_h, n_inliers=n_in,
                             reason=f"only {n_in} inliers (< {min_inliers})")

    par = derotated_parallax(p1n[inl], p2n[inl], R)
    med_par = float(np.median(par)) if par.size else 0.0

    nt = float(np.linalg.norm(t))
    if nt < 1e-9:
        # Pure rotation. The translation is genuinely zero, but the *rotation*
        # is well conditioned and valuable -- returning a failure here would
        # throw away a perfectly good attitude constraint every time the pilot
        # yaws on the spot, which on a Tello is most of the flight.
        t_c1 = np.zeros(3)
        reliable = False
    else:
        # OpenCV -> estimator convention (see TwoViewResult).
        t_c1 = -(R.T @ (t / nt))
        reliable = med_par > np.deg2rad(parallax_thresh_deg)

    return TwoViewResult(
        ok=True, R=R, t=t_c1, inliers=inl, n_inliers=n_in,
        parallax_rad=med_par, model=model, planar_score=r_h,
        translation_reliable=reliable,
        reason="" if reliable else f"low parallax ({np.degrees(med_par):.2f} deg)",
    )


def _select_homography_solution(Rs, Ts, Ns, p1n, p2n):
    """Pick the physically valid one of the (up to 4) homography decompositions.

    ``cv2.decomposeHomographyMat`` returns every algebraically valid solution;
    only some put the observed points in front of both cameras. We score each by
    cheirality (how many triangulated points have positive depth in both views)
    and additionally require the plane normal to face the camera.

    When *no* candidate triangulates -- which is what happens under pure
    rotation, where the baseline is zero -- cheirality cannot discriminate and
    we fall back to the smallest-translation solution, which is the physically
    correct answer in exactly that case.
    """
    best = None
    best_n = -1
    fallback = None
    fallback_t = np.inf
    for R, t, n in zip(Rs, Ts, Ns):
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        n = np.asarray(n, dtype=np.float64).reshape(3)
        if n[2] < 0:                     # plane must be in front of camera 1
            continue
        nt = float(np.linalg.norm(t))
        if nt < fallback_t:
            fallback_t, fallback = nt, (R, t)
        _, good = triangulate(p1n, p2n, R, t)
        cnt = int(np.count_nonzero(good))
        if cnt > best_n:
            best_n, best = cnt, (R, t)
    if best_n > 0:
        return best
    return fallback


def triangulate(p1n: np.ndarray, p2n: np.ndarray, R: np.ndarray, t: np.ndarray,
                max_reproj: float = 0.01, min_parallax_rad: float = np.deg2rad(0.5)):
    """Linear triangulation of normalised correspondences.

    ``R, t`` take camera-1 coordinates into camera-2 (``x2 = R x1 + t``). The
    returned points are in **camera-1** coordinates, scaled by whatever
    ``|t|`` was passed in -- with a unit ``t`` they carry the same arbitrary
    monocular scale as the translation.

    The ``good`` mask enforces the three conditions a triangulated point must
    satisfy to be usable: positive depth in *both* views (cheirality), small
    reprojection error in both, and enough parallax that the depth is not
    numerically arbitrary. Skipping the parallax test is the classic way to
    fill a map with points at effectively infinite range that then destabilise
    every subsequent pose solve.
    """
    p1n = np.asarray(p1n, dtype=np.float64).reshape(-1, 2)
    p2n = np.asarray(p2n, dtype=np.float64).reshape(-1, 2)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3, 1)])

    X = cv2.triangulatePoints(P1, P2, p1n.T, p2n.T)
    w = X[3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    X = (X[:3] / w).T                                   # (N,3) in camera-1 frame

    finite = np.all(np.isfinite(X), axis=1)
    z1 = X[:, 2]
    X2 = X @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64).reshape(3)
    z2 = X2[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = np.linalg.norm(X[:, :2] / X[:, 2:3] - p1n, axis=1)
        e2 = np.linalg.norm(X2[:, :2] / X2[:, 2:3] - p2n, axis=1)
    e1 = np.nan_to_num(e1, nan=1e9, posinf=1e9)
    e2 = np.nan_to_num(e2, nan=1e9, posinf=1e9)

    par = derotated_parallax(p1n, p2n, np.asarray(R, dtype=np.float64))

    good = finite & (z1 > 1e-4) & (z2 > 1e-4) & (e1 < max_reproj) & (e2 < max_reproj) \
        & (par > min_parallax_rad)
    return X, good


def solve_pnp(points3d: np.ndarray, points2d_px: np.ndarray, K: np.ndarray,
              D: np.ndarray | None = None, R_init=None, t_init=None,
              reproj_px: float = 2.0, min_inliers: int = 8):
    """Robust PnP against a local 3-D map: the pose solve that actually holds up.

    Frame-to-frame two-view VO drifts fast because every frame's error is
    independent and they random-walk. Tracking against *triangulated 3-D points*
    ties the current pose to a common structure and slows that drift markedly.
    This is the same reason ORB-SLAM tracks the local map rather than the last
    frame.

    Returns ``(ok, R_cw, t_cw, inlier_index)`` where ``(R_cw, t_cw)`` maps world
    points into the camera: ``x_cam = R_cw @ X_world + t_cw``.
    """
    points3d = np.ascontiguousarray(np.asarray(points3d, dtype=np.float64).reshape(-1, 1, 3))
    points2d = np.ascontiguousarray(np.asarray(points2d_px, dtype=np.float64).reshape(-1, 1, 2))
    if len(points3d) < 4:
        return False, np.eye(3), np.zeros(3), None

    D = np.zeros(5) if D is None else np.asarray(D, dtype=np.float64).reshape(1, -1)
    use_guess = R_init is not None and t_init is not None
    rvec = cv2.Rodrigues(np.asarray(R_init, dtype=np.float64))[0] if use_guess else None
    tvec = np.asarray(t_init, dtype=np.float64).reshape(3, 1).copy() if use_guess else None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d, points2d, np.asarray(K, dtype=np.float64), D,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=bool(use_guess),
        reprojectionError=float(reproj_px), confidence=0.999, iterationsCount=200,
        flags=cv2.SOLVEPNP_ITERATIVE if use_guess else cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return False, np.eye(3), np.zeros(3), None

    idx = inliers.ravel()
    # Refine on the inlier set only; RANSAC's winning hypothesis is computed
    # from a minimal sample and is not itself a least-squares estimate.
    rvec, tvec = cv2.solvePnPRefineLM(
        points3d[idx], points2d[idx], np.asarray(K, dtype=np.float64), D, rvec, tvec
    )
    R, _ = cv2.Rodrigues(rvec)
    return True, R, tvec.reshape(3), idx
