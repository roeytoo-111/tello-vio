"""Monocular visual front-end: track features, choose keyframes, emit relative pose.

Design brief: produce a relative-pose measurement per keyframe, on a laptop CPU,
inside the time budget of a 30 Hz stream, from a soft 960x720 H.264 feed that
arrives 200-300 ms late over WiFi.

Why KLT and not ORB-per-frame
-----------------------------
ORB-SLAM2 detects and describes ~1000 ORB features on *every* frame and matches
them by Hamming distance. That is the right architecture when you also want
relocalisation and loop closure from the same features -- and it is why
ORB-SLAM2 remains the map/loop-closure backend in this project. It is the wrong
architecture for a low-latency odometry front-end: detect+describe dominates the
per-frame cost.

Sparse **Lucas-Kanade** tracking instead reuses the same features across frames
and only pays for detection when the track count falls. On 480x360 with ~150
features, pyramidal KLT is a few milliseconds per frame -- roughly an order of
magnitude cheaper than ORB detect+describe+match, with better sub-pixel accuracy
between adjacent frames (it is a direct photometric alignment, not a
quantised binary descriptor match). Its weakness -- no descriptors, so no
relocalisation after a total loss -- is exactly the gap the ORB-SLAM2 backend
fills. The two are complementary, not redundant.

Three engineering details that decide whether this works or not
--------------------------------------------------------------
* **Forward-backward validation.** Track a point forward, then track the result
  back; if it does not land where it started, the track is wrong. This single
  test removes most of the drift-inducing bad matches that RANSAC would
  otherwise have to absorb, and it costs one extra KLT call.
* **Grid bucketing on detection.** ``goodFeaturesToTrack`` piles features onto
  the highest-contrast corner of the image -- typically one window or one poster.
  Features clustered in a small image region give a nearly degenerate geometry
  and a translation direction with enormous variance. Forcing a per-cell quota
  spreads them out and conditions the whole estimate.
* **Never undistort the image.** Undistorting a full frame every cycle is pure
  waste. Distortion is a smooth warp that KLT handles happily; the correction is
  applied to the handful of *points* that reach the geometry stage
  (:func:`tello_vio.two_view.normalise_points`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .two_view import TwoViewResult, estimate_relative_pose


@dataclass
class FrontendConfig:
    """Front-end tuning. Defaults target a laptop-class CPU at 480x360."""

    #: Working resolution. The Tello streams 960x720; halving it cuts the KLT
    #: and detection cost ~4x and costs surprisingly little accuracy, because
    #: the stream is soft from H.264 compression well before it is
    #: resolution-limited.
    work_width: int = 480

    max_features: int = 180
    min_features: int = 90            # re-detect below this
    quality_level: float = 0.01
    min_distance: int = 12            # px, between detected corners
    grid_cols: int = 6
    grid_rows: int = 5

    klt_win: int = 21
    klt_levels: int = 3
    fb_threshold: float = 1.0         # px, forward-backward consistency

    #: Keyframe triggers. A keyframe is made when *any* fires. When a rotation
    #: prior is supplied the displacement is measured *after* compensating for
    #: rotation -- see :meth:`VoFrontend.process`.
    kf_min_parallax_px: float = 12.0
    kf_max_tracked_ratio: float = 0.6
    kf_max_interval_s: float = 1.0
    kf_min_interval_s: float = 0.08

    ransac_px: float = 1.5
    min_inliers: int = 20
    parallax_thresh_deg: float = 1.0


@dataclass
class VisualMeasurement:
    """One relative-pose measurement between the reference keyframe and now."""

    stamp: float                 # stamp of the *current* frame
    ref_stamp: float             # stamp of the reference keyframe
    R_c1c2: np.ndarray           # rotation, camera-1 -> camera-2
    t_dir_c1: np.ndarray         # unit translation direction in camera-1
    translation_reliable: bool
    n_inliers: int
    parallax_rad: float
    model: str
    planar_score: float
    n_tracked: int
    cost_ms: float = 0.0

    #: Extra rotation uncertainty charged to the homography path, which is
    #: chosen on plane-dominated or low-parallax views where the decomposition
    #: is markedly more fragile. Calibrated, not guessed -- see :meth:`noise_std`.
    HOMOGRAPHY_ROT_PENALTY = 3.5

    def noise_std(self, base_rot_std: float = 0.033,
                  base_dir_std: float = 0.128) -> tuple:
        """Measurement 1-sigmas for *this particular* observation, in radians.

        A fixed ``R`` matrix is wrong here: a 170-inlier essential-matrix solve
        at 3 degrees of parallax and a 30-inlier homography at 0.3 degrees are
        not the same measurement, and handing both to the filter with equal
        confidence lets the bad ones dominate the good ones. The model captures
        the two dominant effects:

        * error falls as ``1/sqrt(N_inliers)`` -- averaging over correspondences;
        * translation *direction* error scales as ``1/parallax`` -- the geometry
          of a long thin triangle, which degrades fast as the baseline shrinks.

        **Provenance of the constants.** They are fitted, not invented. Running
        the rendered fly-through in ``test_frontend.py`` over 8 seeds and 64
        keyframes and regressing the observed errors onto the shape above gives
        ``base_rot_std = 1.87 deg`` and ``base_dir_std = 7.3 deg``, at which the
        normalised errors have a 90th percentile of 1.4-1.7 sigma and only ~2 %
        beyond 3 sigma -- i.e. an honestly calibrated Gaussian.

        Those figures come from *synthetic* imagery with 0.3 px feature noise.
        A real Tello feed -- H.264 artefacts, rolling shutter, motion blur,
        low indoor light -- is worse, so treat these as a floor and re-fit
        against a recorded bag before trusting them in flight. Both are node
        parameters for exactly that reason.
        """
        n = max(8, self.n_inliers)
        k_n = np.sqrt(60.0 / n)
        rot = base_rot_std * k_n
        if self.model == "homography":
            rot *= self.HOMOGRAPHY_ROT_PENALTY
        par_deg = max(0.2, np.degrees(self.parallax_rad))
        direction = base_dir_std * k_n * max(1.0, 3.0 / par_deg)
        return float(rot), float(direction)


class FeatureTracker:
    """Grid-bucketed Shi-Tomasi detection plus forward-backward KLT tracking."""

    def __init__(self, cfg: FrontendConfig):
        self.cfg = cfg
        self.prev_gray: np.ndarray | None = None
        self.pts: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self.ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._next_id = 0
        self._lk = dict(
            winSize=(cfg.klt_win, cfg.klt_win),
            maxLevel=cfg.klt_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def reset(self) -> None:
        self.prev_gray = None
        self.pts = np.zeros((0, 2), dtype=np.float32)
        self.ids = np.zeros((0,), dtype=np.int64)

    # ------------------------------------------------------------------ #

    def track(self, gray: np.ndarray):
        """Advance all tracks into ``gray``. Returns ``(pts, ids, kept_mask)``.

        ``kept_mask`` is over the *previous* point set, so callers can index
        their own parallel arrays (e.g. the keyframe's observations) without
        having to re-match by id.
        """
        if self.prev_gray is None or len(self.pts) == 0:
            self.prev_gray = gray
            return self.pts, self.ids, np.zeros(len(self.pts), dtype=bool)

        p0 = self.pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self._lk)
        if p1 is None:
            self.prev_gray = gray
            kept = np.zeros(len(self.pts), dtype=bool)
            self.pts = np.zeros((0, 2), dtype=np.float32)
            self.ids = np.zeros((0,), dtype=np.int64)
            return self.pts, self.ids, kept

        # Backward pass: a correct track returns to where it started.
        p0r, st0, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self._lk)
        fb = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)

        h, w = gray.shape[:2]
        q = p1.reshape(-1, 2)
        inside = (q[:, 0] >= 1) & (q[:, 0] < w - 1) & (q[:, 1] >= 1) & (q[:, 1] < h - 1)
        kept = (st1.ravel() == 1) & (st0.ravel() == 1) & (fb < self.cfg.fb_threshold) & inside

        self.pts = q[kept].astype(np.float32)
        self.ids = self.ids[kept]
        self.prev_gray = gray
        return self.pts, self.ids, kept

    def detect(self, gray: np.ndarray) -> int:
        """Top the track set back up, one quota per grid cell. Returns count added."""
        cfg = self.cfg
        need = cfg.max_features - len(self.pts)
        if need <= 0:
            return 0

        h, w = gray.shape[:2]
        # Mask out neighbourhoods of existing tracks so new corners are not
        # detected on top of ones we already have.
        mask = np.full((h, w), 255, dtype=np.uint8)
        for x, y in self.pts:
            cv2.circle(mask, (int(x), int(y)), cfg.min_distance, 0, -1)

        cw, ch = w // cfg.grid_cols, h // cfg.grid_rows
        per_cell = max(1, int(np.ceil(need / (cfg.grid_cols * cfg.grid_rows))))
        added = []
        for gy in range(cfg.grid_rows):
            for gx in range(cfg.grid_cols):
                x0, y0 = gx * cw, gy * ch
                x1 = w if gx == cfg.grid_cols - 1 else x0 + cw
                y1 = h if gy == cfg.grid_rows - 1 else y0 + ch
                sub = gray[y0:y1, x0:x1]
                sub_mask = mask[y0:y1, x0:x1]
                if sub.size == 0 or not sub_mask.any():
                    continue
                c = cv2.goodFeaturesToTrack(
                    sub, maxCorners=per_cell, qualityLevel=cfg.quality_level,
                    minDistance=cfg.min_distance, mask=sub_mask,
                )
                if c is None:
                    continue
                c = c.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
                added.append(c)

        if not added:
            return 0
        new = np.vstack(added).astype(np.float32)[:need]
        if len(new):
            # Sub-pixel refinement: KLT is already sub-pixel, but the corner
            # *seeds* are integer, and a half-pixel seeding bias shows up
            # directly as a bias in the estimated translation direction.
            cv2.cornerSubPix(
                gray, new.reshape(-1, 1, 2), (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            ids = np.arange(self._next_id, self._next_id + len(new), dtype=np.int64)
            self._next_id += len(new)
            self.pts = np.vstack([self.pts, new]) if len(self.pts) else new
            self.ids = np.concatenate([self.ids, ids])
        return len(new)


class VoFrontend:
    """Keyframe-based monocular VO producing :class:`VisualMeasurement` objects.

    The scale of ``t_dir_c1`` is deliberately *not* estimated here. This class
    reports geometry only; metric scale is the estimator's job
    (:mod:`tello_vio.eskf`), fed by the drone's own metric sensors.
    """

    def __init__(self, K: np.ndarray, D: np.ndarray | None = None,
                 cfg: FrontendConfig | None = None):
        self.cfg = cfg or FrontendConfig()
        self.K_full = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.zeros(5) if D is None else np.asarray(D, dtype=np.float64).ravel()
        self.tracker = FeatureTracker(self.cfg)

        self._scale = 1.0
        self.K = self.K_full.copy()

        self.kf_pts: np.ndarray | None = None     # keyframe observations
        self.kf_ids: np.ndarray | None = None
        self.kf_stamp: float = 0.0
        self.frame_idx = 0
        self.last_status = "init"

    # ------------------------------------------------------------------ #

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """Grayscale + downscale, and rescale the intrinsics to match.

        Rescaling ``K`` alongside the image is not optional. Downscaling by
        ``s`` scales ``fx, fy, cx, cy`` by ``s``; using the full-resolution
        ``K`` on half-resolution pixels doubles the apparent focal length and
        every derived angle is then wrong by a factor of two.

        The distortion coefficients, by contrast, are defined in *normalised*
        coordinates and are therefore resolution-independent -- they must NOT
        be scaled. Scaling them is a common and quiet error.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        w = gray.shape[1]
        if self.cfg.work_width and w > self.cfg.work_width:
            s = self.cfg.work_width / float(w)
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            s = 1.0

        if s != self._scale:
            self._scale = s
            self.K = self.K_full.copy()
            self.K[0, 0] *= s
            self.K[1, 1] *= s
            self.K[0, 2] = (self.K_full[0, 2] + 0.5) * s - 0.5
            self.K[1, 2] = (self.K_full[1, 2] + 0.5) * s - 0.5
        return gray

    def rotation_compensated_flow(self, R_pred_c1c2: np.ndarray) -> np.ndarray:
        """Per-track pixel displacement with the predicted rotation removed.

        A rotation induces pixel motion identical in character to translation,
        so a raw-displacement keyframe trigger fires on *turning*, not on
        *moving*. That is the worst possible time to make a keyframe: the
        baseline is near zero, so parallax is near zero, the essential matrix is
        ill conditioned, model selection drifts to the homography, and the
        recovered rotation degrades from ~0.2 to several degrees.

        Removing the prior rotation costs one 3x3 matrix and a projective
        divide over ~150 points -- microseconds -- and makes the trigger
        measure the thing it is supposed to measure: parallax. The prior comes
        free from the drone's own attitude stream, which is the one piece of
        inertial data a Tello gives us in good shape.
        """
        if self.kf_pts is None or len(self.kf_pts) == 0:
            return np.zeros(0)
        # Infinite-homography induced by pure rotation: H = K R K^-1.
        Hrot = self.K @ np.asarray(R_pred_c1c2, dtype=np.float64) @ np.linalg.inv(self.K)
        p = np.hstack([self.kf_pts, np.ones((len(self.kf_pts), 1))]) @ Hrot.T
        w = p[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        pred = p[:, :2] / w
        return np.linalg.norm(self.tracker.pts - pred, axis=1)

    def process(self, image: np.ndarray, stamp: float,
                R_pred_c1c2: np.ndarray | None = None) -> VisualMeasurement | None:
        """Feed one frame. Returns a measurement only when a keyframe is made.

        ``R_pred_c1c2`` is an optional rotation prior (camera-1 to camera-2)
        from the estimator or the drone's AHRS. Supplying it makes the keyframe
        trigger measure true parallax instead of total pixel motion, which
        materially improves the quality of every measurement this class emits.
        """
        t0 = time.perf_counter()
        gray = self._prepare(image)
        self.frame_idx += 1

        if self.tracker.prev_gray is None:
            self.tracker.prev_gray = gray
            self.tracker.detect(gray)
            self._set_keyframe(stamp)
            self.last_status = "bootstrap"
            return None

        pts, ids, kept = self.tracker.track(gray)

        # Keep the keyframe's observations aligned with the surviving tracks.
        if self.kf_pts is not None and len(kept) == len(self.kf_pts):
            self.kf_pts = self.kf_pts[kept]
            self.kf_ids = self.kf_ids[kept]

        n_tracked = len(pts)
        if n_tracked < self.cfg.min_features:
            self.tracker.detect(gray)

        if self.kf_pts is None or n_tracked < 8:
            self._set_keyframe(stamp)
            self.last_status = "too few tracks"
            return None

        dt = stamp - self.kf_stamp
        if n_tracked == 0:
            disp = 0.0
        elif R_pred_c1c2 is not None:
            disp = float(np.median(self.rotation_compensated_flow(R_pred_c1c2)))
        else:
            disp = float(np.median(np.linalg.norm(pts - self.kf_pts, axis=1)))
        ratio = n_tracked / max(1, len(self.kf_ids))

        need_kf = (
            dt >= self.cfg.kf_min_interval_s
            and (disp > self.cfg.kf_min_parallax_px
                 or ratio < self.cfg.kf_max_tracked_ratio
                 or dt > self.cfg.kf_max_interval_s)
        )
        if not need_kf:
            self.last_status = f"tracking ({n_tracked})"
            return None

        res: TwoViewResult = estimate_relative_pose(
            self.kf_pts, pts, self.K, self.D,
            px_threshold=self.cfg.ransac_px,
            min_inliers=self.cfg.min_inliers,
            parallax_thresh_deg=self.cfg.parallax_thresh_deg,
        )

        meas = None
        if res.ok:
            meas = VisualMeasurement(
                stamp=stamp, ref_stamp=self.kf_stamp,
                R_c1c2=res.R, t_dir_c1=res.t,
                translation_reliable=res.translation_reliable,
                n_inliers=res.n_inliers, parallax_rad=res.parallax_rad,
                model=res.model, planar_score=res.planar_score,
                n_tracked=n_tracked,
                cost_ms=(time.perf_counter() - t0) * 1e3,
            )
            self.last_status = f"{res.model} {res.n_inliers}in"
        else:
            self.last_status = f"failed: {res.reason}"

        # Re-anchor regardless: a failed keyframe attempt means the current
        # reference has gone stale, and retrying against it will keep failing.
        self.tracker.detect(gray)
        self._set_keyframe(stamp)
        return meas

    def _set_keyframe(self, stamp: float) -> None:
        self.kf_pts = self.tracker.pts.copy()
        self.kf_ids = self.tracker.ids.copy()
        self.kf_stamp = float(stamp)

    # ------------------------------------------------------------------ #

    def draw(self, image: np.ndarray) -> np.ndarray:
        """Debug overlay: tracks, their flow since the keyframe, and status."""
        vis = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        vis = vis.copy()
        s = 1.0 / self._scale if self._scale else 1.0
        pts = self.tracker.pts
        kf = self.kf_pts
        if kf is not None and len(kf) == len(pts):
            for (x0, y0), (x1, y1) in zip(kf * s, pts * s):
                cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 1)
        for x, y in pts * s:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(vis, f"VO: {self.last_status}  n={len(pts)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return vis
