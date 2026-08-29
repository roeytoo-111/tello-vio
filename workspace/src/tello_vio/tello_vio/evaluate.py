"""Compare an estimated trajectory against fiducial ground truth.

Produces the standard SLAM/VIO metrics and a plot:

* **ATE** (absolute trajectory error) after Umeyama alignment. Two variants
  are reported and the difference between them is the point:
    - ``ATE (SE3)``  -- rotation+translation only, scale forced to 1. This is
      the honest metric for a METRIC estimator: it holds the estimator to its
      claim of producing real metres.
    - ``ATE (Sim3)`` -- scale also fitted, and the fitted scale reported. A
      monocular system with no metric sensing can only be judged this way.
      If Sim3 error is far below SE3 error, the trajectory SHAPE is right but
      the SCALE is wrong -- which points at calibration, not at the filter.
* **RPE** (relative pose error) over a fixed window: drift rate, insensitive
  to a single early mistake that ATE would smear across the whole run.

Alignment uses the estimator's own frame convention: both trajectories are
resampled onto common timestamps by interpolation before comparison, because
comparing index-wise across two irregular streams silently pairs samples that
are hundreds of milliseconds apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim3 import umeyama_sim3


@dataclass
class TrajectoryError:
    n: int
    duration_s: float
    path_length_m: float
    ate_se3_rmse: float
    ate_se3_max: float
    ate_sim3_rmse: float
    fitted_scale: float
    rpe_rmse: float
    rpe_window_s: float
    drift_percent: float

    def report(self) -> str:
        return (
            f"samples            : {self.n} over {self.duration_s:.1f} s\n"
            f"path length        : {self.path_length_m:.2f} m\n"
            f"ATE (SE3)  RMSE    : {self.ate_se3_rmse:.3f} m   max {self.ate_se3_max:.3f} m\n"
            f"ATE (Sim3) RMSE    : {self.ate_sim3_rmse:.3f} m   "
            f"(fitted scale {self.fitted_scale:.3f})\n"
            f"RPE over {self.rpe_window_s:.1f} s   : {self.rpe_rmse:.3f} m\n"
            f"drift              : {self.drift_percent:.2f} % of path length")


def resample_to(t_ref: np.ndarray, t_src: np.ndarray, p_src: np.ndarray,
                max_gap_s: float = 0.25):
    """Interpolate ``p_src`` onto ``t_ref``; drop points across long gaps.

    The gap check matters: linear interpolation across a 3 s dropout invents a
    straight line the drone never flew, and that fiction would be scored as
    estimator error.
    """
    t_ref = np.asarray(t_ref, float).ravel()
    t_src = np.asarray(t_src, float).ravel()
    p_src = np.asarray(p_src, float).reshape(-1, 3)
    order = np.argsort(t_src)
    t_src, p_src = t_src[order], p_src[order]

    inside = (t_ref >= t_src[0]) & (t_ref <= t_src[-1])
    out = np.full((len(t_ref), 3), np.nan)
    if not inside.any():
        return out, inside
    idx = np.searchsorted(t_src, t_ref[inside]).clip(1, len(t_src) - 1)
    gap = t_src[idx] - t_src[idx - 1]
    ok = gap <= max_gap_s
    for k in range(3):
        out[inside, k] = np.interp(t_ref[inside], t_src, p_src[:, k])
    sel = np.zeros(len(t_ref), dtype=bool)
    sel[np.where(inside)[0][ok]] = True
    return out, sel


def evaluate(t_est: np.ndarray, p_est: np.ndarray,
             t_gt: np.ndarray, p_gt: np.ndarray,
             rpe_window_s: float = 1.0) -> TrajectoryError:
    """Compute ATE/RPE between an estimate and ground truth."""
    t_gt = np.asarray(t_gt, float).ravel()
    p_gt = np.asarray(p_gt, float).reshape(-1, 3)
    p_on_gt, sel = resample_to(t_gt, t_est, p_est)
    if np.count_nonzero(sel) < 10:
        raise ValueError("fewer than 10 overlapping samples; "
                         "were both topics recorded over the same interval?")
    t = t_gt[sel]
    est = p_on_gt[sel]
    gt = p_gt[sel]

    path = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))

    T_se3 = umeyama_sim3(est, gt, with_scale=False)
    e_se3 = np.linalg.norm(T_se3.apply(est) - gt, axis=1)
    T_sim3 = umeyama_sim3(est, gt, with_scale=True)
    e_sim3 = np.linalg.norm(T_sim3.apply(est) - gt, axis=1)

    # RPE: displacement over a fixed time window, estimate vs truth.
    rpe = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + rpe_window_s)
        if j >= len(t):
            break
        d_est = np.linalg.norm(est[j] - est[i])
        d_gt = np.linalg.norm(gt[j] - gt[i])
        rpe.append(d_est - d_gt)
    rpe = np.asarray(rpe) if rpe else np.zeros(1)

    return TrajectoryError(
        n=int(len(t)),
        duration_s=float(t[-1] - t[0]),
        path_length_m=path,
        ate_se3_rmse=float(np.sqrt(np.mean(e_se3 ** 2))),
        ate_se3_max=float(np.max(e_se3)),
        ate_sim3_rmse=float(np.sqrt(np.mean(e_sim3 ** 2))),
        fitted_scale=float(T_sim3.s),
        rpe_rmse=float(np.sqrt(np.mean(rpe ** 2))),
        rpe_window_s=float(rpe_window_s),
        drift_percent=float(100.0 * np.sqrt(np.mean(e_se3 ** 2)) / max(path, 1e-6)),
    )


def aligned_pair(t_est, p_est, t_gt, p_gt, with_scale=False):
    """Return ``(t, est_aligned, gt)`` for plotting."""
    t_gt = np.asarray(t_gt, float).ravel()
    p_gt = np.asarray(p_gt, float).reshape(-1, 3)
    p_on_gt, sel = resample_to(t_gt, t_est, p_est)
    est, gt, t = p_on_gt[sel], p_gt[sel], t_gt[sel]
    T = umeyama_sim3(est, gt, with_scale=with_scale)
    return t, T.apply(est), gt
