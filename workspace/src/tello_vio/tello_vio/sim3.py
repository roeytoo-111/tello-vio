"""Sim(3) trajectory alignment -- how a scale-free SLAM map joins a metric frame.

ORB-SLAM2 is monocular, so its map and trajectory are defined only up to a
**similarity transform**: an unknown rotation, translation and *scale*. The VIO
estimator, by contrast, is metric, because it is fed the drone's optical-flow
velocity, barometer and ToF. Aligning the two gives you both halves of what you
want:

* the **scale** of the SLAM map, as a by-product; and
* the ``map -> odom`` correction that lets a loop-closing SLAM system remove
  the VIO's accumulated drift without the VIO output ever jumping.

That last point is the REP-105 pattern and it is worth being precise about.
``odom -> base_link`` must be *smooth and continuous* -- controllers
differentiate it, so a discontinuity there is a step input to the vehicle.
``map -> odom`` is allowed to jump, because nothing differentiates it. So a loop
closure is published as a correction to ``map -> odom``, leaving the VIO's
``odom -> base_link`` untouched. Publishing the loop closure straight into the
odometry instead -- the obvious-looking shortcut -- is what makes drones lurch
when a loop closes.

:func:`umeyama_sim3` is the closed-form least-squares solution (Umeyama 1991);
:func:`ransac_sim3` wraps it in RANSAC because a single mis-tracked SLAM segment
would otherwise drag the scale estimate, and scale errors are multiplicative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lie import project_to_so3


@dataclass
class Sim3:
    """``y ~= s * R @ x + t`` -- maps source points into the target frame."""

    s: float = 1.0
    R: np.ndarray = None
    t: np.ndarray = None

    def __post_init__(self):
        if self.R is None:
            self.R = np.eye(3)
        if self.t is None:
            self.t = np.zeros(3)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (self.s * (np.atleast_2d(x) @ self.R.T) + self.t).reshape(x.shape)

    def inverse(self) -> "Sim3":
        return Sim3(1.0 / self.s, self.R.T, -(1.0 / self.s) * (self.R.T @ self.t))

    def matrix(self) -> np.ndarray:
        """4x4 homogeneous form (scale folded into the rotation block)."""
        M = np.eye(4)
        M[:3, :3] = self.s * self.R
        M[:3, 3] = self.t
        return M


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Closed-form least-squares similarity fit (Umeyama 1991).

    Minimises ``sum ||dst_i - (s R src_i + t)||^2`` in one SVD. The subtlety the
    paper exists to fix is the reflection case: naively taking ``R = U V^T``
    from the SVD can return a matrix with ``det = -1`` -- a reflection, not a
    rotation -- when the point sets are nearly coplanar, which for a drone
    flying a mostly-planar path is the normal case, not an edge case. The
    ``diag(1,1,sign(det))`` correction below is what prevents it.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("need at least 3 matched points")

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = float(np.mean(np.sum(xs ** 2, axis=1)))

    Sigma = (xd.T @ xs) / len(src)
    U, Dv, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    R = project_to_so3(R)

    if with_scale and var_s > 1e-12:
        s = float(np.trace(np.diag(Dv) @ S) / var_s)
    else:
        s = 1.0
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return Sim3(s, R, t)


def ransac_sim3(src: np.ndarray, dst: np.ndarray, threshold: float = 0.15,
                iterations: int = 100, min_inlier_ratio: float = 0.5,
                with_scale: bool = True, seed: int = 0):
    """RANSAC around :func:`umeyama_sim3`. Returns ``(Sim3, inlier_mask)``.

    A single bad correspondence -- one SLAM segment that tracked the wrong
    structure -- moves a least-squares scale estimate by a multiplicative
    factor, and multiplicative errors do not average out. Hence RANSAC on a
    problem that has a closed-form solution.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = len(src)
    if n < 3:
        raise ValueError("need at least 3 matched points")
    if n == 3:
        T = umeyama_sim3(src, dst, with_scale)
        return T, np.ones(3, dtype=bool)

    rng = np.random.default_rng(seed)
    best_T, best_inl = None, np.zeros(n, dtype=bool)
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        try:
            T = umeyama_sim3(src[idx], dst[idx], with_scale)
        except (ValueError, np.linalg.LinAlgError):
            continue
        err = np.linalg.norm(T.apply(src) - dst, axis=1)
        inl = err < threshold
        if np.count_nonzero(inl) > np.count_nonzero(best_inl):
            best_T, best_inl = T, inl

    if best_T is None or np.count_nonzero(best_inl) < max(3, int(min_inlier_ratio * n)):
        # Fall back to the plain fit rather than returning nothing: a poor
        # alignment that is flagged as poor is more useful downstream than a
        # missing one.
        return umeyama_sim3(src, dst, with_scale), np.ones(n, dtype=bool)

    # Refit on the consensus set -- the minimal-sample hypothesis is not itself
    # a least-squares estimate.
    return umeyama_sim3(src[best_inl], dst[best_inl], with_scale), best_inl
