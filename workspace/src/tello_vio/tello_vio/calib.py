"""Calibration: IMU noise identification, camera-IMU extrinsics, time offset.

Three quantities have to be known before any VIO result means anything, and on
a Tello none of them is published by the manufacturer:

1. **IMU noise densities and biases.** They set the process noise ``Q``. Guess
   them too small and the filter over-trusts dead reckoning and diverges; too
   large and it ignores the IMU entirely. :func:`identify_imu_noise` measures
   them from a static log using the overlapping Allan deviation, which is the
   standard tool for exactly this and separates white noise from bias
   instability rather than lumping them.

2. **The camera-IMU rotation** ``R_BC``. Every visual measurement is expressed
   in the camera frame and every inertial one in the body frame; without the
   rotation between them the two cannot be combined at all. A 5-degree error
   here couples rotation directly into apparent translation and produces a
   drift that looks exactly like scale error, which is why people chase the
   wrong bug for days. :func:`estimate_camera_imu_rotation` solves it from
   motion alone -- no target, no CAD -- using the rotation-only hand-eye
   identity.

3. **The camera-IMU time offset** ``t_d``. The Tello's video and telemetry
   arrive on different sockets with different, unequal latencies and no
   hardware timestamps. At 1 rad/s of body rate, 50 ms of unmodelled offset is
   50 mrad (~3 deg) of rotation error injected into every single visual
   measurement -- larger than the measurement noise by an order of magnitude.
   :func:`estimate_time_offset` recovers it by cross-correlating angular-rate
   magnitudes, which is the one signal both sensors observe independently.

All three are offline procedures run from a recorded bag, deliberately: they
are far better conditioned as batch problems than as extra states in a filter
that also has to fly the drone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lie import Log, project_to_so3


# --------------------------------------------------------------------------- #
# 1. IMU noise identification
# --------------------------------------------------------------------------- #

def allan_deviation(x: np.ndarray, dt: float, n_clusters: int = 40):
    """Overlapping Allan deviation of a 1-D sequence.

    Returns ``(taus, adev)``. The classic reading of the log-log plot:

    * slope ``-1/2`` region -> **white noise**; the density ``N`` (units/sqrt(Hz))
      is the value of the fitted line at ``tau = 1 s``.
    * flat minimum -> **bias instability**.
    * slope ``+1/2`` region -> **random walk**; ``K`` is the fitted line at
      ``tau = 3 s``.

    The *overlapping* estimator is used rather than the simpler
    non-overlapping one because it makes far better use of a short log --
    and Tello logs are short, since the drone overheats on the bench.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 16:
        raise ValueError("need at least 16 samples for an Allan deviation")

    theta = np.cumsum(x) * dt                     # integrate to the "angle" domain
    max_m = int(n // 5)
    ms = np.unique(np.floor(np.logspace(0, np.log10(max(2, max_m)), n_clusters)).astype(int))
    ms = ms[(ms >= 1) & (ms <= max_m)]

    taus, adev = [], []
    for m in ms:
        tau = m * dt
        k = n - 2 * m
        if k < 1:
            continue
        d = theta[2 * m:] - 2.0 * theta[m:n - m] + theta[:k]
        var = np.sum(d ** 2) / (2.0 * tau ** 2 * k)
        taus.append(tau)
        adev.append(np.sqrt(var))
    return np.array(taus), np.array(adev)


@dataclass
class ImuNoiseIdentification:
    """Per-axis noise identification result, plus the isotropic summary used by
    :class:`tello_vio.preintegration.ImuNoiseModel`."""

    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    accel_noise_density: float
    gyro_noise_density: float
    accel_bias_rw: float
    gyro_bias_rw: float
    gravity_magnitude: float
    n_samples: int
    duration_s: float
    warnings: tuple = ()


def _fit_slope_value(taus, adev, slope, at_tau):
    """Fit a fixed-slope line in log-log and read it off at ``at_tau``.

    Restricted to the decade of ``tau`` where that slope should dominate, so a
    white-noise fit is not contaminated by the random-walk tail and vice versa.
    """
    if slope < 0:
        sel = taus <= max(taus.min() * 10.0, np.percentile(taus, 25))
    else:
        sel = taus >= min(taus.max() / 10.0, np.percentile(taus, 75))
    if np.count_nonzero(sel) < 2:
        sel = np.ones_like(taus, dtype=bool)
    # log(adev) = log(c) + slope*log(tau)  =>  log(c) = mean(log adev - slope log tau)
    log_c = np.mean(np.log(adev[sel]) - slope * np.log(taus[sel]))
    return float(np.exp(log_c) * at_tau ** slope)


def identify_imu_noise(accel: np.ndarray, gyro: np.ndarray | None, dt: float,
                       expect_gravity: bool = True) -> ImuNoiseIdentification:
    """Fit biases and noise densities from a **static** IMU log.

    ``accel`` must be ``(N, 3)`` specific force in m/s^2 and ``gyro`` ``(N, 3)``
    rad/s (or ``None`` on a stock Tello, where no gyro exists and the surrogate
    rate must be supplied by the caller if it wants gyro numbers).

    The drone must be *motionless and level-ish* for the whole log, on a
    surface that is not transmitting vibration. Any real motion inflates the
    apparent noise, which quietly makes the filter sluggish forever after.
    Detected non-stationarity is reported in ``warnings`` rather than silently
    folded into the answer.
    """
    accel = np.asarray(accel, dtype=np.float64).reshape(-1, 3)
    n = len(accel)
    warnings = []

    g_meas = float(np.linalg.norm(np.mean(accel, axis=0)))
    if expect_gravity and not (8.5 < g_meas < 11.0):
        warnings.append(
            f"mean |accel| = {g_meas:.2f} m/s^2 is not ~9.81; units or frame are "
            "probably wrong (raw milli-g is a common cause)")

    # A static accelerometer reads gravity, so the "bias" is only identifiable
    # once gravity is projected out. With the drone level, gravity is along
    # +z and the horizontal components are pure bias; z is confounded with the
    # local gravity magnitude and any tilt, so it is the least trustworthy.
    mean = np.mean(accel, axis=0)
    accel_bias = mean.copy()
    if expect_gravity:
        accel_bias = accel_bias - np.array([0.0, 0.0, g_meas])

    drift = np.linalg.norm(np.mean(accel[: n // 4], axis=0) - np.mean(accel[-n // 4:], axis=0))
    if drift > 0.15:
        warnings.append(f"accel mean drifted {drift:.3f} m/s^2 across the log; "
                        "the drone probably moved or the log is too short")

    a_nd, a_rw = [], []
    for k in range(3):
        taus, adev = allan_deviation(accel[:, k], dt)
        a_nd.append(_fit_slope_value(taus, adev, -0.5, 1.0))
        a_rw.append(_fit_slope_value(taus, adev, +0.5, 3.0))

    if gyro is not None:
        gyro = np.asarray(gyro, dtype=np.float64).reshape(-1, 3)
        gyro_bias = np.mean(gyro, axis=0)
        g_nd, g_rw = [], []
        for k in range(3):
            taus, adev = allan_deviation(gyro[:, k], dt)
            g_nd.append(_fit_slope_value(taus, adev, -0.5, 1.0))
            g_rw.append(_fit_slope_value(taus, adev, +0.5, 3.0))
    else:
        gyro_bias = np.zeros(3)
        g_nd = [np.nan]
        g_rw = [np.nan]
        warnings.append("no gyro supplied: a stock Tello publishes attitude, not "
                        "rate; feed TelloImuSurrogate output to characterise it")

    return ImuNoiseIdentification(
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
        # Use the worst axis, not the mean: the filter is isotropic, and being
        # optimistic on the noisiest axis is what makes it diverge.
        accel_noise_density=float(np.max(a_nd)),
        gyro_noise_density=float(np.max(g_nd)),
        accel_bias_rw=float(np.max(a_rw)),
        gyro_bias_rw=float(np.max(g_rw)),
        gravity_magnitude=g_meas,
        n_samples=n,
        duration_s=n * dt,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# 2. Camera-IMU rotation (rotation-only hand-eye)
# --------------------------------------------------------------------------- #

@dataclass
class HandEyeResult:
    R_BC: np.ndarray
    residual_deg: float
    n_used: int
    n_rejected: int
    angle_mismatch_deg: float


def estimate_camera_imu_rotation(R_cam_rel, R_body_rel,
                                 min_angle_deg: float = 3.0,
                                 max_angle_mismatch_deg: float = 5.0) -> HandEyeResult:
    """Solve ``R_BC`` from paired relative rotations. No calibration target needed.

    The rigid-body identity for a camera bolted to a body is::

        R_C1C2 = R_CB  R_B1B2  R_BC          (R_CB = R_BC^T)

    Conjugation preserves the rotation *angle* and maps the *axis*, so with
    ``n_C``/``n_B`` the unit rotation axes of the two measured increments::

        n_C = R_CB n_B                and       |theta_C| = |theta_B|

    That reduces hand-eye to Wahba's problem -- find the rotation best aligning
    two sets of unit vectors -- solved in closed form by SVD (Kabsch). No
    iteration, no initial guess, no local minimum.

    Two practicalities decide whether it works:

    * **Excitation.** Rotation about a single axis leaves ``R_BC`` undetermined
      about that axis; you must rotate about at least two non-parallel axes.
      Pick the drone up and rotate it about all three.
    * **Screening, in two stages.** Small rotations have noise-dominated axes
      and pollute the sum, so increments below ``min_angle_deg`` are dropped,
      as are pairs whose two angles disagree by more than
      ``max_angle_mismatch_deg`` (a violation of the identity -- almost always
      a time-sync error). That cheap screen is *not sufficient*: a corrupted
      axis that happens to preserve the rotation angle passes it untouched and
      biases the closed-form fit by degrees. A second stage therefore re-fits
      after rejecting pairs whose full conjugation residual exceeds a
      median+3*MAD cut.

    Translation ``p_BC`` is *not* estimated here. It needs metric scale and
    well-excited translation; on a Tello the camera-to-IMU lever arm is a few
    centimetres and is far better taken from the mechanical drawing than fitted
    from noisy data.
    """
    R_cam_rel = list(R_cam_rel)
    R_body_rel = list(R_body_rel)
    if len(R_cam_rel) != len(R_body_rel):
        raise ValueError("paired inputs must have equal length")

    min_ang = np.deg2rad(min_angle_deg)
    max_mis = np.deg2rad(max_angle_mismatch_deg)

    axes_c, axes_b, mismatches, kept_idx = [], [], [], []
    n_rejected = 0
    for i, (Rc, Rb) in enumerate(zip(R_cam_rel, R_body_rel)):
        vc, vb = Log(np.asarray(Rc, float)), Log(np.asarray(Rb, float))
        ac, ab = np.linalg.norm(vc), np.linalg.norm(vb)
        if ac < min_ang or ab < min_ang:
            n_rejected += 1
            continue
        if abs(ac - ab) > max_mis:
            n_rejected += 1
            continue
        mismatches.append(abs(ac - ab))
        # Weight by rotation angle: a 30 deg increment carries far more
        # directional information than a 4 deg one.
        axes_c.append(vc / ac * ac)
        axes_b.append(vb / ab * ab)
        kept_idx.append(i)

    if len(axes_c) < 3:
        raise ValueError(
            f"only {len(axes_c)} usable rotation pairs; rotate the drone about "
            "at least two non-parallel axes by more than "
            f"{min_angle_deg} deg each")

    kept_c = np.asarray(axes_c)      # camera-frame axes, weighted by angle
    kept_b = np.asarray(axes_b)      # body-frame axes
    kept_pairs = list(kept_idx)

    def kabsch(A, B):
        # R_CB = argmin sum ||R n_B - n_C||^2
        return project_to_so3(A.T @ B).T          # returns R_BC

    R_BC = kabsch(kept_c, kept_b)

    # Second stage: reject on the *full rotation* residual, then re-solve.
    #
    # The angle screen above cannot see a blunder that happens to preserve the
    # rotation angle -- a corrupted axis with the right magnitude sails through
    # it, and a single such pair biases the closed-form fit by degrees. The
    # residual of the conjugation identity does see it, so one robust re-fit
    # (median + MAD, the non-parametric analogue of a 3-sigma cut) removes
    # what the cheap screen missed.
    for _ in range(3):
        res_all = np.array([
            np.linalg.norm(Log((R_BC.T @ np.asarray(R_body_rel[i], float) @ R_BC).T
                               @ np.asarray(R_cam_rel[i], float)))
            for i in kept_pairs
        ])
        med = float(np.median(res_all))
        mad = float(np.median(np.abs(res_all - med))) * 1.4826
        thresh = med + 3.0 * max(mad, np.deg2rad(0.2))
        inl = res_all <= thresh
        if inl.all() or np.count_nonzero(inl) < 3:
            break
        n_rejected += int(np.count_nonzero(~inl))
        kept_c = kept_c[inl]
        kept_b = kept_b[inl]
        kept_pairs = [k for k, ok in zip(kept_pairs, inl) if ok]
        R_BC = kabsch(kept_c, kept_b)

    res = [
        np.linalg.norm(Log((R_BC.T @ np.asarray(R_body_rel[i], float) @ R_BC).T
                           @ np.asarray(R_cam_rel[i], float)))
        for i in kept_pairs
    ]
    return HandEyeResult(
        R_BC=R_BC,
        residual_deg=float(np.degrees(np.median(res))),
        n_used=len(kept_pairs),
        n_rejected=n_rejected,
        angle_mismatch_deg=float(np.degrees(np.median(mismatches))) if mismatches else 0.0,
    )


def rotation_excitation(R_rel) -> float:
    """Smallest singular value of the stacked rotation axes: an observability score.

    Near zero means every increment shared one axis, so ``R_BC`` is
    undetermined about it and the hand-eye result is meaningless in that
    direction no matter how small the reported residual is. Report it next to
    the calibration, always.
    """
    V = np.asarray([Log(np.asarray(R, float)) for R in R_rel])
    if len(V) < 3:
        return 0.0
    n = np.linalg.norm(V, axis=1)
    keep = n > 1e-6
    if np.count_nonzero(keep) < 3:
        return 0.0
    U = V[keep] / n[keep, None]
    return float(np.linalg.svd(U, compute_uv=False)[-1] / np.sqrt(len(U)))


# --------------------------------------------------------------------------- #
# 3. Camera-IMU time offset
# --------------------------------------------------------------------------- #

@dataclass
class TimeOffsetResult:
    offset_s: float
    correlation: float
    n_samples: int
    search_range_s: float


def estimate_time_offset(t_a: np.ndarray, s_a: np.ndarray,
                         t_b: np.ndarray, s_b: np.ndarray,
                         max_offset_s: float = 0.6,
                         resample_hz: float = 50.0) -> TimeOffsetResult:
    """Estimate ``t_d`` such that ``s_a(t) ~ s_b(t + t_d)``, by cross-correlation.

    Feed it two scalar signals that both track the same physical quantity --
    in practice ``|omega|`` from the visual front-end and ``|omega|`` from the
    attitude surrogate. Rate magnitude is the right choice because it is
    frame-independent: it does not require ``R_BC`` to be known yet, so time
    offset and extrinsic can be solved in either order.

    The peak is refined to sub-sample resolution by fitting a parabola to the
    three correlation values around the maximum -- without that the answer is
    quantised to ``1/resample_hz`` (20 ms at the default), which is the same
    order as the effect being measured.

    **Sign convention.** The result is ``d`` in the model ``t_b = t_true + d``:
    stream ``b``'s timestamps are ``d`` seconds *later* than the events they
    describe, relative to ``a``. To align the streams, **subtract** ``offset_s``
    from ``b``'s timestamps (equivalently, add it to ``a``'s). A positive
    ``offset_s`` therefore means ``b`` lags ``a`` -- which is what you expect
    for the Tello's video stream measured against its telemetry.
    """
    t_a = np.asarray(t_a, float).ravel()
    t_b = np.asarray(t_b, float).ravel()
    s_a = np.asarray(s_a, float).ravel()
    s_b = np.asarray(s_b, float).ravel()
    if len(t_a) < 8 or len(t_b) < 8:
        raise ValueError("need at least 8 samples in each stream")

    t0 = max(t_a[0], t_b[0]) + max_offset_s
    t1 = min(t_a[-1], t_b[-1]) - max_offset_s
    if t1 <= t0:
        raise ValueError("streams overlap for less than the search window")

    grid = np.arange(t0, t1, 1.0 / resample_hz)
    A = np.interp(grid, t_a, s_a)
    B = np.interp(grid, t_b, s_b)
    # Remove the mean so a DC offset cannot dominate the correlation, and
    # normalise so the peak value is an interpretable correlation coefficient.
    A = A - A.mean()
    B = B - B.mean()
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    if na < 1e-12 or nb < 1e-12:
        raise ValueError("a signal is constant; excite the drone in rotation")
    A, B = A / na, B / nb

    max_lag = int(round(max_offset_s * resample_hz))
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([
        float(np.dot(np.roll(B, int(L)), A)) for L in lags
    ])
    k = int(np.argmax(corr))
    lag = float(lags[k])

    # Parabolic sub-sample refinement around the discrete peak.
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-15:
            lag += 0.5 * (y0 - y2) / denom

    # The correlation peak sits at lag = -d*fs (rolling B forward by L compares
    # A(t) against B(t - L/fs - d)), so negate to report d itself.
    return TimeOffsetResult(
        offset_s=float(-lag / resample_hz),
        correlation=float(corr[k]),
        n_samples=len(grid),
        search_range_s=max_offset_s,
    )
