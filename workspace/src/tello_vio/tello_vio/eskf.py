"""Error-state Kalman filter for Tello visual-inertial odometry.

Why an *error-state* filter and not a plain EKF on the quaternion: attitude
lives on SO(3), which has no global 3-parameter chart. A plain EKF either
carries a 4-vector quaternion with a singular covariance (and needs constant
re-normalisation hacks) or uses Euler angles (and dies at gimbal lock). The
error-state formulation keeps the *nominal* state on the manifold, where it is
free to be large, and keeps a *small* 3-vector attitude error in the tangent
space, where a Gaussian is actually a reasonable model. That is the whole
trick, and it is why every serious VIO filter (MSCKF, OpenVINS, ROVIO, PX4's
EKF2) is built this way.

State layout
------------
Nominal state (on the manifold)::

    p     position of body in world      (ENU, m)
    v     velocity of body in world      (ENU, m/s)
    q     attitude q_WB                  (Hamilton [w,x,y,z])
    b_a   accelerometer bias             (body, m/s^2)
    b_g   gyro bias                      (body, rad/s)
    b_p   barometer bias                 (m)
    p_c   *cloned* position at the last visual keyframe
    q_c   *cloned* attitude at the last visual keyframe

Error state (22-dimensional, the thing the covariance describes)::

    idx  0: 3   dp
    idx  3: 6   dv
    idx  6: 9   dtheta      (body-frame, right-multiplicative: q = q_hat (x) Exp(dtheta))
    idx  9:12   db_a
    idx 12:15   db_g
    idx 15      db_p
    idx 16:19   dp_c
    idx 19:22   dtheta_c

Why the clone
-------------
Monocular vision naturally produces *relative* measurements: "between keyframe
A and now, the camera rotated by this much and moved in this direction".
Feeding a relative measurement to a filter that only knows the current state is
statistically wrong -- the measurement is correlated with the past state error,
and ignoring that correlation makes the filter over-confident and eventually
divergent. **Stochastic cloning** (Roumeliotis & Burdick, 2002) fixes it by
carrying a copy of the state at the reference time inside the same covariance,
so the cross-correlation is tracked explicitly. The clone costs 6 extra states;
a 22x22 filter update is microseconds of NumPy.

Where metric scale comes from
-----------------------------
Deliberately, the visual update below constrains **rotation and translation
*direction* only** -- never translation magnitude. A monocular camera cannot
observe scale, and pretending otherwise is how mono-VIO pipelines acquire a
slowly wrong scale that poisons everything downstream. Magnitude enters the
filter from signals that *are* metric: the Tello's onboard optical-flow body
velocity, the barometer, the downward ToF, and gravity-referenced accelerometer
integration. This split is what makes the estimator metric without needing the
100-200 Hz gyro the drone does not have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lie
from .lie import Exp, I3, Log, right_jacobian, right_jacobian_inv, skew
from .preintegration import GRAVITY_ENU

# Error-state slices, named once so index typos cannot happen silently.
P = slice(0, 3)
V = slice(3, 6)
TH = slice(6, 9)
BA = slice(9, 12)
BG = slice(12, 15)
BP = slice(15, 16)
PC = slice(16, 19)
THC = slice(19, 22)
DIM = 22

#: 95th-percentile chi-square quantiles, indexed by degrees of freedom.
#: Used to reject measurements whose normalised innovation squared is
#: implausible -- the single most effective guard against a bad visual match or
#: a ToF reading off a table edge wrecking the state.
_CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592}


@dataclass
class EskfConfig:
    """Tuning knobs. Densities are continuous-time (per sqrt(Hz))."""

    accel_noise_density: float = 0.08
    gyro_noise_density: float = 0.03
    accel_bias_rw: float = 1.0e-3
    gyro_bias_rw: float = 1.0e-4
    baro_bias_rw: float = 0.01

    #: Initial 1-sigma uncertainties.
    init_pos_std: float = 0.05
    init_vel_std: float = 0.10
    init_att_rp_std: float = 3.0 * np.pi / 180.0
    init_att_yaw_std: float = 30.0 * np.pi / 180.0
    init_ba_std: float = 0.30
    init_bg_std: float = 0.05
    init_bp_std: float = 5.0

    #: Bias magnitudes are physically bounded; clamping stops a temporarily
    #: unobservable direction from ratcheting the bias off to nonsense during,
    #: say, a long stationary-yaw segment.
    max_accel_bias: float = 1.5
    max_gyro_bias: float = 0.30

    #: Largest propagation step accepted. Telemetry gaps longer than this are
    #: treated as dropouts (covariance inflated) rather than integrated across.
    max_dt: float = 0.5

    #: Enable the innovation gate. Leave on except when debugging.
    gating: bool = True
    #: Below this predicted baseline (m) the visual translation direction is
    #: numerically meaningless, so only the rotation part of the visual
    #: measurement is applied. Pure-rotation degeneracy handling.
    min_baseline_m: float = 0.02

    gravity: tuple = tuple(GRAVITY_ENU)


class ErrorStateKF:
    """22-state error-state KF with stochastic cloning.

    Typical call order per cycle::

        kf.propagate(accel, gyro, dt)          # 10-30 Hz, from telemetry
        kf.update_attitude(q_meas, ...)        # Tello AHRS
        kf.update_body_velocity(v_meas, ...)   # Tello optical flow
        kf.update_barometer(baro, ...)
        kf.update_tof(range, ...)              # gated on tilt + range window
        kf.update_visual_relative(...)         # on each VO keyframe
        kf.clone()                             # right after a visual keyframe
    """

    def __init__(self, config: EskfConfig | None = None):
        self.cfg = config or EskfConfig()
        self.g = np.asarray(self.cfg.gravity, dtype=np.float64).reshape(3)

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = lie.quat_identity()
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.bp = 0.0
        self.p_c = np.zeros(3)
        self.q_c = lie.quat_identity()

        self.P = np.zeros((DIM, DIM))
        self.initialised = False
        self.t = 0.0

        #: Diagnostics: counts of accepted/rejected updates by kind.
        self.stats: dict[str, list] = {}

    # ------------------------------------------------------------------ #
    # initialisation
    # ------------------------------------------------------------------ #

    def initialise(self, t: float, q: np.ndarray, baro: float,
                   p: np.ndarray | None = None) -> None:
        """Anchor the filter at the current attitude and pressure altitude.

        The world frame is *defined* here: origin at the current position,
        z along local gravity-up. The barometer bias is initialised to the
        current reading so that ``p_z`` starts at zero -- without this anchor
        ``p_z`` and ``b_p`` are jointly unobservable and the pair would drift
        together forever while the barometer residual stayed happily at zero.
        """
        self.t = float(t)
        self.p = np.zeros(3) if p is None else np.asarray(p, float).reshape(3).copy()
        self.v = np.zeros(3)
        self.q = lie.quat_normalize(q)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.bp = float(baro)
        self.p_c = self.p.copy()
        self.q_c = self.q.copy()

        c = self.cfg
        P0 = np.zeros((DIM, DIM))
        P0[P, P] = np.eye(3) * c.init_pos_std ** 2
        P0[V, V] = np.eye(3) * c.init_vel_std ** 2
        # Attitude uncertainty is anisotropic: roll/pitch are pinned by gravity,
        # yaw is not observable at all from an accelerometer.
        att_world = np.diag([c.init_att_rp_std ** 2, c.init_att_rp_std ** 2,
                             c.init_att_yaw_std ** 2])
        R0 = lie.quat_to_rot(self.q)
        P0[TH, TH] = R0.T @ att_world @ R0
        P0[BA, BA] = np.eye(3) * c.init_ba_std ** 2
        P0[BG, BG] = np.eye(3) * c.init_bg_std ** 2
        P0[BP, BP] = np.eye(1) * c.init_bp_std ** 2
        P0[PC, PC] = P0[P, P]
        P0[THC, THC] = P0[TH, TH]
        self.P = P0
        self.initialised = True

    # ------------------------------------------------------------------ #
    # propagation
    # ------------------------------------------------------------------ #

    @property
    def R(self) -> np.ndarray:
        return lie.quat_to_rot(self.q)

    def propagate(self, accel: np.ndarray, gyro: np.ndarray | None, dt: float) -> None:
        """Advance the state and covariance by one IMU sample.

        ``gyro`` may be ``None`` when the attitude surrogate could not produce a
        rate this cycle (dropout, duplicate packet). In that case the rotation
        is held and only the *uncertainty* grows -- which is honest. Feeding a
        zero rate instead would assert "the drone did not rotate", a confident
        lie that corrupts the attitude covariance and, through it, everything.
        """
        dt = float(dt)
        if dt <= 0.0:
            return
        if dt > self.cfg.max_dt:
            # A dropout: do not integrate across it, just widen the covariance
            # by the amount of process noise the gap would have added.
            self._inflate(dt)
            self.t += dt
            return

        a_m = np.asarray(accel, dtype=np.float64).reshape(3)
        have_gyro = gyro is not None
        w_m = np.asarray(gyro, dtype=np.float64).reshape(3) if have_gyro else np.zeros(3)

        R = self.R
        a = a_m - self.ba
        w = (w_m - self.bg) if have_gyro else np.zeros(3)
        a_world = R @ a + self.g

        # ---- nominal state ------------------------------------------------
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        if have_gyro:
            self.q = lie.quat_boxplus(self.q, w * dt)

        # ---- error-state transition ---------------------------------------
        F = np.eye(DIM)
        a_sk = skew(a)
        F[P, V] = I3 * dt
        F[P, TH] = -0.5 * R @ a_sk * dt * dt
        F[P, BA] = -0.5 * R * dt * dt
        F[V, TH] = -R @ a_sk * dt
        F[V, BA] = -R * dt
        if have_gyro:
            phi = w * dt
            F[TH, TH] = Exp(phi).T
            F[TH, BG] = -right_jacobian(phi) * dt
        else:
            F[TH, BG] = -I3 * dt

        # ---- process noise -------------------------------------------------
        c = self.cfg
        sa2 = c.accel_noise_density ** 2
        sg2 = c.gyro_noise_density ** 2
        Q = np.zeros((DIM, DIM))
        # Velocity/position blocks come from integrating accel white noise;
        # keeping the p-v cross term matters at the dt values we run at.
        Q[V, V] = np.eye(3) * (sa2 * dt)
        Q[P, P] = np.eye(3) * (sa2 * dt ** 3 / 3.0)
        Q[P, V] = np.eye(3) * (sa2 * dt ** 2 / 2.0)
        Q[V, P] = Q[P, V]
        Q[TH, TH] = np.eye(3) * (sg2 * dt)
        if not have_gyro:
            # No rate this cycle: the attitude could have changed by anything
            # up to the airframe's slew rate. 3 rad/s covers a Tello flip.
            Q[TH, TH] += np.eye(3) * (3.0 * dt) ** 2
        Q[BA, BA] = np.eye(3) * (c.accel_bias_rw ** 2 * dt)
        Q[BG, BG] = np.eye(3) * (c.gyro_bias_rw ** 2 * dt)
        Q[BP, BP] = np.eye(1) * (c.baro_bias_rw ** 2 * dt)

        self.P = F @ self.P @ F.T + Q
        self._symmetrise()
        self.t += dt

    def _inflate(self, dt: float) -> None:
        c = self.cfg
        self.P[V, V] += np.eye(3) * (c.accel_noise_density ** 2 * dt)
        self.P[P, P] += np.eye(3) * (c.accel_noise_density ** 2 * dt ** 3 / 3.0)
        self.P[TH, TH] += np.eye(3) * (c.gyro_noise_density ** 2 * dt + (1.0 * dt) ** 2)
        self.P[BA, BA] += np.eye(3) * (c.accel_bias_rw ** 2 * dt)
        self.P[BG, BG] += np.eye(3) * (c.gyro_bias_rw ** 2 * dt)
        self.P[BP, BP] += np.eye(1) * (c.baro_bias_rw ** 2 * dt)
        self._symmetrise()

    # ------------------------------------------------------------------ #
    # generic update machinery
    # ------------------------------------------------------------------ #

    def _symmetrise(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)

    def _record(self, kind: str, accepted: bool, nis: float) -> None:
        s = self.stats.setdefault(kind, [0, 0, 0.0])
        s[0 if accepted else 1] += 1
        s[2] = nis

    def _update(self, H: np.ndarray, r: np.ndarray, R_meas: np.ndarray,
                kind: str, gate_dof: int | None = None) -> bool:
        """Joseph-form EKF update with chi-square innovation gating.

        **Convention, and it is load-bearing:** ``r`` is the innovation
        ``z - h(x_hat)`` and ``H`` is ``dh/dx`` -- the Jacobian of the
        *prediction*, not of the residual. The two differ by a sign, and
        getting it backwards produces a filter that corrects in the wrong
        direction: it stays numerically well-behaved, the covariance still
        shrinks, and the estimate walks steadily away from the truth. Each
        ``H`` below is checked against a finite difference of ``-r`` in
        ``test_eskf.py`` precisely because this failure mode is invisible by
        inspection.

        Joseph form (``(I-KH) P (I-KH)^T + K R K^T``) rather than the shorter
        ``(I-KH) P``: it stays symmetric positive-definite under floating-point
        round-off, which the short form does not once you run for minutes at
        30 Hz with badly scaled states (metres next to milliradians next to
        bias terms). The cost is one extra matrix product on a 22x22 -- free.
        """
        H = np.atleast_2d(H)
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        S = H @ self.P @ H.T + R_meas
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self._record(kind, False, float("inf"))
            return False

        nis = float(r @ S_inv @ r)
        dof = gate_dof if gate_dof is not None else r.size
        if self.cfg.gating and dof in _CHI2_95 and nis > _CHI2_95[dof] * 3.0:
            # 3x the 95 % quantile: loose enough not to reject honest
            # measurements during fast manoeuvres, tight enough to catch a
            # mis-associated visual match or a ToF reading off a table edge.
            self._record(kind, False, nis)
            return False

        K = self.P @ H.T @ S_inv
        dx = K @ r

        IKH = np.eye(DIM) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T
        self._symmetrise()
        self._inject(dx)
        self._record(kind, True, nis)
        return True

    def _inject(self, dx: np.ndarray) -> None:
        """Fold the error estimate into the nominal state and reset it to zero.

        The covariance reset ``G`` accounts for the fact that after injecting
        ``dtheta`` the *meaning* of the remaining attitude error changes. It is
        a second-order effect that most implementations skip; it is two lines
        and it keeps the filter consistent during aggressive rotation.
        """
        self.p = self.p + dx[P]
        self.v = self.v + dx[V]
        self.q = lie.quat_boxplus(self.q, dx[TH])
        self.ba = np.clip(self.ba + dx[BA], -self.cfg.max_accel_bias, self.cfg.max_accel_bias)
        self.bg = np.clip(self.bg + dx[BG], -self.cfg.max_gyro_bias, self.cfg.max_gyro_bias)
        self.bp = float(self.bp + dx[BP][0])
        self.p_c = self.p_c + dx[PC]
        self.q_c = lie.quat_boxplus(self.q_c, dx[THC])

        G = np.eye(DIM)
        G[TH, TH] = I3 - 0.5 * skew(dx[TH])
        G[THC, THC] = I3 - 0.5 * skew(dx[THC])
        self.P = G @ self.P @ G.T
        self._symmetrise()

    # ------------------------------------------------------------------ #
    # measurement updates
    # ------------------------------------------------------------------ #

    def update_attitude(self, q_meas: np.ndarray, std_rp: float, std_yaw: float) -> bool:
        """Absolute attitude from the Tello's onboard AHRS.

        The residual is the body-frame rotation vector ``Log(q_hat^-1 (x) q_m)``.
        Its Jacobian w.r.t. the right-multiplicative error is
        ``-Jr(-r0)^-1`` -- note the sign and the *negated* argument; using
        ``Jr(r0)^-1`` instead is a plausible-looking error that shows up only as
        slightly wrong convergence during fast rotation.

        Roll/pitch are gravity-referenced and genuinely absolute. Yaw is not:
        the Tello has no usable magnetometer, so its yaw free-runs. The
        anisotropic ``R`` below is built in *world* axes and rotated into the
        body, so the large yaw variance stays attached to world-z regardless of
        how the drone is tilted.
        """
        H, r, Rm = self.attitude_hr(q_meas, std_rp, std_yaw)
        return self._update(H, r, Rm, "attitude", gate_dof=3)

    def attitude_hr(self, q_meas: np.ndarray, std_rp: float, std_yaw: float):
        """Return ``(H, r, R)`` for the attitude update. Exposed for testing."""
        # r0 is already the innovation: the measured attitude expressed in the
        # tangent space at the predicted one, i.e. exactly z - h.
        r0 = lie.quat_boxminus(lie.quat_normalize(q_meas), self.q)
        H = np.zeros((3, DIM))
        # d(residual)/d(dtheta) = -Jr(-r0)^-1, so dh/dx = +Jr(-r0)^-1.
        H[:, TH] = right_jacobian_inv(-r0)
        Rw = self.R
        Rm = Rw.T @ np.diag([std_rp ** 2, std_rp ** 2, std_yaw ** 2]) @ Rw
        return H, r0, Rm

    def update_body_velocity(self, v_body: np.ndarray, std: float) -> bool:
        """Metric body-frame velocity from the Tello's downward optical flow.

        This is the measurement that makes the whole thing metric. ``h(x) =
        R^T v``; differentiating the rotation gives the ``skew(R^T v)`` term,
        which is what couples a velocity residual into an attitude correction --
        the mechanism by which sustained horizontal flight observes roll/pitch
        bias.
        """
        Rw = self.R
        h = Rw.T @ self.v
        r = np.asarray(v_body, dtype=np.float64).reshape(3) - h
        H = np.zeros((3, DIM))
        H[:, V] = Rw.T
        H[:, TH] = skew(h)
        return self._update(H, r, np.eye(3) * std ** 2, "vel_body", gate_dof=3)

    def update_barometer(self, baro_m: float, std: float) -> bool:
        """Pressure altitude, as a *biased* measurement of world z."""
        h = self.p[2] + self.bp
        r = np.array([float(baro_m) - h])
        H = np.zeros((1, DIM))
        H[0, 2] = 1.0
        H[0, BP.start] = 1.0
        return self._update(H, r, np.array([[std ** 2]]), "baro", gate_dof=1)

    def update_tof(self, range_m: float, std: float, ground_z: float = 0.0,
                   min_cos_tilt: float = 0.866) -> bool:
        """Downward time-of-flight range, tilt-compensated.

        A downward ToF measures the *slant* range to whatever is under the
        drone. For a locally horizontal floor at ``ground_z`` the geometry is
        ``range = (p_z - ground_z) / cos(tilt)`` with
        ``cos(tilt) = e_z^T R e_z = R[2,2]``. Skipping that division is a
        common shortcut that injects a height error of
        ``h (1/cos - 1)`` -- 6 % at 20 degrees of bank, which is an ordinary
        Tello manoeuvre.

        Gated three ways, because a ToF over a table or a stairwell is not
        measuring what the model says: tilt must be modest, the reading must be
        inside the sensor's honest window, and the innovation must pass the
        chi-square test.
        """
        Rw = self.R
        c_tilt = float(Rw[2, 2])
        if c_tilt < min_cos_tilt:
            self._record("tof", False, float("nan"))
            return False
        dz = self.p[2] - float(ground_z)
        h = dz / c_tilt
        r = np.array([float(range_m) - h])
        H = np.zeros((1, DIM))
        H[0, 2] = 1.0 / c_tilt
        # d(1/R22)/dtheta via dR22/dtheta = (e_z x R^T e_z)^T
        dR22 = np.cross(np.array([0.0, 0.0, 1.0]), Rw.T @ np.array([0.0, 0.0, 1.0]))
        H[0, TH] = -dz / (c_tilt ** 2) * dR22
        return self._update(H, r, np.array([[std ** 2]]), "tof", gate_dof=1)

    def update_zero_velocity(self, std: float = 0.02) -> bool:
        """Zero-velocity update for when the airframe is provably still.

        The highest-information measurement available on this platform. While
        landed, ``v = 0`` is *exact*, which turns the accelerometer bias from
        weakly observable into strongly observable in seconds. Only ever call
        this behind :class:`~tello_vio.tello_model.StationarityDetector`.
        """
        H = np.zeros((3, DIM))
        H[:, V] = I3
        return self._update(H, -self.v, np.eye(3) * std ** 2, "zupt", gate_dof=3)

    def update_zero_angular_rate(self, gyro_meas: np.ndarray, std: float = 0.02) -> bool:
        """Zero-angular-rate update: while still, the measured rate *is* the bias."""
        H = np.zeros((3, DIM))
        H[:, BG] = I3
        r = np.asarray(gyro_meas, dtype=np.float64).reshape(3) - self.bg
        return self._update(H, r, np.eye(3) * std ** 2, "zaru", gate_dof=3)

    # ------------------------------------------------------------------ #
    # visual updates (stochastic cloning)
    # ------------------------------------------------------------------ #

    def clone(self) -> None:
        """Snapshot the current pose into the clone slot, correlations included.

        This is *not* a copy of the mean only. The clone must inherit the full
        covariance and cross-covariance of the state it was taken from,
        otherwise the subsequent relative measurement is treated as independent
        of the past -- which is precisely the error stochastic cloning exists to
        avoid.
        """
        self.p_c = self.p.copy()
        self.q_c = self.q.copy()
        J = np.eye(DIM)
        J[PC, :] = 0.0
        J[THC, :] = 0.0
        J[PC, P] = I3
        J[THC, TH] = I3
        self.P = J @ self.P @ J.T
        self._symmetrise()

    def update_visual_relative(self, R_c1c2: np.ndarray, t_dir_c1: np.ndarray,
                               R_BC: np.ndarray, p_BC: np.ndarray,
                               rot_std: float, dir_std: float,
                               use_translation: bool = True) -> bool:
        """Fuse one relative visual measurement against the clone.

        Parameters
        ----------
        R_c1c2, t_dir_c1:
            Camera-frame relative rotation and **unit** translation direction
            from the keyframe to now, as recovered from the essential matrix.
        R_BC, p_BC:
            Camera-to-body extrinsic (``p_B = R_BC p_C + p_BC``).
        use_translation:
            Set ``False`` for a rotation-only measurement -- correct whenever
            the two views have insufficient parallax, where the essential
            matrix's translation direction is pure noise.

        The translation residual lives in the 2-D tangent plane of the unit
        sphere at the measured bearing, because a bearing carries exactly 2
        degrees of freedom. Treating it as a 3-vector residual (the common
        shortcut) silently adds a phantom constraint along the radial direction
        and biases the scale the rest of the filter worked to establish.
        """
        H, r, Rm, kind, dof = self.visual_relative_hr(
            R_c1c2, t_dir_c1, R_BC, p_BC, rot_std, dir_std, use_translation)
        return self._update(H, r, Rm, kind, gate_dof=dof)

    def visual_relative_hr(self, R_c1c2, t_dir_c1, R_BC, p_BC,
                           rot_std, dir_std, use_translation: bool = True):
        """Return ``(H, r, R, kind, dof)`` for the visual update. For testing."""
        R_hat = self.R
        Rc_hat = lie.quat_to_rot(self.q_c)
        R_BC = np.asarray(R_BC, dtype=np.float64).reshape(3, 3)
        p_BC = np.asarray(p_BC, dtype=np.float64).reshape(3)

        # --- rotation part -------------------------------------------------
        M = R_BC @ np.asarray(R_c1c2, dtype=np.float64).reshape(3, 3) @ R_BC.T
        A_hat = Rc_hat.T @ R_hat                       # predicted body-relative rotation
        # Log(M^T A) is a (prediction - measurement) quantity, the opposite
        # sense to a Kalman innovation. Negate it so that r = z - h holds, and
        # take H as +d(h)/dx accordingly.
        e_rot = Log(M.T @ A_hat)
        JrInv = right_jacobian_inv(e_rot)
        r_rot = -e_rot
        H_rot = np.zeros((3, DIM))
        H_rot[:, TH] = JrInv
        H_rot[:, THC] = -JrInv @ R_hat.T @ Rc_hat

        w_hat = Rc_hat.T @ (self.p - self.p_c)         # body-1 frame baseline
        d_hat = R_BC.T @ (w_hat - p_BC + A_hat @ p_BC)  # camera-1 frame baseline
        baseline = float(np.linalg.norm(d_hat))

        if not use_translation or baseline < self.cfg.min_baseline_m:
            return H_rot, r_rot, np.eye(3) * rot_std ** 2, "visual_rot", 3

        # --- translation-direction part ------------------------------------
        u_hat = d_hat / baseline
        t_meas = np.asarray(t_dir_c1, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(t_meas))
        if n < 1e-9:
            return H_rot, r_rot, np.eye(3) * rot_std ** 2, "visual_rot", 3
        t_meas = t_meas / n

        B = _tangent_basis(t_meas)                     # 3x2
        r_dir = B.T @ (t_meas - u_hat)

        # d(d_hat)/d(error state)
        Dd = np.zeros((3, DIM))
        Dd[:, P] = R_BC.T @ Rc_hat.T
        Dd[:, PC] = -R_BC.T @ Rc_hat.T
        Dd[:, THC] = R_BC.T @ (skew(w_hat) + skew(A_hat @ p_BC))
        Dd[:, TH] = R_BC.T @ (-A_hat @ skew(p_BC))
        # projection onto the sphere: d(u)/d(d) = (I - u u^T)/|d|
        Du = (I3 - np.outer(u_hat, u_hat)) / baseline
        # r_dir = B^T (t_meas - u(x)) is already z - h, so H is +d(h)/dx.
        H_dir = B.T @ Du @ Dd

        H = np.vstack([H_rot, H_dir])
        r = np.concatenate([r_rot, r_dir])
        Rm = np.diag(np.concatenate([np.full(3, rot_std ** 2), np.full(2, dir_std ** 2)]))
        return H, r, Rm, "visual", 5

    # ------------------------------------------------------------------ #

    def pose_covariance_6x6(self) -> np.ndarray:
        """Position+orientation covariance in the ROS ``[x y z rx ry rz]`` order."""
        C = np.zeros((6, 6))
        C[0:3, 0:3] = self.P[P, P]
        C[3:6, 3:6] = self.P[TH, TH]
        C[0:3, 3:6] = self.P[P, TH]
        C[3:6, 0:3] = self.P[TH, P]
        return C

    def summary(self) -> dict:
        return {
            "t": self.t,
            "p": self.p.copy(),
            "v": self.v.copy(),
            "q": self.q.copy(),
            "ba": self.ba.copy(),
            "bg": self.bg.copy(),
            "bp": self.bp,
            "sigma_p": np.sqrt(np.diag(self.P[P, P])),
            "sigma_v": np.sqrt(np.diag(self.P[V, V])),
            "sigma_att_deg": np.degrees(np.sqrt(np.diag(self.P[TH, TH]))),
        }


def _tangent_basis(u: np.ndarray) -> np.ndarray:
    """Orthonormal 3x2 basis of the plane perpendicular to the unit vector ``u``."""
    u = np.asarray(u, dtype=np.float64).reshape(3)
    # Pick the world axis least aligned with u so the cross product is well
    # conditioned; using a fixed axis breaks when u happens to be parallel to it.
    a = np.zeros(3)
    a[int(np.argmin(np.abs(u)))] = 1.0
    b1 = np.cross(u, a)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(u, b1)
    return np.column_stack([b1, b2])
