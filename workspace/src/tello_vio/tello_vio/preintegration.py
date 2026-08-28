"""On-manifold IMU preintegration (Forster et al., T-RO 2017).

Preintegration answers one question: *given a burst of IMU samples between two
keyframes, what single constraint do they impose on the two keyframe states?*
Naively you would re-integrate the whole burst every time the optimiser touches
either keyframe -- ruinously expensive inside a Gauss-Newton loop. Preintegration
instead compresses the burst once into a rotation/velocity/position increment
expressed **in the frame of the first keyframe**, so it stays valid as the
optimiser moves both poses around. The only thing it cannot absorb is a change
of *bias*, which is handled by a first-order correction with the analytic
Jacobians carried alongside.

Definitions (world frame is ENU, gravity ``g = [0, 0, -9.80665]``)::

    dR_ij = prod_k Exp((w_k - b_g) dt)
    dv_ij = sum_k dR_ik (a_k - b_a) dt
    dp_ij = sum_k [ dv_ik dt + 1/2 dR_ik (a_k - b_a) dt^2 ]

relating the two states by::

    R_j = R_i dR_ij
    v_j = v_i + g dt_ij + R_i dv_ij
    p_j = p_i + v_i dt_ij + 1/2 g dt_ij^2 + R_i dp_ij

where ``a_k`` is *specific force* in the body frame (a level, stationary IMU
reads ``[0, 0, +9.81]``, not zero) and ``w_k`` is body angular rate.

Tello caveat
------------
On a stock Tello the ``w_k`` fed in here is the finite-differenced attitude
surrogate from :class:`tello_vio.tello_model.TelloImuSurrogate`, sampled at
~10 Hz rather than a real 200 Hz gyro. Preintegration remains mathematically
correct -- it makes no assumption about rate -- but the *information* it carries
is proportionally weaker, which is why the estimator leans on the body-velocity
and height measurements for anything long-horizon. The class is written to be
rate-agnostic so the same code runs unchanged if you bolt a real IMU onto the
airframe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lie import Exp, Log, right_jacobian, skew, I3

#: Gravity vector in the ENU world frame (points down, i.e. -z).
GRAVITY_ENU = np.array([0.0, 0.0, -9.80665])


@dataclass
class ImuNoiseModel:
    """Continuous-time IMU noise densities.

    ``*_noise_density`` are the white-noise densities (per sqrt(Hz)) and
    ``*_bias_rw`` the bias random-walk densities. Converting to discrete time
    is done inside :meth:`PreintegratedImu.integrate` and depends on the actual
    ``dt`` of each sample -- which matters here, because Tello telemetry does
    not arrive on a fixed period.
    """

    gyro_noise_density: float = 0.03      # rad/s/sqrt(Hz)
    accel_noise_density: float = 0.08     # m/s^2/sqrt(Hz)
    gyro_bias_rw: float = 1.0e-4          # rad/s^2/sqrt(Hz)
    accel_bias_rw: float = 1.0e-3         # m/s^3/sqrt(Hz)


class PreintegratedImu:
    """Accumulates IMU samples into a single relative-motion constraint.

    The 9x9 covariance is ordered ``[dtheta, dv, dp]`` (Forster's ordering);
    the bias Jacobians let the constraint be re-linearised for a new bias
    estimate without replaying the samples, via :meth:`bias_corrected_delta`.
    """

    __slots__ = (
        "noise", "bias_acc", "bias_gyro", "dt", "dR", "dv", "dp", "cov",
        "dR_dbg", "dv_dba", "dv_dbg", "dp_dba", "dp_dbg", "n",
    )

    def __init__(self, noise: ImuNoiseModel, bias_acc: np.ndarray, bias_gyro: np.ndarray):
        self.noise = noise
        #: Bias linearisation point. The stored deltas are only valid for this
        #: bias; :meth:`bias_corrected_delta` extrapolates to nearby values.
        self.bias_acc = np.asarray(bias_acc, dtype=np.float64).reshape(3).copy()
        self.bias_gyro = np.asarray(bias_gyro, dtype=np.float64).reshape(3).copy()
        self.reset()

    def reset(self) -> None:
        self.dt = 0.0
        self.dR = np.eye(3)
        self.dv = np.zeros(3)
        self.dp = np.zeros(3)
        self.cov = np.zeros((9, 9))
        self.dR_dbg = np.zeros((3, 3))
        self.dv_dba = np.zeros((3, 3))
        self.dv_dbg = np.zeros((3, 3))
        self.dp_dba = np.zeros((3, 3))
        self.dp_dbg = np.zeros((3, 3))
        self.n = 0

    # ------------------------------------------------------------------ #

    def integrate(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """Fold one IMU sample (specific force, angular rate, interval) in.

        Uses the midpoint-free "forward" form of Forster's recursion, which is
        what the analytic Jacobians below are derived for. Mixing a midpoint
        state propagation with forward Jacobians is a subtle and popular bug:
        the filter stays stable but the covariance is wrong, so consistency
        checks (NEES) silently fail.
        """
        dt = float(dt)
        if dt <= 0.0:
            return

        a = np.asarray(acc, dtype=np.float64).reshape(3) - self.bias_acc
        w = np.asarray(gyro, dtype=np.float64).reshape(3) - self.bias_gyro

        phi = w * dt
        dR_k = Exp(phi)
        Jr = right_jacobian(phi)
        a_skew = skew(a)
        Ra = self.dR @ a

        # ---- covariance and bias-Jacobian propagation --------------------
        # Both must be evaluated with the *pre-update* dR, so they come first.
        A = np.zeros((9, 9))
        A[0:3, 0:3] = dR_k.T
        A[3:6, 0:3] = -self.dR @ a_skew * dt
        A[3:6, 3:6] = I3
        A[6:9, 0:3] = -0.5 * self.dR @ a_skew * dt * dt
        A[6:9, 3:6] = I3 * dt
        A[6:9, 6:9] = I3

        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = self.dR * dt
        B[6:9, 3:6] = 0.5 * self.dR * dt * dt

        # Discrete white-noise covariance: a continuous density sigma acting
        # over dt has discrete variance sigma^2 / dt once the dt inside B is
        # accounted for (B @ eta then carries std sigma * sqrt(dt)).
        nz = self.noise
        Q = np.diag(
            np.concatenate(
                [
                    np.full(3, nz.gyro_noise_density ** 2 / dt),
                    np.full(3, nz.accel_noise_density ** 2 / dt),
                ]
            )
        )
        self.cov = A @ self.cov @ A.T + B @ Q @ B.T
        self.cov = 0.5 * (self.cov + self.cov.T)  # kill asymmetry drift

        # Position Jacobians consume the *previous* velocity Jacobians, so they
        # are updated before the velocity ones (Forster, appendix B).
        self.dp_dba = self.dp_dba + self.dv_dba * dt - 0.5 * self.dR * dt * dt
        self.dp_dbg = (
            self.dp_dbg + self.dv_dbg * dt
            - 0.5 * self.dR @ a_skew @ self.dR_dbg * dt * dt
        )
        self.dv_dba = self.dv_dba - self.dR * dt
        self.dv_dbg = self.dv_dbg - self.dR @ a_skew @ self.dR_dbg * dt
        self.dR_dbg = dR_k.T @ self.dR_dbg - Jr * dt

        # ---- state increment --------------------------------------------
        self.dp = self.dp + self.dv * dt + 0.5 * Ra * dt * dt
        self.dv = self.dv + Ra * dt
        self.dR = self.dR @ dR_k

        self.dt += dt
        self.n += 1

    # ------------------------------------------------------------------ #

    def bias_corrected_delta(self, bias_acc: np.ndarray, bias_gyro: np.ndarray):
        """First-order re-linearisation of the increment for a new bias.

        Valid while the bias change is small relative to the curvature of the
        integral; the smoother repropagates from scratch when
        ``|db| `` exceeds a threshold (see :mod:`tello_vio.smoother`).
        """
        dba = np.asarray(bias_acc, dtype=np.float64).reshape(3) - self.bias_acc
        dbg = np.asarray(bias_gyro, dtype=np.float64).reshape(3) - self.bias_gyro
        dR = self.dR @ Exp(self.dR_dbg @ dbg)
        dv = self.dv + self.dv_dba @ dba + self.dv_dbg @ dbg
        dp = self.dp + self.dp_dba @ dba + self.dp_dbg @ dbg
        return dR, dv, dp, dbg

    def predict(self, R_i, v_i, p_i, bias_acc=None, bias_gyro=None,
                gravity: np.ndarray = GRAVITY_ENU):
        """Propagate state ``i`` forward through this increment to state ``j``."""
        if bias_acc is None:
            bias_acc = self.bias_acc
        if bias_gyro is None:
            bias_gyro = self.bias_gyro
        dR, dv, dp, _ = self.bias_corrected_delta(bias_acc, bias_gyro)
        dt = self.dt
        R_j = R_i @ dR
        v_j = v_i + gravity * dt + R_i @ dv
        p_j = p_i + v_i * dt + 0.5 * gravity * dt * dt + R_i @ dp
        return R_j, v_j, p_j

    def residual(self, R_i, v_i, p_i, R_j, v_j, p_j,
                 bias_acc=None, bias_gyro=None, gravity: np.ndarray = GRAVITY_ENU):
        """9-vector residual ``[r_R, r_v, r_p]`` for a factor-graph edge."""
        if bias_acc is None:
            bias_acc = self.bias_acc
        if bias_gyro is None:
            bias_gyro = self.bias_gyro
        dR, dv, dp, _ = self.bias_corrected_delta(bias_acc, bias_gyro)
        dt = self.dt
        r_R = Log(dR.T @ R_i.T @ R_j)
        r_v = R_i.T @ (v_j - v_i - gravity * dt) - dv
        r_p = R_i.T @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt * dt) - dp
        return np.concatenate([r_R, r_v, r_p])

    def jacobians(self, R_i, v_i, p_i, R_j, v_j, p_j,
                  bias_acc=None, bias_gyro=None, gravity: np.ndarray = GRAVITY_ENU):
        """Analytic Jacobians of :meth:`residual` w.r.t. both states and the bias.

        Returns ``(J_i, J_j, J_b)`` with column blocks
        ``J_i, J_j = [dtheta | dv | dp]`` (9x9) and ``J_b = [db_a | db_g]``
        (9x6), all in the right-multiplicative body-error convention declared in
        :mod:`tello_vio.lie`.
        """
        if bias_acc is None:
            bias_acc = self.bias_acc
        if bias_gyro is None:
            bias_gyro = self.bias_gyro
        dR, dv, dp, dbg = self.bias_corrected_delta(bias_acc, bias_gyro)
        dt = self.dt
        RiT = R_i.T

        r_R = Log(dR.T @ RiT @ R_j)
        JrInv_rR = np.linalg.inv(right_jacobian(r_R))

        J_i = np.zeros((9, 9))
        J_j = np.zeros((9, 9))
        J_b = np.zeros((9, 6))

        # --- rotation residual ---
        J_i[0:3, 0:3] = -JrInv_rR @ R_j.T @ R_i
        J_j[0:3, 0:3] = JrInv_rR
        J_b[0:3, 3:6] = -JrInv_rR @ Exp(r_R).T @ right_jacobian(self.dR_dbg @ dbg) @ self.dR_dbg

        # --- velocity residual ---
        J_i[3:6, 0:3] = skew(RiT @ (v_j - v_i - gravity * dt))
        J_i[3:6, 3:6] = -RiT
        J_j[3:6, 3:6] = RiT
        J_b[3:6, 0:3] = -self.dv_dba
        J_b[3:6, 3:6] = -self.dv_dbg

        # --- position residual ---
        J_i[6:9, 0:3] = skew(RiT @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt * dt))
        J_i[6:9, 3:6] = -RiT * dt
        J_i[6:9, 6:9] = -RiT
        J_j[6:9, 6:9] = RiT
        J_b[6:9, 0:3] = -self.dp_dba
        J_b[6:9, 3:6] = -self.dp_dbg

        return J_i, J_j, J_b

    #: Largest information any single preintegrated edge is allowed to assert,
    #: per unit. See :meth:`information` for why a cap is necessary.
    MAX_INFORMATION = 1.0e6

    def information(self, floor: float = 1e-9) -> np.ndarray:
        """Information matrix (inverse covariance), regularised *and capped*.

        Two degeneracies have to be handled, and a naive ``inv(cov)`` gets both
        wrong in the same dangerous direction -- by asserting near-infinite
        confidence:

        * **Empty increment** (``n == 0``): the covariance is exactly zero, so
          the inverse is unbounded. An edge like that would weld two keyframes
          rigidly together and dominate every other factor in the graph.
        * **Single sample** (``n == 1``): less obvious, but the covariance is
          only rank 6, not 9. The noise mapping ``B`` makes the position rows a
          fixed multiple of the velocity rows (``B[6:9] = dt/2 * B[3:6]``), so
          three directions carry no uncertainty at all and invert to infinity
          even though the matrix looks harmlessly non-zero.

        Adding a floor fixes the numerics but converts "no information" into
        ``1/floor`` of information, which is the same failure wearing a hat.
        The eigenvalue cap is what actually makes it safe.
        """
        if self.n == 0 or self.dt <= 0.0:
            return np.zeros((9, 9))
        C = 0.5 * (self.cov + self.cov.T) + np.eye(9) * floor
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1.0 / self.MAX_INFORMATION)
        return V @ np.diag(1.0 / w) @ V.T

    def bias_random_walk_cov(self) -> np.ndarray:
        """6x6 covariance of the bias change across this interval."""
        nz = self.noise
        return np.diag(
            np.concatenate(
                [
                    np.full(3, nz.accel_bias_rw ** 2 * self.dt),
                    np.full(3, nz.gyro_bias_rw ** 2 * self.dt),
                ]
            )
        )
