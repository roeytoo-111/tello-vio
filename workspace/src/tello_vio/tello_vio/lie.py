"""Minimal, allocation-light Lie-group utilities for SO(3) / SE(3).

Conventions used *everywhere* in :mod:`tello_vio` (deviating from them is the
single most common source of silent sign errors in VIO code, so they are stated
once here and never re-derived):

* **Rotation matrices** ``R`` map *body* coordinates to *world* coordinates:
  ``p_W = R_WB @ p_B``.
* **Quaternions** are Hamilton quaternions stored ``[w, x, y, z]``.
  ROS ``geometry_msgs/Quaternion`` uses ``[x, y, z, w]``; conversion happens
  only at the ROS boundary (see :mod:`tello_vio.ros_utils`), never inside the
  estimator.
* **Error states are right-multiplicative and expressed in the body frame**::

      R = R_hat @ Exp(dtheta)          q = q_hat (x) Exp_q(dtheta)

  This is the choice made by Forster et al. (preintegration) and by GTSAM.
  It is what makes the IMU Jacobians below constant-ish and well conditioned.
  The alternative (left / world-frame error) yields *different* Jacobians --
  mixing the two is a classic bug.
* ``Exp`` is the SO(3) exponential map from a rotation vector (axis * angle,
  radians) to a rotation matrix; ``Log`` is its inverse on ``(-pi, pi]``.

All functions are pure NumPy, take/return ``float64``, and avoid SciPy so the
estimator can run inside a ROS 2 node without extra dependencies.
"""

from __future__ import annotations

import numpy as np

# Below this rotation angle (rad) the Taylor expansions are both more accurate
# and faster than the trigonometric closed forms. ~1e-4 rad is where the
# float64 cancellation in (1 - cos t) / t^2 starts to bite.
_EPS = 1e-8
_SMALL_ANGLE = 1e-4

I3 = np.eye(3)


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix ``[v]_x`` with ``[v]_x w = v x w``."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def unskew(S: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew` (assumes ``S`` is skew-symmetric)."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def Exp(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map: rotation vector -> rotation matrix (Rodrigues)."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta2 = float(phi @ phi)
    K = skew(phi)
    if theta2 < _SMALL_ANGLE * _SMALL_ANGLE:
        # R = I + K + K^2/2 + O(theta^3); exact to O(theta^4) which is far
        # below float64 noise for theta < 1e-4.
        return I3 + K + 0.5 * (K @ K)
    theta = np.sqrt(theta2)
    s = np.sin(theta) / theta
    c = (1.0 - np.cos(theta)) / theta2
    return I3 + s * K + c * (K @ K)


def Log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm: rotation matrix -> rotation vector in ``(-pi, pi]``.

    Uses the trace formula away from ``theta = pi`` and a numerically stable
    quaternion-based branch near ``pi`` where ``sin(theta) -> 0`` makes the
    trace formula blow up.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = 0.5 * (np.trace(R) - 1.0)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    theta = np.arccos(cos_theta)

    if theta < _SMALL_ANGLE:
        # sin(theta)/theta -> 1; the antisymmetric part *is* the rotation vector.
        return 0.5 * unskew(R - R.T)

    if theta < np.pi - 1e-5:
        return (0.5 * theta / np.sin(theta)) * unskew(R - R.T)

    # Near pi: R - R.T -> 0, so recover the axis from the symmetric part.
    # (R + I)/2 = I + K^2 * (1 - cos)/theta^2 ... simpler: use the largest
    # diagonal element of (R + I) to pick a well-conditioned column.
    A = R + I3
    k = int(np.argmax(np.diag(A)))
    axis = A[:, k].copy()
    n = np.linalg.norm(axis)
    if n < _EPS:  # R == -I is not a valid rotation reachable here, but be safe.
        return np.zeros(3)
    axis /= n
    # Sign is ambiguous at exactly pi; disambiguate with the antisymmetric part.
    v = unskew(R - R.T)
    if float(axis @ v) < 0.0:
        axis = -axis
    return axis * theta


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian ``Jr(phi)`` of SO(3).

    Defined by ``Exp(phi + dphi) ~= Exp(phi) Exp(Jr(phi) dphi)``.
    Appears in every IMU-preintegration and error-state covariance update.
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta2 = float(phi @ phi)
    K = skew(phi)
    if theta2 < _SMALL_ANGLE * _SMALL_ANGLE:
        return I3 - 0.5 * K + (1.0 / 6.0) * (K @ K)
    theta = np.sqrt(theta2)
    a = (1.0 - np.cos(theta)) / theta2
    b = (theta - np.sin(theta)) / (theta2 * theta)
    return I3 - a * K + b * (K @ K)


def right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian ``Jr^{-1}(phi)``."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta2 = float(phi @ phi)
    K = skew(phi)
    if theta2 < _SMALL_ANGLE * _SMALL_ANGLE:
        return I3 + 0.5 * K + (1.0 / 12.0) * (K @ K)
    theta = np.sqrt(theta2)
    # cot(theta/2) written as cos/sin to avoid a division by tan near pi/2.
    c = (1.0 / theta2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return I3 + 0.5 * K + c * (K @ K)


# --------------------------------------------------------------------------- #
# Quaternions (Hamilton, [w, x, y, z])
# --------------------------------------------------------------------------- #

def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < _EPS:
        return quat_identity()
    q = q / n
    # Keep the scalar part non-negative so that repeated small updates do not
    # wander onto the antipodal cover (matters when quaternions are logged,
    # differenced numerically, or compared in tests).
    return -q if q[0] < 0.0 else q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a (x) b`` (rotations compose as ``R(a) @ R(b)``)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion -> rotation matrix."""
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (Shepperd's branch-max method)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    return quat_normalize(q)


def quat_exp(phi: np.ndarray) -> np.ndarray:
    """Rotation vector -> unit quaternion (the SO(3) Exp lifted to S^3)."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    if theta < _SMALL_ANGLE:
        # sinc(theta/2)/2 -> 1/2 - theta^2/48
        k = 0.5 - theta * theta / 48.0
        return quat_normalize(np.array([1.0 - theta * theta / 8.0, k * phi[0], k * phi[1], k * phi[2]]))
    half = 0.5 * theta
    k = np.sin(half) / theta
    return np.array([np.cos(half), k * phi[0], k * phi[1], k * phi[2]])


def quat_log(q: np.ndarray) -> np.ndarray:
    """Unit quaternion -> rotation vector."""
    q = quat_normalize(q)
    v = q[1:]
    nv = float(np.linalg.norm(v))
    if nv < _EPS:
        return np.zeros(3)
    # q[0] >= 0 is guaranteed by quat_normalize, so theta in [0, pi].
    theta = 2.0 * np.arctan2(nv, q[0])
    return v * (theta / nv)


def quat_boxplus(q: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """Right-multiplicative retraction ``q [+] dtheta = q (x) Exp_q(dtheta)``."""
    return quat_normalize(quat_mul(q, quat_exp(dtheta)))


def quat_boxminus(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quat_boxplus`: ``q1 [-] q2 = Log(q2^-1 (x) q1)``."""
    return quat_log(quat_mul(quat_conj(q2), q1))


# --------------------------------------------------------------------------- #
# Euler angles
# --------------------------------------------------------------------------- #

def euler_zyx_to_rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Aerospace ``R = Rz(yaw) Ry(pitch) Rx(roll)`` (intrinsic Z-Y'-X'', radians)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rot_to_euler_zyx(R: np.ndarray) -> tuple:
    """Inverse of :func:`euler_zyx_to_rot`, returning ``(yaw, pitch, roll)``.

    Gimbal lock (``|pitch| -> pi/2``) is handled by collapsing roll into yaw
    rather than returning NaN.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sp = -R[2, 0]
    sp = min(1.0, max(-1.0, sp))
    pitch = np.arcsin(sp)
    if abs(sp) > 1.0 - 1e-9:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    return float(yaw), float(pitch), float(roll)


def euler_zyx_to_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return rot_to_quat(euler_zyx_to_rot(yaw, pitch, roll))


# --------------------------------------------------------------------------- #
# SE(3)
# --------------------------------------------------------------------------- #

def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (transpose + rotated translation)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix to ``M`` in Frobenius norm (SVD / Kabsch).

    Used to re-orthonormalise after long chains of floating-point products and
    to solve the rotation-only hand-eye problem in :mod:`tello_vio.calib`.
    """
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64).reshape(3, 3))
    d = np.linalg.det(U @ Vt)
    S = np.diag([1.0, 1.0, np.sign(d) if d != 0 else 1.0])
    return U @ S @ Vt


def so3_mean(rotations) -> np.ndarray:
    """Karcher (geodesic) mean of a sequence of rotation matrices."""
    rotations = list(rotations)
    if not rotations:
        return I3.copy()
    R = rotations[0].copy()
    for _ in range(20):
        delta = np.zeros(3)
        for Ri in rotations:
            delta += Log(R.T @ Ri)
        delta /= len(rotations)
        R = R @ Exp(delta)
        if float(delta @ delta) < 1e-24:
            break
    return project_to_so3(R)
