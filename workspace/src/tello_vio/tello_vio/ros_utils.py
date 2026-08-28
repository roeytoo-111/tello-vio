"""Conversions between ROS 2 messages and the estimator's internal types.

Kept in one place because every one of these is a chance to flip a convention:

* ROS ``geometry_msgs/Quaternion`` is ``(x, y, z, w)``; this package's maths is
  Hamilton ``[w, x, y, z]`` (see :mod:`tello_vio.lie`). The reordering happens
  here and nowhere else.
* ROS covariances are row-major flattened arrays with a documented ordering
  (``[x, y, z, rot_x, rot_y, rot_z]`` for pose); ``-1`` in element 0 means
  "this quantity is not measured", which is *not* the same as all-zeros
  ("measured, and exactly known"). Publishing zeros for something you do not
  measure is a lie a downstream filter will believe.
"""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3

from . import lie

#: Marker meaning "not available" in the first element of a ROS covariance.
COVARIANCE_UNKNOWN = -1.0


def quat_to_ros(q: np.ndarray) -> Quaternion:
    """Hamilton ``[w, x, y, z]`` -> ``geometry_msgs/Quaternion`` ``(x, y, z, w)``."""
    q = lie.quat_normalize(q)
    m = Quaternion()
    m.w, m.x, m.y, m.z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return m


def quat_from_ros(m) -> np.ndarray:
    """``geometry_msgs/Quaternion`` -> Hamilton ``[w, x, y, z]``."""
    return lie.quat_normalize(np.array([m.w, m.x, m.y, m.z], dtype=np.float64))


def vec3(v: np.ndarray) -> Vector3:
    m = Vector3()
    m.x, m.y, m.z = float(v[0]), float(v[1]), float(v[2])
    return m


def vec3_from(m) -> np.ndarray:
    return np.array([m.x, m.y, m.z], dtype=np.float64)


def stamp_to_sec(stamp: TimeMsg) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def sec_to_stamp(t: float) -> TimeMsg:
    m = TimeMsg()
    m.sec = int(np.floor(t))
    m.nanosec = int(round((t - m.sec) * 1e9))
    if m.nanosec >= 1_000_000_000:          # guard the rounding boundary
        m.sec += 1
        m.nanosec -= 1_000_000_000
    return m


def flatten_cov6(C: np.ndarray) -> list:
    """6x6 covariance -> the 36-element row-major array ROS expects."""
    return [float(x) for x in np.asarray(C, dtype=np.float64).reshape(36)]


def unknown_cov6() -> list:
    """A 36-element covariance flagged as unavailable, per ROS convention."""
    c = [0.0] * 36
    c[0] = COVARIANCE_UNKNOWN
    return c


def unknown_cov3() -> list:
    c = [0.0] * 9
    c[0] = COVARIANCE_UNKNOWN
    return c


def make_transform(parent: str, child: str, stamp, p: np.ndarray,
                   q: np.ndarray) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = float(p[0])
    t.transform.translation.y = float(p[1])
    t.transform.translation.z = float(p[2])
    t.transform.rotation = quat_to_ros(q)
    return t


def camera_from_body_rotation(tilt_deg: float = 0.0) -> np.ndarray:
    """Nominal ``R_BC`` for a forward-facing camera on an FLU body.

    REP-103 optical frame: ``z`` out of the lens, ``x`` right, ``y`` down.
    FLU body frame: ``x`` forward, ``y`` left, ``z`` up. Mapping one to the
    other is the fixed rotation ``Rz(-90 deg) Rx(-90 deg)``::

        camera +z (forward) -> body +x
        camera +x (right)   -> body -y
        camera +y (down)    -> body -z

    ``tilt_deg`` adds a nose-down pitch for a camera mounted looking downward.
    This is only a *starting guess*: the real mount has manufacturing tolerance
    of a few degrees, and a few degrees of extrinsic error looks exactly like
    scale drift in the output. Measure it with
    ``ros2 run tello_vio camera_imu_calib`` and put the answer in the config.
    """
    R = lie.euler_zyx_to_rot(-np.pi / 2, 0.0, -np.pi / 2)
    if tilt_deg:
        # Pitch about the body y axis, applied on the body side of the product.
        R = lie.euler_zyx_to_rot(0.0, np.deg2rad(tilt_deg), 0.0) @ R
    return R
