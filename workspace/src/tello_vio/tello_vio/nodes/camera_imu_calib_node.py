#!/usr/bin/env python3
"""Estimate the camera-IMU rotation and time offset from a hand-waved sequence.

Usage::

    ros2 run tello_vio camera_imu_calib --ros-args -p duration_s:=90.0

    # Then, with the drone powered on but NOT flying, pick it up and rotate it
    # smoothly about all three axes in front of a textured scene. Sharp jerks
    # produce motion blur and break feature tracking; slow sweeps of 30-60 deg
    # about each axis in turn, then some combined motion, is ideal.

Two independent quantities come out:

* ``extrinsic_rpy_deg`` -- the correction to the nominal ``R_BC``, from the
  rotation-only hand-eye identity ``R_C1C2 = R_CB R_B1B2 R_BC``.
* ``time_offset_s`` -- from cross-correlating the two streams' angular-rate
  magnitudes. Solved *first* and applied before the hand-eye fit, because a
  time offset makes the paired rotations disagree and would otherwise be
  absorbed into a wrong extrinsic.

Both are reported with an **observability score**. A small residual with a low
excitation score means the fit is confidently wrong: you rotated about one axis
and the extrinsic is undetermined about it.
"""

from __future__ import annotations

import numpy as np
import rclpy
import yaml
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu

from .. import lie
from ..calib import (estimate_camera_imu_rotation, estimate_time_offset,
                     rotation_excitation)
from ..frontend import FrontendConfig, VoFrontend
from ..ros_utils import camera_from_body_rotation, quat_from_ros, stamp_to_sec

try:
    from cv_bridge import CvBridge
except Exception:                                   # pragma: no cover
    CvBridge = None


class CameraImuCalibNode(Node):

    def __init__(self):
        super().__init__("tello_camera_imu_calib")
        d = self.declare_parameter
        d("image_topic", "/image_raw")
        d("camera_info_topic", "/camera_info")
        d("imu_topic", "/imu")
        d("duration_s", 90.0)
        d("output", "camera_imu_calibration.yaml")
        d("min_rotation_deg", 3.0)
        d("max_offset_s", 0.6)

        self.duration = float(self.get_parameter("duration_s").value)
        self.output = str(self.get_parameter("output").value)
        self.min_rot = float(self.get_parameter("min_rotation_deg").value)
        self.max_offset = float(self.get_parameter("max_offset_s").value)

        self.bridge = CvBridge() if CvBridge is not None else None
        self.frontend = None
        self.t0 = None
        self.done = False

        # Attitude stream, kept raw so it can be re-interpolated after the time
        # offset is known.
        self.imu_t, self.imu_q = [], []
        # Visual relative rotations, each with its own [t_ref, t_cur] interval.
        self.vis = []

        vis_g = MutuallyExclusiveCallbackGroup()
        tel_g = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Imu, str(self.get_parameter("imu_topic").value),
                                 self.on_imu, qos_profile_sensor_data, callback_group=tel_g)
        self.create_subscription(CameraInfo,
                                 str(self.get_parameter("camera_info_topic").value),
                                 self.on_info, qos_profile_sensor_data, callback_group=vis_g)
        self.create_subscription(Image, str(self.get_parameter("image_topic").value),
                                 self.on_image, qos_profile_sensor_data, callback_group=vis_g)
        self.create_timer(3.0, self.on_tick, callback_group=tel_g)
        self.get_logger().info(
            "Rotate the drone smoothly about ALL THREE axes in front of a "
            "textured scene. Avoid jerks (motion blur) and keep it in one place "
            "-- this fit needs rotation, not translation.")

    def on_info(self, msg: CameraInfo) -> None:
        if self.frontend is not None:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 0:
            return
        D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
        # Short keyframe intervals give many rotation pairs, which is what this
        # fit wants -- accuracy per pair matters less than pair count and axis
        # diversity.
        self.frontend = VoFrontend(K, D, FrontendConfig(
            kf_min_parallax_px=6.0, kf_min_interval_s=0.05, kf_max_interval_s=0.4))
        self.get_logger().info("camera_info received; recording")

    def on_imu(self, msg: Imu) -> None:
        if self.done:
            return
        t = stamp_to_sec(msg.header.stamp)
        if self.t0 is None:
            self.t0 = t
        self.imu_t.append(t)
        self.imu_q.append(quat_from_ros(msg.orientation))

    def on_image(self, msg: Image) -> None:
        if self.done or self.frontend is None or self.bridge is None:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        t = stamp_to_sec(msg.header.stamp)
        m = self.frontend.process(img, t)
        if m is not None:
            self.vis.append((m.ref_stamp, m.stamp, m.R_c1c2, m.n_inliers))

    def on_tick(self) -> None:
        if self.done or self.t0 is None or not self.imu_t:
            return
        elapsed = self.imu_t[-1] - self.t0
        if elapsed < self.duration:
            self.get_logger().info(
                f"  {elapsed:5.1f}/{self.duration:.0f}s  imu={len(self.imu_t)} "
                f"visual_pairs={len(self.vis)}")
            return
        self.done = True
        self.finish()

    # ------------------------------------------------------------------ #

    def _body_rotation(self, t_a: float, t_b: float, offset: float = 0.0):
        """Attitude change between two instants, by SLERP on the raw stream."""
        ta, tb = t_a - offset, t_b - offset
        q_a = self._interp_quat(ta)
        q_b = self._interp_quat(tb)
        if q_a is None or q_b is None:
            return None
        return lie.quat_to_rot(q_a).T @ lie.quat_to_rot(q_b)

    def _interp_quat(self, t: float):
        """Spherical interpolation of the attitude stream at time ``t``.

        Linear interpolation of quaternion *components* is the usual shortcut
        and it is wrong twice over: it does not stay on the unit sphere, and it
        traverses the rotation at a non-constant rate. On-manifold interpolation
        (``q_a (x) Exp(a * Log(q_a^-1 q_b))``) is exact and no more expensive.
        """
        T = self.imu_t
        if not T or t < T[0] or t > T[-1]:
            return None
        i = int(np.searchsorted(T, t))
        i = max(1, min(len(T) - 1, i))
        t0, t1 = T[i - 1], T[i]
        if t1 - t0 < 1e-9:
            return self.imu_q[i]
        a = (t - t0) / (t1 - t0)
        q0, q1 = self.imu_q[i - 1], self.imu_q[i]
        return lie.quat_boxplus(q0, a * lie.quat_boxminus(q1, q0))

    def finish(self) -> None:
        if len(self.vis) < 15:
            self.get_logger().error(
                f"only {len(self.vis)} visual rotation pairs; need at least 15. "
                "Is the scene textured enough, and is /image_raw actually flowing?")
            rclpy.shutdown()
            return

        # ---- 1. time offset from angular-rate magnitudes -----------------
        # Visual: |Log(R_c1c2)| / dt over each keyframe interval, stamped at the
        # interval midpoint. Inertial: the same quantity from the attitude
        # stream. Both are frame-independent, so this works before R_BC is known.
        v_t, v_w = [], []
        for (ta, tb, Rc, _n) in self.vis:
            dt = tb - ta
            if dt > 1e-3:
                v_t.append(0.5 * (ta + tb))
                v_w.append(np.linalg.norm(lie.Log(Rc)) / dt)
        i_t, i_w = [], []
        for k in range(1, len(self.imu_t)):
            dt = self.imu_t[k] - self.imu_t[k - 1]
            if 1e-3 < dt < 0.5:
                i_t.append(0.5 * (self.imu_t[k] + self.imu_t[k - 1]))
                i_w.append(np.linalg.norm(
                    lie.quat_boxminus(self.imu_q[k], self.imu_q[k - 1])) / dt)

        offset, corr = 0.0, 0.0
        try:
            # Streams: a = inertial (near-real-time), b = visual (delayed).
            res = estimate_time_offset(np.array(i_t), np.array(i_w),
                                       np.array(v_t), np.array(v_w),
                                       max_offset_s=self.max_offset)
            offset, corr = res.offset_s, res.correlation
            self.get_logger().info(
                f"time offset: image stamps lag telemetry by {offset*1e3:.0f} ms "
                f"(correlation {corr:.2f})")
            if corr < 0.5:
                self.get_logger().warn(
                    "  ! weak correlation -- rotate faster/more, or the streams "
                    "are not observing the same motion")
        except ValueError as e:
            self.get_logger().warn(f"time offset failed: {e}")

        # ---- 2. hand-eye, with the offset applied ------------------------
        Rc_list, Rb_list = [], []
        for (ta, tb, Rc, n) in self.vis:
            Rb = self._body_rotation(ta, tb, offset)
            if Rb is None:
                continue
            Rc_list.append(Rc)
            Rb_list.append(Rb)

        excitation = rotation_excitation(Rb_list)
        try:
            he = estimate_camera_imu_rotation(Rc_list, Rb_list,
                                              min_angle_deg=self.min_rot)
        except ValueError as e:
            self.get_logger().error(str(e))
            rclpy.shutdown()
            return

        nominal = camera_from_body_rotation(0.0)
        delta = nominal.T @ he.R_BC
        yaw, pitch, roll = lie.rot_to_euler_zyx(delta)
        rpy_deg = [float(np.degrees(roll)), float(np.degrees(pitch)),
                   float(np.degrees(yaw))]

        self.get_logger().info("=" * 70)
        self.get_logger().info(f"hand-eye: used {he.n_used} pairs, rejected {he.n_rejected}")
        self.get_logger().info(f"  median residual   : {he.residual_deg:.3f} deg")
        self.get_logger().info(f"  angle mismatch    : {he.angle_mismatch_deg:.3f} deg")
        self.get_logger().info(f"  excitation score  : {excitation:.3f}  "
                               f"({'GOOD' if excitation > 0.25 else 'POOR - rotate about more axes'})")
        self.get_logger().info(f"  R_BC vs nominal   : rpy = {np.round(rpy_deg, 2)} deg")
        if excitation < 0.25:
            self.get_logger().warn(
                "  ! low excitation: the extrinsic is undetermined about the "
                "under-excited axis. A small residual here does NOT mean the "
                "answer is right.")
        if he.residual_deg > 3.0:
            self.get_logger().warn(
                "  ! large residual: check the time offset and that the scene "
                "has enough texture for reliable visual rotation.")
        self.get_logger().info("=" * 70)

        doc = {
            "tello_vio": {
                "ros__parameters": {
                    "extrinsic_rpy_deg": rpy_deg,
                    "time_offset_s": float(offset),
                }
            },
            "_measured": {
                "R_BC": [[float(x) for x in row] for row in he.R_BC],
                "hand_eye_residual_deg": float(he.residual_deg),
                "excitation_score": float(excitation),
                "pairs_used": int(he.n_used),
                "pairs_rejected": int(he.n_rejected),
                "time_offset_correlation": float(corr),
            },
        }
        with open(self.output, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        self.get_logger().info(f"wrote {self.output}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraImuCalibNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
