#!/usr/bin/env python3
"""Publish metric ground truth from a printed ArUco marker, for evaluating VIO.

Usage
-----
1. Print a marker and MEASURE the printed black square with a ruler::

       ros2 run tello_vio make_marker --ros-args -p output:=marker.png
       # print it at 100 % scale (no "fit to page"), then measure it

2. Tape it to a wall. Fly the drone so the marker stays in view.

3. Run alongside the VIO stack::

       ros2 run tello_vio ground_truth --ros-args -p marker_size_m:=0.20

It publishes ``~/pose`` (camera in the marker frame) and ``~/marker_in_camera``
(the robust quantity -- see below), plus TF ``marker -> camera_truth``.

Accuracy, stated honestly
-------------------------
This is a ~1-2 cm reference at 1-2 m range, not a millimetre one, and its
ROTATION is only trustworthy when the marker is viewed obliquely (see the
planar-ambiguity discussion in :mod:`tello_vio.fiducial`). Samples whose
ambiguity or reprojection error is poor are dropped rather than published,
because a silently-wrong reference is worse than no reference.

The scale of the whole reference is set by ``marker_size_m``. Measure the
printed marker; do not trust the nominal size. Printers scale by a few percent
and that error passes straight into every ground-truth number.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

from .. import lie
from ..fiducial import (detect_markers, estimate_marker_pose, make_detector)
from ..ros_utils import make_transform, quat_to_ros, sec_to_stamp, stamp_to_sec

try:
    from cv_bridge import CvBridge
except Exception:                                   # pragma: no cover
    CvBridge = None

import cv2


class GroundTruthNode(Node):

    def __init__(self):
        super().__init__("tello_ground_truth")
        d = self.declare_parameter
        d("image_topic", "/image_raw")
        d("camera_info_topic", "/camera_info")
        d("marker_size_m", 0.20)
        d("marker_id", -1)              # -1 = accept any single marker
        d("dictionary", "DICT_4X4_50")
        d("marker_frame", "marker")
        d("camera_truth_frame", "camera_truth")
        d("max_reproj_px", 3.0)
        d("max_ambiguity", 0.90)
        d("min_side_px", 40.0)
        d("publish_tf", True)

        g = lambda n: self.get_parameter(n).value
        self.size = float(g("marker_size_m"))
        self.want_id = int(g("marker_id"))
        self.marker_frame = str(g("marker_frame"))
        self.cam_frame = str(g("camera_truth_frame"))
        self.max_reproj = float(g("max_reproj_px"))
        self.max_amb = float(g("max_ambiguity"))
        self.min_side = float(g("min_side_px"))
        self.publish_tf = bool(g("publish_tf"))

        self.K = None
        self.D = None
        self.bridge = CvBridge() if CvBridge is not None else None
        self.detector = make_detector(str(g("dictionary")))
        self._n_seen = 0
        self._n_used = 0

        self.pub_pose = self.create_publisher(PoseStamped, "~/pose", 10)
        self.pub_marker = self.create_publisher(PointStamped, "~/marker_in_camera", 10)
        self.pub_reproj = self.create_publisher(Float32, "~/reproj_px", 10)
        self.tf = TransformBroadcaster(self)

        self.create_subscription(CameraInfo, str(g("camera_info_topic")),
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, str(g("image_topic")),
                                 self.on_image, qos_profile_sensor_data)
        self.create_timer(2.0, self.on_status)
        self.get_logger().info(
            f"ground truth: {self.size*100:.1f} cm marker, "
            f"publishing {self.marker_frame} -> {self.cam_frame}")

    def on_info(self, msg: CameraInfo) -> None:
        if self.K is not None:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 0:
            return
        self.K = K
        self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
        self.get_logger().info(f"camera_info: {msg.width}x{msg.height} fx={K[0,0]:.1f}")

    def on_image(self, msg: Image) -> None:
        if self.K is None:
            return
        try:
            if self.bridge is not None:
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            else:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn(f"image conversion failed: {e}",
                                   throttle_duration_sec=5.0)
            return

        corners, ids = detect_markers(self.detector, gray)
        if ids is None or len(ids) == 0:
            return
        self._n_seen += 1

        best = None
        for c, i in zip(corners, ids.ravel()):
            if self.want_id >= 0 and int(i) != self.want_id:
                continue
            det = estimate_marker_pose(c, self.size, self.K, self.D, int(i))
            if det is None:
                continue
            if best is None or det.side_px > best.side_px:
                best = det
        if best is None:
            return

        # Quality gates. A rejected sample is invisible; a bad sample silently
        # corrupts every error statistic computed from it.
        if best.side_px < self.min_side:
            return
        if best.reproj_px > self.max_reproj:
            return
        amb_ok = not np.isfinite(best.ambiguity) or best.ambiguity < self.max_amb

        stamp = msg.header.stamp
        pt = PointStamped()
        pt.header.stamp = stamp
        pt.header.frame_id = msg.header.frame_id or self.cam_frame
        pt.point.x, pt.point.y, pt.point.z = [float(v) for v in best.t_cm]
        self.pub_marker.publish(pt)

        m = Float32(); m.data = float(best.reproj_px); self.pub_reproj.publish(m)

        # Camera-in-marker pose depends on the rotation, so it is only
        # published when the orientation is trustworthy.
        if not amb_ok:
            return
        self._n_used += 1
        q = lie.rot_to_quat(best.R_mc)
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.marker_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = \
            [float(v) for v in best.t_mc]
        ps.pose.orientation = quat_to_ros(q)
        self.pub_pose.publish(ps)

        if self.publish_tf:
            self.tf.sendTransform(make_transform(
                self.marker_frame, self.cam_frame, stamp, best.t_mc, q))

    def on_status(self) -> None:
        if self.K is None:
            self.get_logger().info("ground truth: waiting for camera_info")
        elif self._n_seen == 0:
            self.get_logger().info(
                "ground truth: no marker in view "
                "(check size, lighting, and that it fills enough of the frame)")
        else:
            self.get_logger().info(
                f"ground truth: {self._n_used}/{self._n_seen} detections usable")


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
