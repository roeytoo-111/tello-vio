#!/usr/bin/env python3
"""Publish ``map -> odom`` by aligning a scale-free SLAM pose to metric VIO.

This is the node that lets ORB-SLAM2's loop closures remove VIO drift *without*
making the odometry jump, following REP-105:

* ``odom -> base_link`` (from ``vio_node``) is continuous. Controllers
  differentiate it, so a discontinuity there is a step input to the aircraft.
* ``map -> odom`` (from here) is allowed to jump. Nothing differentiates it.

So when SLAM closes a loop and its trajectory snaps, the correction lands in
``map -> odom`` and the control loop never sees a step.

It also solves the monocular scale problem for the *map*: aligning the SLAM
trajectory to the metric VIO trajectory over a sliding window yields the Sim(3)
between them, whose scale factor is the map's metres-per-SLAM-unit. That is
published so downstream consumers can interpret ORB-SLAM2's map points in
metres.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

from .. import lie
from ..ros_utils import make_transform, quat_from_ros, stamp_to_sec
from ..sim3 import ransac_sim3


class MapAlignNode(Node):

    def __init__(self):
        super().__init__("tello_map_align")
        d = self.declare_parameter
        d("slam_pose_topic", "/orbslam/pose")
        d("vio_odom_topic", "/tello_vio/odom")
        d("map_frame", "map")
        d("odom_frame", "odom")
        d("window_s", 20.0)
        d("min_pairs", 25)
        d("min_span_m", 0.5)
        d("update_hz", 2.0)
        d("max_pair_dt", 0.10)
        d("ransac_threshold_m", 0.20)
        d("fixed_scale", 0.0)         # >0 locks the scale instead of fitting it

        g = lambda n: self.get_parameter(n).value
        self.map_frame = str(g("map_frame"))
        self.odom_frame = str(g("odom_frame"))
        self.window = float(g("window_s"))
        self.min_pairs = int(g("min_pairs"))
        self.min_span = float(g("min_span_m"))
        self.max_pair_dt = float(g("max_pair_dt"))
        self.threshold = float(g("ransac_threshold_m"))
        self.fixed_scale = float(g("fixed_scale"))

        self.lock = threading.Lock()
        self.slam: deque = deque()      # (t, p_slam)
        self.vio: deque = deque()       # (t, p_odom, q_odom)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_scale = self.create_publisher(Float32, "~/map_scale", 1)

        self.create_subscription(PoseStamped, str(g("slam_pose_topic")),
                                 self.on_slam, qos_profile_sensor_data)
        self.create_subscription(Odometry, str(g("vio_odom_topic")),
                                 self.on_vio, qos_profile_sensor_data)
        self.create_timer(1.0 / max(0.1, float(g("update_hz"))), self.on_update)
        self.get_logger().info(
            f"aligning {g('slam_pose_topic')} to {g('vio_odom_topic')} -> "
            f"{self.map_frame} -> {self.odom_frame}")

    def on_slam(self, msg: PoseStamped) -> None:
        t = stamp_to_sec(msg.header.stamp)
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        with self.lock:
            self.slam.append((t, p))
            self._trim(self.slam, t)

    def on_vio(self, msg: Odometry) -> None:
        t = stamp_to_sec(msg.header.stamp)
        p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                      msg.pose.pose.position.z])
        q = quat_from_ros(msg.pose.pose.orientation)
        with self.lock:
            self.vio.append((t, p, q))
            self._trim(self.vio, t)

    def _trim(self, dq: deque, now: float) -> None:
        while dq and (now - dq[0][0]) > self.window:
            dq.popleft()

    def on_update(self) -> None:
        with self.lock:
            slam = list(self.slam)
            vio = list(self.vio)
        if len(slam) < self.min_pairs or len(vio) < self.min_pairs:
            return

        # Pair by nearest timestamp. Both streams are irregular, so a naive
        # index-wise pairing would silently associate poses seconds apart.
        vt = np.array([v[0] for v in vio])
        src, dst = [], []
        for (ts, ps) in slam:
            k = int(np.argmin(np.abs(vt - ts)))
            if abs(vt[k] - ts) <= self.max_pair_dt:
                src.append(ps)
                dst.append(vio[k][1])
        if len(src) < self.min_pairs:
            return
        src = np.asarray(src)
        dst = np.asarray(dst)

        # A similarity fit needs geometric spread. Over a hover, every pair sits
        # in the same place, the fit is degenerate and the scale is arbitrary --
        # so refuse rather than publish a confident wrong transform.
        span = float(np.max(np.linalg.norm(dst - dst.mean(axis=0), axis=1)))
        if span < self.min_span:
            self.get_logger().info(
                f"trajectory span {span:.2f} m < {self.min_span} m: alignment is "
                "degenerate, holding the previous map->odom",
                throttle_duration_sec=10.0)
            return

        with_scale = self.fixed_scale <= 0.0
        T, inl = ransac_sim3(src, dst, threshold=self.threshold, iterations=150,
                             with_scale=with_scale)
        if not with_scale:
            T.s = self.fixed_scale

        # T maps SLAM points into the odom frame. map->odom is then the
        # rigid part of its inverse: the SLAM frame *is* the map frame, so a
        # point at odom coordinates x sits at map coordinates T^-1 x.
        Tinv = T.inverse()
        q = lie.rot_to_quat(Tinv.R)
        stamp = self.get_clock().now().to_msg()
        # TF is a rigid transform and cannot carry scale, so the scale is
        # published separately rather than silently dropped.
        self.tf_broadcaster.sendTransform(
            make_transform(self.map_frame, self.odom_frame, stamp, Tinv.t, q))

        m = Float32()
        m.data = float(T.s)
        self.pub_scale.publish(m)

        resid = float(np.median(np.linalg.norm(T.apply(src) - dst, axis=1)))
        self.get_logger().info(
            f"map->odom: {int(np.count_nonzero(inl))}/{len(src)} inliers, "
            f"scale={T.s:.4f} m/unit, residual={resid:.3f} m, span={span:.2f} m",
            throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = MapAlignNode()
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
