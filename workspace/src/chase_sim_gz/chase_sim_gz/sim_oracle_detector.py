"""sim_oracle_detector -- ground truth in, DroneDetection out
(sim_training_architecture.md 3.3). Tier B's main mode needs NO rendering:
poses come from the bridged odometry, and this node replaces the detector.

The projection uses OUR REAL CALIBRATION (fx = fy = 919.42, c = (480, 360))
imported from chase_gym.constants -- never a sim camera's intrinsics: the
policy must live in the real camera's geometry. Box size follows the
similar-triangles identity the real chase_detector uses; the FOV test drops
out-of-frame targets; corruption applies the Phase-0 statistics. Output is
byte-identical in meaning to the real chase_detector's DroneDetection.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from chase_msgs.msg import DroneDetection
from chase_gym import constants as C
from chase_gym.corruption import CorruptionConfig, CorruptionModel, Measurement
from chase_gym.kinematics import FollowerState, project, world_to_camera

from .gz_iface import quat_to_yaw


class SimOracleDetector(Node):
    def __init__(self):
        super().__init__('sim_oracle_detector')
        self.declare_parameter('rate_hz', 10.0)       # the control rate
        self.declare_parameter('ref_width_m', C.TELLO_BODY_W)
        self.declare_parameter('corruption_file', '')
        self.declare_parameter('corruption_enabled', True)
        self.declare_parameter('seed', 0)

        self.ref_width = float(self.get_parameter('ref_width_m').value)
        path = str(self.get_parameter('corruption_file').value)
        cfg = CorruptionConfig.from_file(path) if path else CorruptionConfig()
        cfg.enabled = bool(self.get_parameter('corruption_enabled').value)
        self.corruption = CorruptionModel(cfg)
        self.rng = np.random.default_rng(
            int(self.get_parameter('seed').value))

        self._follower = None
        self._target = None
        self.create_subscription(Odometry, '/model/follower/odometry',
                                 self._on_follower, 10)
        self.create_subscription(Odometry, '/model/target/odometry',
                                 self._on_target, 10)
        self.pub = self.create_publisher(DroneDetection, '/chase/detection', 10)
        rate = float(self.get_parameter('rate_hz').value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'oracle up: fx={C.FX} c=({C.CX},{C.CY}) ref_width='
            f'{self.ref_width} m (the choice scales every range -- printed, '
            f'never implicit), corruption={"on" if cfg.enabled else "OFF"}')

    def _on_follower(self, msg: Odometry) -> None:
        self._follower = msg

    def _on_target(self, msg: Odometry) -> None:
        self._target = msg

    def _tick(self) -> None:
        # Publish EVERY tick, misses included -- downstream consumers get
        # an explicit visibility signal, never a silent gap (the
        # DroneDetection contract).
        msg = DroneDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.t_receive = msg.header.stamp
        msg.image_width = C.FRAME_W
        msg.image_height = C.FRAME_H
        msg.detected = False
        msg.confidence = 0.0
        msg.class_name = 'drone'
        msg.range_m = float('nan')
        msg.inference_ms = 0.0

        if self._follower is not None and self._target is not None:
            fp = self._follower.pose.pose
            tp = self._target.pose.pose
            st = FollowerState(
                pos=np.array([fp.position.x, fp.position.y, fp.position.z]),
                yaw=quat_to_yaw(fp.orientation.w, fp.orientation.x,
                                fp.orientation.y, fp.orientation.z))
            cam = world_to_camera(
                np.array([tp.position.x, tp.position.y, tp.position.z]), st)
            u, v, w_px, in_frame = project(cam, self.ref_width)
            meas = Measurement(u, v, w_px) if in_frame else None
            meas = self.corruption.apply(meas, self.rng)
            if meas is not None:
                h_px = meas.w_px * (C.TELLO_BODY_H / self.ref_width)
                msg.detected = True
                msg.confidence = 1.0
                msg.xmin = float(meas.u - meas.w_px / 2.0)
                msg.xmax = float(meas.u + meas.w_px / 2.0)
                msg.ymin = float(meas.v - h_px / 2.0)
                msg.ymax = float(meas.v + h_px / 2.0)
                msg.range_m = float(C.FX * self.ref_width / meas.w_px)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimOracleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
