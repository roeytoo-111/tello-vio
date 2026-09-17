"""chase_capture -- record Tello frames to disk for detector training.

    ros2 run chase_detector capture
    ros2 run chase_detector capture --ros-args -p rate_hz:=4.0

Stock COCO weights cannot detect a drone (no such class), so Phase 0 of
implementation_plan.md is a dataset from THIS camera at THIS resolution.
This node is that collector: subscribe to the driver's image_raw, save
JPEGs at a capped rate into a fresh session directory. Filenames carry the
capture stamp in nanoseconds so frames can later be matched against
telemetry or a rosbag.

Frames are saved exactly as published (colour converted to BGR for
imwrite's JPEG writer, which expects BGR -- the on-disk JPEG is correct
either way). Label later with your annotation tool of choice.
"""
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image


class ChaseCaptureNode(Node):
    def __init__(self):
        super().__init__('chase_capture')
        self.declare_parameter('out_dir', 'chase_dataset')
        self.declare_parameter('rate_hz', 2.0)
        self.declare_parameter('jpeg_quality', 92)

        out_root = os.path.expanduser(
            str(self.get_parameter('out_dir').value))
        self.rate_hz = max(0.1, float(self.get_parameter('rate_hz').value))
        self.quality = int(self.get_parameter('jpeg_quality').value)

        session = time.strftime('sess_%Y%m%d_%H%M%S')
        self.dir = os.path.abspath(os.path.join(out_root, session))
        os.makedirs(self.dir, exist_ok=True)

        self.bridge = CvBridge()
        self.count = 0
        self._last_save = 0.0

        self.create_subscription(
            Image, 'image_raw', self.on_image,
            QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       history=QoSHistoryPolicy.KEEP_LAST, depth=1))
        self.get_logger().info(
            f'capturing to {self.dir} at <= {self.rate_hz:.1f} Hz '
            f'(jpeg q{self.quality})')

    def on_image(self, msg: Image):
        now = time.monotonic()
        if now - self._last_save < 1.0 / self.rate_hz:
            return
        self._last_save = now
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'bad frame: {e}', throttle_duration_sec=5.0)
            return
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        path = os.path.join(self.dir, f'frame_{self.count:06d}_{stamp_ns}.jpg')
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        self.count += 1
        if self.count % 25 == 0:
            self.get_logger().info(f'{self.count} frames saved')

    def destroy_node(self):
        self.get_logger().info(f'saved {self.count} frames -> {self.dir}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ChaseCaptureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
