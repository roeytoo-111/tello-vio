"""chase_viewer -- live window: annotated video + box-centre traces.

    ros2 run chase_detector viewer

Subscribes  chase/image_annotated  (the detector draws; this only displays)
            chase/detection        (for the centre-position history)

Split from the detector on purpose: inference must be able to run headless
(rosbag replay, a machine with no display, the eventual flight stack), and a
GUI crash must never take the perception stream down with it. The overlay is
drawn by the detector on the exact frame it was computed from, so box and
frame can never be misaligned here; this node adds the time-dimension view:

    +----------------------------+-----------+
    |                            |  minimap  |   centre trail in the image
    |     annotated video        +-----------+
    |                            |  cx(t)    |   strip charts, misses as
    |                            +-----------+   red ticks on the baseline
    |                            |  cy(t)    |
    +----------------------------+-----------+

Keys: q / ESC quit, c clear the trace, s save a PNG of the canvas.
"""
import collections
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image

from chase_msgs.msg import DroneDetection

PANEL_W = 360
BG = (24, 24, 24)
GRID = (60, 60, 60)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
WINDOW = 'tello-chase'


class ChaseViewerNode(Node):
    def __init__(self):
        super().__init__('chase_viewer')
        self.declare_parameter('trace_seconds', 10.0)
        self.declare_parameter('rate_hz', 30.0)
        self.trace_seconds = float(self.get_parameter('trace_seconds').value)

        self.bridge = CvBridge()
        self.frame = None                     # latest annotated frame (bgr)
        self.image_wh = (960, 720)            # updated from detections
        # (t_capture_s, detected, cx, cy)
        self.history = collections.deque(maxlen=4096)

        latest_wins = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, 'chase/image_annotated',
                                 self.on_image, latest_wins)
        self.create_subscription(
            DroneDetection, 'chase/detection', self.on_detection,
            QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                       history=QoSHistoryPolicy.KEEP_LAST, depth=50))

        self._shown_once = False
        self._live_logged = False
        rate = max(5.0, float(self.get_parameter('rate_hz').value))
        self.create_timer(1.0 / rate, self.render)
        self.get_logger().info(
            'viewer up: waiting for chase/image_annotated + chase/detection '
            '(q quits, c clears the trace, s saves a PNG)')

    def on_image(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'bad frame: {e}', throttle_duration_sec=5.0)

    def on_detection(self, msg: DroneDetection):
        if msg.image_width and msg.image_height:
            self.image_wh = (msg.image_width, msg.image_height)
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        if msg.detected:
            cx = 0.5 * (msg.xmin + msg.xmax)
            cy = 0.5 * (msg.ymin + msg.ymax)
            self.history.append((t, True, cx, cy))
        else:
            self.history.append((t, False, 0.0, 0.0))

    # ------------------------------------------------------------------ #

    def render(self):
        video = self.frame
        if video is None:
            video = np.full((720, 960, 3), BG, np.uint8)
            cv2.putText(video, 'waiting for chase/image_annotated ...',
                        (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GRAY, 2)

        h = video.shape[0]
        panel = np.full((h, PANEL_W, 3), BG, np.uint8)
        third = h // 3
        self.draw_minimap(panel[0:third])
        self.draw_strip(panel[third:2 * third], axis=0, label='cx [px]')
        self.draw_strip(panel[2 * third:3 * third], axis=1, label='cy [px]')
        for y in (third, 2 * third):
            cv2.line(panel, (0, y), (PANEL_W, y), GRID, 1)

        canvas = np.hstack([video, panel])
        try:
            cv2.imshow(WINDOW, canvas)
        except cv2.error as e:
            self.get_logger().fatal(
                f"cannot open a GUI window (DISPLAY="
                f"{os.environ.get('DISPLAY')!r}): {e}\n"
                "  No display in this session. Launch with view:=false and "
                "use chase/detection + the CSV instead.")
            rclpy.shutdown()
            return
        if not self._shown_once:
            # WSLg opens this as its own Windows window WITHOUT stealing
            # focus, so it starts behind the terminal. Pin it on top --
            # this is a monitoring window; move it if it is in the way.
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
            self.get_logger().info(
                "window 'tello-chase' is OPEN and set always-on-top "
                "(WSLg: it is a separate window in the Windows taskbar)")
        if self.frame is not None and not self._live_logged:
            self._live_logged = True
            self.get_logger().info('live annotated video is rendering')
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            self.get_logger().info('quit key -- shutting down')
            rclpy.shutdown()
        elif key == ord('c'):
            self.history.clear()
        elif key == ord('s'):
            path = time.strftime('chase_view_%Y%m%d_%H%M%S.png')
            cv2.imwrite(path, canvas)
            self.get_logger().info(f'saved {path}')
        # Window closed with the mouse: WND_PROP_VISIBLE drops below 1.
        if self._shown_once and \
                cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            rclpy.shutdown()
        self._shown_once = True

    def draw_minimap(self, area):
        ah, aw = area.shape[:2]
        iw, ih = self.image_wh
        scale = min((aw - 20) / iw, (ah - 30) / ih)
        mw, mh = int(iw * scale), int(ih * scale)
        x0, y0 = (aw - mw) // 2, 18 + (ah - 30 - mh) // 2
        cv2.rectangle(area, (x0, y0), (x0 + mw, y0 + mh), GRID, 1)
        cv2.putText(area, 'image plane', (8, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
        pts = [(t, cx, cy) for t, det, cx, cy in self.history if det]
        if not pts:
            return
        tail = pts[-300:]
        for i in range(1, len(tail)):
            a = (x0 + int(tail[i - 1][1] * scale), y0 + int(tail[i - 1][2] * scale))
            b = (x0 + int(tail[i][1] * scale), y0 + int(tail[i][2] * scale))
            shade = int(80 + 175 * i / len(tail))
            cv2.line(area, a, b, (0, shade, 0), 1)
        last = tail[-1]
        cv2.circle(area, (x0 + int(last[1] * scale), y0 + int(last[2] * scale)),
                   4, RED, -1)

    def draw_strip(self, area, axis: int, label: str):
        ah, aw = area.shape[:2]
        left, right, top, bot = 44, 8, 18, 14
        pw, ph = aw - left - right, ah - top - bot
        vmax = float(self.image_wh[axis])
        cv2.putText(area, label, (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
        for frac in (0.0, 0.5, 1.0):
            y = top + int(frac * ph)
            cv2.line(area, (left, y), (left + pw, y), GRID, 1)
            cv2.putText(area, f'{vmax * frac:.0f}', (4, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)
        if not self.history:
            return
        t_now = self.history[-1][0]
        t0 = t_now - self.trace_seconds

        def to_xy(t, v):
            x = left + int((t - t0) / self.trace_seconds * pw)
            y = top + int(np.clip(v / vmax, 0.0, 1.0) * ph)
            return x, y

        prev = None
        for t, det, cx, cy in self.history:
            if t < t0:
                continue
            if not det:
                x = left + int((t - t0) / self.trace_seconds * pw)
                cv2.line(area, (x, top + ph), (x, top + ph - 6), RED, 1)
                prev = None
                continue
            cur = to_xy(t, (cx, cy)[axis])
            if prev is not None:
                cv2.line(area, prev, cur, GREEN, 1)
            prev = cur
        cv2.putText(area, f'{self.trace_seconds:.0f}s window',
                    (aw - 90, ah - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ChaseViewerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
