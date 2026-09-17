"""chase_detector -- YOLO inference on the Tello video stream.

    ros2 run chase_detector detector
    ros2 launch chase_detector chase.launch.py           # driver + detector + viewer

Subscribes  image_raw     (sensor_msgs/Image, best-effort depth 1: latest wins)
            camera_info   (sensor_msgs/CameraInfo, from the tello driver)
Publishes   chase/detection        (chase_msgs/DroneDetection, EVERY frame)
            chase/image_annotated  (sensor_msgs/Image bgr8, when enabled)

Design decisions, each verified against this repo or the installed packages:

* Colour: cv_bridge converts the incoming image to bgr8 whatever the driver's
  `publish_bgr` says. Ultralytics assumes numpy input is BGR; djitellopy
  decodes to RGB -- feeding one to the other unconverted channel-swaps every
  frame (see chase_detector/detector.py docstring for chapter and verse).
* QoS: the driver publishes video with qos_profile_sensor_data (best-effort).
  A reliable subscriber would match nothing and receive nothing, silently.
  Depth 1 makes DDS overwrite the queue while a frame is being processed, so
  a slow (CPU) inference drops stale frames instead of building a latency
  queue -- the freshest frame is always the next one processed.
* Timestamps: every DroneDetection carries t_capture (= the image header
  stamp, which the driver sets on UDP arrival) AND t_receive (this node's
  clock when the callback fired) per implementation_plan.md section 3 --
  latency must be attributable, and both stamps come from the same host
  clock here, so their difference is meaningful.
* Misses are published too: `detected: false` is the visibility signal the
  behaviour FSM keys on. Silence would be indistinguishable from a crash.
"""
import math
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from chase_msgs.msg import DroneDetection

from chase_detector.csv_log import DetectionCsvLogger, default_csv_path
from chase_detector.detector import YoloDroneDetector
from chase_detector.geometry import PinholeGeometry

GREEN = (0, 255, 0)
RED = (0, 0, 255)
GRAY = (160, 160, 160)
WHITE = (255, 255, 255)


class ChaseDetectorNode(Node):
    def __init__(self):
        super().__init__('chase_detector')

        p = self.declare_parameter
        p('weights', 'yolov8n.pt')
        p('device', 'cuda:0')
        p('imgsz', 640)
        p('conf', 0.25)
        p('iou', 0.45)
        p('max_det', 8)
        # Comma-separated class names to keep; empty keeps everything. With
        # one-class drone weights set this to 'drone'. With stock COCO
        # weights leave it empty -- COCO has no drone class (verified), so
        # stock weights are a pipeline smoke test, not a drone detector.
        p('target_classes', '')
        p('min_box_px', 4.0)
        # Real width of what the boxes enclose: 0.180 m Tello prop-tip span,
        # 0.098 m bare fuselage. Wrong choice scales every range by ~1.8x.
        p('ref_width_m', 0.180)
        p('publish_annotated', True)
        # Directory for the per-frame CSV; empty disables logging.
        p('csv_dir', '')

        g = lambda n: self.get_parameter(n).value
        weights = os.path.expanduser(str(g('weights')))
        wanted = [c for c in str(g('target_classes')).split(',') if c.strip()]
        self.ref_width_m = float(g('ref_width_m'))
        self.publish_annotated = bool(g('publish_annotated'))

        if not os.path.isfile(weights):
            # Fail here, not inside ultralytics, which would otherwise try to
            # DOWNLOAD a matching name from the internet -- surprising on a
            # machine that is supposed to be on the drone's WiFi AP.
            raise SystemExit(
                f"weights not found: {os.path.abspath(weights)}\n"
                "Pass weights:=/absolute/path/to/model.pt (the repo copy of "
                "yolov8n.pt works for smoke tests, from the repo root).")

        try:
            self.detector = YoloDroneDetector(
                weights, device=str(g('device')), imgsz=int(g('imgsz')),
                conf=float(g('conf')), iou=float(g('iou')),
                max_det=int(g('max_det')), target_classes=wanted,
                min_box_px=float(g('min_box_px')))
        except ValueError as e:
            # The requested class does not exist in these weights. Loud exit;
            # an eternally-empty detection stream would look like a bug
            # somewhere else.
            raise SystemExit(f'target_classes: {e}')

        if self.detector.device_fell_back:
            self.get_logger().warn(
                f"device '{self.detector.requested_device}' requested but CUDA "
                "is not available; falling back to CPU (~4x slower, still "
                "faster than the 30 fps stream on this class of machine)")

        warm_ms = self.detector.warmup()

        self.bridge = CvBridge()
        self.geometry = None          # set on first camera_info
        self.seq = 0
        self.miss_streak = 0
        self.proc_fps = 0.0           # EMA over processed frames
        self._last_proc_t = None

        self.csv = None
        csv_dir = str(g('csv_dir'))
        if csv_dir:
            self.csv = DetectionCsvLogger(default_csv_path(csv_dir))

        latest_wins = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, 'image_raw', self.on_image, latest_wins)
        self.create_subscription(CameraInfo, 'camera_info', self.on_camera_info,
                                 qos_profile_sensor_data)
        self.pub_detection = self.create_publisher(
            DroneDetection, 'chase/detection',
            QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                       history=QoSHistoryPolicy.KEEP_LAST, depth=50))
        self.pub_annotated = None
        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(
                Image, 'chase/image_annotated', qos_profile_sensor_data)

        classes = self.detector.names
        shown = ', '.join(classes[k] for k in sorted(classes)[:12])
        if len(classes) > 12:
            shown += f', ... ({len(classes)} total)'
        self.get_logger().info(
            '\n'.join([
                'chase_detector up:',
                f'  weights   : {os.path.abspath(weights)}',
                f'  device    : {self.detector.device}'
                + (' (fell back from '
                   f'{self.detector.requested_device})'
                   if self.detector.device_fell_back else ''),
                f'  imgsz={int(g("imgsz"))} conf={float(g("conf"))} '
                f'iou={float(g("iou"))} max_det={int(g("max_det"))}',
                f'  classes   : kept={wanted or "ALL"} of [{shown}]',
                f'  ref_width : {self.ref_width_m:.3f} m (range scale)',
                f'  warmup    : {warm_ms:.1f} ms/frame after warmup',
                f'  csv       : {self.csv.path if self.csv else "disabled"}',
                '  publishing: chase/detection'
                + (' + chase/image_annotated' if self.publish_annotated else ''),
            ]))

    # ------------------------------------------------------------------ #

    def on_camera_info(self, msg: CameraInfo):
        geom = PinholeGeometry.from_k(msg.k)
        if geom is None:
            return
        if self.geometry is None:
            self.get_logger().info(
                f'camera_info: fx={geom.fx:.1f} fy={geom.fy:.1f} '
                f'cx={geom.cx:.1f} cy={geom.cy:.1f} -> '
                f'HFOV {geom.hfov_deg(msg.width):.1f} deg, '
                f'VFOV {geom.vfov_deg(msg.height):.1f} deg')
        self.geometry = geom

    def on_image(self, msg: Image):
        t_receive = self.get_clock().now()
        try:
            self.process(msg, t_receive)
        except Exception as e:  # keep the stream alive on a bad frame
            self.get_logger().error(f'frame dropped: {type(e).__name__}: {e}',
                                    throttle_duration_sec=5.0)

    def process(self, msg: Image, t_receive):
        # Whatever encoding the driver chose (rgb8 or bgr8 via publish_bgr),
        # normalise to the BGR that ultralytics assumes for numpy input.
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = bgr.shape[:2]

        result = self.detector.detect(bgr)
        best = result.best
        transport_ms = (t_receive - Time.from_msg(msg.header.stamp)) \
            .nanoseconds / 1e6

        now_s = t_receive.nanoseconds / 1e9
        if self._last_proc_t is not None:
            dt = now_s - self._last_proc_t
            if dt > 0:
                inst = 1.0 / dt
                self.proc_fps = inst if self.proc_fps == 0.0 \
                    else 0.9 * self.proc_fps + 0.1 * inst
        self._last_proc_t = now_s

        out = DroneDetection()
        out.header = msg.header
        out.t_receive = t_receive.to_msg()
        out.image_width = w
        out.image_height = h
        out.inference_ms = float(result.inference_ms)

        range_m = math.nan
        az = el = math.nan
        if best is None:
            self.miss_streak += 1
            out.detected = False
            out.confidence = 0.0
            out.class_name = ''
            out.xmin = out.ymin = out.xmax = out.ymax = 0.0
            out.range_m = math.nan
        else:
            self.miss_streak = 0
            cx, cy = best.center
            if self.geometry is not None:
                range_m = self.geometry.range_from_width(
                    best.width, self.ref_width_m)
                az, el = self.geometry.bearings_deg(cx, cy)
            out.detected = True
            out.confidence = best.confidence
            out.class_name = best.class_name
            out.xmin, out.ymin = best.xmin, best.ymin
            out.xmax, out.ymax = best.xmax, best.ymax
            out.range_m = range_m
        self.pub_detection.publish(out)

        if self.csv is not None:
            b = best
            self.csv.write(
                seq=self.seq,
                t_capture_ns=Time.from_msg(msg.header.stamp).nanoseconds,
                t_receive_ns=t_receive.nanoseconds,
                transport_ms=transport_ms,
                inference_ms=result.inference_ms,
                detected=int(b is not None),
                class_name=b.class_name if b else None,
                confidence=b.confidence if b else None,
                xmin=b.xmin if b else None, ymin=b.ymin if b else None,
                xmax=b.xmax if b else None, ymax=b.ymax if b else None,
                cx=b.center[0] if b else None, cy=b.center[1] if b else None,
                w=b.width if b else None, h=b.height if b else None,
                area_frac=(b.width * b.height) / (w * h) if b else None,
                range_m=range_m, az_deg=az, el_deg=el,
                image_w=w, image_h=h)
        self.seq += 1

        if self.pub_annotated is not None:
            frame = self.draw(bgr.copy(), result, transport_ms, range_m, az, el)
            am = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            am.header = msg.header
            self.pub_annotated.publish(am)

    # ------------------------------------------------------------------ #

    def draw(self, frame: np.ndarray, result, transport_ms, range_m, az, el):
        h, w = frame.shape[:2]
        if self.geometry is not None:   # principal point crosshair
            pcx, pcy = int(self.geometry.cx), int(self.geometry.cy)
            cv2.drawMarker(frame, (pcx, pcy), GRAY, cv2.MARKER_CROSS, 24, 1)

        for i, d in enumerate(result.boxes):
            colour = GREEN if i == 0 else GRAY
            p1, p2 = (int(d.xmin), int(d.ymin)), (int(d.xmax), int(d.ymax))
            cv2.rectangle(frame, p1, p2, colour, 2)
            cv2.putText(frame, f'{d.class_name} {d.confidence:.2f}',
                        (p1[0], max(14, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        best = result.best
        bar = [f'yolo={result.inference_ms:5.1f}ms  '
               f'transport={transport_ms:5.0f}ms  '
               f'proc={self.proc_fps:4.1f}fps  {w}x{h}']
        if best is not None:
            cx, cy = best.center
            cv2.circle(frame, (int(cx), int(cy)), 5, RED, -1)
            cv2.line(frame, (int(cx) - 12, int(cy)), (int(cx) + 12, int(cy)),
                     RED, 1)
            cv2.line(frame, (int(cx), int(cy) - 12), (int(cx), int(cy) + 12),
                     RED, 1)
            line = (f'px=({cx:4.0f},{cy:4.0f})  '
                    f'box={best.width:3.0f}x{best.height:3.0f}  '
                    f'conf={best.confidence:.2f}')
            if not math.isnan(range_m):
                line += f'  range={range_m:4.2f}m'
            if not math.isnan(az):
                line += f'  az={az:+5.1f}  el={el:+5.1f}'
            bar.insert(0, line)
        else:
            cv2.putText(frame, f'NO TARGET (missed {self.miss_streak})',
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

        y0 = h - 14 - 22 * (len(bar) - 1)
        cv2.rectangle(frame, (0, y0 - 22), (w, h), (0, 0, 0), -1)
        for i, text in enumerate(bar):
            cv2.putText(frame, text, (10, y0 + 22 * i - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        return frame

    def destroy_node(self):
        if self.csv is not None:
            path = self.csv.close()
            self.get_logger().info(f'CSV: {self.csv.rows} rows -> {path}')
            self.csv = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ChaseDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
