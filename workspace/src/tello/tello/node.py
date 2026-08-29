#!/usr/bin/env python3
"""ROS 2 driver for the DJI Tello.

Rewritten around three properties of this aircraft that the obvious
implementation gets wrong, each of which was reachable in flight:

**1. Every djitellopy command is a blocking UDP round trip.**
``takeoff()``, ``land()``, ``flip()`` and the ``query_*`` calls send a datagram
and block up to ``RESPONSE_TIMEOUT`` seconds, retrying three times. Running any
of them inside a ROS callback on a single-threaded executor freezes the whole
node -- including the ``/emergency`` subscription and the RC dead-man timer --
for up to 21 seconds. A drone commanded forward keeps flying forward for that
entire window because no zeroing ``rc`` packet goes out. All blocking commands
therefore run on a dedicated worker thread (:meth:`_command_worker`), and
``/emergency`` additionally bypasses the queue with a fire-and-forget datagram.

**2. Telemetry is a broadcast, not a poll.**
The SDK pushes state at ~10 Hz. Sampling it on an independent 10 Hz timer
beats one aperiodic stream against another: you get duplicated samples,
dropped samples, and -- worst of all -- a timestamp that is the *timer's* firing
instant rather than the packet's arrival, displaced by an unbounded 0-100 ms.
A fixed camera-IMU offset can be calibrated out; a randomly varying one cannot.
This driver polls at 50 Hz and publishes only on an observed change, stamping at
detection, which bounds the timestamp error to the 20 ms poll period.

**3. The units and axes in the SDK state packet are not ROS units and axes.**
Accelerations are milli-g in an FRD body frame; attitude is NED-referenced with
yaw increasing clockwise. ROS mandates SI in FLU/ENU (REP-103). Publishing the
raw values -- as ``/imu`` previously did -- yields ``z = -10 m/s^2`` at rest and a
yaw that runs backwards, which any downstream estimator will faithfully
integrate into nonsense. Conversions live in
:mod:`tello_vio.tello_model` and are mirrored here so this package has no
dependency on ``tello_vio``.

Topics
------
Published: ``/image_raw``, ``/camera_info``, ``/imu``, ``/odom``, ``/tof``,
``/status``, ``/id``, ``/battery``, ``/temperature``.
Subscribed: ``/takeoff``, ``/land``, ``/emergency``, ``/control``, ``/cmd_vel``,
``/flip``, ``/wifi_config``.
"""

from __future__ import annotations

import errno
import logging
import math
import queue
import socket
import threading
import time
from typing import Optional

import numpy
import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import (BatteryState, CameraInfo, Image, Imu, Range,
                             Temperature)
from std_msgs.msg import Empty, String

try:
    from djitellopy import Tello
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing Python dependency 'djitellopy'. Install it with:\n"
        "  python3 -m pip install --user djitellopy\n"
        "or inside your venv, then rebuild/re-source your workspace."
    ) from e

from tello_msg.msg import TelloID, TelloStatus, TelloWifiConfig

# djitellopy logs every rc packet at INFO, which at 20 Hz drowns the console.
logging.getLogger("djitellopy").setLevel(logging.WARNING)

try:
    import cv2
except Exception:                                   # pragma: no cover
    cv2 = None

try:
    from cv_bridge import CvBridge
except Exception:                                   # pragma: no cover
    CvBridge = None

#: Standard gravity, m/s^2.
G0 = 9.80665
#: SDK accelerations are milli-g.
MILLI_G_TO_MPS2 = G0 / 1000.0
#: FRD (drone) -> FLU (ROS). A 180 deg rotation about the body x axis.
FRD_TO_FLU = numpy.diag([1.0, -1.0, -1.0])
DEG = math.pi / 180.0


class TelloNode:

    def __init__(self, node):
        self.node = node
        self._shutdown = threading.Event()

        # ------------------------- parameters --------------------------- #
        p = node.declare_parameter
        p('connect_timeout', 10.0)
        p('tello_ip', '192.168.10.1')
        p('tf_base', 'odom')
        p('tf_drone', 'base_link')
        p('camera_frame', 'camera_optical')
        p('tf_pub', False)
        p('camera_info_file', '')
        p('video_scale', 1.0)
        p('video_target_fps', 30.0)
        p('rc_rate_hz', 20.0)
        p('rc_timeout_sec', 0.35)
        p('telemetry_poll_hz', 50.0)
        p('speed_to_mps', 0.1)
        p('publish_bgr', True)
        p('video_stall_timeout', 3.0)
        p('video_auto_restart', True)

        g = lambda n: node.get_parameter(n).value
        self.connect_timeout = float(g('connect_timeout'))
        self.tello_ip = str(g('tello_ip'))
        self.tf_base = str(g('tf_base'))
        self.tf_drone = str(g('tf_drone'))
        self.camera_frame = str(g('camera_frame'))
        self.tf_pub = bool(g('tf_pub'))
        self.camera_info_file = str(g('camera_info_file'))
        self.video_scale = float(g('video_scale'))
        self.video_target_fps = float(g('video_target_fps'))
        self.rc_rate_hz = float(g('rc_rate_hz'))
        self.rc_timeout_sec = float(g('rc_timeout_sec'))
        self.telemetry_poll_hz = float(g('telemetry_poll_hz'))
        self.speed_to_mps = float(g('speed_to_mps'))
        self.publish_bgr = bool(g('publish_bgr'))
        self.video_stall_timeout = max(1.0, float(g('video_stall_timeout')))
        self.video_auto_restart = bool(g('video_auto_restart'))

        if not (0.0 < self.video_scale <= 1.0):
            node.get_logger().warn(
                f"video_scale={self.video_scale} out of (0, 1]; using 1.0")
            self.video_scale = 1.0
        self.video_target_fps = max(1.0, self.video_target_fps)
        self.rc_rate_hz = max(1.0, self.rc_rate_hz)
        self.rc_timeout_sec = max(0.05, self.rc_timeout_sec)
        self.telemetry_poll_hz = min(200.0, max(10.0, self.telemetry_poll_hz))

        if self.video_scale != 1.0 and cv2 is None:
            node.get_logger().warn(
                "video_scale < 1 requested but OpenCV is unavailable; falling "
                "back to stride subsampling, which ALIASES the image and "
                "measurably degrades feature detection. Install python3-opencv.")

        # ------------------------- camera info -------------------------- #
        if not self.camera_info_file:
            self.camera_info_file = get_package_share_directory('tello') + '/ost.yaml'
        with open(self.camera_info_file, 'r') as f:
            self.camera_info_raw = yaml.safe_load(f) or {}
        self._camera_info_cache: Optional[CameraInfo] = None

        # ------------------------- connect ------------------------------ #
        # The IP must go through the constructor: djitellopy binds the address
        # in __init__, so assigning Tello.TELLO_IP afterwards has no effect.
        # Likewise RESPONSE_TIMEOUT is captured as a default argument value at
        # import time, so it cannot be overridden by a class attribute either;
        # the timeout is passed per call instead.
        problem = check_network_path(self.tello_ip)
        if problem is not None:
            raise RuntimeError('Cannot reach the drone.\n  ' + problem)

        node.get_logger().info(f'Tello: connecting to {self.tello_ip}')
        try:
            self.tello = Tello(host=self.tello_ip)
        except TypeError:                            # very old djitellopy
            Tello.TELLO_IP = self.tello_ip
            self.tello = Tello()
        except OSError as e:
            # djitellopy binds UDP :8889 in Tello.__init__. EADDRINUSE here does
            # NOT mean the drone is unreachable -- it means another process on
            # this machine already owns the socket, almost always a previous
            # driver that outlived its launch. Say so, because the raw errno
            # sends people off debugging their WiFi for an hour.
            if e.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    'UDP port 8889 is already in use, so this driver cannot open '
                    'the Tello command socket.\n'
                    '  This is NOT a connectivity problem: the drone and your WiFi '
                    'are irrelevant here.\n'
                    '  A previous tello driver is still running. Find and stop it:\n'
                    "    ss -lunp | grep 8889\n"
                    "    pkill -f 'lib/tello/tello'\n"
                    '  Then relaunch.') from e
            raise
        # `connect_timeout` cannot be honoured the obvious way: djitellopy
        # binds RESPONSE_TIMEOUT as a *default argument value* at import time,
        # so assigning to the class attribute changes nothing. What is
        # adjustable is the retry count, which multiplies the fixed 7 s
        # per-attempt timeout -- so we translate the requested budget into a
        # retry count and say what we actually did.
        per_try = float(getattr(Tello, 'RESPONSE_TIMEOUT', 7))
        tries = max(1, int(round(self.connect_timeout / max(1e-6, per_try))))
        self.tello.retry_count = tries
        node.get_logger().info(
            f'Tello: response timeout is {per_try:.0f}s per attempt (fixed by '
            f'djitellopy); using {tries} attempt(s) for a ~{tries*per_try:.0f}s budget')
        try:
            self.tello.connect()
        except Exception as e:
            detail = str(e)
            if 'decode' in detail.lower():
                # The drone DID answer -- with bytes that are not valid UTF-8.
                # That is a different fault from silence and has different
                # causes, so it gets its own advice. Usually a stale datagram
                # left in the command socket by a previous session, or a second
                # client (the phone app) talking to the drone at the same time.
                raise RuntimeError(
                    f'The drone at {self.tello_ip} replied with data that is not '
                    'valid text, so the SDK handshake failed.\n'
                    '  This is NOT a network problem: the drone answered, but with '
                    'unexpected bytes.\n'
                    '  Most likely causes, in order:\n'
                    '    1. A previous session left the drone mid-stream. '
                    'POWER-CYCLE the Tello and retry.\n'
                    '    2. Another client (the Tello phone app) is connected. '
                    'Close it.\n'
                    '    3. A stale datagram is queued on UDP :8889 from a '
                    'crashed driver. Check with: ss -lunp | grep 8889\n'
                    f'  Original error: {detail}') from e
            raise RuntimeError(
                f'No response from the drone at {self.tello_ip}.\n'
                '  Check: the Tello is powered on (blinking LED), and this machine '
                'is joined to its WiFi access point (TELLO-XXXXXX).\n'
                '  On WSL2, confirm Windows is on the Tello AP -- WSL shares the '
                'host network.\n'
                f'  Original error: {detail}') from e
        node.get_logger().info('Tello: connected')

        self.tello.streamon()
        try:
            # with_queue=False keeps only the most recent decoded frame. A queue
            # would buffer frames and add latency on top of the ~200 ms the
            # WiFi link already costs, which is the opposite of what VIO needs.
            self._frame_read = self.tello.get_frame_read(with_queue=False)
        except TypeError:
            self._frame_read = self.tello.get_frame_read()

        # ------------------------- state -------------------------------- #
        self._bridge = CvBridge() if CvBridge is not None else None
        self._rc = (0, 0, 0, 0)
        self._rc_last_time = 0.0
        self._prev_state: Optional[dict] = None
        # Hold a REFERENCE to the last frame we published, not a hash of it.
        # djitellopy replaces BackgroundFrameRead.frame with a brand-new numpy
        # array on every successful decode, so object identity is an exact
        # new-frame test. Hashing sampled pixels (the previous approach) reports
        # false duplicates whenever the scene is static -- pointing the drone at
        # a wall was enough to make it discard most real frames -- and holding
        # the reference additionally stops CPython recycling the buffer address.
        self._prev_frame_obj = None
        self._resize_cache = None
        self._video_pub_count = 0
        self._video_win_count = 0
        self._video_win_start = 0.0
        self._video_last_log = 0.0
        self._duplicate_frames = 0
        self._last_new_frame_t = 0.0
        self._video_restarts = 0
        self._video_restart_pending = False
        # Exponential backoff between FAILED restart attempts. Each failed
        # attempt blocks the worker thread ~7 s inside streamoff (djitellopy's
        # fixed timeout); retrying immediately turns a WiFi dropout into a
        # continuous stream of 7 s worker stalls -- during which a queued land
        # command would wait behind them. 5 s -> 10 -> 20 -> 30 cap, reset on
        # the first success.
        self._restart_backoff = 5.0
        self._next_restart_t = 0.0
        self._identity = None
        self._wifi_snr = ''

        # Blocking SDK commands run here, never on an executor thread.
        self._cmd_q: "queue.Queue" = queue.Queue()
        self._cmd_thread = threading.Thread(target=self._command_worker,
                                            name='tello-cmd', daemon=True)
        self._cmd_thread.start()

        self.setup_publishers()
        self.setup_subscribers()

        # Callback groups: video decode and telemetry must not block each other,
        # and neither may delay the RC dead-man. Requires MultiThreadedExecutor.
        self.cg_video = MutuallyExclusiveCallbackGroup()
        self.cg_telemetry = MutuallyExclusiveCallbackGroup()
        self.cg_control = MutuallyExclusiveCallbackGroup()

        self._video_timer = node.create_timer(
            1.0 / self.video_target_fps, self._on_video_timer, callback_group=self.cg_video)
        self._telemetry_timer = node.create_timer(
            1.0 / self.telemetry_poll_hz, self._on_telemetry_timer,
            callback_group=self.cg_telemetry)
        self._slow_timer = node.create_timer(
            2.0, self._on_slow_timer, callback_group=self.cg_telemetry)
        self._rc_timer = node.create_timer(
            1.0 / self.rc_rate_hz, self._on_rc_timer, callback_group=self.cg_control)

        if self.tf_pub:
            self._static_tf = tf2_ros.StaticTransformBroadcaster(node)
            node.get_logger().warn(
                "tf_pub=true: this driver publishes only the static "
                f"{self.tf_drone} -> {self.camera_frame} edge. It never "
                f"publishes {self.tf_base} -> {self.tf_drone}, which belongs to "
                "the state estimator; two publishers on one TF edge is a broken "
                "tree.")
            self._publish_static_camera_tf()

        self._cmd_q.put(('identify', ()))
        node.get_logger().info('Tello: driver ready')

    # ------------------------------------------------------------------ #
    # ROS interface
    # ------------------------------------------------------------------ #

    def setup_publishers(self):
        n = self.node
        self.pub_image_raw = n.create_publisher(Image, 'image_raw', qos_profile_sensor_data)
        self.pub_camera_info = n.create_publisher(CameraInfo, 'camera_info',
                                                  qos_profile_sensor_data)
        self.pub_imu = n.create_publisher(Imu, 'imu', qos_profile_sensor_data)
        self.pub_odom = n.create_publisher(Odometry, 'odom', qos_profile_sensor_data)
        self.pub_tof = n.create_publisher(Range, 'tof', qos_profile_sensor_data)
        self.pub_status = n.create_publisher(TelloStatus, 'status', 1)
        self.pub_id = n.create_publisher(TelloID, 'id', 1)
        self.pub_battery = n.create_publisher(BatteryState, 'battery', 1)
        self.pub_temperature = n.create_publisher(Temperature, 'temperature', 1)

    def setup_subscribers(self):
        n = self.node
        n.create_subscription(Empty, 'emergency', self.cb_emergency, 1)
        n.create_subscription(Empty, 'takeoff', self.cb_takeoff, 1)
        n.create_subscription(Empty, 'land', self.cb_land, 1)
        n.create_subscription(Twist, 'control', self.cb_control, 1)
        n.create_subscription(Twist, 'cmd_vel', self.cb_cmd_vel, 1)
        n.create_subscription(String, 'flip', self.cb_flip, 1)
        n.create_subscription(TelloWifiConfig, 'wifi_config', self.cb_wifi_config, 1)

    # ------------------------------------------------------------------ #
    # blocking-command worker
    # ------------------------------------------------------------------ #

    def _command_worker(self):
        """Serialise every blocking SDK call onto one non-ROS thread.

        The Tello's command socket is single-threaded by protocol -- two
        outstanding commands corrupt each other's responses -- so a single
        worker is both the safe and the correct design.
        """
        while not self._shutdown.is_set():
            try:
                cmd, args = self._cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run_command(cmd, args)
            except Exception as e:
                self.node.get_logger().warn(f"{cmd} failed: {e}")
            finally:
                self._cmd_q.task_done()

    def _run_command(self, cmd, args):
        log = self.node.get_logger()
        if cmd == 'takeoff':
            log.info('takeoff')
            self.tello.takeoff()
        elif cmd == 'land':
            log.info('land')
            self.tello.land()
        elif cmd == 'emergency':
            self.tello.emergency()
        elif cmd == 'flip':
            self.tello.flip(args[0])
        elif cmd == 'wifi':
            self.tello.set_wifi_credentials(args[0], args[1])
        elif cmd == 'identify':
            # Static for the life of the connection: query once, not at 2 Hz.
            # Older firmware answers "unknown command: sdk?" instead of raising,
            # so the error arrives as a perfectly ordinary string. Publishing it
            # as a version number would be worse than admitting we do not know.
            def _clean(v):
                text = str(v).strip()
                return 'unsupported' if 'unknown command' in text.lower() else text

            sdk = _clean(self.tello.query_sdk_version())
            serial = _clean(self.tello.query_serial_number())
            self._identity = (sdk, serial)
            if sdk == 'unsupported' or serial == 'unsupported':
                log.info('Tello firmware does not support sdk?/sn? queries '
                         '(harmless: these are identification only)')
            else:
                log.info(f'Tello SDK {sdk} serial {serial}')
        elif cmd == 'wifi_snr':
            self._wifi_snr = str(self.tello.query_wifi_signal_noise_ratio())
        elif cmd == 'restart_video':
            # streamoff/streamon are blocking SDK round trips, which is exactly
            # why this runs on the worker thread and not in the video timer.
            restarted = False
            try:
                try:
                    self._frame_read.stop()
                except Exception:
                    pass
                self.tello.streamoff()
                time.sleep(0.5)
                self.tello.streamon()
                try:
                    self._frame_read = self.tello.get_frame_read(with_queue=False)
                except TypeError:
                    self._frame_read = self.tello.get_frame_read()
                self._prev_frame_obj = None
                self._last_new_frame_t = time.time()
                self._video_restarts += 1
                restarted = True
                self._restart_backoff = 5.0
                log.info(f'Video stream restarted (#{self._video_restarts})')
            finally:
                if not restarted:
                    self._next_restart_t = time.time() + self._restart_backoff
                    log.warn(
                        f'Video restart failed; next attempt in '
                        f'{self._restart_backoff:.0f}s (the drone is probably '
                        f'out of WiFi range or asleep)')
                    self._restart_backoff = min(30.0, self._restart_backoff * 2.0)
                self._video_restart_pending = False

    # ------------------------------------------------------------------ #
    # telemetry
    # ------------------------------------------------------------------ #

    def _on_telemetry_timer(self):
        """Poll the SDK state and publish only when the packet actually changed.

        djitellopy's receiver thread overwrites one dict, with no sequence
        number, so change detection is the only way to distinguish a fresh
        packet from a re-read of the previous one. Polling faster than the
        broadcast (50 Hz vs ~10 Hz) bounds the stamping error to one poll
        period instead of one telemetry period.
        """
        try:
            state = self.tello.get_current_state()
        except Exception:
            return
        if not state or state == self._prev_state:
            return
        self._prev_state = dict(state)
        stamp = self.node.get_clock().now().to_msg()

        def f(key, default=0.0):
            try:
                return float(state.get(key, default))
            except (TypeError, ValueError):
                return default

        accel_frd = numpy.array([f('agx'), f('agy'), f('agz')]) * MILLI_G_TO_MPS2
        accel_flu = FRD_TO_FLU @ accel_frd
        vel_frd = numpy.array([f('vgx'), f('vgy'), f('vgz')]) * self.speed_to_mps
        vel_flu = FRD_TO_FLU @ vel_frd
        roll_d, pitch_d, yaw_d = f('roll'), f('pitch'), f('yaw')
        q = self._attitude_quaternion(roll_d, pitch_d, yaw_d)

        self._publish_imu(stamp, accel_flu, q)
        self._publish_odom(stamp, vel_flu, q)
        self._publish_tof(stamp, f('tof'))
        self._publish_status(stamp, state, accel_frd, vel_frd, roll_d, pitch_d, yaw_d)

    @staticmethod
    def _attitude_quaternion(roll_deg, pitch_deg, yaw_deg):
        """SDK NED-referenced Euler angles -> ROS ENU-referenced quaternion.

        The SDK reports the FRD body attitude against a NED world frame; ROS
        wants FLU against ENU. Composing both frame flips negates pitch and yaw
        and leaves roll alone. The remaining constant 90 deg heading difference
        between NED-north and ENU-east is moot: the Tello has no usable
        magnetometer, so its yaw origin is wherever it was pointing at power-on.
        """
        roll = roll_deg * DEG
        pitch = -pitch_deg * DEG
        yaw = -yaw_deg * DEG
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        # Hamilton quaternion for R = Rz(yaw) Ry(pitch) Rx(roll), (x, y, z, w).
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    def _publish_imu(self, stamp, accel_flu, q):
        if self.pub_imu.get_subscription_count() == 0:
            return
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.tf_drone
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
        msg.linear_acceleration.x = float(accel_flu[0])
        msg.linear_acceleration.y = float(accel_flu[1])
        msg.linear_acceleration.z = float(accel_flu[2])

        # The Tello exposes NO gyroscope: the state packet carries fused
        # attitude only. ROS defines covariance[0] = -1 as "this field is not
        # measured"; leaving the field zero with a zero covariance instead --
        # as this driver used to -- advertises an *exact* measurement of zero
        # angular rate, which any consumer will believe and integrate.
        msg.angular_velocity_covariance[0] = -1.0

        # Orientation: roll/pitch are gravity-referenced and good to a couple of
        # degrees; yaw free-runs with no absolute reference, so it gets a much
        # larger variance. Values are placeholders until imu_calib is run.
        rp_var = (2.0 * DEG) ** 2
        yaw_var = (15.0 * DEG) ** 2
        msg.orientation_covariance = [rp_var, 0.0, 0.0,
                                      0.0, rp_var, 0.0,
                                      0.0, 0.0, yaw_var]
        a_var = 0.08 ** 2 * 10.0     # density^2 * rate, for the ~10 Hz stream
        msg.linear_acceleration_covariance = [a_var, 0.0, 0.0,
                                              0.0, a_var, 0.0,
                                              0.0, 0.0, a_var]
        self.pub_imu.publish(msg)

    def _publish_odom(self, stamp, vel_flu, q):
        if self.pub_odom.get_subscription_count() == 0:
            return
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.tf_base
        # REP-105: twist is expressed in child_frame_id. Leaving it empty makes
        # the message unusable to robot_localization, whose transform lookup
        # fails on an empty frame id and silently drops the measurement.
        msg.child_frame_id = self.tf_drone
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, \
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q
        # This driver has no position source. Advertise the pose position as
        # unavailable rather than publishing a confident zero at the origin,
        # which would fight any real estimator fusing the same topic.
        cov = [0.0] * 36
        cov[0] = -1.0
        msg.pose.covariance = cov
        msg.twist.twist.linear.x = float(vel_flu[0])
        msg.twist.twist.linear.y = float(vel_flu[1])
        msg.twist.twist.linear.z = float(vel_flu[2])
        tcov = [0.0] * 36
        v_var = 0.15 ** 2
        tcov[0], tcov[7], tcov[14] = v_var, v_var, v_var
        # Angular twist is not measured (there is no gyro). ROS only defines the
        # "unavailable" flag for element 0 of the whole array, and element 0 is
        # a real measurement here, so the honest encoding is a variance large
        # enough that any sane filter gives the angular block no weight.
        big = 1e6
        tcov[21], tcov[28], tcov[35] = big, big, big
        msg.twist.covariance = tcov
        self.pub_odom.publish(msg)

    def _publish_tof(self, stamp, tof_cm):
        if self.pub_tof.get_subscription_count() == 0:
            return
        msg = Range()
        msg.header.stamp = stamp
        msg.header.frame_id = self.tf_drone
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 0.44          # ~25 deg cone, VL53L0X-class sensor
        msg.min_range = 0.10
        msg.max_range = 1.20
        r = float(tof_cm) / 100.0
        # Out-of-range readings are signalled by +inf, per sensor_msgs/Range,
        # rather than being published as a plausible-looking number.
        msg.range = r if msg.min_range <= r <= msg.max_range else float('inf')
        self.pub_tof.publish(msg)

    def _publish_status(self, stamp, state, accel_frd, vel_frd, roll, pitch, yaw):
        def i(key):
            try:
                return int(float(state.get(key, 0)))
            except (TypeError, ValueError):
                return 0

        if self.pub_battery.get_subscription_count() > 0:
            m = BatteryState()
            m.header.stamp = stamp
            m.header.frame_id = self.tf_drone
            m.percentage = i('bat') / 100.0
            m.voltage = float('nan')          # not reported by the SDK
            m.current = float('nan')
            m.charge = float('nan')
            m.capacity = float('nan')
            m.design_capacity = 1.1
            m.present = True
            m.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
            m.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            self.pub_battery.publish(m)

        if self.pub_temperature.get_subscription_count() > 0:
            m = Temperature()
            m.header.stamp = stamp
            m.header.frame_id = self.tf_drone
            m.temperature = 0.5 * (i('templ') + i('temph'))
            m.variance = 0.0
            self.pub_temperature.publish(m)

        if self.pub_status.get_subscription_count() > 0:
            m = TelloStatus()
            m.header.stamp = stamp
            m.header.frame_id = self.tf_drone
            m.acceleration.x, m.acceleration.y, m.acceleration.z = [float(v) for v in accel_frd]
            m.speed.x, m.speed.y, m.speed.z = [float(v) for v in vel_frd]
            m.pitch, m.roll, m.yaw = int(pitch), int(roll), int(yaw)
            try:
                m.barometer = int(round(float(state.get('baro', 0.0)) * 100.0))
            except (TypeError, ValueError):
                m.barometer = 0
            m.distance_tof = i('tof')
            m.height = i('h')
            m.fligth_time = i('time')
            m.battery = max(0, min(255, i('bat')))
            m.highest_temperature = i('temph')
            m.lowest_temperature = i('templ')
            m.temperature = 0.5 * (i('templ') + i('temph'))
            m.wifi_snr = self._wifi_snr
            self.pub_status.publish(m)

    def _on_slow_timer(self):
        self._check_video_health()
        if self.pub_id.get_subscription_count() > 0 and self._identity is not None:
            m = TelloID()
            m.sdk_version, m.serial_number = self._identity
            self.pub_id.publish(m)
        if self.pub_camera_info.get_subscription_count() > 0:
            self.pub_camera_info.publish(self._camera_info_msg())
        # One query every ~10 s, on the worker thread, and only if the queue is
        # otherwise idle -- this is diagnostics, not something worth delaying a
        # land command for.
        if self.pub_status.get_subscription_count() > 0 and self._cmd_q.empty():
            if int(time.time()) % 10 == 0:
                self._cmd_q.put(('wifi_snr', ()))

    # ------------------------------------------------------------------ #
    # video
    # ------------------------------------------------------------------ #

    def _on_video_timer(self):
        if self.pub_image_raw.get_subscription_count() == 0 or self._frame_read is None:
            return
        frame = self._frame_read.frame
        if frame is None:
            return

        # Exact new-frame test: djitellopy hands out a fresh array object per
        # decode, so `is` is True only when no new frame has arrived. Publishing
        # an unchanged frame with a fresh timestamp fabricates motion-free
        # evidence for the estimator and hides a dead decoder behind a
        # healthy-looking topic.
        if frame is self._prev_frame_obj:
            self._duplicate_frames += 1
            return
        self._prev_frame_obj = frame

        now = time.time()
        self._last_new_frame_t = now

        img = numpy.asarray(frame)
        if self.video_scale != 1.0:
            img = self._resize(img, self.video_scale)

        # djitellopy decodes to RGB (`np.array(frame.to_image())`), not BGR.
        # Publishing it labelled 'bgr8' swaps red and blue for every consumer.
        if self.publish_bgr:
            img = img[:, :, ::-1] if img.ndim == 3 and img.shape[2] == 3 else img
            encoding = 'bgr8'
        else:
            encoding = 'rgb8'

        msg = self._to_imgmsg(img, encoding)
        if msg is None:
            return
        msg.header.frame_id = self.camera_frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.pub_image_raw.publish(msg)

        self._video_pub_count += 1
        self._video_win_count += 1
        if self._video_win_start == 0.0:
            self._video_win_start = now
            self._video_last_log = now
            self.node.get_logger().info('Video: first frame published on image_raw')
            return

        # Report the CURRENT rate over the last window. A cumulative average
        # hides a stream that has degraded, and divides by ~0 on the first frame.
        if now - self._video_last_log > 10.0:
            win = max(1e-3, now - self._video_win_start)
            self.node.get_logger().info(
                f'Video: {self._video_win_count / win:.1f} FPS now '
                f'({self._video_pub_count} total, {self._duplicate_frames} idle polls, '
                f'{self._video_restarts} stream restarts)')
            self._video_win_start = now
            self._video_win_count = 0
            self._video_last_log = now

    def _check_video_health(self):
        """Detect a dead decoder and ask the worker thread to restart the stream.

        djitellopy's decode loop runs in its own thread and dies outright on
        `av.error.OSError: [Errno 5]` when the UDP stream stalls -- which the
        Tello does routinely on a congested link. Nothing in djitellopy notices:
        `frame` keeps returning the last decoded image forever, so a naive
        driver republishes a frozen picture indefinitely. Both signals are
        checked because either can happen alone: the thread can die while a
        recent frame is still cached, and the stream can stall for seconds
        without the thread exiting.
        """
        if not self.video_auto_restart or self._frame_read is None:
            return
        if self._video_restart_pending:
            return

        worker = getattr(self._frame_read, 'worker', None)
        decoder_dead = worker is not None and not worker.is_alive()
        stalled = (self._last_new_frame_t > 0.0
                   and (time.time() - self._last_new_frame_t) > self.video_stall_timeout)

        if (decoder_dead or stalled) and time.time() >= self._next_restart_t:
            why = 'decoder thread exited' if decoder_dead else \
                f'no new frame for {self.video_stall_timeout:.0f}s'
            self.node.get_logger().warn(f'Video stalled ({why}); restarting the stream')
            self._video_restart_pending = True
            self._cmd_q.put(('restart_video', ()))

    def _to_imgmsg(self, img, encoding):
        if self._bridge is not None:
            try:
                return self._bridge.cv2_to_imgmsg(numpy.ascontiguousarray(img),
                                                  encoding=encoding)
            except Exception:
                pass
        if img.ndim != 3 or img.shape[2] != 3:
            return None
        img = numpy.ascontiguousarray(img, dtype=numpy.uint8)
        h, w, _ = img.shape
        m = Image()
        m.height, m.width = int(h), int(w)
        m.encoding = encoding
        m.is_bigendian = False
        m.step = int(w * 3)
        m.data = img.tobytes()
        return m

    def _resize(self, img, scale):
        """Downscale with proper area averaging.

        INTER_AREA, not subsampling. Dropping every other pixel aliases
        high-frequency detail into the low frequencies, which is precisely the
        content a corner detector keys on -- an aliased half-resolution image
        yields visibly worse and less repeatable features than a properly
        filtered one. The stride path below exists only as a no-OpenCV fallback
        and is documented as degraded.
        """
        if cv2 is not None:
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        h, w = int(img.shape[0]), int(img.shape[1])
        cache = self._resize_cache
        if cache is None or cache[0] != h or cache[1] != w or abs(cache[2] - scale) > 1e-9:
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            ys = numpy.linspace(0, h - 1, nh).astype(numpy.int32)
            xs = numpy.linspace(0, w - 1, nw).astype(numpy.int32)
            self._resize_cache = (h, w, scale, ys, xs)
        else:
            ys, xs = cache[3], cache[4]
        return img[ys][:, xs]

    def _camera_info_msg(self) -> CameraInfo:
        """Build CameraInfo, **rescaled to the published image size**.

        Downscaling the image without scaling the intrinsics is the single most
        damaging silent bug available here: fx, fy, cx and cy are all in pixels,
        so at video_scale=0.5 the unscaled matrix claims twice the focal length.
        Every bearing computed from it is then wrong by a factor of two, which
        SLAM absorbs as a scale error and never recovers from.

        The pixel-centre convention matters at this precision: pixel *i* spans
        [i, i+1) with centre i+0.5, so ``c' = (c + 0.5) s - 0.5``, not ``c s``.
        Distortion coefficients are dimensionless (defined in normalised
        coordinates) and must NOT be scaled.
        """
        if self._camera_info_cache is not None:
            return self._camera_info_cache

        info = self.camera_info_raw
        s = self.video_scale
        msg = CameraInfo()
        msg.header.frame_id = self.camera_frame
        w = int(info.get('image_width', 0))
        h = int(info.get('image_height', 0))
        msg.width = max(1, int(round(w * s))) if w else 0
        msg.height = max(1, int(round(h * s))) if h else 0
        msg.distortion_model = str(info.get('distortion_model', 'plumb_bob'))
        msg.d = [float(x) for x in info.get('distortion_coefficients', {}).get('data', [])]

        k = [float(x) for x in info.get('camera_matrix', {}).get('data', [])]
        if len(k) == 9:
            k = list(k)
            k[0] *= s
            k[4] *= s
            k[2] = (k[2] + 0.5) * s - 0.5
            k[5] = (k[5] + 0.5) * s - 0.5
            msg.k = k

        r = [float(x) for x in info.get('rectification_matrix', {}).get('data', [])]
        if len(r) == 9:
            msg.r = r

        pr = [float(x) for x in info.get('projection_matrix', {}).get('data', [])]
        if len(pr) == 12:
            pr = list(pr)
            pr[0] *= s
            pr[5] *= s
            pr[2] = (pr[2] + 0.5) * s - 0.5
            pr[6] = (pr[6] + 0.5) * s - 0.5
            pr[3] *= s
            pr[7] *= s
            msg.p = pr

        self._camera_info_cache = msg
        return msg

    def _publish_static_camera_tf(self):
        """Nominal base_link -> camera_optical. Overridden by tello_vio's calibrated one."""
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = self.tf_drone
        t.child_frame_id = self.camera_frame
        t.transform.translation.x = 0.03
        t.transform.translation.y = 0.0
        t.transform.translation.z = -0.01
        # Rz(-90) Rx(-90): optical z forward, x right, y down, on an FLU body.
        t.transform.rotation.x = -0.5
        t.transform.rotation.y = 0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w = 0.5
        self._static_tf.sendTransform(t)

    # ------------------------------------------------------------------ #
    # control
    # ------------------------------------------------------------------ #

    def cb_emergency(self, msg):
        """Cut the motors immediately.

        Sent twice deliberately: once as a fire-and-forget datagram straight
        from this callback so it cannot queue behind a stalled ``land()``, and
        once through the worker so it is retried if the first packet is lost.
        For a motor-cut command, a duplicate is harmless and a delay is not.
        """
        try:
            self.tello.send_command_without_return('emergency')
        except Exception:
            pass
        self._cmd_q.put(('emergency', ()))
        self._rc = (0, 0, 0, 0)

    def cb_takeoff(self, msg):
        self._cmd_q.put(('takeoff', ()))

    def cb_land(self, msg):
        self._rc = (0, 0, 0, 0)
        self._cmd_q.put(('land', ()))

    def cb_flip(self, msg):
        self._cmd_q.put(('flip', (msg.data,)))

    def cb_wifi_config(self, msg):
        self._cmd_q.put(('wifi', (msg.ssid, msg.password)))

    def cb_control(self, msg):
        """Legacy stick-axis control: linear.x = lateral, linear.y = forward.

        This mapping is NOT REP-103 (which puts forward on x) but it is the
        convention ``tello_control``'s keyboard node has always published, and
        silently changing it would invert an operator's controls mid-flight.
        Use ``/cmd_vel`` for the standards-compliant interface.
        """
        self._set_rc(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)

    def cb_cmd_vel(self, msg):
        """REP-103 velocity command: +x forward, +y left, +z up, +yaw counter-clockwise.

        Values are normalised stick units in [-1, 1] and scaled to the SDK's
        [-100, 100]. The sign flips on y and yaw convert ROS FLU/counter-
        clockwise into the Tello's right-positive/clockwise-positive channels.
        """
        self._set_rc(-msg.linear.y * 100.0, msg.linear.x * 100.0,
                     msg.linear.z * 100.0, -msg.angular.z * 100.0)

    def _set_rc(self, lr, fb, ud, yaw):
        clamp = lambda v: max(-100, min(100, int(v)))
        self._rc = (clamp(lr), clamp(fb), clamp(ud), clamp(yaw))
        self._rc_last_time = time.time()

    def _on_rc_timer(self):
        """Send RC at a fixed rate, zeroing on a stale command (dead-man).

        Two jobs at once: the Tello latches the last rc setpoint indefinitely,
        so a lost publisher would leave the drone flying, and the SDK auto-lands
        if it hears nothing for 15 s. A steady stream of (possibly zero) rc
        packets solves both.
        """
        rc = self._rc
        if time.time() - self._rc_last_time > self.rc_timeout_sec:
            rc = (0, 0, 0, 0)
            self._rc = rc
        try:
            self.tello.send_rc_control(*rc)
        except Exception:
            pass

    # ------------------------------------------------------------------ #

    def stop(self):
        self._shutdown.set()
        for fn in (lambda: self.tello.send_rc_control(0, 0, 0, 0),
                   lambda: self.tello.streamoff(),
                   lambda: self.tello.end()):
            try:
                fn()
            except Exception:
                pass



def _local_route_to(host: str):
    """Return the local source address the OS would use to reach ``host``.

    Uses a *connected* UDP socket, which performs a route lookup without
    sending a single packet -- so this is safe to call before the drone is
    known to exist.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.settimeout(0.3)
        probe.connect((host, 9))
        return probe.getsockname()[0]
    except Exception:
        return None
    finally:
        probe.close()


def check_network_path(host: str):
    """Pre-flight: are we plausibly on the drone's network at all?

    The Tello is its own access point handing out 192.168.10.x. If the machine
    sits on some other subnet, packets to 192.168.10.1 get routed upstream to
    whatever gateway is configured, and *something out there may answer* -- which
    surfaces as a baffling 'response decode error' rather than a timeout.
    Diagnosing that from the SDK error alone costs an hour, so check first.

    Returns ``None`` when the path looks fine, otherwise a human-readable
    explanation.
    """
    local = _local_route_to(host)
    if local is None:
        return (f'No route to {host} at all. This machine cannot reach the drone.')

    host_net = host.rsplit('.', 1)[0]
    local_net = local.rsplit('.', 1)[0]
    if host_net == local_net:
        return None

    return (
        f'This machine is on {local}, but the drone is expected at {host}.\n'
        f'  Those are different subnets, so packets to {host} are being routed to\n'
        '  your normal gateway instead of the drone. If anything out there answers,\n'
        "  you get a confusing 'decode error' rather than a clean timeout.\n"
        f'  FIX: join this machine to the Tello access point (TELLO-XXXXXX). You\n'
        f'  should then have a {host_net}.x address.\n'
        '  On WSL2 the WiFi belongs to WINDOWS -- connect Windows to the Tello AP.')


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('tello')
    drone = None
    try:
        drone = TelloNode(node)
        # MultiThreadedExecutor is mandatory here: the callback groups above
        # exist precisely so video, telemetry and the RC dead-man run
        # concurrently. Under the default single-threaded executor they would
        # serialise and the dead-man's timing guarantee would be lost.
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        # Our own pre-flight diagnostics: the message IS the useful output, so
        # do not bury it under a traceback the user has to read upwards.
        for line in str(e).splitlines():
            node.get_logger().error(line)
        return
    except Exception as e:
        node.get_logger().error(f'Tello driver failed: {e}')
        raise
    finally:
        if drone is not None:
            drone.stop()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
