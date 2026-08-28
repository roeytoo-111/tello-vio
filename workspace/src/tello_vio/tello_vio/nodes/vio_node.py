#!/usr/bin/env python3
"""``tello_vio`` main fusion node: visual-inertial odometry for a stock Tello.

Pipeline
--------
::

    /image_raw  ──► VoFrontend (KLT + 2-view geometry) ──┐
                                                          ├──► ErrorStateKF ──► /vio/odom
    /imu, /status, /tof ──► TelloImuSurrogate + ZUPT ─────┘                  └─► TF odom->base_link

Three architectural decisions dominate this file, and each is a response to a
concrete property of this drone rather than a stylistic preference.

**1. The estimator runs on the image clock, not the wall clock.**
Tello video arrives 150-350 ms behind reality; telemetry arrives in ~10 ms.
Fusing them as they land would apply visual measurements to a state that has
already moved on -- an out-of-sequence measurement, which a plain EKF handles by
being wrong. So telemetry is *buffered*, and the filter is advanced only up to
each image's (offset-corrected) capture time. The published odometry is
therefore deliberately ~250 ms old but self-consistent. A separate
``/vio/pose_predicted`` topic forward-propagates to now for anyone flying the
aircraft, clearly labelled as a prediction. Silently mixing the two is how you
get a controller chasing a stale pose.

**2. Two callback groups on a multi-threaded executor.**
The image callback runs feature detection and RANSAC -- tens of milliseconds.
The telemetry callbacks must not queue behind it. Putting them in separate
:class:`MutuallyExclusiveCallbackGroup` s and spinning a
:class:`MultiThreadedExecutor` lets them run concurrently, while the
mutual-exclusivity *within* each group means no callback ever races itself. The
estimator itself is guarded by one lock, which is the only shared mutable state.

**3. The driver's frames are separated.**
IMU data is in ``base_link``, images are in ``camera_optical``, and the fixed
transform between them is published as a static TF from the calibrated
extrinsic. Publishing both in one frame -- as the stock driver did -- makes the
camera-IMU extrinsic literally inexpressible in the system.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, Range
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from .. import lie
from ..eskf import ErrorStateKF, EskfConfig
from ..frontend import FrontendConfig, VoFrontend
from ..ros_utils import (camera_from_body_rotation, flatten_cov6, make_transform,
                         quat_from_ros, quat_to_ros, sec_to_stamp, stamp_to_sec,
                         vec3, vec3_from)
from ..tello_model import (StationarityDetector, TelloImuSurrogate, TelloNoise,
                           TelloUnits)

try:                                    # tello_msg is optional at runtime
    from tello_msg.msg import TelloStatus
    HAVE_TELLO_MSG = True
except Exception:                       # pragma: no cover - depends on workspace
    TelloStatus = None
    HAVE_TELLO_MSG = False

try:
    from cv_bridge import CvBridge
except Exception:                       # pragma: no cover
    CvBridge = None


class TelemetrySample:
    """One buffered telemetry epoch, already in SI + ROS FLU."""

    __slots__ = ("t", "accel", "quat", "gyro", "vel_body", "baro", "tof", "stationary")

    def __init__(self, t, accel, quat, gyro, vel_body, baro, tof, stationary):
        self.t = t
        self.accel = accel
        self.quat = quat
        self.gyro = gyro
        self.vel_body = vel_body
        self.baro = baro
        self.tof = tof
        self.stationary = stationary


class VioNode(Node):

    def __init__(self):
        super().__init__("tello_vio")

        # ---------------- parameters -------------------------------------
        p = self.declare_parameter
        p("image_topic", "/image_raw")
        p("camera_info_topic", "/camera_info")
        p("imu_topic", "/imu")
        p("status_topic", "/status")
        p("tof_topic", "/tof")

        p("odom_frame", "odom")
        p("body_frame", "base_link")
        p("camera_frame", "camera_optical")
        p("publish_tf", True)

        # Camera-IMU extrinsic. Defaults to the nominal forward-facing mount;
        # override from the hand-eye calibration (camera_imu_calib).
        p("camera_tilt_deg", 0.0)
        p("extrinsic_rpy_deg", [0.0, 0.0, 0.0])   # extra rotation on top of nominal
        p("extrinsic_xyz_m", [0.03, 0.0, -0.01])  # camera position in body frame

        # Camera-IMU time offset: t_image_corrected = t_image_header - time_offset.
        p("time_offset_s", 0.0)

        # Sensor scaling (see tello_model.TelloUnits for why these are knobs).
        p("speed_to_mps", 0.1)
        p("use_body_velocity", True)
        p("use_barometer", True)
        p("use_tof", True)
        p("use_attitude", True)
        p("use_zupt", True)

        p("accel_noise_density", 0.08)
        p("gyro_noise_density", 0.03)
        p("accel_bias_rw", 1.0e-3)
        p("gyro_bias_rw", 1.0e-4)
        p("att_rp_std_deg", 2.0)
        p("att_yaw_std_deg", 15.0)
        p("vel_body_std", 0.15)
        p("baro_std", 0.30)
        p("tof_std", 0.03)

        p("vo_work_width", 480)
        p("vo_max_features", 180)
        p("vo_min_features", 90)
        p("vo_kf_parallax_px", 12.0)
        p("vo_ransac_px", 1.5)
        p("vo_base_rot_std", 0.033)
        p("vo_base_dir_std", 0.128)
        p("vo_enable", True)

        p("publish_debug_image", True)
        p("publish_path", True)
        p("path_max_poses", 2000)
        p("telemetry_buffer_s", 3.0)
        p("diagnostics_hz", 2.0)

        g = lambda n: self.get_parameter(n).value

        self.odom_frame = str(g("odom_frame"))
        self.body_frame = str(g("body_frame"))
        self.camera_frame = str(g("camera_frame"))
        self.publish_tf = bool(g("publish_tf"))
        self.time_offset = float(g("time_offset_s"))
        self.units = TelloUnits(speed_to_mps=float(g("speed_to_mps")))
        self.noise = TelloNoise(
            accel_noise_density=float(g("accel_noise_density")),
            gyro_noise_density=float(g("gyro_noise_density")),
            accel_bias_rw=float(g("accel_bias_rw")),
            gyro_bias_rw=float(g("gyro_bias_rw")),
            att_rp_std=np.deg2rad(float(g("att_rp_std_deg"))),
            att_yaw_std=np.deg2rad(float(g("att_yaw_std_deg"))),
            vel_body_std=float(g("vel_body_std")),
            baro_std=float(g("baro_std")),
            tof_std=float(g("tof_std")),
        )

        # ---------------- extrinsic --------------------------------------
        rpy = [float(x) for x in g("extrinsic_rpy_deg")]
        self.R_BC = camera_from_body_rotation(float(g("camera_tilt_deg")))
        if any(abs(v) > 1e-9 for v in rpy):
            self.R_BC = lie.euler_zyx_to_rot(
                np.deg2rad(rpy[2]), np.deg2rad(rpy[1]), np.deg2rad(rpy[0])) @ self.R_BC
        self.p_BC = np.array([float(x) for x in g("extrinsic_xyz_m")])

        # ---------------- estimator --------------------------------------
        self.kf = ErrorStateKF(EskfConfig(
            accel_noise_density=self.noise.accel_noise_density,
            gyro_noise_density=self.noise.gyro_noise_density,
            accel_bias_rw=self.noise.accel_bias_rw,
            gyro_bias_rw=self.noise.gyro_bias_rw,
        ))
        self.lock = threading.Lock()
        self.surrogate = TelloImuSurrogate()
        self.stationary = StationarityDetector(self.noise)

        self.buffer: deque = deque()
        self.buffer_span = float(g("telemetry_buffer_s"))
        self.last_fused_t = None
        self.have_camera_info = False
        self.frontend: VoFrontend | None = None
        self.vo_enable = bool(g("vo_enable"))
        self.vo_cfg = FrontendConfig(
            work_width=int(g("vo_work_width")),
            max_features=int(g("vo_max_features")),
            min_features=int(g("vo_min_features")),
            kf_min_parallax_px=float(g("vo_kf_parallax_px")),
            ransac_px=float(g("vo_ransac_px")),
        )
        self.vo_base_rot_std = float(g("vo_base_rot_std"))
        self.vo_base_dir_std = float(g("vo_base_dir_std"))
        self.bridge = CvBridge() if CvBridge is not None else None

        self._path = Path()
        self._path.header.frame_id = self.odom_frame
        self._path_max = int(g("path_max_poses"))
        self._n_images = 0
        self._n_visual = 0
        self._n_telemetry = 0
        self._last_vo_cost_ms = 0.0
        self._last_meas = None

        # ---------------- ROS interface ----------------------------------
        # Separate groups: heavy vision must never delay telemetry ingest.
        self.cb_vision = MutuallyExclusiveCallbackGroup()
        self.cb_telemetry = MutuallyExclusiveCallbackGroup()

        self.pub_odom = self.create_publisher(Odometry, "~/odom", 10)
        self.pub_pose_pred = self.create_publisher(PoseWithCovarianceStamped,
                                                   "~/pose_predicted", 10)
        self.pub_path = self.create_publisher(Path, "~/path", 1)
        self.pub_debug = self.create_publisher(Image, "~/debug_image",
                                               qos_profile_sensor_data)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_extrinsic()

        self.create_subscription(Imu, str(g("imu_topic")), self.on_imu,
                                 qos_profile_sensor_data, callback_group=self.cb_telemetry)
        self.create_subscription(Range, str(g("tof_topic")), self.on_tof,
                                 qos_profile_sensor_data, callback_group=self.cb_telemetry)
        if HAVE_TELLO_MSG:
            self.create_subscription(TelloStatus, str(g("status_topic")), self.on_status,
                                     10, callback_group=self.cb_telemetry)
        else:
            self.get_logger().warn(
                "tello_msg not available: body-velocity and barometer updates are "
                "disabled, so the estimate will not be metric. Build tello_msg.")

        self.create_subscription(CameraInfo, str(g("camera_info_topic")),
                                 self.on_camera_info, qos_profile_sensor_data,
                                 callback_group=self.cb_vision)
        self.create_subscription(Image, str(g("image_topic")), self.on_image,
                                 qos_profile_sensor_data, callback_group=self.cb_vision)

        self.create_timer(1.0 / max(0.2, float(g("diagnostics_hz"))), self.on_diagnostics,
                          callback_group=self.cb_telemetry)
        self.create_service(Trigger, "~/reset", self.on_reset,
                            callback_group=self.cb_telemetry)

        self._latest_tof = None
        self._latest_status = None

        self.get_logger().info(
            "tello_vio ready | frames %s -> %s -> %s | t_d=%.3fs | VO %s"
            % (self.odom_frame, self.body_frame, self.camera_frame,
               self.time_offset, "on" if self.vo_enable else "off"))

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #

    def _publish_static_extrinsic(self) -> None:
        """Publish base_link -> camera_optical once, from the calibration."""
        q = lie.rot_to_quat(self.R_BC)
        tf = make_transform(self.body_frame, self.camera_frame,
                            self.get_clock().now().to_msg(), self.p_BC, q)
        self.static_tf.sendTransform(tf)

    def on_camera_info(self, msg: CameraInfo) -> None:
        if self.have_camera_info:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 0.0:
            self.get_logger().warn("CameraInfo has fx <= 0; ignoring")
            return
        D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
        self.frontend = VoFrontend(K, D, self.vo_cfg)
        self.have_camera_info = True
        self.get_logger().info(
            f"camera_info: {msg.width}x{msg.height} fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
            f"cx={K[0,2]:.1f} cy={K[1,2]:.1f} model={msg.distortion_model}")
        if msg.width and abs(K[0, 2] / msg.width - 0.5) > 0.15:
            self.get_logger().warn(
                "principal point is far from the image centre -- if you changed "
                "video_scale, make sure camera_info was rescaled with it")

    # ------------------------------------------------------------------ #
    # telemetry ingest
    # ------------------------------------------------------------------ #

    def on_status(self, msg) -> None:
        self._latest_status = msg

    def on_tof(self, msg: Range) -> None:
        r = float(msg.range)
        # Range messages use +inf / -inf for "out of range" -- honour that
        # rather than feeding an infinity into the filter.
        if not np.isfinite(r) or r < msg.min_range or r > msg.max_range:
            self._latest_tof = None
        else:
            self._latest_tof = r

    def on_imu(self, msg: Imu) -> None:
        """The telemetry heartbeat: one buffered sample per IMU message."""
        t = stamp_to_sec(msg.header.stamp)
        accel = vec3_from(msg.linear_acceleration)
        quat = quat_from_ros(msg.orientation)

        # Angular rate: prefer a real gyro if the driver ever supplies one
        # (covariance[0] = -1 is the ROS marker for "not measured"), otherwise
        # differentiate the attitude on the manifold.
        if msg.angular_velocity_covariance[0] >= 0.0:
            gyro = vec3_from(msg.angular_velocity)
        else:
            gyro = self.surrogate.update(t, quat)

        st = self._latest_status
        vel_body = np.zeros(3)
        baro = None
        if st is not None:
            s = self.units.speed_to_mps
            # TelloStatus carries the drone's FRD axes; convert to FLU.
            vel_body = np.array([st.speed.x * s, -st.speed.y * s, -st.speed.z * s])
            baro = float(st.barometer) / 100.0

        still = self.stationary.update(accel, vel_body, gyro)

        self.buffer.append(TelemetrySample(t, accel, quat, gyro, vel_body, baro,
                                           self._latest_tof, still))
        while self.buffer and (t - self.buffer[0].t) > self.buffer_span:
            self.buffer.popleft()
        self._n_telemetry += 1

        with self.lock:
            if not self.kf.initialised and baro is not None:
                self.kf.initialise(t, quat, baro)
                self.last_fused_t = t
                self.get_logger().info("estimator initialised")
            elif not self.kf.initialised and not HAVE_TELLO_MSG:
                self.kf.initialise(t, quat, 0.0)
                self.last_fused_t = t

        # Vision is what drives fusion forward (see the module docstring). With
        # VO disabled or no camera yet, fuse telemetry directly so the node is
        # still useful as an attitude/height estimator.
        if not self.vo_enable or not self.have_camera_info:
            with self.lock:
                self._advance_to(t)
            self._publish(t)
        else:
            self._publish_prediction(t)

    # ------------------------------------------------------------------ #
    # fusion
    # ------------------------------------------------------------------ #

    def _advance_to(self, t_target: float) -> None:
        """Propagate + apply telemetry updates up to ``t_target``. Caller holds the lock."""
        if not self.kf.initialised or self.last_fused_t is None:
            return
        gp = self.get_parameter
        use_vel = bool(gp("use_body_velocity").value)
        use_baro = bool(gp("use_barometer").value)
        use_tof = bool(gp("use_tof").value)
        use_att = bool(gp("use_attitude").value)
        use_zupt = bool(gp("use_zupt").value)

        for s in list(self.buffer):
            if s.t <= self.last_fused_t or s.t > t_target:
                continue
            dt = s.t - self.last_fused_t
            self.kf.propagate(s.accel, s.gyro, dt)
            self.last_fused_t = s.t

            if use_att:
                self.kf.update_attitude(s.quat, self.noise.att_rp_std,
                                        self.noise.att_yaw_std)
            if use_zupt and s.stationary:
                self.kf.update_zero_velocity(0.02)
                if s.gyro is not None:
                    self.kf.update_zero_angular_rate(s.gyro, 0.02)
            elif use_vel:
                # Optical-flow velocity degrades with height: the flow sensor's
                # metric conversion divides by range, so its error grows with it.
                std = self.noise.vel_body_std + \
                    self.noise.vel_body_std_per_m * max(0.0, self.kf.p[2])
                self.kf.update_body_velocity(s.vel_body, std)
            if use_baro and s.baro is not None:
                self.kf.update_barometer(s.baro, self.noise.baro_std)
            if use_tof and s.tof is not None:
                self.kf.update_tof(s.tof, self.noise.tof_std)

    def on_image(self, msg: Image) -> None:
        if not self.have_camera_info or self.frontend is None or not self.vo_enable:
            return
        img = self._to_cv(msg)
        if img is None:
            return

        # Correct the header stamp for the measured camera-IMU offset. The
        # driver stamps at decode time; the true capture instant is earlier.
        t_img = stamp_to_sec(msg.header.stamp) - self.time_offset
        self._n_images += 1

        with self.lock:
            if not self.kf.initialised:
                return
            # Rotation prior for the keyframe trigger, from the current attitude
            # relative to the cloned keyframe attitude. Free, and it is what
            # makes the trigger measure parallax rather than total pixel flow.
            R_b1b2 = lie.quat_to_rot(self.kf.q_c).T @ self.kf.R
            R_prior = self.R_BC.T @ R_b1b2 @ self.R_BC

        meas = self.frontend.process(img, t_img, R_pred_c1c2=R_prior)

        with self.lock:
            self._advance_to(t_img)
            if meas is not None:
                self._last_meas = meas
                self._n_visual += 1
                self._last_vo_cost_ms = meas.cost_ms
                rot_std, dir_std = meas.noise_std(self.vo_base_rot_std,
                                                  self.vo_base_dir_std)
                self.kf.update_visual_relative(
                    meas.R_c1c2, meas.t_dir_c1, self.R_BC, self.p_BC,
                    rot_std, dir_std, use_translation=meas.translation_reliable)
                # Re-clone *after* the update: the new keyframe's reference pose
                # is the corrected one, not the pre-update guess.
                self.kf.clone()

        self._publish(t_img)
        if bool(self.get_parameter("publish_debug_image").value) and \
                self.pub_debug.get_subscription_count() > 0:
            self._publish_debug(img, msg.header.stamp)

    def _to_cv(self, msg: Image):
        try:
            if self.bridge is not None:
                enc = "mono8" if msg.encoding in ("mono8", "8UC1") else "bgr8"
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding=enc)
            # Minimal fallback so the node runs without cv_bridge installed.
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("mono8", "8UC1"):
                return buf.reshape(msg.height, msg.step)[:, :msg.width]
            ch = 3
            return buf.reshape(msg.height, msg.step // ch, ch)[:, :msg.width]
        except Exception as e:                      # pragma: no cover
            self.get_logger().warn(f"image conversion failed: {e}", throttle_duration_sec=5.0)
            return None

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #

    def _publish(self, t: float) -> None:
        with self.lock:
            if not self.kf.initialised:
                return
            p, v, q = self.kf.p.copy(), self.kf.v.copy(), self.kf.q.copy()
            C = self.kf.pose_covariance_6x6()
            Cv = self.kf.P[3:6, 3:6].copy()
        stamp = sec_to_stamp(t)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.body_frame     # twist is in the body frame
        odom.pose.pose.position.x = float(p[0])
        odom.pose.pose.position.y = float(p[1])
        odom.pose.pose.position.z = float(p[2])
        odom.pose.pose.orientation = quat_to_ros(q)
        odom.pose.covariance = flatten_cov6(C)
        R = lie.quat_to_rot(q)
        odom.twist.twist.linear = vec3(R.T @ v)   # body frame, per REP-105
        tw = np.zeros((6, 6))
        tw[0:3, 0:3] = R.T @ Cv @ R
        tw[3:6, 3:6] = np.eye(3) * (self.noise.gyro_noise_density ** 2)
        odom.twist.covariance = flatten_cov6(tw)
        self.pub_odom.publish(odom)

        if self.publish_tf:
            self.tf_broadcaster.sendTransform(
                make_transform(self.odom_frame, self.body_frame, stamp, p, q))

        if bool(self.get_parameter("publish_path").value) and \
                self.pub_path.get_subscription_count() > 0:
            ps = PoseStamped()
            ps.header = odom.header
            ps.pose = odom.pose.pose
            self._path.poses.append(ps)
            if len(self._path.poses) > self._path_max:
                del self._path.poses[:len(self._path.poses) - self._path_max]
            self._path.header.stamp = stamp
            self.pub_path.publish(self._path)

    def _publish_prediction(self, t_now: float) -> None:
        """Forward-propagate the fused state to ``t_now`` for control consumers.

        This is explicitly *not* the odometry output: it is an open-loop
        extrapolation across the video latency, so its covariance is inflated by
        the process noise over that interval. Publishing it on its own topic,
        with honest covariance, lets a controller use it while making it obvious
        that it is not a measurement.
        """
        with self.lock:
            if not self.kf.initialised or self.last_fused_t is None:
                return
            dt = max(0.0, t_now - self.last_fused_t)
            p = self.kf.p + self.kf.v * dt
            q = self.kf.q.copy()
            C = self.kf.pose_covariance_6x6()
            Cv = self.kf.P[3:6, 3:6].copy()
        C = C.copy()
        C[0:3, 0:3] += Cv * dt * dt + np.eye(3) * (self.noise.accel_noise_density ** 2 *
                                                   dt ** 3 / 3.0)
        C[3:6, 3:6] += np.eye(3) * (self.noise.gyro_noise_density ** 2 * dt)

        m = PoseWithCovarianceStamped()
        m.header.stamp = sec_to_stamp(t_now)
        m.header.frame_id = self.odom_frame
        m.pose.pose.position.x = float(p[0])
        m.pose.pose.position.y = float(p[1])
        m.pose.pose.position.z = float(p[2])
        m.pose.pose.orientation = quat_to_ros(q)
        m.pose.covariance = flatten_cov6(C)
        self.pub_pose_pred.publish(m)

    def _publish_debug(self, img, stamp) -> None:
        try:
            vis = self.frontend.draw(img)
            if self.bridge is not None:
                out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            else:
                return
            out.header.stamp = stamp
            out.header.frame_id = self.camera_frame
            self.pub_debug.publish(out)
        except Exception as e:                      # pragma: no cover
            self.get_logger().warn(f"debug image failed: {e}", throttle_duration_sec=5.0)

    def on_diagnostics(self) -> None:
        with self.lock:
            init = self.kf.initialised
            s = self.kf.summary() if init else None
            stats = dict(self.kf.stats)

        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = "tello_vio: estimator"
        st.hardware_id = "tello"

        kv = []
        if not init:
            st.level = DiagnosticStatus.WARN
            st.message = "waiting for telemetry"
        else:
            sp = float(np.linalg.norm(s["sigma_p"]))
            att = float(np.max(s["sigma_att_deg"]))
            # These thresholds are advisory: they exist so a human watching
            # rqt_robot_monitor sees degradation before the estimate is useless.
            if sp > 2.0 or att > 20.0:
                st.level = DiagnosticStatus.WARN
                st.message = f"uncertain (sigma_p={sp:.2f} m, sigma_att={att:.1f} deg)"
            else:
                st.level = DiagnosticStatus.OK
                st.message = "tracking"
            kv += [
                KeyValue(key="position", value=np.array2string(s["p"], precision=3)),
                KeyValue(key="velocity", value=np.array2string(s["v"], precision=3)),
                KeyValue(key="accel_bias", value=np.array2string(s["ba"], precision=4)),
                KeyValue(key="gyro_bias", value=np.array2string(s["bg"], precision=4)),
                KeyValue(key="baro_bias_m", value=f"{s['bp']:.2f}"),
                KeyValue(key="sigma_p_m", value=np.array2string(s["sigma_p"], precision=3)),
                KeyValue(key="sigma_att_deg",
                         value=np.array2string(s["sigma_att_deg"], precision=2)),
            ]

        for name, (ok, rej, nis) in stats.items():
            total = ok + rej
            kv.append(KeyValue(key=f"{name}_accept",
                               value=f"{ok}/{total} ({100.0*ok/max(1,total):.0f}%) nis={nis:.1f}"))
        kv += [
            KeyValue(key="images", value=str(self._n_images)),
            KeyValue(key="visual_updates", value=str(self._n_visual)),
            KeyValue(key="telemetry", value=str(self._n_telemetry)),
            KeyValue(key="vo_cost_ms", value=f"{self._last_vo_cost_ms:.1f}"),
            KeyValue(key="vo_status",
                     value=self.frontend.last_status if self.frontend else "no camera_info"),
            KeyValue(key="time_offset_s", value=f"{self.time_offset:.3f}"),
        ]
        if self._last_meas is not None:
            m = self._last_meas
            kv.append(KeyValue(
                key="last_visual",
                value=f"{m.model} inliers={m.n_inliers} "
                      f"parallax={np.degrees(m.parallax_rad):.2f}deg "
                      f"t_ok={m.translation_reliable}"))
        st.values = kv
        arr.status = [st]
        self.pub_diag.publish(arr)

    def on_reset(self, request, response):
        with self.lock:
            self.kf = ErrorStateKF(self.kf.cfg)
            self.last_fused_t = None
        self.surrogate.reset()
        if self.frontend is not None:
            self.frontend.tracker.reset()
        self._path.poses.clear()
        response.success = True
        response.message = "estimator reset"
        self.get_logger().info("estimator reset by service call")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = VioNode()
    # MultiThreadedExecutor is required, not optional: with the default
    # single-threaded executor the vision callback group blocks telemetry.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
