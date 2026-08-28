#!/usr/bin/env python3
"""Record a static IMU log and fit biases + noise densities into a YAML config.

Usage::

    # 1. Put the drone on a solid, level, non-vibrating surface. Do NOT arm it.
    #    A desk that a fan or a laptop is resting on will contaminate the log.
    ros2 run tello_vio imu_calib --ros-args -p duration_s:=120.0

The node records, fits with :func:`tello_vio.calib.identify_imu_noise`, prints a
report, and writes a YAML file you can pass straight to ``vio_node``.

Why 120 s and not 10: the Allan-deviation estimate at cluster time ``tau``
averages over ``N - 2 tau/dt`` overlapping windows, so the bias-instability and
random-walk regions -- the ones that set how fast the filter is allowed to trust
dead reckoning -- only become visible once the log is many times longer than the
``tau`` you care about. At the Tello's ~10 Hz telemetry, two minutes gives usable
statistics out to a few seconds of ``tau``. Ten seconds gives you the white-noise
number and nothing else.
"""

from __future__ import annotations

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from ..calib import identify_imu_noise
from ..ros_utils import quat_from_ros, stamp_to_sec, vec3_from
from ..tello_model import TelloImuSurrogate


class ImuCalibNode(Node):

    def __init__(self):
        super().__init__("tello_imu_calib")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("duration_s", 120.0)
        self.declare_parameter("output", "imu_calibration.yaml")

        self.duration = float(self.get_parameter("duration_s").value)
        self.output = str(self.get_parameter("output").value)

        self.t, self.acc, self.gyr = [], [], []
        self.surrogate = TelloImuSurrogate()
        self.t0 = None
        self.done = False

        self.create_subscription(Imu, str(self.get_parameter("imu_topic").value),
                                 self.on_imu, qos_profile_sensor_data)
        self.create_timer(2.0, self.on_tick)
        self.get_logger().info(
            f"Recording {self.duration:.0f}s of static IMU data. Keep the drone "
            "still, level, and on a surface that is not vibrating.")

    def on_imu(self, msg: Imu) -> None:
        if self.done:
            return
        t = stamp_to_sec(msg.header.stamp)
        if self.t0 is None:
            self.t0 = t
        self.t.append(t)
        self.acc.append(vec3_from(msg.linear_acceleration))
        if msg.angular_velocity_covariance[0] >= 0.0:
            self.gyr.append(vec3_from(msg.angular_velocity))
        else:
            w = self.surrogate.update(t, quat_from_ros(msg.orientation))
            self.gyr.append(w if w is not None else np.full(3, np.nan))

    def on_tick(self) -> None:
        if self.done or self.t0 is None:
            return
        elapsed = self.t[-1] - self.t0
        if elapsed < self.duration:
            self.get_logger().info(
                f"  {elapsed:5.1f}/{self.duration:.0f}s  ({len(self.t)} samples)")
            return
        self.done = True
        self.finish()

    def finish(self) -> None:
        t = np.asarray(self.t)
        acc = np.asarray(self.acc)
        gyr = np.asarray(self.gyr)

        # Use the *measured* mean period, not the nominal rate: Tello telemetry
        # is jittery, and an assumed dt propagates straight into every density.
        dts = np.diff(t)
        dts = dts[(dts > 1e-4) & (dts < 1.0)]
        if len(dts) < 10:
            self.get_logger().error("not enough samples to calibrate")
            rclpy.shutdown()
            return
        dt = float(np.median(dts))
        rate = 1.0 / dt

        finite = np.all(np.isfinite(gyr), axis=1)
        gyro_in = gyr[finite] if np.count_nonzero(finite) > 100 else None

        res = identify_imu_noise(acc, gyro_in, dt)

        self.get_logger().info("=" * 68)
        self.get_logger().info(f"samples={res.n_samples} duration={res.duration_s:.1f}s "
                               f"median rate={rate:.2f} Hz (jitter "
                               f"{100*np.std(dts)/dt:.1f}% of the period)")
        self.get_logger().info(f"|mean accel|      = {res.gravity_magnitude:.4f} m/s^2 "
                               f"(expect ~9.807)")
        self.get_logger().info(f"accel bias (x,y)  = {np.round(res.accel_bias[:2], 4)} m/s^2")
        self.get_logger().info(f"gyro  bias        = {np.round(res.gyro_bias, 5)} rad/s")
        self.get_logger().info(f"accel_noise_density = {res.accel_noise_density:.5f} m/s^2/sqrt(Hz)")
        self.get_logger().info(f"gyro_noise_density  = {res.gyro_noise_density:.5f} rad/s/sqrt(Hz)")
        self.get_logger().info(f"accel_bias_rw       = {res.accel_bias_rw:.6f}")
        self.get_logger().info(f"gyro_bias_rw        = {res.gyro_bias_rw:.6f}")
        for w in res.warnings:
            self.get_logger().warn("  ! " + w)
        self.get_logger().info("=" * 68)

        doc = {
            "tello_vio": {
                "ros__parameters": {
                    "accel_noise_density": float(res.accel_noise_density),
                    "gyro_noise_density": float(res.gyro_noise_density),
                    "accel_bias_rw": float(res.accel_bias_rw),
                    "gyro_bias_rw": float(res.gyro_bias_rw),
                }
            },
            "_measured": {
                "samples": int(res.n_samples),
                "duration_s": float(res.duration_s),
                "median_rate_hz": float(rate),
                "period_jitter_pct": float(100 * np.std(dts) / dt),
                "gravity_magnitude": float(res.gravity_magnitude),
                "accel_bias": [float(x) for x in res.accel_bias],
                "gyro_bias": [float(x) for x in res.gyro_bias],
                "warnings": list(res.warnings),
            },
        }
        with open(self.output, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        self.get_logger().info(f"wrote {self.output}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ImuCalibNode()
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
