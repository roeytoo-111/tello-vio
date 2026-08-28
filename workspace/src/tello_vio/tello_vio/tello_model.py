"""Physical model of what a DJI Tello actually measures, and how to get it into ROS.

Everything in this file exists because the Tello is **not** a normal VIO
platform, and pretending otherwise is how Tello VIO projects fail. The facts
that drive the whole design:

1. **There is no raw gyroscope on the wire.** The SDK state broadcast
   (UDP :8890) carries *fused attitude* -- ``pitch``, ``roll``, ``yaw`` as
   whole degrees -- not angular rate. Textbook tightly-coupled VIO (VINS-Mono,
   OpenVINS, ORB-SLAM3-VI) preintegrates 100-200 Hz gyro + accel between camera
   frames. That input does not exist here. See :class:`TelloImuSurrogate` for
   what we do instead.
2. **The state broadcast is slow.** ~10 Hz nominal, jittery over WiFi. Whole-
   degree attitude quantisation at 10 Hz means a differentiated "gyro" carries
   roughly ``1 deg / 0.1 s = 10 deg/s`` of quantisation noise. Useful for
   outlier gating and short-horizon prediction; useless as a precision rate.
3. **Video and state arrive on different sockets with different latencies.**
   H.264 over the Tello's own WiFi AP typically lands 150-350 ms behind reality
   and jitters tens of ms. Neither stream is hardware-timestamped, so the
   camera-IMU time offset is *unknown and time-varying* and must be estimated
   (:mod:`tello_vio.calib`) and compensated.
4. **The Tello already runs its own downward optical-flow + ToF velocity
   estimator**, published as ``vgx/vgy/vgz``. This is the single most valuable
   signal on the drone: it is *metric*, which is exactly what monocular vision
   cannot give you. It is what makes scale observable in practice.
5. **Units in the state packet are firmware-dependent.** The SDK documentation
   is thin and community reports disagree (notably for ``vg*`` and ``tof``).
   Rather than hard-code a guess, every conversion below goes through an
   explicit, overridable scale factor in :class:`TelloUnits`, and
   ``tello_vio.nodes.imu_calib_node`` measures the ones that matter.

Frame conventions
-----------------
The Tello reports in an **aerospace FRD/NED-style body frame** (x forward,
y right, z down). ROS mandates **FLU/ENU** (REP-103: x forward, y left,
z up). The conversion is a 180-degree rotation about the body x axis::

    R_FLU_FRD = diag(1, -1, -1)

Evidence this is the right reading of the hardware: a Tello sitting still on a
table reports ``agz ~= -1000`` (i.e. -1 g on its z axis). An accelerometer
measures *specific force* ``f = a - g``. At rest ``a = 0``, so in FRD (z down)
``f = -g_vec = [0, 0, -9.81]`` -- exactly the observed sign. After the FLU
conversion the same drone reports ``+9.81`` on z, which is what REP-145 requires
of a ``sensor_msgs/Imu`` publisher.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lie

#: Standard gravity (CODATA / ISO 80000-3), m/s^2.
G0 = 9.80665

#: Rotation taking a vector expressed in the Tello's FRD body axes to ROS FLU.
#: It is its own inverse (a 180 deg rotation about x).
R_FLU_FRD = np.diag([1.0, -1.0, -1.0])

DEG = np.pi / 180.0


@dataclass
class TelloUnits:
    """Scale factors converting raw SDK state fields to SI.

    Defaults reflect the most widely reproduced readings on stock firmware, but
    **they are parameters, not constants** -- verify them for your airframe with
    ``ros2 run tello_vio imu_calib`` before trusting the metric output.

    Attributes
    ----------
    accel_to_mps2:
        ``agx/agy/agz`` are reported in milli-g (0.001 g). At rest one axis
        reads ~1000. Hence ``G0 / 1000``.

        A note on a bug this replaces: dividing the raw value by 100 gives
        ``1000/100 = 10``, which *looks* like 9.81 m/s^2 and therefore passes a
        casual eyeball check while being 1.97 % high. Over a 10 s dead-reckoned
        segment that alone integrates to roughly 1 m of position error, on top
        of everything else.
    speed_to_mps:
        ``vgx/vgy/vgz``. The SDK document gives no unit. Decimetres per second
        is the reading consistent with the Tello's 8 m/s top speed against
        observed field magnitudes (~30 at full stick); centimetres per second
        would cap the drone at 1 m/s, which it plainly exceeds. Set
        ``0.01`` if your firmware disagrees -- ``imu_calib`` will tell you.
    baro_to_m:
        ``djitellopy.Tello.get_barometer()`` already multiplies the raw metres
        by 100 and documents its return as centimetres, so the driver divides
        by 100 again. The result is a *pressure altitude* with a large,
        slowly drifting offset -- never an altitude above the takeoff point.
        The estimator therefore carries an explicit barometer-bias state.
    tof_to_m:
        ``get_distance_tof()`` returns centimetres. The downward ToF is only
        trustworthy roughly within 0.1-1.2 m of a flat surface and reads
        garbage over stairs, tables and drop-offs, so the estimator gates it.
    """

    accel_to_mps2: float = G0 / 1000.0
    speed_to_mps: float = 0.1
    baro_to_m: float = 1.0
    tof_to_m: float = 1.0
    tof_min_m: float = 0.10
    tof_max_m: float = 1.20


@dataclass
class TelloNoise:
    """Noise model for the Tello's published signals.

    These are *starting points* measured on a stock Tello sitting on a desk and
    hovering indoors; ``imu_calib_node`` refits ``accel_*`` and ``att_*`` from
    your own static log. They are deliberately pessimistic: an over-confident
    R matrix makes a Kalman filter diverge silently, an under-confident one
    only makes it sluggish.

    Units follow the usual IMU convention: densities are per sqrt(Hz), so the
    discrete-time variance at rate ``f`` is ``sigma^2 * f`` for white noise and
    ``sigma^2 / f`` for a random walk.
    """

    #: Accelerometer white noise density, m/s^2/sqrt(Hz). Large because what we
    #: receive is a heavily filtered, 10 Hz, telemetry-decimated signal rather
    #: than a raw MEMS output -- decimation aliases the high-frequency content
    #: of a vibrating airframe straight into the band we use.
    accel_noise_density: float = 0.08
    #: Accelerometer bias random walk, m/s^3/sqrt(Hz).
    accel_bias_rw: float = 1.0e-3

    #: Surrogate-gyro white noise density, rad/s/sqrt(Hz). Dominated by the
    #: 1-degree attitude quantisation, not by the physical MEMS gyro:
    #: a uniform quantiser of width q has std q/sqrt(12), differentiated over
    #: dt this becomes q/(dt*sqrt(6)) ~= 0.07 rad/s at 10 Hz.
    gyro_noise_density: float = 0.03
    #: Surrogate-gyro bias random walk, rad/s^2/sqrt(Hz).
    gyro_bias_rw: float = 1.0e-4

    #: Attitude measurement std (rad) for roll/pitch. Gravity-referenced by the
    #: onboard AHRS, so these are absolute and drift-free.
    att_rp_std: float = 2.0 * DEG
    #: Attitude measurement std (rad) for yaw. Yaw has *no* absolute reference
    #: (no usable magnetometer), so it free-runs. A large value here makes the
    #: filter treat the Tello's yaw as a smoothness hint, not as truth.
    att_yaw_std: float = 15.0 * DEG

    #: Body-velocity measurement std (m/s) from the onboard optical flow.
    vel_body_std: float = 0.15
    #: Extra velocity std added per metre of height (flow scales with range and
    #: degrades as the ground texture thins out).
    vel_body_std_per_m: float = 0.05

    #: Barometer measurement std (m) and bias random walk (m/sqrt(s)).
    baro_std: float = 0.30
    baro_bias_rw: float = 0.01

    #: ToF measurement std (m).
    tof_std: float = 0.03

    #: Stationarity thresholds used by the zero-velocity detector.
    zupt_accel_std_thresh: float = 0.35
    zupt_speed_thresh: float = 0.08
    zupt_gyro_thresh: float = 0.08


@dataclass
class TelloState:
    """One decoded SDK state broadcast, already converted to SI + ROS FLU.

    ``stamp`` is the receive time in seconds. It is *not* a capture time: the
    Tello does not timestamp its telemetry, so this carries the WiFi + OS
    scheduling latency of the state socket (small, ~5-15 ms, and much smaller
    than the video path's).
    """

    stamp: float
    #: Specific force in the FLU body frame, m/s^2 (at rest: ``[0, 0, +9.81]``).
    accel: np.ndarray
    #: Attitude of the body in the ENU-ish world frame, Hamilton [w,x,y,z].
    quat: np.ndarray
    #: Onboard velocity estimate in the FLU body frame, m/s.
    vel_body: np.ndarray
    #: Pressure altitude, m (arbitrary offset).
    baro: float
    #: Downward ToF range, m, or ``None`` when out of its valid window.
    tof: float | None
    #: Reported height above takeoff, m (integer decimetre resolution).
    height: float
    battery: int = 0
    #: True while the airframe is judged to be sitting still (see ZUPT).
    stationary: bool = False


def frd_to_flu(v: np.ndarray) -> np.ndarray:
    """Re-express a vector from the Tello's FRD axes in ROS FLU axes."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([v[0], -v[1], -v[2]])


def tello_attitude_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float,
                           yaw_offset_rad: float = 0.0) -> np.ndarray:
    """Convert the SDK's NED-referenced Euler angles to a ROS ENU-referenced quaternion.

    The SDK reports the FRD body attitude relative to a NED world frame. ROS
    wants the FLU body attitude relative to an ENU world frame. Composing the
    two frame flips gives, for the Euler angles themselves::

        roll_ENU  =  roll_NED
        pitch_ENU = -pitch_NED
        yaw_ENU   = -yaw_NED            (+ a constant heading offset)

    The heading offset between NED-north and ENU-east is a constant 90 degrees,
    but the Tello's yaw origin is whatever direction it happened to be facing at
    power-on -- it has no usable magnetometer -- so the offset is arbitrary
    anyway. ``yaw_offset_rad`` exists to align the estimator's world frame with
    something you care about (e.g. the first keyframe); it does not make the
    yaw absolute.
    """
    roll = float(roll_deg) * DEG
    pitch = -float(pitch_deg) * DEG
    yaw = -float(yaw_deg) * DEG + float(yaw_offset_rad)
    return lie.euler_zyx_to_quat(yaw, pitch, roll)


class TelloImuSurrogate:
    """Turn the Tello's 10 Hz *attitude* stream into a usable angular-rate signal.

    Why this class exists: the estimator's propagation model wants
    ``omega_body``. The drone gives orientation. Differentiating orientation is
    the only route, and doing it naively -- subtracting Euler angles -- breaks
    at wrap-around and is wrong whenever roll and yaw both change. The correct
    finite difference is taken *on the manifold*::

        omega ~= Log(q_{k-1}^{-1} (x) q_k) / dt

    which is exact for a constant-rate rotation over the interval and has no
    wrap-around pathology.

    Two refinements matter in practice:

    * **Quantisation.** Whole-degree attitude at 10 Hz gives a rate signal
      quantised to ~10 deg/s. A short zero-phase-ish smoother (a 3-tap
      symmetric FIR applied to the *rotation vectors*, not the angles) halves
      the variance without adding meaningful lag at these rates.
    * **Jitter.** The state socket does not arrive on a metronome. Dividing by
      the *nominal* period instead of the actual one injects a rate error
      proportional to the jitter, so we always use measured ``dt`` and reject
      samples whose ``dt`` is implausible.
    """

    def __init__(self, max_dt: float = 0.5, min_dt: float = 5e-3, smooth: int = 3):
        self._prev_q: np.ndarray | None = None
        self._prev_t: float | None = None
        self.max_dt = float(max_dt)
        self.min_dt = float(min_dt)
        self._hist: list[np.ndarray] = []
        self._smooth = max(1, int(smooth))

    def reset(self) -> None:
        self._prev_q = None
        self._prev_t = None
        self._hist.clear()

    def update(self, stamp: float, quat: np.ndarray) -> np.ndarray | None:
        """Feed one attitude sample; return ``omega_body`` (rad/s) or ``None``.

        ``None`` means "no usable rate this cycle" -- first sample, a duplicate
        telemetry packet, or a gap so long that a finite difference across it
        would be meaningless. Callers must treat that as *missing data* and let
        the filter propagate on its bias estimate, not silently substitute zero:
        substituting zero tells the filter the drone is not rotating, which is a
        confident lie and will corrupt the attitude covariance.
        """
        quat = lie.quat_normalize(quat)
        if self._prev_q is None or self._prev_t is None:
            self._prev_q, self._prev_t = quat, float(stamp)
            return None

        dt = float(stamp) - self._prev_t
        if dt < self.min_dt or dt > self.max_dt:
            # Duplicate packet or a dropout: re-anchor without emitting a rate.
            self._prev_q, self._prev_t = quat, float(stamp)
            return None

        omega = lie.quat_boxminus(quat, self._prev_q) / dt
        self._prev_q, self._prev_t = quat, float(stamp)

        self._hist.append(omega)
        if len(self._hist) > self._smooth:
            self._hist.pop(0)
        return np.mean(self._hist, axis=0)


class StationarityDetector:
    """Decide whether the airframe is sitting still, for zero-velocity updates.

    A ZUPT is the cheapest high-value measurement in the whole system: while the
    drone is on the ground, ``v = 0`` and ``omega = 0`` are *exact*, which pins
    down the accelerometer and gyro biases far better than any amount of flight
    data. Getting a ZUPT wrong is expensive though -- declaring "stationary"
    during a steady hover injects a hard false constraint -- so the test
    requires *all* of low accel variance, low reported speed and low rate,
    sustained over a window.
    """

    def __init__(self, noise: TelloNoise, window: int = 10):
        self.noise = noise
        self.window = max(3, int(window))
        self._accel: list[np.ndarray] = []
        self._ok_run = 0

    def update(self, accel: np.ndarray, vel_body: np.ndarray,
               omega: np.ndarray | None) -> bool:
        self._accel.append(np.asarray(accel, dtype=np.float64).reshape(3))
        if len(self._accel) > self.window:
            self._accel.pop(0)
        if len(self._accel) < self.window:
            return False

        A = np.asarray(self._accel)
        # Variance of the accel *magnitude* is gravity-invariant, so this test
        # does not care how the drone is tilted.
        accel_std = float(np.std(np.linalg.norm(A, axis=1)))
        speed = float(np.linalg.norm(vel_body))
        rate = float(np.linalg.norm(omega)) if omega is not None else 0.0

        still = (
            accel_std < self.noise.zupt_accel_std_thresh
            and speed < self.noise.zupt_speed_thresh
            and rate < self.noise.zupt_gyro_thresh
        )
        self._ok_run = self._ok_run + 1 if still else 0
        return self._ok_run >= self.window
