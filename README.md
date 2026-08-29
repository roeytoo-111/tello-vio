# Tello VIO — Visual-Inertial Odometry on a DJI Tello (ROS 2 Humble)

Metric, drift-bounded pose estimation on a stock DJI Tello, with no GPS and no
motion capture. A KLT/two-view visual front-end feeds an error-state Kalman
filter (real-time) and an optional fixed-lag factor-graph smoother (accuracy),
fused with the drone's own attitude, optical-flow velocity, barometer and ToF.
ORB-SLAM2 runs behind it as an optional loop-closing map backend.

> **Pipeline diagram (one page):** [`docs/Tello_VIO_Pipeline.pdf`](docs/Tello_VIO_Pipeline.pdf)
> — every stage from UDP packet to published pose, with rates and algorithms.
>
> **Full technical report (32 pages):** [`docs/Tello_VIO_Technical_Report.pdf`](docs/Tello_VIO_Technical_Report.pdf)
> — the algorithms, the derivations, the ROS 2 architecture and the measured results.

---

## Start here: what a Tello can and cannot do

This shapes every design decision in the repository, so it is stated before
anything else.

| | |
|---|---|
| **No gyroscope** | The SDK state broadcast carries *fused attitude* (whole degrees), not angular rate. Textbook tightly-coupled VIO — VINS-Mono, OpenVINS, ORB-SLAM3-VI — assumes 100–200 Hz gyro + accel. **That input does not exist on this aircraft.** |
| **~10 Hz telemetry** | And jittery. A rate differentiated from whole-degree attitude at 10 Hz carries ~10 °/s of quantisation noise. |
| **150–350 ms video latency** | H.264 over the drone's own WiFi AP, no hardware timestamps, jitter of tens of ms. |
| **Metric velocity, for free** | The Tello runs its own downward optical-flow + ToF velocity estimator (`vgx/vgy/vgz`). This is the single most valuable signal on the drone: it is **metric**, which is exactly what a monocular camera cannot be. |
| **Off-board compute** | Everything here runs on your laptop. The drone streams; it does not compute. "Efficient" means *fits in a 33 ms frame budget on one core*, not *fits on the drone*. |

**The consequence:** this is a *loosely-coupled* estimator with an explicit
metric-scale source, not a tightly-coupled one. Vision supplies rotation and
translation **direction**; magnitude comes from the flow sensor, barometer and
ToF. Pretending a monocular camera observes scale is how mono-VIO pipelines
acquire a slowly-wrong scale that poisons everything downstream.

---

## Results

From `workspace/src/tello_vio/test/test_integration.py`, a simulated 60 s
figure-of-eight with the real drone's limitations modelled (10 Hz jittery
telemetry, 1° attitude quantisation, milli-g accelerometer, integer-decimetre
velocity, drifting barometer, free-running yaw):

| Metric | Result |
|---|---|
| Absolute trajectory error (RMS), 39.8 m flown | **0.57 m** |
| Final position drift | **1.4 % of path length** |
| Metric scale accuracy (path-length ratio) | **1.034** |
| Roll/pitch RMS error | **1.8°** |
| Position NEES (ideal 3.0) | **3.2** at 40 s, **3.8** at 60 s — mildly over-confident |
| Accelerometer bias, estimated vs true | `[0.134 −0.098 0.055]` vs `[0.12 −0.08 0.05]` |

**What the camera is actually worth**, averaged over 5 seeds:

| Optical-flow quality | With vision | Without vision | Improvement |
|---|---|---|---|
| Healthy (σ = 0.08 m/s) | 0.364 m | 0.348 m | −5 % (neutral) |
| Degraded (σ = 0.30 m/s) | 0.530 m | 0.666 m | **+20 %** |
| Bad (σ = 0.80 m/s) | 0.779 m | 1.647 m | **+53 %** |
| Flow sensor dead | 0.75 m | 15.1 m | **20× better** |

Read that honestly: **when the Tello's flow sensor is working well over
textured ground, monocular VO adds little to short-horizon odometry.** Its
value is (a) redundancy — flow degrades above ~1 m altitude, over plain floors,
in low light, over moving surfaces — and (b) global consistency through loop
closure, via the ORB-SLAM2 backend. The repository does not overstate this.

`116` unit and integration tests, all passing, none requiring a drone:

```bash
cd workspace/src/tello_vio && python3 -m pytest test/ -q
```

---

## Quick start

```bash
# 1. Install (Ubuntu 22.04 + ROS 2 Humble)
./scripts/install.sh

# 2. Build
./scripts/build.sh

# 3. Connect to the Tello's WiFi AP (TELLO-XXXXXX), then:
source install/setup.bash
ros2 launch tello_vio vio.launch.py rviz:=true
```

Before the numbers mean anything, calibrate — see **Calibration** below. Running
uncalibrated produces a plausible-looking trajectory in the wrong units.

---

## Packages

```
workspace/src/
  tello/          Driver: video, telemetry, control. Publishes SI units in ROS frames.
  tello_msg/      TelloStatus / TelloID / TelloWifiConfig messages.
  tello_control/  Keyboard control GUI with a live video window.
  tello_vio/      The estimator. Pure NumPy/OpenCV core + ROS 2 nodes.
slam/src/
  orbslam2/       ORB-SLAM2 wrapper. Optional; publishes pose, TF and the map.
```

`tello_vio` is layered so everything except `nodes/` is testable and profilable
without ROS installed:

| Module | Contents |
|---|---|
| `lie.py` | SO(3)/SE(3) operations. **All conventions are declared here**, once. |
| `tello_model.py` | What the drone actually measures: units, frames, the surrogate gyro. |
| `preintegration.py` | On-manifold IMU preintegration (Forster et al., T-RO 2017). |
| `eskf.py` | 22-state error-state KF with stochastic cloning. *Real-time path.* |
| `smoother.py` | Fixed-lag factor graph with Schur-complement marginalisation. |
| `two_view.py` | Essential/homography model selection, relative pose, triangulation. |
| `frontend.py` | KLT tracking, keyframing, relative-pose measurements. |
| `calib.py` | Allan-variance IMU identification, hand-eye extrinsic, time offset. |
| `sim3.py` | Umeyama similarity alignment — `map ↔ odom`, monocular map scale. |

---

## Frames (REP-103 / REP-105)

```
map ──────────► odom ──────────► base_link ──────────► camera_optical
     may jump         continuous          static, calibrated
  (map_align)        (tello_vio)          (from camera_imu_calib)
```

* `odom → base_link` is **continuous**. Controllers differentiate it, so a jump
  there is a step input to the aircraft. Published by `tello_vio`.
* `map → odom` is **allowed to jump**. Nothing differentiates it. Published by
  `map_align`, which aligns ORB-SLAM2's scale-free trajectory to the metric VIO
  one; a loop closure lands here and the control loop never sees it.
* `base_link` is FLU (x forward, y left, z up); `camera_optical` is the REP-103
  optical frame (z out of the lens, x right, y down).

**Exactly one node owns each TF edge.** Two publishers on one edge is a broken
tree, not a redundant one.

---

## Calibration

Three quantities must be measured per airframe. Skipping any of them gives you a
confident wrong answer rather than an obviously wrong one.

**1 — Camera intrinsics.** The shipped `ost.yaml` is from somebody else's drone,
and its tangential distortion (`p1 = −0.023`) is large enough to suggest an
under-constrained fit.

```bash
./scripts/cameracalib.sh          # SIZE=8x6 SQUARE=0.025 to override
```

Fill the image corners, vary distance, and **tilt the board** — a target held
flat and centred trades focal length against distance and yields a confident
wrong `fx`.

**2 — IMU noise and biases.** Drone still, motors off, on a surface that is not
vibrating:

```bash
ros2 launch tello_vio calibrate.launch.py target:=imu duration:=120.0
```

Two minutes, not ten seconds: the Allan deviation only reveals the
bias-instability and random-walk regions once the log is many times longer than
the cluster times you care about.

**3 — Camera-IMU rotation and time offset.** Pick the drone up (motors off) and
rotate it smoothly about **all three axes** in front of a textured scene:

```bash
ros2 launch tello_vio calibrate.launch.py target:=camera_imu duration:=90.0
```

Rotating about one axis leaves the extrinsic undetermined about that axis, so
the node reports an **excitation score** alongside the residual. A small
residual with a low excitation score means the fit is confidently wrong.

Merge both YAML fragments into `workspace/src/tello_vio/config/tello_vio.yaml`.

---

## Topics

**Driver** (`tello`) publishes:

| Topic | Type | Notes |
|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | ~30 Hz, `bgr8`, duplicate frames suppressed |
| `/camera_info` | `sensor_msgs/CameraInfo` | **rescaled** when `video_scale != 1.0` |
| `/imu` | `sensor_msgs/Imu` | SI, FLU. `angular_velocity_covariance[0] = -1` — there is no gyro |
| `/odom` | `nav_msgs/Odometry` | Twist only; `pose.covariance[0] = -1` — no position source |
| `/tof` | `sensor_msgs/Range` | `+inf` when out of the sensor's honest window |
| `/status`, `/id`, `/battery`, `/temperature` | | `TelloStatus` now carries a `Header` |

**Estimator** (`tello_vio`) publishes `~/odom`, `~/pose_predicted`, `~/path`,
`~/debug_image`, `/diagnostics`, and the `odom → base_link` transform.
`~/pose_predicted` is an *extrapolation* across the video latency for
controllers — separate topic, inflated covariance, never confused with the
fused estimate.

**Control:** `/takeoff`, `/land`, `/emergency`, `/flip`, `/wifi_config`, and two
velocity topics — `/control` (legacy stick axes: `linear.x` = lateral) and
`/cmd_vel` (REP-103: `linear.x` = forward, normalised to ±1).

---

## Notable fixes in this revision

An adversarial audit confirmed 30 defects in the original code. The ones that
were reachable in flight:

* **Blocking SDK calls in ROS callbacks.** `land()` retries for up to 21 s inside
  its subscriber callback. On a single-threaded executor that froze
  `/emergency` *and* the RC dead-man for the whole window, leaving a drone
  commanded forward flying forward. All blocking commands now run on a worker
  thread; `/emergency` bypasses the queue.
* **`/imu` published a fabricated zero angular velocity with zero covariance** —
  ROS semantics for "measured, and exactly known". Any consumer would believe it.
* **IMU in the drone's FRD frame with a 2 % scale error** — `z ≈ −10 m/s²` at
  rest instead of `+9.81`.
* **Attitude published without the NED→ENU conversion** — yaw and pitch ran
  backwards.
* **`camera_info` was not rescaled with `video_scale`** — at `0.5` it claimed
  twice the true focal length, which SLAM absorbs as an unrecoverable scale error.
* **`AttributeError` on most `video_scale` values** — `_resize_cache` was read
  but never initialised.
* **The ORB-SLAM2 wrapper computed a pose every frame and discarded it** — no TF,
  no odometry. It now publishes pose, TF, path and tracking state, with the
  optical→ROS rotation applied.
* **`ORB_SLAM2_ENABLE=OFF` forced the build instead of skipping it.**
* **`scripts/build.sh` did `cd ../workspace && rm -rf build install log`** — a
  relative path plus a recursive delete.

Full detail, with failure scenarios, is in the technical report.

---

## Optional: ORB-SLAM2 backend

```bash
./scripts/orbslam.sh                       # builds into libs/ORB_SLAM2
export ORB_SLAM2_ROOT_DIR="$PWD/libs/ORB_SLAM2"
./scripts/build.sh

ros2 launch tello_vio vio.launch.py slam:=true \
    vocabulary:=$PWD/libs/ORB_SLAM2/Vocabulary/ORBvoc.txt \
    slam_config:=$(ros2 pkg prefix orbslam2)/share/orbslam2/config.yaml
```

Without `ORB_SLAM2_ROOT_DIR` the package configures, warns, and skips its node,
so the rest of the workspace still builds. **The settings file must match the
resolution actually being published** — ORB-SLAM2 reads that YAML, not
`/camera_info`, so nothing will warn you if `video_scale` disagrees with it.

---

## Troubleshooting

**`ros2 topic list` / `rqt` hangs forever (WSL2).** The `ros2` daemon fails to
start under WSL2; DDS itself is fine. Every introspection command works with the
daemon bypassed:

```bash
ros2 topic list --no-daemon
```

Launching nodes is unaffected — only the CLI introspection tools use the daemon.

**`AttributeError: _ARRAY_API not found` or `KeyError: 16` from `cv_bridge`.**
ROS 2 Humble's compiled Python extensions are built against numpy 1.x and
OpenCV 4.x. A newer numpy or OpenCV in `~/.local` shadows them and breaks
`cv_bridge` — which also breaks `camera_calibration`. Pin them back:

```bash
python3 -m pip install --user "numpy<2" "opencv-python<5"
```

`tello_vio` itself runs on either stack (the test suite passes on both); only
ROS's own extensions are version-sensitive. The driver and VIO node degrade
gracefully without `cv_bridge`, but you lose debug images and calibration.

**`CMake Error: ... CMakeCache.txt directory is different`.** A stale `build/`
tree from another path. Wipe it:

```bash
CLEAN=1 ./scripts/build.sh
```

**`OSError: [Errno 98] Address already in use` on startup.** Not a connectivity
problem. A previous driver is still holding UDP :8889.

The usual cause is stopping the launch with **Ctrl-Z instead of Ctrl-C**.
Ctrl-Z *suspends* the process — it keeps every socket it owns, indefinitely, and
sessions stack up invisibly. **Use Ctrl-C.** Check for suspended jobs with
`ps -eo pid,stat,cmd | grep tello` and look for `T` in the STAT column, or `jobs`
in the shell you launched from (`fg` then Ctrl-C, or `kill -9 %1`).

Find and stop whatever holds the port:

```bash
ss -lunp | grep -E '8889|8890|11111'
```

```bash
pkill -f 'lib/tello/tello'
```

The driver now detects this case and prints those instructions instead of a
traceback.

**The drone stops responding after ~15 s.** The Tello auto-lands if it hears no
command for 15 s. The driver sends RC at 20 Hz precisely to prevent this — check
the driver node is actually running and that `/status` is updating.

## Known limitations

* **Yaw is unbounded.** No magnetometer, so heading free-runs (~13° RMS over
  60 s in simulation). Bounding it needs loop closure or an external reference.
* **Monocular scale depends on the flow sensor.** Above ~1–2 m, or over a
  featureless floor, the metric anchor weakens — the table above quantifies it.
* **Preintegration is fed a 10 Hz surrogate rate**, not a gyro. The mathematics
  is rate-agnostic and would run unchanged on a real IMU; the *information* it
  carries here is proportionally weaker.
* **No relocalisation in the fast path.** KLT has no descriptors. Recovering
  from total tracking loss is ORB-SLAM2's job.
* **Simulation is not flight.** The results above come from a simulator built
  from the SDK's documented behaviour. Validate on a recorded bag before
  trusting any number.

## Overheating

The Tello's motor drivers overheat when it sits powered on but not flying —
exactly what calibration involves. Point a fan at it, or expect the two-minute
IMU log to end in a thermal shutdown.

## Licence

MIT. ORB-SLAM2 is GPLv3 and is *not* vendored here; `scripts/orbslam.sh` fetches
it separately.
