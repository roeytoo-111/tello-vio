# Implementation Plan — Research Proposal on ROS 2

Companion to [block_diagram.md](block_diagram.md). Planning only — no code on this branch.

---

## 1. Ground rules taken from the proposal (they shape every decision below)

1. **Simulation first, hardware second** (PDF §4): the decisive measurements need exact ground
   truth and controlled sweeps; safety asks that closed-loop behaviour be exercised in
   simulation before any flight.
2. **One module per switch** (PDF §4.3): the ablation is a configuration matrix, never a code
   change. Concretely: every switch in the [switchboard](block_diagram.md#6-the-ablation-switchboard-pdf-43)
   is a ROS 2 parameter of exactly one node.
3. **Dual timestamps everywhere** (PDF §4.3): every message carries the time it was captured
   *and* the time it was received; latency is measured, never assumed.
4. **The controller is fixed** (PDF §4.1): it closes the loop for E6 and is held constant
   across all perception/estimation variants.
5. **Report per range band** (PDF §3.1, §5.1): Eq. (2) makes any single average dominated by
   the far field.
6. **Statistics up front** (PDF §5.3): every comparison gets confidence intervals over ≥10
   seeds; filter consistency over ≥50 repeated runs.
7. **Cheapest decisive experiment first** (PDF §5.4, §7): E1 on downsampled public crops runs
   in month one, before anything is built for flight.

---

## 2. Verified platform baseline (what `workspace/src` actually provides)

All facts checked in code/config on this repo — file references given so they can be re-verified.

| Fact | Value | Where verified |
|---|---|---|
| Camera stream | 960×720, `bgr8`, ~30 Hz target, duplicate frames suppressed, auto-restart on stall | `tello/tello/node.py` video path; `README.md` topics table |
| Camera intrinsics | fx ≈ 919.42 px, fy ≈ 911.93 px (rescaled automatically if `video_scale` ≠ 1) | `tello/resource/ost.txt`; `node.py:971-1024` |
| Video latency | **150–350 ms**, jitter of tens of ms, arrival-stamped | `README.md` platform constraints |
| Inertial data | **No gyroscope.** Fused attitude (whole degrees, NED→ENU converted) + accel (milli-g→SI), ~10 Hz jittery; `angular_velocity_covariance[0] = -1` | `node.py:558-588`; `README.md` |
| Telemetry stamping | 50 Hz change-detection polling → stamp error ≤ 20 ms | `node.py:496-511` |
| Metric velocity | downward optical-flow + ToF velocity (`vgx/vgy/vgz`), FLU on `/odom` twist; **metric** — the platform's most valuable signal | `node.py:590-621`; `README.md` |
| Command interface | `/cmd_vel` REP-103, sticks normalised to [−1, 1] → SDK ±100; RC resent at 20 Hz; **dead-man zeroes a command stale by > 0.35 s** → controllers must publish ≥ 10 Hz | `node.py:1084-1114` |
| Abort paths | `/emergency` double path (fire-and-forget bypass + retried queue); `/land`; SDK auto-land after 15 s of RC silence | `node.py:1046-1059`; `README.md` |
| Absolute reference (fallback) | Mission pads, Tello **EDU** only: drift-free pose relative to a printed pad, 10–20 Hz, ~cm | `tello_msg/msg/TelloMissionPad.msg`; driver `_publish_mission_pad` |
| Namespacing | all driver topics are relative names → two-aircraft namespacing works; the launch file does not yet expose a `namespace` argument (small future change) | `node.py:355-381`; `tello/launch/tello.launch.py` |
| Two drivers, one host | **impossible as-is**: djitellopy binds UDP :8889 in its constructor (second process gets EADDRINUSE), and both drones in AP mode are 192.168.10.1 | `node.py:205-220` |
| Simulator provisioning | Gazebo 11 (classic) install script exists; nothing proposal-specific yet | `scripts/gazebo.sh` |

### The pixel-size regime, with the repo's own numbers

Pinhole: `w = fx · W / d`. Using fx = 919.4 px and the Tello's official body width
W = 0.098 m (Ryze specs — PDF §4.2 says "roughly 10 × 9 cm"):

| Range d | Apparent width w |
|---|---|
| 1 m | ~90 px |
| 2 m | ~45 px |
| 3 m | ~30 px |
| 4.5 m | ~20 px |
| 6 m | ~15 px |
| 9 m | ~10 px |

An indoor mocap volume (2–8 m working range) covers exactly the tens-of-pixels regime the
proposal targets (average target in the public air-to-air benchmark: ~31×17 px, PDF §1.1).
Note `video_scale` can halve fx to extend the sweep at fixed range, but PDF §4.1 prefers
moving the target over degrading the image — use it only as a checked supplement.
**Caveat for the dimension DB:** a detector's box spans the *visible* extent (usually
prop-tip to prop-tip), which is larger than the 98 mm body. The DB must record the
measurement convention and the annotation rule must match it (see §6 below).

---

## 3. Packages to build (names, roles, interfaces — no code yet)

Proposed new packages under `workspace/src/`, one module per switch (PDF §4.3):

| Package | Type | Role | Key parameters (= switches) |
|---|---|---|---|
| `rtr_msgs` | msgs | the topic contract: `DroneDetection`, `RangeMixture`, `TargetTrack`, `ObserverMotion` — every msg carries `t_capture` + `t_receive` | — |
| `rtr_dimension_db` | data + lib | machine-readable dimension table (YAML/CSV) + lookup; the released C1 artefact | table path |
| `rtr_detect` | node | drone detection + model-recognition head; consumes image + motion channel; emits bbox + class posterior incl. `unknown` | model checkpoint, score threshold |
| `rtr_motion_comp` | node | background alignment for the motion channel + bearing de-rotation | `mode: gyro / homography / none` (S4) |
| `rtr_observer_motion` | node | hw: surrogate angular rate from ~10 Hz attitude (differentiation + smoothing, noise stated); sim: passthrough of true gyro; inter-frame preintegration | source: `attitude_surrogate / gyro` |
| `rtr_range` | node | Eq. (4) mixture over the DB, aspect-angle marginalisation (Eq. 3), box-noise model σw(w) via Eq. (2) | `size_prior: fixed / model / model+aspect` (S1), `output: mixture / point` (S3) |
| `rtr_tracker` | node | multiple-model filter on target relative state; NEES/NIS logged | `r_inflation: on/off` (S5), `delayed_replay: on/off` (S6), `filter_rate_hz` (S7) |
| `rtr_follow` | node | the FIXED simple follower (e.g. proportional bearing-centring + standoff-range hold) → `cmd_vel` | gains — frozen after tuning, never swept |
| `rtr_safety` | node | supervisor between controller and driver: engagement gate, geofence (mocap-fed), closing-speed cap, auto-disarm (link/target loss, breach, low battery) → forwards or zeroes `cmd_vel`, can issue `land`/`emergency` | volume bounds, v_max, disarm rules |
| `rtr_sim` | node + assets | simulator interface publishing the same topic contract + `/gt/*`; CAD models of DB drones; all sweep knobs (S8, S9, S11) | delay, cam-IMU offset, exposure, focal length, target range/behaviour, observer mode, seed |
| `rtr_gt_bridge` | node | mocap → `/gt/observer_pose`, `/gt/target_pose` (hardware phase) | mocap backend, rigid-body names |
| `rtr_experiments` | tools | `experiment_manager` (runs one config-matrix row: switches + knobs + seed; records bags) + offline `evaluation` (AP by size band, range error per band, NEES over ≥50 runs, threshold-curve closed-loop scores, CIs over ≥10 seeds) + the E2 zero-shot depth-foundation-model arm (offline on recorded frames) | config matrix YAML |

**Reused from the baseline, unchanged:** `tello` driver, `tello_msg`, `tello_control`
(manual override + abort), the calibration tooling. **Deliberately not used:** the `tello_vio`
estimator (decision recorded from the branch owner; this thesis tracks *another* drone, not
the observer's own pose — observer attitude/velocity come from the driver, observer ground
truth from mocap).

**Only baseline change eventually needed:** a `namespace` launch argument for
`tello.launch.py` (topics are already relative). Not done on this branch.

### Message contracts (field sketches)

- `DroneDetection` — header (stamp = capture time), `t_receive`, bbox (cx, cy, w, h, px),
  detection score, `class_names[]` (matching DB keys, incl. `unknown`), `class_posterior[]`.
- `RangeMixture` — header, `t_receive`, per-component {model_id, weight πk, mean dk, sigma}, 
  collapsed mean/sigma for convenience, de-rotated bearing (az/el) + flag.
- `TargetTrack` — header, relative position + velocity (frame stated) + 6×6 covariance,
  model posterior, NIS value, active filter mode.
- `ObserverMotion` — header, angular rate + source (`gyro`/`surrogate`), rate variance,
  preintegrated inter-frame rotation, `t_capture`/`t_receive`.

---

## 4. Experiment → module mapping (PDF Table 1, verbatim falsification criteria)

| Exp | Tests | Runs on | Modules exercised | Falsified if (PDF) |
|---|---|---|---|---|
| **E1** | H1 | downsampled public crops (month 1), then sim | `rtr_detect`, `evaluation` — **no flight, no new code beyond the recognition head** | "No clear fall-off, or a fall-off so early that ranging only works at close range." |
| **E2** | H2 | sim → hardware (Q6) | `rtr_range` (S1 sweep), depth-model offline arm (S2), `rtr_dimension_db`, `evaluation` | "The recognised-model size is not clearly better than a fixed size for small targets, or a depth model matches it." |
| **E3** | H3 | sim (≥50 repeats) | `rtr_range` (S3), `rtr_tracker`, misclassification injection (S10), NEES in `evaluation` | "The mixture-fed filter is no more consistent than the point-fed one." |
| **E4** | H4 | sim (full rate) → hardware (lower bound) | `rtr_motion_comp` (S4), `rtr_tracker` (S5–S7), three observer modes (S11) | "No improvement under manoeuvre, or a gain at filter rates below the camera rate." |
| **E5** | H4 | sim only (needs the injectable offset) | `rtr_sim` (S8), full stack | "No crossover, or a crossover so tight it needs special hardware." |
| **E6** | all | sim → staged hardware ladder | full stack + `rtr_follow` + `rtr_safety`; threshold curves (S12) | "Perception and estimation gains do not reach any closed-loop number." |

Hypotheses H1–H4 are each owned by exactly one experiment chain above (PDF §3.6), and the
headline results figure is E6's success-rate-vs-target-speed with the capture threshold as a
family of curves (PDF §7).

---

## 5. Open decisions (each with a recommendation and a deadline)

### 5.1 Simulator (decide in Q1, before M2)

PDF §4.1 requires: photorealistic rendering, CAD models of the DB drones, adjustable exposure /
image delay / camera-IMU offset, focal-length and range sweeps, three observer modes, a fixed
following controller in the loop.

Recommendation: **split the requirement in two**, because it is really two requirements —
- *Perception realism* (E1, E2, dataset v0): a photorealistic renderer with drone CAD models.
  Candidates: modern Gazebo (Harmonic) with PBR scenes, NVIDIA Isaac Sim, or Flightmare-style
  rendering. The repo's `scripts/gazebo.sh` targets Gazebo **classic 11** (EOL, weak
  photorealism) — treat it as legacy, do not build on it without re-evaluating.
- *Timing fidelity* (E3, E4, E5): a lightweight deterministic sensor-timing simulator (exact
  control of rates, delays, offsets, seeds; no rendering needed — measurements can be
  synthesised from geometry + the σw(w) noise model). Cheap, fast, exactly repeatable — this
  is where the ≥50-run consistency batteries and offset sweeps live.

Both publish the same topic contract, so the split is invisible downstream.

### 5.2 Two-Tello control infrastructure (decide before Q6)

Verified conflict: one host cannot run two driver processes (UDP :8889 bind), and both drones
in AP mode share 192.168.10.1.

- **Recommended:** target Tello is *not* on the perception host at all — flown scripted from a
  second laptop (or manually). Ground truth comes from mocap; the perception host never needs
  the target's telemetry. Zero new infrastructure.
- Alternatives if ROS-driven target control is wanted: (a) Tello EDU station mode — both
  drones join a lab AP with distinct IPs — plus either one composed process for both drivers
  or per-driver network namespaces; (b) two Wi-Fi adapters + Linux netns for two AP-mode
  drones. Both are real work; only do it if scripted-from-second-host proves insufficient.

### 5.3 Ground-truth bridge (decide before Q6)

Depends on the available lab: OptiTrack (NatNet → `mocap4ros2`) or Vicon (VRPN client).
Requirements either way: both rigid bodies tracked, ≥100 Hz, time-aligned to the ROS clock,
and the measured residual latency recorded in the dataset datasheet (PDF §4.3). Fallback for
early smoke tests only: the driver's mission-pad reference (Tello EDU, 10–20 Hz, ~cm) — not a
substitute for mocap in scored runs.

### 5.4 Detector/recognition backbone (decide in Q2)

Constraints from the proposal: classes = drone models + `unknown` (ADG-YOLO style), must
ingest an auxiliary motion channel (GLAD/YOLOMG style), must run in the 33 ms frame budget on
the ground laptop (`README.md` compute constraint). Start from a small YOLO-family detector
with an added recognition head; train on synthetic (sim) + real crops (Q2, PDF Table 3).

---

## 6. The dimension database (C1 artefact — design notes)

Schema per model (keyed to the recognition class names): `model_id`, width / length / height
(m), **measurement convention** (props included? arms folded?), manufacturer source URL,
retrieval date, tolerance. Plus the `unknown` entry returning a broad distribution over all
sizes (PDF §3.2). The annotation guideline for bounding boxes must state the same convention,
otherwise Eq. (1) is fed a W that does not match its w. The bias mechanism is exact: Eq. (1)
is linear in W, so a fractional mismatch between the annotated extent and the table width is
the same fractional range bias at every distance. For the Tello only the 98 mm body width is
officially specified; the prop tips extend the visible span beyond it by an amount that is
**not** in the spec and must be measured when the table is built — which is precisely what the
per-entry datasheet is for (PDF §4.3). Released with the dataset + datasheet.

---

## 7. Dataset recording spec (C1, PDF §4.3 verbatim requirements)

Every sequence stores: synchronised video, IMU/state stream, true poses of both aircraft,
target range and bearing, target model identity and viewing angle, box annotations, and the
measured stream latencies. Stratification axes: observer mode (hover / level / manoeuvre) ×
target behaviour (crossing / approaching / receding) × range band (spanning the px regime) ×
background. Recording is `rosbag2` of the full topic contract — the bag *is* the dataset
export source.

---

## 8. Quarter plan (PDF Table 3) mapped onto this repo

| Q | Proposal activities | Repo deliverables |
|---|---|---|
| Q1 | re-read ADG-YOLO/DroneDAR; dimension table v0; **E1 on downsampled public crops**; simulator setup | `rtr_dimension_db` v0; E1 notebook + accuracy-vs-size curve; simulator decision (§5.1) |
| Q2 | sim scenes + sensor models; train recognition head; range module | `rtr_sim` v0; `rtr_detect` v0; `rtr_range` v0; simulated dataset v0 |
| Q3 | E2 + E3 | `rtr_tracker` v0 (for E3); `rtr_experiments` config matrix; C2 results; first paper draft |
| Q4 | tracking filter with gyro switches; E4; filter-rate sweep | `rtr_motion_comp`, `rtr_observer_motion`, full `rtr_tracker`; C3 sim results |
| Q5 | E5 offset sweep; E6 closed-loop in sim; statistics | `rtr_follow` + `rtr_safety` (sim-exercised); full sim results |
| Q6 | Tello pair in mocap; latency measurement; hardware E2 + E4 | `rtr_gt_bridge`; namespace launch arg; hardware runs; dataset release |
| Q7 | optional evidence-gated extensions | inertial motion gate for a memory tracker — **or the RL controller arm, if adopted ([rl_analysis.md](rl_analysis.md))** |
| Q8 | thesis + journal paper; code/data release | — |

---

## 9. Risks (proposal's own, plus platform-verified additions)

From PDF §5.4: recognition may collapse at 40 px instead of 15 (→ E1 first, month one); the
Tello's slow state stream may hide the inertial effect (→ sim establishes it, hardware
reported as a lower bound); time synchronisation may dominate (→ E5 becomes the headline);
two closest papers appeared during writing (→ repeat the literature search at each milestone).

Platform-verified additions:
- **150–350 ms video latency** is far above the "tens of milliseconds" the delayed-filter
  literature assumes — it makes Switch S6 *more* valuable but means hardware E4/E6 numbers
  will be latency-dominated; measure and report per PDF §4.2.
- **Surrogate gyro noise** (~10 °/s from whole-degree 10 Hz attitude) may swamp the gyro-aiding
  benefit on hardware — this is exactly the PDF's "lower bound" framing; do not oversell.
- **Yaw free-runs** — no magnetometer; the repo's own figure is ~13° RMS over 60 s, measured
  in its simulation (README, Known limitations) — so bearing de-rotation must use
  short-horizon relative rotation, never absolute yaw.
- **Wi-Fi congestion with mocap + two drones + video** in one room: schedule a link budget
  check in Q6 week 1 (the driver's stream auto-restart is a mitigation, not a fix).
- **No true hardware kill-switch** on a Tello — record the honest equivalents
  (emergency motor-cut + battery pull) in the safety section of the thesis.
