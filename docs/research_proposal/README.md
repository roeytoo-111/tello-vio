# Research Proposal — Baseline Architecture & Implementation Plan

**Branch purpose.** This branch (`feat/research-proposal`) plans the ROS 2 implementation of the
research proposal *"Recognise, Then Range: Model-Based Monocular Distance Estimation and
Gyroscope-Aided Tracking of Small Drones from a Moving Drone"*
(source document: `preview_content-only.pdf`, version of 2 Sep 2026, 16 pages).
It contains **design documents only — no implementation code**, by explicit decision.

The VIO/SLAM estimator developed on `feat/visual-inertial-odometry` (`tello_vio` package) is
**not part of this architecture**. What *is* reused is the hardware baseline:
the `tello` driver, `tello_msg` messages, and `tello_control` keyboard GUI in `workspace/src/`.

## Documents

| File | Contents |
|---|---|
| [block_diagram.md](block_diagram.md) | The system block diagrams: proposal pipeline, ROS 2 node graph (hardware and simulation), safety chain, ablation switchboard |
| [implementation_plan.md](implementation_plan.md) | Verified platform baseline, packages/interfaces to build, experiment → module mapping (E1–E6), simulation choice, two-drone infrastructure, quarter plan, risks |
| [rl_analysis.md](rl_analysis.md) | **Verified finding: the proposal contains no reinforcement-learning method.** What the proposal actually specifies for closed-loop control, and a clearly-marked RL extension design (states / actions / reward / algorithm) in case a learned controller is intended |

## Verification record

Every load-bearing claim in these documents was checked against a primary source.
Nothing is assumed from memory.

| Claim | Verified against |
|---|---|
| Proposal pipeline, method equations (1)–(4), hypotheses H1–H4, experiments E1–E6, quarterly plan | `preview_content-only.pdf` §1.2, §3, §3.6, §5 (Table 1), §7 (Table 3) — full text extracted and read |
| The proposal contains **no RL component**; the only occurrence of "reinforcement" is the Henderson et al. [27] citation about seed statistics (§5.3) | Keyword sweep of the full extracted text (`reinforcement`, `reward`, `policy`, `MDP`, `PPO`, `SAC`, `DQN`, `actor`, `critic`, `agent`, …) — sole hit is the [27] citation; all other hits are substrings of "f**actor**" |
| Closed-loop controller is a **fixed** simple follower, deliberately not part of what is tested | PDF §4.1: "A simple following controller closes the loop; the controller is held fixed across all perception and estimation variants, because it is not part of what is being tested." |
| The Tello exposes **no gyroscope** — fused attitude only, whole degrees, ~10 Hz | `workspace/src/tello/tello/node.py:569-574` (`angular_velocity_covariance[0] = -1`), `README.md` platform-constraints table |
| Video: 960×720, ~30 Hz, `bgr8`, 150–350 ms Wi-Fi latency, arrival-stamped | `workspace/src/tello/resource/ost.txt`, `node.py` video path, `README.md` |
| Camera focal length fx ≈ 919.4 px | `workspace/src/tello/resource/ost.txt` camera matrix |
| `/cmd_vel` is REP-103, normalised sticks ∈ [−1, 1], scaled to SDK ±100; RC sent at 20 Hz with a 0.35 s dead-man that zeroes stale commands | `node.py:1084-1114` |
| Emergency has a queue-bypassing fire-and-forget path plus a retried queued path | `node.py:1046-1059` |
| A second Tello driver process on the same host cannot start: djitellopy binds UDP :8889 in its constructor (EADDRINUSE) | `node.py:205-220` |
| Both Tellos in AP mode present the identical IP 192.168.10.1 | `node.py` default parameter + driver network pre-flight check |
| Mission pads (Tello EDU only) give an absolute drift-free reference at 10–20 Hz | `tello_msg/msg/TelloMissionPad.msg`, driver `_publish_mission_pad` |
| Driver topics use relative names → the whole driver is namespace-able for a two-aircraft setup | `node.py:355-381` (`create_publisher('image_raw', …)` etc.) |
| Gazebo 11 (classic) is the simulator the repo has provisioning for | `scripts/gazebo.sh` |

## The one sentence that matters before reading further

The proposal's science is **perception and estimation** (contributions C1–C3), measured through
ablation switches under a *fixed* controller. If a reinforcement-learning controller is wanted,
it is an **extension** (the proposal's Q7 slot is the natural home) and must be evaluated as a
separate arm of experiment E6 — see [rl_analysis.md](rl_analysis.md) before planning any RL work.
