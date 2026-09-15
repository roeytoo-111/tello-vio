# Implementation Plan — Drone Chasing Drone on ROS 2

Companion to [block_diagram.md](block_diagram.md) and [rl_specification.md](rl_specification.md).
Planning only — no code on this branch. Tags: **[P]** paper, **[V]** verified here, **[D]** design
decision.

---

## 1. Verified platform baseline

Everything the paper needs from the aircraft already exists in `workspace/src`. Re-verified this
session, with references.

| Capability the paper requires | Status | Verified where |
|---|---|---|
| Video at 960 × 720 | **exists** — `/image_raw`, `bgr8`, ~30 Hz target, duplicate frames suppressed, stream auto-restart on stall | `tello/resource/ost.txt` (width 960, height 720); driver video path |
| Frame centre (480, 360) | **matches exactly** | recomputed from the above |
| Camera intrinsics (needed only for calibrating the depth-rule thresholds) | **exists** — fx ≈ 919.42 px, correctly rescaled if `video_scale` ≠ 1 | `ost.txt`; `node.py:971-1024` |
| Velocity command channel | **exists** — `/cmd_vel`, REP-103, normalised sticks in [−1, 1] scaled to SDK ±100 | `node.py:1084-1092`; clamp `node.py:1094-1097` |
| Command watchdog | **exists** — RC resent at 20 Hz; a command stale by > 0.35 s is zeroed | `node.py:1099-1114` |
| Abort path | **exists** — `/emergency` sends a fire-and-forget datagram that bypasses the command queue, plus a retried copy; `/land`; SDK auto-lands after 15 s of RC silence | `node.py:1046-1059` |
| Manual override | **exists** — keyboard GUI with takeoff/land/emergency and manual sticks | `tello_control/src/main.cpp` |
| Off-board compute model | **exists and is the repo's stated architecture** | `README.md` platform constraints |
| Link latency quantified | **measured: 150–350 ms**, jitter of tens of ms — the paper's stated dominant limitation, unquantified there | `README.md` platform constraints |
| Detection / RL / Gym code | **none** — verified absent | grep over `workspace/src` for yolo, darknet, ddpg, gym, torch, tensorflow |

**Aircraft facts** [V, manufacturer]: 80 g, max flight time 13 min, **max speed 8 m/s**
([ryzerobotics.com/tello/specs](https://www.ryzerobotics.com/tello/specs)). The paper's Table A.1
matches weight and endurance exactly but prints the speed as 8 cm/s — a 100× unit error. Treat
every speed figure in the paper as unit-ambiguous and derive your own operating envelope; note the
repository has been bitten by exactly this class of error before (commit `7539d95`, *"Fix the
velocity scale: vg\* is cm/s, not dm/s — the cause of the runaway"*), which is a good reason to
verify units on this platform rather than inherit them.

**Not used by this method** [P + D]: IMU, odometry, ToF, mission pads, transform tree, and the
entire `tello_vio` estimator. The paper's method is camera-only, so the VIO/SLAM work on this
repository is genuinely irrelevant here, exactly as scoped.

---

## 2. Packages to build

One package per functional block, so each of the paper's stages can be disabled, swapped, or
replayed independently [D, following the block diagram].

| Package | Role | Key parameters |
|---|---|---|
| `chase_msgs` | the interface contract: `DroneDetection` (box, confidence, `t_capture`, `t_receive`), `TrackingState` (errors, dist, ratio, visibility, staleness), `PolicyAction` (raw action + mapped Twist, checkpoint id) | — |
| `chase_detector` | YOLO inference on `/image_raw` → `DroneDetection` | model path, score threshold, input size, device |
| `chase_state` | box → normalised observation; owns the observation definition and the visibility/staleness logic | `observation_mode: faithful \| latency_aware`, normalisation constants |
| `chase_policy` | loads the trained actor, runs **inference only**, maps action → Twist | checkpoint path, `horizontal_axis: yaw \| lateral`, rate limit, policy rate |
| `chase_depth_rule` | box-ratio thresholds → forward / hold / backward | `ratio_mode: linear \| area`, `near`/`far` thresholds, forward/back speeds |
| `chase_behaviour` | TRACK / REACQUIRE / PATROL state machine and the search pattern | lost-frame count, loss timeout, search yaw rate, patrol pattern |
| `chase_mixer` | assembles one Twist from policy + depth rule + behaviour | axis ownership map |
| `chase_safety` | engage gate, speed cap, auto-land on link loss / low battery / prolonged loss | volume limits, `v_max`, battery floor |
| `chase_gym` | the training environment: image-plane dynamics, target-motion generators, latency injection, detection noise | all randomisation ranges |
| `chase_train` | DDPG (and TD3/SAC arms) trainer; emits a versioned checkpoint | Table 2 hyperparameters + §8 additions |
| `chase_eval` | offline scoring from rosbag2: time-in-view %, detection %, latency distribution, per-seed statistics | scenario definitions |

**Reused unchanged**: `tello` driver, `tello_msg`, `tello_control`. **Only baseline change
eventually needed** [V]: a `namespace` launch argument for `tello.launch.py` if two aircraft are
ever driven from one host — the driver's topic names are already relative
(`node.py:355-381`), so nothing else has to change. Not required for this method, since the target
is flown manually [P Appendix A: *"The following drone was a standard drone that can be controlled
with a remote control"*].

---

## 3. Interface contracts

**Every message carries `t_capture` and `t_receive`** [D]. This is not ceremony: the paper's
dominant failure mode is latency, and the only way to attribute an error to latency rather than to
the policy is to have both timestamps on every hop. The driver already stamps video on arrival and
telemetry at change-detection with a 20 ms bound [V], so the hardware half of this is in place.

**Axis ownership is explicit and exclusive** [D]:

| Axis | Owner | Source |
|---|---|---|
| `linear.z` (up/down) | `chase_policy` | learned [P §1] |
| `angular.z` (yaw) | `chase_policy` | learned [P §1] |
| `linear.x` (forward/back) | `chase_depth_rule` | hand-coded [P §2.1.2] |
| `linear.y` (lateral) | unused by default | left free by the paper; ablation only |

The mixer enforces this. Two controllers writing one axis is the classic way to get an
uncontrollable aircraft, and it is worth making structurally impossible rather than relying on
discipline.

---

## 4. Phases

### Phase 0 — Dataset and detector (no flight)

[P Appendix A] YOLOv3, one class, 100 epochs, batch 20, LR 1e-4, ≈95 % accuracy, on ≈5000 [P §1]
or ≈6500 [P Appendix A] images — the paper states both. The reported weakness is explicit:
*"the accuracy rate given for the detected drone is lower in small images"*, with the paper's own
remedy being more small-target images in the training set.

**[D] Recommendations**: (a) record your own images from this exact camera — the detector's job is
this drone, at this resolution, in this room; (b) use a current small model (YOLOv8/v11-nano
class) rather than YOLOv3 from 2018 — both accuracy on small objects and inference cost have
improved substantially, and the ~33 ms frame budget on the ground machine [V `README.md`] is real;
(c) report accuracy **stratified by target pixel size**, because that is where the paper's
detector degrades and where the whole system's failures originate.

Deliverable: a detector, plus an accuracy-versus-target-size curve.

### Phase 1 — Geometry, depth rule, and a P-controller baseline (flight, no RL)

Build `chase_state`, `chase_depth_rule`, `chase_mixer`, `chase_safety`, and a **proportional
controller** on the pixel error. Fly the five scenarios.

**[D] Why this comes before any RL.** It gives a working chase, exercises the whole safety chain,
produces the latency and target-dynamics measurements the simulator needs, and — most importantly
— produces the baseline number that makes the RL result interpretable. The paper never compares
against a classical controller, so "the policy achieves 95 %" currently has no denominator.

Deliverable: baseline scenario results, a measured latency distribution, and a step-response
estimate of the follower's velocity lag.

### Phase 2 — Simulator

Build `chase_gym` per [rl_specification.md](rl_specification.md) §9, with the Phase 1
measurements baked in: real latency distribution, real lag constant, detection dropout matched to
the Phase 0 curve. **Validate it** by replaying a recorded flight's box trajectory and comparing.

> **UPDATE (2026-09-15).** Phase 2 is now the first tier of a three-tier architecture — Gazebo
> (lockstep physics tier) and Unreal/Colosseum (perception tier: YOLO dataset factory +
> vision-in-the-loop evaluation) are fully specified in
> [sim_training_architecture.md](sim_training_architecture.md), together with the
> FOLLOW→INTERCEPT task extension and its PN baseline.

Deliverable: a validated environment, with the validation plot.

### Phase 3 — Training

Train DDPG with the paper's Table 2 hyperparameters as the fidelity arm, then the repaired reward
(§6) and the corrected episode length (§8). Multiple seeds throughout. Optionally add TD3/SAC as a
second learner arm.

Deliverable: checkpoints versioned with their observation/action specification, plus convergence
curves per arm and seed.

### Phase 4 — Deployment and the scenarios

`chase_policy` loads a checkpoint and runs inference behind `chase_safety`. Fly the paper's five
scenarios plus the brightness study for scenarios 2 and 3 [P Table 5].

**[D] Staged flight ladder**, adapted from ordinary practice and the paper's own indoor setup:
policy in simulation → policy in flight against a **stationary** target → slow target → full
scenarios. The safety supervisor and manual override are live throughout.

Deliverable: the comparison table — P-controller vs paper-faithful DDPG vs repaired DDPG — across
five scenarios and three brightness levels, with per-seed spread.

---

## 5. Safety

The paper says nothing about fail-safes. **[D]** The following comes from ordinary practice plus
what the platform already provides, and costs nothing to state:

| Mechanism | Owner | Status |
|---|---|---|
| Manual override and motor-cut at any moment | keyboard GUI → driver `/emergency` | **exists** [V `node.py:1046-1059`] |
| Command watchdog — stale command zeroed in 0.35 s | driver | **exists** [V `node.py:1099-1114`] |
| Auto-land after 15 s of command silence | Tello SDK | **exists** [V] |
| Engagement gate — policy output ignored until armed | `chase_safety` | to build |
| Speed cap on the assembled command | `chase_safety` | to build |
| Auto-disarm on link loss, low battery, prolonged target loss | `chase_safety` | to build |
| Flight-volume limit | `chase_safety` | to build — note there is no absolute position source in this method, so this is necessarily an operator-enforced boundary plus a conservative speed cap, not a true geofence |

**[D] The honest limitation**: because this method estimates no metric position, a software
geofence is not available in the way it would be with a motion-capture system. That makes the
speed cap, the short flight volume, and the human override the real safety mechanisms, and they
should be described that way rather than overclaimed. The Tello has no external hardware kill
line; the equivalents are the emergency motor-cut and pulling the battery.

---

## 6. Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| **The follower cannot out-run the target** | [P Table A.1] 8 vs 15 in the paper's units — the chaser is about half as fast | cap the target's speed in the protocol, or report the speed ratio with every result so kinematics and control can be separated |
| **Latency dominates the control loop** | 150–350 ms measured [V]; the paper names this as its primary limitation twice | train with injected latency; include the last action in the observation; report per-run latency |
| **Small targets break the detector** | [P Appendix A] accuracy drops on small images; the system's failures start here | stratified accuracy reporting; more small-target training data; the depth rule keeps the target large in-frame, which helps |
| **Reward discontinuity distorts training** | verified: an ~18-unit cliff at dist = 110 | use the repaired continuous reward; keep the original as a comparison arm |
| **DDPG seed variance** | well known, and the paper reports single numbers; its own comparison table cites a study where DQN beat DDPG [P Table 1, ref 29] | ≥ 5 seeds with reported spread; TD3/SAC as a second arm |
| **Unvalidated simulator** | the paper never describes its Gym environment's dynamics | validate against recorded flight before trusting any policy |
| **Unit ambiguity in speeds** | verified 100× error in [P Table A.1]; this repo has had a similar bug before (`7539d95`) | derive the envelope from your own flight tests; never inherit the paper's speed numbers |
| **Wi-Fi congestion indoors** | the paper attributes bounding-box lag to *"the capacity of the wireless communication"* | measure the link during every run; the driver's stream auto-restart is a mitigation, not a fix |

---

## 7. What would make this more than a reproduction

Four things, each cheap and each addressing something the paper leaves open:

1. **Quantify the latency and train against it.** The paper blames latency for its failures without
   measuring it; this platform measures it [V]. Turning the headline limitation into a controlled
   training variable is the strongest available contribution.
2. **Add the missing baseline.** A proportional controller on the same error signal answers
   whether the learned policy is actually earning its complexity.
3. **Fix and ablate the reward.** The discontinuity at dist = 110 is verifiable from the paper's
   own formula; measuring what it costs is a clean, self-contained result.
4. **Separate kinematics from control.** Reporting the follower/target speed ratio alongside
   tracking accuracy would show how much of Scenario 5's 42 % is a control failure and how much is
   simply a slower aircraft — which the paper's presentation cannot distinguish.
