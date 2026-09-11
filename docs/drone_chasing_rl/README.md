# Drone-Chasing-Drone with Deep RL — Baseline Architecture & Implementation Plan

**Source document.** Ziya Tan and Mehmet Karaköse, *"A new drone chasing drone approach based on
deep reinforcement learning with accelerated rewards"*, **SoftwareX 31 (2025) 102201**,
doi:10.1016/j.softx.2025.102201 (open access, CC BY). 13 pages. Local copy:
`A new drone chasing drone approach based on deep reinforcement learning.pdf`.

Reference implementation named by the paper (metadata C2/C3):
`github.com/ElsevierSoftwareX/SOFTX-D-24-00498` and `github.com/ziya44/Drone_tracking_with_drone`,
MIT licence, Python, Ubuntu 20.04.

**This directory contains design documents only — no implementation code**, by explicit decision.

## Why this is a new document set

The earlier set in [`../research_proposal/`](../research_proposal/) was built from a *different*
source (`preview_content-only.pdf`, "Recognise, Then Range" — model-based monocular ranging with
gyroscope aiding). That PDF is no longer in the working tree and the method here is not a variant
of it: **this paper learns the chase directly with DDPG from bounding-box geometry**, with no
dimension database, no pinhole ranging, and no inertial aiding. Mixing the two would confuse
which claims come from which source, so this is a parallel, self-contained set. The earlier
documents stay valid for their own source; what carries across is the **verified platform
baseline**, which is the same repository and the same aircraft.

Also worth recording: last session's verified finding was that the *previous* PDF contained no
reinforcement learning at all. This paper is the RL source, and it resolves that discrepancy.

## Documents

| File | Contents |
|---|---|
| [block_diagram.md](block_diagram.md) | Six block diagrams: the published approach, the hybrid control decomposition, DDPG internals, the ROS 2 node graph, the behaviour state machine, and the training→deployment path |
| [rl_foundations.md](rl_foundations.md) | **The deep explanation, from the basics**: MDP/POMDP, Bellman and TD learning, the actor-critic mechanism, why DDPG needs its replay buffer / target networks / noise / twin-critic machinery — every concept cashed out against this project's verified numbers, with a worked update, the assembled annotated algorithm, and a diagnostics table |
| [rl_specification.md](rl_specification.md) | **The main reference for the RL work**: state, action, reward, algorithm, hyperparameters, episode structure — what the paper specifies, what it leaves undefined, and a concrete recommended specification for each gap |
| [implementation_plan.md](implementation_plan.md) | ROS 2 packages and interfaces, mapping onto the existing `workspace/src` baseline, training and deployment phases, evaluation protocol, risks |
| [drl_ros2_reference_analysis.md](drl_ros2_reference_analysis.md) | Verified read of the `reiniscimurs/DRL-Robot-Navigation-ROS2` reference codebase: which methods transfer to the Tello and which cannot, two empirically-confirmed defects not to copy, and the consequences of **both drones being Tellos** — including the verified finding that the paper's box-ratio thresholds are unsafe at Tello scale |

## Why this paper fits this repository unusually well

Verified alignments, not assumptions:

| Paper requires | This repo has | Verified where |
|---|---|---|
| Follower = DJI Tello, 80 g, 13 min endurance | the same aircraft, with a working ROS 2 driver | Table A.1 vs [ryzerobotics.com/tello/specs](https://www.ryzerobotics.com/tello/specs) and `workspace/src/tello/` |
| 960 × 720 image frame, centre (480, 360) | `/image_raw` published at exactly 960 × 720 | `workspace/src/tello/resource/ost.txt`; frame centre recomputed = (480, 360) ✓ |
| Camera as the **only** input source | driver publishes `/image_raw` independently of IMU/odom — no inertial data needed | `node.py` publisher set |
| All compute on a central computer, drone streams only | the repo's stated architecture ("The drone streams; it does not compute") | `README.md` platform constraints |
| Command channel for up/down, yaw, forward/back | `/cmd_vel`, REP-103, normalised sticks → SDK ±100 | `node.py:1084-1092`, `_set_rc` clamp `node.py:1094-1097` |
| Wireless latency is the paper's **primary stated limitation**, unquantified | the same link, latency **measured**: 150–350 ms with tens of ms jitter | `README.md` platform constraints |

That last row is the most valuable one: the paper names its own dominant error source and does not
measure it. This repository already does, which turns a hand-waved limitation into a controlled
variable.

## Verification record

Every load-bearing claim was checked against the paper text, the repository, or the manufacturer's
specification. Nothing is taken from memory.

| Claim | Verified against |
|---|---|
| YOLOv3 → bounding box → DDPG agent; hybrid control; central computer off-board | Paper §2.1, §2.1.1, §2.1.2 (full text extracted and read) |
| Frame 960×720, centre (480, 360), forward/back thresholds 20 % / 55 % of the box-to-screen ratio | Paper §2.1.2 |
| Reward Eq. (5)–(6): Euclidean bbox-centre-to-frame-centre distance, threshold 110 units | Paper §2.2.3 |
| Hyperparameters: actor/critic LR 1e-4, γ = 0.99, σ = 0.3, 10 steps/episode, 100 000 episodes, interval 100 | Paper Table 2 |
| Selected hyperparameter set → total reward −4.2, 25 000 episodes, high stability | Paper Table 3 |
| YOLOv3: 1 class, 100 epochs, batch 20, LR 1e-4, ≈95 % accuracy, weaker on small targets | Paper Appendix A |
| Five scenarios and their tracking/detection rates (99/95, 95/94, 86/90, 82/90, 42/60) | Paper Table 4 + Appendix A narrative |
| Brightness study covers **only** scenarios 2 and 3 | Paper Table 5 |
| Max centre-to-corner distance in a 960×720 frame = **exactly 600 px**, so the 110-px reward threshold is 18.3 % of the maximum | recomputed: √(480² + 360²) = 600.0 |
| Reward Eq. (6) is **discontinuous** at dist = 110 (−10 inside vs −27.5 outside) and **undefined** exactly at 110 | recomputed from the paper's own formula |
| Tello real max speed is **8 m/s**, not the 8 cm/s of Table A.1; weight (80 g) and endurance (13 min) in the same table match the spec exactly | [ryzerobotics.com/tello/specs](https://www.ryzerobotics.com/tello/specs) |
| Follower max speed < target max speed (8 vs 15 in the paper's own units) | Paper Table A.1 |
| Repo has **no** existing detection, YOLO, RL, or Gym code — only the driver, messages, keyboard control, and the (excluded) VIO stack | grep over `workspace/src` for yolo/darknet/ddpg/gym/torch/tensorflow |
| Statistics: 5-scenario means 80.8 % / 85.8 % and their sample standard deviations reproduce exactly | recomputed with `statistics` from Table 4 |

### Inconsistencies found in the paper (they affect implementation)

These are stated plainly because each one forces a decision. Details and recommended resolutions
are in [rl_specification.md](rl_specification.md).

| # | Inconsistency | Where |
|---|---|---|
| 1 | **Which DoF the agent controls**: §1 says "up-down and yaw"; §2.1.2 says "up, down, right, left" | §1 vs §2.1.2 |
| 2 | **What the agent observes**: abstract says box *size*; §2.1 says "position, orientation, and speed of the followed drone"; §2.2.3 uses the box *centre*. The paper also states the camera is the only input, so orientation/speed of the target are not directly measurable | abstract vs §2.1 vs §2.2.3 |
| 3 | **Reward maximum**: prose says the reward at the centre is "0 or close to it"; Eq. (6) gives **100** at dist = 0 | §2.2.3 |
| 4 | **Reward discontinuity/undefined point** at dist = 110 (verified numerically) | Eq. (6) |
| 5 | **Training length**: Table 2 says 100 000 episodes, Table 3 says 25 000, Conclusions say "after 500 training steps" | Table 2 vs Table 3 vs §5 |
| 6 | **Dataset size**: §1 says ≈5000 images, Appendix A says ≈6500 | §1 vs Appendix A |
| 7 | **Tello max speed** given as 8 cm/s; the official spec is 8 m/s (same table's weight and endurance match the spec exactly) | Table A.1 vs manufacturer |
| 8 | **Dispersion figures for tracking accuracy do not reproduce**: the two *detection* rows recompute exactly (S2: s = 2.52, s² = 6.33; S3: s = 4.04, s² = 16.33), but the two *tracking* rows (reported S2 s = 2.08, s² = 4.33; S3 s = 7.64, s² = 58.33) match neither the sample nor the population estimator on Table 5's own values | §3 vs Table 5, recomputed |
| 9 | **γ = 0.99 against a 10-step episode**: the discount's effective horizon (~100 steps) is an order of magnitude longer than an episode, so γ has almost no effect on the return | Table 2 |
| 10 | **Box-ratio definition undefined** — "ratio of the detected bounding box to the primary screen size" does not say area or linear, and the two imply materially different standoff distances | §2.1.2 |

None of these invalidate the approach. They are the decisions a faithful reimplementation has to
make explicitly, and each is given a recommended resolution in the RL specification.
