# Baseline Block Diagram — Drone Chasing Drone with DDPG

Six views, from the published concept to the ROS 2 wiring. Source for every claim marked
**[P]** is the paper (Tan & Karaköse, SoftwareX 31 (2025) 102201, section cited); **[V]** is
verified in this repository or against the manufacturer specification; **[D]** is a design
decision of this plan, with its derivation given.

1. [The published approach](#1-the-published-approach-p) — what the paper actually does
2. [The hybrid control decomposition](#2-the-hybrid-control-decomposition--the-papers-key-structural-choice) — which degrees of freedom are learned and which are hand-coded
3. [DDPG internals and the training loop](#3-ddpg-internals-and-the-training-loop-p-§222)
4. [ROS 2 node graph](#4-ros-2-node-graph--deployment) — deployment on this repository's baseline
5. [Behaviour state machine](#5-behaviour-state-machine-p-§21-item-3)
6. [Training → deployment path](#6-training--deployment-path)

Colour code: **green** = exists and verified in `workspace/src` · **orange** = to be built ·
**blue** = physical world / sensor / actuator · **purple** = offline training only · grey =
standard machinery.

---

## 1. The published approach [P]

The paper's system has three physical components [P §2.1]: the follower drone, the followed
drone, and a central computer; *"The follower drone is wirelessly connected to the central
computer"* and all perception and control run on that computer. The loop below is Fig. 1 and
Fig. 2 of the paper, expanded to the level of detail an implementation needs.

```mermaid
flowchart TB
    subgraph WORLD["Physical world — indoor sports hall in the paper"]
        TGT["<b>Followed drone</b><br/>standard quadcopter, 147 g<br/>remote-controlled by a human<br/>max speed 15 (paper units)"]:::phys
        FOL["<b>Follower drone — DJI Tello</b><br/>80 g · 13 min · no onboard compute<br/>camera 960 x 720 · max speed 8"]:::phys
    end

    subgraph CC["Central computer — all perception and control"]
        YOLO["<b>YOLOv3 detector</b><br/>1 class 'drone' · 100 epochs<br/>batch 20 · LR 1e-4 · ~95% accuracy<br/>weaker on small targets"]:::new
        GEOM["<b>Bounding-box geometry</b><br/>centre (xc, yc) of the box<br/>error vs frame centre (480, 360)<br/>dist = sqrt(ex^2 + ey^2), range 0..600 px<br/>box-to-screen ratio"]:::new
        DDPG["<b>DDPG policy — LEARNED</b><br/>continuous action<br/>vertical + horizontal correction<br/>trained offline in an OpenAI Gym env"]:::new
        RULE["<b>Depth rule — HAND-CODED</b><br/>ratio &lt; 20% : move forward<br/>20% .. 55% : hold<br/>ratio &gt; 55% : move backward"]:::new
        MIX["<b>Command assembly</b><br/>one velocity command<br/>from the two controllers"]:::new
        FSM["<b>Search / rescan logic</b><br/>no detection : hold, then rotate<br/>on a predefined pattern<br/>prolonged loss : patrol mode"]:::new
    end

    TGT -- "enters field of view" --> FOL
    FOL -- "video over Wi-Fi<br/>150-350 ms latency (measured in this repo)" --> YOLO
    YOLO -- "bounding box + confidence" --> GEOM
    YOLO -- "no detection" --> FSM
    GEOM -- "centre error ex, ey" --> DDPG
    GEOM -- "box-to-screen ratio" --> RULE
    DDPG -- "up/down + yaw" --> MIX
    RULE -- "forward/back" --> MIX
    FSM -- "search command" --> MIX
    MIX -- "velocity command over Wi-Fi" --> FOL
    FOL -- "follower moves, image changes" --> FOL

    classDef phys fill:#e8eef8,stroke:#4a6fa5,color:#1d3050
    classDef new fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
```

**The objective in one sentence** [P §2.1.2]: *"the difference between the red and blue dots is
minimized"* — the red dot is the detected box centre, the blue dot is the frame centre. The whole
control problem is driving that pixel error to zero while holding the box size inside a band.

**What is genuinely load-bearing here**, and easy to miss on a first read:

- The system never estimates metric distance. There is no ranging, no depth, no pose. Distance is
  regulated purely by the **apparent size** of the box against two thresholds. That is why the
  approach needs no IMU, no gyroscope, and no VIO — and why the VIO/SLAM stack on this repository
  is genuinely irrelevant to it.
- The loop is closed **through the image**, so every latency in the link appears directly as
  control delay. The paper identifies this as its main limitation twice (§4.1 and Appendix A:
  *"the latency caused by image processing and wireless communication … led to occasional
  inaccuracies in real-time tracking, especially in high-speed scenarios"*) but never quantifies
  it. This repository measures it at **150–350 ms** [V].
- **The follower is slower than the target** [P Table A.1]: 8 versus 15 in the paper's own units.
  A chaser that cannot out-run its quarry will lose it whenever the quarry commits to a direction,
  which is exactly what the worst scenario reports (42 % tracking at high target speed).

---

> **Our setup differs from the paper's here.** The diagram above describes the *published*
> system, whose target was a 147 g quadcopter flown by hand. In our project **both aircraft are
> DJI Tellos**, which (a) removes the paper's follower-slower-than-target handicap, (b) requires
> the target to be commanded rather than hand-flown, and (c) invalidates the 20 % / 55 % box-ratio
> thresholds at Tello scale. See
> [drl_ros2_reference_analysis.md §6](drl_ros2_reference_analysis.md#6-your-clarification-the-followed-drone-is-also-a-dji-tello).

## 2. The hybrid control decomposition — the paper's key structural choice

This is the single most important thing to understand before implementing, and the paper states
it in a single sentence [P §1]: *"The DDPG agent determines the up-down and yaw movements of the
drone, and the forward-backward movements are determined by the size of the box framing the
detected drone."* The controller is **not** end-to-end. Two degrees of freedom are learned, one is
a hand-written rule, and one is unused.

```mermaid
flowchart LR
    IMG["Detected bounding box<br/>centre (xc, yc) + size"]:::src

    subgraph LEARNED["LEARNED — DDPG, continuous action"]
        VERT["<b>Vertical</b><br/>driven by ey = yc - 360<br/>maps to linear.z"]:::rl
        HORZ["<b>Horizontal</b><br/>driven by ex = xc - 480<br/>maps to angular.z (yaw)<br/>see ambiguity note"]:::rl
    end

    subgraph CODED["HAND-CODED — threshold rule"]
        DEPTH["<b>Depth / standoff</b><br/>driven by box-to-screen ratio<br/>20% and 55% thresholds<br/>maps to linear.x"]:::rule
    end

    subgraph UNUSED["NOT USED"]
        LAT["<b>Lateral translation</b><br/>linear.y<br/>left free by the paper"]:::none
    end

    IMG --> VERT
    IMG --> HORZ
    IMG --> DEPTH
    IMG -.-> LAT

    VERT --> CMD["Single velocity command<br/>to the follower drone"]:::out
    HORZ --> CMD
    DEPTH --> CMD

    classDef src fill:#e8eef8,stroke:#4a6fa5,color:#1d3050
    classDef rl fill:#f3e8fd,stroke:#8e44ad,color:#3d1e57
    classDef rule fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
    classDef none fill:#eef0f2,stroke:#8a97a0,color:#3b4750
    classDef out fill:#e2f4e3,stroke:#2e8b57,color:#14401f
```

**Why the split exists, as far as the paper explains it.** Box size is a monotone, low-noise proxy
for distance that needs no learning, so spending policy capacity on it would be wasteful; the
centring problem, by contrast, is a continuous two-axis regulation task under delay, which is what
DDPG is for. The consequence for implementation is that **the reward function only ever sees the
centring error** [P §2.2.3] — the depth rule is outside the learning problem entirely and cannot
be improved by training.

**Ambiguity #1, and how to resolve it.** §1 says the agent controls *"up-down and yaw"*; §2.1.2
says *"other actions (up, down, right, left) are determined by the agent"*. These disagree on
whether the horizontal degree of freedom is a yaw rotation or a lateral translation. **[D]
Recommendation: implement yaw.** Three reasons: §1 is the paper's own summary of its contribution
and names yaw explicitly; yaw rotates the camera and therefore moves the target in the image
without changing the standoff distance the depth rule is simultaneously regulating, so the two
controllers stay decoupled; and the paper's own search behaviour rotates the aircraft about its
axis [P Appendix A, Scenario 3], which only makes sense if yaw is the re-centring authority.
Keep lateral translation as a documented ablation.

---

## 3. DDPG internals and the training loop [P §2.2.2]

DDPG is *"an actor-critic method … which uses two separate neural networks, one for the 'actor'
(policy) and one for the 'critic' (value function)"* and *"uses a form of off-policy learning …
learn from a replay buffer that stores many past experiences"* [P §2.2.2]. The diagram shows the
data path the paper's equations (3) and (4) describe, plus the pieces the paper leaves undefined
(dashed borders) that an implementation must nevertheless choose.

```mermaid
flowchart TB
    subgraph ENV["OpenAI Gym environment — offline training only"]
        SIM["<b>Chase environment</b><br/>simulates target motion in the image plane<br/>state = box centre error<br/>episode = 10 steps (Table 2)"]:::train
        REW["<b>Reward — Eq. (6)</b><br/>dist &lt; 110 : r = 100 - dist<br/>dist &gt; 110 : r = -0.25 * dist<br/>discontinuous at 110 — see note"]:::train
    end

    subgraph AGENT["DDPG agent"]
        ACTOR["<b>Actor</b> mu(s | theta_mu)<br/>deterministic policy<br/>outputs continuous action"]:::rl
        NOISE["<b>Exploration noise</b><br/>sigma = 0.3<br/>process type not specified"]:::gap
        CRITIC["<b>Critic</b> Q(s, a | theta_Q)<br/>action-value estimate"]:::rl
        TACTOR["<b>Target actor</b> mu'"]:::rl
        TCRITIC["<b>Target critic</b> Q'"]:::rl
        BUF["<b>Replay buffer</b><br/>off-policy experience<br/>capacity + batch size<br/>not specified"]:::gap
    end

    LOSS["<b>Critic update — Eq. (3) and (4)</b><br/>y_i = r_i + gamma * Q'(s_i+1, mu'(s_i+1))<br/>L = mean of (y_i - Q(s_i, a_i))^2<br/>gamma = 0.99 · LR = 1e-4"]:::rl
    POLUP["<b>Actor update</b><br/>deterministic policy gradient<br/>ascend Q with respect to the action<br/>update rule not written in the paper"]:::gap
    SOFT["<b>Target network update</b><br/>soft update rate tau<br/>not specified"]:::gap

    SIM -- "state s" --> ACTOR
    ACTOR --> NOISE
    NOISE -- "action a" --> SIM
    SIM --> REW
    REW -- "reward r, next state s'" --> BUF
    NOISE -- "a" --> BUF
    SIM -- "s" --> BUF
    BUF -- "minibatch" --> LOSS
    TACTOR --> LOSS
    TCRITIC --> LOSS
    LOSS -- "gradient" --> CRITIC
    CRITIC --> POLUP
    POLUP -- "gradient" --> ACTOR
    ACTOR -. "soft copy" .-> TACTOR
    CRITIC -. "soft copy" .-> TCRITIC
    SOFT -.-> TACTOR
    SOFT -.-> TCRITIC

    classDef train fill:#f0e6fa,stroke:#7d4fa6,color:#3d1e57
    classDef rl fill:#f3e8fd,stroke:#8e44ad,color:#3d1e57
    classDef gap fill:#fdf0dc,stroke:#d9822b,color:#5c3a10,stroke-dasharray: 5 4
```

**Dashed blocks are gaps in the paper.** Network architecture, replay-buffer capacity, minibatch
size, target-update rate τ, optimiser, and the exploration-noise process are all unspecified.
Recommended values for each are in [rl_specification.md](rl_specification.md) §6 — these must be
chosen and *recorded*, or the result is not reproducible.

**Verified defect in the reward.** Evaluating Eq. (6) at the boundary gives −10 just inside the
threshold and −27.5 just outside: an ~18-unit cliff at dist = 110, and no value defined exactly at
110 because both branches use strict inequalities. A continuous repair that preserves the paper's
intent (steep gradient near the centre, gentle far away) is given in the RL specification.

**Verified scale mismatch.** γ = 0.99 has an effective horizon near 100 steps, while an episode is
10 steps [P Table 2]. The discount therefore barely influences the return, and the agent is
effectively near-myopic within an episode whatever γ says.

---

## 4. ROS 2 node graph — deployment

Green nodes exist and are verified in `workspace/src`; orange nodes are to be built. Every edge
label is a verified property, not a nominal one. Note what is **absent**: no IMU, no odometry, no
transform tree, no state estimator — the paper's method does not use them, which is why the
VIO/SLAM stack on this repository plays no part.

```mermaid
flowchart TB
    subgraph AIR["Aircraft"]
        TELLO["<b>Follower — DJI Tello</b>"]:::phys
        TARGET["<b>Target — also a DJI Tello</b><br/>commanded for repeatable scenarios<br/>needs its own driver on a SECOND HOST<br/>(UDP :8889 bind + shared 192.168.10.1)"]:::phys
    end

    DRV["<b>tello driver — EXISTS</b><br/>workspace/src/tello<br/>RC resent at 20 Hz · dead-man 0.35 s<br/>emergency bypass path"]:::exist

    DET["<b>drone_detector — NEW</b><br/>YOLO inference<br/>publishes box + confidence<br/>stamped with capture and receive time"]:::new
    VS["<b>visual_state — NEW</b><br/>box centre, error ex/ey, dist,<br/>box-to-screen ratio, visibility<br/>builds the policy observation"]:::new
    POL["<b>ddpg_policy — NEW</b><br/>trained actor, inference only<br/>observation to continuous action<br/>no learning in flight"]:::new
    RULE["<b>depth_rule — NEW</b><br/>ratio thresholds 20% / 55%<br/>forward / hold / backward"]:::new
    BEH["<b>behaviour_manager — NEW</b><br/>TRACK / REACQUIRE / PATROL<br/>owns the search pattern"]:::new
    MIX["<b>command_mixer — NEW</b><br/>assembles one Twist from<br/>policy + depth rule + behaviour"]:::new
    SAFE["<b>safety_supervisor — NEW</b><br/>engage gate · speed cap ·<br/>auto-land on link loss / low battery"]:::new
    KB["<b>tello_control GUI — EXISTS</b><br/>manual override + emergency key"]:::exist
    BAG["<b>rosbag2</b><br/>records every run for offline scoring"]:::grey

    TARGET -. "seen by the camera" .-> TELLO
    TELLO -- "video" --> DRV
    DRV -- "/image_raw · 960x720 · bgr8 · ~30 Hz<br/>150-350 ms Wi-Fi latency" --> DET
    DRV -- "/camera_info · fx 919.4 px" --> VS
    DET -- "/chase/detection" --> VS
    DET -- "no detection" --> BEH
    VS -- "/chase/observation" --> POL
    VS -- "box ratio" --> RULE
    VS -- "visibility + staleness" --> BEH
    POL -- "vertical + yaw action" --> MIX
    RULE -- "forward / back" --> MIX
    BEH -- "mode + search command" --> MIX
    MIX -- "/chase/cmd_raw" --> SAFE
    SAFE -- "/cmd_vel · REP-103 sticks in [-1,1]<br/>must publish at 10 Hz or faster<br/>or the dead-man zeroes it" --> DRV
    KB -- "/control · /emergency<br/>manual override, always live" --> DRV
    KB -. "arm / abort" .-> SAFE
    DRV --> TELLO

    DET --> BAG
    VS --> BAG
    POL --> BAG
    DRV --> BAG

    classDef phys fill:#e8eef8,stroke:#4a6fa5,color:#1d3050
    classDef exist fill:#e2f4e3,stroke:#2e8b57,color:#14401f
    classDef new fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
    classDef grey fill:#eef0f2,stroke:#8a97a0,color:#3b4750
```

**Why this node split** [D, derived from P]: one node per box in diagram 1, so that each of the
paper's functional stages can be swapped, disabled, or replayed independently. Two specific
choices are worth defending. `visual_state` is separate from `drone_detector` because the
observation definition is the single most under-specified part of the paper (ambiguity #2) and
will be revised more often than the detector; keeping it separate means changing the observation
never touches the perception code. And `ddpg_policy` runs **inference only** — no learning
in flight — because the paper trains entirely in the Gym environment and deploys a fixed
checkpoint [P §2.2.1, Appendix A], and because an updating policy on a live aircraft has no
safety argument.

**The dead-man is a design constraint, not a detail** [V `node.py:1099-1114`]: the driver zeroes
the command if it goes stale by more than 0.35 s, so the policy loop must publish at 10 Hz or
faster. This is a feature worth keeping — a stalled or crashed policy node stops the aircraft
rather than latching the last command.

---

## 5. Behaviour state machine [P §2.1 item 3]

The paper describes a hand-coded recovery behaviour that is **not** learned, and it matters
because three of the five scenarios exercise it: *"If the followed drone moves out of sight, the
follower drone initiates a rescan operation"*, *"It first maintains its position and then adjusts
its field of view"*, *"The scanning operation is performed according to a predefined movement
pattern"*, and *"If no drone is detected for an extended period, the follower drone switches to a
predefined patrol mode"*. Scenario 3 tells us the pattern in practice: the follower *"switches to
search mode and scans the environment by rotating around its axis"*.

```mermaid
stateDiagram-v2
    [*] --> SEARCH
    SEARCH: SEARCH<br/>no target yet<br/>rotate about own axis
    TRACK: TRACK<br/>target detected<br/>DDPG + depth rule active
    REACQUIRE: REACQUIRE<br/>target just lost<br/>hold position, then rotate
    PATROL: PATROL<br/>lost for an extended period<br/>predefined patrol pattern

    SEARCH --> TRACK: drone detected
    TRACK --> REACQUIRE: detection lost
    REACQUIRE --> TRACK: target reacquired
    REACQUIRE --> PATROL: loss exceeds the timeout
    PATROL --> TRACK: drone detected
    TRACK --> [*]: operator disengages
```

**[D] What the paper leaves to the implementer**: the loss timeout before PATROL, the rotation
rate during search, the number of consecutive missed frames that count as "lost", and the patrol
pattern itself. Recommended defaults are in the implementation plan; all four belong in one
parameter file, because they are the knobs that decide whether scenarios 3 and 4 succeed.

---

## 6. Training → deployment path

The paper trains in simulation and tests on hardware [P abstract, §2.2.1, Appendix A]: *"the
OpenAI Gym simulator is designed according to our problem, the DDPG agent is trained, the
performance of the developed system is tested in a real-world environment"*. Training took about
four hours on an Intel i5 with 8 GB of RAM, **CPU only** [P §2.2.4] — this is a small problem, and
that is a useful fact: the whole training run is reproducible on a laptop.

```mermaid
flowchart LR
    subgraph OFF["Offline — training"]
        GYM["<b>Gym chase environment — NEW</b><br/>image-plane target motion model<br/>latency injection 150-350 ms<br/>domain randomisation"]:::train
        TRAIN["<b>DDPG trainer — NEW</b><br/>actor/critic + replay buffer<br/>Table 2 hyperparameters<br/>~4 h on CPU (paper)"]:::train
        CKPT[("<b>Actor checkpoint</b><br/>versioned with the exact<br/>observation and action spec")]:::train
    end

    subgraph ON["Online — deployment"]
        POLN["<b>ddpg_policy node</b><br/>loads the checkpoint<br/>inference only"]:::new
        REAL["<b>Real flight</b><br/>Tello + target, indoors"]:::phys
    end

    EVAL["<b>Offline evaluation</b><br/>time-in-view % and detection %<br/>per scenario, per brightness"]:::grey

    GYM <--> TRAIN
    TRAIN --> CKPT
    CKPT --> POLN
    POLN --> REAL
    REAL -- "rosbag2" --> EVAL
    EVAL -. "measured latency and target dynamics<br/>feed back into the sim model" .-> GYM

    classDef train fill:#f0e6fa,stroke:#7d4fa6,color:#3d1e57
    classDef new fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
    classDef phys fill:#e8eef8,stroke:#4a6fa5,color:#1d3050
    classDef grey fill:#eef0f2,stroke:#8a97a0,color:#3b4750
```

**The one substantive addition to the paper's pipeline** [D]: the dashed feedback edge. The paper
trains in a Gym environment whose fidelity it never describes, then reports that latency was the
dominant failure mode in the real world. Since this repository has already measured that latency
at 150–350 ms, the environment can *model* it and the policy can be trained against it — turning
the paper's headline limitation into a training condition. This is the clearest place where a
replication on this platform can exceed the original.

**The checkpoint must be versioned with its observation and action specification** [D]. A DDPG
actor is meaningless without the exact input scaling and output mapping it was trained under;
storing them together is what makes the sim-to-real transfer auditable rather than folklore.
