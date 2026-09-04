# Baseline Block Diagram — "Recognise, Then Range"

This document turns the proposal (PDF §1.2 Fig. 1 + §3 + §4) into an implementable ROS 2
architecture. Five views, from concept to wiring:

1. [The proposal pipeline](#1-the-proposal-pipeline-what-the-thesis-claims) — what the thesis claims, with the novelty colour code of the PDF's Fig. 1
2. [The one-contract principle](#2-the-one-contract-principle-simulation--hardware) — how simulation-first / hardware-second becomes a config change
3. [ROS 2 node graph, hardware phase](#3-ros-2-node-graph--hardware-phase-two-tellos-indoors)
4. [ROS 2 node graph, simulation phase](#4-ros-2-node-graph--simulation-phase)
5. [Safety chain](#5-safety-chain-ieee-7009-mechanisms-pdf-62) and the [ablation switchboard](#6-the-ablation-switchboard-pdf-43)

Colour code (identical to PDF Fig. 1): **red** = published, not claimed as new · **orange** =
exists, but in a different setting or without a fair comparison · **green** = no published
instance found (the thesis's novelty) · grey = standard machinery, not under test.

---

## 1. The proposal pipeline (what the thesis claims)

The pipeline has two halves (PDF §3): the **perception half** turns an image of a small drone
into a metric range *with honest uncertainty*, conditioned on the recognised model (C2); the
**estimation half** turns those measurements, taken from a moving observer, into a stable
track using the observer's gyroscope (C3). Every added part is a **switch** that can be turned
off so its contribution is measurable on its own (PDF §3, §4.3).

```mermaid
flowchart TB
    subgraph PERC["PERCEPTION HALF — C2: image of a tiny drone → metric range with honest uncertainty"]
        CAM["Camera frames<br/>moving observer"]:::input
        DET["<b>Detect drone</b><br/>appearance + motion channel<br/>output: bbox width w px"]:::red
        REC["<b>Recognise model</b><br/>posterior πk over K models<br/>+ explicit 'unknown' class"]:::orange
        DB[("<b>Dimension database</b><br/>per-model W, L, H from specs<br/>'unknown' → broad prior over all sizes")]:::green
        ASP["<b>Aspect-angle model</b><br/>Wk(ψ) ≈ Wk·|cos ψ| + Lk·|sin ψ|<br/>ψ prior sharpened by target heading"]:::green
        PIN["<b>Pinhole range</b><br/>d = f·W / w<br/>sensitivity: ∂d/∂w = −d² / (f·W)"]:::red
        MIX["<b>Range mixture — Eq. (4)</b><br/>p(d) = Σk πk · N(d; f·Wk/w, σ²dk)<br/>confident recognition → narrow<br/>uncertain / wrong → wide, not biased"]:::green
    end

    subgraph EST["ESTIMATION HALF — C3: measurements from a moving observer → stable target track"]
        GYRO["Observer gyroscope<br/>(sim: true rate, 100–200 Hz)<br/>(Tello: surrogate from ~10 Hz attitude)"]:::input
        PREINT["IMU preintegration<br/>rotation between frames"]:::grey
        SW_A{{"<b>Switch A — motion compensation</b><br/>gyro-aided vs image-only homography<br/>→ background alignment + de-rotated bearing"}}:::orange
        SW_B{{"<b>Switch B — R inflation</b><br/>scale measurement noise with |ω|<br/>(Pliska et al.)"}}:::orange
        SW_C{{"<b>Switch C — delayed filter</b><br/>run at inertial rate, roll state<br/>to image capture time (Yang et al.)<br/>filter rate = swept parameter"}}:::orange
        TRK["<b>Multiple-model tracking filter</b><br/>target relative state + covariance<br/>consistency-checked (NEES)"]:::grey
    end

    subgraph LOOP["CLOSED LOOP — E6 (threshold-swept test is the novel part)"]
        CTL["<b>Following controller — FIXED</b><br/>deliberately simple, held constant<br/>across all perception/estimation variants"]:::grey
        SAFE["Safety supervisor<br/>geofence · speed cap · human engage<br/>auto-disarm · kill path"]:::grey
        UAV["Observer drone<br/>velocity commands"]:::input
        E6["<b>Closed-loop scoring</b><br/>success rate as a curve over the<br/>capture threshold · time-to-close ·<br/>closest approach"]:::green
    end

    CAM --> DET
    CAM --> REC
    DET -- "w (px), box noise σw(w)" --> PIN
    REC -- "πk + unknown" --> MIX
    REC --> DB
    DB -- "Wk, Lk + spread" --> ASP
    ASP -- "effective Wk(ψ) + variance" --> PIN
    PIN --> MIX
    MIX -- "p(d) mixture" --> TRK

    GYRO --> PREINT
    PREINT --> SW_A
    SW_A -- "aligned motion channel" --> DET
    SW_A -- "de-rotated bearing" --> TRK
    GYRO -- "|ω|" --> SW_B
    SW_B --> TRK
    GYRO -- "high-rate propagation" --> SW_C
    SW_C --> TRK
    TRK -- "estimated heading → ψ prior" --> ASP

    TRK -- "target relative state" --> CTL
    CTL --> SAFE
    SAFE --> UAV
    UAV -. "observer moves, loop closes" .-> CAM
    SAFE --> E6

    classDef red fill:#fde8e8,stroke:#c0392b,color:#5b1f16
    classDef orange fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
    classDef green fill:#e2f4e3,stroke:#2e8b57,color:#14401f
    classDef grey fill:#eef0f2,stroke:#7f8c8d,color:#2c3e50
    classDef input fill:#e8eef8,stroke:#4a6fa5,color:#1d3050
```

**How to read it.**
- The two feedback edges matter: the tracker's heading estimate sharpens the aspect-angle
  prior (PDF §3.3), and the closed loop moves the camera that feeds perception.
- The three orange switches are the entire C3 experiment: each has an image-only or naive
  counterpart, giving the single-switch ablation rows (PDF §3.5).
- Range error grows with the **square** of range (Eq. 2). Two design consequences propagate
  through everything downstream: errors are reported **per range band**, and the range
  uncertainty **must widen with distance** (PDF §3.1).
- The controller is grey on purpose: the proposal holds it fixed because it is not part of
  what is being tested (PDF §4.1). See [rl_analysis.md](rl_analysis.md) before changing that.

---

## 2. The one-contract principle (simulation ⇄ hardware)

PDF §4 mandates simulation-first, hardware-second, and §4.3 mandates that the ablation be a
**configuration matrix, not a code change**. The architectural consequence: the perception /
estimation / control stack must not know whether it is flying. Simulator and hardware drivers
publish the **same topic contract**; swapping phases is a launch-file substitution.

```mermaid
flowchart LR
    subgraph HW["Hardware providers (Phase 2, Q6)"]
        DRV["tello driver — EXISTS<br/>workspace/src/tello<br/>image_raw · camera_info · imu ·<br/>odom · status ← cmd_vel"]:::exist
        MOCAP["mocap bridge — NEW<br/>true poses of both aircraft"]:::new
    end
    subgraph SIM["Simulation provider (Phase 1, Q1–Q5)"]
        SIMU["quadrotor simulator — NEW<br/>same topics, plus knobs:<br/>image delay · camera-IMU offset ·<br/>exposure · gyro rate · focal length ·<br/>target range and model (CAD)"]:::new
        SGT["simulated ground truth<br/>exact by construction"]:::new
    end
    CONTRACT["<b>THE TOPIC CONTRACT</b><br/>sensor topics + ground-truth topics +<br/>velocity command topic<br/>every message carries t_capture AND t_receive"]:::contract
    STACK["Perception + Estimation + Control stack<br/>(identical binaries in both phases —<br/>only parameters change)"]:::stack

    DRV --> CONTRACT
    MOCAP --> CONTRACT
    SIMU --> CONTRACT
    SGT --> CONTRACT
    CONTRACT <--> STACK

    classDef exist fill:#e2f4e3,stroke:#2e8b57,color:#14401f
    classDef new fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
    classDef contract fill:#e8eef8,stroke:#4a6fa5,color:#1d3050
    classDef stack fill:#eef0f2,stroke:#7f8c8d,color:#2c3e50
```

The dual timestamp (`t_capture` + `t_receive`) is a hard requirement of PDF §4.3 — "latency is
always a measured quantity". The existing driver already stamps on arrival with bounded error
(50 Hz change-detection polling, `node.py`), which is the hardware half of this promise.

---

## 3. ROS 2 node graph — hardware phase (two Tellos, indoors)

Green = exists on this repo's baseline (`workspace/src/`), verified. Orange = to be built.
The observer's driver topics are shown with their **measured** properties, not nominal ones.

```mermaid
flowchart TB
    subgraph OBSHW["Observer Tello (namespace /observer)"]
        ODRV["tello driver node — EXISTS<br/>relative topic names → namespace-able"]:::exist
    end

    subgraph TGT["Target Tello (no driver needed on the perception host)"]
        TDRV["scripted / manual flight<br/>(second host or operator)<br/>+ passive mocap markers"]:::neutral
    end

    subgraph GT["Ground truth"]
        MC["motion-capture volume"]:::neutral
        MCB["mocap bridge node — NEW<br/>/gt/observer_pose · /gt/target_pose"]:::new
    end

    subgraph PIPE["Perception + estimation stack (one node per switch — PDF §4.3)"]
        OMO["observer_motion — NEW<br/>hw: surrogate rate from ~10 Hz attitude<br/>+ inter-frame preintegration"]:::new
        MOC["motion_compensator — NEW<br/>mode: gyro | homography | none"]:::new
        DR["detector_recognizer — NEW<br/>bbox w + class posterior πk + unknown"]:::new
        DDB[("dimension_db — NEW<br/>machine-readable table + lookup")]:::new
        RE["range_estimator — NEW<br/>Eq. (4) mixture + aspect marginalisation"]:::new
        TT["target_tracker — NEW<br/>multiple-model filter<br/>switches: R-inflation · delayed replay · rate"]:::new
    end

    subgraph CTRL["Closed loop"]
        FC["follow_controller — NEW<br/>FIXED simple follower"]:::new
        SS["safety_supervisor — NEW<br/>gates every command"]:::new
        KB["tello_control GUI — EXISTS<br/>keyboard: takeoff/land/EMERGENCY/manual"]:::exist
    end

    REC["rosbag2 recorder<br/>records every topic of every run"]:::neutral

    ODRV -- "/observer/image_raw<br/>960×720 · ~30 Hz · bgr8<br/>150–350 ms Wi-Fi latency" --> DR
    ODRV -- "/observer/image_raw" --> MOC
    ODRV -- "/observer/camera_info<br/>fx ≈ 919.4 px (ost.txt)" --> RE
    ODRV -- "/observer/imu<br/>fused attitude + accel · ~10 Hz<br/>NO gyro (covariance flag −1)" --> OMO
    ODRV -- "/observer/odom<br/>metric flow velocity" --> TT

    OMO -- "surrogate ω + frame rotation" --> MOC
    OMO -- "|ω| for R-inflation" --> TT
    MOC -- "aligned motion channel" --> DR
    MOC -- "de-rotated bearing" --> TT
    DR -- "/pipeline/detection<br/>t_capture + t_receive" --> RE
    DDB -.-> RE
    RE -- "/pipeline/range_mixture" --> TT
    TT -- "heading → aspect prior" --> RE
    TT -- "/pipeline/track<br/>rel. position/velocity + covariance + NIS" --> FC

    FC -- "/control/cmd_vel_raw" --> SS
    SS -- "/observer/cmd_vel<br/>REP-103 · sticks ∈ [−1,1] · ≥10 Hz<br/>(driver: 20 Hz RC + 0.35 s dead-man)" --> ODRV
    KB -- "/observer/control · /observer/emergency<br/>manual override + abort" --> ODRV
    KB -. "engage / disengage" .-> SS

    MC --> MCB
    MCB -- "/gt/*_pose (also feeds geofence)" --> SS
    MCB --> REC
    ODRV --> REC
    DR --> REC
    RE --> REC
    TT --> REC

    TDRV -.- MC

    classDef exist fill:#e2f4e3,stroke:#2e8b57,color:#14401f
    classDef new fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
    classDef neutral fill:#eef0f2,stroke:#7f8c8d,color:#2c3e50
```

**Verified hardware facts this wiring respects** (details + file references in
[implementation_plan.md](implementation_plan.md)):

- **There is no gyroscope on the Tello.** `/observer/imu` carries fused attitude (whole
  degrees, ~10 Hz, jittery) and accelerometer only; the driver marks angular velocity
  "not measured". The C3 "gyro" input on hardware is therefore a **surrogate rate**
  differentiated from attitude (~10 °/s quantisation noise) — exactly why PDF §4.2 promises
  the hardware inertial result only as a **lower bound**, with the full-rate result from
  simulation.
- **The target drone needs no driver on the perception host.** Its ground truth comes from
  mocap; it only has to fly patterns (crossing / approaching / receding — PDF §4.3). This
  sidesteps a real conflict: djitellopy binds UDP :8889 per process and both Tellos in AP
  mode share IP 192.168.10.1, so two drivers on one host do not coexist
  (options, if the target must be ROS-driven, in the plan).
- **The pixel-size regime works indoors.** With fx ≈ 919.4 px and the Tello's 98 mm body
  width: ~45 px at 2 m, ~30 px at 3 m, ~15 px at 6 m, ~10 px at 9 m — the tens-of-pixels
  regime of PDF §1.1 inside a mocap volume.

---

## 4. ROS 2 node graph — simulation phase

Identical stack; the two aircraft and the mocap system are replaced by the simulator, which
adds the sweep knobs the experiments need (PDF §4.1). Nothing downstream changes.

```mermaid
flowchart TB
    subgraph SIMW["Simulator world"]
        WORLD["photorealistic scene<br/>observer + target from CAD models<br/>of the dimension-table drones"]:::new
        KNOBS["sweep knobs (parameters):<br/>image delay · camera-IMU time offset (E5) ·<br/>exposure · focal length + target range (H1 size sweep) ·<br/>gyro rate + noise (H4 filter-rate sweep) ·<br/>observer mode: hover / level / manoeuvre ·<br/>target behaviour: crossing / approaching / receding"]:::knobs
    end
    SIF["sim interface node — NEW<br/>publishes the SAME topic contract:<br/>/observer/image_raw · camera_info · imu (TRUE gyro, 100–200 Hz) ·<br/>odom · /gt/observer_pose · /gt/target_pose ·<br/>/gt/target_model · /gt/target_range<br/>consumes /observer/cmd_vel"]:::new
    STACK["Perception + estimation + control stack<br/>(unchanged binaries)"]:::grey
    EXP["experiment_manager — NEW<br/>loads one config-matrix row per run:<br/>switch settings + knob values + seed<br/>≥10 seeds per comparison (PDF §5.3)"]:::new
    EVAL["evaluation package — NEW (offline)<br/>AP by size band · range error per band ·<br/>NEES consistency (≥50 runs) ·<br/>closed-loop threshold curves + CIs"]:::new

    KNOBS --> WORLD
    WORLD --> SIF
    SIF <--> STACK
    EXP -- "parameters per run" --> SIF
    EXP -- "switch settings per run" --> STACK
    SIF -- "bags: data + ground truth" --> EVAL
    STACK -- "bags: estimates" --> EVAL

    classDef new fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
    classDef knobs fill:#e8eef8,stroke:#4a6fa5,color:#1d3050
    classDef grey fill:#eef0f2,stroke:#7f8c8d,color:#2c3e50
```

Two proposal-mandated details: the H1 size sweep is produced by **moving the target and
varying focal length, not by blurring the image** (PDF §4.1); and the three observer modes are
matched to Pliska et al. so the C3 numbers are directly comparable (PDF §4.1).

---

## 5. Safety chain (IEEE 7009 mechanisms, PDF §6.2)

Each fail-safe is a concrete mechanism with an owner. Several already exist in the verified
driver — they are the reason the closed loop can be attempted at all.

```mermaid
flowchart LR
    subgraph AUTH["Human authority (always present)"]
        H["operator<br/>engage is opt-in, per run"]:::human
    end
    subgraph SUP["safety_supervisor — NEW"]
        GF["hard geofence<br/>mocap-fed, edge of flight volume"]:::new
        VC["closing-speed cap<br/>clamps commanded velocity"]:::new
        AD["auto-disarm on:<br/>link loss · target loss ·<br/>geofence breach · low battery"]:::new
        EN["engagement gate<br/>zero command until human-armed"]:::new
    end
    subgraph DRVS["tello driver — EXISTS, verified"]
        DM["RC dead-man: stale command<br/>→ zeroed within 0.35 s"]:::exist
        EM["emergency: fire-and-forget datagram<br/>bypassing the queue + retried copy"]:::exist
        AL["SDK auto-land after 15 s silence"]:::exist
    end
    LAD["staged testing ladder (PDF §6.2):<br/>simulation → static target →<br/>slow target → full following<br/>each fail-safe exercised in sim first"]:::plan

    H -- "arm / abort (keyboard E, battery pull)" --> EN
    H --> EM
    EN --> VC --> GF --> DM
    AD --> DM
    GF -- "breach" --> AD
    AD -- "worst case" --> EM
    DM --> AL
    LAD -.-> SUP

    classDef human fill:#e8eef8,stroke:#4a6fa5,color:#1d3050
    classDef new fill:#fdf0dc,stroke:#d9822b,color:#5c3a10
    classDef exist fill:#e2f4e3,stroke:#2e8b57,color:#14401f
    classDef plan fill:#eef0f2,stroke:#7f8c8d,color:#2c3e50
```

One honest gap to record: the proposal promises a "hardware kill-switch on both aircraft".
The Tello has no external kill line; the practical equivalents are the keyboard emergency
(motor cut via the driver's bypass path) and physically pulling the battery. This should be
stated in the thesis rather than papered over.

---

## 6. The ablation switchboard (PDF §4.3)

"The whole ablation is a configuration matrix rather than a code change." Every switch below
is a ROS 2 parameter of exactly one node; `experiment_manager` sweeps them.

| # | Switch | Options | Owner node | Swept by |
|---|--------|---------|-----------|----------|
| S1 | Size prior | fixed single width · recognised model width · recognised + aspect marginalisation | `range_estimator` | **E2** (H2) |
| S2 | Range baseline | depth foundation model, zero-shot (offline arm) | `evaluation` | **E2** (H2) |
| S3 | Range representation | mixture Eq. (4) · collapsed point estimate | `range_estimator` → `target_tracker` | **E3** (H3) |
| S4 | Motion compensation | gyro-aided · image-only homography · none | `motion_compensator` | **E4** (H4) |
| S5 | Measurement-noise inflation | scale R with observer rate · off | `target_tracker` | **E4** (H4) |
| S6 | Delay bridging | delayed filter at inertial rate · naive (filter at camera rate) | `target_tracker` | **E4** (H4) |
| S7 | Filter rate | swept, below → above camera rate | `target_tracker` | **E4** (H4) |
| S8 | Camera–IMU time offset | injected 0 → tens of ms | sim interface | **E5** (H4) |
| S9 | Target pixel size | range × focal length sweep | sim interface | **E1** (H1) |
| S10 | Recognition errors | forced misclassification injection | `experiment_manager` | **E3** (H3) |
| S11 | Observer mode | hover · level flight · manoeuvre | sim / flight plan | **E4, E6** |
| S12 | Capture threshold | family of thresholds (reported as curves, never one point) | `evaluation` | **E6** |

The controller has **no switch** — PDF §4.1 holds it fixed. If a learned (RL) controller is
ever added, it becomes a separate comparison arm of E6, not a row of this matrix
([rl_analysis.md](rl_analysis.md)).
