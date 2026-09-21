# The Three-Tier Simulation Pipeline, Explained

*How the Tello chase system was trained and proven — every stage before the
real drone, in plain language. All numbers in this document are from real
runs on this machine (September 19–21, 2026).*

---

## 1. The Big Idea: Why Three Tiers?

We are building a drone (a DJI Tello) that can **chase another drone using
only its camera**. Before it ever flies for real, we must answer three very
different questions:

1. **Can a policy *learn* the chasing behavior at all?** — needs *millions*
   of practice attempts.
2. **Does that behavior survive *real physics*?** — motors, inertia, drag,
   delays.
3. **Does it survive *real vision*?** — an actual camera image and an actual
   object detector, with all their noise and mistakes.

No single simulator can answer all three well. A simulator with beautiful
graphics is far too slow for millions of training steps; a fast math
simulator has no camera and no real physics. So the work is split into
**three tiers**, like a pilot's education:

| | Analogy | What it is | Speed | What it proves |
|---|---|---|---|---|
| **Tier A** | classroom | pure-math world (Python/Gym) | thousands of steps/minute | the behavior can be *learned* |
| **Tier B** | flight simulator | real physics engine (Gazebo) | ~real time, headless | the behavior survives *physics* |
| **Tier C** | practice field | rendered 3D world (Unreal Engine) | slow (rendering) | the behavior survives *real vision* |

The tiers do **not** talk to each other while running. They connect through
two things only:

- **Shared code** — every tier imports the *same* Python functions for
  observations, rewards, and control laws (one contract, never copied).
- **File hand-offs** — Tier A produces a trained policy file; Tier C
  produces an image dataset and then a trained detector file; each later
  stage consumes the earlier stage's files.

### The two "trained models"

It's important to keep these apart:

- **The brain** — the RL policy (TD3 network). Learns *how to move* from a
  detection. Trained in Tier A.
- **The eyes** — the YOLO detector. Learns *where the drone is in a camera
  image*. Trained on images rendered by Tier C.

Tier C's final exam plugs the eyes into the brain and lets them fly
together, with nothing faked.

### Who controls what (a key design decision)

The policy does **not** control everything. The forward speed is a
**hand-written law** shared by every tier and by the future real flight
code: it converts the detected box's width into a distance estimate and
commands approach/standoff speed from that. The RL policy controls the two
*perceptual* axes — **vertical speed and yaw (turning)** — i.e., keeping
the target centered in the image. Why? Monocular distance is weak and
forward speed decides collision energy, so it stays deterministic and
auditable; the genuinely hard part under camera noise and delays — keeping
the target in frame — is what the network learns.

---

## 2. Tier A — The Classroom (Gym, pure Python)

**Where:** `workspace/src/chase_gym`, `chase_train`, `chase_eval`
**Needs:** nothing but Python + a GPU. No simulator installed, no window.

### What the world looks like

A 14-number observation vector (where the box is in the image, how wide it
is, what the last actions were…), simple flight kinematics with a
first-order lag (a model of the Tello's sluggishness), and — crucially —
**simulated sensor imperfection**: the "camera" measurements are delayed,
jittered, and sometimes dropped, exactly the abuse a real detector will
deliver. The policy never experiences a clean world.

### What ran

- **Four algorithm "arms"** trained head-to-head: `td3` (modern),
  `ddpg_repaired`, `ddpg_faithful` (an exact reproduction of the source
  paper's algorithm, bugs and all), and `sb3_td3` (an independent library
  implementation as a cross-check).
- **The matrix:** 20 runs (4 arms × 5 random seeds), each scored on a
  held-out evaluation suite the training never saw.
- **The acceptance rule:** an arm is accepted only if the *lower* end of
  its confidence interval beats the *upper* end of the P-controller
  baseline's interval on moving-target scenarios. No cherry-picking.

### The result

✅ **ACCEPT — every RL arm beat the classical P-controller** on moving
targets. The selected deployable policy:

- Follow task (the deployment mission — shadow the target at 2 m):
  `runs/td3_s4_20260919_211433/td3_s4_step75000_tv100.pt`
- Intercept task (catch the target): `runs/td3_s4_20260920_113322/`
  `td3_s4_step70000_tv098.pt` — held-out time-in-view **98.05%**

Every checkpoint is a *bundle*: weights **plus** the exact observation
specification, action mapping, config hash, git commit, and RNG state — so
a checkpoint can never silently be used with the wrong input format (it
refuses to load). Each selected policy is also exported to **ONNX** with a
bit-level parity check, ready for the real drone's inference runtime.

**What Tier A cannot prove:** that its math world resembles reality. That
is exactly Tier B's job.

---

## 3. Tier B — The Flight Simulator (Gazebo, real physics)

**Where:** `workspace/src/chase_sim_gz`
**Needs:** Gazebo Harmonic installed on this machine. Runs **headless** —
there is deliberately no window (see below).

### Why it exists

Tier A *approximates* flight with simple equations. Gazebo simulates the
real thing: spinning rotors, thrust curves, inertia, drag. If a policy's
behavior only works in the approximation, it is worthless. Tier B is the
referee between the two worlds.

### The three checks, and their results

1. **Sign test** ✅ — pushes each control axis one at a time and verifies
   the world responds in the *direction* the code believes (three
   simulators plus one aircraft equals four chances for a silently flipped
   sign). Passed on real Gazebo.

2. **Lag calibration** ✅ — measures how sluggishly the simulated Tello
   responds (measured 0.54–0.84 s). Deliberately left uncalibrated for
   now: it will be tuned to the *real* Tello's measured lag in the
   hardware phase, not to a guess.

3. **The A↔B replay gate** ✅ — the main event. The *same commands* are
   replayed in Tier A's math world and in Gazebo, and the two resulting
   image-space trajectories are compared pixel by pixel:

   ```
   episode seed 5000: median 21.0 px
   episode seed 5001: median  4.0 px
   episode seed 5002: median  8.0 px
   overall median 8.0 px vs tolerance 40.0  ->  PASS
   ```

   **Meaning: the world the policy trained in matches real physics to
   within a few pixels.** This gate could not even run until a reset bug
   was fixed (the world was given only 12 *milliseconds* to physically
   settle after teleporting the drones into place; it now holds 1.5 s).

### Why there is no window

Tier B has **no camera** — measurements come from mathematically
projecting the true positions ("oracle detector"). Rendering would add
nothing. It also runs in *lockstep*, stepping physics in exact chunks
faster than real time; a 30-second flight finishes in a few wall-clock
seconds. (If you ever want to peek: `gz sim -g` attaches a viewer to a
running server.)

**What Tier B cannot prove:** anything about cameras or detectors. That is
Tier C.

---

## 4. Tier C — The Practice Field (Unreal Engine, real vision)

**Where:** `sim/tier_c_airsim` (client, WSL) + Blocks.exe / Unreal 5.7 on
the Windows side (renderer, RTX 5050).
**This is the only tier with pictures — because pictures are the point.**

Tier C answers the question the other tiers cannot: *does the whole thing
work when the input is an actual camera image processed by an actual
detector?* It ran in four stages.

### Stage C1 — Build the world (the warehouse)

The deployment target is indoor flight, so `warehouse.py` builds an
enclosed warehouse *inside* the empty Blocks level at runtime: floor,
ceiling, walls, shelving racks down both sides of a central aisle, loaded
with CC0 3D props (crates, boxes, barrels), lit by ceiling lights whose
intensity is actually driven by the `--lighting low|medium|high` setting.
The target drone is a quadrotor mesh scaled to the Tello's 18 cm span.
A camera is mounted on the chaser's nose (in front of its own rotors),
matching the real Tello camera's resolution and field of view.

### Stage C2 — Manufacture the dataset

`dataset_factory.py` places the target at a grid of ranges (0.5–6 m),
angles, and headings, in three lighting classes, and saves each camera
frame with a **pixel-perfect label** taken from the engine's own ground
truth (no human labeling, no label errors):

- **3,003 labeled frames** across the three lighting classes.
- Plus — after a hard lesson, see below — **360 "negative" frames**:
  warehouse views with *no drone anywhere*, captured by a patrol that
  hides the target under the floor and looks at walls, racks, and dark
  corners.

### Stage C3 — Train the eyes (the detector)

`train_detector.py` fine-tunes a YOLOv8-nano on that dataset (split
stratified by target range so tiny far-away targets appear in both train
and validation).

**The phantom lesson — why the negatives exist.** The first detector (v1)
scored a beautiful precision of 0.995 — but its validation set contained
*only images with a drone in them*. The moment the chase flew off the
aisle's center line, v1 hallucinated huge, high-confidence "drone" boxes
on dark walls and shelf shadows (confidences up to 0.92!), and every
hallucination became a fake "capture." A detector that has never seen a
drone-free image cannot say "no drone here."

The v2 detector, retrained with ~11% negatives:

- **Precision 0.996 — now measured *with* 72 drone-free images in the
  exam.**
- Recall 0.705, mAP50 0.724 (the misses are targets smaller than ~30 px —
  4 m and beyond at extreme angles).
- All three of v1's phantom frames come back clean.
- Inference: ~1.2 ms per frame on the GPU.

### Stage C4 — The final exam: vision-in-the-loop

`vision_in_loop_eval.py` runs the complete closed loop with **nothing
faked**:

```
rendered camera frame → the trained YOLO (v2) → the same observation
assembler as training → the trained TD3 actor → stick commands →
the engine flies → next frame
```

**Follow (the deployment mission):**
✅ **100% detection across 5 episodes × 300 steps (1,500 frames)** — the
policy held the target in the detectable envelope the entire time. This is
the mission the drone will actually fly, and it passed completely.

**Intercept (catch it):**
✅ One **verified genuine capture** — the drone flew 3.6 m down the aisle
and closed to 0.48 m at step 79, with the target's rotors visibly inside
the detection box on the saved frame. Detection was 100% in all five
episodes; the other four tracked flawlessly but ran out the 300-step clock
during the slow final meter. Important honesty note: the eval loop is
rendering-bound (~2 commands/second), which roughly halves the drone's
motion duty — so the *capture rate* measured here is a **lower bound**,
not the deployed system's number. The deployed flight stack streams
commands continuously.

Every capture claim in this project is **verified frame-by-frame** —
because Stage C4's history is a museum of convincing fakes:

- captures "in 1 step" → the previous episode's momentum (fixed: episodes
  now start from a measured standstill);
- captures on giant boxes → v1's phantoms (fixed: negatives + v2);
- 80% capture rate → all three artifacts stacked (discarded).

The rule that emerged: **a number is not a result until the frame behind
it has been looked at.**

### The engine war (lessons that cost two days)

Project AirSim's `simple_flight` controller has undocumented behavior that
consumed most of the debugging time. The distilled rules, all measured:

1. **Fire-and-forget ("streamed") commands are unreliable** — a vertical
   component freezes the vehicle in place always; yaw or even pure forward
   freeze on some connects. Which paths work is re-rolled per connection.
2. **The one pattern that never failed anywhere:** a single *blocking*
   velocity command per control tick, sent from the main thread. The final
   eval uses exactly this (0.4 s windows).
3. **Between episodes the vehicle must be actively settled to a stop**
   (a diving policy carries ~1 m/s through a naive reset).
4. **The engine process degrades** with uptime and many scene reloads —
   including one outright crash mid-run. Standing procedure: restart
   Blocks.exe before runs that matter, and run `stream_probe.py` (a
   60-second health check) at the start of a session.

---

## 5. What Is Now Proven — and What Is Not

**Proven, with numbers:**

| Claim | Evidence |
|---|---|
| The chasing behavior is learnable, and RL beats the classical baseline | Tier A matrix, 20 runs, ACCEPT |
| The training world matches real physics | Tier B gate, 8.0 px median on real Gazebo |
| A detector can be manufactured entirely from the sim, and made honest | v2: P .996 with negatives in the exam |
| The full vision loop works with nothing faked | Follow: 100% detection × 1,500 frames |
| The intercept mission can genuinely complete | Verified capture at 0.48 m, frame on record |

**Not yet proven (this is exactly the hardware phase):**

- The **real world's appearance** — the detector has only ever seen
  rendered images. Real camera frames must be mixed in before outdoor
  trust (the code's own docs insist on this).
- The **real Tello's dynamics** — its true command lag and stick scaling
  are still placeholders everywhere.
- The **flight software** — the six ROS nodes that will run all of this on
  the real drone are designed but not written.

---

## 6. The Road to the Real Tello

1. **Measure the real drone** — step-response tests for command lag and
   stick scale (Phase 1 of the plan). Replaces the placeholders.
2. **Recalibrate Tier B** to the measured numbers, re-run the A↔B gate,
   and **fine-tune the policy in Gazebo** if the gap demands it.
3. **Collect real camera frames**, mix them into the dataset, retrain the
   detector (v3) — closing the rendered-vs-real appearance gap.
4. **Build the six flight nodes** — state, policy (loads the ONNX),
   depth_rule (the same forward law), behaviour, mixer, safety.
5. **Fly** — first tethered/protected tests, then the mission.

---

## 7. Quick Reference

| Artifact | Path |
|---|---|
| Follow policy (deployable) | `runs/td3_s4_20260919_211433/td3_s4_step75000_tv100.pt` |
| Intercept policy | `runs/td3_s4_20260920_113322/td3_s4_step70000_tv098.pt` |
| Detector (v2, with negatives) | `sim/tier_c_airsim/detector_runs/chase_yolo_v2/weights/best.pt` |
| Dataset (3,363 frames) | `sim/tier_c_airsim/chase_dataset/` |
| Tier B gate result | `ab_replay_gate.json` (median 8.0 px, PASS) |
| Chase videos | `sim/tier_c_airsim/chase_frames9/chase.mp4` |

| Task | Command |
|---|---|
| Engine health check (60 s) | `~/.venvs/tier_c/bin/python stream_probe.py` |
| Vision-in-the-loop eval | `~/.venvs/tier_c/bin/python -u vision_in_loop_eval.py --checkpoint <policy.pt> --yolo <best.pt> --lighting medium --episodes 5` |
| Tier B replay gate | `python -m chase_sim_gz.ab_replay_gate --episodes 3` (with the workspace on `PYTHONPATH`) |
| Retrain detector | `~/.venvs/tier_c/bin/python train_detector.py --data chase_dataset --epochs 100` |

*(All Tier C commands run from `sim/tier_c_airsim/` with Blocks.exe up.)*
