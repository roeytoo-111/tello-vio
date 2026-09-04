# Design Rationale — Why Every Diagram, Why Every Block

Companion to [block_diagram.md](block_diagram.md). This file answers, for each diagram and each
block: **why it was chosen, what it means, how it will be implemented, and what already exists.**

## 0. Method: how blocks earned their place

Every element carries exactly one of three provenance tags, so each claim is checkable:

- **[P]** — *PDF-mandated*: the proposal requires it; the section (and where useful the sentence)
  is cited from `preview_content-only.pdf`.
- **[V]** — *verified platform fact*: checked in this repo's code or docs; `file:line` given.
- **[D]** — *my design decision*: not stated in the PDF; the derivation from [P]/[V] facts is
  given, and it is open to challenge.

Selection rule for a **diagram**: it exists only if it settles a question the implementation
must answer (what the thesis claims; how sim and hardware coexist; which processes talk over
which topics; what stops an unsafe command; what changes if RL is added). Selection rule for a
**block**: something must be built, reused, or measured at that boundary — a box with no owner
and no experiment was cut.

### 0.1 Corrections from this verification pass

Re-auditing my own output against the sources found four inaccuracies, now fixed:

1. **Yaw drift figure.** I wrote "~13°/min-scale drift regime measured on this repo". The
   README actually says *"~13° RMS over 60 s in simulation"* (Known limitations). Corrected in
   [implementation_plan.md](implementation_plan.md) §9 — the figure is simulation-derived, not
   a hardware measurement.
2. **Prop-span numbers.** I wrote "the ~80 mm gap … is a ~45% range bias". The Tello spec gives
   only the 98 mm body width; the prop-tip span is **not** officially specified, so both numbers
   were false precision. Replaced with the exact mechanism: Eq. (1) is linear in W, so any
   fractional mismatch between annotation convention and table width is the same fractional
   range bias — and the actual span must be *measured* when the dimension table is built.
3. **Recognition input edge.** Diagram 1 previously drew `camera → recognise` directly. PDF
   §3.1 says a *"recognition head"* — attached to the detector — and §4.3 lists one combined
   *"detection-and-recognition module"*. The edge is now `detect → recognise (target region)`.
4. **Dimension-DB edge semantics.** The `recognise → DB` edge is *not* an argmax lookup. Eq. (4)
   sums over **all** models k, so the range module reads the whole table and weights components
   by the posterior πk; the recogniser contributes the class-key set and the posterior. The edge
   label now says "class keys".

---

## 1. Diagram 1 — the proposal pipeline

**Why this diagram exists.** The PDF's own Fig. 1 is the thesis's claim structure: eight stages,
each coloured by how well the literature covers it. The implementation must *preserve those
boundaries* — the red blocks must not be innovated on (they are someone else's published work),
and the orange/green blocks are exactly what the experiments E1–E6 measure. Diagram 1 is Fig. 1
expanded with the mechanisms of §3, so that every box corresponds to something buildable and
every dashed box to something switchable. [P: Fig. 1, §3, §4.3]

### Perception half (C2)

**Camera frames** — [P §3.1] *"The observer carries a calibrated camera and an IMU with known
mounting and a small, to-be-calibrated time offset."*
*Implemented today:* fully, by the existing driver — `/image_raw` 960×720 `bgr8` at ~30 Hz,
stamped on arrival, duplicate frames suppressed by object identity (`node.py:852-855`), dead
decoder detected and stream auto-restarted (`node.py:899-926`); calibration in `ost.txt`
(fx = 919.42 px) with intrinsics correctly rescaled if the image is downscaled
(`node.py:971-1024`). The "calibrated camera" premise of §3.1 is already satisfied.

**Detect drone — red** — [P Fig. 1, §2.1] Red because detection from a moving camera is
published work (Rozantsev [14], Dogfight [15], GLAD [6], YOLOMG [16]) and the proposal
explicitly does not claim it. It is in the diagram because it produces the pipeline's first
measured quantity: §3.1 *"the detector returns a box of pixel width w around the target."* The
motion-channel input is drawn because the modern tiny-target detectors *"improve on tiny
targets by fusing appearance with a motion channel"* [P §2.1] — and that channel is where
Switch A couples in.
*Implementation:* a small published-family detector inside `rtr_detect`, kept within the 33 ms
frame budget [V README compute constraint]. *Today:* nothing built; the input feed and the
budget constraint exist.

**Recognise model — orange** — [P §3.1] *"a recognition head returns a probability πk for each
known model k plus an unknown class."* Orange because recognition has only been shown on large,
close targets (~92% on three DJI models [21]) and *"no study reports how this accuracy changes
as the target shrinks"* [P §2.4] — measuring that fall-off **is** E1/H1. The `unknown` class is
load-bearing: it is what the mixture widens toward (§3.4) and the ADG-YOLO limitation the
thesis fixes (§1.3: *"cannot handle unknown models"*).
*Implementation:* a classification head on the detector (one module per PDF §4.3), classes =
dimension-table keys + `unknown`, trained in Q2 on synthetic + real crops [P Table 3]. *Today:* none.

**Dimension database — green** — [P §3.2] Green because *"this is the reusable resource
ADG-YOLO's authors call for and that no perception paper has published"*, and [P §2.4] *"there
is no public, machine-readable table of drone dimensions."* Why it matters is quantified in the
PDF: a DJI Mini 3 and a Matrice 300 differ by ~3.5× in width, *"so using the wrong model's size
is a factor-of-three range error."* The `unknown` entry *"returns a broad distribution over all
sizes."*
*Implementation:* YAML/CSV keyed to the recognition class names; per entry: W/L/H, the
**measurement convention** (props included or not — see correction 0.1-2), source URL,
retrieval date, tolerance; consumed as a library by the range module — [D] no runtime service,
because the data is static and a service round-trip buys nothing. *Today:* none (v0 is Q1
deliverable D1).

**Aspect model — green** — [P §3.3] Eq. (3): Wk(ψ) ≈ Wk·|cos ψ| + Lk·|sin ψ|; *"for a square
airframe [the apparent width] swings by up to 41% between head-on and corner-on. That swing is
the same size as the error the size prior is meant to remove, so it cannot be ignored."* It is
a separate block because it is a separate **switch** (S1's third option) and a separate
question: *"Whether this term measurably helps is one of the experiments, not an assumption."*
The dashed feedback edge from the tracker exists because §3.3 gives ψ *"a prior that is
sharpened by the target's estimated heading (drones usually fly nose-forward)."*
*Implementation:* marginalise each mixture component over the ψ prior inside `rtr_range`;
heading prior from the track's velocity direction. *Today:* none.

**Pinhole range — red** — [P Fig. 1, §2.2] Red because ADG-YOLO already *"recognises the drone
model, selects that model's width, and puts it into the pinhole formula, reaching a few percent
error over tens of metres from a fixed camera."* Eq. (1) d = f·W/w. The block also carries
Eq. (2), ∂d/∂w = −d²/(f·W), because that single fact *"drives two design choices later: errors
must be reported per range band … and the estimate needs an uncertainty that widens with
distance"* [P §3.1].
*Implementation:* a pure function in `rtr_range` reading fx from `CameraInfo`. *Today:* the fx
it needs is verified (919.42 px, `ost.txt`), and the driver's intrinsics-rescale correctness
(`node.py:971-1024`) protects it from the classic silent bug the driver's own docstring warns
about — a wrong fx would be an undetectable multiplicative range error.

**Range mixture — green** — [P §3.4] Eq. (4): p(d) = Σk πk·N(d; f·W̄k/w, σ²dk). Green because
*"the single-model version of this idea is used for pedestrians in existing work; the
multi-model version, driven by a recognition posterior over a dimension table, is what we
add."* The block's contract is behavioural: *"when recognition is confident the mixture is
narrow and model-specific; when recognition is uncertain — as it must be for the smallest
targets — the mixture widens toward the unknown prior on its own; and a genuine
misclassification shows up as a spread-out estimate, not a confident wrong one."* σdk combines
the box-width noise (*"itself larger for smaller targets"*) with the table + aspect spread
through Eq. (2).
*Implementation:* `rtr_range` publishes a `RangeMixture` message (components + collapsed
summary); the σw(w) box-noise model is calibrated in simulation in Q2. Tested by E2 (accuracy)
and E3 (S3: mixture vs collapsed point, judged by filter consistency). *Today:* none.

### Estimation half (C3)

**Observer gyro (input)** — [P §3.5] used in three places; **[V]** the hardware has none:
`node.py:569-574` sets `angular_velocity_covariance[0] = -1` with the comment *"The Tello
exposes NO gyroscope: the state packet carries fused attitude only."* The dual annotation on
the block (sim: true 100–200 Hz; hw: surrogate from ~10 Hz attitude) is **[D]**, forced by [V]
and consistent with the PDF's own framing [P §4.2]: *"Its state stream is far slower than the
200 Hz IMUs in the interception literature, so the filter-rate part of H4 is run at full rate
in simulation and at the achievable rate on hardware, and the hardware number is reported as a
lower bound."* Evidence the surrogate approach is workable at all: this repo's VIO branch
already runs preintegration on a 10 Hz surrogate rate [V README, Known limitations] — the
technique is proven in-repo even though that code is excluded here.

**Preintegration — grey** — [P §3.5] *"summarised between frames by preintegration [22]"*
(Forster et al.). Grey = accepted standard tool; innovating here would be claiming someone
else's ground. *Implementation:* rotation-only on-manifold preintegration between consecutive
frame timestamps, inside `rtr_observer_motion`. *Today:* none in this stack.

**Switch A · motion compensation — orange, dashed** — Orange because [P §1.3] *"published
air-to-air detectors cancel the observer's own motion using an image homography, or ignore it;
none uses the observer's inertial sensor"* — and because the 2026 ODD-SEC paper deliberately
avoids IMU compensation, *"a useful reminder that whether inertial aiding helps here is an open
question, not a settled one"* [P §2.1]. The mechanism [P §3.5]: *"At air-to-air ranges the
background is effectively at infinity, so the frame-to-frame image motion caused by the
observer turning is almost pure rotation, and a gyroscope predicts it directly."* Two outputs,
hence two out-edges: background alignment for the detector's motion channel, and the de-rotated
bearing for the tracker. The baseline arm is *"the image-only homography that current detectors
use [6, 16]"*, compared *"on identical data"*.
*Implementation:* `rtr_motion_comp`, parameter `mode: gyro | homography | none` (S4, swept by
E4). *Today:* none.

**Switch B · R inflation — orange, dashed** — [P §3.5] *"Following Pliska et al., the target's
measurement uncertainty is scaled up when the observer is turning fast, using its angular rate.
Their results predict the expected effect: no change in hover, a clear improvement under
manoeuvre"* — which is also the expected effect size stated up front in §5.3 (*"roughly a fifth
to a quarter less state error under manoeuvre and none in hover"*).
*Implementation:* measurement covariance scaled by a monotone function of |ω| inside
`rtr_tracker` (S5). *Today:* none.

**Switch C · delay bridging — orange, dashed** — [P §3.5] *"The filter therefore runs at the
IMU rate and re-uses the buffered inertial data to roll the state forward to each image's true
capture time, as in the delayed Kalman filter of Yang et al. [3]. Their evidence is that this
helps only when the filter is faster than the camera, so the filter rate is a swept parameter,
not a fixed choice"* (S6 + S7; E4's falsifier is *"a gain at filter rates below the camera
rate"*). **[V]** platform twist: the PDF's *"tens of milliseconds"* is here **150–350 ms** of
Wi-Fi video latency [V README] — an order of magnitude more, which makes this switch more
valuable and the hardware numbers latency-dominated (recorded as a risk in the plan).
*Implementation:* state buffer + measurement application at `t_capture` — this is the block
that *consumes* the dual-timestamp contract. *Today:* the substrate half-exists: the driver
stamps telemetry at change-detection with ≤ 20 ms bound (`node.py:17-24, 496-511`) and video on
arrival.

**Multiple-model tracker — grey** — [P §3.5] *"The target filter itself is a standard
multiple-model tracker, as used by Pliska et al., fed the mixture range from (4)."* Grey
because the filter is standard; the science is in its inputs and switches. The NEES annotation
comes from [P §5.1]: consistency *"checked with the standard normalised error-squared tests
against their expected bounds, over at least fifty repeated runs … This is more than the
closest published work reports."*
*Implementation:* IMM-style filter over target relative position/velocity; measurement = mixture
range + de-rotated bearing; NIS logged every update for E3. *Today:* none.

### Closed loop

**Following controller — grey, FIXED** — [P §4.1] *"A simple following controller closes the
loop; the controller is held fixed across all perception and estimation variants, because it is
not part of what is being tested."* Grey and switch-less by design: a controller that adapted
would confound every C2/C3 attribution.
*Implementation:* proportional bearing-centring + range closure under the speed cap; gains
tuned once in simulation, then frozen and documented. *Today:* none (only manual keyboard
control exists).

**Safety supervisor — grey** — the mechanisms are [P §6.2] (geofence, capped closing speed,
human-authorised engagement, auto-disarm on loss of link / loss of target / breach / low
battery); its **placement** as a separate node between any controller and the driver is [D],
justified three ways: IEEE 7009 asks for *mechanisms*, and a separate process is testable in
isolation and cannot be disabled by a controller bug; it must gate the RL arm identically if
that is ever added; and it composes with, rather than duplicates, the driver's own verified
fail-safes. *Today:* the driver-side pieces exist — see Diagram 4.

**Observer drone (actuator)** — [V] the command contract is fully implemented:
`node.py:1084-1092` maps REP-103 `cmd_vel` (normalised sticks ∈ [−1,1]) to SDK ±100 with the
FLU→Tello sign flips.

**E6 scoring — green** — green per Fig. 1 (*"a threshold-swept closed-loop test"* has no
published instance). The three-metrics-together rule is [P §5.1]: *"because a method can win on
one and lose on another"*; and the curve rule verbatim: *"success is defined very differently
across the field, from passing within a 2 m net plane down to sub-decimetre contact, so any
comparison must state its threshold, and we report success as a curve over the threshold rather
than at one point."* *Implementation:* offline evaluation over recorded bags + ground truth.
*Today:* none.

### The two edges that carry the design

- **tracker → aspect (dashed, "heading → ψ prior")** [P §3.3] — the one place estimation feeds
  perception; it must be drawn or the implementation would wire the pipeline as a pure feed-forward
  chain and quietly lose the sharpened prior.
- **drone → camera (dashed, "observer moves")** — the physical loop closure. It is why §4.1
  requires the *simulator* to be closed-loop, and why E6 cannot be replaced by open-loop replay:
  the perception input distribution depends on the controller flying.

---

## 2. Diagram 2 — one topic contract, two providers

**Why this diagram exists.** It is the one pure **[D]** architecture decision, so it gets its
own picture with its derivation on display. Premises: [P §4] simulation-first (for exact ground
truth + sweeps, and because safety requires closed-loop behaviour be exercised in sim before
flight); [P §4.3] *"the whole ablation is a configuration matrix rather than a code change"*;
[P §4.3] *"Every message carries both the time it was captured and the time it was received, so
latency is always a measured quantity."* Conclusion: the simulator and the hardware drivers
must publish **the same topic contract**, so that the phase swap is a launch substitution and
the Q6 hardware runs of E2/E4 [P Table 3] are configuration, not re-integration. If sim and
hardware had different interfaces, the config-matrix promise would silently break at the exact
moment (hardware) where it matters most.

**What exists today of this contract** [V]: the hardware side already honours the spirit of the
timestamp rule — the driver polls telemetry at 50 Hz and stamps on observed change, bounding
stamp error to 20 ms (`node.py:17-24`), and `TelloStatus` explicitly gained a `Header` *"so
consumers can time-align this with /imu and /image_raw"* (`TelloStatus.msg`). **What is new**
[D]: explicit `t_capture` + `t_receive` fields in every `rtr_msgs` message; on hardware,
`t_capture` is arrival time minus the *measured* latency — matching [P §4.2] *"the video and
state streams are time-stamped on arrival and aligned in software"* and *"Its Wi-Fi video
latency is measured, not assumed."*

---

## 3. Diagram 3 — ROS 2 node graph, hardware phase

**Why this diagram exists.** It is where the PDF meets the platform: processes, topics, and the
three deployment decisions (namespacing, target-off-host, mocap) live here, and each is either
verified or mandated. The node decomposition is not mine — it is [P §4.3] **verbatim**:

> "a detection-and-recognition module, a motion-compensation module (image-only or gyro-aided),
> the dimension-database lookup, the range module implementing (4), and the tracking filter."

mapping one-to-one to `rtr_detect`, `rtr_motion_comp`, `rtr_dimension_db`, `rtr_range`,
`rtr_tracker`. The additions beyond that list are each forced by something:

| Node | Provenance | Forced by |
|---|---|---|
| `rtr_observer_motion` | [D] from [V] | the PDF assumes a gyro; the Tello has none (`node.py:569-574`) — someone must manufacture the surrogate rate and the preintegrated rotation |
| `rtr_safety` | [P §6.2] + [D] placement | fail-safes must be concrete mechanisms; see Diagram 4 |
| `rtr_follow` | [P §4.1] | the fixed controller that closes E6 |
| `rtr_gt_bridge` | [P §4.2] | *"Ground truth comes from an indoor motion-capture volume, which gives the true pose of both aircraft: true range and bearing to score C2, and the observer's true attitude to score C3"* |
| `rtr_experiments` / evaluation | [P §4.3, §5] | the config matrix, ≥10 seeds with CIs, NEES over ≥50 runs, per-band reporting |
| `rtr_sim` | [P §4.1] | the sweep knobs E1/E5/H4 need |

**The target-off-host decision** [D from V]. Two verified facts make one host with two drivers
impossible as-is: djitellopy binds UDP :8889 in its constructor, so a second driver process
dies with `EADDRINUSE` (`node.py:205-220` handles exactly this error with advice); and both
drones in factory AP mode present the identical IP 192.168.10.1 (`node.py` default +
`check_network_path`, `node.py:1148-1176`). The resolution is not infrastructure but scope: the
perception host never needs the target's telemetry, because [P §4.2] the target's truth comes
from mocap — so the target flies scripted from a second machine and carries only markers.
Alternatives (EDU station mode, network namespaces) are documented in the plan §5.2 for the
case where ROS-driven target control becomes necessary.

**Namespacing** [V]: every driver topic is created with a relative name
(`node.py:355-381`), so `ns /observer` works without touching the driver; only the launch file
lacks a `namespace` argument (small future change, noted, not done — no code on this branch).

**Every edge label is a verified quantity:**

| Edge label | Verified where |
|---|---|
| `/observer/image_raw` 960×720, ~30 Hz, 150–350 ms late | `ost.txt` (size); `video_target_fps` default 30 (`node.py:118`) + README topics table; README platform constraints (latency) |
| `/observer/imu` ~10 Hz, no gyro | `node.py:569-574`; README (*"~10 Hz telemetry. And jittery."*) |
| `/observer/odom` metric flow velocity | `node.py:590-621` (twist-only; pose flagged unavailable); README (*"Metric velocity, for free … the single most valuable signal"*) |
| `/observer/cmd_vel` sticks [−1,1], ≥ 10 Hz, dead-man 0.35 s | `node.py:1084-1092` (mapping); `rc_timeout_sec` 0.35 (`node.py:120`, enforced `node.py:1099-1114`); README (*"Publish at 10 Hz or faster or the dead-man will keep zeroing it"*) |
| manual override `/observer/control` + `/observer/emergency` | `tello_control/src/main.cpp` — keys t/l/e/f + arrows/WASD; `/control` deliberately keeps the legacy axis convention (x = lateral) with the driver comment explaining why it must not be silently changed (`node.py:1074-1082`) |
| `/gt/*` ≥ 100 Hz | [D] — a *requirement* on the mocap bridge, not a measured fact; recorded as such |

The rosbag note is [P §4.3]: each sequence stores synchronised video, IMU, both true poses,
range/bearing, model identity, viewing angle, boxes — recording the full topic contract makes
the bag the dataset export source (C1).

---

## 4. Diagram 4 — the safety chain

**Why this diagram exists.** [P §6.2] states fail-safes *"as concrete mechanisms … rather than
promising to be careful"*, and *"Each is exercised in simulation before any flight."* E6 cannot
be flown before this chain exists. The diagram's specific job is **ownership**: three of the
mechanisms already exist in the verified driver, and drawing the chain prevents both
double-building them and assuming the rest.

| Block | Provenance | Detail |
|---|---|---|
| Operator | [P §6.1] | *"any engagement is human-authorised and a person keeps a hardware abort on both aircraft"* — the meaningful-human-control position [29] the thesis commits to |
| Engage gate | [P §6.2] | "human-authorised engagement" → zero command until armed, per run |
| Closing-speed cap | [P §6.2] | "a capped closing speed" → clamp on commanded velocity |
| Hard geofence | [P §6.2] + [D] | "a hard geofence at the edge of the flight volume"; mocap-fed is [D] — indoors, mocap is the only absolute position reference available |
| Auto-disarm | [P §6.2] | verbatim trigger list: *"loss of link, loss of target, geofence breach, or low battery"* |
| RC dead-man | [V] | `node.py:1099-1114`: stale command (> 0.35 s) → zeroed at the 20 Hz RC rate; exists precisely because the Tello *latches the last rc setpoint indefinitely* |
| Emergency motor-cut | [V] | `node.py:1046-1059`: sent twice by design — a fire-and-forget datagram that cannot queue behind a stalled `land()`, plus a retried queued copy |
| SDK auto-land | [V] | the aircraft lands itself after 15 s of RC silence (driver comment `node.py:1104-1105`; README troubleshooting) |
| Staged ladder | [P §6.2] | *"staged testing from simulation, to a static target, to a slow target, to full following"* |

**The honest gap, kept visible:** the proposal promises *"a hardware kill-switch on both
aircraft"* [P §6.2]; a Tello has no external kill line [V — no such interface exists in the SDK
or driver]. The practical equivalents are the keyboard emergency (the bypass path above) and
physically pulling the battery — to be stated in the thesis, not papered over.

---

## 5. Diagram 5 — the RL extension

**Why this diagram exists at all.** The user requires an RL plan; the verified finding
([rl_analysis.md](rl_analysis.md) §1) is that the PDF contains none — its only RL text is the
Henderson [27] statistics citation, and §4.1 fixes the controller. So the diagram's job is to
**draw the difference**: exactly one node changes between the proposal's baseline and the RL
variant, and both arms feed the same unchanged supervisor. That picture *is* the anti-confound
protocol: run all C2/C3 ablations under arm A, freeze the best configuration, then compare arms
on E6 with ≥ 10 seeds and confidence intervals [P §5.3].

| Block | Provenance | Why |
|---|---|---|
| `target_tracker` (unchanged) | [D] | the policy consumes the track, not pixels — end-to-end RL would bypass contributions C1–C3, need pixel-scale data, and make E6 unattributable |
| `follow_controller` arm A | [P §4.1] | the proposal's baseline and the comparison anchor; without it the RL result has no denominator |
| `rl_policy` arm B | [D] | the extension itself — every design choice justified below |
| `safety_supervisor` (unchanged) | [P §6.2] + [D] | the policy is never trusted; identical gating for both arms preserves §6 of the thesis untouched |
| driver / sim (unchanged) | [V] | byte-identical stack is what makes the arm comparison attributable |

**Per-choice justification of the RL formulation** (full design in
[rl_analysis.md](rl_analysis.md) §4):

- **POMDP framing** [V]: 150–350 ms video latency, ~10 Hz jittery telemetry, σd ∝ d² noise
  (Eq. 2), and a target that can leave the frame — a memoryless policy cannot be Markov against
  its own delayed action effects.
- **Observation entries** — each is there for a reason: relative position (the task variable);
  relative velocity (closing-speed control, and compliance with the cap); **σd from the
  mixture** (the one entry unique to this thesis — it lets the policy behave differently on an
  honest-but-wide range, connecting C2 to control); visibility flag + staleness (the dominant
  failure is losing the target); observer velocity (metric, from the flow sensor — the
  platform's best signal [V README]); roll/pitch only (yaw excluded because it free-runs [V]);
  last k = 4 actions (350 ms of latency at 10 Hz — the latency-compensation memory).
- **Action** [V]: the driver's exact `cmd_vel` contract (`node.py:1084-1092`), ≥ 10 Hz because
  of the dead-man — which is kept deliberately: a hung policy *stops* the aircraft.
- **Reward** — every term maps to an E6 metric [P Table 1: success, time-to-close, closest
  approach] or a platform fact (smoothness term because Tello sticks saturate; loss penalty
  because FOV loss dominates; the terminal success condition requires closing speed ≤ the cap so
  success cannot be bought by violating safety). Success is evaluated as a threshold **curve**
  [P §5.1].
- **PPO first, SAC documented alternative** [D]: robust on noisy delay-dominated dynamics,
  parallelises over the timing-faithful sim, most reproducible seed-to-seed — and
  reproducibility is not optional here, because the proposal already commits to the Henderson
  protocol [P §5.3], which exists *because of* RL seed sensitivity.
- **Shielded training** [D]: train with the supervisor's clamps active so there is no
  train/deploy mismatch; curriculum mirrors the safety ladder [P §6.2], so training stages and
  flight-clearance stages are one story.

---

## 6. Why the switchboard is a table, not a sixth diagram

The twelve switches are configuration dimensions, not dataflow: drawing them as boxes would
duplicate Diagram 1's dashed borders while hiding the two things that matter operationally —
**which single node owns each switch** (the §4.3 one-module-per-switch rule) and **which
experiment sweeps it**. A table carries exactly those two columns; a diagram cannot without
becoming a table wearing boxes.

---

## 7. The implemented / planned ledger (precise)

**Implemented and verified today (reused as-is):**
- `tello` driver — all topics, the cmd_vel contract, dead-man, emergency dual path, video
  health restart, intrinsics rescaling, change-detection stamping (`workspace/src/tello/tello/node.py`)
- `tello_msg` — `TelloStatus` (with Header), `TelloMissionPad` (EDU-only absolute reference,
  fallback for smoke tests), `TelloID`, `TelloWifiConfig`
- `tello_control` — keyboard GUI: manual sticks, takeoff/land/flip, emergency key
- camera calibration (`ost.txt`/`ost.yaml`, fx = 919.42 px) and the launch file
- provisioning scripts (`scripts/gazebo.sh` = Gazebo classic 11 — treated as legacy, §5.1 of the plan)

**Implemented on this branch:** documentation only — the four planning documents plus this one.
No code, by explicit decision.

**Not implemented (all planned):** every `rtr_*` package (msgs, dimension DB, detect,
motion-comp, observer-motion, range, tracker, follow, safety, sim, gt-bridge, experiments),
the simulator scenes, the mocap bridge, and all experiments E1–E6.

**Explicitly excluded:** the `tello_vio` estimator (branch owner's decision — this thesis
tracks *another* drone, not the observer's own pose). One lesson from it is kept: its
surrogate-rate preintegration [V README] is the existence proof for `rtr_observer_motion`'s
hardware mode.
