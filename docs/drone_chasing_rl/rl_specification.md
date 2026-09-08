# The Reinforcement Learning Specification — States, Actions, Reward, Method

This is the reference document for the RL half of the work. For each element it gives **what the
paper specifies**, **what it leaves undefined**, and **a concrete recommendation** with the
reasoning. Tags: **[P]** from the paper (section cited), **[V]** verified in this repository or
against the manufacturer, **[D]** a design decision of this plan.

---

## 1. The problem as an MDP

**[P]** The paper never writes the MDP down formally, so this section states it explicitly from
what the paper does describe. The task is *visual servoing*: keep the detected target's bounding
box centred in the follower's image, and hold its apparent size inside a band.

| Element | Content |
|---|---|
| Agent | The follower drone's controller, running on the central computer |
| Environment | The follower's own dynamics, the target's motion, the camera projection, and the wireless link |
| Observation | Derived from the YOLO bounding box only — **the camera is the sole input source** [P abstract] |
| Action | Continuous velocity corrections on two axes [P §1] |
| Reward | A function of the pixel distance between box centre and frame centre [P §2.2.3] |
| Episode | 10 steps [P Table 2] |
| Objective | Minimise the pixel error, i.e. *"the difference between the red and blue dots is minimized"* [P §2.1.2] |

**It is a POMDP, not an MDP** [D]. The observation is a single image-plane projection, so target
depth, target velocity, and the follower's own motion are not directly observable; and the
measured 150–350 ms link latency [V] means the observation describes the world as it was, not as
it is. The paper does not address this. The practical consequence is in §3 below: the observation
must carry some history, or the policy is being asked to control a delayed system from a
memoryless snapshot.

---

## 2. Geometry — the numbers the whole problem runs on

**[P §2.1.2]** The frame is 960 × 720 and the centre is fixed at (480, 360). **[V]** This matches
this repository's camera exactly (`workspace/src/tello/resource/ost.txt`), and the centre
recomputes to (480, 360) as stated.

**[V, recomputed]** Useful derived quantities the paper does not state but that every threshold
depends on:

| Quantity | Value | Why it matters |
|---|---|---|
| Maximum centre-to-corner distance | **exactly 600.0 px** (√(480² + 360²)) | the natural normaliser for `dist`; it makes the reward bounded and scale-free |
| Reward threshold 110 px as a fraction of that | **18.3 %** | the "close enough" zone is a disc of radius 110 px, about a fifth of the way to the corner |
| Half-width / half-height | 480 / 360 px | the natural per-axis normalisers for the error components |

**[D] Normalise everything by these constants.** Working in raw pixels ties the policy to one
resolution; normalising by 480, 360 and 600 makes the trained actor transferable if the stream
size ever changes (for example if `video_scale` is used, which this repository supports and which
rescales intrinsics correctly [V `node.py:971-1024`]).

---

## 3. State / observation

### What the paper says — and the contradiction

**[P]** Three different descriptions appear:

1. Abstract: *"The **size information of the box** was sent to the agent trained with the
   DDPG … algorithm to determine the action"*
2. §2.1: *"These observations include information such as the **position, orientation, and speed
   of the followed drone**"*
3. §2.2.3: the reward is computed from the **centre of the box** versus the centre of the screen

**[P + D] These cannot all be true.** The paper states the camera is *"the only input source"*, so
the target's orientation and speed are not measurable quantities in this system; description 2 is
loose language. Description 1 names size, but size is what the *hand-coded depth rule* consumes,
not the agent — and the reward, which defines what the agent optimises, is built purely from the
box **centre**. The only self-consistent reading is that **the agent observes the centring
error**.

### [D] Recommended observation vector

Two versions. Implement the faithful one first so the paper's result is reproducible, then the
recommended one as the improvement — and keep both behind a parameter so the comparison is a
configuration change, not a code change.

**Faithful (paper-minimal), 2 dimensions:**

```
s = [ ex_n , ey_n ]
  ex_n = (xc - 480) / 480      in [-1, 1]
  ey_n = (yc - 360) / 360      in [-1, 1]
```

**Recommended (latency-aware), 8 dimensions:**

```
s = [ ex_n , ey_n ,                    current centring error
      ex_n_prev , ey_n_prev ,          previous error  -> implicit image-plane velocity
      ratio_n ,                        box-to-screen ratio, normalised
      visible ,                        1.0 if detected this step, else 0.0
      staleness_n ,                    time since last detection / loss timeout
      a_prev_v , a_prev_h ]            last action actually sent
```

That is 9 entries as written; drop `ratio_n` if the depth rule is to stay strictly outside the
learning problem as the paper intends.

**Why each addition** [D]:

- **Previous error** gives the policy image-plane velocity. Without it the agent cannot tell an
  approaching target from a receding one, and cannot lead a moving target at all — which is
  precisely where the paper's scenarios degrade (86 %, 82 %, and 42 % as target motion grows).
- **Last action** is the standard remedy for a delayed actuator: with 150–350 ms of measured lag
  [V] the effect of an action is not visible in the next observation, and a memoryless policy
  responds by oscillating. This is the single highest-value addition.
- **Visibility and staleness** let the policy behave differently when the box is missing, instead
  of acting on a stale error. The behaviour manager handles the hard cases, but the policy should
  not be fed a phantom error in the meantime.

**[D] Normalisation and clipping.** Clip `ex_n`, `ey_n` to [−1, 1] and treat a missing detection
by zeroing the error terms and setting `visible = 0` — never by holding the last error, which
teaches the policy to chase a ghost.

---

## 4. Action

### What the paper says

**[P §1]** *"The DDPG agent determines the **up-down and yaw** movements of the drone, and the
forward-backward movements are determined by the size of the box."*
**[P §2.1.2]** *"Other actions (**up, down, right, left**) are determined by the agent trained by
the DDPG algorithm."*

**[P §2.2.2]** DDPG is used precisely because it learns *"continuous control policies"*, so the
action space is continuous — this is not in doubt even though the two sentences above disagree on
which horizontal axis is meant.

### [D] Recommended action space

```
a = [ a_v , a_h ]  ,  each in [-1, 1]      continuous, 2-dimensional

a_v  -> vertical velocity      -> geometry_msgs/Twist linear.z
a_h  -> yaw rate               -> geometry_msgs/Twist angular.z
```

**Why yaw rather than lateral translation** [D]: §1 is the paper's own contribution summary and
says yaw; yaw rotates the camera without changing the standoff distance that the depth rule is
regulating at the same time, so the two controllers stay decoupled; and the paper's search
behaviour rotates the aircraft about its own axis [P Appendix A, Scenario 3], which only works if
yaw is the re-centring authority. Keep lateral translation (`linear.y`) as a documented ablation
rather than the default.

**[V] Mapping to the aircraft.** The action maps directly onto the existing driver contract with
no conversion layer: `/cmd_vel` takes REP-103 normalised sticks in [−1, 1] and scales them to the
SDK's ±100 internally (`node.py:1084-1092`, clamp at `node.py:1094-1097`). So the actor's output
range and the driver's input range are already the same interval — a rare piece of luck worth
preserving by **not** rescaling anywhere in between.

**[D] The third axis stays outside the policy.** `linear.x` comes from the depth rule
(§5 below), and the policy must never write it, or the learned and hand-coded controllers will
fight over the same degree of freedom.

**[D] Rate limiting.** Clamp the change in action per step. The Tello's sticks saturate quickly
and a policy trained without a smoothness constraint produces visible judder; the paper
acknowledges it considered *"penalizing abrupt direction changes"* and dropped it *"due to the
increased training complexity"* [P §2.2.3]. A rate limit on the output achieves most of that
benefit without touching the reward.

---

## 5. The hand-coded depth rule (outside the learning problem)

**[P §2.1.2]** *"The follower drone determines its distance from the target drone according to the
ratio of the detected bounding box to the primary screen size. If this ratio is below 20 %, it is
aimed to move forward; if it is above 55 %, it is aimed to move backward. If this ratio is between
20 % and 55 %, it will not move."*

**Ambiguity #10 — "ratio" is not defined** [P + V]. The paper does not say whether this is an
**area** ratio or a **linear** ratio, and the two imply very different standoff distances.
Recomputed for a 960 × 720 frame:

| Interpretation | 20 % threshold | 55 % threshold |
|---|---|---|
| Area ratio (box area ÷ frame area) | equivalent square side ≈ 372 px | ≈ 617 px |
| Linear ratio (box width ÷ frame width) | 192 px wide | 528 px wide |

Under either reading the standoff band is **close** — of the order of one to two metres for a
typical small quadrotor at this focal length. That is consistent with an indoor sports hall, but
it is worth being deliberate about, because it also means the "25 m" starting distance of
Scenario 1 [P Appendix A] is an *approach* phase, not a following distance.

**[D] Recommendation**: implement the linear ratio (box width ÷ frame width), because it is
linear in 1/distance and therefore gives a well-behaved control signal, and **calibrate the two
thresholds** against the actual target airframe rather than inheriting 20 % / 55 % blindly —
the thresholds encode a distance, and the distance they encode depends on the target's real size.
Record the calibrated values with the results.

---

## 6. Reward

### What the paper specifies [P §2.2.3]

```
dist   = Euclidean distance between the box centre and the frame centre, in pixels
reward = -0.25 * dist        if dist > 110
       =  100 - dist         if dist < 110
```

The 110-unit threshold *"was determined as 110 units due to the experiments"*.

### Verified problems

**[V, recomputed]** Three defects, all mechanical consequences of the formula as printed:

| # | Problem | Evidence |
|---|---|---|
| 1 | **Discontinuity at the threshold.** Just inside: 100 − 110 = **−10**. Just outside: −0.25 × 110 = **−27.5**. Crossing outward from 109 to 111 px costs about 18.75 reward for one pixel of motion | evaluated directly from Eq. (6) |
| 2 | **Undefined exactly at dist = 110** — both branches use strict inequalities | Eq. (6) as printed |
| 3 | **The prose contradicts the formula.** The text says that at the centre *"the reward reaches its maximum value (e.g. 0 or close to it)"*, but Eq. (6) returns **100** at dist = 0. The text's other example is consistent: *"a very low value (e.g., −50)"* matches −0.25 × 200 | §2.2.3 prose vs Eq. (6) |

A discontinuity of that size at the boundary is not cosmetic: it creates a cliff in the value
function exactly where the agent spends most of its time, which is a plausible contributor to the
negative total reward reported for even the best hyperparameter set (−4.2, [P Table 3]).

### [D] Recommended reward

Keep the paper's intent — a steep gradient near the centre and a gentler one far away — but make
it continuous by anchoring the outer branch at the threshold value:

```
r = 100 - dist                       if dist <  110
r = -10 - 0.25 * (dist - 110)        if dist >= 110
```

At dist = 110 both branches give **−10**, so the function is continuous and everywhere defined,
and the far-field slope is unchanged from the paper. Add two terms the paper's own failure
analysis argues for [D]:

```
r -= w_loss   on the step where the target leaves the field of view   (terminal)
r -= w_smooth * || a_t - a_(t-1) ||^2                                 (actuation smoothness)
```

The loss penalty matters because losing the target *is* the failure mode the evaluation measures
(the reported metric is percentage of time in view). The smoothness term is the one the paper
considered and dropped [P §2.2.3]; with a rate-limited action it can be given a small weight or
omitted — decide once, record it.

**[D] Report both.** Train and report the paper's exact reward as the baseline arm and the
repaired reward as the comparison. That single ablation is a genuine, publishable contribution:
it isolates the cost of the discontinuity.

---

## 7. Algorithm — DDPG

**[P §2.2.2]** Actor-critic, off-policy, replay buffer, target networks. The paper gives the two
update equations:

```
(3)  y_i = r_i + gamma * Q'( s_(i+1), mu'(s_(i+1) | theta_mu') | theta_Q' )
(4)  L   = (1/N) * sum_i ( y_i - Q(s_i, a_i | theta_Q) )^2
```

where *"Q' is the state-action value estimated by the target Critical Network, and μ' is the
action chosen by the target actor network"* [P §2.2.2].

**Note on equations (1) and (2)** [P §2.2.1]: these are tabular Q-learning and the optimal
action-value definition. They are background, **not** what DDPG implements — the paper presents
them while motivating why function approximation is needed for *"large and continuous state
spaces"*. Do not implement them.

**[D] Why DDPG is a defensible choice here, and where it is weak.** It fits because the action
space is continuous and low-dimensional, the reward is dense (every step yields a distance-based
signal), and the problem is small enough to train on a CPU in hours [P §2.2.4]. Its known
weaknesses are directly relevant: DDPG is sensitive to hyperparameters and prone to
overestimation in the critic, which is exactly what TD3 (twin critics, delayed policy updates,
target-policy smoothing) or SAC (entropy-regularised, generally more robust out of the box) were
designed to fix. **Recommendation: reproduce with DDPG for fidelity, then run TD3 or SAC as a
second arm.** Both accept the identical observation and action spaces, so this is a swap of the
learner, not of the architecture — and given that this paper's own comparison table cites a study
finding DQN outperforming DDPG on a similar task [P Table 1, ref 29], the algorithm choice is
visibly not settled in this literature.

---

## 8. Hyperparameters

### [P Table 2] As published

| Parameter | Value | Paper's description |
|---|---|---|
| Learning rate, actor **and** critic | 0.0001 | step size for weight updates |
| Gamma (γ) | 0.99 | discount factor |
| Sigma (σ) | 0.3 | exploration noise added to actions |
| `nb_max_episode_steps` | **10** | max steps per episode before termination |
| Number of episodes | 100 000 | total training episodes |
| Interval | 100 | logging/evaluation frequency |

Selection method: empirical grid search followed by Bayesian optimisation on γ, α and σ
[P §2.2.4]. Training hardware: Intel i5, 8 GB RAM, **CPU only**, ≈ 4 hours [P §2.2.4, Appendix A].

**[P Table 3]** Sensitivity, as reported:

| Parameter set | Total reward | Episodes to train | Stability |
|---|---|---|---|
| **Selected**: γ = 0.99, α = 1e-4, σ = 0.3 | −4.2 | 25 000 | High |
| γ = 0.95, α = 1e-3, σ = 0.1 | −12.3 | 35 000 | Medium |
| γ = 0.9, α = 5e-4, σ = 0.5 | −20.1 | 50 000 | Low |

### Verified problems with the published set

**Ambiguity #9 — γ = 0.99 against a 10-step episode** [V, derived]. The effective horizon of a
0.99 discount is on the order of 1/(1−γ) = 100 steps, ten times the entire episode. Within a
10-step episode the discount is nearly inert, so the reported preference for γ = 0.99 over 0.95
and 0.9 is unlikely to be a genuine long-horizon effect. **[D]** Either lengthen the episode
substantially (200–500 steps, which also lets the agent experience a target that manoeuvres), or
acknowledge that the task is effectively a bandit-like immediate-reward problem and say so.
Lengthening is the better choice: a 10-step episode cannot represent a chase.

**Ambiguity #5 — three different training lengths.** Table 2 says 100 000 episodes, Table 3 says
25 000, and the Conclusions say *"After 500 training steps were performed, it was concluded that
the loss and reward values were at the desired level."* [P §5]. **[D]** Treat 25 000 episodes as
the operative convergence figure (it is the one attached to the selected parameter set) and report
your own convergence curve rather than inheriting any of the three.

### [D] The parameters the paper does not give — and recommended values

None of the following appear anywhere in the paper, and DDPG cannot be run without them. These are
starting points, to be recorded with the results:

| Parameter | Recommended | Reasoning |
|---|---|---|
| Actor network | 2 hidden layers, 256 units, ReLU, `tanh` output | standard for low-dimensional continuous control; `tanh` gives the [−1, 1] action range for free |
| Critic network | 2 hidden layers, 256 units, ReLU, action injected at the first hidden layer | standard DDPG |
| Replay buffer capacity | 1e5 transitions | ample for a problem this small; the whole run fits in memory |
| Minibatch size N | 64 | the N of Eq. (4), unspecified in the paper |
| Target update rate τ | 0.005 (soft) | the paper mentions target networks but never their update rule |
| Optimiser | Adam | implied by the learning-rate framing |
| Exploration noise process | Gaussian, σ = 0.3, decayed over training | the paper gives σ but not the process; Gaussian is simpler than Ornstein-Uhlenbeck and generally performs as well |
| Warm-up steps before learning | 1000 | avoids updating from a near-empty buffer |
| Control / policy rate | **10–20 Hz** | [V] forced by the driver's 0.35 s dead-man (`node.py:1099-1114`); also sets the environment's step duration |
| Episode length | 200–500 steps | see the γ discussion above |
| Random seeds | ≥ 5, report mean and spread | DDPG results vary substantially across seeds; a single-seed number is not evidence |

**[D] The control rate deserves emphasis**: it is the bridge between training and reality. The Gym
environment's step must correspond to the real control period, or every velocity the policy learns
is scaled wrongly on the aircraft. Fix it at one value, use it in both places, and record it.

---

## 9. The training environment

**[P]** The paper says only that *"the OpenAI Gym simulator is designed according to our
problem"* and that training took about four hours on a CPU. **The environment's dynamics model is
never described** — there is no statement of how target motion, follower response, or the camera
projection are simulated. This is the largest single gap for reproduction.

**[D] Recommended environment specification** — deliberately minimal, because the observation is
only image-plane geometry:

- **State to simulate**: relative position of the target in the camera frame, expressed as the
  bounding-box centre and size. A full 6-DoF flight simulator is unnecessary: the policy only ever
  sees the projection.
- **Follower response**: first-order lag from commanded velocity to achieved velocity, with a time
  constant calibrated from a real step-response flight. Cheap, and far closer to the truth than an
  instantaneous response.
- **Target motion**: a generator covering the paper's five scenarios — stationary, constant
  velocity, vertical oscillation, horizontal oscillation, and high-speed variable motion — with
  randomised amplitude and frequency.
- **Latency injection**: delay the observation by a sample drawn from the **measured** 150–350 ms
  distribution [V]. This is the key addition; the paper's own post-hoc analysis blames latency for
  its failures, so training under it is the obvious fix.
- **Detection noise**: bounding-box jitter, and dropout at a rate matched to the ~95 % detection
  accuracy the detector actually achieves, rising for small targets [P Appendix A].
- **Episode termination**: target leaves the frame, or the step limit is reached.
- **Domain randomisation**: over latency, target dynamics, detection dropout, and the follower's
  lag constant — so the policy is not tuned to one particular simulated world.

**[D] Validate the environment before trusting the policy.** Record a real flight under manual
control, replay the observed box trajectory through the environment's dynamics, and compare. An
unvalidated simulator is the standard reason a policy that works in training fails in the air.

---

## 10. Evaluation protocol

**[P Table 4, Appendix A]** The paper's five scenarios, with its reported results:

| Scenario | Target behaviour | Duration | Tracking accuracy | Detection accuracy |
|---|---|---|---|---|
| 1 | stationary, 25 m initial distance | 4 min | 99 % | 95 % |
| 2 | constant-speed route | 9 min | 95 % | 94 % |
| 3 | vertical oscillation, variable speed, irregular intervals | 10 min | 86 % | 90 % |
| 4 | horizontal oscillation, variable speed | 9 min | 82 % | 90 % |
| 5 | variable **high** speed | 5 min | **42 %** | **60 %** |

**[P]** The two metrics are: **tracking accuracy** = the fraction of flight time the target
remained in the follower's field of view; **detection accuracy** = the fraction of time the
detector found it. **[P Table 5]** A brightness study repeats **only scenarios 2 and 3** at low,
medium and high illumination; medium is best in every cell.

**[V, recomputed]** The five-scenario summary statistics check out exactly: means of 80.8 % and
85.8 %, with sample standard deviations of 0.2273 and 0.146 and variances of 0.051 and 0.021 — all
as printed [P §3]. However, **the brightness-study dispersion figures for tracking do not
reproduce**: the two *detection* rows recompute exactly (S2: s = 2.52, s² = 6.33; S3: s = 4.04,
s² = 16.33, both matching the paper), but the reported *tracking* figures (S2: s = 2.08,
s² = 4.33; S3: s = 7.64, s² = 58.33) match neither the sample nor the population estimator applied
to Table 5's own values. They are internally consistent (σ² ≈ σ × σ), which points to a
transcription slip rather than a method error. **[D] Use the raw per-cell values, not the
published dispersions.**

### [D] Additions to the protocol

- **Cap the target's speed, or the result is predetermined.** [P Table A.1] gives the follower a
  max speed of 8 and the target 15 in the same units — the chaser is roughly half as fast as its
  quarry. Scenario 5's collapse to 42 % is at least partly a kinematic inevitability rather than a
  control failure. Either limit the target to a speed the follower can actually match, or report
  the speed ratio alongside every result so the two effects can be separated.
- **Report the latency you measured**, per run. It is the paper's stated dominant limitation and
  this platform can quantify it [V].
- **Report per-seed results.** The paper reports single numbers; DDPG's seed variance makes single
  numbers weak evidence.
- **Add a null baseline.** A simple proportional controller on the same pixel error is a few lines
  of code and answers the question the paper never asks: *does the learned policy beat a P
  controller?* Without that comparison the value of the RL component is unquantified. This is the
  single most valuable addition to the evaluation.

---

## 11. Summary — what to build, in order

1. **Detector** first: without boxes there is no observation, no reward, and no evaluation.
2. **Geometry and the observation vector**, with the faithful 2-D version and the recommended
   latency-aware version behind one parameter.
3. **The Gym environment**, with measured latency injected — and validated against a real flight.
4. **DDPG training**, paper hyperparameters first, then the fixes from §6 and §8.
5. **The P-controller baseline**, in parallel — it is cheap and it calibrates every later claim.
6. **Deployment** as an inference-only node behind the safety supervisor.
7. **The five scenarios**, with the speed ratio recorded and multiple seeds.

The honest summary of the source: the paper gives a clear and workable **skeleton** — hybrid
learned/coded control, a distance-based reward, DDPG, and a five-scenario evaluation — but it
underspecifies the observation, omits the network and buffer configuration entirely, prints a
discontinuous reward, and reports a training length three different ways. All of that is
recoverable, and every gap above has a recommendation attached. What makes this platform the right
place to do it is that the paper's own headline limitation, the wireless latency, is already a
measured quantity here.
