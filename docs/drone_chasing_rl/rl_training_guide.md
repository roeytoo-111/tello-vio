# The RL Training Guide — From Beginner to Expert, for the Chase Project

**What this file is.** The complete manual for the learning half of the project, built as a
ladder: **Part I** teaches reinforcement learning from nothing, with derivations, not just
statements; **Part II** builds DDPG and TD3 equation by equation, every symbol defined;
**Part III** specifies the training system to implement; **Part IV** is the expert layer —
reward-design theory, the statistics of RL claims, sim-to-real reasoning, and an oral-exam
section with the questions a committee asks and the answers an expert gives. Read Parts I–II to
*understand*, Part III to *build*, Part IV to *defend*.

Reading order of the document set, so you know where you are:

| Document | Role |
|---|---|
| [rl_foundations.md](rl_foundations.md) | the conceptual companion — *why* each mechanism exists, narrated |
| [rl_specification.md](rl_specification.md) | *what* to build — the decision record for states, actions, reward, hyperparameters |
| [rl_block_diagram.md](rl_block_diagram.md) | *what the original actually did* — the paper's published code, line-verified |
| [implementation_plan.md](implementation_plan.md) | *where it runs* — ROS 2 packages, phases, safety |
| **this guide** | *how to understand, build, train, and defend it* — end to end |

**Evidence tags**: **[P]** the chase paper (§ cited) · **[C]** its published code
(`ziya44/Drone_tracking_with_drone` @ `840487e`, line-cited) · **[V]** verified in this
repository or by recomputation (every derived number in this file was recomputed in a script
before being written down) · **[D]** a design decision of this project · **[L1]–[L9]** the
literature, each fetched and read for this guide (2026-09-13), not quoted from memory:

| Tag | Source | Provides |
|---|---|---|
| [L1] | Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018 (ch. 3, 6, 11) | MDPs, returns, Bellman, TD, the deadly triad |
| [L2] | Silver et al., *Deterministic Policy Gradient Algorithms*, ICML 2014 | the DPG theorem |
| [L3] | Lillicrap et al., *Continuous control with deep RL*, ICLR 2016 — **PDF fetched, §7 Experiment Details read in full** | DDPG; original hyperparameters and initialisation |
| [L4] | Fujimoto, van Hoof, Meger, *Addressing Function Approximation Error in Actor-Critic Methods*, ICML 2018 | TD3: overestimation analysis, the three fixes |
| [L5] | OpenAI Spinning Up, DDPG & TD3 pages — **fetched, equations and defaults quoted** | canonical modern presentation, documented defaults |
| [L6] | `sfujim/TD3`, the author's implementation — **fetched, code quoted** | exact network sizes, defaults, update code |
| [L7] | Mnih et al., *Human-level control through deep RL*, Nature 2015 | replay buffer + target networks (DQN lineage) |
| [L8] | Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018 — **abstract fetched** | seed variance and reporting standards |
| [L9] | Ng, Harada, Russell, *Policy Invariance Under Reward Transformations*, ICML 1999 — **citation verified** | which reward changes preserve the optimal policy |

---

## 0. Notation — every symbol used anywhere in this guide

| Symbol | Name | Meaning here | Shape / value |
|---|---|---|---|
| `t`, `Δt` | step, control period | one tick of the loop; **the same constant in simulation and flight** | Δt = 1/f_ctrl, f_ctrl = 10 Hz [D] |
| `s_t` (`o_t`) | state (observation) | the vector the policy sees (§4.4) | ℝ^(6+2k), entries in [−1,1] |
| `a_t` | action | (vertical velocity, yaw rate) as normalised sticks | ℝ², each in [−1,1] |
| `r_t` | reward | scalar payment for step t (§16.2) | ≈ [−1.3, +1] after scaling [D] |
| `d_t` | terminal flag | 1 only if the **task** ended — never a timeout (§11.3) | {0, 1} |
| `γ` | discount factor | how much the future counts (§5) | 0.99 |
| `G_t` | return | discounted sum of rewards from t onward | ℝ |
| `π(a\|s)` | stochastic policy | probability of a in s (general theory) | — |
| `μ_θ(s)` | actor | our deterministic policy network, parameters θ | obs → [−1,1]² |
| `Q_φ(s,a)` | critic | action-value network, parameters φ | obs × ℝ² → ℝ |
| `V^π`, `Q^π`, `Q*` | value functions | expected return under π; under π given first action; optimal | ℝ |
| `μ′, Q′` (θ′, φ′) | target networks | slow copies used only to build labels (§11.5) | same shapes |
| `τ` | soft-update rate | blend of live weights into targets per gradient step | 0.005 [D] |
| `D`, `B`, `N` | buffer, minibatch, size | replay store; a sampled batch; its size | 10⁵ · — · 64 [D] |
| `y_i`, `δ` | TD target, TD error | regression label; `y − Q(s,a)`, the surprise (§7) | ℝ |
| `α` | learning rate | step size (bandit: averaging; deep: Adam) | 3e-4 [D] |
| `ε`, `σ` | exploration noise, scale | added to executed actions, **training only** | σ 0.3 → 0.05 decayed [D] |
| `θ_OU, μ_OU, σ_OU, Δt_OU` | OU parameters | mean reversion, mean, volatility, integration step (§11.2) | see §11.2 |
| `σ̃`, `c` | target-smoothing std, clip | TD3 trick ② (§13.2) | 0.2, 0.5 [L4][L6] |
| `p(s′\|s,a)` | transition law | environment dynamics — sampled, never modelled | — |
| `ρ₀`, `ρ^β` | initial / behaviour state distribution | where episodes start; which states the data visits | — |
| `T` | Bellman operator | the map whose fixed point is Q* (§6.3) | — |
| `Φ(s)` | shaping potential | any function of state used in reward shaping (§22) | ℝ |
| `k` | action-history length | past actions carried in the observation | k = ⌈Δ_max·f_ctrl⌉ = 4 |
| `Δ` | link latency | camera-to-computer delay | **150–350 ms measured** [V] |
| `fx` | focal length | this camera's calibration | 919.42 px [V `ost.txt`] |
| `dist` | centring error | box centre ↔ frame centre (480, 360) | 0–600 px [V] |

A plain-words **glossary** of the spoken vocabulary is Appendix A.

---

# PART I — FOUNDATIONS (beginner → intermediate)

## 1. What RL is — and why supervised learning cannot solve this

Supervised learning needs labelled examples: *input → correct output*. To train the chase that
way you would need, for millions of frames, the **correct stick command** — which nobody knows.
There is no oracle pilot function to imitate; there is only a *measurable outcome*: was the box
near the centre, did the target stay in view. Reinforcement learning is the branch of machine
learning for exactly this situation [L1]:

| | Supervised learning | Reinforcement learning |
|---|---|---|
| Training signal | the correct answer per input | a scalar **reward** — how good things turned out |
| Feedback timing | immediate, per example | possibly **delayed** — an action's consequence appears later |
| Data distribution | fixed dataset | **the agent's own behaviour** decides what data exists |
| Core difficulty | generalisation | **credit assignment** + **exploration** |

The two named difficulties, in our system's terms:

- **Credit assignment.** The target leaves the frame at t = 3.0 s. Which of the last thirty
  commands caused it? With 150–350 ms of latency [V], the guilty command is *seconds* old.
  Value functions (§6) are the machinery that spreads outcome-credit backward over time.
- **Exploration.** The learner only knows about actions it has tried. A policy that never yaws
  hard left can never learn that hard-left was sometimes right. Something must force variety
  (§2, §11.2) — and this creates RL's signature tension, **exploration vs exploitation**: use
  what you know, or probe for something better.

One more distinction to speak precisely about: we are **model-free**. We never learn or write
down `p(s′|s,a)` — the physics, Wi-Fi and detector are only ever *sampled* by acting. The
simulator (§16) exists to make that sampling cheap and safe, not to hand the agent a model.

## 2. Warm-up: one frozen moment — the bandit

Strip time away and RL's kernel is already visible. Freeze one situation: target 200 px left.
Two candidate actions: `yaw-left-soft`, `yaw-left-hard`. Each, when tried, yields some reward
(next-frame centring, noisy). This is a **two-armed bandit** [L1 ch. 2] — no states, no future,
just: which arm pays more, and how do you find out while paying for every trial?

Keep a running estimate per arm, updated incrementally after each pull:

```
Q_{n+1} = Q_n + (1/n) · ( r_n − Q_n )
```

| Term | Meaning |
|---|---|
| `Q_n` | current estimate of the arm's average payout after n−1 pulls |
| `r_n` | the n-th observed reward |
| `(r_n − Q_n)` | the **surprise**: how much reality differed from the estimate |
| `1/n` | the step size — shrinking it makes the estimate the exact running mean |

Worked [V]: rewards 0.2, 0.8, 0.5 → estimates 0.2, then 0.2 + ½(0.8−0.2) = 0.5, then
0.5 + ⅓(0.5−0.5) = 0.5. And to keep exploring, act **ε-greedy**: with probability 1−ε take the
best-estimate arm, with probability ε take a random one.

Three ideas here survive unchanged into deep RL: *estimate = old estimate + step·surprise* (TD
learning, §7, is exactly this with a bootstrapped surprise); *behave greedily but inject
randomness* (exploration noise, §11.2); *estimates of untried actions are garbage* (why the
buffer must contain variety, §12.1). Everything the full problem adds is **time**: actions now
change the situation, so the reward for an action includes the value of where it leads — which
is Part I's remaining work to define.

## 3. The five objects, and where the boundary sits

- **Agent**: the thing being trained — here, *only* the actor network μ, a function from a
  vector to two numbers. Not the detector, not the safety supervisor, not the depth rule.
- **Environment**: everything else — both aircraft, the Wi-Fi and its 150–350 ms latency [V],
  the codec, **the YOLO detector with its jitter, misses and small-target weakness**
  [P Appendix A], the target's behaviour, the clamps.
- **Action** `a_t ∈ [−1,1]²` → `linear.z`, `angular.z`. The driver consumes exactly this range
  [V `node.py:1084-1092`]; **no rescaling exists between network and aircraft, and none may be
  added** [D, spec §4].
- **Reward** `r_t`: the task definition itself, §16.2.
- **Episode**: reset → run to a terminal event or a step budget → reset. Every step emits the
  tuple all learning consumes:

```
(s_t, a_t, r_t, s_{t+1}, d_t)
 what I saw · what I did · what it paid · what I saw next · did the TASK end
```

The boundary has one consequence beginners miss and experts lead with: **whatever sits outside
the agent must be present in training.** Detector dropout is environment dynamics. A policy
trained in a world without latency meets a *different MDP* in the air
([rl_foundations.md §2](rl_foundations.md)).

## 4. States, the Markov property, and our POMDP

### 4.1 The Markov property — the load-bearing assumption

Every guarantee in Part I's mathematics silently assumes [L1 ch. 3]:

> **p(s_{t+1} | s_t, a_t)** — the current state and action determine the distribution of the
> next state. Nothing earlier adds information.

A state that satisfies this "summarises the past". Positions-and-velocities of both aircraft
would qualify. Now test what we actually observe — a single frame's centring error `(ex, ey)`:

1. **Velocity is hidden.** Target drifting left vs drifting right: identical error, opposite
   correct actions. One frame cannot distinguish them.
2. **The pipe is hidden.** Commands from the last 150–350 ms have not yet changed any image we
   have seen. Two identical observations — one preceded by a hard yaw, one by stillness — have
   very different futures.

So the raw observation is **not** Markov: formally a **POMDP** (partially observed MDP).

### 4.2 The expert framing, then the engineering fix

The theoretically exact treatment of a POMDP is to maintain a **belief state** — a probability
distribution over true states given the whole history — and act on that. Exact belief tracking
is intractable here and unnecessary: the standard engineering approximation is to **feed a short
window of history and let the network learn its own sufficient statistic** — the same move as
frame-stacking in Atari DQN [L7], where four frames recover ball velocity.

What is hidden, and which observation entry recovers it:

| Hidden quantity | Recovered by | Why it works |
|---|---|---|
| target image-plane velocity | previous error `(ex, ey)_prev` | one-step finite difference ≈ velocity; at ≈ 46 px/step [V, §5.1] this signal is large |
| the latency pipe's contents | the last **k** actions actually sent | the policy can see what it already ordered and stop re-ordering it |
| detection validity | `visible`, `staleness` | a missing box must *look different* from a centred one |

**The k-rule** [D] (reconciling the spec's k = 1 baseline with the foundations' k = 4):
**k = ⌈Δ_max · f_ctrl⌉** — carry as many past actions as fit in the worst latency. At 0.35 s ×
10 Hz: **k = 4**, observation dimension **6 + 2k = 14**. Change the control rate and k changes
with it; record them as one pair.

### 4.3 The observation vector, exactly

| Index | Entry | Definition | Range |
|---|---|---|---|
| 0–1 | `ex, ey` | (box_x − 480)/480, (box_y − 360)/360, clipped | [−1, 1] |
| 2–3 | `ex_prev, ey_prev` | previous step's errors | [−1, 1] |
| 4 | `visible` | 1 if detected this step, else 0 | {0, 1} |
| 5 | `staleness` | time since last detection ÷ loss timeout, clipped | [0, 1] |
| 6 … 5+2k | `a_{t−1} … a_{t−k}` | the last k actions **sent** (post-clamp) | [−1, 1] each |

Corruption rules [D, spec §3]: on a miss, zero the errors and set `visible = 0` — never hold
the last error (that trains ghost-chasing). Store what was *sent*, not what the actor proposed.
The **faithful arm** uses none of this: raw pixels `(box_x, box_y)`, exactly as the original
code [C `drone_sim_env.py:24-33`] — kept as a reproduction arm, not a design.

## 5. Return and discount — with the derivations

### 5.1 The return

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …  =  Σ_{j≥0} γ^j · r_{t+j}
```

Why a *sum*: the agent must care about consequences, not moments — a command that centres the
box now but sets up an overshoot must lose to one that centres it slightly later and holds.
Why *discounted*: three derivations, each one line, each a different true reading of γ:

1. **Finiteness.** Constant reward r forever: `G = r·Σγ^j = r/(1−γ)` — the geometric series
   converges only because γ < 1. (Also the source of the useful constant `1/(1−γ)`.)
2. **Soft termination.** Suppose the task survives each step with probability γ and rewards are
   undiscounted. Expected total reward = `Σ_j r_j·P(alive at j) = Σ_j γ^j·r_j` — the discounted
   return. γ = 0.99 ⇔ a 1 %-per-step chance the chase ends.
3. **Horizon.** `γ^n = e^{n·ln γ} ≈ e^{−n(1−γ)}`, so a reward's weight falls to 1/e after
   `n ≈ 1/(1−γ)` steps. Verified exactly: 0.99¹⁰⁰ = 0.366 ≈ 1/e [V].

Grounded [V, recomputed]: at 10 Hz, the paper's own Table 3 candidates have horizons
γ = 0.9 → 10 steps = **1 s**; γ = 0.95 → 20 steps = **2 s**; γ = 0.99 → 100 steps = **10 s** —
only the last is the scale of a chase segment, and "let the target drift now, pay in three
seconds" is invisible to the other two. And the original's pairing of γ = 0.99 with 10-step
episodes leaves the deepest discount at γ¹⁰ = 0.904 — the discount nearly inert. **γ = 0.99
requires episodes of hundreds of steps**; ours are 200–500 [D, spec §8]. Also grounded, the
motion number used throughout: a 1 m/s crossing target at 2 m sweeps
`fx·(v/d)·Δt = 919.42·0.5·0.1 ≈ 46 px per control step` [V].

## 6. Value functions and the Bellman equation — the heart of the method

### 6.1 Definitions

```
V^π(s)    = E_π[ G_t | s_t = s ]             how good the situation is under π
Q^π(s, a) = E_π[ G_t | s_t = s, a_t = a ]    do a now, follow π afterwards
Q*(s, a)  = max_π Q^π(s, a)                  the best achievable
```

Q is the one to learn because **acting means comparing actions**, and Q is the object indexed
by action: given a good Q, the best action is simply its maximiser — no planning, no model. In
our words: *"target 200 px left and drifting; if I yaw left at 40 % stick now and then behave
as usual — how well does the rest of this chase go?"* One number.

### 6.2 Deriving the Bellman equation

Three lines, worth doing once by hand. Split the return after its first term:

```
G_t = r_t + γ·G_{t+1}                                        (definition, regrouped)
Q^π(s,a) = E[ r_t + γ·G_{t+1} | s,a ]                        (take expectations)
         = E[ r + γ·Q^π(s′, π(s′)) ]                          (inner expectation IS Q at s′)
```

and for the optimal function, the continuation is the best available:

```
Q*(s,a) = E[ r + γ·max_{a′} Q*(s′, a′) ]                      Bellman optimality equation
```

Read as a promise: *the value of now equals what you collected plus the discounted value of
where you landed.* Any function violating it is wrong somewhere — and the violation is
measurable from one stored tuple (§7).

### 6.3 The expert layer: the Bellman operator is a contraction

Define the operator `T` by `(TQ)(s,a) = E[r + γ·max_{a′} Q(s′,a′)]`. For any two value
functions, `‖TQ₁ − TQ₂‖∞ ≤ γ·‖Q₁ − Q₂‖∞` [L1] — applying T shrinks the worst-case disagreement
by the factor γ. By the Banach fixed-point theorem, T has **exactly one** fixed point, and it
is Q*; iterating T from anywhere converges. This is why the whole enterprise is sound *in
principle*, and it quantifies the speed: at γ = 0.99, each exact sweep shrinks error by ×0.99 —
**69 sweeps per halving** [V] — slow, but guaranteed. Two honest caveats an expert states
unprompted: the guarantee is for *exact* backups over *all* states (tabular); with sampling
and neural networks it becomes the fragile situation of §9. This is the difference between
"RL converges" (theorem, narrow) and "deep RL converges" (empirical, managed).

## 7. Learning from samples: Monte Carlo vs temporal difference

Two ways to estimate Q from experience, and the trade decides the whole method family.

**Monte Carlo (MC)**: run the episode to the end, use the *actual* return G_t as the label.
Unbiased — it is the real return — but high variance (every future random event is inside the
label) and unusable until the episode ends.

**Temporal difference (TD)**: use the Bellman promise as the label after **one** step:

```
y = r + γ·Q(s′, a′)                the bootstrapped target
δ = y − Q(s, a)                    the TD error — the measurable surprise
Q(s,a) ← Q(s,a) + α·δ              the update (the bandit rule of §2, with a bootstrapped surprise)
```

| Property | MC | TD |
|---|---|---|
| Bias | none | yes — the label contains the current (wrong) Q |
| Variance | high (whole future in the label) | low (one reward + one estimate) |
| Available | at episode end | **every step** |
| Behind DDPG? | no | **yes** — every critic update is a TD update |

Using an estimate to improve an estimate is **bootstrapping** — the method's power (every step
teaches; no waiting) and the origin of every instability §12 exists to manage. Worked with our
real magnitudes [V] ([rl_foundations.md §6](rl_foundations.md)): reward −50 at 200 px error,
critic believed Q(s,a) = −800, next situation rated −700:

```
y = −50 + 0.99·(−700) = −743        δ = −743 − (−800) = +57
```

The step went better than believed → raise Q(s,a). Learning is δ shrinking across all visited
transitions.

**Scaling corollary** [V]: raw ±100-scale rewards over a 300-step γ = 0.99 episode give returns
spanning ≈ `−150·Σγ^j ≈ −150·95 ≈ −1.4×10⁴`. Regressing ±10⁴ labels from ±1 inputs is a
conditioning disaster — hence observations normalised (§4.3) **and the reward ÷100** (§16.2), a
change §22 proves harmless.

## 8. On-policy vs off-policy — the distinction that buys us everything

Two policies exist during learning: the **behaviour policy** β that generated the data (our
actor + exploration noise — or the hand-coded controller, or an old checkpoint), and the
**target policy** π being evaluated/improved (the current actor). **On-policy** methods (SARSA,
PPO) require data from the current policy — every update discards the past. **Off-policy**
methods learn about π from β's data; Q-learning's target does this by construction: it asks
*"what would the current policy do at s′"* — `μ′(s′)` — not *"what did we happen to do next"*.

What off-policy buys this project, concretely [D]:

1. **Replay**: every real transition trains many updates (§12.1) — on hardware, the difference
   between hours and weeks of flight.
2. **Demonstrations**: the P-controller's flights are just more off-policy data — pre-fill the
   buffer, pretrain before the first learned flight (§17).
3. **Noise tolerance**: the exploration perturbations themselves are off-policy actions; the
   stored pair `(s, a)` with the *executed* a keeps the critic's regression honest.

The price is §9.

## 9. Function approximation and the deadly triad

Our state space is continuous — a table with one Q entry per state is impossible, so Q and μ
are neural networks that **generalise**: an update at one state moves nearby states too. That
is the point, and the danger: generalisation lets one bad label contaminate a neighbourhood.

Sutton & Barto name the combination that can make value learning *unstable, even divergent* —
**the deadly triad** [L1 ch. 11]:

1. **function approximation** (networks, not tables),
2. **bootstrapping** (TD labels containing the learned function),
3. **off-policy training** (data distribution ≠ target policy's distribution).

DDPG uses all three **by construction**. That is the expert's one-sentence explanation of why
Part II contains so much machinery: replay sampling, target networks, twin critics, delayed
updates are not decoration — they are triad management. Nothing in Part II is understandable
without this frame, and nothing in it is mysterious with it.

---

# PART II — THE ALGORITHMS (practitioner)

## 10. Two roads from values to actions — and why ours is the deterministic one

With **discrete** actions the policy is free: evaluate Q per action, take the argmax — that is
DQN [L7]. Our actions are **continuous**: `argmax_a Q(s,a)` over a continuum is itself an
optimisation problem, with no budget to run one inside every 100 ms control period. Two escape
roads exist.

**Road 1 — stochastic policy gradients (REINFORCE family)** [L1 ch. 13]: make the policy a
distribution π_θ(a|s), sample actions, and push up the log-probability of actions in proportion
to how well things went:

```
∇_θ J = E_{τ~π_θ}[ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]
```

| Term | Meaning |
|---|---|
| `τ ~ π_θ` | whole trajectories sampled by running the current policy — **on-policy** |
| `∇_θ log π_θ(a_t\|s_t)` | the direction in θ that makes the taken action more probable |
| `G_t` | the return that followed — the "how well things went" weight |

Sound, general — and high-variance (the whole noisy return multiplies the gradient) and
on-policy (every update wants fresh rollouts). PPO is this road, stabilised. It forfeits replay
and demonstrations — the two things §8 showed we need (§14 returns to this).

**Road 2 — the deterministic policy gradient** [L2]: make the policy a deterministic function
μ_θ(s) and differentiate the critic **through the action**:

```
J(θ)  = E_{s~D}[ Q_φ(s, μ_θ(s)) ]                                 maximise
∇_θ J = E_{s~D}[ ∇_a Q_φ(s,a)|_{a=μ_θ(s)} · ∇_θ μ_θ(s) ]          the DPG
```

| Term | Meaning |
|---|---|
| `s ~ D` | states from the replay buffer (the practical stand-in for the behaviour distribution ρ^β in [L2]'s theorem) |
| `∇_a Q_φ(s,a)\|_{a=μ_θ(s)}` | the critic's slope **in action space** at the actor's output — "a little more up-stick would have scored higher" |
| `∇_θ μ_θ(s)` | how the actor's output moves when its weights move |
| product | the chain rule: the critic's action-slope pushed back through the actor |

Why the expert prefers this road here: **no log-probability term and no return in the gradient**
— the noisy G_t is replaced by the smooth, learned surface Q, so variance is far lower; the
expectation is over buffer states, so it is **off-policy-compatible**; and Silver et al. proved
the form above is the true gradient of performance (the subtle part of the theorem is that the
term from the state distribution's dependence on θ can be dropped) [L2]. The actor is an
**amortised argmax**: optimisation at training time, one forward pass — microseconds — at
flight time.

Two structural consequences: the critic must take the action as an **input** (one network over
concat(s, a) — differentiability w.r.t. a is the whole mechanism), and the actor ends in `tanh`,
landing natively in the driver's [−1, 1] contract.

## 11. DDPG, complete [L3]

DDPG = the DPG (§10) + the two DQN stabilisers (replay, targets [L7]) + exploration noise.
Components and shapes for our problem:

| Component | Improved arms | Faithful arm [C] |
|---|---|---|
| Actor μ_θ | Linear(6+2k→256) ReLU → Linear(256→256) ReLU → Linear(256→2) tanh | Flatten → 16·16·16 ReLU → 2 tanh → ×60 |
| Critic Q_φ | concat(s,a) → 256·256 ReLU → 1 linear | concat(a,s) → 32·32·32 ReLU → 1 |
| Targets μ′, Q′ | same shapes, parameters θ′, φ′ | same |
| Buffer D | ring, 10⁵ | SequentialMemory 10⁵, window 1 |
| Noise | Gaussian, §11.2 | OU, §11.2 — including the dt finding |

### 11.1 Experience collection — the behaviour policy

```
a_t = clip( μ_θ(s_t) + ε_t ,  −1, +1 )          ε_t: training only; evaluation and flight run ε ≡ 0
```

Without ε a deterministic policy never discovers better actions — the buffer fills with one
behaviour's consequences (§1's exploration problem, §2's ε-greedy, continuous edition). Modern
collection rules [L5]: **warm-start** with uniformly random actions for the first `start_steps`
(ours 1 000 [D]) so the buffer opens with coverage; **always evaluate with noise off**.

### 11.2 The noise process — exact equations, statistics, and a verified finding

**Ornstein–Uhlenbeck**, chosen by [L3] "for exploration efficiency in physical control problems
with inertia" (its samples are correlated in time, so the perturbation *pushes* for a while
instead of dithering):

```
x_{t+1} = x_t + θ_OU·(μ_OU − x_t)·Δt_OU + σ_OU·√Δt_OU · N(0, I)        ε_t = x_t
```

| Term | Meaning | [L3] | Chase code [C] |
|---|---|---|---|
| `θ_OU` | mean-reversion rate — pull toward μ_OU; autocorrelation time ≈ 1/θ_OU | 0.15 | 0.15 |
| `μ_OU` | long-run mean | 0 | 0 |
| `σ_OU` | volatility per unit time | 0.2 | 0.3 |
| `Δt_OU` | integration step | 1 (implicit) | **10⁻² — library default, never set** |

Expert-level statistics of the process [V, recomputed]: its stationary distribution is Gaussian
with std `σ_OU/√(2θ_OU)` — for the chase parameters, **0.548** regardless of Δt_OU — but the
time to *reach* stationarity is ~`1/(θ_OU·Δt_OU)` steps. With Δt_OU = 10⁻² that is **≈ 667
steps**, against **10-step episodes**.

> **Verified finding (2026-09-13).** The chase code builds keras-rl's
> `OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3)` with `dt` unset
> [C `rl_drone.py:55`]; the library default is `dt = 1e-2`, per-step increment `σ√dt·N(0,1)`,
> and episode resets draw `x ~ N(0, σ)` (keras-rl `rl/random.py`, fetched and quoted). Net
> effect [V]: fresh noise of std `0.3·√0.01 = 0.03` per step, growing to only ≈ 0.09 by step
> 10, around a per-episode offset of std 0.3 — all added **after** the ×60 output scaling, i.e.
> **0.05–0.5 % of the action half-range**. Table 2's "σ = 0.3" never acted at face value; the
> original explored through uniform random resets over the frame, not through action noise.
> Reproduce the faithful arm as shipped — "fixing" its noise makes it a different arm.

**Gaussian** — the improved arms [D]: `ε ~ N(0, σ²I)`, σ = 0.3 decayed to ~0.05, in normalised
units. Support: "uncorrelated, mean-zero Gaussian noise works perfectly well" [L5]; the TD3
author uses N(0, 0.1) [L6]. σ is the first knob to lower if actions pile at ±1 (§23).

### 11.3 The critic target — and the (1−d) rule

```
y = r + γ · (1 − d) · Q′_{φ′}( s′ , μ′_{θ′}(s′) )
```

| Term | Meaning |
|---|---|
| `y` | the regression label: one step of collected truth + discounted bootstrap |
| `(1 − d)` | terminal ⇒ the future is truncated; the label is just r |
| `Q′(s′, μ′(s′))` | the **target** critic's rating of what the **target** actor would do next — both slow copies, so the label barely moves between updates (§11.5) |

**The strongest warning in this guide — timeouts are not terminal.** `d = 1` only when the task
truly ended: target left the frame (with its penalty), or a safety abort. A step-budget timeout
is an accounting boundary — the chase was still worth its future value when the clock expired —
so the stored flag must be 0 and the target must bootstrap. Storing timeouts as terminal
teaches the critic that chases die at an arbitrary wall, silently biasing every value downward.
Gym's `terminated`/`truncated` split exists for exactly this; store `terminated` only.

### 11.4 The two updates

```
Critic:  L(φ) = (1/N) Σ_i ( y_i − Q_φ(s_i, a_i) )²          minimise — Adam, lr 3e-4 [D]
Actor:   L(θ) = −(1/N) Σ_i Q_φ( s_i , μ_θ(s_i) )            minimise (−Q ⇒ ascend Q) — Adam, lr 3e-4 [D]
```

Critic: regression on the **stored** pair — the action actually executed, noise included; that
is what makes the update off-policy-correct. (This is the chase paper's Eq. (4) with N made
explicit [P §2.2.2].) Actor: the DPG of §10 by autodiff — gradient flows from Q's output,
through Q's **action input**, into θ, with φ frozen for this step; literally
`actor_loss = −Q1(state, actor(state)).mean()` [L6]. It is also the equation the chase paper
never prints — it exists in their system only because keras-rl implements it
[C, [rl_block_diagram.md §4](rl_block_diagram.md)]. If divergence is ever observed, two damping
options with precedent — critic L2 weight decay 10⁻² [L3], gradient clipnorm 1 [C] — and start
with neither [D].

### 11.5 Target networks and soft updates

```
φ′ ← τ·φ + (1−τ)·φ′        θ′ ← τ·θ + (1−τ)·θ′        after every gradient step
```

The label y contains the network being trained; without slow copies, every gradient step moves
the target of the next — a fixed-point iteration whose target refuses to sit still, the
canonical divergence driver [L3][L7]. τ makes the label source an exponential moving average
with time constant ≈ 1/τ gradient steps: τ = 0.005 → ~200; τ = 0.001 ([L3] and [C]) → ~1 000
[V]. Two cautions with verified precedent: count **gradient steps** (the reference codebase
counts training *calls* and bursts 500 target updates at once — mechanism silently disabled
[drl_ros2_reference_analysis.md §2.2](drl_ros2_reference_analysis.md)); initialise targets as
exact copies.

### 11.6 Initialisation and saturation — small details with mechanisms

- Final layers of actor and critic: uniform **±3×10⁻³** [L3 §7] — the initial policy is
  near-zero-action and initial values near zero, instead of random confident nonsense.
- Why saturation matters, with numbers [V]: tanh's gradient is `1 − tanh²`. At output 0.9 the
  gradient is 0.19; at 0.96, **0.078**; at 0.99, 0.02. An actor pinned at ±1 is an actor that
  has almost stopped learning — which is why the near-zero init, the reward scaling (§7), and
  the σ-decay all exist, and why the action histogram is a first-class diagnostic (§23).
- [L3] used batch normalisation (cross-task unit mismatch); **we don't need it** — inputs are
  [−1,1] by construction, and [L6] uses none. Actions enter our critic at the input concat
  (today's standard; also what the chase code does [C `rl_drone.py:40`]).

### 11.7 DDPG assembled — annotated

```
initialise μ_θ, Q_φ;  copy targets θ′ ← θ, φ′ ← φ                        (§11.5–11.6)
pre-fill D from hand-coded-controller flights; recompute rewards on load  (§17)

for env_step = 1 … total_steps:                 # 10 Hz sim time; the world never pauses
    s = observation (assembled per §4.3)
    a = uniform random                if env_step < start_steps           (§11.1)
      = clip(μ_θ(s) + ε, −1, 1)      otherwise                            (§11.1–11.2)
    execute a; environment advances one control period                    (§16.1)
    observe r, s′, terminated, truncated
    store (s, a, r, s′, d = terminated)          # truncated ⇒ d = 0 — bootstrap! (§11.3)
    if terminated or truncated: reset

    if env_step ≥ update_after:                  # one gradient step per env step
        sample minibatch B of N = 64 from D                               (§12.1)
        y  = r + γ(1−d)·Q′(s′, μ′(s′))                                    (§11.3)
        φ  ← Adam step on ∇_φ (1/N)Σ (y − Q_φ(s,a))²                      (§11.4)
        θ  ← Adam step on ∇_θ −(1/N)Σ Q_φ(s, μ_θ(s))                      (§11.4)
        φ′ ← τφ + (1−τ)φ′ ;  θ′ ← τθ + (1−τ)θ′                            (§11.5)

every eval_interval: E episodes, ε = 0, fixed scenarios — log return,
time-in-view, avg_Q vs realised return, action histogram                  (§23)
```

### 11.8 Hyperparameters — four verified columns and ours

Sources: [L3] the DDPG paper's §7 (fetched); [C] the chase repo (cloned, line-cited); [L6] the
TD3 author's code (fetched); Spinning Up defaults [L5] noted where different. **Ours** [D] =
the improved arms; the faithful arm uses the [C] column verbatim.

| Knob | DDPG paper [L3] | Chase code [C] | TD3 author [L6] | Ours [D] | Failure symptom |
|---|---|---|---|---|---|
| Hidden layers | 400, 300 ReLU | 16³ / 32³ ReLU | 256, 256 ReLU | 256, 256 ReLU | too small → plateaus; too big → noisier |
| Actor output | tanh | tanh ×60 | tanh ×max_action | tanh ([−1,1] native) | — |
| Final-layer init | ±3e-3 | keras default ±0.05 | PyTorch default | ±3e-3 | big init → early saturation (§11.6) |
| Actor / critic lr | 1e-4 / 1e-3 | 1e-3 both, clipnorm 1 | 3e-4 / 3e-4 | 3e-4 / 3e-4 | high → actions slam ±1 / avg_Q runs away |
| Critic L2 | 1e-2 | — | — | none (option) | — |
| γ · τ | 0.99 · 0.001 | 0.99 · 0.001 | 0.99 · 0.005 | 0.99 · 0.005 | τ high → tail-chasing; low → slow |
| Buffer · batch | 1e6 · 64 | 1e5 · 32 (keras-rl default) | 1e6 · 256 (runner; 100 [L5]) | 1e5 · 64 | — |
| Warm-up · update_after | — | 100 · 100 | 25e3 (10e3 · 1e3 [L5]) | 1 000 · 1 000 | early sampling → overfitting noise |
| Update cadence | per step | per step | per step | 1 gradient step / env step | count gradient steps for τ |
| Noise | OU θ.15 σ.2 | OU θ.15 σ.3 **dt 1e-2** (≈0.03/step effective) | Gaussian 0.1 | Gaussian 0.3→0.05 | §11.2 |
| Episode length | task-dep. | 10 | 1 000 | 200–500 | must exceed γ's horizon scale (§5) |
| Total steps · seeds | ~1e6 · — | 1e5 · seed 123 | 1e6 · 10 | ~2e5/arm · ≥5 | single-seed numbers are anecdotes (§24) |

## 12. Why the naive version diverges — four failures, four mechanisms

Take §7 and §10 literally — one critic, one actor, learn from the latest transition — and
training diverges, notoriously. Each piece of machinery answers one failure; every knob in
§11.8 belongs to one of these stories. (Narrative version:
[rl_foundations.md §9](rl_foundations.md); here, the mechanisms.)

**12.1 Correlated data → replay buffer.** Consecutive transitions are near-duplicates of one
corner of state space. SGD assumes i.i.d. samples; feed it a correlated stream and the network
overfits the present neighbourhood while the Q-surface elsewhere deforms (catastrophic
forgetting is literal). Uniform minibatches from a large buffer restore approximate
independence, reuse every real sample many times, and — because the method is off-policy (§8) —
legitimately mix old policies, noise, and demonstrations.

**12.2 The moving target → target networks.** The regression `Q ← y(Q)` is a fixed-point
iteration performed by gradient descent while the fixed point moves with every step. Freezing
the label source (targets, τ-averaged) turns it into ordinary supervised regression against a
slowly drifting label — the tabular contraction's (§6.3) function-approximation surrogate.

**12.3 Determinism → injected noise.** §11.2. On hardware, noise passes through the safety
supervisor like every other command; exploration is overwhelmingly a *simulation* activity [D].

**12.4 Max-bias → (eventually) TD3.** The target maximises over actions — explicitly in
Q-learning, implicitly in DDPG (the actor is trained to climb Q, so `Q′(s′, μ′(s′))` is an
approximate max). A max of noisy estimates is biased upward, and the bias is computable [V]:

```
max(X₁, X₂) = (X₁+X₂)/2 + |X₁−X₂|/2
X₁, X₂ ~ N(0, σ²) independent  ⇒  X₁−X₂ ~ N(0, 2σ²),  E|N(0, s²)| = s·√(2/π)
⇒  E[max] = ½·σ√2·√(2/π) = σ/√π ≈ 0.564·σ        (simulated: 0.566 [V])
```

Two *unbiased* estimators, and the max manufactures +0.56σ of optimism from pure noise. The
optimism enters the label, the label trains the critic, the actor climbs the inflated region,
the buffer fills with its consequences — a feedback loop. In curves: `avg_Q` racing above the
returns actually achieved. That loop is DDPG's signature failure and TD3's entire reason to
exist.

## 13. TD3 — three surgical repairs [L4]

Spinning Up states the field's experience plainly: DDPG is "frequently brittle … the learned
Q-function begins to dramatically overestimate Q-values, which then leads to the policy
breaking" [L5]. TD3 ("Twin Delayed DDPG") changes exactly three things; everything else in §11
stands.

### 13.1 Trick ① — clipped double-Q

Two critics, independent inits; every label uses the **minimum** of their target copies:

```
y = r + γ·(1−d) · min_{i=1,2} Q′_{φ′_i}( s′ , a′(s′) )
L(φ₁, φ₂) = (1/N) Σ [ (y − Q_{φ1}(s,a))² + (y − Q_{φ2}(s,a))² ]        (shared label; summed MSE [L6])
```

Why min works — the order-statistics view: §12.4 showed E[max] = +0.564σ; by symmetry
E[min] = −0.564σ. The min of two noisy estimates is *pessimistic* by about what the max was
optimistic — the bias flipped to the safe side, where it damps rather than feeds the loop. (A
pessimistic critic undervalues some good actions; that costs some learning speed and buys
stability — a trade [L4] shows empirically is worth it.) The actor trains against Q₁ only [L6].

### 13.2 Trick ② — target-policy smoothing

```
a′(s′) = clip( μ′_{θ′}(s′) + clip(ε̃, −c, +c) ,  −1, +1 ),      ε̃ ~ N(0, σ̃²·I)
```

| Term | Meaning | Value [L4][L6] |
|---|---|---|
| `ε̃` | fresh noise **inside the label only** — never executed | σ̃ = 0.2 |
| `clip(ε̃, −c, c)` | bounds the perturbation so the label stays near the proposal | c = 0.5 |
| outer clip | actions stay legal | [−1, 1] |

Mechanism: the label becomes the value of a small *neighbourhood* of the target action, so a
narrow spurious spike in a critic cannot be harvested as a label — a regulariser encoding the
prior that similar actions have similar value. Unit caution [D]: σ̃ and c are in normalised
action units — correct for [−1,1] as-is; any other range must scale them ([L6] multiplies by
`max_action`).

### 13.3 Trick ③ — delayed policy and target updates

```
policy_delay = 2:   one actor step, and one soft-update of ALL targets, per two critic steps
                    (if total_it % policy_freq == 0 — [L6])
```

The two-timescale intuition: let the fast learner (critic) settle between moves of the slow
learner (actor), so the actor climbs a surface that has stopped shifting underfoot; an actor
updated against a half-converged critic amplifies its errors. Coupling to note [V]: with
per-step critic updates, delay 2 doubles the targets' effective time constant
(τ = 0.005 → ≈ 400 critic steps).

### 13.4 The diff against §11.7 — only the changed lines

```
initialise Q_{φ1}, Q_{φ2} (+ their targets)                       # was: one critic
…
    ε̃    = clip(N(0, σ̃), −c, c)                                   # trick ②
    a′   = clip(μ′(s′) + ε̃, −1, 1)
    y    = r + γ(1−d)·min( Q′₁(s′,a′), Q′₂(s′,a′) )               # trick ①
    update BOTH critics toward y (summed MSE)
    if gradient_step % policy_delay == 0:                          # trick ③
        θ  ← Adam step on −(1/N)Σ Q_{φ1}(s, μ_θ(s))
        soft-update θ′, φ′₁, φ′₂                                   # targets move only here
```

### 13.5 One trainer, three arms [D]

DDPG is a configuration of the TD3 trainer, not a second codebase (the structuring decision of
[drl_ros2_reference_analysis.md §5](drl_ros2_reference_analysis.md)):

| Flag | **T** — TD3 (primary) | **R** — repaired DDPG | **F** — faithful DDPG |
|---|---|---|---|
| critics | 2, min target | 1 | 1 |
| target smoothing σ̃ / c | 0.2 / 0.5 | off | off |
| policy_delay | 2 | 1 | 1 |
| exploration | Gaussian 0.3→0.05 | Gaussian 0.3→0.05 | OU θ.15 σ.3 dt 1e-2, post-scale [C] |
| observation | §4.3, k = 4 | §4.3, k = 4 | raw pixels (box_x, box_y) [C] |
| reward | §16.2 repaired, scaled | §16.2 repaired, scaled | original threshold-100, unscaled [C] |
| everything else | §11.8 "Ours" | §11.8 "Ours" | §11.8 [C] verbatim (16³/32³, τ 1e-3, batch 32, seed 123) |

Arm F carries the reproduction claim; **R vs T isolates the algorithm**; **F vs R isolates the
reward/observation repairs** — each comparison changes one thing, which is what makes the
ablation publishable.

## 14. The algorithm landscape — and the defense of the choice

| Method | Action space | On/off-policy | Policy | Fit here |
|---|---|---|---|---|
| DQN [L7] | discrete only | off | implicit argmax | wrong action space (would need to discretise two continuous axes) |
| **DDPG** [L3] | continuous | off | deterministic | the paper's method — required as the reproduction arm; brittle alone (§12.4) |
| **TD3** [L4] | continuous | off | deterministic | **primary**: fixes DDPG's failure at near-zero added complexity, same skeleton |
| SAC | continuous | off | stochastic (entropy-regularised) | respectable; adds a stochastic head + temperature machinery a 2-D action doesn't need; optional third arm (the reference repo's clean SAC critic is the template) |
| PPO | both | **on** | stochastic | wrong shape: no replay, no demonstration pre-fill, pays for every update with fresh rollouts — exactly wrong where real samples cost battery and crash risk |

The chase paper's own comparison table contains the warning that plain DDPG is unsettled for
this task class — its ref [29] found DQN outperforming DDPG [P Table 1]. Our answer is not to
argue the point but to instrument it: three arms, ≥ 5 seeds each, P-controller baseline, and
the flight checkpoint is the best arm-T seed unless arm R beats it across seeds (§28.4).

---

# PART III — THE TRAINING SYSTEM (implementer)

The algorithm is ~80 lines; the system around it decides whether it learns anything real.

## 15. Layout — RL core with no ROS in it

Two-layer split ([drl_ros2_reference_analysis.md §3.7](drl_ros2_reference_analysis.md)): the
algorithm modules import **no ROS symbols**; the environment implements plain Gym; ROS appears
only in the deployment shell (§21). Package homes per
[implementation_plan.md §2](implementation_plan.md):

```
chase_gym/    env.py             # ChaseEnv(gym.Env): §16
              target_motion.py   # the five scenario generators
              latency.py         # the delay queue
chase_train/  networks.py        # actor, twin critic (§11, §13)
              buffer.py          # ring buffer + demonstration loader (§17)
              td3.py             # ONE class, the flags of §13.5
              train.py           # the loop of §11.7, config-driven, seeds by CLI
chase_eval/   eval.py            # fixed scenarios, noise off, per-seed statistics
```

And one config block, versioned with every run [D]:

```yaml
algo: td3                    # td3 | ddpg_repaired | ddpg_faithful
obs:   {k: 4, normalize: true}          # faithful arm: raw_pixels
nets:  {hidden: [256, 256], final_init: 3.0e-3}
train: {gamma: 0.99, tau: 0.005, batch: 64, buffer: 100000,
        actor_lr: 3.0e-4, critic_lr: 3.0e-4,
        start_steps: 1000, update_after: 1000,
        policy_delay: 2, target_noise: 0.2, noise_clip: 0.5,
        expl_noise: {type: gaussian, sigma0: 0.3, sigma_min: 0.05}}
env:   {dt: 0.1, episode_steps: 300, latency_ms: [150, 350],
        t_lag_jitter: 0.30, scenario: curriculum}
eval:  {every: 5000, episodes_per_scenario: 10, noise: off}
run:   {seeds: [0,1,2,3,4], total_steps: 200000}
```

## 16. The environment — the most important build item

> **UPDATE (2026-09-15).** This section specifies Tier A of what is now a three-tier simulation
> architecture — Gazebo (physics truth, lockstep fine-tune, ROS rehearsal) and Unreal/Colosseum
> (YOLO dataset factory, vision-in-the-loop evaluation) are specified component-by-component in
> [sim_training_architecture.md](sim_training_architecture.md), including the FOLLOW→INTERCEPT
> task extension. Gradient training at scale stays here, in Tier A.

The original's environment turned out to be a static point-mass with **no target motion at
all** [C, [rl_block_diagram.md §3](rl_block_diagram.md)] — ours is not a refinement of it; it
is the first simulation of the chase. Internal (hidden) state: box centre `(u, v)` and width
`w` in px, follower velocities, generator state. No 6-DoF simulator: the policy only ever sees
the projection, so simulating the projection is sufficient — and validatable.

### 16.1 One step, `Δt = 0.1 s` — the deployment constant

```
def step(a):                                   # a ∈ [−1,1]², the deployment convention
  1  target.advance(Δt)                        # scenario generator (§16.3)
  2  v ← v + (Δt/T_lag)·(a_mapped − v)         # follower FIRST-ORDER LAG
  3  (u, v_px, w) ← project(pose, fx=919.42)   # pinhole; Tello true width 0.098 m (0.18 m variant)
  4  queue.push((u, v_px, w), t)               # TRUE geometry enters the delay line
  5  meas = queue.sample(t − Δ_t)              # observation is Δ old
  6  meas = jitter(meas) or DROPPED            # detector noise
  7  s′  = assemble_observation(meas, history) # §4.3 — the SAME MODULE as deployment
  8  r   = reward(TRUE current geometry)       # §16.2 — reward uses truth, obs uses the pipe
  9  terminated = target outside frame (truth) # loss penalty applied here
 10  truncated  = step budget reached          # NOT terminal — §11.3
```

| Line | Parameter | Value / source |
|---|---|---|
| 2 | `T_lag` | **measured** from a Phase-1 step-response flight [implementation_plan.md]; randomised ±30 % per episode [D] |
| 5 | `Δ_t` | per-episode base from the **measured 150–350 ms** distribution + per-frame jitter of tens of ms [V] — at 10 Hz, a delay of 1.5–3.5 control periods, exactly what k = 4 compensates |
| 6 | jitter / dropout | box jitter ~N(0, σ_px) from Phase-0 detector statistics; dropout matched to measured detection rates, rising as the box shrinks [P Appendix A] |
| 10 | budget | 200–500 steps [D] |

**Line 8 is a design principle, not a convenience** [D]: the reward is computed from the true,
current geometry while the observation comes through the delay-and-noise pipe. Reward is
training-time information — it never needs to be computable in flight — and defining it on the
truth makes it judge what we actually care about (where the target *is*), while latency is
genuinely *felt*: acting on old data is punished by the true state of the world. §25 gives this
its theoretical footing.

### 16.2 The reward, concretely [D]

```
r_track  = (100 − dist)/100               if dist ≤ 100        ∈ (0, 1]
         = −0.25·(dist − 100)/100         if dist  > 100        ∈ (−1.25, 0]
r_smooth = −w_smooth · ‖a_t − a_{t−1}‖²                          w_smooth = 0.05 initial
r_loss   = −w_loss   on the terminal target-lost step            w_loss = 5
r = r_track + r_smooth (+ r_loss when terminal)
```

Anchors [V]: r_track(0) = 1.0 · r_track(50) = 0.5 · r_track(100) = 0 · r_track(300) = −0.5 ·
r_track(600) = −1.25. Threshold **100** px is the original *code's* value [C], with the
one-term repair that removes its 25-point cliff
([rl_block_diagram.md §5](rl_block_diagram.md)); ÷100 is the §7 scaling. The smoothness term is
the one the paper considered and dropped [P §2.2.3] — under latency, nothing else discourages
stick jitter. The loss penalty makes losing the target unambiguously the worst event (it *is*
the evaluation metric) and defines the reward on the no-detection path. The theory of which of
these changes are "free" and which change the task is §22 — read it before defending the
reward. The faithful arm ignores all of this and runs the original discontinuous reward,
unscaled [C].

### 16.3 Target motion, reset, validation

**Generators**, one per evaluation scenario family [P Table 4], randomised amplitude/frequency/
speed, curriculum-capped speed (equal airframes — the target must not outrun the follower by
construction [drl_ros2_reference_analysis.md §6.1](drl_ros2_reference_analysis.md)): `static`,
`constant_velocity`, `vertical_oscillation`, `horizontal_oscillation`, `aggressive`.
**Reset** (= ρ₀): target uniform in frame at a standoff drawn from the operating band; follower
velocities zero; latency base and T_lag drawn; generator seeded. Uniform placement is one thing
the original got right — keep it [C]. **Validation gate** [D, spec §9]: replay a recorded real
flight's command log through the environment and compare the simulated box track to the
recorded YOLO track — median error within detector jitter, latency histograms matching. **An
unvalidated simulator invalidates everything downstream; this gate is not skippable.**

## 17. Replay buffer and demonstrations

Ring buffer, 10⁵, storing `(s, a, r, s′, d_terminated)` plus the real per-step Δt
(diagnostics). Before the first learned step, **pre-fill from the hand-coded controller** (the
Phase-1 P-controller + metric standoff, in sim and from recorded flights) and **recompute
rewards on load**, so the reward function can be revised without re-flying — the
demonstration-bootstrap pattern with its storage changed to `.npz`
([drl_ros2_reference_analysis.md §3.1](drl_ros2_reference_analysis.md)). Legitimate precisely
because the method is off-policy (§8).

## 18. Networks

```
Actor:      Linear(6+2k → 256) ReLU → Linear(256 → 256) ReLU → Linear(256 → 2) tanh
Critic ×2:  Linear(6+2k+2 → 256) ReLU → Linear(256 → 256) ReLU → Linear(256 → 1)
```

Final layers ±3e-3 [L3]; no normalisation layers (§11.6); ~90 k parameters — CPU-trainable by a
wide margin (the original trained a far smaller net in ~4 h on an i5 [P Appendix A]).

## 19. Trainer rules and the checkpoint contract

- One gradient step per env step; **count gradient steps** for τ and policy_delay (§11.5).
- Eval every 5 000 env steps: ≥ 10 episodes per scenario family, ε = 0, fixed eval seeds; log
  the §23 dashboard.
- ≥ 5 seeds per arm; the 3-arm × 5-seed matrix scripted, not manual; never tune on eval seeds.
- **Checkpoint contract** [D] — sim-to-real is auditable or it is folklore: weights (actor +
  critics) **plus** observation layout + k + normalisation constants, action map (index → axis,
  sign), control rate, reward version, env config hash, git SHA, seed, env-step count, library
  versions. The inference node refuses a checkpoint whose observation spec it cannot reproduce.

## 20. Curriculum and budget

Stage the generator — static → constant velocity → oscillations → aggressive — advancing when
eval time-in-view > 90 % on the current stage across seeds [D]. Budget, stated as an estimate
to revise against curves: ~2×10⁵ env steps per arm per seed; point-kinematics simulation is
minutes-cheap, wall-clock is the gradient steps — hours per run on CPU, parallel across seeds.

## 21. From checkpoint to the big pipeline — inference

What deploys is the **actor alone** — critics, targets, buffer, noise are training-only.
Contract (owner: [implementation_plan.md §2–3](implementation_plan.md)):

- `chase_policy` loads the bundle, verifies the observation spec, runs one forward pass per
  tick — **inference only, no noise, no learning in flight** (the original's deployment is the
  existence proof: `training` never enabled, `backward()` never called [C]).
- **Publish at ≥ 10 Hz with hold-last-command** under the driver's 0.35 s dead-man [V] — never
  the original's pulse-and-zero pattern (≈ 17 % command duty cycle
  [C, [rl_block_diagram.md §6](rl_block_diagram.md)]).
- Axis ownership exclusive (policy → `linear.z` + `angular.z`; standoff rule → `linear.x`;
  `linear.y` unused), enforced by the mixer; every output passes the safety supervisor.
- Flight ladder: sim → stationary target → slow target → full scenarios; manual override live.
- The observation assembly in flight is **the same module** as §16.1 line 7 — one
  implementation imported by both, so train/deploy skew is structurally impossible [D].

---

# PART IV — MASTERY (expert)

## 22. Reward design theory — which changes are free, and which change the task [L9]

Ng, Harada & Russell (ICML 1999) answered precisely the question our reward edits raise: *which
transformations of the reward function leave the optimal policy unchanged?* Two results matter
here:

1. **Positive linear transformations** (`r → a·r + b`, a > 0) preserve the optimal policy —
   the utility-theory result. **Our ÷100 scaling is therefore provably free** (§7, §16.2).
2. **Potential-based shaping**: adding `F(s, s′) = γ·Φ(s′) − Φ(s)` for *any* state function Φ
   preserves the optimal policy — and this form is *necessary* for guaranteed invariance. The
   telescoping intuition: along any trajectory the added terms collapse to
   `γ^T·Φ(s_T) − Φ(s₀)`, a policy-independent constant (up to termination handling), so no
   ranking of policies changes.

Now the expert-grade honesty about our own edits, in exactly these terms [D]:

| Edit | Form | Verdict |
|---|---|---|
| ÷100 scaling | positive linear | **free** — provably no change to the optimal policy |
| Continuity repair (`−0.25·(dist−100)` outside) | a change to r(s), not potential-based | **changes the task in principle** — it re-ranks behaviours near the threshold. That is *why* arm F vs arm R exists: "what did the discontinuity cost" is measured, not assumed |
| Smoothness term `−w‖a_t − a_{t−1}‖²` | depends on actions, not expressible as γΦ(s′) − Φ(s) | **changes the task by design** — we *want* the optimum moved toward smooth control; the weight is reported with results |
| Loss penalty (terminal) | task definition, not shaping | defines the failure event the evaluation measures; also removes an undefined reward case |

Being able to say "this change is provably policy-invariant, that one is deliberate task
design, and here is the arm that measures the difference" is the difference between tuning and
science — and it is a two-sentence answer in the exam (§26, Q11).

## 23. Reading training curves like an expert

Each signal, what it is mechanically, and what its pathologies mean. (Companion table:
[rl_foundations.md §12](rl_foundations.md).)

| Signal | What it is | Healthy | Pathology → diagnosis |
|---|---|---|---|
| Eval return (ε = 0) | the objective itself | rising, then flat | the only signal that *is* the goal; everything else is instrumentation |
| Time-in-view % | the flight metric [P §3] | rising with return | diverges from return → reward is optimising something else — reread §16.2 weights |
| avg_Q vs realised return | critic's belief vs reality | tracking | avg_Q ≫ realised → overestimation loop (§12.4); confirm arm T immune, else lower critic lr |
| max_Q | worst-case belief | bounded | secular growth = divergence beginning — hours before the return curve shows it |
| Critic loss | TD regression error | modest, non-zero | ≈ 0 with no progress → collapsed critic or one-behaviour buffer. **Not a success metric** — the label moves by design (§11.5) |
| Q₁ − Q₂ gap (TD3) | twin disagreement | small, stable | growing without bound → one critic diverging; check smoothing on, lower critic lr |
| Action histogram | what the policy actually outputs | using the range | piled at ±1 → saturation: tanh gradient 0.078 at output 0.96 [V] — learning stalls; lower actor lr / check reward scale / σ decay |
| TD-δ distribution | per-sample surprise | spread shrinking | persistent heavy tails → a discontinuity (reward) or mislabelled terminals (§11.3) |
| Episode length (train) | survival | rising toward budget | stuck short → target lost early: curriculum too fast, or loss penalty overwhelming r_track |

Debugging order, always: **environment → reward → done flags → hyperparameters.** Most "RL
doesn't work" is one of the first three.

## 24. The statistics of RL claims [L8]

Henderson et al. (AAAI 2018) demonstrated that in deep RL, "non-determinism in standard
benchmark environments, combined with variance intrinsic to the methods, can make reported
results tough to interpret" — identical code and hyperparameters, different seeds, materially
different curves; DDPG-family methods are among the worst offenders. The protocol this project
therefore commits to [D]:

- **≥ 5 seeds per arm**, all reported — mean **and** spread; a single-seed win is an anecdote.
- **Fixed evaluation scenarios and seeds**, disjoint from anything tuned on.
- **Compare distributions, not best runs**: arm T beats the P-controller only if the margin
  exceeds the seed spread (§28.4).
- **Report everything that moved**: the four-column table (§11.8) plus the config hash make
  every run reconstructible; the chase paper's own three-way-inconsistent training length
  [P Table 2 vs 3 vs §5] is the cautionary example of what happens otherwise.

The one-line expert stance: *in this field, error bars are not politeness — the seed is a
hyperparameter of the result.*

## 25. Sim-to-real — why this transfer is credible

The **reality gap**: a policy is optimal for the MDP it trained in; deploy it in a different
MDP and optimality claims evaporate. Three commitments close the gap here:

1. **Model what is outside the agent** (§3): the sim injects the *measured* latency
   distribution, measured lag constant, and detector noise matched to Phase-0 statistics —
   the known dominant error sources of this platform, which the source paper named but never
   measured [P Appendix A; V].
2. **Domain randomisation** = training against a *distribution over MDPs* (latency draw, lag
   constant, target size, detector dropout per episode). The policy that survives a
   neighbourhood of worlds treats the real world as one more sample from it — robustness by
   construction rather than by hope.
3. **Validation before belief** (§16.3): the replay-comparison gate bounds the sim's error in
   the *observation channel* with real data. Reward-from-truth (§16.1 line 8) is sound
   precisely because reward is a training-time construct: nothing at deployment ever needs it.

And the honest residual an expert names before being asked: the sim's *follower dynamics* are
first-order-lag kinematics, not aerodynamics; the flight ladder (§21) exists because the last
gap is closed by staged exposure, not by simulation fidelity claims.

## 26. The oral exam — twenty questions, expert answers

*Foundations*

1. **Why is this reinforcement learning and not supervised learning?**
   No oracle provides the correct stick command per frame; only outcomes are measurable
   (centring, time-in-view). RL is learning a controller from measurable outcomes under
   delayed credit — the guilty command is seconds old by the time the target is lost (§1).
2. **What exactly is your agent?**
   Only the actor network: 14 numbers in, 2 numbers out at 10 Hz. Detector, depth rule, safety
   supervisor, both aircraft — all environment (§3). That boundary is why detector noise and
   latency must exist in training.
3. **Why is your problem a POMDP, and what did you do about it?**
   A single frame hides target velocity and the 150–350 ms of commands still in flight — two
   states demanding opposite actions can look identical. Exact belief tracking being
   unnecessary, we append history: previous error and the last k = ⌈Δ_max·f_ctrl⌉ = 4 actions,
   the same sufficient-statistic move as Atari frame-stacking (§4).
4. **What does γ = 0.99 mean, physically, in your system?**
   A ~100-step ≈ 10-second attention horizon at 10 Hz (weight falls to 1/e at 1/(1−γ) steps;
   0.99¹⁰⁰ = 0.366). Equivalently a 1 %/step task-termination prior. It forces episodes of
   hundreds of steps — the original's 10-step episodes left γ nearly inert (γ¹⁰ = 0.904) (§5).
5. **What is Q(s,a), in one sentence, for your system?**
   The expected discounted future payment for issuing this stick pair in this situation and
   flying my current policy afterwards — the single number that makes actions comparable (§6.1).
6. **What is bootstrapping, and what risk does it carry?**
   Training a value estimate toward a label built from the same estimate one step later
   (y = r + γQ(s′,a′)). It gives per-step learning without waiting for outcomes, and it is one
   leg of the deadly triad — with function approximation and off-policy data it can diverge,
   which is what the target-network and TD3 machinery manage (§7, §9).
7. **Why can you train on the hand-coded controller's flights?**
   The method is off-policy: the target `Q′(s′, μ′(s′))` asks what the *current* policy would
   do next, regardless of who produced the transition. Demonstrations are just more behaviour
   data — that is the entire theory of the pre-fill (§8, §17).

*Design*

8. **Why learn only two of four axes?**
   Depth already has a monotone, low-noise proxy (box size → metric standoff) that needs no
   learning; centring is a two-axis regulation problem under delay — the part worth a learned
   controller. Splitting also keeps the learned and hand-coded controllers off each other's
   axes, enforced by the mixer (§1, §21).
9. **Why an actor network at all — why not Q-learning alone?**
   Continuous actions: argmax over a continuum is an optimisation problem per tick. The actor
   *is* the learned argmax — trained by ascending the critic's action-gradient (the DPG
   theorem), then a single forward pass in flight (§10).
10. **Defend every entry of your 14-dimensional observation.**
    Current error (the task), previous error (finite-difference velocity — 46 px/step is a
    large signal), visibility and staleness (a missing box must look different from a centred
    one), and four past actions (the latency pipe's contents, k from the rule
    k = ⌈0.35 s × 10 Hz⌉). Nothing else is observable through a camera-only pipeline (§4.3).
11. **You changed the paper's reward — did you change the task?**
    Partly, and deliberately, and we can prove which part. The ÷100 is a positive linear
    transform — policy-invariant by Ng et al. 1999. The continuity repair and the smoothness
    term are *not* potential-based, so in principle they move the optimum — the repair is
    measured by the arm-F-vs-arm-R ablation, and the smoothness term is intentional task design
    against stick jitter under latency (§22).
12. **Why is the training environment credible?**
    Because it simulates what the agent actually faces — the projection, the *measured*
    latency, measured lag, detector noise from our own detector's statistics — and because it
    must pass a replay-validation gate against a recorded real flight before any result counts.
    The original's simulator, for contrast, contained no target motion at all [C] (§16, §25).
13. **Why compute reward from true geometry but observe through the delay pipe?**
    Reward is training-time information — it defines what we want (where the target *is*), and
    nothing at deployment needs it. Observing through the pipe is what makes latency hurt, so
    the policy is paid to compensate for it (§16.1).

*Algorithms*

14. **Why do you need target networks?**
    The TD label contains the network being trained; without a slow copy, each gradient step
    moves the next step's target — a fixed-point iteration whose target won't sit still.
    Polyak-averaged targets (τ = 0.005, time constant ≈ 200 gradient steps) restore a
    near-stationary regression (§11.5, §12.2).
15. **Why does DDPG overestimate, and by how much?**
    The target is an (implicit) max over actions, and a max of noisy unbiased estimates is
    biased up — for two estimates with noise σ, by exactly σ/√π ≈ 0.564σ (three-line
    derivation via E|X₁−X₂|). The optimism compounds through bootstrapping and the actor learns
    to exploit critic error (§12.4).
16. **What exactly does TD3 change, and why does the min help?**
    Three things only: twin critics with a min-target (the min is pessimistic by about what the
    max was optimistic — bias flipped to the damping side), clipped noise on the target action
    (labels become neighbourhood values — spike-proof), and actor/target updates at half the
    critic cadence (climb a settled surface) (§13).
17. **Why not PPO? Why not SAC?**
    PPO is on-policy: no replay, no demonstration pre-fill, fresh rollouts per update — wrong
    where samples cost battery and crash risk. SAC is a fine off-policy alternative but adds a
    stochastic head and temperature machinery a 2-D action doesn't need; it is the optional
    third arm, not the default. TD3 fixes DDPG's known failure on the same skeleton the
    reproduction already requires (§14).
18. **The original trained with almost no exploration noise and still worked — how, and why
    doesn't that transfer?**
    Verified from its code: OU with dt = 10⁻² yields ~0.03-per-step noise on a ±60 action
    scale, so exploration came from uniform random resets over the whole frame in a
    disturbance-free point-mass world with dense reward. Our world adds latency, target
    motion, and noise — coverage by resets alone no longer spans behaviour, so real action
    noise (and a warm-start) is required (§11.2, §16).

*Results scepticism*

19. **One seed of arm T beats the P-controller. Is the claim made?**
    No. DDPG-family variance across seeds is notorious (Henderson et al.): the protocol is ≥ 5
    seeds per arm, fixed eval scenarios, and a win only if the margin exceeds the seed spread.
    Anything less is an anecdote (§24, §28.4).
20. **What result would make you abandon the learned controller?**
    If no arm beats the P-controller on moving-target scenarios under injected latency by more
    than the seed spread — then the learning didn't earn its complexity, we ship the
    P-controller, and the paper reports that honestly. The baseline exists to make this
    decision possible, and saying so is the strongest credibility signal in the defence (§28.4).

## 27. The learning path — beginner to expert, staged

| Stage | Read / do | You are ready when |
|---|---|---|
| 1 · Vocabulary | §§1–5 + Appendix A; recompute the bandit and γ tables yourself | you can explain the transition tuple and γ to a non-RL engineer |
| 2 · The math core | §§6–9; derive Bellman by hand; run the −743/+57 update on paper | you can state the deadly triad and point to each leg in DDPG |
| 3 · The algorithms | §§10–14 + [rl_foundations.md](rl_foundations.md); [L5] both pages | you can write §11.7's pseudocode from memory and name each line's failure mode |
| 4 · The reference code | [L6] — read all ~200 lines of the author's TD3 | you can map every line to an equation in §11/§13 |
| 5 · The build | Part III + [implementation_plan.md](implementation_plan.md); implement env unit tests first (§28.2) | the smoke run solves the static world in a few thousand steps |
| 6 · Mastery | Part IV; then [L3], [L4] in full; [L8], [L9] abstracts+theorems | you pass §26 cold, including Q11 and Q18 |

External reading order: [L1 ch. 3, 6] → [L5] → [L3] → [L4] → [L8] → [L9]. The chase paper [P]
last — by then, every one of its gaps and discrepancies ([README.md](README.md)) will be
visible to you unaided.

## 28. Playbook — running it, and when it misbehaves

### 28.1 Pre-training checklist — the silent-bug list

1. Timeouts stored with `d = 0` (bootstrap); only true task-endings terminal (§11.3).
2. Target updates counted in **gradient** steps; targets initialised as exact copies (§11.5).
3. One observation-assembly module shared by env and deployment; checkpoint carries the
   normalisation constants (§19, §21).
4. No rescaling between tanh output and `/cmd_vel`; action-map signs verified once, recorded in
   the checkpoint (§3, §21).
5. Reward defined on every path — including the no-detection step (terminal + penalty) (§16.2).
6. Evaluation runs ε = 0, on seeds never used for tuning (§11.1, §24).
7. No gradient step before `update_after`; batch ≤ buffer fill (§11.8).
8. Noise semantics end-to-end: σ in normalised units, applied before the clip; any OU has an
   explicit dt (the keras-rl dt = 10⁻² finding, §11.2).
9. γ = 0.99 paired with ≥ 200-step episodes (§5).
10. Sim Δt equals the deployment control period exactly (§16.1).
11. Environment validated against a recorded real flight **before** any training conclusions
    (§16.3).
12. ≥ 5 seeds planned from the start; the 3-arm × 5-seed matrix scripted, not manual (§24).

### 28.2 First-run protocol

Unit-test before the loop: reward anchors (§16.2 values [V]); the latency queue delivers the
right vintage; terminated/truncated split behaves; observation matches a hand-computed vector.
Then a smoke run — `static` target, no latency, no noise: TD3 should reach near-perfect
time-in-view within a few thousand steps. If it cannot solve the trivial world, the bug is in
the system, not the hyperparameters. Then switch the full randomisation on, and the curriculum.

### 28.3 Symptom → diagnosis

The §23 dashboard is the full read; the fastest triage rows:

| Symptom | Diagnosis | First move |
|---|---|---|
| avg_Q ≫ realised eval return | overestimation loop (§12.4) | confirm arm T immune; else lower critic lr |
| Q₁−Q₂ gap unbounded (TD3) | one critic diverging | smoothing on? lower critic lr |
| actions pinned at ±1 early | actor lr / reward scale / init | halve actor lr; verify §7 scaling; §11.6 init |
| perfect in sim, oscillates in flight | sim latency weaker than reality, or k too small | re-measure Δ; recheck the k-rule (§4.2) |
| learns static, collapses on movers | curriculum too fast; history features broken | verify `ex_prev ≠ ex` in stored data |
| critic loss ≈ 0, nothing improves | collapsed critic / one-behaviour buffer | action-histogram diversity; longer warm-start |
| return great, flight jerky | w_smooth too low for the airframe | raise w_smooth; output rate limit [spec §4] |

### 28.4 Acceptance — when training is finished

Across ≥ 5 seeds on the fixed eval scenarios, the deployment candidate must: (1) beat the
P-controller on moving-target scenarios under injected latency by more than the seed spread;
(2) match it on the static scenario; (3) show no avg_Q/return divergence. If no arm clears
(1): **ship the P-controller**, and report it — see exam Q20.

---

## Appendix A — Glossary (the spoken vocabulary)

| Term | One sentence |
|---|---|
| agent / environment | the thing being trained (here: only the actor) / everything else, detector and latency included |
| policy (π, μ) | the rule mapping observations to actions; ours is deterministic (μ) |
| episode / rollout | one reset-to-end run; the trajectory it produces |
| return (G) | discounted sum of future rewards — what is actually maximised |
| discount (γ) | geometric down-weighting of the future; sets the attention horizon 1/(1−γ) |
| value function (V, Q) | expected return from a state / from a state-action pair |
| Bellman equation | the self-consistency law: value now = reward + discounted value next |
| TD error (δ) | the measurable violation of that law on one transition — the surprise |
| bootstrapping | using the current value estimate inside its own training label |
| Monte Carlo | labelling with actual episode returns instead of bootstrapped ones |
| on-policy / off-policy | must train on the current policy's data / may train on anyone's (ours) |
| behaviour vs target policy | who generated the data vs who is being improved |
| replay buffer | store of past transitions, sampled uniformly to decorrelate SGD |
| target network | slow Polyak-averaged copy that stabilises the TD label |
| Polyak averaging (τ) | θ′ ← τθ + (1−τ)θ′ — exponential moving average of weights |
| exploration / exploitation | trying actions to learn about them / using what is known |
| ε-greedy, action noise | discrete and continuous ways to force exploration |
| OU process | temporally correlated noise; mean-reverting random walk |
| actor-critic | policy network trained against a learned value network |
| DPG | the deterministic policy gradient: chain rule through the critic's action input |
| overestimation (max-bias) | max over noisy estimates is optimistic — by σ/√π for two of them |
| clipped double-Q | TD3's min over twin critics — bias flipped to the safe side |
| target policy smoothing | noise inside the label so value spikes can't be exploited |
| deadly triad | approximation + bootstrapping + off-policy: the unstable combination |
| POMDP / belief state | partially observed MDP / the distribution over true states given history |
| reward shaping / potential-based | reward edits to speed learning / the provably policy-invariant kind |
| domain randomisation | training over a distribution of simulated worlds |
| sim-to-real / reality gap | transferring a sim-trained policy / the MDP mismatch that breaks it |
| curriculum | staged difficulty, gated on performance |
| warm-start | initial uniform-random action phase filling the buffer with coverage |
| checkpoint contract | weights + the full spec needed to reproduce their meaning |

## Sources and verification record

| # | Source | Verified how, for this guide |
|---|---|---|
| [L1] | Sutton & Barto 2018, ch. 2–3, 6, 11, 13 | standard material (bandits, MDPs, TD, deadly triad, REINFORCE); no numeric claims taken |
| [L2] | Silver et al., ICML 2014 | DPG theorem form and the dropped state-distribution term (§10) |
| [L3] | Lillicrap et al., ICLR 2016 | **PDF fetched 2026-09-13, §7 read**: Adam 1e-4/1e-3, L2 1e-2, γ 0.99, τ 0.001, 400/300, action at 2nd critic layer, final init ±3e-3, batch 64, buffer 1e6, OU θ.15 σ.2, batch-norm usage, OU rationale quote |
| [L4] | Fujimoto et al., ICML 2018 | page fetched (title/venue); parameters cross-checked against [L6] |
| [L5] | Spinning Up, DDPG & TD3 pages | fetched 2026-09-13: (1−d) target form, pseudocode, defaults, Gaussian-vs-OU statement, brittleness/overestimation quote |
| [L6] | `sfujim/TD3` | fetched 2026-09-13: 256-256 nets, lr 3e-4, τ 0.005, policy_noise 0.2·max_action, clip 0.5, freq 2, min target, summed MSE, actor vs Q₁ |
| [L7] | Mnih et al., Nature 2015 | replay + target-network lineage; frame-stacking analogy |
| [L8] | Henderson et al., AAAI 2018 (arXiv:1709.06560) | **abstract fetched 2026-09-13**; quote in §24 |
| [L9] | Ng, Harada, Russell, ICML 1999, pp. 278–287 | **citation and theorem content verified via search 2026-09-13**: positive-linear + potential-based invariance, necessity |
| [C] | chase code @ `840487e` + keras-rl source | cloned 2026-09-11 ([rl_block_diagram.md](rl_block_diagram.md)); keras-rl `rl/random.py` fetched — dt = 1e-2 default, σ√dt increments, N(μ, σ) resets |
| [V] | recomputation for this guide | bandit example · γ table (0.9/0.95/0.99 → 1/2/10 s; γ¹⁰; 0.99¹⁰⁰ = 0.366; Σγ ≈ 95) · contraction rate (69 sweeps per halving) · E[max] = σ/√π ≈ 0.564 (simulated 0.566) · OU stationary std 0.548, ~667 steps to stationarity, per-step 0.03, 10-step growth 0.094 · tanh gradients 0.19/0.078/0.02 · reward anchors · 46 px/step · τ time constants 200/400/1 000 · return scale ±1.4×10⁴ |

**What deliberately stays open** (decide once, record with results): final `w_smooth` /
`w_loss` after the first sweeps; the per-stage curriculum gates; whether the optional SAC arm
earns its cost. Everything else is pinned to a source or marked [D] with its reasoning.
