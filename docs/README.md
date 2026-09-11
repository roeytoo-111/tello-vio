# Project Planning — How to Read It

Ten design documents, ~2,900 lines, two source papers, no code. This page is the **only entry
point you need**; everything else is reached from here. Read this section first — it tells you
which of the two document sets is live, and which parts of the older one no longer apply.

---

## The 60-second orientation

| | |
|---|---|
| **What the project is** | A DJI Tello that chases **another DJI Tello**, using a camera only, with the chase policy learned by deep RL. All compute is off-board. |
| **The live source** | Tan & Karaköse, *"A new drone chasing drone approach based on deep reinforcement learning with accelerated rewards"*, SoftwareX 31 (2025) 102201. YOLOv3 → DDPG. |
| **What exists in code** | Only the flight baseline: `workspace/src/tello` (driver), `tello_msg`, `tello_control`. **No detection, no RL, no training code has been written.** |
| **What these docs are** | The architecture and the decisions, with every claim traced to a source. They are a plan, not a description of working software. |

### The two document sets, and which one is live

```
docs/
├── drone_chasing_rl/     <- LIVE. The RL project. Start here.
└── research_proposal/    <- A DIFFERENT paper ("Recognise, Then Range"). Not the RL work.
```

`research_proposal/` was built from a different source PDF that is no longer in the tree, and that
paper contains **no reinforcement learning at all** — its controller is deliberately fixed. Do not
read it looking for the RL design. It stays because two things in it are still useful: the
verified Tello platform baseline, and its pinhole ranging equation, which §6.4 of the live set now
reuses.

---

## Reading paths — pick the one that matches your question

### Path A — "I want to understand the project" (~45 min, 3 documents)

1. **[drone_chasing_rl/README.md](drone_chasing_rl/README.md)** — what the paper claims, why this
   repo fits it, and the **ten verified inconsistencies in the paper**. Read the inconsistency
   table properly; it is the reason several later decisions exist.
2. **[drone_chasing_rl/block_diagram.md](drone_chasing_rl/block_diagram.md)** §1–2 — the published
   approach, then the hybrid split (DDPG owns vertical + yaw; forward/back is a hand-coded rule).
   That split is the single most important structural fact about the design.
3. **[drone_chasing_rl/rl_specification.md](drone_chasing_rl/rl_specification.md)** §1–4 — the MDP,
   the geometry, then state and action.

Stop there. You now know what is being built and why.

### Path B — "I am implementing the RL" (the working path)

Read in this order, and treat the last one as authoritative where they disagree:

0. **[rl_foundations.md](drone_chasing_rl/rl_foundations.md)** — if RL is not already second
   nature, start here: the basics (MDP, Bellman, actor-critic, DDPG's machinery) explained
   deeply and entirely in this project's terms. The spec then reads as conclusions instead of
   assertions.
1. **rl_specification.md** — the whole file. It is the reference: state, action, reward, algorithm,
   hyperparameters, episode structure. Every section separates *what the paper says* from *what is
   recommended*, so you can always see which is which.
2. **[drl_ros2_reference_analysis.md](drone_chasing_rl/drl_ros2_reference_analysis.md)** §3–4 —
   what to take from the reference codebase and what cannot cross over. §4.1 (their environment
   pauses Gazebo's physics; a Tello cannot be paused) is the constraint that shapes the whole
   training loop.
3. **§2 of the same file** — two defects confirmed by running the code, not by reading it. Do not
   copy the TD3 critic.
4. **[implementation_plan.md](drone_chasing_rl/implementation_plan.md)** — packages, interfaces,
   phases, risks.

### Path C — "I need to decide something today"

Jump straight to the decision:

| Question | Go to |
|---|---|
| What does the agent see / do / get rewarded for? | rl_specification.md §3, §4, §6 |
| DDPG or something else? | rl_specification.md §7 + reference analysis §5 |
| How far behind should the follower fly? | **reference analysis §6.3–6.4** (the published thresholds are unsafe — see below) |
| How do I run two Tellos? | reference analysis §6.2 |
| What can I reuse from the existing repo? | implementation_plan.md §1 |
| What is safe to copy from the reference GitHub repo? | reference analysis §3 (adopt) and §4 (do not) |
| What would falsify / break this? | implementation_plan.md §6 |

---

## How these documents are written — the conventions that make them fast to read

Every claim carries one of three tags. Once you know them you can read at speed, because the tag
tells you how much to trust a line and whether it is arguable:

| Tag | Means | How to treat it |
|---|---|---|
| **[P]** | Stated in the source paper (section cited) | Fixed. If you disagree, you disagree with the paper. |
| **[V]** | Verified in this repository's code or by computation (`file:line` given) | Fact. Re-checkable in seconds. |
| **[D]** | A design decision made in the documents, not in the paper | **Arguable — this is where your judgement is wanted.** |

So: **skim [P], trust [V], argue with [D].** If you only have time to review one category, review
the [D]s; they are the open choices.

Two more conventions: each document names verified problems in its own source rather than
smoothing them over, and every number that could be recomputed *was* recomputed.

---

## Known-stale sections — read the correction, not the original

The documents were written across several sessions, and the newest analysis revised parts of the
older ones. Corrections are now marked **in place**, so linear reading is safe. For completeness:

| Where | What changed | Authoritative version |
|---|---|---|
| rl_specification.md §5 | 20 % / 55 % box-ratio thresholds are **unsafe at Tello scale** — "forward" only fires at 0.47 m | reference analysis §6.3–6.4 |
| block_diagram.md §4 | The target is a **commanded Tello on a second host**, not a hand-flown quadcopter | reference analysis §6.2 |
| README.md (RL set) | The paper's "follower slower than target" limitation is **void** — both aircraft are Tellos | reference analysis §6.1 |

A full list is kept at the end of the reference analysis (§8).

---

## The three things most worth knowing before you build anything

1. **The published depth thresholds would fly the aircraft into each other.** Re-derived with this
   repo's calibration and the Tello's real width, "too far — move forward" is only reached at
   0.47 m. Replace with the metric standoff `d = fx·W/w`. Because both drones are the same known
   model, this needs no dimension table and no recognition step. *(reference analysis §6.3–6.4)*
2. **The reference codebase's timing model cannot be reused.** It pauses Gazebo around every step,
   giving a synchronous zero-latency MDP. A Tello cannot be paused and carries 150–350 ms of
   measured video latency. The policy must publish continuously at ≥ 10 Hz, because the driver
   zeroes any command older than 0.35 s. *(reference analysis §4)*
3. **The paper contradicts itself in ten places**, including which axes the agent controls and a
   reward that is discontinuous at its own threshold. Each has a recommended resolution; none is
   fatal. *(RL set README, inconsistency table)*

---

## Where the code will go

Nothing below exists yet. This is the target layout, for orientation when reading the plan:

```
workspace/src/
├── tello/          EXISTS — driver: /image_raw, /cmd_vel, dead-man, emergency
├── tello_msg/      EXISTS — TelloStatus, TelloID, ...
├── tello_control/  EXISTS — keyboard GUI, manual override + abort
└── rtr_*/          PLANNED — detection, RL policy, follow controller, safety supervisor
```
