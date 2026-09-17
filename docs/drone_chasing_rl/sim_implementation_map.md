# Simulator Implementation Map — branch `simulation-ddpg-train`

**What this file is.** The code-level map of the simulator-training implementation on this
branch, and the log of every decision the code had to make that the doc set left open. The
*specifications* live in [sim_training_architecture.md](sim_training_architecture.md) (the
three tiers), [rl_training_guide.md](rl_training_guide.md) Part III (Tier A + trainer),
[rl_specification.md](rl_specification.md) (obs/action/reward), and
[rl_block_diagram.md](rl_block_diagram.md) (the faithful arm, verbatim from the paper's code).
This file only says **where each specified component lives, what was decided during the build,
and how to run it** — nothing here re-specifies the design. Tags as in the doc set; **[D-impl]**
marks a decision made on this branch, to be revisited when measurements arrive.

## 1. Where everything lives

```
workspace/src/
  chase_gym/        Tier A + the shared contract modules (ament_python, imports NO ROS)
    chase_gym/constants.py       verified numbers, one place: fx=fy=919.42, c=(480,360),
                                 960x720, Δt=0.1 s, Tello 0.098/0.041/0.18 m, k=4
    chase_gym/observation.py     §4.3 assembler (14-dim, k=4) + faithful raw-pixel mode +
                                 the spec-hash the checkpoint contract carries
    chase_gym/reward.py          §16.2 repaired+scaled / faithful threshold-100 unscaled /
                                 INTERCEPT potential shaping + capture bonus (Ng et al. form)
    chase_gym/latency.py         sim-time delay queue (measured-distribution base + jitter)
    chase_gym/corruption.py      σ_px jitter · size-dependent dropout · burst loss (two-state)
    chase_gym/target_motion.py   the five generator families + curriculum speed caps
    chase_gym/kinematics.py      follower first-order lag + world→camera pinhole projection
    chase_gym/env.py             ChaseEnv (gymnasium.Env): FOLLOW and INTERCEPT, §16.1 step
    chase_gym/pipeline.py        MeasurementPipe + the cross-tier reset DRAW ORDER CONTRACT
    chase_gym/faithful_env.py    exact port of the original drone_sim_env.py [C]
    chase_gym/baselines.py       P-controller + PN — get_action(obs) interface (demo source)
  chase_train/      the one trainer (ament_python, imports NO ROS)
    chase_train/networks.py      256×256 ±3e-3 actor/twin-critic + faithful 16³/32³ variant
    chase_train/noise.py         Gaussian σ 0.3→0.05 + OU with explicit dt (faithful semantics)
    chase_train/buffer.py        preallocated ring, d=terminated only, per-step Δt,
                                 .npz demo prefill with reward recompute, min-fill guard
    chase_train/td3.py           ONE class; §13.5 flags make arms T / R / F
    chase_train/config.py        the §15 config block; YAML presets per arm
    chase_train/checkpoint.py    §19 bundle + versioned names + best.json + ONNX (opset 17)
    chase_train/train.py         §11.7 loop, curriculum, eval hook, stopping rule; CLI
    chase_train/run_matrix.py    arms × seeds matrix, scripted (--with-sb3, auto-report)
    chase_train/report_matrix.py IQM + bootstrap CIs on held-out episodes, P-controller
                                 baseline CI, the guide-28.4 acceptance verdict
    chase_train/sb3_check.py     SB3 2.9 TD3 on the identical env (external cross-check)
    config/{td3,ddpg_repaired,ddpg_faithful,td3_intercept}.yaml
  chase_eval/       evaluation + gates (ament_python, imports NO ROS)
    chase_eval/evaluate.py       frozen-scenario suite (per-episode scores), ActorPolicy,
                                 MOVING_FAMILIES + the shared selection metric
    chase_eval/cli.py            evaluate a bundle or the P/PN baselines on the suite
    chase_eval/stats.py          IQM + episode/stratified bootstrap CIs (rliable protocol)
    chase_eval/replay_validation.py  the §16.3 gate: static-target protocol, centre+width
  chase_sim_gz/     Tier B (ament_python, ROS 2 Humble + gz-sim Harmonic)
    chase_sim_gz/gz_iface.py     the only file that talks gz-transport; mockable interface
    chase_sim_gz/gz_chase_env.py GzChaseEnv: lockstep WorldControl multi-step (100 × 1 ms)
    chase_sim_gz/tello_sim_shim.py   sticks[-1,1]→velocity node, dead-man 0.35 s replicated
    chase_sim_gz/sim_oracle_detector.py  truth→DroneDetection with OUR calibration + corruption
    chase_sim_gz/latency_shim.py     sim-time delay of DroneDetection messages
    chase_sim_gz/sign_test.py    §3.2-item-4 protocol, automated (run after install)
    chase_sim_gz/calibrate_lag.py    step-response fit vs the measured T_lag (§3.2)
    chase_sim_gz/ab_replay_gate.py   identical action log through Tier A and Tier B
    chase_sim_gz/finetune.py     continue a Tier-A bundle ~1e4 lockstep steps (§3.5)
    chase_sim_gz/gen_world.py    emits worlds/chase.sdf from one parameterisation
    worlds/chase.sdf             two multicopters, MulticopterVelocityControl + motor models
    launch/tier_b.launch.py · config/bridge_*.yaml
sim/tier_c_airsim/  Tier C (engine-hosted machine; not a ROS package)
    engine.py                    ONE adapter over Project AirSim / classic (units, yaw
                                 handedness, image-type traps contained here)
    dataset_factory.py           YOLO dataset with exact auto-labels, operator-attested lighting
    vision_in_loop_eval.py       real YOLO + trained actor closed loop, flight metrics
    tier_c_sign_test.py          the §4.4 sign protocol against a live engine
    settings/ · README.md        camera 960×720 FOV 55.1°; Project AirSim default,
                                 Colosseum-pinned fallback
scripts/install_gz_harmonic.sh   apt steps for gz-sim Harmonic + ros_gz on 22.04 (needs sudo)
```

One-contract enforcement: `chase_sim_gz` and `sim/tier_c_airsim` **import**
`chase_gym.observation`, `chase_gym.reward`, `chase_gym.constants` — the same modules the
trainer used — so a checkpoint runs in Tiers B/C without one line changing.

## 2. Decisions made on this branch [D-impl]

| # | Decision | Value chosen | Why / when to revisit |
|---|---|---|---|
| 1 | Host reality: this WSL2 box has only Gazebo **Classic 11** and no sudo session | Tier B code is written against gz-sim Harmonic per spec, exercised in CI-style tests through a kinematic fake of `gz_iface`; `scripts/install_gz_harmonic.sh` + `sign_test.py` complete the real-gz verification after the user runs the install (needs sudo) | Harmonic supports Ubuntu 22.04 (Jammy); Classic 11 is EOL and explicitly excluded [S1] |
| 2 | Stick gains Tier A/B share | `v_max = 1.5 m/s` (all linear axes), `ω_max = 1.5 rad/s` | v_max from the shim spec [D safety, sim_training_architecture §3.2]; ω_max set to keep a full-stick yaw sweep of the 55.1° FOV ≈ 0.64 s — parameter, calibrate in Phase 1 |
| 3 | `T_lag` placeholder until the Phase-1 step-response flight | 0.25 s, randomised ±30 % per episode | mid-range of small-quad velocity-loop constants; **flagged loudly in config; the measured value replaces it** |
| 4 | Latency model default | base ~ U(150, 350) ms per episode + N(0, 20 ms) per frame, floor 0 | the measured range [V]; an empirical histogram file can replace the uniform draw (`latency.from_samples`) |
| 5 | Corruption defaults until Phase-0 CSVs exist | σ_px = 2 px; dropout p(w): 2 % at w ≥ 90 px rising smoothly to 30 % at w ≤ 15 px; burst loss two-state Markov (enter 0.5 %/step, mean length 5 steps) | placeholder shape follows [P App. A] small-target weakness + the CTS burst-loss failure mode [offline_training_recipe §6.1]; `chase_detector` CSVs will fit the real curve |
| 6 | Staleness normaliser (obs index 5) | loss_timeout = 1.0 s | matches the REACQUIRE-scale timeout; recorded in the checkpoint's obs spec |
| 7 | Faithful arm timeout handling | 10-step cutoff stored as **terminal** (d = 1) | keras-rl forces `done=True` at `nb_max_episode_steps` — reproducing the artifact requires reproducing its bug; arms R/T store truncations with d = 0 per §11.3 |
| 8 | Faithful arm env | `FaithfulPointMassEnv`, an exact port of `drone_sim_env.py` [C]: p += 2a, edge-terminal, threshold-100 reward, uniform reset, ±60 actions, OU noise added post-scale | the reproduction claim needs the original's world, not ChaseEnv |
| 9 | FOLLOW forward axis inside Tier A | metric standoff on the **delayed, corrupted** measurement (d = fx·W/w), setpoint 2.0 m, deadband ±0.25 m, gain 0.8 s⁻¹, hold 0 on miss | the real depth rule will also act on delayed detections; using truth here would understate coupling. Setpoint mid `[1.5, 2.5]` m band [drl_ros2_reference_analysis §6.4] |
| 10 | INTERCEPT closing law | v_cmd = clip(1.0·(range−r_cap), **v_min = 0.15**, 1.2 m/s) while outside r_cap, 0 inside, on the same delayed measurement; r_cap = 0.5 m, B = +10, λ = 0.1, approach band 2–6 m | directly from sim_training_architecture §1; the floor keeps the approach *crossing* r_cap instead of stalling asymptotically on it (found empirically: capture never fired without it); gains are config, recorded with results |
| 11 | Reset draw (FOLLOW) | range ~ U(1.5, 2.5) m; box centre uniform inside a 15 %-margin frame rectangle, back-projected | "uniform in frame at a standoff drawn from the operating band" [guide §16.3] |
| 12 | Yaw/vertical sign convention (Tier A) | REP-103: +yaw (CCW) moves the box **right** (+u); +linear.z (up) moves the box **down** (+v); derived and unit-tested, recorded in the checkpoint action map | the doc set's standing sign-test rule [S5]; Tier B/C re-test it per tier |
| 13 | Curriculum gate | advance when rolling eval time-in-view > 90 % on the current stage; can step back down on < 70 % | guide §20 + the rolling-rate correction from [drl_ros2_reference_analysis §3.6] |
| 14 | Stopping rule | budget (2×10⁵) exhausted **or** eval-IQM plateau over 5 consecutive evals after the curriculum completes | offline_training_recipe §7 stage 3 |
| 15 | Tier C client | Written against the classic AirSim-lineage Python API (Colosseum-compatible), with the Project AirSim port isolated behind one adapter file | Project AirSim's client API differs; the adapter keeps the jobs engine-portable — verified statuses in offline_training_recipe §6.3 |
| 16 | Reset draw-order contract | The shared draw prefix (placement → family → latency base → generator params; tier-specific draws LAST) lives in `chase_gym/pipeline.py` and is regression-tested cross-tier with latency ON | the contract broke twice while it lived as two textual copies (caught by the A↔B gate and again in review); one implementation cannot drift |
| 17 | Replay-validation protocol | **Static target** during the validation flight; its world position estimated from the earliest detections, comparison timestamp-matched on the detection timeline with substepped replay | the follower's pose is never measured (no mocap), so a moving-target reconstruction through replayed states would compare the sim against itself; self-test asserts the gate separates true from 3×-wrong T_lag |
| 18 | Eval seed blocks | Selection (curriculum, best-checkpoint, plateau) uses `eval_seed0`; the REPORTED numbers come from a disjoint held-out block (`eval_seed0 + 500000`) run once at training end | best-checkpoint selection on the same seeds it reports is optimistic bias (guide §24); both numbers land in metrics.jsonl |
| 19 | Tier-B odometry sync | `GzTransportBackend.step()` waits for the world clock AND for odometry stamped **after the step began** (steps ≥ one 10 ms publish period guarantee a fresh sample); world reset waits for the clock to rewind **below the pre-reset time** | odometry arrives on its own topic; a target-relative slack would accept pre-step (pre-teleport) poses on short settle steps, and an absolute rewind threshold is fooled by stale callbacks on short-lived worlds |
| 20 | Curriculum ↔ eval coupling | With curriculum enabled, eval cadence **legitimately** changes the training trajectory: stage advancement consumes eval results (guide §20). Seeded reproducibility holds exactly at fixed cadence; cadence-independence holds with `curriculum.enabled=false` (verified empirically) | this is spec-mandated coupling, not an implementation leak — the diag-RNG isolation removed the only illegitimate channel |
| 21 | Faithful arm stopping | `eval.plateau_enabled: false` in ddpg_faithful.yaml: the original `fit(nb_steps=100000)` ran unconditionally, so the reproduction arm must too | verification caught the default plateau stop silently applying to arm F |
| 22 | Tier-B per-episode dynamics randomisation | **Deferred**: gz-sim motor/controller constants are fixed at world load; per-episode ±30 % randomisation (arch §3.2 step 3) has no supported runtime lever. The residual randomisation lives in Tier A (where all gradients run); Tier B's fine-tune arm runs the calibrated nominal dynamics | revisit if gz exposes runtime plugin reconfiguration; record with fine-tune results |
| 23 | Lockstep bypasses the ROS shim | `GzChaseEnv` maps sticks→metric itself (the same constants as `tello_sim_shim`) instead of routing through the ROS node: the lockstep RL path is deliberately ROS-free for determinism. The shim + oracle + latency-shim **nodes** are rehearsed by `tier_b.launch.py`; `sign_test.py` covers plugin→physics→projection | arch §3.4's "through tello_sim_shim" is satisfied semantically (identical mapping), not topologically, in the training path |
| 24 | Tier-C lighting axis | The dataset factory takes `--lighting <class>` as an operator-set label, one run per class — the client API cannot set scene lighting, and a fake in-process loop would stamp three labels onto identical frames | verification caught the original fake loop |
| 25 | PN baseline status | Implementation-verified: 60 % capture in the clean world (par with early RL arms), **0 % under the measured latency+noise** — its differentiated LOS-rate term is what delay and dropout destroy | not a bug: "beat PN under latency" is the meaningful bar (arch §1); report PN's clean-world number alongside, per the honesty rule |

## 3. The verification story on this branch

1. **Unit tests** (pytest, no ROS, no gz): reward anchors (§16.2 values exactly), latency-queue
   vintage, terminated/truncated split, observation vector vs hand-computed, sign conventions,
   projection round-trip, buffer d-flags, OU/Gaussian noise statistics, TD3-update shapes and
   target-freeze, faithful-env port vs the published code's arithmetic.
2. **Smoke training** (runs on this machine, CUDA): `static` target, no latency, no noise —
   TD3 must reach near-perfect time-in-view within a few thousand steps [guide §28.2];
   plus a short faithful-arm run reproducing the original's reward scale.
3. **Tier B without gz**: `GzChaseEnv` and the three nodes run against `FakeGzBackend`
   (kinematic double implementing the same `gz_iface` calls) — proves lockstep bookkeeping,
   oracle projection, latency shim on sim time. The **real** gz path is completed by
   `scripts/install_gz_harmonic.sh` (sudo) → `sign_test.py` → `calibrate_lag.py`.
4. **Tier C**: cannot run without an Unreal host; scripts ship with a `--dry-run` that
   exercises the label/bbox arithmetic against synthetic segmentation masks.

## 4. How to run (quick reference)

```
# Tier A smoke test (fast, CPU or CUDA)
python3 -m chase_train.train --config workspace/src/chase_train/config/td3.yaml \
        --override env.scenario=static env.latency=off train.total_steps=8000 --seed 0

# One full arm, one seed
python3 -m chase_train.train --config .../td3.yaml --seed 0

# The matrix (3 arms × 5 seeds + SB3 cross-check)
python3 -m chase_train.run_matrix --arms td3 ddpg_repaired ddpg_faithful --seeds 0 1 2 3 4

# Evaluate a checkpoint on the frozen suite
python3 -m chase_eval.evaluate --checkpoint runs/<...>/best.pt

# Tier B (after scripts/install_gz_harmonic.sh):
python3 -m chase_sim_gz.sign_test          # FIRST — the standing sign rule
ros2 launch chase_sim_gz tier_b.launch.py  # full graph rehearsal
```

Outputs land in `runs/<arm>_s<seed>_<stamp>/` (gitignored): config snapshot, JSONL metrics,
checkpoints `arm_s{seed}_step{n}_tv{score}.pt`, `best.json`, ONNX export.
