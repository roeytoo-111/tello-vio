# Tier C — Photoreal perception jobs (Project AirSim / classic AirSim)

The perception tier of
[sim_training_architecture.md §4](../../docs/drone_chasing_rl/sim_training_architecture.md):
**Job 1** `dataset_factory.py` (YOLO training set with exact auto-labels,
stratified by target pixel size) and **Job 2** `vision_in_loop_eval.py`
(real YOLO on rendered frames, closed loop, the last gate before hardware).
Nothing gradient-critical lives here — the jobs are batch and replaceable
by design.

## Engine (verified 2026-09-17, offline_training_recipe.md §6.3)

| Engine | Status | Use |
|---|---|---|
| **Project AirSim** (`iamaisim/ProjectAirSim`) | active, MIT, `pip install projectairsim` (1.0.2), **prebuilt Linux env binaries** on its Releases, `-RenderOffScreen` documented | **default** — native 2D bbox annotations make the dataset factory trivial |
| Colosseum | **archived 2026-07-11**, no prebuilt envs (UE 5.6 build required) | fallback, pinned commit only |
| Cosys-AirSim (`pip install cosysairsim`, 3.5.0) | active | the healthier classic-API fallback |

Three portability traps `engine.py` contains (do not spread them):
Project AirSim yaw is **radians**, classic is **degrees**; segmentation
ImageType is **3** vs classic **5**; the Project AirSim command API is
asyncio.

## Where this runs

**Not on the WSL2 training box.** UE rendering inside WSL2 is undocumented
territory for both engines; the documented topology is split-host: engine
on the Windows host (or a native-Linux GPU box), client scripts connect
over the LAN (`ProjectAirSimClient(address=...)` ports 8989/8990, or
classic `settings.json` `LocalHostIp`/`ApiServerPort 41451`).

Quick start with the prebuilt Project AirSim environment:

```bash
./Blocks.sh -RenderOffScreen                 # engine host
pip install projectairsim ultralytics
python3 dataset_factory.py --engine projectairsim --out chase_dataset/
python3 vision_in_loop_eval.py --checkpoint <best.pt> --yolo <weights.pt>
```

No engine at hand? `python3 dataset_factory.py --dry-run` exercises the
label arithmetic.

## Non-negotiables carried from the doc set

- Camera = the **real** sensor: 960×720, H-FOV **55.1°** (our ost.txt
  calibration) — configured in `settings/` for both engines. Matching FOV
  matters more than matching lens model.
- The dataset **must be mixed with real captures** before detector
  training (implementation_plan.md Phase 0) — synthetic-only detectors
  inherit the renderer's domain.
- The measured accuracy/jitter/dropout-vs-size curves from Job 2 + the
  metadata.csv stratification are exported as the corruption-model file
  Tier A and the Tier-B oracle consume — the "C feeds A" arrow.
- Sign/units test before trusting anything (§4.4): NED, z-down; command
  one axis at a time and assert the image-space effect, exactly as
  `chase_sim_gz/sign_test.py` does for Tier B.
- Vision-in-the-loop is an **evaluation** (~10 Hz): promotion requires
  ≥ the PN / P-controller baseline under real YOLO noise.
