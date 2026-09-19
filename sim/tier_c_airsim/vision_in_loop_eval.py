"""Job 2 -- vision-in-the-loop evaluation (sim_training_architecture.md
4.3): the full pipeline with NOTHING mocked.

    rendered frame -> REAL YOLO -> the SAME observation assembler ->
    TRAINED actor (inference) -> stick semantics -> engine velocity API
    -> the renderer moves the world -> next frame

This is where "the policy tolerates real detector noise, not our model of
it" is established -- the last gate before hardware. It is an EVALUATION,
not a training ground (rendering-bound, ~10 Hz: statistics, not
gradients). Scored exactly like the flight: time-in-view %, detection %,
capture rate / time-to-capture.

Needs (on the engine host): the repo's chase_gym + chase_train on
PYTHONPATH, ultralytics YOLO weights, a running engine. The follower is
flown by velocity commands in its BODY frame mapped from the policy's
stick convention -- with the same v_max / omega_max as every other tier.

    python3 vision_in_loop_eval.py --checkpoint best.pt --yolo weights.pt \
        --episodes 10 [--baseline p_controller]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import make_engine  # noqa: E402
import warehouse  # noqa: E402

# chase_gym/chase_train come from the repo workspace (PYTHONPATH); the
# imports are the one-contract property, not a convenience.
from chase_gym import constants as C
from chase_gym.corruption import Measurement
from chase_gym.env import EnvConfig, forward_command
from chase_gym.observation import ObservationAssembler, ObservationSpec
from chase_gym.baselines import PController, PNController
from chase_eval.evaluate import ActorPolicy

DT = C.DT


def load_policy(checkpoint: str, spec: ObservationSpec):
    """Returns (policy, env_cfg): the actor AND the EnvConfig it trained
    under -- v_max/omega_max/standoff/intercept params must be the trained
    ones, and the obs-spec hash cannot police those."""
    from chase_train.checkpoint import actor_from_bundle, load_bundle
    bundle = load_bundle(checkpoint, expect_obs_spec=spec)
    env_cfg = EnvConfig(**bundle['config']['env'])
    return ActorPolicy(actor_from_bundle(bundle)), env_cfg


def detect(yolo, frame: np.ndarray):
    """Real YOLO -> best-box Measurement or None, plus inference ms --
    the same best-box rule as the deployed chase_detector."""
    t0 = time.time()
    results = yolo(frame, verbose=False)
    ms = (time.time() - t0) * 1000.0
    best = None
    for r in results:
        for b in r.boxes:
            conf = float(b.conf[0])
            if best is None or conf > best[0]:
                x1, y1, x2, y2 = map(float, b.xyxy[0])
                best = (conf, x1, y1, x2, y2)
    if best is None:
        return None, ms
    _, x1, y1, x2, y2 = best
    return Measurement((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1), ms


def run_episode(eng, yolo, policy, cfg: EnvConfig, spec: ObservationSpec,
                seed: int, max_steps: int):
    rng = np.random.default_rng(seed)
    asm = ObservationAssembler(spec)
    if hasattr(policy, 'reset'):
        policy.reset()             # baselines carry filter/LOS state
    # Reset geometry RELATIVE to the vehicle's live pose: the target goes
    # r0 ahead along the current heading at the current altitude. The
    # vehicle is never teleported -- that does not stick in flight -- so
    # episodes start wherever the last one ended, with correct relative
    # geometry.
    (x, y, z), yaw = eng.get_vehicle_pose()
    r0 = rng.uniform(*cfg.resolved_reset_range())
    eng.set_object_pose('TargetTello',
                        (x + r0 * math.cos(yaw), y + r0 * math.sin(yaw), z),
                        yaw)
    time.sleep(0.5)

    stats = {'steps': 0, 'detected': 0, 'captured': False}
    for k in range(max_steps):
        frame = eng.get_rgb()
        meas, inf_ms = detect(yolo, frame)
        # Detection rate is the reported metric here; exact in-view truth
        # needs engine ground truth and is a Tier-B/A quantity.
        stats['detected'] += int(meas is not None)
        a = np.clip(np.asarray(policy.get_action(asm.assemble(meas, DT)),
                               dtype=np.float32), -1.0, 1.0)
        asm.record_action(a)
        v_fwd = forward_command(cfg, meas)
        # BODY-frame command (the Tello stick semantics): the engine's own
        # heading applies -- no client-side yaw dead-reckoning, which
        # drifts against controller lag and YOLO wall-time.
        v_up = float(a[0]) * cfg.v_max
        yaw_rate = float(a[1]) * cfg.omega_max
        eng.move_by_velocity_body(v_fwd, 0.0, -v_up, yaw_rate, DT)
        stats['steps'] += 1
        if meas is not None and cfg.task == 'intercept':
            d_est = C.FX * cfg.ref_width_m / max(meas.w_px, 1.0)
            if d_est <= cfg.r_cap_m:
                stats['captured'] = True
                break
    eng.hover()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', default='projectairsim',
                    choices=['projectairsim', 'classic'])
    ap.add_argument('--checkpoint')
    ap.add_argument('--baseline', choices=['p_controller', 'pn'])
    ap.add_argument('--yolo', required=True, help='ultralytics weights')
    ap.add_argument('--task', default=None,
                    choices=['follow', 'intercept'],
                    help='baselines only (default follow); a checkpoint '
                         'brings its own task')
    ap.add_argument('--address', default=None,
                    help='engine host (default: the local machine)')
    ap.add_argument('--no-warehouse', action='store_true',
                    help='use the packaged outdoor level instead of the '
                         'indoor warehouse scene')
    ap.add_argument('--lighting', default='medium',
                    choices=['low', 'medium', 'high'],
                    help='warehouse interior light level')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--out', default='vision_in_loop_result.json')
    args = ap.parse_args(argv)
    if not args.checkpoint and not args.baseline:
        ap.error('need --checkpoint or --baseline')

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    spec = ObservationSpec()
    if args.checkpoint:
        policy, cfg = load_policy(args.checkpoint, spec)
        if args.task and args.task != cfg.task:
            raise SystemExit(
                f'checkpoint was trained for task={cfg.task!r}; evaluate '
                f'it there (got --task {args.task})')
        label = os.path.basename(args.checkpoint)
    else:
        cfg = EnvConfig(task=args.task or 'follow')
        policy = (PController(spec) if args.baseline == 'p_controller'
                  else PNController(spec))
        label = args.baseline

    eng = make_engine(args.engine,
                      **({'address': args.address} if args.address else {}))
    eng.connect()
    if not args.no_warehouse:
        n_props, n_lights = warehouse.build_and_light(eng, seed=20,
                                                      lighting=args.lighting)
        print(f'warehouse: {n_props} props, {n_lights} lights, '
              f'lighting={args.lighting}')
    eng.takeoff()                    # velocity commands need flight mode
    episodes = []
    try:
        for e in range(args.episodes):
            st = run_episode(eng, yolo, policy, cfg, spec, seed=20_000 + e,
                             max_steps=args.steps)
            st['detection_rate'] = st['detected'] / max(st['steps'], 1)
            episodes.append(st)
            print(f"  ep {e}: steps {st['steps']}  detect "
                  f"{st['detection_rate']:.1%}  captured {st['captured']}")
    finally:
        eng.disconnect()

    agg = {
        'label': label, 'engine': args.engine, 'task': cfg.task,
        'episodes': episodes,
        'mean_detection_rate': float(np.mean(
            [e['detection_rate'] for e in episodes])),
        # Keyed on the RESOLVED task (a checkpoint brings its own):
        # args.task alone dropped the headline number for every intercept
        # checkpoint evaluated without an explicit --task.
        'capture_rate': float(np.mean(
            [e['captured'] for e in episodes])) if cfg.task == 'intercept'
        else None,
    }
    with open(args.out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"vision-in-the-loop: detection {agg['mean_detection_rate']:.1%} "
          f"-> {args.out}")


if __name__ == '__main__':
    main()
