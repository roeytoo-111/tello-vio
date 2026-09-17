"""The A<->B dynamics gate (sim_training_architecture.md 3.5 item 1):
run IDENTICAL action logs through Tier A (point kinematics) and Tier B
(rigid body / fake), compare the box trajectories. A standing gate, not a
one-off -- it is what bounds the sim2sim gap before any checkpoint is
trusted across tiers.

    python3 -m chase_sim_gz.ab_replay_gate [--fake] [--episodes 3]
                                           [--tolerance-px 40]

Both envs run the same config with latency and corruption OFF (dynamics
comparison, not pipeline comparison), same seeds, same scripted action
log (a seeded P-controller run in Tier A, replayed open-loop in Tier B).
The report gives per-step |Delta u, Delta v| statistics; the tolerance is
the acceptance bound to tighten as calibration improves.
"""
import argparse
import json
import sys

import numpy as np

from chase_gym import ChaseEnv, EnvConfig, PController

from .gz_iface import FakeGzBackend, GzTransportBackend
from .gz_chase_env import GzChaseEnv


def collect_tier_a(cfg: EnvConfig, seed: int, max_steps: int):
    env = ChaseEnv(cfg)
    ctl = PController(env.obs_spec)
    obs, info = env.reset(seed=seed)
    actions, track = [], []
    for _ in range(max_steps):
        a = ctl.get_action(obs)
        obs, r, term, trunc, info = env.step(a)
        actions.append(a)
        track.append((info['u'], info['v'])
                     if info['in_frame'] else (np.nan, np.nan))
        if term or trunc:
            break
    return actions, track


def replay_tier_b(env: GzChaseEnv, seed: int, actions):
    obs, info = env.reset(seed=seed)
    track = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        track.append((info['u'], info['v'])
                     if info['in_frame'] else (np.nan, np.nan))
        if term or trunc:
            break
    return track


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fake', action='store_true',
                    help='Tier B = kinematic fake (CI); default real gz')
    ap.add_argument('--world', default=None)
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--scenario', default='static',
                    help='static isolates follower dynamics (the target '
                         'generator is stochastic per-backend otherwise)')
    ap.add_argument('--tolerance-px', type=float, default=40.0)
    ap.add_argument('--out', default='ab_replay_gate.json')
    args = ap.parse_args(argv)

    # t_lag_jitter 0: the gate compares DYNAMICS, so Tier A runs its
    # nominal lag; against the fake backend the same constant is used for
    # its response, making the CI gate isolate integration/placement error.
    cfg = EnvConfig(scenario=args.scenario, latency=False, corruption=False,
                    episode_steps=args.steps, t_lag_jitter=0.0)
    if args.fake:
        backend = FakeGzBackend(response_tau=cfg.t_lag)
    else:
        import os
        world = args.world or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'worlds', 'chase.sdf')
        backend = GzTransportBackend(world)
    env_b = GzChaseEnv(backend, cfg)

    per_episode = []
    try:
        for e in range(args.episodes):
            seed = 5000 + e
            actions, track_a = collect_tier_a(cfg, seed, args.steps)
            track_b = replay_tier_b(env_b, seed, actions)
            n = min(len(track_a), len(track_b))
            err = [float(np.hypot(track_a[i][0] - track_b[i][0],
                                  track_a[i][1] - track_b[i][1]))
                   for i in range(n)
                   if np.isfinite(track_a[i][0])
                   and np.isfinite(track_b[i][0])]
            per_episode.append({
                'seed': seed, 'steps_compared': len(err),
                'median_px': float(np.median(err)) if err else float('nan'),
                'p90_px': float(np.percentile(err, 90)) if err else float('nan'),
            })
            print(f"  episode seed {seed}: {len(err)} steps, median "
                  f"{per_episode[-1]['median_px']:.1f} px, p90 "
                  f"{per_episode[-1]['p90_px']:.1f} px")
    finally:
        env_b.close()

    medians = [e['median_px'] for e in per_episode
               if e['median_px'] == e['median_px']]
    overall = float(np.median(medians)) if medians else float('nan')
    ok = bool(medians) and overall <= args.tolerance_px
    result = {'tier_b_backend': 'fake' if args.fake else 'gz',
              'scenario': args.scenario, 'episodes': per_episode,
              'overall_median_px': overall,
              'tolerance_px': args.tolerance_px, 'pass': ok}
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'A<->B gate: overall median {overall:.1f} px vs tolerance '
          f'{args.tolerance_px} -> {"PASS" if ok else "FAIL"} ({args.out})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
