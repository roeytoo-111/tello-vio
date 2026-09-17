"""Evaluate a checkpoint bundle or a named baseline on the frozen suite.

    python3 -m chase_eval.cli --checkpoint runs/<...>/best.pt
    python3 -m chase_eval.cli --baseline p_controller [--task intercept]
    python3 -m chase_eval.cli --baseline pn --task intercept

Prints the per-family table and writes eval_result.json next to the
checkpoint (or into --out for baselines). Baselines run on the same suite
with the same frozen seeds -- the comparison the source paper never made
(rl_specification.md 10).
"""
import argparse
import json
import os

import numpy as np

from chase_gym import ChaseEnv, EnvConfig, PController, PNController

from .evaluate import default_env_factory, run_suite


def _print_table(suite: dict) -> None:
    cols = ('family', 'return_mean', 'time_in_view', 'loss_rate',
            'capture_rate', 'mean_dist_px')
    print(f"{'family':<24}{'return':>9}{'tv%':>8}{'loss%':>8}"
          f"{'capt%':>8}{'dist':>8}")
    for fam, s in suite['families'].items():
        print(f"{fam:<24}{s['return_mean']:>9.2f}"
              f"{100 * s['time_in_view']:>8.1f}{100 * s['loss_rate']:>8.1f}"
              f"{100 * s['capture_rate']:>8.1f}{s['mean_dist_px']:>8.1f}")
    a = suite['aggregate']
    print(f"{'AGGREGATE':<24}{a['return_mean']:>9.2f}"
          f"{100 * a['time_in_view']:>8.1f}{100 * a['loss_rate']:>8.1f}"
          f"{100 * a['capture_rate']:>8.1f}{'':>8}  "
          f"action-sat {a['action_sat_frac']:.1%}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint')
    g.add_argument('--baseline', choices=['p_controller', 'pn'])
    ap.add_argument('--task', choices=['follow', 'intercept'], default=None)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--eval-seed0', type=int, default=10_000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    if args.checkpoint:
        # Lazy import: chase_train depends on chase_eval, not the reverse.
        from chase_train.checkpoint import actor_from_bundle, load_bundle
        import torch
        bundle = load_bundle(args.checkpoint)
        env_kwargs = {k: v for k, v in bundle['config']['env'].items()}
        if args.task:
            env_kwargs['task'] = args.task
        if bundle['obs_spec']['mode'] == 'raw_pixels':
            raise SystemExit(
                'the faithful arm evaluates in its own world during '
                'training; the frozen suite needs the latency-aware spec')
        # The refusal handle: rebuild the spec THIS code produces and
        # compare hashes (guide 19).
        probe = ChaseEnv(EnvConfig(**env_kwargs))
        load_bundle(args.checkpoint, expect_obs_spec=probe.obs_spec)
        actor = actor_from_bundle(bundle)

        def policy(obs):
            with torch.no_grad():
                t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                return actor(t).squeeze(0).numpy()
        label = os.path.basename(args.checkpoint)
        out_path = args.out or args.checkpoint.replace(
            '.pt', '_eval_result.json')
    else:
        env_kwargs = {'task': args.task or 'follow', 'scenario': 'mix'}
        probe = ChaseEnv(EnvConfig(**env_kwargs))
        policy = (PController(probe.obs_spec)
                  if args.baseline == 'p_controller'
                  else PNController(probe.obs_spec))
        label = args.baseline
        out_path = args.out or f'eval_{args.baseline}.json'

    suite = run_suite(policy, default_env_factory(env_kwargs),
                      episodes_per_family=args.episodes,
                      eval_seed0=args.eval_seed0)
    print(f'== {label}  (task={env_kwargs.get("task", "follow")}, '
          f'{args.episodes} eps/family, frozen seeds from {args.eval_seed0})')
    _print_table(suite)
    with open(out_path, 'w') as f:
        json.dump({'label': label, 'env': env_kwargs, 'suite': suite}, f,
                  indent=2)
    print(f'-> {out_path}')


if __name__ == '__main__':
    main()
