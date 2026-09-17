"""The matrix report: IQM + stratified bootstrap CIs per arm on the
HELD-OUT numbers, and the guide-28.4 acceptance verdict.

    python3 -m chase_train.report_matrix --runs runs/ [--episodes 10]

This is the piece the recipe (stage 3: "Report IQM + stratified bootstrap
CIs") and guide 28.4 ("beat the P-controller ... by more than the seed
spread") demand and run_matrix alone did not produce. It reads every run
directory under --runs, groups by arm, pools each run's final_heldout
per-episode-family scores, runs the P-controller on the SAME held-out seed
block, and prints per-arm IQM [CI] plus the acceptance verdict:

  accept arm X  iff  IQM_lo(X) > IQM_hi(P-controller)  on the moving
  families AND IQM(X) >= IQM(P) - epsilon on static (guide 28.4 items
  1-2). If no arm clears: SHIP THE P-CONTROLLER, and say so (exam Q20).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from chase_gym import ChaseEnv, EnvConfig, PController
from chase_eval.evaluate import default_env_factory, run_suite
from chase_eval.stats import iqm, stratified_bootstrap_ci

MOVING = ('constant_velocity', 'vertical_oscillation',
          'horizontal_oscillation', 'aggressive')


def collect_runs(root: str):
    """{arm: {seed: heldout_record}} from every metrics.jsonl under root."""
    out = defaultdict(dict)
    for entry in sorted(os.listdir(root)):
        mpath = os.path.join(root, entry, 'metrics.jsonl')
        cpath = os.path.join(root, entry, 'config.json')
        if not (os.path.isfile(mpath) and os.path.isfile(cpath)):
            continue
        with open(cpath) as f:
            cfg = json.load(f)['config']
        heldout = None
        with open(mpath) as f:
            for line in f:
                rec = json.loads(line)
                if 'final_heldout' in rec:
                    heldout = rec
        if heldout is not None:
            out[cfg['arm']][cfg['seed']] = {'record': heldout, 'config': cfg}
    return out


def family_scores(heldout: dict, metric: str, families) -> list:
    fams = heldout['final_heldout'].get('families', {})
    return [fams[f][metric] for f in families if f in fams]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--episodes', type=int, default=10,
                    help='episodes/family for the P-controller pass')
    ap.add_argument('--epsilon', type=float, default=0.01,
                    help='static-family tolerance (tv fraction)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    runs = collect_runs(args.runs)
    if not runs:
        print(f'no completed runs with final_heldout under {args.runs}',
              file=sys.stderr)
        return 2

    # One env config + held-out seed block, taken from the first ChaseEnv
    # arm found (all arms share the env by construction; faithful is
    # reported but never enters the flight comparison -- it lives in the
    # point-mass world).
    ref = None
    for arm, seeds in runs.items():
        if arm != 'ddpg_faithful':
            ref = next(iter(seeds.values()))
            break
    report = {'arms': {}, 'runs_root': args.runs}
    baseline_by_family = None
    if ref is not None:
        env_kwargs = ref['config']['env']
        heldout_seed0 = ref['record']['heldout_seed0']
        probe = ChaseEnv(EnvConfig(**env_kwargs))
        psuite = run_suite(PController(probe.obs_spec),
                           default_env_factory(env_kwargs),
                           episodes_per_family=args.episodes,
                           eval_seed0=heldout_seed0)
        baseline_by_family = psuite['families']
        report['p_controller'] = {
            f: {'time_in_view': s['time_in_view'],
                'return_mean': s['return_mean']}
            for f, s in baseline_by_family.items()}

    print(f"{'arm':<16}{'seeds':>6}{'tv IQM':>10}{'tv CI':>18}"
          f"{'ret IQM':>10}")
    accepted = []
    for arm, seeds in sorted(runs.items()):
        per_seed_tv = {s: family_scores(v['record'], 'time_in_view',
                                        MOVING) or
                       [v['record']['final_heldout']['aggregate']
                        ['time_in_view']]
                       for s, v in seeds.items()}
        per_seed_ret = {s: family_scores(v['record'], 'return_mean', MOVING)
                        or [v['record']['final_heldout']['aggregate']
                            ['return_mean']]
                        for s, v in seeds.items()}
        tv_iqm, tv_lo, tv_hi = stratified_bootstrap_ci(per_seed_tv)
        ret_iqm = iqm([x for xs in per_seed_ret.values() for x in xs])
        entry = {'seeds': sorted(seeds), 'moving_tv_iqm': tv_iqm,
                 'moving_tv_ci': [tv_lo, tv_hi],
                 'moving_return_iqm': ret_iqm}
        print(f"{arm:<16}{len(seeds):>6}{tv_iqm:>10.3f}"
              f"{f'[{tv_lo:.3f},{tv_hi:.3f}]':>18}{ret_iqm:>10.2f}")

        if baseline_by_family is not None and arm != 'ddpg_faithful':
            p_moving = iqm([baseline_by_family[f]['time_in_view']
                            for f in MOVING if f in baseline_by_family])
            static_arm = iqm([x for s, v in seeds.items()
                              for x in family_scores(v['record'],
                                                     'time_in_view',
                                                     ('static',))])
            p_static = baseline_by_family.get('static', {}) \
                .get('time_in_view', float('nan'))
            beats_moving = (tv_lo == tv_lo and tv_lo > p_moving)
            matches_static = static_arm >= p_static - args.epsilon
            entry.update({'p_moving_tv_iqm': p_moving,
                          'static_tv_iqm': static_arm,
                          'p_static_tv': p_static,
                          'beats_p_on_moving_beyond_spread': beats_moving,
                          'matches_p_on_static': matches_static})
            if beats_moving and matches_static:
                accepted.append(arm)
        report['arms'][arm] = entry

    if baseline_by_family is not None:
        if accepted:
            verdict = (f"ACCEPT {', '.join(accepted)}: CI lower bound "
                       f"beats the P-controller on moving families and "
                       f"matches it on static (guide 28.4)")
        else:
            verdict = ('NO ARM CLEARS guide 28.4 -> SHIP THE '
                       'P-CONTROLLER and report it (exam Q20)')
            if any(len(e['seeds']) < 2 for e in report['arms'].values()):
                verdict += (' [note: arms with <2 seeds cannot clear -- '
                            'the CI needs seed replication; run the full '
                            'matrix]')
        report['verdict'] = verdict
        print('\n' + verdict)
    out_path = args.out or os.path.join(args.runs, 'matrix_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'-> {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
