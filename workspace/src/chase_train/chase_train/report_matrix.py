"""The matrix report: per-arm IQM + stratified bootstrap CIs on the
HELD-OUT per-episode scores, and the guide-28.4 acceptance verdict.

    python3 -m chase_train.report_matrix --runs runs/ [--episodes 10]

This is the piece the recipe (stage 3: "Report IQM + stratified bootstrap
CIs") and guide 28.4 ("beat the P-controller ... by more than the seed
spread") demand. Statistics, stated precisely:

* Each ChaseEnv arm's population = the PER-EPISODE moving-family
  time-in-view scores from every seed's final_heldout (episode
  replication, not family means: a bootstrap over four fixed family means
  would measure design heterogeneity, not sampling variance). The CI is
  the stratified bootstrap over seeds, then episodes within seed.
* The P-controller baseline runs on the SAME held-out seed block and gets
  its own bootstrap CI over its per-episode scores -- comparing an arm's
  CI lower bound against a noisy baseline point estimate is not "beyond
  the seed spread".
* accept arm X  iff  tv_lo(X) > tv_hi(P) on the moving families AND
  IQM_static(X) >= IQM_static(P) - epsilon (guide 28.4 items 1-2).
  If no arm clears: SHIP THE P-CONTROLLER, and say so (exam Q20).
* The faithful arm lives in the point-mass world: reported in its own
  row with n/a moving columns, never compared across worlds.

The moving-family set and the selection metric are IMPORTED from
chase_eval.evaluate (selection and acceptance judge the same population).
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from chase_gym import ChaseEnv, EnvConfig, PController
from chase_eval.evaluate import (MOVING_FAMILIES, default_env_factory,
                                 run_suite)
from chase_eval.stats import bootstrap_ci, iqm, stratified_bootstrap_ci


def collect_runs(root: str):
    """{arm: {seed: {'record', 'config'}}} from metrics.jsonl under root."""
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


def episode_scores(heldout: dict, families) -> list:
    """Pooled per-episode tv over `families`; [] when absent (faithful, or
    pre-per-episode-suite records)."""
    fams = heldout['final_heldout'].get('families', {})
    return [x for f in families if f in fams
            for x in fams[f].get('tv_episodes', [])]


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

    # Env config + held-out seed block from the first ChaseEnv arm.
    ref = None
    for arm, seeds in runs.items():
        if arm != 'ddpg_faithful':
            ref = next(iter(seeds.values()))
            break
    report = {'arms': {}, 'runs_root': args.runs}
    p_stats = None
    if ref is not None:
        env_kwargs = ref['config']['env']
        heldout_seed0 = ref['record']['heldout_seed0']
        probe = ChaseEnv(EnvConfig(**env_kwargs))
        psuite = run_suite(PController(probe.obs_spec),
                           default_env_factory(env_kwargs),
                           episodes_per_family=args.episodes,
                           eval_seed0=heldout_seed0)
        p_moving_eps = [x for f in MOVING_FAMILIES
                        for x in psuite['families'][f]['tv_episodes']]
        # The baseline has no seed axis, only episode sampling noise.
        p_iqm, p_lo, p_hi = bootstrap_ci(p_moving_eps)
        p_static = iqm(psuite['families']['static']['tv_episodes'])
        p_stats = {'moving_iqm': p_iqm, 'moving_ci': [p_lo, p_hi],
                   'static_iqm': p_static}
        report['p_controller'] = dict(
            p_stats, families={f: s['time_in_view']
                               for f, s in psuite['families'].items()})

    print(f"{'arm':<16}{'seeds':>6}{'tv IQM':>10}{'tv CI':>18}"
          f"{'ret IQM':>10}")
    accepted = []
    for arm, seeds in sorted(runs.items()):
        if arm == 'ddpg_faithful':
            # Point-mass world: survival-fraction aggregate, no moving
            # families -- own row, never in the flight comparison.
            aggs = [v['record']['final_heldout']['aggregate']
                    for v in seeds.values()]
            report['arms'][arm] = {
                'seeds': sorted(seeds), 'world': 'point-mass (own scale)',
                'survival_iqm': iqm([a['time_in_view'] for a in aggs]),
                'return_iqm': iqm([a['return_mean'] for a in aggs])}
            print(f"{arm:<16}{len(seeds):>6}{'n/a':>10}{'(point-mass)':>18}"
                  f"{report['arms'][arm]['return_iqm']:>10.2f}")
            continue

        per_seed_tv = {s: episode_scores(v['record'], MOVING_FAMILIES)
                       for s, v in seeds.items()}
        per_seed_tv = {s: xs for s, xs in per_seed_tv.items() if xs}
        if not per_seed_tv:
            print(f"{arm:<16}{len(seeds):>6}{'no data':>10}")
            continue
        per_seed_ret = {s: [x for f in MOVING_FAMILIES
                            if f in v['record']['final_heldout']['families']
                            for x in v['record']['final_heldout']
                            ['families'][f].get('return_episodes', [])]
                        for s, v in seeds.items()}
        tv_iqm, tv_lo, tv_hi = stratified_bootstrap_ci(per_seed_tv)
        ret_iqm = iqm([x for xs in per_seed_ret.values() for x in xs])
        entry = {'seeds': sorted(seeds), 'moving_tv_iqm': tv_iqm,
                 'moving_tv_ci': [tv_lo, tv_hi],
                 'moving_return_iqm': ret_iqm}
        ci_txt = 'n/a (needs >=2 seeds)' if math.isnan(tv_lo) \
            else f'[{tv_lo:.3f},{tv_hi:.3f}]'
        print(f"{arm:<16}{len(per_seed_tv):>6}{tv_iqm:>10.3f}"
              f"{ci_txt:>18}{ret_iqm:>10.2f}")

        if p_stats is not None:
            static_eps = [x for s, v in seeds.items()
                          for x in episode_scores(v['record'], ('static',))]
            static_arm = iqm(static_eps) if static_eps else float('nan')
            beats_moving = (not math.isnan(tv_lo)
                            and tv_lo > p_stats['moving_ci'][1])
            matches_static = (not math.isnan(static_arm)
                              and static_arm >= p_stats['static_iqm']
                              - args.epsilon)
            entry.update({'static_tv_iqm': static_arm,
                          'beats_p_on_moving_beyond_spread': beats_moving,
                          'matches_p_on_static': matches_static})
            if beats_moving and matches_static:
                accepted.append(arm)
        report['arms'][arm] = entry

    if p_stats is not None:
        p_ci_txt = (f"[{p_stats['moving_ci'][0]:.3f},"
                    f"{p_stats['moving_ci'][1]:.3f}]")
        print(f"{'p_controller':<16}{'--':>6}"
              f"{p_stats['moving_iqm']:>10.3f}{p_ci_txt:>18}")
        if accepted:
            verdict = (f"ACCEPT {', '.join(accepted)}: CI lower bound "
                       f"beats the P-controller's CI upper bound on the "
                       f"moving families and matches it on static "
                       f"(guide 28.4)")
        else:
            verdict = ('NO ARM CLEARS guide 28.4 -> SHIP THE '
                       'P-CONTROLLER and report it (exam Q20)')
            if any(math.isnan(e.get('moving_tv_ci', [float('nan')])[0])
                   for e in report['arms'].values()
                   if 'moving_tv_ci' in e):
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
