"""The arms x seeds matrix, scripted -- never manual (guide 24).

    python3 -m chase_train.run_matrix --arms td3 ddpg_repaired ddpg_faithful \
        --seeds 0 1 2 3 4 --jobs 2

Each cell is a subprocess of chase_train.train with its own run directory;
a matrix.json index is written at the end. --jobs parallelises across
processes (each run is single-threaded-torch friendly at this model size).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

def _config_dir() -> str:
    """Source tree first; installed layout (share/chase_train/config via
    colcon/ament prefixes) second -- site-packages/../config does not
    exist when the package is installed."""
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'config')
    if os.path.isdir(src):
        return src
    prefixes = [sys.prefix] + \
        os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep)
    for prefix in filter(None, prefixes):
        cand = os.path.join(prefix, 'share', 'chase_train', 'config')
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        'chase_train config dir not found in the source tree or any '
        'AMENT_PREFIX_PATH share directory')


CONFIG_DIR = _config_dir()


def run_cell(arm: str, seed: int, extra_overrides, out_root: str) -> dict:
    if arm == 'sb3_check':
        config = os.path.join(CONFIG_DIR, 'td3.yaml')
        cmd = [sys.executable, '-m', 'chase_train.sb3_check', '--config',
               config, '--seed', str(seed)]
    else:
        config = os.path.join(CONFIG_DIR, f'{arm}.yaml')
        cmd = [sys.executable, '-m', 'chase_train.train', '--config', config,
               '--seed', str(seed)]
    overrides = list(extra_overrides) + [f'out_root={out_root}']
    cmd += ['--override'] + overrides
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    run_dir = ''
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith('[done] run dir: '):
            run_dir = line.split(': ', 1)[1]
            break
    if not ok:
        sys.stderr.write(f'--- {arm} s{seed} FAILED ---\n{proc.stderr[-2000:]}\n')
    return {'arm': arm, 'seed': seed, 'ok': ok, 'run_dir': run_dir,
            'wall_s': round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', nargs='+',
                    default=['td3', 'ddpg_repaired', 'ddpg_faithful'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--out-root', default='runs')
    ap.add_argument('--override', nargs='*', default=[])
    ap.add_argument('--with-sb3', action='store_true',
                    help='add the SB3 TD3 cross-check per seed '
                         '(recipe stage 3: the 4th arm)')
    ap.add_argument('--report', action='store_true', default=True,
                    help='aggregate IQM/CI + 28.4 acceptance at the end')
    args = ap.parse_args(argv)

    cells = [(arm, seed) for arm in args.arms for seed in args.seeds]
    if args.with_sb3:
        cells += [('sb3_check', seed) for seed in args.seeds]
    print(f'matrix: {len(cells)} runs ({len(args.arms)} arms x '
          f'{len(args.seeds)} seeds), jobs={args.jobs}')
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_cell, arm, seed, args.override,
                               args.out_root) for arm, seed in cells]
        for fut in futures:
            res = fut.result()
            results.append(res)
            print(f"[matrix] {res['arm']} s{res['seed']}: "
                  f"{'ok' if res['ok'] else 'FAILED'} ({res['wall_s']}s) "
                  f"{res['run_dir']}")
    index_path = os.path.join(args.out_root, 'matrix.json')
    os.makedirs(args.out_root, exist_ok=True)
    with open(index_path, 'w') as f:
        json.dump(results, f, indent=2)
    failed = [r for r in results if not r['ok']]
    print(f'matrix complete: {len(results) - len(failed)}/{len(results)} ok; '
          f'index: {index_path}')
    if args.report and len(results) > len(failed):
        from .report_matrix import main as report_main
        report_main(['--runs', args.out_root])
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
