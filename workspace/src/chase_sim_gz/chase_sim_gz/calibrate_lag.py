"""Step-response calibration (sim_training_architecture.md 3.2 procedure).

Commands a velocity step on each axis, records the achieved velocity from
odometry at every control period, fits the first-order time constant
T = -dt / ln(1 - v_k/v_cmd) averaged over the rise, and compares against
the target (the MEASURED Phase-1 T_lag once it exists; the Tier-A
placeholder until then). Tune motor time constants + velocity-controller
gains in gen_world.py until sim matches measurement within ~10%, then
randomise the residual +/-30% per episode in the env config.

    python3 -m chase_sim_gz.calibrate_lag [--fake] [--target-t-lag 0.25]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from .gz_iface import add_backend_args, make_backend

AXES = {
    'forward': ((1.0, 0.0, 0.0), 0.0),
    'vertical': ((0.0, 0.0, 1.0), 0.0),
    'yaw': ((0.0, 0.0, 0.0), 1.0),
}
STEP_CMD = 0.5          # m/s or rad/s step
CONTROL_DT = 0.1
SETTLE_S = 1.0
RISE_S = 2.0


def achieved(backend, axis: str) -> float:
    od = backend.get_odom('follower')
    if axis == 'forward':
        return float(od.lin_body[0])
    if axis == 'vertical':
        return float(od.lin_body[2])
    return float(od.yaw_rate)


def fit_t_lag(times, vels, v_cmd: float) -> float:
    """Least-squares fit of v(t) = v_cmd*(1-exp(-t/T)) on the rise."""
    ts, ys = [], []
    for t, v in zip(times, vels):
        frac = v / v_cmd
        if 0.05 < frac < 0.95:
            ts.append(t)
            ys.append(-math.log(1.0 - frac))
    if len(ts) < 2:
        return float('nan')
    ts, ys = np.asarray(ts), np.asarray(ys)
    # y = t / T  ->  1/T from a through-origin least squares.
    return float((ts @ ts) / (ts @ ys))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_backend_args(ap)
    ap.add_argument('--target-t-lag', type=float, default=0.25,
                    help='the measured Phase-1 T_lag (placeholder 0.25 s '
                         'until the step-response flight exists)')
    ap.add_argument('--out', default='lag_calibration.json')
    args = ap.parse_args(argv)

    backend = make_backend(args)
    backend.start()

    results = {}
    try:
        for axis, (lin, yaw) in AXES.items():
            backend.reset_world()
            backend.set_pose('follower', (0.0, 0.0, 1.5), 0.0)
            backend.send_twist('follower', (0.0, 0.0, 0.0), 0.0)
            backend.step(int(SETTLE_S / backend.physics_dt))

            cmd_lin = tuple(STEP_CMD * c for c in lin)
            backend.send_twist('follower', cmd_lin, STEP_CMD * yaw)
            times, vels = [], []
            n = int(RISE_S / CONTROL_DT)
            for k in range(1, n + 1):
                backend.step(int(CONTROL_DT / backend.physics_dt))
                times.append(k * CONTROL_DT)
                vels.append(achieved(backend, axis))
            t_fit = fit_t_lag(times, vels, STEP_CMD)
            final = vels[-1] / STEP_CMD
            dev = (t_fit - args.target_t_lag) / args.target_t_lag \
                if t_fit == t_fit else float('nan')
            results[axis] = {'t_lag_fit_s': t_fit, 'final_fraction': final,
                             'deviation_vs_target': dev}
            print(f'  {axis:<9} T_lag {t_fit:6.3f} s   final {final:5.1%} '
                  f'of command   vs target {args.target_t_lag}: '
                  f'{dev:+.0%}' if t_fit == t_fit else
                  f'  {axis:<9} fit failed (response never rose)')
    finally:
        backend.stop()

    ok = all(r['t_lag_fit_s'] == r['t_lag_fit_s']
             and abs(r['deviation_vs_target']) <= 0.10 for r in results.values())
    out = {'target_t_lag_s': args.target_t_lag, 'axes': results,
           'within_10pct': ok}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(('calibration WITHIN 10% of target -> ' if ok else
           'calibration OUTSIDE 10% -- tune gen_world.py constants '
           '(timeConstantUp/Down, velocityGain) and rerun -> ') + args.out)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
