"""The section 16.3 validation gate: replay a real flight through the sim.

"An unvalidated simulator invalidates everything downstream; this gate is
not skippable." (rl_training_guide.md 16.3)

PROTOCOL [D-impl]: the validation flight uses a STATIC target (hovering or
mounted) while the follower is flown manually. That choice is what makes
the gate non-circular: the follower's own pose is never measured (no
mocap), so a moving target reconstructed through replayed states would be
compared against itself. With a static target, its one world position is
estimated from the EARLIEST detections -- where the replayed state is
still the known initial hover pose regardless of dynamics parameters --
and every later frame then tests the integrated dynamics against the
recorded YOLO track on the DETECTION timeline (timestamp-matched, never
index-matched: detections arrive at ~25-30 Hz, commands at 10-20 Hz).

Inputs:
  --detections  chase_detector per-frame CSV (csv_log.py schema);
  --commands    CSV of the commands sent during the same flight, columns:
                t_ns, a_v, a_h, v_fwd  (sticks in [-1,1], v_fwd m/s) --
                exported from the rosbag of /cmd_vel;
  --t-lag       the measured lag constant under test.

PASS when the median pixel error is within the detector jitter band
(--tolerance-px). `--self-test` runs the REAL pipeline on a synthetic
flight and asserts both directions: correct T_lag passes, a grossly wrong
T_lag scores worse.
"""
import argparse
import bisect
import csv
import json
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from chase_gym import constants as C
from chase_gym.kinematics import (FollowerKinematics, FollowerState, project,
                                  world_to_camera)


@dataclass
class Snapshot:
    t: float
    pos: np.ndarray
    yaw: float

    def state(self) -> FollowerState:
        return FollowerState(pos=self.pos.copy(), yaw=self.yaw)


def load_detections(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row.get('detected', '0') or 0):
                rows.append({'t': int(row['t_capture_ns']) * 1e-9,
                             'cx': float(row['cx']), 'cy': float(row['cy']),
                             'w': float(row['w'])})
    return rows


def load_commands(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({'t': int(row['t_ns']) * 1e-9,
                         'a_v': float(row['a_v']), 'a_h': float(row['a_h']),
                         'v_fwd': float(row.get('v_fwd', 0.0))})
    return rows


def integrate_states(commands: List[dict], t_lag: float, v_max: float,
                     omega_max: float, n_sub: int = 10) -> List[Snapshot]:
    """Replay the command log through the Tier-A follower kinematics,
    returning DEEP-COPIED snapshots (the kinematics object mutates one
    state in place -- aliasing it was the original sin this gate shipped
    with). Each command interval is integrated in n_sub substeps and a
    snapshot stored per substep: the aircraft is continuous, and coarse
    Euler at the command period plus zero-order-hold state lookup at the
    30 Hz detection times costs >10 px against a 6 px tolerance."""
    fk = FollowerKinematics(t_lag)
    fk.reset()
    t0 = commands[0]['t']
    snaps = [Snapshot(0.0, fk.state.pos.copy(), fk.state.yaw)]
    for i in range(1, len(commands)):
        dt = commands[i]['t'] - commands[i - 1]['t']
        if not (0.0 < dt < 1.0):
            continue
        c = commands[i - 1]
        t_base = commands[i - 1]['t'] - t0
        for j in range(1, n_sub + 1):
            fk.step(c['v_fwd'], c['a_v'] * v_max, c['a_h'] * omega_max,
                    dt / n_sub)
            snaps.append(Snapshot(t_base + j * dt / n_sub,
                                  fk.state.pos.copy(), fk.state.yaw))
    return snaps


def state_at(snaps: List[Snapshot], times: List[float], t: float
             ) -> Snapshot:
    """Newest snapshot with timestamp <= t (bisect on the shared times)."""
    i = bisect.bisect_right(times, t) - 1
    return snaps[max(i, 0)]


def back_project(det: dict, st: FollowerState) -> np.ndarray:
    """One detection + follower state -> target world position (pinhole,
    our calibration)."""
    z = C.FX * C.TELLO_BODY_W / max(det['w'], 1.0)
    x_cam = (det['cx'] - C.CX) * z / C.FX
    y_cam = (det['cy'] - C.CY) * z / C.FY
    body = np.array([z, -x_cam, -y_cam])
    cy_, sy_ = math.cos(st.yaw), math.sin(st.yaw)
    return st.pos + np.array([cy_ * body[0] - sy_ * body[1],
                              sy_ * body[0] + cy_ * body[1], body[2]])


def validate(dets: List[dict], cmds: List[dict], t_lag: float,
             v_max: float, omega_max: float,
             n_anchor: int = 10) -> dict:
    """The gate computation. Returns per-frame errors and the estimate."""
    t0 = cmds[0]['t']
    dets = [dict(d, t=d['t'] - t0) for d in dets if d['t'] >= t0]
    snaps = integrate_states(cmds, t_lag, v_max, omega_max)
    times = [s.t for s in snaps]

    # Static-target estimate from the earliest frames, where the replayed
    # state is ~the initial pose whatever the dynamics constants are.
    anchors = dets[:n_anchor]
    est = np.median(np.stack([
        back_project(d, state_at(snaps, times, d['t']).state())
        for d in anchors]), axis=0)

    errors = []
    for d in dets:
        st = state_at(snaps, times, d['t']).state()
        u, v, w_px, in_frame = project(world_to_camera(est, st))
        errors.append(math.hypot(u - d['cx'], v - d['cy']))
    return {'target_estimate': est.tolist(), 'errors_px': errors}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detections')
    ap.add_argument('--commands')
    ap.add_argument('--t-lag', type=float, default=0.25)
    ap.add_argument('--v-max', type=float, default=1.5)
    ap.add_argument('--omega-max', type=float, default=1.5)
    ap.add_argument('--tolerance-px', type=float, default=6.0)
    ap.add_argument('--out', default='replay_validation.json')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args.tolerance_px)
    if not (args.detections and args.commands):
        ap.error('--detections and --commands are required '
                 '(or use --self-test)')

    dets = load_detections(args.detections)
    cmds = load_commands(args.commands)
    if len(dets) < 50 or len(cmds) < 50:
        print(f'not enough data: {len(dets)} detections, {len(cmds)} '
              f'commands', file=sys.stderr)
        return 2

    res = validate(dets, cmds, args.t_lag, args.v_max, args.omega_max)
    err = res['errors_px']
    median = float(np.median(err))
    p90 = float(np.percentile(err, 90))
    ok = median <= args.tolerance_px
    result = {'frames': len(err), 'median_px': median, 'p90_px': p90,
              'tolerance_px': args.tolerance_px, 'pass': ok,
              't_lag': args.t_lag,
              'target_estimate_m': res['target_estimate'],
              'protocol': 'static-target [D-impl]'}
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'replay validation: median {median:.2f} px, p90 {p90:.2f} px '
          f'over {len(err)} frames -> {"PASS" if ok else "FAIL"} '
          f'({args.out})')
    return 0 if ok else 1


def _synthetic_flight(t_lag_true: float, seed: int = 0
                      ) -> Tuple[List[dict], List[dict]]:
    """A ground-truth flight for the self-test: known static target, known
    dynamics, sinusoid stick commands at 10 Hz; detections at 30 Hz with
    2 px jitter -- the DIFFERENT rates the timestamp matching must handle."""
    rng = np.random.default_rng(seed)
    target = np.array([2.0, 0.3, 1.2])
    fk = FollowerKinematics(t_lag_true)
    fk.reset()
    cmds, dets = [], []
    det_dt, cmd_dt = 1.0 / 30.0, 0.1
    t, next_det = 0.0, 0.0
    for k in range(200):
        a_v = 0.3 * math.sin(0.7 * t)
        a_h = 0.25 * math.sin(0.4 * t + 1.0)
        v_fwd = 0.2 * math.sin(0.3 * t)
        cmds.append({'t': t, 'a_v': a_v, 'a_h': a_h, 'v_fwd': v_fwd})
        # integrate the "real" aircraft at fine steps, emitting detections
        for _ in range(10):
            fk.step(v_fwd, a_v * 1.5, a_h * 1.5, cmd_dt / 10.0)
            t += cmd_dt / 10.0
            if t >= next_det:
                from chase_gym.kinematics import world_to_camera
                u, v, w, in_frame = project(
                    world_to_camera(target, fk.state))
                if in_frame:
                    dets.append({'t': t,
                                 'cx': u + rng.normal(0, 2.0),
                                 'cy': v + rng.normal(0, 2.0),
                                 'w': max(w + rng.normal(0, 1.0), 1.0)})
                next_det += det_dt
    return dets, cmds


def self_test(tolerance_px: float) -> int:
    """Runs the REAL validate() pipeline on a synthetic flight, both
    directions: the true T_lag must pass; a 3x-wrong T_lag must be
    measurably worse (the gate detects dynamics error, not just noise)."""
    t_true = 0.25
    dets, cmds = _synthetic_flight(t_true)
    good = validate(dets, cmds, t_true, 1.5, 1.5)
    bad = validate(dets, cmds, 3.0 * t_true, 1.5, 1.5)
    m_good = float(np.median(good['errors_px']))
    m_bad = float(np.median(bad['errors_px']))
    ok = m_good <= tolerance_px and m_bad > 1.5 * m_good
    print(f'[self-test] {len(dets)} detections / {len(cmds)} commands; '
          f'median @true T_lag {m_good:.2f} px (tol {tolerance_px}), '
          f'@3x T_lag {m_bad:.2f} px -> '
          f'{"PASS" if ok else "FAIL"} (gate separates good from bad '
          f'dynamics)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
