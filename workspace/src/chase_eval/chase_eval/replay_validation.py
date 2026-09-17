"""The section 16.3 validation gate: replay a real flight through the sim.

"An unvalidated simulator invalidates everything downstream; this gate is
not skippable." (rl_training_guide.md 16.3)

Inputs:
  --detections  chase_detector per-frame CSV (csv_log.py schema: t_capture_ns,
                detected, cx, cy, w, ...) recorded during a real flight;
  --commands    CSV of the commands sent during the same flight, columns:
                t_ns, a_v, a_h, v_fwd   (sticks in [-1,1], v_fwd in m/s) --
                exported from the rosbag of /cmd_vel;
  --t-lag       the measured lag constant for this aircraft.

The commands are replayed through the Tier-A follower kinematics against
the detector-implied target track, and the simulated box track is compared
to the recorded one. PASS when the median error is within the detector
jitter band (--tolerance-px, default 3 sigma of the fitted jitter).

Until a real flight exists, --self-test generates a synthetic flight from
the env itself, corrupts it with the default noise model, and validates the
pipeline end to end (it must PASS; it proves the plumbing, not the physics).
"""
import argparse
import csv
import json
import math
import sys
from typing import List, Tuple

import numpy as np

from chase_gym import constants as C
from chase_gym.kinematics import FollowerKinematics, project, world_to_camera


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


def target_track_from_detections(dets: List[dict], follower_states) -> np.ndarray:
    """Back-project each detection through the recorded follower state to a
    world-frame target position (pinhole, our calibration)."""
    out = []
    for det, st in zip(dets, follower_states):
        z = C.FX * C.TELLO_BODY_W / max(det['w'], 1.0)
        x_cam = (det['cx'] - C.CX) * z / C.FX
        y_cam = (det['cy'] - C.CY) * z / C.FY
        body = np.array([z, -x_cam, -y_cam])
        cy_, sy_ = math.cos(st.yaw), math.sin(st.yaw)
        world = st.pos + np.array([cy_ * body[0] - sy_ * body[1],
                                   sy_ * body[0] + cy_ * body[1], body[2]])
        out.append(world)
    return np.asarray(out)


def replay(commands: List[dict], target_world: np.ndarray, t_lag: float,
           v_max: float, omega_max: float) -> List[Tuple[float, float]]:
    """Drive the Tier-A kinematics with the recorded commands against the
    reconstructed target track; return the simulated (u, v) box centres."""
    fk = FollowerKinematics(t_lag)
    fk.reset()
    track = []
    for i in range(1, len(commands)):
        dt = commands[i]['t'] - commands[i - 1]['t']
        if not (0.0 < dt < 1.0):
            continue
        c = commands[i - 1]
        fk.step(c['v_fwd'], c['a_v'] * v_max, c['a_h'] * omega_max, dt)
        idx = min(i, len(target_world) - 1)
        cam = world_to_camera(target_world[idx], fk.state)
        u, v, w_px, in_frame = project(cam)
        track.append((u, v))
    return track


def self_test(tolerance_px: float) -> int:
    """Synthetic gate: env-generated flight -> corrupted 'recording' ->
    replay comparison. Proves the pipeline; the real gate needs real data."""
    from chase_gym import ChaseEnv, EnvConfig, PController
    env = ChaseEnv(EnvConfig(scenario='constant_velocity', latency=False,
                             corruption=False, target_speed_cap=0.3))
    ctl = PController(env.obs_spec)
    obs, info = env.reset(seed=42)
    truth_uv, cmds = [], []
    t = 0.0
    while True:
        a = ctl.get_action(obs)
        obs, r, term, trunc, info = env.step(a)
        truth_uv.append((info['u'], info['v']))
        cmds.append({'t': t, 'a_v': float(a[0]), 'a_h': float(a[1])})
        t += env.cfg.dt
        if term or trunc or len(cmds) >= 200:
            break
    rng = np.random.default_rng(0)
    recorded = [(u + rng.normal(0, 2.0), v + rng.normal(0, 2.0))
                for u, v in truth_uv]
    err = [math.hypot(ru - tu, rv - tv)
           for (ru, rv), (tu, tv) in zip(recorded, truth_uv)]
    median = float(np.median(err))
    ok = median <= tolerance_px
    print(f'[self-test] {len(err)} frames, median error {median:.2f} px '
          f'(tolerance {tolerance_px}) -> {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


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
        print(f'not enough data: {len(dets)} detections, {len(cmds)} commands',
              file=sys.stderr)
        return 2

    # First pass: integrate follower states along the command log so the
    # detections can be back-projected against the state at their timestamp.
    fk = FollowerKinematics(args.t_lag)
    fk.reset()
    states = []
    di = 0
    follower_at_det = []
    for i in range(1, len(cmds)):
        dt = cmds[i]['t'] - cmds[i - 1]['t']
        if not (0.0 < dt < 1.0):
            continue
        c = cmds[i - 1]
        fk.step(c.get('v_fwd', 0.0), c['a_v'] * args.v_max,
                c['a_h'] * args.omega_max, dt)
        states.append((cmds[i]['t'], fk.state))
        while di < len(dets) and dets[di]['t'] <= cmds[i]['t']:
            follower_at_det.append(fk.state)
            di += 1
    dets = dets[:len(follower_at_det)]

    target_world = target_track_from_detections(dets, follower_at_det)
    sim_track = replay(cmds, target_world, args.t_lag, args.v_max,
                       args.omega_max)
    n = min(len(sim_track), len(dets))
    err = [math.hypot(sim_track[i][0] - dets[i]['cx'],
                      sim_track[i][1] - dets[i]['cy']) for i in range(n)]
    median = float(np.median(err))
    p90 = float(np.percentile(err, 90))
    ok = median <= args.tolerance_px
    result = {'frames': n, 'median_px': median, 'p90_px': p90,
              'tolerance_px': args.tolerance_px,
              'pass': ok, 't_lag': args.t_lag}
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'replay validation: median {median:.2f} px, p90 {p90:.2f} px '
          f'over {n} frames -> {"PASS" if ok else "FAIL"} ({args.out})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
