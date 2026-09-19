"""Tier-C sign test -- the section 4.4 protocol against a LIVE engine.

Tier B has chase_sim_gz/sign_test.py; this is its Tier-C counterpart: the
REP-103 -> NED yaw negation in engine.py is convention-derived, and the
standing doc-set rule is that sign conventions are TESTED per tier, never
assumed (three simulators + one aircraft = four chances for a silent
flip). Run on the engine host after the scene is up:

    python3 tier_c_sign_test.py --engine projectairsim

Protocol (one axis at a time, image-space assertions via the engine's own
bbox annotations -- no YOLO in the loop):
    +yaw rate (REP-103 CCW) -> target bbox centre moves RIGHT (+u)
    +v_up (stick up)        -> target bbox centre moves DOWN  (+v)
    +v_fwd                  -> bbox width GROWS
A FAIL means the adapter's negation is wrong for this engine build: fix it
in engine.py's _ned_yaw_rate helper and record the result next to the
checkpoints -- never patch signs downstream.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import make_engine  # noqa: E402

SETTLE_S = 1.0
PUSH_S = 1.6
TICK_S = 0.2
RANGE_M = 3.0


def bbox_centre(eng):
    eng.get_rgb()
    boxes = eng.get_bboxes()
    if not boxes:
        return None
    b = boxes[0]
    return ((b.xmin + b.xmax) / 2.0, (b.ymin + b.ymax) / 2.0, b.w)


def fmt(b, i):
    return f'{b[i]:.1f}' if b is not None else 'none'


def settle(eng):
    """Re-centre the TARGET on the vehicle's current camera axis. The
    vehicle is NEVER hovered or teleported here: a hover re-parks
    simple_flight so the next yaw command is silently dropped (measured on
    Blocks 1.0.1), and teleporting a flying physics vehicle does not stick.
    The sign checks only need the relative geometry, so we place the target
    on the live axis and let the vehicle keep flying."""
    (x, y, z), yaw = eng.get_vehicle_pose()
    eng.set_object_pose('TargetTello',
                        (x + RANGE_M * math.cos(yaw),
                         y + RANGE_M * math.sin(yaw), z), yaw)
    time.sleep(SETTLE_S)


def push(eng, v_fwd=0.0, v_up=0.0, yaw_rate=0.0):
    """Drive CONTINUOUSLY in short ticks, exactly as the vision-in-loop
    eval does -- a single isolated command is unreliable for yaw on
    simple_flight, but the continuous closed-loop pattern actuates it (a
    P-controller centres a 0.35 rad off-axis target this way). No hover: it
    would re-park the controller."""
    b0 = bbox_centre(eng)
    n = max(1, int(round(PUSH_S / TICK_S)))
    for _ in range(n):
        eng.move_by_velocity_body(v_fwd, 0.0, -v_up, yaw_rate, TICK_S)
    b1 = bbox_centre(eng)
    return b0, b1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', default='projectairsim',
                    choices=['projectairsim', 'classic'])
    ap.add_argument('--address', default=None,
                    help='engine host (default: the local machine)')
    ap.add_argument('--out', default='tier_c_sign_test_result.json')
    args = ap.parse_args(argv)

    eng = make_engine(args.engine,
                      **({'address': args.address} if args.address else {}))
    eng.connect()
    checks = []
    try:
        eng.takeoff()                # velocity commands need flight mode
        eng.move_by_velocity_body(0.0, 0.0, -0.1, 0.0, 0.4)  # unpark (UP)
        settle(eng)
        b0, b1 = push(eng, v_up=0.15, yaw_rate=+0.3)
        ok = b0 is not None and b1 is not None and b1[0] > b0[0] + 5.0
        checks.append(('+yaw_rate (REP-103 CCW) -> +u (image right)', ok,
                       f'u {fmt(b0, 0)} -> {fmt(b1, 0)}'))
        settle(eng)
        b0, b1 = push(eng, v_up=+0.3)
        ok = b0 is not None and b1 is not None and b1[1] > b0[1] + 5.0
        checks.append(('+v_up -> +v (image down)', ok,
                       f'v {fmt(b0, 1)} -> {fmt(b1, 1)}'))
        settle(eng)
        b0, b1 = push(eng, v_fwd=+0.3)
        ok = b0 is not None and b1 is not None and b1[2] > b0[2] + 1.0
        checks.append(('+v_fwd -> bbox width grows', ok,
                       f'w {fmt(b0, 2)} -> {fmt(b1, 2)}'))
    finally:
        eng.disconnect()

    all_ok = all(ok for _, ok, _ in checks)
    for name, ok, detail in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}   ({detail})')
    with open(args.out, 'w') as f:
        json.dump({'engine': args.engine,
                   'checks': [{'name': n, 'pass': bool(ok), 'detail': d}
                              for n, ok, d in checks],
                   'all_pass': all_ok}, f, indent=2)
    print(('TIER-C SIGN TEST PASS -> ' if all_ok else
           'TIER-C SIGN TEST FAIL -- fix engine.py _ned_yaw_rate, record '
           'it -> ') + args.out)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
