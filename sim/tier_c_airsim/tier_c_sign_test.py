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
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import make_engine  # noqa: E402

SETTLE_S = 1.0
PUSH_S = 1.0


def bbox_centre(eng):
    eng.get_rgb()
    boxes = eng.get_bboxes()
    if not boxes:
        return None
    b = boxes[0]
    return ((b.xmin + b.xmax) / 2.0, (b.ymin + b.ymax) / 2.0, b.w)


def settle(eng):
    eng.set_vehicle_pose((0.0, 0.0, -1.0), 0.0)
    eng.set_object_pose('TargetTello', (2.0, 0.0, -1.0), 0.0)
    eng.hover()
    time.sleep(SETTLE_S)


def push(eng, v_fwd=0.0, v_up=0.0, yaw_rate=0.0):
    b0 = bbox_centre(eng)
    eng.move_by_velocity_body(v_fwd, 0.0, -v_up, yaw_rate, PUSH_S)
    b1 = bbox_centre(eng)
    eng.hover()
    return b0, b1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', default='projectairsim',
                    choices=['projectairsim', 'classic'])
    ap.add_argument('--out', default='tier_c_sign_test_result.json')
    args = ap.parse_args(argv)

    eng = make_engine(args.engine)
    eng.connect()
    checks = []
    try:
        settle(eng)
        b0, b1 = push(eng, yaw_rate=+0.5)
        ok = b0 is not None and b1 is not None and b1[0] > b0[0] + 5.0
        checks.append(('+yaw_rate (REP-103 CCW) -> +u (image right)', ok,
                       f'u {b0 and b0[0]:.1f} -> {b1 and b1[0]:.1f}'))
        settle(eng)
        b0, b1 = push(eng, v_up=+0.3)
        ok = b0 is not None and b1 is not None and b1[1] > b0[1] + 5.0
        checks.append(('+v_up -> +v (image down)', ok,
                       f'v {b0 and b0[1]:.1f} -> {b1 and b1[1]:.1f}'))
        settle(eng)
        b0, b1 = push(eng, v_fwd=+0.3)
        ok = b0 is not None and b1 is not None and b1[2] > b0[2] + 1.0
        checks.append(('+v_fwd -> bbox width grows', ok,
                       f'w {b0 and b0[2]:.1f} -> {b1 and b1[2]:.1f}'))
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
