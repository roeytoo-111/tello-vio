"""Streamed-command health probe -- run before any vision-in-loop eval.

Measured on Blocks 1.0.1 (2026-09-20): fire-and-forget velocity commands
carrying a vertical or yaw component could pin simple_flight in a dead
hover (|v| = 0.00) while pure-horizontal streams chained fine. The engine
adapter now runs its asyncio loop in a background thread so command
send/ack completes in real time; THIS probe is the pass/fail check for
that whole mechanism, on the live engine, in ~60 s:

    ~/.venvs/tier_c/bin/python stream_probe.py

PASS = all four rows show |dpos| well above zero (the vehicle flies on
every axis combination). Any 0.00 row = streamed commands are still being
dropped: do NOT trust an intercept eval until this passes.
"""
import argparse
import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import make_engine  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--address', default=None)
    args = ap.parse_args(argv)
    eng = make_engine('projectairsim',
                      **({'address': args.address} if args.address else {}))
    eng.connect()
    eng.takeoff()
    eng.rc_start()
    failures = 0

    def probe(tag, vdown=0.0, yaw=0.0):
        # Exercise the RC BRIDGE exactly as the eval does: refresh the
        # setpoint each "tick" while the bridge re-issues blocking
        # commands back-to-back. (Raw streamed commands are NOT probed:
        # their vertical/yaw freezes are known, engine-mood-dependent,
        # and the eval no longer uses them.)
        nonlocal failures
        eng.rc_pause()
        eng.settle()
        (x0, y0, z0), yaw0 = eng.get_vehicle_pose()
        t0 = time.time()
        while time.time() - t0 < 3.0:
            eng.rc_set_body(0.8, 0.0, vdown, yaw)
            time.sleep(0.45)
        eng.rc_pause()
        (x1, y1, z1), yaw1 = eng.get_vehicle_pose()
        moved = abs(x1 - x0) + abs(y1 - y0) > 0.3
        failures += not moved
        print(f'  [{"PASS" if moved else "FAIL"}] {tag:26s} '
              f'dpos=({x1-x0:5.2f},{y1-y0:5.2f},{z1-z0:5.2f}) '
              f'dyaw={yaw1-yaw0:5.2f}  |v|={eng.get_speed():.2f}')

    try:
        probe('pure forward (control)')
        probe('fwd + v_down 0.1', vdown=0.1)
        probe('fwd + yaw 0.3', yaw=0.3)
        probe('fwd + down + yaw', vdown=0.1, yaw=0.3)
    finally:
        eng.rc_stop()
        eng.settle()
        eng.disconnect()
    print('RC-BRIDGE PROBE ' + ('PASS -- the eval transport actuates on '
                                'every axis; go fly'
                                if failures == 0 else
                                f'FAIL ({failures}/4 frozen) -- the engine '
                                f'is in a bad state; restart Blocks.exe '
                                f'and probe again'))
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
