"""The sign test -- run FIRST, before anything else in Tier B
(sim_training_architecture.md 3.2 item 4 / 4.4).

gz-sim has an OPEN, unconfirmed report that the quadrotor positive-yaw
rotation direction is inverted in the multicopter path (issue #2657,
checked 2026-09-17: open, 0 comments, no fix in any release). The original
chase code needed a -yaw sign flip on real hardware [C main.py:225]. Three
simulators + one aircraft = four chances for a silent sign flip. So:
command each axis for one second, assert the IMAGE-SPACE effect through
the whole chain (twist -> plugin -> physics -> oracle projection), and
record the result next to the checkpoints.

    python3 -m chase_sim_gz.sign_test          # against real gz-sim
    python3 -m chase_sim_gz.sign_test --fake   # against the kinematic fake

Expected convention (REP-103 body frame, forward camera; derived and
unit-tested in chase_gym/test/test_kinematics.py):
    +yaw rate (CCW)  -> target image moves RIGHT (+u)
    +linear.z (up)   -> target image moves DOWN  (+v)
    +linear.x (fwd)  -> box width w GROWS
A FAIL here means: flip the sign in the shim/action map AND record it in
the checkpoint bundle -- never patch it silently downstream.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from chase_gym.kinematics import FollowerState, project, world_to_camera

from .gz_iface import add_backend_args, make_backend


def project_target(backend) -> tuple:
    od_f = backend.get_odom('follower')
    od_t = backend.get_odom('target')
    st = FollowerState(pos=od_f.pos, yaw=od_f.yaw)
    return project(world_to_camera(od_t.pos, st))


def run_axis(backend, lin, yaw_rate, seconds: float = 1.0):
    u0, v0, w0, in0 = project_target(backend)
    backend.send_twist('follower', lin, yaw_rate)
    backend.step(int(seconds / backend.physics_dt))
    u1, v1, w1, in1 = project_target(backend)
    backend.send_twist('follower', (0.0, 0.0, 0.0), 0.0)
    backend.step(int(0.5 / backend.physics_dt))
    return (u0, v0, w0, in0), (u1, v1, w1, in1)


def settle(backend):
    backend.reset_world()
    backend.set_pose('follower', (0.0, 0.0, 1.0), 0.0)
    backend.set_pose('target', (2.0, 0.0, 1.0), 0.0)
    backend.send_twist('follower', (0.0, 0.0, 0.0), 0.0)
    backend.send_twist('target', (0.0, 0.0, 0.0), 0.0)
    backend.step(int(0.5 / backend.physics_dt))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_backend_args(ap)
    ap.add_argument('--out', default='sign_test_result.json')
    args = ap.parse_args(argv)

    backend = make_backend(args)
    backend.start()

    checks = []
    try:
        # +yaw -> image right (+u)
        settle(backend)
        b0, b1 = run_axis(backend, (0.0, 0.0, 0.0), +0.5)
        checks.append(('+yaw_rate -> +u (image right)', b1[0] > b0[0] + 5.0,
                       f'u {b0[0]:.1f} -> {b1[0]:.1f}'))
        # +linear.z -> image down (+v)
        settle(backend)
        b0, b1 = run_axis(backend, (0.0, 0.0, +0.3), 0.0)
        checks.append(('+linear.z -> +v (image down)', b1[1] > b0[1] + 5.0,
                       f'v {b0[1]:.1f} -> {b1[1]:.1f}'))
        # +linear.x -> box grows
        settle(backend)
        b0, b1 = run_axis(backend, (+0.3, 0.0, 0.0), 0.0)
        checks.append(('+linear.x -> w grows (closing)', b1[2] > b0[2] + 1.0,
                       f'w {b0[2]:.1f} -> {b1[2]:.1f}'))
        # hover sanity: altitude holds within 0.2 m over 1 s of zeros
        settle(backend)
        od0 = backend.get_odom('follower')
        backend.step(int(1.0 / backend.physics_dt))
        od1 = backend.get_odom('follower')
        checks.append(('hover holds altitude (calibration sanity)',
                       abs(od1.pos[2] - od0.pos[2]) < 0.2,
                       f'z {od0.pos[2]:.2f} -> {od1.pos[2]:.2f}'))
    finally:
        backend.stop()

    all_ok = all(ok for _, ok, _ in checks)
    for name, ok, detail in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}   ({detail})')
    result = {'backend': 'fake' if args.fake else 'gz',
              'checks': [{'name': n, 'pass': bool(ok), 'detail': d}
                         for n, ok, d in checks],
              'convention': '+yaw->+u, +z->+v, +x->w-grows (REP-103)',
              'all_pass': all_ok}
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(('SIGN TEST PASS -- convention recorded in ' if all_ok else
           'SIGN TEST FAIL -- flip in the shim + action map, record it; '
           'result in ') + args.out)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
