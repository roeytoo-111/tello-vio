"""Open and LOOK at the warehouse scene -- no dataset, no policy.

Blocks.exe is only an empty stage; the warehouse (shelving, crates, barrels)
is spawned into it by warehouse.py. This helper connects to a running engine,
reloads a clean scene, builds the lit warehouse, parks the follower in the
aisle with the target ahead, saves one camera frame, and leaves the scene
populated so you can look around in the engine window.

    # 1) start the engine on Windows (once):
    #    C:\ProjectAirSim\Blocks\Blocks.exe
    # 2) then, from WSL, in this folder, in the tier_c venv:
    ~/.venvs/tier_c/bin/python open_warehouse.py --lighting medium
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import make_engine  # noqa: E402
import warehouse  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', default='projectairsim',
                    choices=['projectairsim', 'classic'])
    ap.add_argument('--address', default=None)
    ap.add_argument('--lighting', default='medium',
                    choices=['low', 'medium', 'high'])
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--range', type=float, default=3.0,
                    help='where to place the target ahead, metres')
    ap.add_argument('--out', default='warehouse_view.png')
    args = ap.parse_args(argv)

    eng = make_engine(args.engine,
                      **({'address': args.address} if args.address else {}))
    eng.connect()                       # reloads a clean scene, spawns target
    n_props, n_lights = warehouse.build_and_light(
        eng, seed=args.seed, lighting=args.lighting)
    print(f'warehouse built: {n_props} props, {n_lights} lights, '
          f'lighting={args.lighting}')
    eng.takeoff()
    eng.move_by_velocity_body(0.0, 0.0, -0.1, 0.0, 0.4)   # unpark/settle
    (x, y, z), yaw = eng.get_vehicle_pose()
    eng.set_object_pose('TargetTello',
                        (x + args.range * math.cos(yaw),
                         y + args.range * math.sin(yaw), z), yaw)
    try:
        import cv2
        from projectairsim.types import ImageType
        from projectairsim.utils import unpack_image
        img = unpack_image(eng._drone.get_images(
            eng.camera, [ImageType.SCENE])[ImageType.SCENE])
        cv2.imwrite(args.out, img)
        print(f'saved the follower camera view -> {args.out}')
    except Exception as exc:
        print(f'(camera snapshot skipped: {exc})')
    # Disconnect the CLIENT (so this process exits cleanly). The spawned
    # objects are server-side state and stay in the running engine, so the
    # warehouse remains on screen to look around after this returns.
    eng.disconnect()
    print('warehouse is left standing in the running engine. '
          'Re-run to rebuild, or restart Blocks.exe for an empty stage.')


if __name__ == '__main__':
    main()
