"""Job 1 -- the YOLO dataset factory (sim_training_architecture.md 4.2).

Loops over lighting x target pose grid x ranges 0.5-6 m x yaw, captures
RGB with EXACT auto-labels (native bbox annotations on Project AirSim;
Detection API / segmentation mask on classic), and writes a YOLO-format
dataset STRATIFIED BY TARGET PIXEL SIZE -- attacking the source paper's
own named weakness (small targets [P App. A]) with unlimited exact labels.

Output layout (ultralytics-ready):
    <out>/images/*.png
    <out>/labels/*.txt          one line: 0 cx cy w h (normalised)
    <out>/metadata.csv          range_m, w_px, lighting, yaw -- the
                                stratification the accuracy-vs-size curve
                                and the corruption-model fit consume

The set MUST be mixed with real captures before detector training
(implementation_plan.md Phase 0): synthetic-only detectors inherit the
renderer's domain.

    python3 dataset_factory.py --engine projectairsim --out chase_dataset/
    python3 dataset_factory.py --dry-run     # label arithmetic, no engine
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chase_paths  # noqa: F401,E402  (puts chase_* on sys.path)
from engine import BBox, bbox_from_mask, make_engine  # noqa: E402
import warehouse  # noqa: E402

# The real camera's geometry comes from the shared constants (the repo's
# chase_gym must be on PYTHONPATH, as for vision_in_loop_eval.py) -- never
# restated (chase_gym/constants.py contract).
from chase_gym import constants as _C  # noqa: E402

FRAME_W, FRAME_H = _C.FRAME_W, _C.FRAME_H

# The follower hovers at FOLLOWER_NED looking north (NED +x); the target
# is placed on a polar grid CENTRED ON THE CAMERA. The grid must share the
# camera's altitude: measured against the shipped scene, a grid around the
# world origin put only 356 of 1120 poses in frame and none nearer than
# 2 m, because the camera sits 1 m up.
FOLLOWER_NED = (0.0, 0.0, -1.0)
RANGES_M = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
AZIMUTHS_DEG = [-20, -12, -5, 0, 5, 12, 20]
ELEVATIONS_DEG = [-12, -6, 0, 6, 12]
TARGET_YAWS_DEG = [0, 45, 90, 180]
# Lighting is scene-side state the CLIENT CANNOT SET through either
# engine's stock API: the operator configures the environment (sun angle /
# intensity per the paper's low/medium/high study) and passes the matching
# label per invocation -- one dataset run per lighting class. A fake
# in-process lighting loop would stamp three labels onto identical frames.
LIGHTING_CLASSES = ('low', 'medium', 'high')

# Where the target goes while NEGATIVE frames are captured: under the
# floor slab, so no view can contain it.
NEG_HIDE_NED = (9.0, 0.0, 2.0)


def yolo_line(b: BBox, img_w: int, img_h: int) -> str:
    cx = (b.xmin + b.xmax) / 2.0 / img_w
    cy = (b.ymin + b.ymax) / 2.0 / img_h
    return (f'0 {cx:.6f} {cy:.6f} '
            f'{b.w / img_w:.6f} {b.h / img_h:.6f}')


def pose_grid(origin=FOLLOWER_NED, yaw: float = 0.0):
    """Polar grid centred on the CAMERA (origin, yaw): azimuth is measured
    from the camera axis, so the grid is valid wherever the vehicle
    actually is -- run() reads the live pose rather than assuming the
    spawn point."""
    for r in RANGES_M:
        for az in AZIMUTHS_DEG:
            for el in ELEVATIONS_DEG:
                for tyaw in TARGET_YAWS_DEG:
                    az_r = math.radians(az) + yaw
                    el_r = math.radians(el)
                    x = origin[0] + r * math.cos(el_r) * math.cos(az_r)
                    y = origin[1] + r * math.cos(el_r) * math.sin(az_r)
                    z = origin[2] - r * math.sin(el_r)   # NED z is DOWN
                    yield r, az, el, tyaw, (x, y, z)


def capture_negatives(eng, out_dir, meta, target_object, lighting,
                      n_frames: int) -> int:
    """BACKGROUND frames: the warehouse with NO drone anywhere in view,
    saved with EMPTY label files (the ultralytics background convention).

    Why: a detector trained only on frames that all contain the target
    never learns what 'no drone' looks like -- measured live, ours
    hallucinated 0.6-0.9-confidence boxes on dark walls/shelf shadows the
    moment the intercept eval flew off the aisle axis, and every phantom
    box became a fake width-based capture. Standard practice is ~10%
    negatives.

    The patrol deliberately produces the OFF-AXIS views that fooled it:
    the target is hidden under the floor, then the vehicle spins in place
    (with the slight vertical bias that makes simple_flight yaw actuate)
    at several spots down the aisle, capturing walls, racks, dark corners
    and the ceiling. Every frame is verified target-free via the engine's
    own annotations before it is saved."""
    import time
    import cv2
    eng.set_object_pose(target_object, NEG_HIDE_NED, 0.0)
    time.sleep(0.5)
    n = 0
    spot = 0
    v_up_sign = 1.0
    while n < n_frames:
        # a slow spin with a small alternating climb/descend bias (yaw
        # only actuates alongside vertical motion on simple_flight)
        for _ in range(max(6, n_frames // 4)):
            if n >= n_frames:
                break
            eng.move_by_velocity_body(0.0, 0.0, -0.06 * v_up_sign, 0.55,
                                      1.0, wait=False)
            time.sleep(0.55)
            img = eng.get_rgb()
            if eng.get_bboxes():
                continue                 # target somehow in view: not a negative
            stem = f'{lighting}_neg_{n:03d}'
            cv2.imwrite(os.path.join(out_dir, 'images', stem + '.png'), img)
            open(os.path.join(out_dir, 'labels', stem + '.txt'), 'w').close()
            meta.writerow([stem, 'neg', '', '', '', lighting, '', ''])
            n += 1
        v_up_sign = -v_up_sign
        if hasattr(eng, 'settle'):
            eng.settle()
        # hop to the next spot down the aisle for fresh backgrounds
        spot += 1
        if spot < 3 and n < n_frames:
            for _ in range(4):
                eng.move_by_velocity_body(0.8, 0.0, 0.0, 0.0, 1.0,
                                          wait=False)
                time.sleep(0.5)
            if hasattr(eng, 'settle'):
                eng.settle()
    return n


def run(engine_name: str, out_dir: str, target_object: str,
        lighting: str, address: str = None, warehouse_scene: bool = True,
        negatives: int = 0, grid: bool = True) -> int:
    import cv2
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    meta_path = os.path.join(out_dir, 'metadata.csv')
    # APPEND: the workflow is one run per lighting class into one dataset
    # dir -- truncating here would wipe the previous class's rows while
    # its images stay on disk.
    write_header = not (os.path.exists(meta_path)
                        and os.path.getsize(meta_path) > 0)
    eng = make_engine(engine_name,
                      **({'address': address} if address else {}))
    eng.connect()
    if warehouse_scene:
        # Build the warehouse and set the interior light from the lighting
        # class -- so --lighting physically changes the scene, and each class
        # also gets its own clutter arrangement.
        seed = {'low': 10, 'medium': 20, 'high': 30}.get(lighting, 0)
        n_props, n_lights = warehouse.build_and_light(eng, seed=seed,
                                                      lighting=lighting)
        print(f'warehouse: {n_props} props, {n_lights} lights, '
              f'lighting={lighting}')
    eng.takeoff()                                # so the camera is at flight pose
    eng.move_by_velocity_body(0.0, 0.0, -0.1, 0.0, 0.4)  # unpark/settle
    origin, yaw = eng.get_vehicle_pose()         # grid centred on the camera
    n = 0
    try:
        with open(meta_path, 'a', newline='') as mf:
            meta = csv.writer(mf)
            if write_header:
                meta.writerow(['stem', 'range_m', 'azimuth_deg',
                               'elevation_deg', 'target_yaw_deg',
                               'lighting', 'w_px', 'h_px'])
            for r, az, el, tyaw, ned in (
                    pose_grid(origin, yaw) if grid else ()):
                eng.set_object_pose(target_object, ned,
                                    yaw + math.radians(tyaw))
                img = eng.get_rgb()
                boxes = eng.get_bboxes()
                if not boxes:
                    continue                     # occluded / out of FOV
                b = boxes[0]
                stem = (f'{lighting}_r{r:0.2f}_az{az}_el{el}_y{tyaw}'
                        .replace('.', 'p').replace('-', 'm'))
                cv2.imwrite(os.path.join(out_dir, 'images', stem + '.png'),
                            img)
                with open(os.path.join(out_dir, 'labels', stem + '.txt'),
                          'w') as lf:
                    lf.write(yolo_line(b, img.shape[1], img.shape[0])
                             + '\n')
                meta.writerow([stem, r, az, el, tyaw, lighting,
                               f'{b.w:.1f}', f'{b.h:.1f}'])
                n += 1
            if negatives > 0:
                n_neg = capture_negatives(eng, out_dir, meta,
                                          target_object, lighting,
                                          negatives)
                print(f'negatives: {n_neg} background frames at '
                      f'lighting={lighting}')
    finally:
        eng.disconnect()
    print(f'dataset: {n} labelled frames at lighting={lighting} -> '
          f'{out_dir} (set the scene lighting and rerun per class; mix '
          f'with real captures before training!)')
    return 0


def dry_run() -> int:
    """No engine: verify the label arithmetic on synthetic segmentation
    masks + BBox inputs, and print the grid size."""
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    mask[300:340, 500:560] = True
    box = bbox_from_mask(mask)
    assert box == (500.0, 300.0, 559.0, 339.0), box
    b = BBox('t', *box)
    line = yolo_line(b, FRAME_W, FRAME_H)
    parts = [float(x) for x in line.split()[1:]]
    assert abs(parts[0] - (529.5 / FRAME_W)) < 1e-6
    assert abs(parts[2] - (59.0 / FRAME_W)) < 1e-6
    assert bbox_from_mask(np.zeros((4, 4), dtype=bool)) is None
    grid = sum(1 for _ in pose_grid())
    print(f'[dry-run] label arithmetic OK; grid = {grid} captures per '
          f'lighting class ({len(RANGES_M)} ranges x {len(AZIMUTHS_DEG)} '
          f'az x {len(ELEVATIONS_DEG)} el x {len(TARGET_YAWS_DEG)} yaws; '
          f'x {len(LIGHTING_CLASSES)} operator-set lighting classes)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', default='projectairsim',
                    choices=['projectairsim', 'classic'])
    ap.add_argument('--out', default='chase_dataset')
    ap.add_argument('--target-object', default='TargetTello')
    ap.add_argument('--lighting', default=None,
                    choices=LIGHTING_CLASSES,
                    help='REQUIRED (except --dry-run): attestation of the '
                         'scene lighting you set in the engine for THIS '
                         'run -- a default would turn forgetfulness into '
                         'wrong labels')
    ap.add_argument('--address', default=None,
                    help='engine host (default: the local machine)')
    ap.add_argument('--no-warehouse', action='store_true',
                    help='use the packaged outdoor level instead of building '
                         'the indoor warehouse scene')
    ap.add_argument('--negatives', type=int, default=0,
                    help='ALSO capture this many target-free background '
                         'frames (empty labels) via a patrol -- a detector '
                         'trained with zero negatives hallucinates drones '
                         'on dark walls (measured)')
    ap.add_argument('--negatives-only', action='store_true',
                    help='skip the labelled pose grid; only patrol for '
                         'background frames (append to an existing set)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    if args.dry_run:
        return dry_run()
    if args.lighting is None:
        ap.error('--lighting is required for a real run (the attestation '
                 'of the scene lighting you set in the engine)')
    if args.negatives_only and args.negatives <= 0:
        ap.error('--negatives-only needs --negatives N')
    return run(args.engine, args.out, args.target_object, args.lighting,
               args.address, warehouse_scene=not args.no_warehouse,
               negatives=args.negatives, grid=not args.negatives_only)


if __name__ == '__main__':
    sys.exit(main())
