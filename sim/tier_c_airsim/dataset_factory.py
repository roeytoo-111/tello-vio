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
from engine import BBox, bbox_from_mask, make_engine  # noqa: E402

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


def run(engine_name: str, out_dir: str, target_object: str,
        lighting: str, address: str = None) -> int:
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
    origin, yaw = eng.get_vehicle_pose()         # grid centred on the camera
    n = 0
    try:
        with open(meta_path, 'a', newline='') as mf:
            meta = csv.writer(mf)
            if write_header:
                meta.writerow(['stem', 'range_m', 'azimuth_deg',
                               'elevation_deg', 'target_yaw_deg',
                               'lighting', 'w_px', 'h_px'])
            for r, az, el, tyaw, ned in pose_grid(origin, yaw):
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
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    if args.dry_run:
        return dry_run()
    if args.lighting is None:
        ap.error('--lighting is required for a real run (the attestation '
                 'of the scene lighting you set in the engine)')
    return run(args.engine, args.out, args.target_object, args.lighting,
               args.address)


if __name__ == '__main__':
    sys.exit(main())
