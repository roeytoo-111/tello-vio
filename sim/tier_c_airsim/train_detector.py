"""Train the tello-vio drone detector on OUR Tier-C dataset.

dataset_factory.py renders the warehouse and writes an ultralytics-ready set
(images/*.png + labels/*.txt, one class `0`, plus metadata.csv). This turns
that set into a trained detector:

    stratified train/val split (by target RANGE, so the small-target frames
    the source paper struggles with appear in BOTH splits) -> data.yaml
    (one class: `drone`) -> ultralytics fine-tune from a small base model
    -> best.pt

The result is IN-DOMAIN for Tier C: it learns the warehouse renderer the
vision-in-loop eval flies in, which a stock COCO model (no `drone` class) or
another project's thermal model never sees. It is a SIM detector -- for real
Tello flight, mix real captures in before trusting it outdoors
(dataset_factory.py's own warning; sim_training_architecture.md 4.2).

    # after generating the dataset on the live engine (all three lightings):
    ~/.venvs/tier_c/bin/python train_detector.py --data chase_dataset \
        --epochs 100
    # -> detector_runs/chase_yolo/weights/best.pt

    ~/.venvs/tier_c/bin/python train_detector.py --data chase_dataset \
        --dry-run          # split + data.yaml only, no engine, no training
"""
import argparse
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# yolov8-nano fine-tunes fast on a few thousand frames and needs no GPU to
# be usable; the repo ships the weight so this runs offline.
DEFAULT_MODEL = os.path.join(os.path.dirname(HERE), 'yolov8n.pt')
IMG_EXTS = ('.png', '.jpg', '.jpeg')


def _range_of(stem: str, meta: dict) -> str:
    """Stratification key = target range bucket. Prefer metadata.csv;
    fall back to parsing the stem (`..._r0p50_...` -> '0p50')."""
    if stem in meta:
        return meta[stem]
    for tok in stem.split('_'):
        if tok.startswith('r') and len(tok) > 1:
            return tok[1:]
    return '_'


def _load_meta(data_dir: str) -> dict:
    path = os.path.join(data_dir, 'metadata.csv')
    out = {}
    if os.path.exists(path):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('stem') and row.get('range_m'):
                    out[row['stem']] = row['range_m']
    return out


def split(data_dir: str, val_frac: float, seed: int):
    """Deterministic train/val split STRATIFIED by target range, so each
    split spans the whole target-size spectrum (near=large .. far=tiny).
    Returns (train_paths, val_paths) as absolute image paths."""
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')
    if not os.path.isdir(img_dir):
        raise SystemExit(f'no images/ under {data_dir} -- run '
                         f'dataset_factory.py first (needs the live engine)')
    meta = _load_meta(data_dir)
    buckets = {}
    for fn in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in IMG_EXTS:
            continue
        # Only keep frames that actually have a label (an unlabelled image
        # would train the net that the drone is background).
        if not os.path.exists(os.path.join(lbl_dir, stem + '.txt')):
            continue
        buckets.setdefault(_range_of(stem, meta), []).append(
            os.path.join(img_dir, fn))
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(buckets):
        items = sorted(buckets[key])
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    if not train:
        raise SystemExit(f'no labelled frames found under {data_dir}')
    return sorted(train), sorted(val)


def write_dataset_yaml(data_dir: str, train, val):
    """train.txt / val.txt (absolute image paths; ultralytics derives each
    label by swapping images/ -> labels/) + a one-class data.yaml."""
    train_txt = os.path.join(data_dir, 'train.txt')
    val_txt = os.path.join(data_dir, 'val.txt')
    with open(train_txt, 'w') as f:
        f.write('\n'.join(train) + '\n')
    with open(val_txt, 'w') as f:
        f.write('\n'.join(val) + '\n')
    yaml_path = os.path.join(data_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'path: {os.path.abspath(data_dir)}\n'
                f'train: train.txt\n'
                f'val: val.txt\n'
                f'names:\n'
                f'  0: drone\n')
    return yaml_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='chase_dataset',
                    help='dataset dir from dataset_factory (images/ labels/)')
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help='base weights to fine-tune from (default: repo '
                         'yolov8n.pt)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    # 8 dataloader workers (ultralytics' default) OOM-killed this 7.6 GB WSL
    # box -- each worker holds ~1.3 GB of augmentation/shared buffers, and
    # the orphaned workers then wedged the machine. 2 is safe here; raise it
    # only if `free -h` shows plenty of headroom.
    ap.add_argument('--workers', type=int, default=2,
                    help='dataloader workers (low on purpose: high counts '
                         'OOM-kill a small-RAM WSL)')
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None,
                    help="ultralytics device (default: auto; e.g. '0' or "
                         "'cpu')")
    ap.add_argument('--project', default=os.path.join(HERE, 'detector_runs'))
    ap.add_argument('--name', default='chase_yolo')
    ap.add_argument('--dry-run', action='store_true',
                    help='split + write data.yaml only; no training')
    args = ap.parse_args(argv)

    train, val = split(args.data, args.val_frac, args.seed)
    yaml_path = write_dataset_yaml(args.data, train, val)
    print(f'split: {len(train)} train / {len(val)} val  ->  {yaml_path}')
    if args.dry_run:
        print('[dry-run] data.yaml written; skipping training')
        return 0

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, workers=args.workers, cache=False,
                seed=args.seed, project=args.project, name=args.name,
                exist_ok=True,
                **({'device': args.device} if args.device else {}))
    best = os.path.join(args.project, args.name, 'weights', 'best.pt')
    print(f'\ndetector trained -> {best}\n'
          f'fly it in the warehouse with:\n'
          f'  ~/.venvs/tier_c/bin/python vision_in_loop_eval.py \\\n'
          f'      --checkpoint <td3 .pt> --yolo {best} --lighting medium')
    return 0


if __name__ == '__main__':
    sys.exit(main())
