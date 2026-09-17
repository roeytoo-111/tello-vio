"""Per-frame CSV logging -- pure Python, no ROS.

One row per PROCESSED frame, misses included, because the miss statistics
(how often, how long, at what target size) are precisely the detector-noise
model the RL simulator needs to inject (sim_training_architecture.md). Kept
as its own module so an offline evaluator writes the identical schema.

Timestamps are integer nanoseconds to survive float round-tripping; empty
string means "not applicable this row" (e.g. box fields on a miss).
"""
import csv
import math
import os
import time
from typing import Optional


COLUMNS = [
    'seq',              # processed-frame counter, this run
    't_capture_ns',     # image header.stamp: driver's arrival stamp
    't_receive_ns',     # when the detector node picked the frame up
    'transport_ms',     # (t_receive - t_capture), same host clock
    'inference_ms',     # YOLO pre+infer+post
    'detected',         # 0/1
    'class_name',
    'confidence',
    'xmin', 'ymin', 'xmax', 'ymax',
    'cx', 'cy',         # box centre, px
    'w', 'h',           # box size, px
    'area_frac',        # box area / frame area (the paper's depth-rule input)
    'range_m',          # pinhole range, empty when unavailable
    'az_deg', 'el_deg', # bearings of the centre, empty when no camera_info
    'image_w', 'image_h',
]


def _fmt(v) -> str:
    if v is None:
        return ''
    if isinstance(v, float):
        if math.isnan(v):
            return ''
        return f'{v:.4f}'
    return str(v)


def default_csv_path(directory: str) -> str:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(directory, f'detections_{stamp}.csv')


class DetectionCsvLogger:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        self._fh = open(self.path, 'w', newline='')
        self._writer = csv.writer(self._fh)
        self._writer.writerow(COLUMNS)
        self._fh.flush()
        self.rows = 0

    def write(self, **fields) -> None:
        unknown = set(fields) - set(COLUMNS)
        if unknown:
            raise KeyError(f'unknown CSV fields: {sorted(unknown)}')
        self._writer.writerow([_fmt(fields.get(c)) for c in COLUMNS])
        self.rows += 1
        if self.rows % 30 == 0:      # ~once a second at stream rate
            self._fh.flush()

    def close(self) -> Optional[str]:
        if self._fh is None:
            return None
        self._fh.flush()
        self._fh.close()
        self._fh = None
        return self.path
