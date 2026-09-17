"""YOLO inference wrapper -- pure Python, no ROS.

Everything ROS-independent about detection lives here so it can be unit
tested without a drone, a GPU, or a running graph, and reused verbatim by a
future offline evaluator (rosbag replay) or the training pipeline.

COLOUR CONTRACT -- the one that silently breaks Tello + YOLO code:
``detect()`` takes a **BGR** uint8 HxWx3 array. Ultralytics assumes numpy
input is BGR (`ultralytics/data/loaders.py`, LoadPilAndNumpy._single_check:
"NumPy color inputs are assumed to use OpenCV-compatible BGR order";
`engine/predictor.py` preprocess then does ``im[..., ::-1]``). djitellopy
2.5.0 hands out **RGB** (``np.array(frame.to_image())`` -- PIL is RGB), so a
raw djitellopy frame fed straight to YOLO is channel-swapped. The ROS node
avoids this by construction: cv_bridge converts whatever encoding the driver
publishes ('rgb8' or 'bgr8') to bgr8 before calling into here.

CLASS CONTRACT: stock COCO weights (yolov8n.pt et al.) have NO 'drone'
class -- verified against this repo's copy (80 classes; nearest are
'airplane', 'bird', 'kite'). Detecting a Tello for real requires custom
one-class weights (implementation_plan.md, Phase 0). This wrapper is
weights-agnostic: point ``weights`` at the fine-tuned .pt and set
``target_classes=['drone']`` and nothing else changes. Asking for a class
the weights do not have is a hard error naming the classes that exist,
because a silent empty filter looks exactly like "the detector is broken".
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One box, pixels, in the source frame's own resolution."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax))


@dataclass(frozen=True)
class FrameResult:
    """All surviving boxes for one frame, best first."""
    best: Optional[Detection]         # None on a miss
    boxes: Tuple[Detection, ...]      # sorted by confidence, descending
    inference_ms: float               # preprocess + inference + postprocess


def resolve_class_ids(names: Dict[int, str], wanted: Sequence[str]) -> List[int]:
    """Map class names to ids against a model's own name table.

    Case-insensitive. Raises ValueError naming the available classes when a
    wanted name does not exist -- the yolov8n-has-no-drone-class trap must be
    loud, not an eternally-empty detection stream.
    """
    lookup = {v.lower(): k for k, v in names.items()}
    ids = []
    missing = []
    for w in wanted:
        w = w.strip()
        if not w:
            continue
        cid = lookup.get(w.lower())
        if cid is None:
            missing.append(w)
        else:
            ids.append(cid)
    if missing:
        available = ', '.join(names[k] for k in sorted(names))
        raise ValueError(
            f"class(es) {missing} not in this model's classes. "
            f"The loaded weights know only: {available}")
    return sorted(set(ids))


def select_boxes(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray,
                 names: Dict[int, str], min_box_px: float) -> List[Detection]:
    """Filter degenerate boxes and sort by confidence, descending.

    Class filtering is done inside ultralytics via ``classes=``; this only
    rejects boxes thinner than ``min_box_px`` in either dimension, which at
    Tello scale are decode artefacts, not aircraft.
    """
    out = []
    for i in range(len(conf)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        if (x2 - x1) < min_box_px or (y2 - y1) < min_box_px:
            continue
        cid = int(cls[i])
        out.append(Detection(x1, y1, x2, y2, float(conf[i]), cid,
                             names.get(cid, str(cid))))
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out


class YoloDroneDetector:
    """Owns the ultralytics model and the inference settings.

    Import of ultralytics/torch happens here, at construction, so that pure
    consumers of this module's dataclasses never pay for it.
    """

    def __init__(self, weights: str, device: str = 'cuda:0', imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45, max_det: int = 8,
                 target_classes: Sequence[str] = (), min_box_px: float = 4.0):
        import torch
        from ultralytics import YOLO

        self.requested_device = device
        self.device_fell_back = False
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
            self.device_fell_back = True
        self.device = device

        self.model = YOLO(weights)
        # Plain dict copy: ultralytics may hand back a lazy mapping.
        self.names: Dict[int, str] = dict(self.model.names)
        self.class_ids: Optional[List[int]] = None
        if target_classes:
            self.class_ids = resolve_class_ids(self.names, target_classes)

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.min_box_px = float(min_box_px)

    def warmup(self, shape=(720, 960, 3), n: int = 2) -> float:
        """Run n dummy inferences so the first real frame does not eat the
        CUDA context + cuDNN autotune cost. Returns the last per-frame ms."""
        dummy = np.zeros(shape, dtype=np.uint8)
        ms = 0.0
        for _ in range(n):
            ms = self.detect(dummy).inference_ms
        return ms

    def detect(self, bgr: np.ndarray) -> FrameResult:
        """Run the model on one BGR frame. See module docstring for why BGR."""
        results = self.model.predict(
            bgr, device=self.device, imgsz=self.imgsz, conf=self.conf,
            iou=self.iou, max_det=self.max_det, classes=self.class_ids,
            verbose=False)
        r = results[0]
        ms = float(sum(r.speed.values()))
        b = r.boxes
        dets = select_boxes(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                            b.cls.cpu().numpy(), self.names, self.min_box_px)
        return FrameResult(best=dets[0] if dets else None,
                           boxes=tuple(dets), inference_ms=ms)
