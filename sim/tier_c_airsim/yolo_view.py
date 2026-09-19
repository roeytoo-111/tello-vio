"""See a YOLO detector's boxes on the LIVE warehouse scene.

Builds the warehouse, places the target at a spread of ranges down the aisle,
captures each frame, runs a YOLO detector on it, and saves annotated images
plus a montage:
    GREEN box  = the engine's ground-truth target box (always correct)
    YELLOW box = the detector's output, with confidence
    IoU printed per frame = overlap of the best detection with the truth
      (0.00 means the detector missed the drone entirely)

Two detector backends:
    --yolo  weights.pt     ultralytics (a trained/fine-tuned YOLO)
    --onnx  weights.onnx   raw onnxruntime for a DeepStream-style export
                           ([1,N,6] = x1,y1,x2,y2,conf,cls), CPU

    C:\\ProjectAirSim\\Blocks\\Blocks.exe                 # engine, on Windows
    ~/.venvs/tier_c/bin/python yolo_view.py --yolo runs/detect/train/weights/best.pt
    ~/.venvs/tier_c/bin/python yolo_view.py --onnx ~/shared/yolo-thermal-gan-ds/uav_yolo26s_640_op17_ds.onnx
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import make_engine  # noqa: E402
import warehouse  # noqa: E402


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class UltralyticsDet:
    def __init__(self, weights, device='cpu', conf=0.25):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.device = device
        self.conf = conf

    def __call__(self, bgr):
        r = self.m.predict(bgr, device=self.device, conf=self.conf,
                           verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append((np.array([x1, y1, x2, y2]), float(b.conf[0])))
        return out


class OnnxDet:
    """Raw onnxruntime for a DeepStream-style [1,N,6] export (CPU)."""
    def __init__(self, weights, conf=0.25, imgsz=640):
        import onnxruntime as ort
        self.s = ort.InferenceSession(weights, providers=['CPUExecutionProvider'])
        self.inp = self.s.get_inputs()[0].name
        self.conf = conf
        self.sz = imgsz

    def _letterbox(self, im):
        h, w = im.shape[:2]
        r = min(self.sz / h, self.sz / w)
        nh, nw = int(round(h*r)), int(round(w*r))
        c = np.full((self.sz, self.sz, 3), 114, np.uint8)
        t, l = (self.sz-nh)//2, (self.sz-nw)//2
        c[t:t+nh, l:l+nw] = cv2.resize(im, (nw, nh))
        return c, r, l, t

    @staticmethod
    def _nms(boxes, scores, thr=0.5):
        idx = scores.argsort()[::-1]
        keep = []
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
            w = np.clip(xx2-xx1, 0, None)
            h = np.clip(yy2-yy1, 0, None)
            inter = w*h
            a = (boxes[i, 2]-boxes[i, 0])*(boxes[i, 3]-boxes[i, 1])
            b = (boxes[idx[1:], 2]-boxes[idx[1:], 0])*(boxes[idx[1:], 3]-boxes[idx[1:], 1])
            iou = inter/(a+b-inter+1e-9)
            idx = idx[1:][iou < thr]
        return keep

    def __call__(self, bgr):
        cv, r, l, t = self._letterbox(bgr)
        x = cv[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)/255.0
        out = self.s.run(None, {self.inp: x})[0][0]
        m = out[:, 4] >= self.conf
        d = out[m]
        if not len(d):
            return []
        b = d[:, :4].copy()
        b[:, [0, 2]] = (b[:, [0, 2]]-l)/r
        b[:, [1, 3]] = (b[:, [1, 3]]-t)/r
        keep = self._nms(b, d[:, 4])
        return [(b[i], float(d[i, 4])) for i in keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--yolo', help='ultralytics weights (.pt)')
    ap.add_argument('--onnx', help='DeepStream-style ONNX weights')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--lighting', default='medium',
                    choices=['low', 'medium', 'high'])
    ap.add_argument('--ranges', default='1,2,3,4,6',
                    help='comma-separated target ranges, metres')
    ap.add_argument('--out', default='yolo_view')
    ap.add_argument('--address', default=None)
    args = ap.parse_args(argv)
    if not (args.yolo or args.onnx):
        ap.error('give --yolo weights.pt or --onnx weights.onnx')
    det = (OnnxDet(args.onnx, args.conf) if args.onnx
           else UltralyticsDet(args.yolo, args.device, args.conf))
    name = os.path.basename(args.onnx or args.yolo)

    os.makedirs(args.out, exist_ok=True)
    eng = make_engine('projectairsim',
                      **({'address': args.address} if args.address else {}))
    eng.connect()
    warehouse.build_and_light(eng, seed=20, lighting=args.lighting)
    eng.takeoff()
    eng.move_by_velocity_body(0.0, 0.0, -0.1, 0.0, 0.4)
    tiles, hits = [], 0
    ranges = [float(x) for x in args.ranges.split(',')]
    try:
        for rng in ranges:
            (x, y, z), yaw = eng.get_vehicle_pose()
            eng.set_object_pose('TargetTello',
                                (x + rng*math.cos(yaw), y + rng*math.sin(yaw), z),
                                yaw)
            import time
            time.sleep(0.8)
            eng.get_rgb()
            boxes = eng.get_bboxes()
            img = eng.get_rgb()
            gt = None
            if boxes:
                b = boxes[0]
                gt = np.array([b.xmin, b.ymin, b.xmax, b.ymax])
                cv2.rectangle(img, (int(b.xmin), int(b.ymin)),
                              (int(b.xmax), int(b.ymax)), (0, 255, 0), 2)
            dets = det(img)
            best_iou = 0.0
            for box, c in dets:
                cv2.rectangle(img, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 255), 2)
                cv2.putText(img, f'{c:.2f}', (int(box[0]), int(box[1])-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if gt is not None:
                    best_iou = max(best_iou, _iou(gt, box))
            hit = best_iou >= 0.3
            hits += hit
            cv2.putText(img, f'{rng:.0f}m  det={len(dets)}  IoU={best_iou:.2f}'
                        f'  {"HIT" if hit else "miss"}', (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(args.out, f'r{rng:.0f}.png'), img)
            tiles.append(cv2.resize(img, (480, 360)))
            print(f'{rng:>4}m: detections={len(dets)} best_IoU={best_iou:.2f} '
                  f'{"HIT" if hit else "miss"}')
    finally:
        eng.disconnect()
    if tiles:
        row = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
        wmax = max(r.shape[1] for r in row)
        row = [np.pad(r, ((0, 0), (0, wmax-r.shape[1]), (0, 0))) for r in row]
        cv2.imwrite(os.path.join(args.out, 'montage.png'), np.vstack(row))
    print(f'\n{name}: hit the drone in {hits}/{len(ranges)} frames '
          f'(IoU>=0.3). Annotated frames -> {args.out}/')


if __name__ == '__main__':
    main()
