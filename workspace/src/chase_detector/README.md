# chase_detector — YOLO inference on the Tello video stream

The perception block of the drone-chasing stack
([implementation_plan.md §2](../../../docs/drone_chasing_rl/implementation_plan.md)):
subscribe to the driver's `image_raw`, run YOLO, publish a
`chase_msgs/DroneDetection` for **every** processed frame (misses included —
`detected: false` is the visibility signal), draw the box + centre on an
annotated stream, log a per-frame CSV, and show a live window with the
box-centre traces.

```
tello driver ──image_raw──▶ detector ──chase/detection────────▶ (chase_state, later)
                 │              └──────chase/image_annotated──▶ viewer (window + cx/cy traces)
                 └────────────────────▶ capture (training-set frames)
```

## Run

```bash
source install/setup.bash
# from the repo root (weights path resolves against your cwd):
ros2 launch chase_detector chase.launch.py                     # driver+detector+viewer
ros2 launch chase_detector chase.launch.py capture:=true      # + dataset recorder
ros2 launch chase_detector chase.launch.py driver:=false      # rosbag replay
ros2 launch chase_detector chase.launch.py control:=true      # + keyboard GUI to fly
```

Per-frame CSV lands in `chase_logs/` (arg `csv_dir`, empty disables);
captured frames in `chase_dataset/` (both gitignored).

## The two traps this package exists to not fall into

Both verified against the installed packages, not folklore:

1. **Colour order.** djitellopy 2.5.0 decodes to **RGB**
   (`np.array(frame.to_image())`); ultralytics assumes numpy input is **BGR**
   (`LoadPilAndNumpy._single_check`, and `predictor.preprocess` does
   `im[..., ::-1]`). Feed one to the other raw and every frame is
   channel-swapped. Here cv_bridge normalises whatever the driver publishes
   (`rgb8` or `bgr8`) to BGR before inference.
2. **Stock COCO weights have no `drone` class** (80 classes; checked against
   this repo's `yolov8n.pt`). On a real close-up of this repo's Tello, stock
   yolov8n answers *clock 0.36 / scissors 0.25 / airplane 0.19*. Stock
   weights are therefore a **pipeline smoke test only**. The node is
   weights-agnostic: after fine-tuning (Phase 0), pass
   `weights:=/path/drone.pt target_classes:=drone` and nothing else changes.
   Asking for a class the weights don't have is a startup **error** naming
   the classes that exist — never a silently empty stream.

## Key parameters (detector)

| Param | Default | Meaning |
|---|---|---|
| `weights` | `yolov8n.pt` | path to `.pt`, resolved against cwd; must exist (no auto-download) |
| `target_classes` | `''` | comma-separated names to keep; empty = all |
| `conf` / `iou` / `imgsz` / `max_det` | 0.25 / 0.45 / 640 / 8 | ultralytics thresholds |
| `device` | `cuda:0` | falls back to CPU with a warning |
| `ref_width_m` | 0.180 | real width of the boxed object for range: 0.180 prop span, 0.098 body |
| `csv_dir` | `''` | per-frame CSV directory, empty disables |
| `publish_annotated` | `true` | the drawn stream the viewer displays |

Range is `fx · ref_width / box_width`; `fx` comes from the driver's
`camera_info` (already rescaled for `video_scale`), never hardcoded. No
camera_info yet ⇒ range/bearings are NaN, not a guess.

Measured on this machine (RTX 5050, 960×720 frames, imgsz 640): median
12.4 ms/frame, p95 16.5 ms — an order of magnitude inside the 30 fps budget;
CPU fallback ≈38 ms still keeps up.

## Tests

```bash
cd workspace/src/chase_detector && python3 -m pytest test/ -q   # 20 tests, no drone needed
```
