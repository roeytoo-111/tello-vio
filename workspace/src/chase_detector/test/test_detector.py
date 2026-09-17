"""Pure detection logic, plus one real-model integration test."""
import pathlib

import numpy as np
import pytest

from chase_detector.detector import (Detection, resolve_class_ids,
                                     select_boxes)

REPO = pathlib.Path(__file__).resolve().parents[4]

COCO_SLICE = {0: 'person', 4: 'airplane', 14: 'bird', 33: 'kite'}


def test_resolve_is_case_insensitive_and_dedupes():
    assert resolve_class_ids(COCO_SLICE, ['Airplane', 'bird', 'BIRD']) == [4, 14]


def test_resolve_skips_empty_entries():
    assert resolve_class_ids(COCO_SLICE, ['', '  ', 'person']) == [0]


def test_resolve_missing_class_is_loud_and_names_the_options():
    # THE trap: asking stock COCO weights for 'drone'.
    with pytest.raises(ValueError) as e:
        resolve_class_ids(COCO_SLICE, ['drone'])
    msg = str(e.value)
    assert 'drone' in msg and 'airplane' in msg


def make_arrays(rows):
    xyxy = np.array([r[:4] for r in rows], dtype=float).reshape(-1, 4)
    conf = np.array([r[4] for r in rows], dtype=float)
    cls = np.array([r[5] for r in rows], dtype=float)
    return xyxy, conf, cls


def test_select_sorts_by_confidence_desc():
    xyxy, conf, cls = make_arrays([
        (0, 0, 50, 50, 0.30, 4),
        (100, 100, 200, 180, 0.90, 4),
        (300, 300, 340, 330, 0.60, 14),
    ])
    out = select_boxes(xyxy, conf, cls, COCO_SLICE, min_box_px=4.0)
    assert [d.confidence for d in out] == [0.90, 0.60, 0.30]
    assert out[0].class_name == 'airplane'


def test_select_drops_degenerate_slivers():
    xyxy, conf, cls = make_arrays([
        (10, 10, 12, 300, 0.95, 4),   # 2 px wide: decode artefact
        (10, 10, 300, 12, 0.95, 4),   # 2 px tall
        (10, 10, 60, 60, 0.40, 4),
    ])
    out = select_boxes(xyxy, conf, cls, COCO_SLICE, min_box_px=4.0)
    assert len(out) == 1 and out[0].confidence == 0.40


def test_select_empty_input():
    out = select_boxes(np.zeros((0, 4)), np.zeros(0), np.zeros(0),
                       COCO_SLICE, 4.0)
    assert out == []


def test_detection_derived_properties():
    d = Detection(10.0, 20.0, 110.0, 70.0, 0.5, 4, 'airplane')
    assert d.width == 100.0 and d.height == 50.0
    assert d.center == (60.0, 45.0)


@pytest.mark.slow
def test_real_model_end_to_end_cpu():
    """The full wrapper against the repo weights on a real photo. Proves the
    ultralytics call signature and result plumbing, not detection quality --
    stock COCO has no drone class, so no content assertion is made."""
    weights = REPO / 'yolov8n.pt'
    photo = REPO / 'readme' / 'drone_a.jpg'
    if not weights.is_file() or not photo.is_file():
        pytest.skip('repo weights/photo not present')
    ultralytics = pytest.importorskip('ultralytics')
    assert ultralytics is not None
    cv2 = pytest.importorskip('cv2')
    from chase_detector.detector import YoloDroneDetector

    det = YoloDroneDetector(str(weights), device='cpu', conf=0.10)
    bgr = cv2.imread(str(photo))          # imread returns BGR: the contract
    assert bgr is not None
    r = det.detect(bgr)
    assert r.inference_ms > 0
    for b in r.boxes:
        assert 0 <= b.xmin <= b.xmax <= bgr.shape[1]
        assert 0 <= b.ymin <= b.ymax <= bgr.shape[0]
        assert 0.10 <= b.confidence <= 1.0
    if r.boxes:
        assert r.best == r.boxes[0]


@pytest.mark.slow
def test_real_model_refuses_drone_class_on_coco_weights():
    weights = REPO / 'yolov8n.pt'
    if not weights.is_file():
        pytest.skip('repo weights not present')
    pytest.importorskip('ultralytics')
    from chase_detector.detector import YoloDroneDetector
    with pytest.raises(ValueError):
        YoloDroneDetector(str(weights), device='cpu',
                          target_classes=['drone'])
