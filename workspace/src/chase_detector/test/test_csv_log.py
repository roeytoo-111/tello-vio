"""CSV schema round-trip, including the miss-row convention."""
import csv
import math

import pytest

from chase_detector.csv_log import COLUMNS, DetectionCsvLogger, default_csv_path


def test_roundtrip_detection_and_miss(tmp_path):
    p = tmp_path / 'out.csv'
    log = DetectionCsvLogger(str(p))
    log.write(seq=0, t_capture_ns=1000, t_receive_ns=1500, transport_ms=0.0005,
              inference_ms=9.7, detected=1, class_name='drone',
              confidence=0.91, xmin=100.0, ymin=200.0, xmax=260.0, ymax=320.0,
              cx=180.0, cy=260.0, w=160.0, h=120.0, area_frac=0.0278,
              range_m=1.034, az_deg=-2.5, el_deg=1.1, image_w=960, image_h=720)
    log.write(seq=1, t_capture_ns=2000, t_receive_ns=2400, transport_ms=0.0004,
              inference_ms=9.5, detected=0, class_name=None, confidence=None,
              xmin=None, ymin=None, xmax=None, ymax=None, cx=None, cy=None,
              w=None, h=None, area_frac=None, range_m=math.nan,
              az_deg=math.nan, el_deg=math.nan, image_w=960, image_h=720)
    assert log.close() == str(p)

    rows = list(csv.reader(open(p)))
    assert rows[0] == COLUMNS
    assert len(rows) == 3
    det = dict(zip(COLUMNS, rows[1]))
    assert det['detected'] == '1'
    assert det['range_m'] == '1.0340'
    miss = dict(zip(COLUMNS, rows[2]))
    assert miss['detected'] == '0'
    # A miss reports nothing it does not know: empty, never 0.0 or 'nan'.
    for field in ('confidence', 'xmin', 'cx', 'range_m', 'az_deg'):
        assert miss[field] == ''


def test_unknown_field_is_an_error(tmp_path):
    log = DetectionCsvLogger(str(tmp_path / 'x.csv'))
    with pytest.raises(KeyError):
        log.write(seq=0, bogus=1)
    log.close()


def test_default_path_is_timestamped_inside_dir(tmp_path):
    p = default_csv_path(str(tmp_path))
    assert p.startswith(str(tmp_path))
    assert p.endswith('.csv') and 'detections_' in p
