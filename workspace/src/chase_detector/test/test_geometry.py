"""Pinhole geometry against the repo's real calibration (ost.txt)."""
import math

import pytest

from chase_detector.geometry import PinholeGeometry

# workspace/src/tello/resource/ost.txt, 960x720
FX, FY, CX, CY = 919.424717, 911.926190, 459.655779, 323.551997


def real_geom():
    return PinholeGeometry(FX, FY, CX, CY)


def test_from_k_reads_row_major_layout():
    k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    g = PinholeGeometry.from_k(k)
    assert (g.fx, g.fy, g.cx, g.cy) == (FX, FY, CX, CY)


def test_from_k_rejects_uncalibrated_zero_matrix():
    assert PinholeGeometry.from_k([0.0] * 9) is None
    assert PinholeGeometry.from_k([]) is None


def test_fov_matches_calibrated_tello():
    g = real_geom()
    # 55.1 H / 43.1 V, the values recorded for this camera.
    assert g.hfov_deg(960) == pytest.approx(55.1, abs=0.15)
    assert g.vfov_deg(720) == pytest.approx(43.1, abs=0.15)


def test_range_similar_triangles():
    g = real_geom()
    # 180 mm prop span filling 160 px at fx=919.4 -> 1.034 m
    assert g.range_from_width(160.0, 0.180) == pytest.approx(1.034, abs=0.002)
    # Half the pixels, twice the range.
    assert g.range_from_width(80.0, 0.180) == \
        pytest.approx(2 * g.range_from_width(160.0, 0.180))


def test_range_invalid_inputs_are_nan_not_garbage():
    g = real_geom()
    assert math.isnan(g.range_from_width(0.0, 0.180))
    assert math.isnan(g.range_from_width(-5.0, 0.180))
    assert math.isnan(g.range_from_width(100.0, 0.0))


def test_bearings_zero_at_principal_point():
    az, el = real_geom().bearings_deg(CX, CY)
    assert az == pytest.approx(0.0)
    assert el == pytest.approx(0.0)


def test_bearing_sign_and_magnitude():
    g = real_geom()
    u = CX + FX * math.tan(math.radians(10.0))
    az, _ = g.bearings_deg(u, CY)
    assert az == pytest.approx(10.0, abs=1e-6)   # +right
    _, el = g.bearings_deg(CX, CY - 100.0)
    assert el < 0                                 # above axis -> negative


def test_rejects_nonpositive_focal():
    with pytest.raises(ValueError):
        PinholeGeometry(0.0, FY, CX, CY)
