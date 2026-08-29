"""Fiducial ground-truth tests, against synthetically rendered markers.

The marker is rendered from a KNOWN camera pose by warping the marker image
through the exact homography that pose implies, then the pose is recovered
from the rendered pixels. That makes the ground truth of the ground truth
exact, which is the only way to state an accuracy bound honestly.
"""
import numpy as np
import cv2
import pytest

from tello_vio import lie
from tello_vio.fiducial import (detect_markers, estimate_marker_pose,
                                generate_marker_png, make_detector,
                                marker_object_points)

W, H = 960, 720
K = np.array([[919.424717, 0.0, 459.655779],
              [0.0, 911.926190, 323.551997],
              [0.0, 0.0, 1.0]])
D = np.zeros(5)
SIZE = 0.20            # 20 cm marker, i.e. most of an A4 sheet

# A marker FACING the camera is not R = I.
#
# OpenCV's marker frame has +z out of the printed face, towards the viewer.
# The camera optical frame has +z pointing away from the camera, into the
# scene. So R_cm = I would align the marker's +z with the camera's +z, i.e.
# the marker faces AWAY and we see its back: a mirrored code, which ArUco
# correctly refuses to decode. The fronto-parallel "flat on the wall in front
# of me" pose is a 180 deg rotation about x, which also flips y (marker +y up
# vs image +y down).
R_FACING = np.diag([1.0, -1.0, -1.0])


def render_marker(R_cm, t_cm, marker_px=700, size_m=SIZE):
    """Render a marker seen from a camera at the given marker->camera pose."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "generateImageMarker"):
        m = cv2.aruco.generateImageMarker(d, 0, marker_px)
    else:
        m = cv2.aruco.drawMarker(d, 0, marker_px)
    pad = marker_px // 5
    tile = np.full((marker_px + 2 * pad, marker_px + 2 * pad), 255, np.uint8)
    tile[pad:pad + marker_px, pad:pad + marker_px] = m

    # Tile pixel -> marker metres (tile spans the marker plus its quiet zone).
    span = size_m * (marker_px + 2 * pad) / marker_px
    s = span / tile.shape[0]
    src = np.float32([[0, 0], [tile.shape[1], 0],
                      [tile.shape[1], tile.shape[0]], [0, tile.shape[0]]])
    obj = np.array([[-span / 2,  span / 2, 0], [ span / 2,  span / 2, 0],
                    [ span / 2, -span / 2, 0], [-span / 2, -span / 2, 0]])
    rvec, _ = cv2.Rodrigues(R_cm)
    proj, _ = cv2.projectPoints(obj, rvec, t_cm.reshape(3, 1), K, D)
    Hm = cv2.getPerspectiveTransform(src, proj.reshape(4, 2).astype(np.float32))
    img = cv2.warpPerspective(tile, Hm, (W, H), borderValue=255)
    return img


def recover(R_cm, t_cm):
    img = render_marker(R_cm, t_cm)
    det = make_detector()
    corners, ids = detect_markers(det, img)
    if ids is None or len(ids) == 0:
        return None
    return estimate_marker_pose(corners[0], SIZE, K, D, int(ids[0][0]))


@pytest.mark.parametrize("dist", [0.8, 1.5, 2.5])
def test_recovers_known_distance(dist):
    """Straight-on view: translation must come back to ~1 cm."""
    R = R_FACING
    t = np.array([0.0, 0.0, dist])
    d = recover(R, t)
    assert d is not None, f"marker not detected at {dist} m"
    err = float(np.linalg.norm(d.t_cm - t))
    assert err < 0.02, f"{err*100:.1f} cm error at {dist} m"
    assert d.reproj_px < 2.0


@pytest.mark.parametrize("yaw_deg", [0, 15, 30, 45])
def test_translation_is_accurate_at_every_viewing_angle(yaw_deg):
    """Translation is the robust quantity: ~1.5 cm regardless of obliquity."""
    R = R_FACING @ lie.euler_zyx_to_rot(0.0, np.deg2rad(yaw_deg), 0.0)
    t = np.array([0.0, 0.0, 1.5])
    d = recover(R, t)
    assert d is not None, f"not detected at {yaw_deg} deg"
    pos_err = float(np.linalg.norm(d.t_cm - t))
    assert pos_err < 0.03, f"{pos_err*100:.1f} cm at {yaw_deg} deg"


@pytest.mark.parametrize("yaw_deg", [20, 30, 45])
def test_rotation_is_accurate_only_when_viewed_obliquely(yaw_deg):
    """Away from head-on, the planar ambiguity resolves and rotation is good."""
    R = R_FACING @ lie.euler_zyx_to_rot(0.0, np.deg2rad(yaw_deg), 0.0)
    d = recover(R, np.array([0.0, 0.0, 1.5]))
    assert d is not None
    rot_err = np.degrees(np.linalg.norm(lie.Log(R.T @ d.R_cm)))
    assert rot_err < 5.0, f"{rot_err:.1f} deg at {yaw_deg} deg obliquity"


def test_head_on_rotation_is_ambiguous_and_is_flagged():
    """The documented failure mode, asserted so it cannot silently change.

    Head-on, rotation error is large AND the ambiguity metric reports it --
    which is what lets a consumer reject the sample instead of trusting it.
    """
    d = recover(R_FACING, np.array([0.0, 0.0, 2.0]))
    assert d is not None
    rot_err = np.degrees(np.linalg.norm(lie.Log(R_FACING.T @ d.R_cm)))
    assert rot_err > 3.0, "expected the head-on ambiguity to show up"
    assert np.isfinite(d.ambiguity), "ambiguity must be measurable"
    assert d.ambiguity > 0.5, f"ambiguity {d.ambiguity:.2f} should be near 1"
    # Translation is still fine even though rotation is not.
    assert float(np.linalg.norm(d.t_cm - np.array([0.0, 0.0, 2.0]))) < 0.03


def test_camera_pose_in_marker_frame_is_the_inverse():
    """R_mc/t_mc must be the exact inverse of R_cm/t_cm -- the mirror-image bug."""
    R = R_FACING @ lie.euler_zyx_to_rot(0.1, 0.2, -0.15)
    t = np.array([0.15, -0.08, 1.8])
    d = recover(R, t)
    assert d is not None
    assert np.allclose(d.R_mc @ d.R_cm, np.eye(3), atol=1e-9)
    assert np.allclose(d.R_mc @ d.t_cm + d.t_mc, np.zeros(3), atol=1e-9)
    # Camera sits ~1.8 m from the marker along its normal.
    assert abs(np.linalg.norm(d.t_mc) - np.linalg.norm(t)) < 0.03


def test_relative_motion_is_measured_in_the_camera_frame():
    """The actual evaluation use: a 0.5 m displacement must read as 0.5 m.

    Measured via t_cm (marker in camera), NOT t_mc (camera in marker): the
    latter is contaminated by the rotation ambiguity documented above, and
    would report 0.81 m for this same 0.5 m motion when viewed head-on.
    """
    t0 = np.array([0.0, 0.0, 2.0])
    t1 = np.array([0.5, 0.0, 2.0])
    d0, d1 = recover(R_FACING, t0), recover(R_FACING, t1)
    assert d0 is not None and d1 is not None
    moved = float(np.linalg.norm(d1.t_cm - d0.t_cm))
    assert abs(moved - 0.5) < 0.03, f"measured {moved:.3f} m for a 0.5 m move"


def test_marker_png_is_written_and_redetectable(tmp_path):
    p = str(tmp_path / "marker.png")
    generate_marker_png(p, marker_id=7)
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape[0] > 1000
    corners, ids = detect_markers(make_detector(), img)
    assert ids is not None and int(ids[0][0]) == 7


def test_object_point_order_matches_detector_order():
    """Corner ordering is the classic silent-mirror bug; pin it."""
    obj = marker_object_points(0.2)
    assert obj.shape == (4, 3)
    assert np.allclose(obj[0], [-0.1, 0.1, 0.0])   # top-left
    assert np.allclose(obj[2], [0.1, -0.1, 0.0])   # bottom-right
    assert np.allclose(obj[:, 2], 0.0)             # coplanar


# --------------------------------------------------------------------------- #
# trajectory evaluation
# --------------------------------------------------------------------------- #

from tello_vio.evaluate import evaluate, resample_to   # noqa: E402


def _traj(T=20.0, hz=30.0):
    t = np.arange(0.0, T, 1.0 / hz)
    p = np.column_stack([1.5 * np.sin(0.4 * t), 0.8 * t * 0.1, 1.2 + 0.2 * np.sin(t)])
    return t, p


def test_perfect_estimate_scores_zero_error():
    t, p = _traj()
    r = evaluate(t, p, t, p)
    assert r.ate_se3_rmse < 1e-6
    assert abs(r.fitted_scale - 1.0) < 1e-6
    assert r.path_length_m > 1.0


def test_rigid_offset_is_removed_by_alignment():
    """A different world origin/heading is not an error; alignment removes it."""
    t, p = _traj()
    R = lie.euler_zyx_to_rot(0.7, 0.0, 0.0)
    est = p @ R.T + np.array([3.0, -2.0, 1.0])
    r = evaluate(t, est, t, p)
    assert r.ate_se3_rmse < 1e-6, "rigid transform must not count as error"


def test_scale_error_shows_up_in_se3_but_not_sim3():
    """The diagnostic that separates a scale problem from a shape problem."""
    t, p = _traj()
    est = p * 1.25                     # 25 % scale error, shape perfect
    r = evaluate(t, est, t, p)
    assert r.ate_se3_rmse > 0.15, "SE3 must expose the scale error"
    assert r.ate_sim3_rmse < 1e-6, "Sim3 must absorb it"
    assert abs(r.fitted_scale - 0.8) < 1e-6   # est -> gt needs 1/1.25


def test_known_drift_is_measured():
    t, p = _traj()
    drift = np.column_stack([0.02 * t, np.zeros_like(t), np.zeros_like(t)])
    r = evaluate(t, p + drift, t, p)
    assert 0.05 < r.ate_se3_rmse < 0.5
    assert r.rpe_rmse < 0.1, "RPE should stay small for slow linear drift"


def test_resample_refuses_to_interpolate_across_a_dropout():
    t_gt = np.linspace(0, 10, 101)
    t_src = np.concatenate([np.linspace(0, 3, 31), np.linspace(7, 10, 31)])
    p_src = np.column_stack([t_src, t_src, t_src])
    _, sel = resample_to(t_gt, t_src, p_src, max_gap_s=0.25)
    kept = t_gt[sel]
    assert not np.any((kept > 3.3) & (kept < 6.7)), "interpolated across the gap"
    assert sel.sum() > 40


def test_evaluate_rejects_non_overlapping_streams():
    t, p = _traj(T=5.0)
    with pytest.raises(ValueError, match="overlapping"):
        evaluate(t + 1000.0, p, t, p)
