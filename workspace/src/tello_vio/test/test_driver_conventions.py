"""Cross-package check: the driver's conversions must match ``tello_model``.

The ``tello`` driver deliberately does not import ``tello_vio`` -- a driver
should not depend on an estimator -- so the FRD->FLU and NED->ENU conversions
exist in both packages. That duplication is a standing risk: if one is edited
and the other is not, the estimator silently receives data in a convention it
does not expect, and the failure looks like drift rather than like a bug.

This test extracts the driver's conversion functions from source and compares
them numerically against ``tello_model``, so the two cannot diverge unnoticed.
"""
import ast
import math
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tello_vio import lie
from tello_vio.tello_model import (G0, R_FLU_FRD, TelloUnits, frd_to_flu,
                                   tello_attitude_to_quat)

DRIVER = Path(__file__).resolve().parents[2] / "tello" / "tello" / "node.py"


def _extract(func_name: str):
    """Pull one self-contained function out of the driver without importing it.

    Importing node.py requires rclpy and djitellopy, neither of which is needed
    to check arithmetic.
    """
    if not DRIVER.exists():
        pytest.skip(f"driver source not found at {DRIVER}")
    tree = ast.parse(DRIVER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            src = textwrap.dedent(ast.get_source_segment(DRIVER.read_text(), node))
            # Drop the @staticmethod decorator line if present.
            src = "\n".join(l for l in src.splitlines()
                            if not l.strip().startswith("@"))
            # Supply the module-level constants the function closes over.
            ns = {"math": math, "numpy": np, "DEG": _constant("DEG"),
                  "G0": _constant("G0"),
                  "MILLI_G_TO_MPS2": _constant("MILLI_G_TO_MPS2"),
                  "FRD_TO_FLU": _constant("FRD_TO_FLU")}
            exec(src, ns)
            return ns[func_name]
    pytest.fail(f"{func_name} not found in {DRIVER}")


def _constant(name: str):
    if not DRIVER.exists():
        pytest.skip("driver source not found")
    tree = ast.parse(DRIVER.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            ns = {"math": math, "numpy": np, "G0": G0}
            exec(compile(ast.Module([node], []), "<driver>", "exec"), ns)
            return ns[name]
    pytest.fail(f"constant {name} not found in {DRIVER}")


def test_driver_uses_the_same_accel_scale():
    assert np.isclose(_constant("MILLI_G_TO_MPS2"), TelloUnits().accel_to_mps2)
    # And the value itself is right: a resting axis reads ~1000 milli-g = 1 g.
    assert np.isclose(1000.0 * _constant("MILLI_G_TO_MPS2"), 9.80665)


def test_driver_uses_the_same_frame_flip():
    assert np.allclose(_constant("FRD_TO_FLU"), R_FLU_FRD)
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(_constant("FRD_TO_FLU") @ v, frd_to_flu(v))


def test_resting_drone_reports_gravity_upward_in_ros_axes():
    """The end-to-end statement of correctness for the accelerometer path."""
    raw_milli_g = np.array([0.0, 0.0, -1000.0])     # what a level Tello reports
    flu = _constant("FRD_TO_FLU") @ (raw_milli_g * _constant("MILLI_G_TO_MPS2"))
    assert np.allclose(flu, [0.0, 0.0, 9.80665], atol=1e-6), flu


@pytest.mark.parametrize("roll,pitch,yaw", [
    (0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10),
    (-15, 25, 130), (5, -5, -179), (0, 0, 180), (30, -20, 90),
])
def test_driver_attitude_matches_tello_model(roll, pitch, yaw):
    fn = _extract("_attitude_quaternion")
    x, y, z, w = fn(roll, pitch, yaw)
    q_driver = lie.quat_normalize(np.array([w, x, y, z]))   # -> Hamilton order
    q_model = tello_attitude_to_quat(roll, pitch, yaw)
    # Compare rotations, not quaternion components: q and -q are the same
    # rotation and either sign is a valid encoding.
    ang = np.linalg.norm(lie.Log(lie.quat_to_rot(q_model).T @ lie.quat_to_rot(q_driver)))
    assert ang < 1e-9, f"{np.degrees(ang)} deg apart"


def test_ned_to_enu_actually_flips_yaw_and_pitch():
    """Guards the conversion itself, not just agreement between two copies."""
    # Nose-up 10 deg in the SDK's NED convention is nose-up +10 in ENU too...
    q = tello_attitude_to_quat(0, 10, 0)
    _, pitch, _ = lie.rot_to_euler_zyx(lie.quat_to_rot(q))
    assert np.isclose(np.degrees(pitch), -10.0, atol=1e-9)
    # ... and a clockwise (SDK-positive) yaw is a NEGATIVE yaw in right-handed ENU.
    q = tello_attitude_to_quat(0, 0, 30)
    yaw, _, _ = lie.rot_to_euler_zyx(lie.quat_to_rot(q))
    assert np.isclose(np.degrees(yaw), -30.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Frame de-duplication
# --------------------------------------------------------------------------- #

def _fake_decoder(n_frames, static=True, seed=0):
    """Mimic djitellopy's BackgroundFrameRead: a NEW array object per decode.

    `self.frame = np.array(frame.to_image())` rebinds to a freshly allocated
    array every time, whether or not the picture changed.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (72, 96, 3), dtype=np.uint8)
    out = []
    for _ in range(n_frames):
        img = base.copy() if static else rng.integers(0, 255, base.shape, dtype=np.uint8)
        out.append(img)
    return out


def test_identity_dedup_accepts_every_real_frame_of_a_static_scene():
    """The regression this replaces: a Tello pointed at a wall.

    The old detector hashed a handful of sub-sampled pixels. On a static scene
    consecutive frames hash identically, so genuinely new frames were discarded
    -- observed in flight as ~5 FPS published out of ~30 decoded. Object
    identity has no such failure mode.
    """
    frames = _fake_decoder(50, static=True)

    # What the driver does now.
    prev, published = None, 0
    for f in frames:
        if f is prev:
            continue
        prev = f
        published += 1
    assert published == 50, "identity check must accept every decoded frame"

    # What the driver used to do, reproduced to show why it failed.
    def sampled_hash(fr):
        return (fr.shape,
                int(fr[::max(1, fr.shape[0] // 8), ::max(1, fr.shape[1] // 8)].sum()))

    prev_h, published_old = None, 0
    for f in frames:
        h = sampled_hash(f)
        if h == prev_h:
            continue
        prev_h = h
        published_old += 1
    assert published_old == 1, \
        "the sampled-pixel hash should collapse a static scene to one frame"


def test_identity_dedup_still_rejects_a_repolled_frame():
    """The property that made de-duplication worth having in the first place."""
    frames = _fake_decoder(3, static=False, seed=1)
    polls = [frames[0], frames[0], frames[0], frames[1], frames[1], frames[2]]

    prev, published, skipped = None, 0, 0
    for f in polls:
        if f is prev:
            skipped += 1
            continue
        prev = f
        published += 1
    assert (published, skipped) == (3, 3)


def test_identity_dedup_accepts_a_changing_scene():
    frames = _fake_decoder(30, static=False, seed=2)
    prev, published = None, 0
    for f in frames:
        if f is prev:
            continue
        prev = f
        published += 1
    assert published == 30


# --------------------------------------------------------------------------- #
# Mission Pad ground truth (Tello EDU / RoboMaster TT, SDK 2.0+)
# --------------------------------------------------------------------------- #

def _driver_src():
    if not DRIVER.exists():
        pytest.skip("driver source not found")
    return DRIVER.read_text()


def test_mission_pad_sentinels_match_the_message_definition():
    """Driver sentinels must agree with TelloMissionPad.msg or consumers
    silently misread 'no pad' as a real pad id."""
    msg = (DRIVER.parents[2] / "tello_msg" / "msg" / "TelloMissionPad.msg")
    if not msg.exists():
        pytest.skip("TelloMissionPad.msg not found")
    text = msg.read_text()
    for name, value in (("PAD_UNSUPPORTED", -3), ("PAD_DISABLED", -2),
                        ("PAD_NONE", -1)):
        # Whitespace-tolerant: the .msg aligns the '=' for readability.
        assert re.search(rf"int32\s+{name}\s*=\s*{value}\b", text), \
            f"{name} missing or renamed in TelloMissionPad.msg"
    src = _driver_src()
    for name in ("PAD_UNSUPPORTED", "PAD_DISABLED", "PAD_NONE"):
        assert f"TelloMissionPad.{name}" in src, f"driver never sets {name}"


def test_mission_pad_units_are_converted_from_centimetres():
    """The SDK reports cm; ROS requires metres (REP-103)."""
    src = _driver_src()
    assert "/ 100.0 * self.mission_pad_signs" in src, \
        "mission pad position must be divided by 100 (cm -> m)"


def test_mission_pad_mpry_order_is_corrected():
    """'mpry' is pitch,yaw,roll -- NOT roll,pitch,yaw.

    Reading it in the conventional order silently swaps two axes, which looks
    plausible on a level drone and is wrong the moment it banks.
    """
    src = _driver_src()
    assert "pitch_d, yaw_d, roll_d" in src, \
        "mpry must be unpacked as (pitch, yaw, roll) per the SDK"


def test_mission_pad_axis_signs_are_configurable_not_hardcoded():
    """The SDK documents the pad origin and plane but not the axis signs, so
    they must be a parameter the user can verify, like speed_to_mps."""
    src = _driver_src()
    assert "mission_pad_axis_signs" in src
    assert "p('mission_pad_axis_signs'" in src, "must be a declared parameter"


def test_mission_pad_conversion_arithmetic():
    """Reproduce the driver's conversion and check a concrete case."""
    signs = np.array([1.0, -1.0, 1.0])
    raw_cm = np.array([50, 30, 80], dtype=float)     # 50 cm fwd, 30 right, 80 up
    pos = raw_cm / 100.0 * signs
    # Forward stays positive; the SDK's right-positive y becomes ROS left-positive.
    assert np.isclose(pos[0], 0.50)
    assert np.isclose(pos[1], -0.30)
    assert np.isclose(pos[2], 0.80)


# --------------------------------------------------------------------------- #
# Drone capability probe: reply classification
# --------------------------------------------------------------------------- #

def _load_probe():
    """Import scripts/probe_drone.py without it being an installed package."""
    import importlib.util
    path = DRIVER.parents[4] / "scripts" / "probe_drone.py"
    if not path.exists():
        pytest.skip(f"probe script not found at {path}")
    spec = importlib.util.spec_from_file_location("probe_drone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("reply", ["ok", "OK", " ok ", "Ok\r\n"])
def test_probe_accepts_only_the_success_token(reply):
    assert _load_probe().is_ok(reply) is True


@pytest.mark.parametrize("reply", [
    "error",
    "unknown command: mon",        # what a standard Tello actually says
    "unknown command: sdk?",
    "out of range",
    "error Not joystick",
    "error Motor stop",
    "",
    None,
])
def test_probe_rejects_every_form_of_refusal(reply):
    """Regression: an 'error'-prefix test classified "unknown command: mon" as
    SUCCESS and told a standard Tello it supported Mission Pads."""
    assert _load_probe().is_ok(reply) is False


def test_probe_does_not_use_a_prefix_or_substring_test():
    """Guard the implementation, not just its current outputs."""
    src = (DRIVER.parents[4] / "scripts" / "probe_drone.py").read_text()
    assert 'startswith("error")' not in src, "prefix test reintroduced"
    assert "strip().lower() == \"ok\"" in src, "exact-match test removed"
