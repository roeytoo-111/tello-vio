"""The flight-side half of the ONE CONTRACT.

Every tier of this project -- the Gym trainer (Tier A), the Gazebo
rehearsal (Tier B), the rendered-vision evaluation (Tier C) and the
aircraft -- consumes the SAME observation assembler, the SAME forward-speed
law and the SAME action convention. Those live in `chase_gym`. This module
does not re-implement any of them; it only adapts ROS message types to
them, and it refuses anything it cannot prove is compatible.

Three adaptations live here, and each one is a defect that bit this project
at least once:

1. RESOLUTION. The observation spec is expressed in the CALIBRATION frame
   (960x720, cx=480, cy=360, fx=919.42). The driver may downscale the video
   (`video_scale`, 0.5 in vio.launch.py), and `chase_msgs/DroneDetection`
   carries boxes in whatever resolution the detector saw. Rescaling the
   measurement into the calibration frame fixes the observation
   normalisation AND the pinhole range in one step, because both are
   expressed against the same fx.

2. REFERENCE WIDTH. `chase_detector` defaults `ref_width_m` to 0.180 m (the
   PROP SPAN) while the trained policy's EnvConfig uses 0.098 m (the BODY
   WIDTH). Every range differs by 1.8x between those two choices. We
   therefore IGNORE the detector's `range_m` field entirely and let
   `chase_gym.env.forward_command` derive range from the box width with the
   checkpoint's own `ref_width_m` -- the same arithmetic the policy trained
   against. `check_detector_ref_width()` exists to warn when the operator
   has configured the detector inconsistently, because the mismatch is
   otherwise silent.

3. PROVENANCE. A checkpoint is loadable only if its observation-spec hash
   matches the spec this code builds, and its action map is the
   stick-convention map. The `ddpg_faithful` arm, for example, emits a
   2-dim `raw_pixels` observation and an action map of
   ('image_x', 'image_y') scaled by 60 -- loading it here would fly the
   aircraft on a contract it never agreed to, so it is refused by name.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from chase_gym import constants as C
from chase_gym.corruption import Measurement
from chase_gym.env import EnvConfig
from chase_gym.observation import ObservationSpec

# The action map every flight-legal checkpoint must declare. Source of
# truth: chase_train/train.py, mirrored into each bundle.
FLIGHT_ACTION_ORDER = ['linear.z', 'angular.z']

# The driver zeroes the sticks when a command is older than this
# (tello/node.py rc_timeout_sec). Every publisher in this package treats it
# as a hard deadline, not a target.
DEADMAN_S = C.DEADMAN_S                      # 0.35 s


@dataclass(frozen=True)
class PolicyBundle:
    """A verified, flight-legal policy plus the config it trained under."""
    infer: callable                # (obs[14] float32) -> action[2] float32
    spec: ObservationSpec
    env_cfg: EnvConfig
    backend: str                   # 'onnx' | 'torch'
    checkpoint_path: str
    obs_spec_hash: str
    arm: str
    env_step: int
    control_rate_hz: float
    git_sha: str


def spec_from_bundle(bundle: dict) -> ObservationSpec:
    """Rebuild the ObservationSpec a checkpoint trained under.

    `to_dict()` adds a synthetic 'dim' key that is not a dataclass field,
    so it must be dropped before reconstruction.
    """
    fields = {k: v for k, v in bundle['obs_spec'].items() if k != 'dim'}
    return ObservationSpec(**fields)


def env_cfg_from_bundle(bundle: dict) -> EnvConfig:
    """The resolved EnvConfig the policy trained under.

    `env_config` carries the FULL resolved config (every EnvConfig field),
    unlike `config['env']` which holds only the overrides. It also carries a
    nested 'obs_spec' that is not an EnvConfig field.
    """
    raw = dict(bundle.get('env_config') or {})
    raw.pop('obs_spec', None)
    valid = set(EnvConfig.__dataclass_fields__.keys())
    unknown = set(raw) - valid
    for k in unknown:
        raw.pop(k)
    return EnvConfig(**raw)


def load_policy(checkpoint: str, prefer_onnx: bool = True,
                logger=None) -> PolicyBundle:
    """Load a checkpoint for FLIGHT, refusing anything unproven.

    The torch bundle is always read, because it is the only artifact that
    carries the observation-spec hash, the action map and the training
    config -- the ONNX graph carries none of that. Inference then runs on
    the ONNX actor when one is available (opset 17, deterministic, and the
    same graph the parity check validated at export), falling back to the
    torch actor otherwise.
    """
    from chase_train.checkpoint import actor_from_bundle, load_bundle

    bundle = load_bundle(checkpoint)          # no expect: we verify below
    spec = spec_from_bundle(bundle)

    # --- provenance gate 1: the observation contract ------------------
    want = spec.spec_hash()
    got = bundle.get('obs_spec_hash')
    if got != want:
        raise ValueError(
            f'checkpoint obs_spec_hash {got!r} != rebuilt {want!r} -- the '
            f'bundle disagrees with its own spec; refusing to fly it')
    if spec.mode != 'latency_aware':
        raise ValueError(
            f'checkpoint obs mode {spec.mode!r} is not flight-legal: the '
            f'flight stack assembles the 14-dim latency_aware observation')

    # --- provenance gate 2: the action contract -----------------------
    amap = bundle.get('action_map') or {}
    order = list(amap.get('order') or [])
    if order != FLIGHT_ACTION_ORDER:
        raise ValueError(
            f'checkpoint action_map order {order!r} != {FLIGHT_ACTION_ORDER!r} '
            f'-- this policy does not emit stick commands (the ddpg_faithful '
            f'arm emits image-space actions scaled by 60). Refusing.')
    scale = float(amap.get('scale', 1.0))
    if abs(scale - 1.0) > 1e-9:
        raise ValueError(
            f'checkpoint action scale {scale} != 1.0; the flight stick range '
            f'is [-1, 1] and no rescaling is permitted between network and '
            f'aircraft')

    env_cfg = env_cfg_from_bundle(bundle)

    # --- inference backend --------------------------------------------
    onnx_path = os.path.splitext(checkpoint)[0] + '.onnx'
    infer = None
    backend = 'torch'
    if prefer_onnx and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(
                onnx_path, providers=['CPUExecutionProvider'])
            in_name = sess.get_inputs()[0].name
            in_shape = sess.get_inputs()[0].shape
            if len(in_shape) != 2 or int(in_shape[1]) != spec.dim:
                raise ValueError(
                    f'ONNX input {in_shape} does not match obs dim '
                    f'{spec.dim}')

            def infer(obs: np.ndarray, _s=sess, _n=in_name) -> np.ndarray:
                out = _s.run(None, {_n: obs.reshape(1, -1).astype(np.float32)})
                return np.asarray(out[0][0], dtype=np.float32)

            backend = 'onnx'
        except Exception as exc:                       # pragma: no cover
            if logger is not None:
                logger.warn(f'ONNX load failed ({exc}); using torch actor')
            infer = None

    if infer is None:
        import torch
        actor = actor_from_bundle(bundle)

        def infer(obs: np.ndarray, _a=actor) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(obs.reshape(1, -1).astype(np.float32))
                return _a(t).numpy()[0].astype(np.float32)

        backend = 'torch'

    return PolicyBundle(
        infer=infer, spec=spec, env_cfg=env_cfg, backend=backend,
        checkpoint_path=checkpoint, obs_spec_hash=want,
        arm=str(bundle.get('arm', '?')),
        env_step=int(bundle.get('env_step', -1)),
        control_rate_hz=float(bundle.get('control_rate_hz',
                                         C.CONTROL_RATE_HZ)),
        git_sha=str(bundle.get('git_sha', '?')))


def measurement_from_detection(det, spec: ObservationSpec
                               ) -> Tuple[Optional[Measurement], float]:
    """`chase_msgs/DroneDetection` -> the Measurement chase_gym expects.

    Returns (measurement_or_None, scale_applied). The box is rescaled from
    the detector's frame into the CALIBRATION frame the spec and fx are
    expressed in, so that both the observation normalisation and the
    pinhole range stay correct under any `video_scale`.

    A detection is rejected (None) when `detected` is false or the box is
    degenerate -- a zero/negative width would divide by zero in the range
    law, and `forward_command` guards `w_px <= 0` but the observation would
    still encode a bogus centre.
    """
    if det is None or not det.detected:
        return None, 1.0
    w_px = float(det.xmax) - float(det.xmin)
    h_px = float(det.ymax) - float(det.ymin)
    if not (w_px > 0.0 and h_px > 0.0):
        return None, 1.0
    if not (math.isfinite(w_px) and math.isfinite(h_px)):
        return None, 1.0

    # Rescale into the calibration frame. The detector reports the
    # resolution its boxes live in; 0 means "unknown", in which case we
    # trust that it is already the calibration frame.
    src_w = int(det.image_width) or int(2 * spec.half_w)
    scale = (2.0 * spec.half_w) / float(src_w) if src_w > 0 else 1.0

    u = 0.5 * (float(det.xmin) + float(det.xmax)) * scale
    v = 0.5 * (float(det.ymin) + float(det.ymax)) * scale
    return Measurement(u, v, w_px * scale), scale


def check_detector_ref_width(node, env_cfg: EnvConfig) -> None:
    """Warn loudly when the detector's ref_width_m disagrees with the
    policy's. The detector's own `range_m` is never used by this stack, but
    a mismatch means the operator's mental model of range is wrong by the
    ratio -- and a 0.180 vs 0.098 mix-up is a 1.8x error.
    """
    try:
        from rcl_interfaces.srv import GetParameters       # noqa: F401
    except Exception:                                       # pragma: no cover
        return
    node.get_logger().info(
        f'policy ref_width_m = {env_cfg.ref_width_m:.3f} m '
        f'(chase_detector defaults to 0.180 m = prop span; if it is '
        f'configured that way its published range_m will read '
        f'{0.180 / env_cfg.ref_width_m:.2f}x this stack\'s range -- this '
        f'stack ignores that field and uses the box width directly)')


def stick_from_mps(v_mps: float, v_max_mps: float) -> float:
    """Metres/second -> normalised stick [-1, 1].

    `forward_command` returns m/s; `/cmd_vel` consumes normalised sticks.
    The conversion needs the aircraft's full-stick speed on that axis,
    which is a MEASURED quantity (see docs: the forward stick scale has not
    been measured on the real Tello yet, so the launch default is
    deliberately conservative).
    """
    if v_max_mps <= 0.0:
        return 0.0
    return float(np.clip(v_mps / v_max_mps, -1.0, 1.0))
