"""Engine adapter for Tier C (sim_training_architecture.md 4): ONE
interface, two AirSim-lineage backends, so the dataset factory and the
vision-in-the-loop evaluation are engine-portable.

Backends, per the verified 2026 survey (offline_training_recipe.md 6.3 +
API facts verified at source 2026-09-17):

* ProjectAirSimEngine (DEFAULT): iamaisim/ProjectAirSim, pip
  `projectairsim` (1.0.2 on PyPI), prebuilt Linux env binaries with
  -RenderOffScreen. NATIVE 2D bbox annotations (camera
  "annotation-settings") -- no segmentation-mask step needed.
* ClassicEngine (FALLBACK): the classic `airsim` msgpack-rpc API --
  Colosseum (ARCHIVED 2026-07-11, pin it) or the active Cosys-AirSim
  fork (`pip install cosysairsim`, 3.5.0). Bboxes via the Detection API,
  segmentation-mask fallback included.

The three verified portability traps this file exists to contain:
  1. yaw units -- Project AirSim: RADIANS (rad/s); classic: DEGREES.
     The adapter's contract is SI (rad, rad/s), converted per backend.
  2. ImageType.Segmentation -- Project AirSim: 3; classic: 5. Never
     hardcode the int downstream.
  3. Project AirSim's command API is asyncio; classic returns futures.
     The adapter is synchronous.

Coordinates: both engines are NED, m, m/s (verified). The adapter keeps
NED; callers convert to/from the project's REP-103 world as needed.
"""
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BBox:
    """Axis-aligned 2D box, pixels, in the capture resolution."""
    object_id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def w(self) -> float:
        return self.xmax - self.xmin

    @property
    def h(self) -> float:
        return self.ymax - self.ymin


class EngineBase:
    """connect/disconnect, teleport, capture, command -- the whole surface
    the two Tier-C jobs need."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def set_object_pose(self, name: str, ned_xyz, yaw_rad: float) -> None: ...
    def set_vehicle_pose(self, ned_xyz, yaw_rad: float) -> None: ...
    def get_rgb(self) -> np.ndarray: ...
    def get_bboxes(self) -> List[BBox]: ...
    def move_by_velocity(self, v_north: float, v_east: float, v_down: float,
                         yaw_rate_rad_s: float, duration_s: float) -> None: ...
    def move_by_velocity_body(self, v_fwd: float, v_right: float,
                              v_down: float, yaw_rate_rad_s: float,
                              duration_s: float) -> None:
        """Body-frame velocity + yaw rate -- the Tello stick semantics.
        Preferred for closed-loop control: no client-side yaw
        dead-reckoning (which drifts against the engine's true heading)."""
        ...
    def hover(self) -> None: ...


class ProjectAirSimEngine(EngineBase):
    """Project AirSim backend. Scene/robot JSONC configs live in
    settings/; the camera there is 960x720 FOV 55.1 deg [V ost.txt] with
    annotation-settings enabled for the target object ids."""

    def __init__(self, scene_config: str = 'scene_chase.jsonc',
                 sim_config_path: str = 'settings',
                 address: str = '127.0.0.1',
                 vehicle: str = 'Follower', camera: str = 'FrontCamera',
                 target_ids: Tuple[str, ...] = ('TargetTello',)):
        self.scene_config = scene_config
        self.sim_config_path = sim_config_path
        self.address = address
        self.vehicle = vehicle
        self.camera = camera
        self.target_ids = target_ids
        self._client = None
        self._world = None
        self._drone = None

    def connect(self) -> None:
        # Lazy import: this module must be importable (and dry-runnable)
        # on machines without any engine installed.
        from projectairsim import ProjectAirSimClient, World, Drone
        self._client = ProjectAirSimClient(address=self.address)
        self._client.connect()
        self._world = World(self._client, self.scene_config,
                            sim_config_path=self.sim_config_path,
                            delay_after_load_sec=2)
        self._drone = Drone(self._client, self._world, self.vehicle)
        self._drone.enable_api_control()
        self._drone.arm()

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = None

    @staticmethod
    def _pose(ned_xyz, yaw_rad: float) -> dict:
        return {'translation': {'x': float(ned_xyz[0]), 'y': float(ned_xyz[1]),
                                'z': float(ned_xyz[2])},
                'rotation': {'w': math.cos(yaw_rad / 2.0), 'x': 0.0,
                             'y': 0.0, 'z': math.sin(yaw_rad / 2.0)},
                'frame_id': 'DEFAULT_FRAME'}

    def set_object_pose(self, name: str, ned_xyz, yaw_rad: float) -> None:
        self._world.set_object_pose(name, self._pose(ned_xyz, yaw_rad),
                                    teleport=True)

    def set_vehicle_pose(self, ned_xyz, yaw_rad: float) -> None:
        self._drone.set_pose(self._pose(ned_xyz, yaw_rad),
                             reset_kinematics=True)

    def get_rgb(self) -> np.ndarray:
        from projectairsim.types import ImageType
        from projectairsim.utils import unpack_image
        images = self._drone.get_images(self.camera, [ImageType.SCENE])
        self._last_image_msg = images[ImageType.SCENE]
        return unpack_image(self._last_image_msg)

    def get_bboxes(self) -> List[BBox]:
        """Native annotations: every image message carries
        annotations[].bbox2d {center{x,y}, size{x,y}} for the configured
        object-ids (camera annotation-settings) -- exact labels for free."""
        msg = getattr(self, '_last_image_msg', None)
        if msg is None:
            self.get_rgb()
            msg = self._last_image_msg
        out = []
        for ann in msg.get('annotations', []):
            if ann.get('object_id') not in self.target_ids:
                continue
            c, s = ann['bbox2d']['center'], ann['bbox2d']['size']
            out.append(BBox(ann['object_id'],
                            c['x'] - s['x'] / 2.0, c['y'] - s['y'] / 2.0,
                            c['x'] + s['x'] / 2.0, c['y'] + s['y'] / 2.0))
        return out

    def move_by_velocity(self, v_north, v_east, v_down, yaw_rate_rad_s,
                         duration_s) -> None:
        import asyncio
        from projectairsim.drone import YawControlMode
        # Project AirSim yaw is rad/s [V drone.py docstring] -- no
        # conversion; the API is asyncio, adapter stays synchronous.
        task = self._drone.move_by_velocity_async(
            float(v_north), float(v_east), float(v_down), float(duration_s),
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
            yaw_is_rate=True, yaw=float(yaw_rate_rad_s))
        asyncio.get_event_loop().run_until_complete(task)

    def move_by_velocity_body(self, v_fwd, v_right, v_down, yaw_rate_rad_s,
                              duration_s) -> None:
        import asyncio
        from projectairsim.drone import YawControlMode
        task = self._drone.move_by_velocity_body_frame_async(
            float(v_fwd), float(v_right), float(v_down), float(duration_s),
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
            yaw_is_rate=True, yaw=float(yaw_rate_rad_s))
        asyncio.get_event_loop().run_until_complete(task)

    def hover(self) -> None:
        self.move_by_velocity(0.0, 0.0, 0.0, 0.0, 0.1)


class ClassicEngine(EngineBase):
    """Classic AirSim API (Colosseum pinned / Cosys-AirSim). settings.json
    (settings/settings.json here) must carry CaptureSettings 960x720
    FOV_Degrees 55.1 and the camera name used below."""

    SEGMENTATION_IMAGE_TYPE = 5          # classic; Project AirSim uses 3

    def __init__(self, camera: str = '0', vehicle: str = '',
                 target_mesh_regex: str = 'target.*',
                 detection_radius_cm: float = 2000_00):
        self.camera = camera
        self.vehicle = vehicle
        self.target_mesh_regex = target_mesh_regex
        self.detection_radius_cm = detection_radius_cm
        self._client = None

    def connect(self) -> None:
        try:
            import cosysairsim as airsim      # active fork, if present
        except ImportError:
            import airsim                     # Colosseum / classic
        self._airsim = airsim
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True, self.vehicle)
        self._client.armDisarm(True, self.vehicle)
        # Detection API: exact 2D boxes for named meshes [verified
        # client.py:691 simGetDetections / simAddDetectionFilterMeshName].
        self._client.simSetDetectionFilterRadius(
            self.camera, airsim.ImageType.Scene, self.detection_radius_cm,
            vehicle_name=self.vehicle)
        self._client.simAddDetectionFilterMeshName(
            self.camera, airsim.ImageType.Scene, self.target_mesh_regex,
            vehicle_name=self.vehicle)

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.enableApiControl(False, self.vehicle)
            self._client = None

    def set_object_pose(self, name: str, ned_xyz, yaw_rad: float) -> None:
        a = self._airsim
        pose = a.Pose(a.Vector3r(*map(float, ned_xyz)),
                      a.to_quaternion(0.0, 0.0, float(yaw_rad)))
        self._client.simSetObjectPose(name, pose, teleport=True)

    def set_vehicle_pose(self, ned_xyz, yaw_rad: float) -> None:
        a = self._airsim
        pose = a.Pose(a.Vector3r(*map(float, ned_xyz)),
                      a.to_quaternion(0.0, 0.0, float(yaw_rad)))
        self._client.simSetVehiclePose(pose, ignore_collision=True,
                                       vehicle_name=self.vehicle)

    def get_rgb(self) -> np.ndarray:
        a = self._airsim
        resp = self._client.simGetImages(
            [a.ImageRequest(self.camera, a.ImageType.Scene, False, False)],
            vehicle_name=self.vehicle)[0]
        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        return img.reshape(resp.height, resp.width, 3)

    def get_bboxes(self) -> List[BBox]:
        dets = self._client.simGetDetections(
            self.camera, self._airsim.ImageType.Scene,
            vehicle_name=self.vehicle)
        return [BBox(d.name, d.box2D.min.x_val, d.box2D.min.y_val,
                     d.box2D.max.x_val, d.box2D.max.y_val) for d in dets]

    def move_by_velocity(self, v_north, v_east, v_down, yaw_rate_rad_s,
                         duration_s) -> None:
        a = self._airsim
        # Classic yaw_mode is DEGREES/s [V apis.md] -- convert from SI.
        self._client.moveByVelocityAsync(
            float(v_north), float(v_east), float(v_down), float(duration_s),
            drivetrain=a.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=a.YawMode(is_rate=True,
                               yaw_or_rate=math.degrees(yaw_rate_rad_s)),
            vehicle_name=self.vehicle).join()

    def move_by_velocity_body(self, v_fwd, v_right, v_down, yaw_rate_rad_s,
                              duration_s) -> None:
        a = self._airsim
        self._client.moveByVelocityBodyFrameAsync(
            float(v_fwd), float(v_right), float(v_down), float(duration_s),
            drivetrain=a.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=a.YawMode(is_rate=True,
                               yaw_or_rate=math.degrees(yaw_rate_rad_s)),
            vehicle_name=self.vehicle).join()

    def hover(self) -> None:
        self._client.hoverAsync(vehicle_name=self.vehicle).join()


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Tight box around a boolean segmentation mask -- the fallback
    labelling route when no bbox API exists. Pure numpy, unit-testable."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def make_engine(name: str, **kw) -> EngineBase:
    if name == 'projectairsim':
        return ProjectAirSimEngine(**kw)
    if name == 'classic':
        return ClassicEngine(**kw)
    raise ValueError(f'unknown engine {name!r} (projectairsim | classic)')
