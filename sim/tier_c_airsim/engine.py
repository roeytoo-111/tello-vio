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
  fork (`pip install cosysairsim`, 3.5.0). Bboxes via the Detection API;
  bbox_from_mask() below is the segmentation-mask HELPER for scenes where
  the Detection API misbehaves -- wiring it needs the scene's seg IDs and
  is left to the operator (it is not called by get_bboxes).

The verified portability traps this file exists to contain:
  1. yaw units -- Project AirSim: RADIANS (rad/s); classic: DEGREES.
     The adapter's contract is SI (rad, rad/s), converted per backend.
  2. yaw HANDEDNESS -- the adapter's yaw-rate contract is REP-103
     (z-up: positive = CCW viewed from above), the project convention
     every other tier uses. Both engines are NED (z-down: positive yaw =
     CW), so the adapter NEGATES yaw rates at the boundary. Without this
     every yaw command executes in the opposite direction.
  3. ImageType.Segmentation -- Project AirSim: 3; classic: 5. Never
     hardcode the int downstream.
  4. Project AirSim's command API is asyncio; classic returns futures.
     The adapter is synchronous.

Coordinates: positions stay NED (m); velocities m/s; yaw-rate inputs are
REP-103 CCW-positive and converted here.
"""
import math
import os
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
    def takeoff(self) -> None:
        """Put the vehicle in flight. Velocity commands are only honoured
        once the controller is flying (verified live: before takeoff a yaw
        command did nothing and a climb crawled; after takeoff both track
        their setpoints)."""
        ...
    def get_vehicle_pose(self):
        """-> ((x, y, z) NED metres, yaw rad). Scripts place the TARGET
        relative to this instead of teleporting the vehicle: teleporting a
        flying physics vehicle does not stick (verified live -- commanded
        (0,0,-1), the vehicle stayed near its altitude and only its yaw
        reset)."""
        ...
    def set_object_pose(self, name: str, ned_xyz, yaw_rad: float) -> None: ...
    def set_vehicle_pose(self, ned_xyz, yaw_rad: float) -> None: ...
    def get_rgb(self) -> np.ndarray: ...
    def get_bboxes(self) -> List[BBox]: ...
    def move_by_velocity(self, v_north: float, v_east: float, v_down: float,
                         yaw_rate_rad_s: float, duration_s: float) -> None:
        """yaw_rate_rad_s is REP-103 CCW-positive (negated internally for
        the NED engines)."""
        ...
    def move_by_velocity_body(self, v_fwd: float, v_right: float,
                              v_down: float, yaw_rate_rad_s: float,
                              duration_s: float) -> None:
        """Body-frame velocity + yaw rate -- the Tello stick semantics,
        yaw REP-103 CCW-positive. Preferred for closed-loop control: no
        client-side yaw dead-reckoning (which drifts against the engine's
        true heading)."""
        ...
    def hover(self) -> None: ...


class ProjectAirSimEngine(EngineBase):
    """Project AirSim backend. Scene/robot JSONC configs live in
    settings/; the camera there is 960x720 FOV 55.1 deg [V ost.txt] with
    annotation-settings enabled for the target object ids.

    The target is SPAWNED at connect() rather than declared in the scene:
    env actors re-apply their own trajectory pose on every engine tick
    (upstream UnrealEnvActor::Tick), so a teleported env actor snaps back
    one frame later, while a spawned static object stays put (verified
    live on Blocks 1.0.1, 2026-09-19)."""

    # 'Quadrotor1' spans 1.303 m at scale 1 (measured live via its bbox2d
    # at a known range); 0.138 shrinks it to the Tello's 0.18 m prop span.
    TARGET_ASSET = 'Quadrotor1'
    TARGET_SCALE = 0.18 / 1.303

    def __init__(self, scene_config: str = 'scene_chase.jsonc',
                 sim_config_path: str = None,
                 address: str = '127.0.0.1',
                 vehicle: str = 'Follower', camera: str = 'FrontCamera',
                 target_ids: Tuple[str, ...] = ('TargetTello',),
                 target_asset: str = None, target_scale: float = None):
        self.scene_config = scene_config
        # Anchor to this file, not the CWD: World joins the two paths and
        # opens the result relative to the process working directory.
        self.sim_config_path = sim_config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'settings')
        self.address = address
        self.vehicle = vehicle
        self.camera = camera
        self.target_ids = target_ids
        self.target_asset = target_asset or self.TARGET_ASSET
        self.target_scale = target_scale or self.TARGET_SCALE
        self._client = None
        self._world = None
        self._drone = None
        self._loop = None
        self._parked = True

    def connect(self) -> None:
        # Lazy import: this module must be importable (and dry-runnable)
        # on machines without any engine installed.
        import asyncio
        from projectairsim import ProjectAirSimClient, World, Drone
        # One private loop for the whole session: get_event_loop() outside
        # a running loop is deprecated (an error from Python 3.14).
        self._loop = asyncio.new_event_loop()
        self._client = ProjectAirSimClient(address=self.address)
        self._client.connect()
        self._world = World(self._client, self.scene_config,
                            sim_config_path=self.sim_config_path,
                            delay_after_load_sec=2)
        self._drone = Drone(self._client, self._world, self.vehicle)
        self._drone.enable_api_control()
        self._drone.arm()
        # Static (physics-off) target; annotations are configured for this
        # exact name, and spawn_object may rename on collision -- refuse
        # rather than label a target the camera is not annotating.
        name = self._world.spawn_object(
            self.target_ids[0], self.target_asset,
            self._pose((2.0, 0.0, -1.0), 0.0),
            [self.target_scale] * 3, False)
        if name != self.target_ids[0]:
            raise RuntimeError(
                f'target spawned as {name!r}, not {self.target_ids[0]!r} '
                f'(name taken in the scene?) -- bbox annotations would '
                f'silently miss it')

    def takeoff(self) -> None:
        self._run(self._drone.takeoff_async())
        self._parked = True

    # Measured on Blocks 1.0.1 (2026-09-19): from a parked state (just
    # after takeoff or a hover), a yaw-ONLY command -- move_by_velocity or
    # rotate_by_yaw_rate -- is silently ignored (heading moved 0.0000 rad).
    # A command carrying any nonzero linear velocity unparks the controller
    # so the NEXT yaw takes effect. An UPWARD wake works (verified at
    # -0.3/-0.1/-0.05 and a lateral variant -> subsequent yaw moved
    # -0.14..-0.20 rad); a downward pulse is itself ignored and does not
    # unpark. Trap 5 of this adapter: before a yaw-only command from the
    # parked state, send a brief upward wake (few cm) so the yaw takes
    # effect. It fires only from the parked state, so a continuously-
    # commanding controller pays it at most once per hover.
    _WAKE_V = -0.1        # NED z-up (negative), verified to unpark
    _WAKE_S = 0.3

    def _wake_if_parked(self, vx, vy, vz, yaw_rate) -> None:
        lin = abs(vx) + abs(vy) + abs(vz)
        if lin > 1e-6:
            self._parked = False
            return
        if yaw_rate != 0.0 and self._parked:
            from projectairsim.drone import YawControlMode
            self._run(self._drone.move_by_velocity_async(
                0.0, 0.0, self._WAKE_V, self._WAKE_S,
                yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
                yaw_is_rate=True, yaw=0.0))
            self._parked = False

    def get_vehicle_pose(self):
        k = self._drone.get_ground_truth_kinematics()
        p, q = k['pose']['position'], k['pose']['orientation']
        yaw = math.atan2(2.0 * (q['w'] * q['z'] + q['x'] * q['y']),
                         1.0 - 2.0 * (q['y'] ** 2 + q['z'] ** 2))
        return (p['x'], p['y'], p['z']), yaw

    def _run(self, coro):
        """Drive an SDK command to COMPLETION. The async APIs are
        two-stage: awaiting the coroutine only SENDS the request and
        returns the motion's own asyncio.Task, which must be awaited too
        (upstream hello_drone.py awaits twice). One await returns before
        the vehicle has moved."""
        task = self._loop.run_until_complete(coro)
        return self._loop.run_until_complete(task)

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

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
        # NOTE: despite the name, bytes are BGR (server packs B,G,R) --
        # which is exactly what cv2.imwrite and ultralytics expect.
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

    @staticmethod
    def _ned_yaw_rate(rep103_rad_s: float) -> float:
        """REP-103 (+CCW, z-up) -> Project AirSim NED rad/s (+CW, z-down).
        THE one conversion point: a new velocity API must call this or it
        yaws backwards."""
        return -float(rep103_rad_s)

    def move_by_velocity(self, v_north, v_east, v_down, yaw_rate_rad_s,
                         duration_s) -> None:
        from projectairsim.drone import YawControlMode
        self._wake_if_parked(v_north, v_east, v_down, yaw_rate_rad_s)
        # Project AirSim yaw is rad/s [V drone.py docstring].
        self._run(self._drone.move_by_velocity_async(
            float(v_north), float(v_east), float(v_down), float(duration_s),
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
            yaw_is_rate=True, yaw=self._ned_yaw_rate(yaw_rate_rad_s)))

    def move_by_velocity_body(self, v_fwd, v_right, v_down, yaw_rate_rad_s,
                              duration_s) -> None:
        # Route body-frame commands through the WORLD API. Measured on
        # Blocks 1.0.1 (2026-09-19): move_by_velocity_body_frame_async does
        # NOT actuate yaw at all (5/5 reps of a 2 s body yaw moved the
        # heading 0.0000 rad), while the world move_by_velocity yaw does.
        # Rotating the body velocity into world with the live heading, then
        # calling move_by_velocity, gives working yaw AND keeps the linear
        # command carrying motion (so a chase command self-unparks). The
        # heading drifts within the command window, but a 10 Hz caller
        # re-reads and re-converts every tick.
        (_, _, _), yaw = self.get_vehicle_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        v_north = v_fwd * c - v_right * s
        v_east = v_fwd * s + v_right * c
        self.move_by_velocity(v_north, v_east, v_down, yaw_rate_rad_s,
                              duration_s)

    def hover(self) -> None:
        self.move_by_velocity(0.0, 0.0, 0.0, 0.0, 0.1)
        self._parked = True


class ClassicEngine(EngineBase):
    """Classic AirSim API (Colosseum pinned / Cosys-AirSim). settings.json
    (settings/settings.json here) must carry CaptureSettings 960x720
    FOV_Degrees 55.1 and the camera name used below."""

    SEGMENTATION_IMAGE_TYPE = 5          # classic; Project AirSim uses 3

    def __init__(self, camera: str = '0', vehicle: str = '',
                 target_mesh_regex: str = 'TargetTello*',
                 detection_radius_cm: float = 2000_00,
                 address: str = ''):
        # NOTE: simAddDetectionFilterMeshName takes a UE WILDCARD pattern
        # ('*' globs, '.' is literal), not a regex -- and the default must
        # match the default target object name used across Tier C.
        self.camera = camera
        self.vehicle = vehicle
        self.target_mesh_regex = target_mesh_regex
        self.detection_radius_cm = detection_radius_cm
        self.address = address                 # '' = 127.0.0.1
        self._client = None

    def connect(self) -> None:
        try:
            import cosysairsim as airsim      # active fork, if present
        except ImportError:
            import airsim                     # Colosseum / classic
        self._airsim = airsim
        # Cosys-AirSim diverges from classic in two silent ways handled
        # below: euler_to_quaternion(roll,pitch,yaw) instead of
        # to_quaternion(pitch,roll,yaw), and RGB image bytes instead of
        # classic's BGR.
        self._is_cosys = 'cosys' in airsim.__name__
        self._client = airsim.MultirotorClient(ip=self.address)
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

    def _yaw_pose(self, ned_xyz, yaw_rad: float):
        a = self._airsim
        if hasattr(a, 'to_quaternion'):        # classic / Colosseum
            q = a.to_quaternion(0.0, 0.0, float(yaw_rad))
        else:                                  # Cosys 3.5: no to_quaternion
            q = a.utils.euler_to_quaternion(0.0, 0.0, float(yaw_rad))
        return a.Pose(a.Vector3r(*map(float, ned_xyz)), q)

    def set_object_pose(self, name: str, ned_xyz, yaw_rad: float) -> None:
        self._client.simSetObjectPose(name, self._yaw_pose(ned_xyz, yaw_rad),
                                      teleport=True)

    def set_vehicle_pose(self, ned_xyz, yaw_rad: float) -> None:
        self._client.simSetVehiclePose(self._yaw_pose(ned_xyz, yaw_rad),
                                       ignore_collision=True,
                                       vehicle_name=self.vehicle)

    def takeoff(self) -> None:
        self._client.takeoffAsync(vehicle_name=self.vehicle).join()

    def get_vehicle_pose(self):
        pose = self._client.simGetVehiclePose(vehicle_name=self.vehicle)
        p, q = pose.position, pose.orientation
        yaw = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                         1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2))
        return (p.x_val, p.y_val, p.z_val), yaw

    def get_rgb(self) -> np.ndarray:
        a = self._airsim
        resp = self._client.simGetImages(
            [a.ImageRequest(self.camera, a.ImageType.Scene, False, False)],
            vehicle_name=self.vehicle)[0]
        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img = img.reshape(resp.height, resp.width, 3)
        # Contract: BGR out (cv2/ultralytics convention). Classic and
        # Colosseum pack B,G,R; Cosys packs R,G,B and must be flipped.
        return img[:, :, ::-1] if self._is_cosys else img

    def get_bboxes(self) -> List[BBox]:
        dets = self._client.simGetDetections(
            self.camera, self._airsim.ImageType.Scene,
            vehicle_name=self.vehicle)
        return [BBox(d.name, d.box2D.min.x_val, d.box2D.min.y_val,
                     d.box2D.max.x_val, d.box2D.max.y_val) for d in dets]

    @staticmethod
    def _ned_yaw_rate_deg(rep103_rad_s: float) -> float:
        """REP-103 (+CCW, rad/s) -> classic AirSim NED yaw_or_rate
        (+CW, DEGREES/s). THE one conversion point for this backend."""
        return -math.degrees(rep103_rad_s)

    def move_by_velocity(self, v_north, v_east, v_down, yaw_rate_rad_s,
                         duration_s) -> None:
        a = self._airsim
        self._client.moveByVelocityAsync(
            float(v_north), float(v_east), float(v_down), float(duration_s),
            drivetrain=a.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=a.YawMode(is_rate=True,
                               yaw_or_rate=self._ned_yaw_rate_deg(
                                   yaw_rate_rad_s)),
            vehicle_name=self.vehicle).join()

    def move_by_velocity_body(self, v_fwd, v_right, v_down, yaw_rate_rad_s,
                              duration_s) -> None:
        a = self._airsim
        self._client.moveByVelocityBodyFrameAsync(
            float(v_fwd), float(v_right), float(v_down), float(duration_s),
            drivetrain=a.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=a.YawMode(is_rate=True,
                               yaw_or_rate=self._ned_yaw_rate_deg(
                                   yaw_rate_rad_s)),
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
