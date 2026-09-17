"""The ONLY file that speaks gz-transport -- everything else talks to this
interface, which is what makes Tier B testable on a machine without gz-sim
and swappable if the transport API moves.

Verified API facts this file is built on (all read from gz-sim8/gz-msgs10/
gz-transport13 sources, 2026-09-17; see sim_implementation_map.md):

  * Python bindings: `from gz.transport13 import Node`,
    `from gz.msgs10.<x>_pb2 import <X>`; `node.request(service, req,
    ReqType, RespType, timeout_ms) -> (ok, resp)`;
    `node.subscribe(MsgType, topic, cb)`.
  * Lockstep: `/world/<w>/control` takes gz.msgs.WorldControl; the correct
    request is {pause: true, multi_step: N} -- pause:false + multi_step
    FREE-RUNS. The service returns IMMEDIATELY (the command is queued);
    completion must be detected from `/world/<w>/clock` (gz.msgs.Clock,
    published every loop iteration, NOT throttled) -- `/world/<w>/stats`
    is throttled to 10 msgs/s wall-clock and pose/info to 60/s wall-clock,
    both useless at high real-time factors.
  * Reset: {pause: true, reset: {all: true}}; systems without ISystemReset
    (MulticopterVelocityControl, MulticopterMotorModel) are destroyed and
    re-Configured -- clean, but the controller then waits for a FIRST
    Twist before publishing rotor velocities: re-send a command after
    reset.
  * set_pose: `/world/<w>/set_pose` (gz.msgs.Pose in, Boolean out),
    provided by the UserCommands system, QUEUED -- executed on the next
    PreUpdate, so flush with a small step.
  * OdometryPublisher output `/model/<name>/odometry` (gz.msgs.Odometry)
    is throttled in SIM time -- deterministic under lockstep, unlike
    pose/info.

Measured against a live gz-sim 8.15 server (2026-09-17), not read:

  * `Node.request_raw` blocks WITHOUT releasing the GIL. Once the process
    holds any subscription, the transport thread delivering the reply is
    stuck acquiring the GIL for a callback, so every request times out
    (ok=False) even though the server executed it -- a multi_step still
    advanced sim time. Service calls therefore go through a spawned helper
    process whose Node never subscribes (_REQUEST_WORKER_SRC).
  * The apt gz.msgs10 _pb2 files predate protoc 3.19; a pip protobuf >= 4
    (onnx pulls one into ~/.local) refuses them unless the pure-Python
    implementation is selected before google.protobuf is first imported.
"""
import math
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from importlib import metadata
from typing import Dict, Optional, Tuple

import numpy as np


def ensure_protobuf_compat() -> None:
    try:
        major = int(metadata.version('protobuf').split('.')[0])
    except (metadata.PackageNotFoundError, ValueError):
        return
    if major < 4:
        return
    impl = os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION')
    if impl == 'python':
        return
    if 'google.protobuf' in sys.modules:
        raise RuntimeError(
            f'protobuf {major}.x was imported before gz.msgs10 and cannot '
            f'load its _pb2 files; export '
            f'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python before starting '
            f'Python')
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


# Run with `python -c`, not multiprocessing: spawn re-imports the caller's
# __main__, which breaks any unguarded script, and fork is unsafe once torch
# or gz-transport threads exist.
_REQUEST_WORKER_SRC = r'''
import pickle, struct, sys
from gz.transport13 import Node
node = Node()
rd, wr = sys.stdin.buffer, sys.stdout.buffer
while True:
    head = rd.read(4)
    if len(head) < 4:
        break
    item = pickle.loads(rd.read(struct.unpack("<I", head)[0]))
    out = pickle.dumps(node.request_raw(*item))
    wr.write(struct.pack("<I", len(out)) + out)
    wr.flush()
'''


@dataclass
class OdomSample:
    t_sim: float
    pos: np.ndarray                   # world xyz
    yaw: float
    lin_body: np.ndarray              # body-frame linear velocity
    yaw_rate: float


class GzBackendBase:
    """The contract GzChaseEnv drives. Implementations: GzTransportBackend
    (real gz-sim server) and FakeGzBackend (kinematic double for tests)."""

    physics_dt: float = 0.001
    # OdometryPublisher period (sim time) in the generated world; steps
    # shorter than this guarantee no fresh pose and are refused by BOTH
    # backends -- the fake enforces the caller contract the real one needs.
    ODOM_PERIOD_S = 0.010

    def check_step_span(self, iterations: int) -> None:
        span = iterations * self.physics_dt
        if span < self.ODOM_PERIOD_S:
            raise ValueError(
                f'step of {span * 1000:.0f} ms is shorter than the odom '
                f'publish period ({self.ODOM_PERIOD_S * 1000:.0f} ms): no '
                f'fresh pose is guaranteed -- step at least one period')

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def reset_world(self) -> None: ...
    def set_pose(self, model: str, pos, yaw: float) -> None: ...
    def send_twist(self, model: str, lin_body, yaw_rate: float) -> None: ...
    def step(self, iterations: int) -> None: ...
    def sim_time(self) -> float: ...
    def get_odom(self, model: str) -> Optional[OdomSample]: ...


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class GzTransportBackend(GzBackendBase):
    """Drives a headless `gz sim -s` server over gz-transport13. One backend
    per server; concurrent instances isolate with GZ_PARTITION."""

    def __init__(self, world_sdf: str, world_name: str = 'chase',
                 models: Tuple[str, ...] = ('follower', 'target'),
                 partition: Optional[str] = None,
                 server_cmd: str = 'gz sim -s -r',
                 request_timeout_ms: int = 2000,
                 step_timeout_s: float = 30.0):
        self.world_sdf = world_sdf
        self.world_name = world_name
        self.models = models
        self.partition = partition
        self.server_cmd = server_cmd
        self.request_timeout_ms = request_timeout_ms
        self.step_timeout_s = step_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._req_proc: Optional[subprocess.Popen] = None
        self._node = None
        self._pubs: Dict[str, object] = {}
        self._odom: Dict[str, OdomSample] = {}
        self._sim_time = 0.0
        self._paused = True
        self._clock_cv = threading.Condition()
        self._WorldControl = None
        self._Boolean = None
        self._Pose = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        try:
            self._start()
        except BaseException:
            self.stop()          # never orphan a server the caller can't see
            raise

    def _start(self) -> None:
        env = dict(os.environ)
        if self.partition:
            env['GZ_PARTITION'] = self.partition
        cmd = self.server_cmd.split() + [self.world_sdf]
        self._proc = subprocess.Popen(cmd, env=env,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        if self.partition:
            os.environ['GZ_PARTITION'] = self.partition
        ensure_protobuf_compat()

        self._req_proc = subprocess.Popen(
            [sys.executable, '-c', _REQUEST_WORKER_SRC],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        from gz.transport13 import Node
        from gz.msgs10.world_control_pb2 import WorldControl
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_pb2 import Pose
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.twist_pb2 import Twist
        self._WorldControl, self._Boolean, self._Pose = WorldControl, Boolean, Pose
        self._Twist = Twist
        self._node = Node()

        def on_clock(msg: Clock):
            with self._clock_cv:
                self._sim_time = msg.sim.sec + msg.sim.nsec * 1e-9
                self._clock_cv.notify_all()

        ok = self._node.subscribe(Clock, f'/world/{self.world_name}/clock',
                                  on_clock)
        if not ok:
            raise RuntimeError('failed to subscribe world clock')

        for model in self.models:
            def make_cb(name):
                def on_odom(msg: Odometry):
                    q = msg.pose.orientation
                    sample = OdomSample(
                        t_sim=(msg.header.stamp.sec
                               + msg.header.stamp.nsec * 1e-9),
                        pos=np.array([msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z]),
                        yaw=quat_to_yaw(q.w, q.x, q.y, q.z),
                        lin_body=np.array([msg.twist.linear.x,
                                           msg.twist.linear.y,
                                           msg.twist.linear.z]),
                        yaw_rate=msg.twist.angular.z)
                    with self._clock_cv:
                        self._odom[name] = sample
                        self._clock_cv.notify_all()
                return on_odom
            if not self._node.subscribe(Odometry, f'/model/{model}/odometry',
                                        make_cb(model)):
                raise RuntimeError(f'failed to subscribe odometry of {model}')
            self._pubs[model] = self._node.advertise(
                f'/{model}/cmd_vel', Twist)

        self._wait_for_server()

    def _wait_for_server(self, timeout_s: float = 20.0) -> None:
        deadline = time.time() + timeout_s
        with self._clock_cv:
            while self._sim_time == 0.0 and time.time() < deadline:
                self._clock_cv.wait(0.5)
            if self._sim_time == 0.0:
                # A silent fall-through here poisons everything downstream
                # (reset_world would read pre=0 on a live server).
                raise TimeoutError(
                    f'gz server produced no clock within {timeout_s}s -- '
                    f'is {self.world_sdf} loadable?')
        # Pause immediately: lockstep owns the clock from here on.
        self._control(pause=True)

    def stop(self) -> None:
        if self._req_proc is not None:
            try:
                self._req_proc.stdin.close()   # EOF ends the worker loop
                self._req_proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self._req_proc.kill()
            self._req_proc = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    # -- control ----------------------------------------------------------
    def _request(self, service: str, req, ReqType, RespType):
        blob = pickle.dumps((service, req.SerializeToString(),
                             ReqType.DESCRIPTOR.full_name,
                             RespType.DESCRIPTOR.full_name,
                             self.request_timeout_ms))
        proc = self._req_proc
        proc.stdin.write(struct.pack('<I', len(blob)) + blob)
        proc.stdin.flush()
        head = proc.stdout.read(4)
        if len(head) < 4:
            raise RuntimeError(
                f'gz request worker exited (code {proc.poll()}) during '
                f'{service}')
        ok, raw = pickle.loads(proc.stdout.read(struct.unpack('<I', head)[0]))
        resp = RespType()
        if ok:
            resp.ParseFromString(raw)
        return ok, resp

    def _control(self, pause: Optional[bool] = None, multi_step: int = 0,
                 reset_all: bool = False) -> None:
        req = self._WorldControl()
        if pause is not None:
            req.pause = pause
        if multi_step:
            req.multi_step = multi_step
        if reset_all:
            req.reset.all = True
        ok, resp = self._request(
            f'/world/{self.world_name}/control', req, self._WorldControl,
            self._Boolean)
        if not (ok and resp.data):
            raise RuntimeError(
                f'world control failed (pause={pause}, step={multi_step}, '
                f'reset={reset_all})')

    def step(self, iterations: int) -> None:
        """Lockstep: {pause: true, multi_step: N}, wait on the unthrottled
        world clock until sim time reaches the target, THEN wait until
        every model's odometry has caught up to the step boundary.

        The second wait is load-bearing: odometry arrives on its own topic
        asynchronously to the clock, so without it get_odom() can return
        None (before the first publish) or a pose up to one publish period
        behind the boundary -- silently breaking reward/termination truth
        and the one-seed-one-geometry contract on the real backend (the
        kinematic fake, with instant odometry, cannot show this)."""
        start = self.sim_time()
        target = start + iterations * self.physics_dt
        self._control(pause=True, multi_step=iterations)
        deadline = time.time() + self.step_timeout_s
        with self._clock_cv:
            while self._sim_time < target - 0.25 * self.physics_dt:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f'lockstep: sim time {self._sim_time:.4f} never '
                        f'reached {target:.4f}')
                self._clock_cv.wait(min(remaining, 0.5))
        # Freshness bound, both invariants at once: the sample must be
        # POST-START (a pre-step/pre-teleport pose must never satisfy the
        # wait) AND NEAR-TARGET (accepting the step's first odom would
        # leave truth up to a whole control period stale and let the two
        # models pair time-skewed samples). max() enforces both; steps
        # shorter than one publish period cannot guarantee any fresh
        # sample and are refused outright rather than served stale data.
        self.check_step_span(iterations)
        want = max(start + 0.25 * self.physics_dt,
                   target - 1.5 * self.ODOM_PERIOD_S)
        self._wait_odoms(want, deadline)

    def _wait_odoms(self, want: float, deadline: float) -> None:
        # The odom callbacks notify the same condition variable as the
        # clock, so this waits instead of sleep-polling (a 1 ms poll tax
        # on every lockstep step adds real minutes over a fine-tune run).
        with self._clock_cv:
            while True:
                stale = [m for m in self.models
                         if self._odom.get(m) is None
                         or self._odom[m].t_sim < want]
                if not stale:
                    return
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f'odometry for {stale} never reached sim time '
                        f'{want:.3f} (have: '
                        f'{ {m: getattr(self._odom.get(m), "t_sim", None) for m in stale} })')
                self._clock_cv.wait(min(remaining, 0.5))

    def reset_world(self) -> None:
        pre = self.sim_time()
        self._control(pause=True, reset_all=True)
        # The rewind applies on the next loop iteration while paused; wait
        # until the (unthrottled, publishes-while-paused) clock shows a
        # time BELOW the pre-reset time -- an absolute threshold would be
        # satisfied by a stale pre-reset callback whenever the world had
        # only run briefly. If the world never advanced, there is nothing
        # to rewind and nothing to wait for.
        if pre > 0.0:
            deadline = time.time() + self.step_timeout_s
            rewound_below = min(0.5 * pre, 0.5)
            with self._clock_cv:
                self._sim_time = float('inf')  # ignore pre-reset callbacks
                while self._sim_time > rewound_below:
                    if time.time() > deadline:
                        raise TimeoutError(
                            'world reset never rewound sim time')
                    self._clock_cv.wait(0.5)
        self._odom.clear()                     # pre-reset poses are stale
        for model in self.models:
            self.send_twist(model, (0.0, 0.0, 0.0), 0.0)
        # Flush at least one odometry period so _wait_odoms has a sample
        # to see (the publisher emits every ODOM_PERIOD_S of sim time).
        self.step(int(math.ceil(self.ODOM_PERIOD_S / self.physics_dt)) + 2)

    def set_pose(self, model: str, pos, yaw: float) -> None:
        req = self._Pose()
        req.name = model
        req.position.x, req.position.y, req.position.z = map(float, pos)
        req.orientation.w = math.cos(yaw / 2.0)
        req.orientation.z = math.sin(yaw / 2.0)
        ok, resp = self._request(
            f'/world/{self.world_name}/set_pose', req, self._Pose,
            self._Boolean)
        if not (ok and resp.data):
            raise RuntimeError(f'set_pose({model}) refused')
        # Queued by UserCommands: takes effect on the next PreUpdate.

    def send_twist(self, model: str, lin_body, yaw_rate: float) -> None:
        msg = self._Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = map(float, lin_body)
        msg.angular.z = float(yaw_rate)
        self._pubs[model].publish(msg)

    def sim_time(self) -> float:
        with self._clock_cv:
            return self._sim_time

    def get_odom(self, model: str) -> Optional[OdomSample]:
        return self._odom.get(model)


class FakeGzBackend(GzBackendBase):
    """Kinematic double of the gz world: two 'multicopters' whose achieved
    body velocity follows the commanded twist through a first-order lag
    (what MulticopterVelocityControl produces at Tier-A fidelity),
    integrated at the physics step. Exercises every line of GzChaseEnv --
    lockstep bookkeeping, oracle, latency shim -- without a gz install.
    The REAL backend is exercised by sign_test.py after
    scripts/install_gz_harmonic.sh."""

    def __init__(self, models: Tuple[str, ...] = ('follower', 'target'),
                 response_tau: float = 0.15):
        self.models = models
        self.response_tau = response_tau
        self._t = 0.0
        self._state: Dict[str, dict] = {}
        self._cmd: Dict[str, tuple] = {}
        self.reset_world()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reset_world(self) -> None:
        self._t = 0.0
        self._state = {m: {'pos': np.zeros(3), 'yaw': 0.0,
                           'v_body': np.zeros(3), 'yaw_rate': 0.0}
                       for m in self.models}
        self._cmd = {m: ((0.0, 0.0, 0.0), 0.0) for m in self.models}

    def set_pose(self, model: str, pos, yaw: float) -> None:
        st = self._state[model]
        st['pos'] = np.asarray(pos, dtype=float).copy()
        st['yaw'] = float(yaw)
        st['v_body'] = np.zeros(3)
        st['yaw_rate'] = 0.0

    def send_twist(self, model: str, lin_body, yaw_rate: float) -> None:
        self._cmd[model] = (tuple(map(float, lin_body)), float(yaw_rate))

    def step(self, iterations: int) -> None:
        self.check_step_span(iterations)
        dt = self.physics_dt
        alpha = min(dt / self.response_tau, 1.0)
        for _ in range(iterations):
            for m, st in self._state.items():
                cmd_lin, cmd_yaw = self._cmd[m]
                st['v_body'] += alpha * (np.asarray(cmd_lin) - st['v_body'])
                st['yaw_rate'] += alpha * (cmd_yaw - st['yaw_rate'])
                cy, sy = math.cos(st['yaw']), math.sin(st['yaw'])
                vx, vy, vz = st['v_body']
                st['pos'] += np.array([cy * vx - sy * vy,
                                       sy * vx + cy * vy, vz]) * dt
                st['yaw'] += st['yaw_rate'] * dt
            self._t += dt

    def sim_time(self) -> float:
        return self._t

    def get_odom(self, model: str) -> Optional[OdomSample]:
        st = self._state[model]
        return OdomSample(t_sim=self._t, pos=st['pos'].copy(),
                          yaw=st['yaw'], lin_body=st['v_body'].copy(),
                          yaw_rate=st['yaw_rate'])


# ---- shared script plumbing (sign_test, calibrate_lag, ab_replay_gate,
# ---- finetune all take the same backend choice) --------------------------

DEFAULT_WORLD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'worlds', 'chase.sdf')


def add_backend_args(ap) -> None:
    ap.add_argument('--fake', action='store_true',
                    help='kinematic fake backend (no gz install needed)')
    ap.add_argument('--world', default=None,
                    help=f'world SDF for the real backend '
                         f'(default: {DEFAULT_WORLD})')
    ap.add_argument('--partition', default=None,
                    help='GZ_PARTITION (default: chase-<pid>, so two '
                         'concurrent scripts never share transport)')


def make_backend(args, **fake_kwargs) -> GzBackendBase:
    if args.fake:
        return FakeGzBackend(**fake_kwargs)
    partition = args.partition or f'chase-{os.getpid()}'
    return GzTransportBackend(args.world or DEFAULT_WORLD,
                              partition=partition)
