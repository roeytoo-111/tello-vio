"""Tests for the safety supervisor's FAULT logic.

test_flight_graph.py deliberately disables every auto-disarm rule so that
it can exercise the control path; this file does the opposite. It drives
`chase_safety` alone and asserts that each rule fires, that each one fails
CLOSED when its input is missing, and that a non-finite command cannot
become a deflection.

Several of these encode defects found by review rather than by accident:

  * `min()`/`max()` do not clamp NaN -- a naive clamp turns NaN into FULL
    POSITIVE stick on every axis, and reports itself as un-capped.
  * The detector publishes a message for every frame INCLUDING misses, so
    message age is ~33 ms forever and a target-loss rule keyed on it can
    never fire.
  * A level-triggered arm latch is cleared once a second by a repeating
    `ros2 topic pub` (i.e. by forgetting `-1`), defeating every fault.
  * Height reads ~0 on the ground, so an altitude floor enforced from
    boot latches before anyone can arm.
"""
import math
import threading
import time

import pytest

rclpy = pytest.importorskip('rclpy')

from chase_msgs.msg import SafetyStatus, TrackingState        # noqa: E402
from geometry_msgs.msg import Twist, TwistStamped             # noqa: E402
from rclpy.executors import SingleThreadedExecutor            # noqa: E402
from rclpy.node import Node                                   # noqa: E402
from sensor_msgs.msg import BatteryState                      # noqa: E402
from std_msgs.msg import Bool, Empty                          # noqa: E402
from tello_msg.msg import TelloStatus                         # noqa: E402

from chase_flight.nodes.safety_node import ChaseSafetyNode    # noqa: E402

MAX, FWD = 0.6, 0.15


class Rig(Node):
    def __init__(self):
        super().__init__('safety_rig')
        self.cmd = self.create_publisher(TwistStamped, 'chase/cmd_raw', 10)
        self.arm_p = self.create_publisher(Bool, 'chase/arm', 10)
        self.st_p = self.create_publisher(TelloStatus, 'status', 10)
        self.bat_p = self.create_publisher(BatteryState, 'battery', 10)
        self.obs_p = self.create_publisher(TrackingState, 'chase/observation', 10)
        self.emg_p = self.create_publisher(Empty, 'emergency', 10)
        self.out, self.status, self.landed = [], [], []
        self.create_subscription(Twist, 'cmd_vel', self.out.append, 10)
        self.create_subscription(SafetyStatus, 'chase/safety',
                                 self.status.append, 10)
        self.create_subscription(Empty, 'land', self.landed.append, 10)

    def send_cmd(self, x=0.0, z=0.0, yaw=0.0):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.twist.linear.x, m.twist.linear.z, m.twist.angular.z = x, z, yaw
        self.cmd.publish(m)

    def send_status(self, height_cm=100, battery=90, flight_s=10):
        m = TelloStatus()
        m.height = int(height_cm)
        m.battery = int(battery)
        m.fligth_time = int(flight_s)
        self.st_p.publish(m)
        b = BatteryState()
        b.percentage = battery / 100.0
        self.bat_p.publish(b)

    def send_obs(self, lost_s=0.0):
        m = TrackingState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.detection_age_s = 0.03          # detector alive (publishes misses too)
        m.time_since_detection_s = float(lost_s)
        m.obs_spec_hash = 'x'
        self.obs_p.publish(m)

    def arm(self, on=True):
        b = Bool()
        b.data = on
        self.arm_p.publish(b)


class Fixture:
    def __init__(self, **over):
        import rclpy.node as rn
        self._real = rn.Node.declare_parameter
        defaults = dict(max_stick=MAX, max_forward_stick=FWD,
                        publish_rate_hz=50.0, land_on_disarm=True)
        defaults.update(over)

        def patched(node, name, value=None, *a, **kw):
            if name in defaults:
                value = defaults[name]
            return self._real(node, name, value, *a, **kw)

        rn.Node.declare_parameter = patched
        try:
            if not rclpy.ok():
                rclpy.init()
            self.node = ChaseSafetyNode()
        finally:
            rn.Node.declare_parameter = self._real
        self.rig = Rig()
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        self.ex.add_node(self.rig)
        self._t = threading.Thread(target=self.ex.spin, daemon=True)
        self._t.start()

    def close(self):
        try:
            self.ex.shutdown()
            self.node.destroy_node()
            self.rig.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def spin(fx, n=12, dt=0.05, healthy=True, cmd=None):
    for _ in range(n):
        if healthy:
            fx.rig.send_status()
            fx.rig.send_obs()
        if cmd:
            fx.rig.send_cmd(**cmd)
        time.sleep(dt)


@pytest.fixture
def fx(request):
    f = Fixture(**(getattr(request, 'param', {}) or {}))
    yield f
    f.close()


# ---------------------------------------------------------------- NaN ---
def test_nan_command_is_zeroed_not_full_deflection(fx):
    """A NaN must never become a deflection. min()/max() do not clamp it."""
    spin(fx, 6)
    fx.rig.arm(True)
    spin(fx, 8)
    fx.rig.out.clear()
    spin(fx, 10, cmd={'x': float('nan'), 'z': float('nan'),
                      'yaw': float('nan')})
    assert fx.rig.out
    for c in fx.rig.out:
        for v in (c.linear.x, c.linear.z, c.angular.z):
            assert math.isfinite(v), 'non-finite reached /cmd_vel'
            assert v == 0.0, f'NaN became a {v} deflection'


def test_forward_axis_has_a_tighter_cap(fx):
    """The forward axis scale is unmeasured, so it is capped separately."""
    spin(fx, 6)
    fx.rig.arm(True)
    spin(fx, 8)
    fx.rig.out.clear()
    spin(fx, 10, cmd={'x': 1.0, 'z': 1.0, 'yaw': 1.0})
    assert fx.rig.out
    last = fx.rig.out[-1]
    assert abs(last.linear.x) <= FWD + 1e-9
    assert abs(last.linear.z) <= MAX + 1e-9
    assert last.linear.x < last.linear.z, 'forward must be capped tighter'


# ------------------------------------------------------------- arming ---
def test_cannot_arm_without_telemetry(fx):
    """Every auto-disarm rule depends on telemetry; arming without it
    would fly a live policy with no supervision at all."""
    fx.rig.arm(True)                       # no status ever published
    spin(fx, 10, healthy=False, cmd={'z': 0.5})
    assert fx.rig.out
    assert all(c.linear.z == 0.0 for c in fx.rig.out)
    assert not fx.node.armed


def test_arm_is_edge_triggered(fx):
    """A repeating arm publisher must not clear a latched fault."""
    spin(fx, 6)
    fx.rig.arm(True)
    spin(fx, 6)
    assert fx.node.armed
    # induce a latching fault, then hold arm HIGH the way `ros2 topic pub`
    # without -1 would
    for _ in range(16):
        fx.rig.send_status(battery=5)      # below the 20 % floor
        fx.rig.send_obs()
        fx.rig.arm(True)
        time.sleep(0.05)
    assert not fx.node.armed, 'a held-high arm cleared a latched fault'
    assert 'battery' in fx.node._disarm_reason


# -------------------------------------------------------------- rules ---
def test_low_battery_disarms_and_lands(fx):
    spin(fx, 6)
    fx.rig.arm(True)
    spin(fx, 6)
    for _ in range(12):
        fx.rig.send_status(battery=5)
        fx.rig.send_obs()
        time.sleep(0.05)
    assert not fx.node.armed
    assert fx.rig.landed, 'no land was commanded'
    assert fx.rig.out[-1].linear.z == 0.0


def test_link_loss_disarms(fx):
    spin(fx, 6)
    fx.rig.arm(True)
    spin(fx, 6)
    assert fx.node.armed
    time.sleep(1.5)                        # telemetry_timeout_s = 1.0
    assert not fx.node.armed
    assert 'link loss' in fx.node._disarm_reason


@pytest.mark.parametrize('fx', [{'detection_timeout_s': 0.4}], indirect=True)
def test_prolonged_target_loss_disarms(fx):
    """Keyed on time since the last POSITIVE detection -- message age is
    ~33 ms forever because the detector publishes misses too."""
    spin(fx, 6)
    fx.rig.arm(True)
    spin(fx, 6)
    assert fx.node.armed
    for _ in range(14):
        fx.rig.send_status()
        fx.rig.send_obs(lost_s=5.0)        # detector alive, target gone
        time.sleep(0.05)
    assert not fx.node.armed
    assert 'target loss' in fx.node._disarm_reason


def test_altitude_floor_does_not_fire_on_the_ground(fx):
    """Height reads ~0 before takeoff; enforcing the floor from boot would
    latch a fault before the operator can arm."""
    for _ in range(10):
        fx.rig.send_status(height_cm=0)
        fx.rig.send_obs()
        time.sleep(0.05)
    fx.rig.arm(True)
    spin(fx, 6, healthy=False)
    for _ in range(6):
        fx.rig.send_status(height_cm=0)
        fx.rig.send_obs()
        time.sleep(0.05)
    assert fx.node.armed, f'grounded aircraft disarmed: {fx.node._disarm_reason}'


def test_altitude_ceiling_disarms_once_airborne(fx):
    spin(fx, 6)
    fx.rig.arm(True)
    spin(fx, 6)
    for _ in range(12):
        fx.rig.send_status(height_cm=400)   # 4 m, ceiling is 2.5
        fx.rig.send_obs()
        time.sleep(0.05)
    assert not fx.node.armed
    assert 'ceiling' in fx.node._disarm_reason


def test_emergency_disarms_the_stack(fx):
    spin(fx, 6)
    fx.rig.arm(True)
    spin(fx, 6)
    assert fx.node.armed
    fx.rig.emg_p.publish(Empty())
    spin(fx, 6)
    assert not fx.node.armed
    assert 'emergency' in fx.node._disarm_reason


# --------------------------------------------------------- parameters ---
def test_absurd_max_stick_is_refused():
    """max_stick in SDK units (5.0) would silently defeat the cap."""
    import rclpy.node as rn
    real = rn.Node.declare_parameter

    def patched(node, name, value=None, *a, **kw):
        if name == 'max_stick':
            value = 5.0
        return real(node, name, value, *a, **kw)

    rn.Node.declare_parameter = patched
    try:
        if not rclpy.ok():
            rclpy.init()
        with pytest.raises(SystemExit):
            ChaseSafetyNode()
    finally:
        rn.Node.declare_parameter = real
        if rclpy.ok():
            rclpy.shutdown()
