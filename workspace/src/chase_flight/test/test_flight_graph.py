"""Integration tests for the flight graph -- the whole chain, in process.

These spin up the REAL six nodes, inject synthetic `DroneDetection`
messages and assert on what reaches `/cmd_vel`. No aircraft and no
simulator are involved, so they run in CI and they are the last line of
defence before the stack is pointed at a real drone.

The most important one is `test_sign_convention_*`: the standing rule of
this project is that the sign chain is TESTED per tier, never assumed --
three simulators plus one aircraft is four chances for a silent flip, and
the source work needed a yaw negation on real hardware. These tests pin the
convention on the flight side:

    target RIGHT in the image (ex > 0)  ->  angular.z < 0   (turn right)
    target BELOW centre       (ey > 0)  ->  linear.z  < 0   (descend)

both of which follow from the checkpoint's own action map: "+angular.z
(CCW) moves box +u (right); +linear.z (up) moves box +v (down)".
"""
import math
import threading
import time

import pytest

rclpy = pytest.importorskip('rclpy')

from chase_msgs.msg import DroneDetection                      # noqa: E402
from geometry_msgs.msg import Twist                            # noqa: E402
from rclpy.executors import MultiThreadedExecutor              # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from std_msgs.msg import Bool                                  # noqa: E402

from chase_gym import constants as C                           # noqa: E402
from chase_flight.nodes.behaviour_node import ChaseBehaviourNode   # noqa: E402
from chase_flight.nodes.depth_rule_node import ChaseDepthRuleNode  # noqa: E402
from chase_flight.nodes.mixer_node import ChaseMixerNode           # noqa: E402
from chase_flight.nodes.policy_node import ChasePolicyNode         # noqa: E402
from chase_flight.nodes.safety_node import ChaseSafetyNode         # noqa: E402
from chase_flight.nodes.state_node import ChaseStateNode           # noqa: E402

MAX_STICK = 0.6


class Harness(Node):
    """Injects detections, captures /cmd_vel."""

    def __init__(self):
        super().__init__('test_harness')
        self.pub_det = self.create_publisher(DroneDetection,
                                             'chase/detection', 10)
        self.pub_arm = self.create_publisher(Bool, 'chase/arm', 10)
        self.cmds = []
        self.create_subscription(Twist, 'cmd_vel', self._on_cmd, 10)

    def _on_cmd(self, msg):
        self.cmds.append(msg)

    def detect(self, u, v, w_px, detected=True):
        m = DroneDetection()
        m.header.stamp = self.get_clock().now().to_msg()
        m.detected = detected
        m.confidence = 0.9
        m.class_name = 'drone'
        m.xmin, m.xmax = u - w_px / 2.0, u + w_px / 2.0
        m.ymin, m.ymax = v - w_px / 4.0, v + w_px / 4.0
        m.image_width, m.image_height = C.FRAME_W, C.FRAME_H
        m.range_m = float('nan')
        self.pub_det.publish(m)

    def arm(self, on=True):
        b = Bool()
        b.data = on
        self.pub_arm.publish(b)


def _params(**kw):
    from rclpy.parameter import Parameter
    return [Parameter(k, value=v) for k, v in kw.items()]


class Graph:
    """The six nodes plus a harness, spinning in a background thread."""

    def __init__(self, baseline='p_controller', **safety_over):
        if not rclpy.ok():
            rclpy.init()
        sp = dict(max_stick=MAX_STICK, publish_rate_hz=50.0,
                  start_armed=False, land_on_disarm=False,
                  telemetry_timeout_s=1e9, detection_timeout_s=1e9,
                  height_max_m=1e9, height_min_m=-1e9)
        sp.update(safety_over)
        self.nodes = [
            ChaseStateNode(),
            ChasePolicyNode(),
            ChaseDepthRuleNode(),
            ChaseBehaviourNode(),
            ChaseMixerNode(),
            ChaseSafetyNode(),
        ]
        self.harness = Harness()
        self.exec = MultiThreadedExecutor()
        for n in self.nodes + [self.harness]:
            self.exec.add_node(n)
        self._t = threading.Thread(target=self.exec.spin, daemon=True)
        self._t.start()

    def shutdown(self):
        try:
            self.exec.shutdown()
            for n in self.nodes + [self.harness]:
                n.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


@pytest.fixture
def graph(monkeypatch, request):
    """Build the graph with node parameters injected via declare defaults."""
    over = getattr(request, 'param', {}) or {}
    baseline = over.pop('baseline', 'p_controller')
    import rclpy.node as rnode
    real_declare = rnode.Node.declare_parameter
    defaults = dict(baseline=baseline, max_stick=MAX_STICK,
                    publish_rate_hz=50.0, land_on_disarm=False,
                    telemetry_timeout_s=1e9, detection_timeout_s=1e9,
                    height_max_m=50.0, height_min_m=-10.0,
                    control_rate_hz=20.0,
                    # No driver in these tests: the supervisor's telemetry
                    # and height gates are exercised in test_safety.py.
                    require_telemetry_to_arm=False, require_height=False,
                    max_forward_stick=MAX_STICK)
    defaults.update(over)

    def patched(self, name, value=None, *a, **kw):
        if name in defaults:
            value = defaults[name]
        return real_declare(self, name, value, *a, **kw)

    monkeypatch.setattr(rnode.Node, 'declare_parameter', patched)
    g = Graph()
    yield g
    g.shutdown()


def _settle(g, n=25, dt=0.05, u=None, v=None, w=60.0, detected=True):
    """Drive the graph for a while, optionally feeding a fixed detection."""
    for _ in range(n):
        if u is not None:
            g.harness.detect(u, v, w, detected)
        time.sleep(dt)


def test_disarmed_commands_are_exactly_zero(graph):
    """The engage gate: a good detection must not move the aircraft."""
    _settle(graph, u=C.CX + 200, v=C.CY + 100)
    assert graph.harness.cmds, 'safety must publish continuously'
    for c in graph.harness.cmds:
        assert c.linear.x == 0.0 and c.linear.y == 0.0
        assert c.linear.z == 0.0 and c.angular.z == 0.0


def test_lateral_axis_is_structurally_unused(graph):
    graph.harness.arm(True)
    _settle(graph, u=C.CX + 150, v=C.CY - 80)
    assert graph.harness.cmds
    assert all(c.linear.y == 0.0 for c in graph.harness.cmds)


def test_speed_cap_is_enforced(graph):
    graph.harness.arm(True)
    # A wildly off-centre target drives the controller to saturation.
    _settle(graph, u=C.FRAME_W - 5, v=C.FRAME_H - 5, w=20.0)
    assert graph.harness.cmds
    for c in graph.harness.cmds:
        for val in (c.linear.x, c.linear.z, c.angular.z):
            assert abs(val) <= MAX_STICK + 1e-6, f'{val} exceeds cap'


def test_sign_convention_yaw_turns_toward_target(graph):
    """Target on the RIGHT must produce a RIGHT (negative, CW) yaw."""
    graph.harness.arm(True)
    _settle(graph, u=C.CX + 300, v=C.CY, w=60.0)
    acted = [c for c in graph.harness.cmds if c.angular.z != 0.0]
    assert acted, 'no yaw command produced for an off-centre target'
    assert acted[-1].angular.z < 0.0, (
        'target right of centre must yaw right (angular.z < 0); got '
        f'{acted[-1].angular.z}')


def test_sign_convention_vertical_descends_for_low_target(graph):
    """Target BELOW centre must produce a DOWN (negative) vertical stick."""
    graph.harness.arm(True)
    _settle(graph, u=C.CX, v=C.CY + 200, w=60.0)
    acted = [c for c in graph.harness.cmds if c.linear.z != 0.0]
    assert acted, 'no vertical command produced for an off-centre target'
    assert acted[-1].linear.z < 0.0, (
        'target below centre must descend (linear.z < 0); got '
        f'{acted[-1].linear.z}')


def test_forward_axis_holds_when_target_lost(graph):
    """The forward axis must not coast toward a target that is gone."""
    graph.harness.arm(True)
    _settle(graph, u=C.CX, v=C.CY, w=20.0)       # far target -> closing
    graph.harness.cmds.clear()
    _settle(graph, n=30)                          # nothing published at all
    assert graph.harness.cmds
    assert graph.harness.cmds[-1].linear.x == 0.0


def test_standoff_sign_far_target_moves_forward(graph):
    """Beyond the standoff the forward stick is positive; inside, negative."""
    graph.harness.arm(True)
    # w_px small => far. d = fx*W/w = 919.42*0.098/20 = 4.5 m > 2.0 standoff
    _settle(graph, u=C.CX, v=C.CY, w=20.0)
    far = graph.harness.cmds[-1].linear.x
    graph.harness.cmds.clear()
    # w_px large => close. 919.42*0.098/120 = 0.75 m < 2.0 standoff
    _settle(graph, u=C.CX, v=C.CY, w=120.0)
    near = graph.harness.cmds[-1].linear.x
    assert far > 0.0, f'far target must command forward, got {far}'
    assert near < 0.0, f'close target must back off, got {near}'
