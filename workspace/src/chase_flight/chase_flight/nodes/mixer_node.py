"""chase_mixer -- one Twist, assembled under exclusive axis ownership.

AXIS OWNERSHIP IS EXCLUSIVE AND ENFORCED HERE. Two controllers writing one
axis is the classic way to get an uncontrollable aircraft, so the mixer is
the single place a command is assembled and the ownership is structural
rather than a matter of discipline:

    linear.z   (up/down)      chase_policy       learned
    angular.z  (yaw)          chase_policy       learned   [see precedence]
    linear.x   (forward/back) chase_depth_rule   hand-coded
    linear.y   (lateral)      UNUSED -- always exactly 0.0

UNITS, AND THE ONE CONVERSION THAT IS LEGITIMATE. The design set contains a
contradiction: one document says the mixer owns a [-1,1] -> +/-60 -> stick
mapping, another says no rescaling may exist between the network output and
/cmd_vel. The second is right for the axes the network owns, and the first
is a leftover from the source work's own action scale. This mixer therefore:

  * passes the POLICY axes through UNSCALED. The checkpoint's action map
    declares stick_range [-1,1] and scale 1.0, /cmd_vel consumes normalised
    sticks in [-1,1], and the driver scales to the SDK's +/-100 internally.
    The ranges already coincide; inserting a factor here would silently
    change what the policy's learned magnitudes mean. The loader refuses
    any checkpoint whose action scale is not 1.0.
  * converts the FORWARD axis from m/s to sticks, because that axis is not
    a network output at all: `forward_command` is a metric law and returns
    metres per second. That conversion needs the airframe's full-stick
    forward speed, which is a MEASURED quantity -- and it has NOT yet been
    measured on this aircraft. The default below is deliberately
    conservative and the parameter is the first thing the step-response
    test should replace.

PRECEDENCE (undefined in the design set; defined here). The behaviour node
overrides the policy on yaw whenever the mode is not TRACK, because outside
TRACK the policy is acting on a zeroed observation -- there is no target in
frame, so its output carries no information about where to point:

    TRACK              policy owns z and yaw; depth rule owns x
    SEARCH / PATROL    behaviour owns yaw; z = 0; x = 0
    REACQUIRE (hold)   everything 0 -- hold position, do not coast
    REACQUIRE (rotate) behaviour owns yaw; z = 0; x = 0

TIMING. The mixer re-mixes and publishes the moment ANY input arrives, and
also on a control-rate timer that acts only as a watchdog. It used to be
purely timer-driven, and measurement showed that design cost more latency
than anything else in the stack: a new action waited up to a full 100 ms
for the mixer's own tick (and then up to 50 ms more for the supervisor's),
while the computation itself takes ~0.03 ms. None of that waiting exists
in training, where an action is applied in the same step it is computed --
so it was pure train/deploy skew. The watchdog timer is kept so that the
mixer still emits, with stale axes zeroed, if every upstream node goes
quiet.
"""
import numpy as np
import rclpy
from chase_msgs.msg import FlightMode, PolicyAction
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from chase_gym import constants as C

from chase_flight.contract import stick_from_mps


class ChaseMixerNode(Node):

    def __init__(self):
        super().__init__('chase_mixer')

        self.declare_parameter('control_rate_hz', C.CONTROL_RATE_HZ)
        # [!][UNMEASURED] Full-stick forward speed of this airframe, m/s.
        # A LOWER value here makes the aircraft fly FASTER for a given
        # commanded m/s (stick = v / scale), so the conservative direction
        # is to keep this at or above the truth. The simulator assumed 1.5,
        # the manufacturer quotes 8 -- this in-code default must be as safe
        # as the shipped YAML, so that running the node without its config
        # file cannot silently pick the dangerous value.
        self.declare_parameter('fwd_full_stick_mps', 6.0)
        # Inputs older than this many control periods are treated as absent.
        self.declare_parameter('stale_ticks', 3.0)

        rate = float(self.get_parameter('control_rate_hz').value)
        self.dt = 1.0 / rate
        self.fwd_scale = float(self.get_parameter('fwd_full_stick_mps').value)
        self.stale_s = float(self.get_parameter('stale_ticks').value) * self.dt

        self._action = None
        self._action_t = None
        self._fwd_mps = 0.0
        self._fwd_t = None
        self._mode = None
        self._mode_t = None

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(PolicyAction, 'chase/action',
                                 self._on_action, qos)
        self.create_subscription(TwistStamped, 'chase/forward_cmd',
                                 self._on_forward, qos)
        self.create_subscription(FlightMode, 'chase/mode', self._on_mode, qos)
        self.pub = self.create_publisher(TwistStamped, 'chase/cmd_raw', qos)
        self.create_timer(self.dt, self._tick)

        self.get_logger().warn(
            f'forward stick scale = {self.fwd_scale:.2f} m/s at full stick '
            f'[D][UNMEASURED placeholder]. Until a step response is flown, '
            f'the forward axis magnitude is an assumption.')

    # ------------------------------------------------------------------
    def _on_action(self, msg: PolicyAction):
        self._action = np.asarray(msg.action, dtype=np.float32)
        self._action_t = self.get_clock().now()
        self._tick()                # event-driven: publish now

    def _on_forward(self, msg: TwistStamped):
        self._fwd_mps = float(msg.twist.linear.x)
        self._fwd_t = self.get_clock().now()
        self._tick()                # event-driven: publish now

    def _on_mode(self, msg: FlightMode):
        self._mode = msg
        self._mode_t = self.get_clock().now()
        self._tick()                # event-driven: publish now

    def _age(self, t):
        if t is None:
            return float('inf')
        return (self.get_clock().now() - t).nanoseconds / 1e9

    # ------------------------------------------------------------------
    def _tick(self):
        a_fresh = self._age(self._action_t) <= self.stale_s
        f_fresh = self._age(self._fwd_t) <= self.stale_s
        m_fresh = self._age(self._mode_t) <= self.stale_s

        a_v = a_yaw = 0.0
        if a_fresh and self._action is not None and len(self._action) >= 2:
            a_v = float(self._action[0])
            a_yaw = float(self._action[1])
        v_fwd_mps = self._fwd_mps if f_fresh else 0.0

        mode = self._mode.mode if (m_fresh and self._mode) else None
        if mode == FlightMode.TRACK:
            out_v, out_yaw = a_v, a_yaw
            out_x = stick_from_mps(v_fwd_mps, self.fwd_scale)
        elif mode in (FlightMode.SEARCH, FlightMode.PATROL):
            out_v, out_yaw, out_x = 0.0, float(self._mode.search_yaw), 0.0
        elif mode == FlightMode.REACQUIRE:
            if self._mode.hold_position:
                out_v, out_yaw, out_x = 0.0, 0.0, 0.0
            else:
                out_v, out_yaw, out_x = 0.0, float(self._mode.search_yaw), 0.0
        else:
            # No fresh mode: the mission state is unknown, so command
            # nothing. The safety supervisor sees the zeros and the age.
            out_v = out_yaw = out_x = 0.0

        clip = lambda x: float(np.clip(x, -1.0, 1.0))      # noqa: E731
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.twist.linear.x = clip(out_x)      # depth rule (m/s -> stick)
        out.twist.linear.y = 0.0              # UNUSED, structurally
        out.twist.linear.z = clip(out_v)      # policy, unscaled
        out.twist.angular.z = clip(out_yaw)   # policy or behaviour
        self.pub.publish(out)


def main(argv=None):
    rclpy.init(args=argv)
    node = ChaseMixerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
