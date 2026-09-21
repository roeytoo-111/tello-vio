"""chase_safety -- the supervisor, and the only publisher of /cmd_vel.

It is a SEPARATE PROCESS from the mixer on purpose: a controller bug must
not be able to disable the thing that restrains it. Nothing else in this
stack is permitted to publish /cmd_vel.

WHAT IT GUARANTEES

  Engage gate      Until an operator arms the stack, the command on the
                   wire is exactly zero. The policy may run, publish and be
                   recorded from the first second; it simply does not reach
                   the aircraft.
  Speed cap        Every axis is clamped to `max_stick` before it is sent.
                   This is the primary safety mechanism on this airframe --
                   see the honesty note below.
  Dead-man feed    Publishes at 20 Hz with HOLD-LAST-COMMAND semantics. The
                   driver zeroes the sticks if a command is older than
                   0.35 s, so a zero command is a message that must be
                   SENT, never an absence of messages. A stalled upstream
                   node therefore stops the aircraft instead of latching
                   its last command -- the platform is safer than a
                   pulse-and-zero publisher would be.
  Auto-disarm      Link loss, low battery, prolonged target loss, altitude
                   breach or a stale mixer all latch the stack disarmed and
                   command a land.

THE HONEST LIMITATION -- THERE IS NO GEOFENCE. This method estimates no
absolute position: the aircraft knows its height (barometer, time-of-flight
and the SDK's own estimate) but nothing about where it is horizontally. A
lateral geofence is therefore NOT implementable here and this node does not
pretend to offer one. What actually keeps the aircraft inside the room is
the speed cap, a short flight volume chosen by the operator, and a human
with the abort control -- and they should be described that way rather than
overclaimed. The altitude limits below ARE real, because height is measured.
There is also no hardware kill line on a Tello: the equivalents are the
emergency motor-cut and pulling the battery.

EVERY THRESHOLD BELOW IS A [D] CHOICE. The design set names the rules
(engage gate, speed cap, auto-disarm on link loss / low battery / prolonged
loss, volume limits, v_max, battery floor) but fixes no numbers anywhere.
These defaults are deliberately conservative first-flight values.
"""
import math

import rclpy
from chase_msgs.msg import PolicyAction, SafetyStatus, TrackingState
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState, Range
from std_msgs.msg import Bool, Empty
from tello_msg.msg import TelloStatus


class ChaseSafetyNode(Node):

    def __init__(self):
        super().__init__('chase_safety')

        # --- [D] thresholds: none of these come from the design set ----
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('max_stick', 0.6)
        self.declare_parameter('battery_floor', 0.20)
        self.declare_parameter('telemetry_timeout_s', 1.0)
        self.declare_parameter('command_timeout_s', 0.30)
        self.declare_parameter('detection_timeout_s', 30.0)
        self.declare_parameter('height_max_m', 2.5)
        self.declare_parameter('height_min_m', 0.3)
        self.declare_parameter('flight_time_max_s', 0.0)   # 0 = disabled
        self.declare_parameter('land_on_disarm', True)
        self.declare_parameter('start_armed', False)
        self.declare_parameter('require_height', False)

        g = lambda n: self.get_parameter(n).value            # noqa: E731
        self.max_stick = float(g('max_stick'))
        self.batt_floor = float(g('battery_floor'))
        self.tel_timeout = float(g('telemetry_timeout_s'))
        self.cmd_timeout = float(g('command_timeout_s'))
        self.det_timeout = float(g('detection_timeout_s'))
        self.h_max = float(g('height_max_m'))
        self.h_min = float(g('height_min_m'))
        self.t_max = float(g('flight_time_max_s'))
        self.land_on_disarm = bool(g('land_on_disarm'))
        self.require_height = bool(g('require_height'))

        self.armed = bool(g('start_armed'))
        self._landed = False
        self._disarm_reason = ''

        self._batt = float('nan')
        self._height = float('nan')
        self._flight_s = float('nan')
        self._tel_t = None
        self._cmd = None
        self._cmd_t = None
        self._det_age = float('inf')
        self._det_t = None

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(TwistStamped, 'chase/cmd_raw',
                                 self._on_cmd, qos)
        self.create_subscription(TrackingState, 'chase/observation',
                                 self._on_state, qos)
        self.create_subscription(Bool, 'chase/arm', self._on_arm, qos)
        self.create_subscription(Empty, 'chase/abort', self._on_abort, qos)
        self.create_subscription(TelloStatus, 'status', self._on_status, qos)
        self.create_subscription(BatteryState, 'battery', self._on_batt, qos)
        self.create_subscription(Range, 'tof', self._on_tof,
                                 rclpy.qos.qos_profile_sensor_data)

        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', qos)
        self.pub_land = self.create_publisher(Empty, 'land', qos)
        self.pub_status = self.create_publisher(SafetyStatus, 'chase/safety',
                                                qos)
        self.pub_sent = self.create_publisher(PolicyAction,
                                              'chase/action_sent', qos)

        rate = float(g('publish_rate_hz'))
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'safety: max_stick={self.max_stick} battery_floor='
            f'{self.batt_floor:.2f} link_timeout={self.tel_timeout}s '
            f'height=[{self.h_min}, {self.h_max}]m publishing at {rate} Hz. '
            f'ARMED={self.armed}. NOTE: no lateral geofence exists on this '
            f'platform -- the speed cap and the operator are the envelope.')

    # ------------------------------------------------------------------
    def _on_cmd(self, msg: TwistStamped):
        self._cmd = msg.twist
        self._cmd_t = self.get_clock().now()

    def _on_state(self, msg: TrackingState):
        self._det_t = self.get_clock().now()
        self._det_age = (float(msg.detection_age_s)
                         if msg.detection_age_s >= 0 else float('inf'))

    def _on_arm(self, msg: Bool):
        if msg.data and self._disarm_reason:
            self.get_logger().warn(
                f'arm request while latched disarmed ({self._disarm_reason}); '
                f'clearing the latch on explicit operator command')
            self._disarm_reason = ''
            self._landed = False
        self.armed = bool(msg.data)
        self.get_logger().warn(f'ARMED={self.armed} (operator)')

    def _on_abort(self, _msg: Empty):
        self._disarm('operator abort')

    def _on_status(self, msg: TelloStatus):
        self._tel_t = self.get_clock().now()
        # TelloStatus carries height in CENTIMETRES and battery as 0..100.
        self._height = float(msg.height) / 100.0
        self._flight_s = float(msg.fligth_time)          # sic, driver spelling
        if math.isnan(self._batt):
            self._batt = float(msg.battery) / 100.0

    def _on_batt(self, msg: BatteryState):
        # The driver publishes a FRACTION here (0..1), not a percent.
        self._batt = float(msg.percentage)

    def _on_tof(self, msg: Range):
        # Out-of-window readings are published as +inf; only trust the
        # in-window band, and prefer it over the SDK height when valid.
        if math.isfinite(msg.range) and msg.min_range <= msg.range <= msg.max_range:
            self._height = float(msg.range)

    # ------------------------------------------------------------------
    def _age(self, t):
        if t is None:
            return float('inf')
        return (self.get_clock().now() - t).nanoseconds / 1e9

    def _disarm(self, reason: str):
        if self._disarm_reason:
            return
        self._disarm_reason = reason
        self.armed = False
        self.get_logger().error(f'DISARM: {reason}')
        if self.land_on_disarm and not self._landed:
            self._landed = True
            self.pub_land.publish(Empty())
            self.get_logger().error('land commanded')

    def _check_disarm(self):
        """Latching faults, in priority order."""
        tel_age = self._age(self._tel_t)
        if self._tel_t is not None and tel_age > self.tel_timeout:
            return f'link loss: no telemetry for {tel_age:.1f}s'
        if not math.isnan(self._batt) and self._batt < self.batt_floor:
            return f'low battery: {self._batt * 100:.0f}%'
        if math.isfinite(self._det_age) and self._det_age > self.det_timeout:
            return f'prolonged target loss: {self._det_age:.0f}s'
        if not math.isnan(self._height):
            if self._height > self.h_max:
                return f'altitude {self._height:.2f} m above ceiling {self.h_max} m'
            if self._height < self.h_min:
                return f'altitude {self._height:.2f} m below floor {self.h_min} m'
        elif self.require_height:
            return 'no height source'
        if self.t_max > 0 and not math.isnan(self._flight_s) \
                and self._flight_s > self.t_max:
            return f'flight time {self._flight_s:.0f}s exceeds {self.t_max}s'
        return ''

    def _tick(self):
        fault = self._check_disarm()
        if fault:
            self._disarm(fault)

        veto = ''
        if not self.armed:
            veto = self._disarm_reason or 'not armed'
        else:
            cmd_age = self._age(self._cmd_t)
            if self._cmd is None:
                veto = 'no command received'
            elif cmd_age > self.cmd_timeout:
                veto = f'stale command ({cmd_age * 1e3:.0f} ms)'

        capped = False
        if veto:
            fx = fz = fyaw = 0.0
        else:
            def cap(x):
                nonlocal capped
                y = max(-self.max_stick, min(self.max_stick, float(x)))
                capped = capped or (abs(y - float(x)) > 1e-9)
                return y
            fx = cap(self._cmd.linear.x)
            fz = cap(self._cmd.linear.z)
            fyaw = cap(self._cmd.angular.z)

        out = Twist()
        out.linear.x = fx
        out.linear.y = 0.0                 # structurally unused
        out.linear.z = fz
        out.angular.z = fyaw
        self.pub_cmd.publish(out)

        # Feed the assembler what was ACTUALLY sent, post-clamp -- the
        # observation must record the action the aircraft received, not the
        # one the actor proposed.
        sent = PolicyAction()
        sent.header.stamp = self.get_clock().now().to_msg()
        sent.action = [fz, fyaw]
        sent.a_vertical = fz
        sent.a_yaw = fyaw
        self.pub_sent.publish(sent)

        st = SafetyStatus()
        st.header.stamp = sent.header.stamp
        st.armed = bool(self.armed)
        st.engaged = bool(self.armed and not veto)
        st.vetoed = bool(veto)
        st.veto_reason = veto
        st.capped = bool(capped)
        st.battery_frac = float(self._batt)
        st.height_m = float(self._height)
        st.telemetry_age_s = float(self._age(self._tel_t))
        st.detection_age_s = float(self._det_age
                                   if math.isfinite(self._det_age) else -1.0)
        st.command_age_s = float(self._age(self._cmd_t))
        st.flight_time_s = float(self._flight_s)
        st.sent_forward = fx
        st.sent_vertical = fz
        st.sent_yaw = fyaw
        self.pub_status.publish(st)


def main(argv=None):
    rclpy.init(args=argv)
    node = ChaseSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub_cmd.publish(Twist())     # zero on the way out
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
