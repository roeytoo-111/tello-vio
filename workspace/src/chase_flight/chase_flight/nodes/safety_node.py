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
        # The forward axis gets its OWN, tighter cap. Its m/s -> stick
        # conversion depends on a full-stick speed that has never been
        # measured on this airframe, so it is the one axis whose commanded
        # magnitude is not trustworthy -- and it is the axis that closes
        # distance on another aircraft. It does not get to share a cap with
        # the two axes whose units are exact.
        self.declare_parameter('max_forward_stick', 0.15)
        self.declare_parameter('battery_floor', 0.20)
        self.declare_parameter('telemetry_timeout_s', 1.0)
        self.declare_parameter('command_timeout_s', 0.30)
        self.declare_parameter('detection_timeout_s', 30.0)
        self.declare_parameter('height_max_m', 2.5)
        self.declare_parameter('height_min_m', 0.3)
        self.declare_parameter('flight_time_max_s', 0.0)   # 0 = disabled
        self.declare_parameter('land_on_disarm', True)
        self.declare_parameter('start_armed', False)
        self.declare_parameter('require_height', True)
        # Refuse to arm before telemetry has ever been seen. Without this
        # every auto-disarm rule FAILS OPEN: link loss, battery, altitude
        # and flight-time are all skipped when their input has never
        # arrived, so the stack would fly a live policy unsupervised.
        self.declare_parameter('require_telemetry_to_arm', True)

        g = lambda n: self.get_parameter(n).value            # noqa: E731
        self.max_stick = float(g('max_stick'))
        self.max_fwd = float(g('max_forward_stick'))
        self.batt_floor = float(g('battery_floor'))
        self.tel_timeout = float(g('telemetry_timeout_s'))
        self.cmd_timeout = float(g('command_timeout_s'))
        self.det_timeout = float(g('detection_timeout_s'))
        self.h_max = float(g('height_max_m'))
        self.h_min = float(g('height_min_m'))
        self.t_max = float(g('flight_time_max_s'))
        self.land_on_disarm = bool(g('land_on_disarm'))
        self.require_height = bool(g('require_height'))
        self.require_tel = bool(g('require_telemetry_to_arm'))
        self._validate_params()

        self.armed = bool(g('start_armed'))
        self._prev_arm_cmd = self.armed       # arm is EDGE triggered
        self._landed = False
        self._disarm_reason = ''
        self._airborne = False                # latched on first real height
        self._flight_s0 = None                # flight time at arm
        self._arm_t = None                    # when the operator armed

        self._batt = float('nan')
        self._height = float('nan')           # best estimate, see _on_tof
        self._sdk_height = float('nan')
        self._flight_s = float('nan')
        self._tel_t = None
        self._cmd = None
        self._cmd_t = None
        self._det_age = float('inf')
        self._lost_s = -1.0                   # uncapped seconds since seen
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
        # An emergency motor-cut must also disarm THIS stack. Otherwise the
        # supervisor keeps publishing a live command, and the aircraft flies
        # under policy the instant anyone sends /takeoff to recover it.
        self.create_subscription(Empty, 'emergency', self._on_emergency, qos)

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
    def _validate_params(self) -> None:
        """Refuse absurd parameters at construction.

        A `max_stick` of 5.0 -- the natural mistake if someone thinks in SDK
        units -- would silently defeat the cap entirely, because the driver
        clamps to its own range afterwards and says nothing.
        """
        bad = []
        if not 0.0 < self.max_stick <= 1.0:
            bad.append(f'max_stick={self.max_stick} (expected 0 < x <= 1)')
        if not 0.0 < self.max_fwd <= 1.0:
            bad.append(f'max_forward_stick={self.max_fwd} (expected 0 < x <= 1)')
        if not 0.0 <= self.batt_floor < 1.0:
            bad.append(f'battery_floor={self.batt_floor} (a FRACTION, '
                       f'not a percent)')
        if not 0.0 < self.cmd_timeout < 0.35:
            bad.append(f'command_timeout_s={self.cmd_timeout} (must be under '
                       f"the driver's 0.35 s dead-man)")
        if self.h_max <= self.h_min:
            bad.append(f'height_max_m={self.h_max} <= height_min_m={self.h_min}')
        if self.h_max > 100.0:
            bad.append(f'height_max_m={self.h_max} (metres, not centimetres)')
        if bad:
            raise SystemExit('chase_safety: refusing to start -- '
                             + '; '.join(bad))

    def _on_cmd(self, msg: TwistStamped):
        t = msg.twist
        # A non-finite command must never become the held command: the cap
        # below cannot clamp a NaN (min/max propagate it into the bound),
        # and a held NaN would be republished at 20 Hz forever.
        for v in (t.linear.x, t.linear.y, t.linear.z, t.angular.z):
            if not math.isfinite(v):
                self.get_logger().error(
                    'non-finite command on chase/cmd_raw -- rejected')
                return
        self._cmd = t
        self._cmd_t = self.get_clock().now()

    def _on_state(self, msg: TrackingState):
        self._det_t = self.get_clock().now()
        self._det_age = (float(msg.detection_age_s)
                         if msg.detection_age_s >= 0 else float('inf'))
        # Uncapped seconds since the last POSITIVE detection; -1 = never.
        self._lost_s = float(msg.time_since_detection_s)

    def _on_arm(self, msg: Bool):
        """Arming is EDGE triggered.

        A level-triggered latch clear is defeated by a repeating publisher:
        `ros2 topic pub` without `-1` republishes at 1 Hz, which would clear
        a latched fault once a second forever while the 20 Hz tick re-latched
        it -- and emit a fresh `land` on every cycle, flooding the driver's
        single blocking-command thread.
        """
        want = bool(msg.data)
        rising = want and not self._prev_arm_cmd
        self._prev_arm_cmd = want
        if not want:
            if self.armed:
                self.get_logger().warn('DISARMED (operator)')
            self.armed = False
            return
        if not rising:
            return                      # level held high: ignore
        if self.require_tel and self._tel_t is None:
            self.get_logger().error(
                'arm REFUSED: no telemetry received yet. Every auto-disarm '
                'rule depends on it, so arming now would fly unsupervised.')
            return
        if self._disarm_reason:
            still = self._check_disarm()
            if still:
                self.get_logger().error(
                    f'arm REFUSED: fault still present ({still})')
                return
            self.get_logger().warn(
                f'clearing latched fault ({self._disarm_reason}) on operator '
                f'arm edge')
            self._disarm_reason = ''
            self._landed = False
        self.armed = True
        self._arm_t = self.get_clock().now()
        self._flight_s0 = (None if math.isnan(self._flight_s)
                           else self._flight_s)
        self.get_logger().warn('ARMED (operator)')

    def _on_abort(self, _msg: Empty):
        self._disarm('operator abort')

    def _on_emergency(self, _msg: Empty):
        # The motors are already cut; landing on top of that is meaningless,
        # so suppress the land and simply make sure the stack is dead.
        self._landed = True
        self._disarm('emergency motor cut')

    def _on_status(self, msg: TelloStatus):
        self._tel_t = self.get_clock().now()
        # TelloStatus carries height in CENTIMETRES and battery as 0..100.
        self._sdk_height = float(msg.height) / 100.0
        self._height = self._sdk_height
        self._flight_s = float(msg.fligth_time)          # sic, driver spelling
        if math.isnan(self._batt):
            self._batt = float(msg.battery) / 100.0
        if self._sdk_height > self.h_min:
            self._airborne = True

    def _on_batt(self, msg: BatteryState):
        # The driver publishes a FRACTION here (0..1), not a percent.
        self._batt = float(msg.percentage)

    def _on_tof(self, msg: Range):
        """The time-of-flight ranger measures whatever is BENEATH the
        aircraft, not altitude. Over a table, a person or a stair it reads
        short, and treating that as altitude would command a descent onto
        the obstacle. It is therefore used only to RAISE the height estimate
        (agreeing with the SDK, or catching a genuine low hover), never to
        lower it below what the SDK reports.
        """
        if not (math.isfinite(msg.range)
                and msg.min_range <= msg.range <= msg.max_range):
            return
        r = float(msg.range)
        if math.isnan(self._sdk_height):
            self._height = r
        else:
            self._height = max(self._sdk_height, r)

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
        """Latching faults, in priority order.

        Every rule here FAILS CLOSED: an input that has never arrived is a
        fault, not a reason to skip the check. The opposite convention --
        skipping a rule whose input is missing -- silently disables link
        loss, battery, altitude and flight-time together, which is exactly
        the set that matters when something is wrong.
        """
        # `require_telemetry_to_arm` governs BOTH the arming gate and these
        # rules. With it True (the flight default) a missing telemetry
        # source is a fault, never a skipped check. Setting it False is an
        # explicit statement that this deployment has no telemetry -- a
        # Gazebo rehearsal or an in-process test -- and then link-loss and
        # battery simply do not apply.
        tel_age = self._age(self._tel_t)
        if self._tel_t is None:
            if self.require_tel:
                return 'no telemetry has ever arrived'
        else:
            if tel_age > self.tel_timeout:
                return f'link loss: no telemetry for {tel_age:.1f}s'
        if math.isnan(self._batt):
            if self.require_tel:
                return 'no battery reading'
        elif self._batt < self.batt_floor:
            return f'low battery: {self._batt * 100:.0f}%'

        # Target loss uses the UNCAPPED time since the last POSITIVE
        # detection. Message age cannot be used: the detector publishes one
        # message per frame including misses, so it reads ~33 ms forever and
        # this rule would never fire.
        # Liveness rules get a grace period measured from ARMING: the
        # supervisor ticks faster than chase_state, so the first tick after
        # an arm legitimately precedes the first observation. Without the
        # grace, arming instantly disarms itself.
        since_arm = self._age(self._arm_t) if self._arm_t is not None else 0.0
        obs_age = self._age(self._det_t)
        if self._det_t is None:
            if since_arm > self.tel_timeout:
                return 'no observation has ever arrived (is chase_state alive?)'
        elif obs_age > self.tel_timeout:
            return f'observation stalled for {obs_age:.1f}s'
        if self._lost_s > self.det_timeout:
            return f'prolonged target loss: {self._lost_s:.0f}s'
        if self._lost_s < 0.0 and self._det_t is not None \
                and since_arm > self.det_timeout:
            return (f'target never acquired within {self.det_timeout:.0f}s '
                    f'of arming')

        if math.isnan(self._height):
            if self.require_height:
                return 'no height source'
        else:
            if self._height > self.h_max:
                return (f'altitude {self._height:.2f} m above ceiling '
                        f'{self.h_max} m')
            # The floor only applies once the aircraft has actually been
            # airborne. On the ground height reads ~0, so enforcing it from
            # the start latches a fault before anyone can arm -- and the
            # natural workaround (zeroing the floor) throws the protection
            # away entirely.
            if self._airborne and self._height < self.h_min:
                return (f'altitude {self._height:.2f} m below floor '
                        f'{self.h_min} m')
        if self.t_max > 0 and self._flight_s0 is not None \
                and not math.isnan(self._flight_s) \
                and (self._flight_s - self._flight_s0) > self.t_max:
            return (f'flight time {self._flight_s - self._flight_s0:.0f}s '
                    f'since arming exceeds {self.t_max}s')
        return ''

    def _tick(self):
        # Faults are only evaluated while armed. On the ground, disarmed,
        # an aircraft with no telemetry is not a fault -- it is Tuesday.
        if self.armed:
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
            def cap(x, limit):
                """Clamp to +/-limit, treating non-finite as zero.

                min()/max() do NOT clamp a NaN: `min(0.6, nan)` returns 0.6,
                so a naive clamp turns a NaN into FULL POSITIVE deflection
                -- and the difference test `abs(0.6 - nan) > eps` is False,
                so it would be reported as un-capped. Measured, not assumed.
                """
                nonlocal capped
                v = float(x)
                if not math.isfinite(v):
                    capped = True
                    return 0.0
                y = max(-limit, min(limit, v))
                capped = capped or (abs(y - v) > 1e-9)
                return y
            # The forward axis has its own, tighter cap: its stick scale is
            # unmeasured, and it is the axis that closes on another aircraft.
            fx = cap(self._cmd.linear.x, self.max_fwd)
            fz = cap(self._cmd.linear.z, self.max_stick)
            fyaw = cap(self._cmd.angular.z, self.max_stick)

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
        st.detection_age_s = float(self._lost_s)
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
