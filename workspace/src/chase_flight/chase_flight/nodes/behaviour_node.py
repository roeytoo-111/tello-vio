"""chase_behaviour -- SEARCH / TRACK / REACQUIRE / PATROL.

The mission logic, and the only node that decides what to do when there is
no target. The source work describes it plainly: if the followed drone
leaves the field of view the follower "initiates a rescan operation", it
"first maintains its position and then adjusts its field of view", the scan
follows "a predefined movement pattern", and after a prolonged loss it
"switches to a predefined patrol mode". Scenario 3 gives the pattern:
rotate about its own axis.

    [*] -> SEARCH                 no target yet; rotate in place
    SEARCH    -> TRACK            drone detected
    TRACK     -> REACQUIRE        detection lost
    REACQUIRE -> TRACK            target reacquired
    REACQUIRE -> PATROL           loss exceeds the timeout
    PATROL    -> TRACK            drone detected

REACQUIRE has two phases, in the order the source describes: HOLD first
(stop, do not coast toward a target you can no longer see), then ROTATE.

EVERY TIMEOUT HERE IS A [D] CHOICE, NOT A MEASUREMENT. The design documents
name these four knobs -- lost-frame count, loss timeout, search yaw rate,
patrol pattern -- and deliberately leave the values to the implementer;
there is a dangling promise of "recommended defaults" that was never
written. The defaults below are conservative starting points chosen to be
obviously safe indoors, and they are exactly the parameters to tune on the
first flights. The one value with any provenance is the 1.0 s loss timeout,
which matches the `loss_timeout_s` baked into the checkpoint's observation
spec -- i.e. the staleness normaliser the policy was trained against.

This node never touches the vertical or forward axes. It publishes a yaw
search command and a mode; the mixer decides how they combine with the
policy, under an explicit precedence rule.
"""
import rclpy
from chase_msgs.msg import FlightMode, TrackingState
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

_NAMES = {FlightMode.SEARCH: 'SEARCH', FlightMode.TRACK: 'TRACK',
          FlightMode.REACQUIRE: 'REACQUIRE', FlightMode.PATROL: 'PATROL'}


class ChaseBehaviourNode(Node):

    def __init__(self):
        super().__init__('chase_behaviour')

        # --- [D] operator-tunable mission knobs -----------------------
        # Consecutive missed control ticks before TRACK concedes the target.
        # At 10 Hz, 5 ticks = 0.5 s: long enough to ride out a dropped
        # frame or one late Wi-Fi packet, short enough that the aircraft
        # does not keep closing on a target that is no longer there.
        self.declare_parameter('lost_ticks', 5)
        # How long REACQUIRE holds position before it starts rotating.
        self.declare_parameter('reacquire_hold_s', 1.0)
        # Total loss duration after which REACQUIRE concedes to PATROL.
        self.declare_parameter('patrol_after_s', 8.0)
        # Yaw stick used while searching/patrolling, normalised [-1, 1].
        # 0.25 of full stick is a slow sweep: at the airframe's nominal
        # 1.5 rad/s full-stick yaw that is ~0.37 rad/s, so the 55.1 deg
        # field of view sweeps past in ~2.6 s -- fast enough to scan, slow
        # enough that the detector gets several frames on any target it
        # crosses. Sweeping faster than the detector can see is the classic
        # way to rotate straight past the thing you are looking for.
        self.declare_parameter('search_yaw', 0.25)
        # PATROL here is a continuous slow rotation -- the simplest
        # "predefined pattern" that needs no absolute position, which this
        # method does not have. A translating patrol would need a position
        # source the aircraft lacks.
        self.declare_parameter('patrol_yaw', 0.20)

        self.lost_ticks = int(self.get_parameter('lost_ticks').value)
        self.hold_s = float(self.get_parameter('reacquire_hold_s').value)
        self.patrol_after_s = float(self.get_parameter('patrol_after_s').value)
        self.search_yaw = float(self.get_parameter('search_yaw').value)
        self.patrol_yaw = float(self.get_parameter('patrol_yaw').value)

        self.mode = FlightMode.SEARCH
        self._miss = 0
        self._mode_since = self.get_clock().now()
        self._lost_since = None

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(FlightMode, 'chase/mode', qos)
        self.create_subscription(TrackingState, 'chase/observation',
                                 self._on_state, qos)
        self.get_logger().info(
            f'behaviour: lost_ticks={self.lost_ticks} '
            f'hold={self.hold_s}s patrol_after={self.patrol_after_s}s '
            f'search_yaw={self.search_yaw} (all [D] defaults -- tune in '
            f'flight)')

    # ------------------------------------------------------------------
    def _enter(self, mode):
        if mode != self.mode:
            self.get_logger().info(
                f'{_NAMES[self.mode]} -> {_NAMES[mode]}')
            self.mode = mode
            self._mode_since = self.get_clock().now()

    def _on_state(self, msg: TrackingState):
        now = self.get_clock().now()
        if msg.visible:
            self._miss = 0
            self._lost_since = None
            self._enter(FlightMode.TRACK)
        else:
            self._miss += 1
            if self._miss >= self.lost_ticks:
                if self._lost_since is None:
                    self._lost_since = now
                lost_s = (now - self._lost_since).nanoseconds / 1e9
                if self.mode == FlightMode.TRACK:
                    self._enter(FlightMode.REACQUIRE)
                if (self.mode == FlightMode.REACQUIRE
                        and lost_s >= self.patrol_after_s):
                    self._enter(FlightMode.PATROL)

        in_mode_s = (now - self._mode_since).nanoseconds / 1e9

        search_yaw = 0.0
        hold = False
        if self.mode == FlightMode.SEARCH:
            search_yaw = self.search_yaw
        elif self.mode == FlightMode.REACQUIRE:
            if in_mode_s < self.hold_s:
                hold = True          # phase 1: maintain position
            else:
                search_yaw = self.search_yaw   # phase 2: adjust the view
        elif self.mode == FlightMode.PATROL:
            search_yaw = self.patrol_yaw

        out = FlightMode()
        out.header.stamp = now.to_msg()
        out.mode = self.mode
        out.mode_name = _NAMES[self.mode]
        out.search_yaw = float(search_yaw)
        out.hold_position = bool(hold)
        out.time_in_mode_s = float(in_mode_s)
        out.detection_age_s = float(msg.detection_age_s)
        self.pub.publish(out)


def main(argv=None):
    rclpy.init(args=argv)
    node = ChaseBehaviourNode()
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
