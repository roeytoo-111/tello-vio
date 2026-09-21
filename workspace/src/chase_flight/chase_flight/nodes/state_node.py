"""chase_state -- the observation clock.

Owns the control tick (10 Hz, Delta_t = 0.1 s: "the deployment constant",
and the same period the Gym environment stepped at). On each tick it

    1. records the action that was actually SENT last tick,
    2. assembles the 14-dim observation from the newest detection,
    3. publishes it.

Both steps run through `chase_gym.observation.ObservationAssembler` -- the
same module the trainer imported -- so the vector the network sees in
flight is produced by the same code that produced its training inputs. The
ordering (record-then-assemble) mirrors `chase_gym/env.py` exactly: slot
6..7 of the observation is the action just sent, not the one about to be
computed.

Two rules this node must not get wrong, both of which are documented
consequences of how the policy was trained:

  * ON A MISS THE ERRORS ARE ZEROED, never held. Holding the last error
    trains -- and flies -- ghost-chasing. The assembler does this; this
    node's job is simply to pass `None` rather than a stale Measurement.
  * WHAT IS RECORDED IS WHAT WAS SENT, post-clamp, not what the actor
    proposed. The safety supervisor is the only node that knows what
    actually reached the wire, so it publishes it back here.

The tick is driven by a timer, not by detection arrival: detections come at
the detector's pace (and stop entirely when the target is lost), while the
observation's `staleness` element only advances if something keeps
assembling. A detection-driven clock would freeze staleness at the exact
moment it matters.
"""
import math

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from chase_msgs.msg import DroneDetection, PolicyAction, TrackingState
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from chase_gym import constants as C
from chase_gym.observation import ObservationAssembler, ObservationSpec

from chase_flight.contract import load_policy, measurement_from_detection


class ChaseStateNode(Node):

    def __init__(self):
        super().__init__('chase_state')

        self.declare_parameter('checkpoint', '')
        self.declare_parameter('control_rate_hz', C.CONTROL_RATE_HZ)
        self.declare_parameter('detection_timeout_s', 1.0)

        ckpt = self.get_parameter('checkpoint').value
        rate = float(self.get_parameter('control_rate_hz').value)
        self._det_timeout = float(
            self.get_parameter('detection_timeout_s').value)

        # The spec must come from the checkpoint that will fly, not from
        # defaults: a policy is meaningless without the exact input layout
        # it trained under. Loading the bundle here (rather than trusting a
        # parameter) is what makes the hash check meaningful.
        if ckpt:
            pol = load_policy(ckpt, prefer_onnx=False, logger=self.get_logger())
            self.spec = pol.spec
            self.env_cfg = pol.env_cfg
            self.get_logger().info(
                f'observation spec from {ckpt}: dim={self.spec.dim} '
                f'hash={pol.obs_spec_hash} '
                f'ref_width_m={self.env_cfg.ref_width_m:.3f}')
            if abs(pol.control_rate_hz - rate) > 1e-6:
                self.get_logger().warn(
                    f'checkpoint trained at {pol.control_rate_hz} Hz but this '
                    f'node ticks at {rate} Hz -- every velocity the policy '
                    f'learned is scaled by the ratio')
        else:
            self.spec = ObservationSpec()
            from chase_gym.env import EnvConfig
            self.env_cfg = EnvConfig()
            self.get_logger().warn(
                'no checkpoint given: using DEFAULT observation spec '
                f'(hash {self.spec.spec_hash()}). Set the `checkpoint` '
                'parameter so the spec is proven against the policy.')

        self.asm = ObservationAssembler(self.spec)
        self.dt = 1.0 / rate

        self._det = None            # newest DroneDetection
        self._det_stamp = None      # rclpy Time it arrived
        self._det_consumed = True   # consume-once: see _fresh_detection
        # Last POSITIVE detection. The detector publishes a message for every
        # frame INCLUDING misses, so message age is a detector-liveness
        # signal and says nothing about whether the target is still there.
        # The supervisor's target-loss rule needs this one.
        self._last_seen = None
        self._last_action = np.zeros(self.spec.action_dim, dtype=np.float32)

        reliable = QoSProfile(depth=10)
        reliable.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(DroneDetection, 'chase/detection',
                                 self._on_detection, reliable)
        # What was SENT, fed back by the safety supervisor.
        self.create_subscription(PolicyAction, 'chase/action_sent',
                                 self._on_action_sent, reliable)
        self.pub = self.create_publisher(TrackingState, 'chase/observation',
                                         reliable)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f'chase_state ticking at {rate:.1f} Hz (dt={self.dt:.3f} s)')

    # ------------------------------------------------------------------
    def _on_detection(self, msg: DroneDetection):
        self._det = msg
        self._det_stamp = self.get_clock().now()
        self._det_consumed = False

    def _on_action_sent(self, msg: PolicyAction):
        a = np.asarray(msg.action, dtype=np.float32)
        if a.shape == (self.spec.action_dim,):
            self._last_action = a

    # ------------------------------------------------------------------
    def _fresh_detection(self):
        """The newest UNCONSUMED detection, else None.

        Consume-once is the contract: each detection informs exactly one
        control tick. Deciding freshness purely by an age threshold is not
        enough -- a detection younger than the threshold would be re-used on
        every tick until it aged out, which both double-counts it as
        'visible' and STALLS the staleness counter at zero for the whole
        window. Staleness stalling is the dangerous half: the observation
        would claim the target is being seen at exactly the moment it was
        lost, and `visible` is what the behaviour node trips on.

        The age bound remains as an upper guard, so that a detection
        delayed past the loss timeout is never presented as current.
        """
        if self._det is None or self._det_stamp is None or self._det_consumed:
            age = (math.inf if self._det_stamp is None else
                   (self.get_clock().now() - self._det_stamp).nanoseconds / 1e9)
            return None, age
        age = (self.get_clock().now() - self._det_stamp).nanoseconds / 1e9
        if age > self._det_timeout:
            return None, age
        self._det_consumed = True
        return self._det, age

    def _tick(self):
        # 1. record what was SENT (ordering contract: before assemble)
        self.asm.record_action(self._last_action)

        # 2. assemble
        det, age = self._fresh_detection()
        meas, _scale = measurement_from_detection(det, self.spec)
        if meas is None:
            det_for_stamp = None
        else:
            det_for_stamp = det
        obs = self.asm.assemble(meas, self.dt)

        # 3. publish
        out = TrackingState()
        now = self.get_clock().now()
        out.header.stamp = now.to_msg()
        out.header.frame_id = 'camera_optical'
        out.t_capture = (det_for_stamp.header.stamp if det_for_stamp
                         else TimeMsg())
        out.t_receive = now.to_msg()
        out.observation = [float(x) for x in obs]
        out.obs_spec_hash = self.spec.spec_hash()
        out.ex = float(obs[0])
        out.ey = float(obs[1])
        out.visible = bool(obs[4] >= 0.5)
        out.staleness = float(obs[5])
        if meas is not None:
            out.w_px = float(meas.w_px)
            out.range_m = float(C.FX * self.env_cfg.ref_width_m /
                                max(meas.w_px, 1e-6))
        else:
            out.w_px = 0.0
            out.range_m = float('nan')
        out.detection_age_s = float(age if math.isfinite(age) else -1.0)
        if meas is not None:
            self._last_seen = now
        out.time_since_detection_s = (
            -1.0 if self._last_seen is None
            else float((now - self._last_seen).nanoseconds / 1e9))
        self.pub.publish(out)


def main(argv=None):
    rclpy.init(args=argv)
    node = ChaseStateNode()
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
