"""chase_depth_rule -- the forward axis, hand-coded and shared.

This node owns `linear.x` and nothing else. It imports
`chase_gym.env.forward_command`: the SAME function Tier A stepped its
environment with and Tier B's GzChaseEnv calls directly. The policy
therefore learned to centre the target in a world whose forward motion
obeyed exactly this law, and the aircraft flies that same law. Re-deriving
it here -- however carefully -- would be the classic train/deploy skew.

WHY THE PUBLISHED RATIO RULE IS NOT USED. The source work commands
forward/hold/backward from the fraction of the frame the box covers (20 % /
55 % linear, 25 % / 50 % by area in its code). Against a Tello-sized target
and this camera, the "too far, move forward" threshold is not reached until
roughly 0.47 m (body box) and the "too close" threshold sits near 0.17 m --
so a faithful port would command forward at every safe separation and stop
only when the aircraft are a few tens of centimetres apart. That is a
collision policy, not a chase policy. Because the target's width is a known
constant, the box width converts directly to metric range through the
pinhole relation d = fx * W / w, and the rule becomes a metric standoff
controller whose setpoint is stated in metres and can be chosen for safety.

The law, for the record (chase_gym/env.py):
  FOLLOW     d = fx*W/w ; err = d - standoff ; deadband +/- 0.25 m ;
             v = clip(0.8 * err, -v_fwd_max, +v_fwd_max)
  INTERCEPT  v = clip(close_gain * (d - r_cap), v_close_min, v_close_max),
             and 0 once inside r_cap
  MISS       v = 0 -- the forward axis holds on a lost target; it does not
             coast, and it does not search (that is the behaviour node's
             job, on the yaw axis).

Output is METRES PER SECOND. The conversion to normalised sticks belongs to
the mixer, which owns the (measured) full-stick speed of this airframe.
"""
import math

import rclpy
from chase_msgs.msg import TrackingState
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from chase_gym import constants as C
from chase_gym.corruption import Measurement
from chase_gym.env import EnvConfig, forward_command

from chase_flight.contract import env_cfg_from_bundle


class ChaseDepthRuleNode(Node):

    def __init__(self):
        super().__init__('chase_depth_rule')

        self.declare_parameter('checkpoint', '')
        # Overrides, used only when no checkpoint is supplied. With a
        # checkpoint the config comes from the bundle, so the flown law is
        # provably the trained law.
        self.declare_parameter('task', 'follow')
        self.declare_parameter('standoff_m', 2.0)
        self.declare_parameter('v_fwd_max', 0.8)

        ckpt = str(self.get_parameter('checkpoint').value)
        if ckpt:
            from chase_train.checkpoint import load_bundle
            self.cfg = env_cfg_from_bundle(load_bundle(ckpt))
            src = f'checkpoint {ckpt}'
        else:
            self.cfg = EnvConfig(
                task=str(self.get_parameter('task').value),
                standoff_m=float(self.get_parameter('standoff_m').value),
                v_fwd_max=float(self.get_parameter('v_fwd_max').value))
            src = 'parameters (NO checkpoint -- law may differ from training)'

        self.get_logger().info(
            f'depth rule: task={self.cfg.task} from {src}; '
            f'ref_width_m={self.cfg.ref_width_m:.3f} '
            f'standoff={self.cfg.standoff_m:.2f} m '
            f'deadband=+/-{self.cfg.standoff_deadband_m:.2f} m '
            f'gain={self.cfg.standoff_gain:.2f} /s '
            f'v_fwd_max={self.cfg.v_fwd_max:.2f} m/s '
            f'| intercept: r_cap={self.cfg.r_cap_m:.2f} m '
            f'v_close=[{self.cfg.v_close_min:.2f}, '
            f'{self.cfg.v_close_max:.2f}] m/s')

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(TwistStamped, 'chase/forward_cmd',
                                         qos)
        self.create_subscription(TrackingState, 'chase/observation',
                                 self._on_state, qos)

    def _on_state(self, msg: TrackingState):
        # Rebuild the Measurement the law expects. Only w_px matters to it;
        # u and v are carried for completeness. When the target is not
        # visible we pass None, which is how the law yields a hold.
        meas = None
        if msg.visible and msg.w_px > 0.0 and math.isfinite(msg.w_px):
            meas = Measurement(C.CX, C.CY, float(msg.w_px))
        v_fwd = forward_command(self.cfg, meas)

        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.twist.linear.x = float(v_fwd)      # METRES PER SECOND
        self.pub.publish(out)


def main(argv=None):
    rclpy.init(args=argv)
    node = ChaseDepthRuleNode()
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
