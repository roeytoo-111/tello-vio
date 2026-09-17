"""tello_sim_shim -- makes Gazebo speak Tello
(sim_training_architecture.md 3.2).

One thin node per vehicle:
  in:  /<ns>/cmd_vel        geometry_msgs/Twist, sticks in [-1, 1]
                            -- the SAME contract as the real driver
  map: linear.x,y,z -> v = stick * v_max ; angular.z -> stick * omega_max
  out: /<ns>/cmd_vel_metric geometry_msgs/Twist, metric body-frame
                            velocities, bridged by ros_gz to the model's
                            gz cmd_vel (MulticopterVelocityControl input)

Extras replicated from the real driver [V node.py:1099-1114]:
  * dead-man: input stale > 0.35 s -> zero command;
  * continuous republish at 20 Hz (the driver's RC resend rate) --
    hold-last-command semantics, never pulse-and-zero
    [rl_block_diagram.md 6 step 6].

Everything ABOVE this shim -- chase_state, chase_policy, chase_mixer,
chase_safety -- runs unchanged against sim topics: that is the entire
point of Tier B's ROS-rehearsal role.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class TelloSimShim(Node):
    def __init__(self):
        super().__init__('tello_sim_shim')
        self.declare_parameter('v_max', 1.5)          # m/s  [D safety]
        self.declare_parameter('omega_max', 1.5)      # rad/s
        self.declare_parameter('deadman_s', 0.35)     # the driver's timeout
        self.declare_parameter('republish_hz', 20.0)  # the driver's RC rate

        self.v_max = float(self.get_parameter('v_max').value)
        self.omega_max = float(self.get_parameter('omega_max').value)
        self.deadman_s = float(self.get_parameter('deadman_s').value)

        self._last_cmd = Twist()
        self._last_stamp = self.get_clock().now()

        self.sub = self.create_subscription(Twist, 'cmd_vel', self._on_cmd, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel_metric', 10)
        period = 1.0 / float(self.get_parameter('republish_hz').value)
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f'shim up: sticks[-1,1] -> v_max {self.v_max} m/s, '
            f'omega_max {self.omega_max} rad/s, dead-man {self.deadman_s}s')

    @staticmethod
    def _clamp(x: float) -> float:
        return max(-1.0, min(1.0, x))

    def _on_cmd(self, msg: Twist) -> None:
        self._last_cmd = msg
        self._last_stamp = self.get_clock().now()

    def _tick(self) -> None:
        out = Twist()
        age = (self.get_clock().now() - self._last_stamp).nanoseconds * 1e-9
        if age <= self.deadman_s:
            c = self._last_cmd
            out.linear.x = self._clamp(c.linear.x) * self.v_max
            out.linear.y = self._clamp(c.linear.y) * self.v_max
            out.linear.z = self._clamp(c.linear.z) * self.v_max
            out.angular.z = self._clamp(c.angular.z) * self.omega_max
        # else: all zeros -- the dead-man, exactly like the real driver.
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TelloSimShim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
