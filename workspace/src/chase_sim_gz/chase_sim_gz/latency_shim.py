"""latency_shim -- delays DroneDetection messages by the measured
distribution, on SIM time (sim_training_architecture.md 3.3): a sim-time
queue so it works in lockstep, where wall-clock delays would be meaningless.

Per episode (or per node lifetime here), a base delay is drawn from the
measured 150-350 ms band [V]; each message adds jitter. Downstream
consumers -- chase_state onward -- receive exactly the staleness the real
link imposes, with header stamps preserved (t_capture stays the capture
time; arrival is simply later), which is how the real transport behaves.
"""
import numpy as np
import rclpy
from rclpy.node import Node

from chase_msgs.msg import DroneDetection

from chase_gym import constants as C


class LatencyShim(Node):
    def __init__(self):
        super().__init__('latency_shim')
        self.declare_parameter('base_min_s', C.LATENCY_RANGE_S[0])
        self.declare_parameter('base_max_s', C.LATENCY_RANGE_S[1])
        self.declare_parameter('jitter_std_s', 0.020)
        self.declare_parameter('seed', 0)
        self.rng = np.random.default_rng(int(self.get_parameter('seed').value))
        lo = float(self.get_parameter('base_min_s').value)
        hi = float(self.get_parameter('base_max_s').value)
        self.base = float(self.rng.uniform(lo, hi))
        self.jitter = float(self.get_parameter('jitter_std_s').value)

        self._pending = []          # (release_sim_time, msg), time-ordered
        self.sub = self.create_subscription(
            DroneDetection, '/chase/detection', self._on_msg, 50)
        self.pub = self.create_publisher(
            DroneDetection, '/chase/detection_delayed', 50)
        self.create_timer(0.005, self._tick)   # sim-time timer (use_sim_time)
        self.get_logger().info(
            f'latency shim up: base {1000 * self.base:.0f} ms '
            f'(drawn from measured {1000 * lo:.0f}-{1000 * hi:.0f} ms), '
            f'jitter sd {1000 * self.jitter:.0f} ms, sim-time queue')

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_msg(self, msg: DroneDetection) -> None:
        delay = max(0.0, self.base + self.rng.normal(0.0, self.jitter))
        self._pending.append((self._now_s() + delay, msg))

    def _tick(self) -> None:
        now = self._now_s()
        while self._pending and self._pending[0][0] <= now:
            _, msg = self._pending.pop(0)
            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LatencyShim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
