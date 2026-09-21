"""chase_policy -- the trained actor, INFERENCE ONLY.

No noise, no learning, no exploration in flight: the aircraft flies a fixed
checkpoint. (An updating policy on a live aircraft has no safety argument,
and the source work deploys a frozen actor too.)

What deploys is the ACTOR ALONE. Critics, targets, replay buffer and noise
process are training-only and never loaded. Inference runs on the exported
ONNX graph (opset 17, deterministic, parity-checked at export), with the
torch bundle as the source of truth for everything the graph cannot carry:
the observation-spec hash, the action map and the training config. Loading
both and cross-checking them is the point -- the ONNX alone could not prove
it belongs to this observation layout.

The node REFUSES to run when the observation it is handed does not carry
the spec hash it loaded. That refusal is the whole sim-to-real audit trail
in one line: an actor is meaningless without the exact input contract it
trained under, and a silent mismatch would fly a network on numbers that
mean something different from what it learned.

Normalisation and action unscaling are deliberately NOT in the graph -- they
live in the shared modules (the assembler upstream, the stick convention
downstream), so that train and deploy cannot drift apart.
"""
import os
import time

import numpy as np
import rclpy
from chase_msgs.msg import PolicyAction, TrackingState
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from chase_flight.contract import load_policy


class _BaselinePolicy:
    """Adapter giving a chase_gym baseline the PolicyBundle surface, so the
    node body is identical whether a network or a hand-written controller
    is flying."""

    def __init__(self, ctl, spec):
        self._ctl = ctl
        self.spec = spec
        self.backend = 'baseline'
        self.obs_spec_hash = spec.spec_hash()
        self.arm = 'baseline'
        self.env_step = -1
        self.git_sha = ''

    def infer(self, obs):
        return np.asarray(self._ctl.get_action(obs), dtype=np.float32)


class ChasePolicyNode(Node):

    def __init__(self):
        super().__init__('chase_policy')

        self.declare_parameter('checkpoint', '')
        self.declare_parameter('prefer_onnx', True)
        # The documented flight ladder puts a hand-written controller on the
        # aircraft BEFORE any learned policy: it exercises the whole safety
        # chain, produces the latency and target-dynamics measurements the
        # simulator needs, and yields the baseline number that makes the RL
        # result interpretable. Setting `baseline` flies chase_gym's
        # PController/PNController through this same node, so the graph,
        # the observation and the safety path are identical either way.
        self.declare_parameter('baseline', '')      # '' | p_controller | pn
        # Rate limit on the action (per control step, per axis). The Tello's
        # sticks saturate quickly and an unconstrained policy produces
        # visible judder; a rate limit on the OUTPUT achieves most of what a
        # smoothness reward would, without touching the trained policy.
        self.declare_parameter('rate_limit', 0.5)

        ckpt = str(self.get_parameter('checkpoint').value)
        baseline = str(self.get_parameter('baseline').value).strip()
        self.rate_limit = float(self.get_parameter('rate_limit').value)
        prefer_onnx = bool(self.get_parameter('prefer_onnx').value)

        if baseline:
            from chase_gym.baselines import PController, PNController
            from chase_gym.observation import ObservationSpec
            if baseline not in ('p_controller', 'pn'):
                raise SystemExit(
                    f'chase_policy: unknown baseline {baseline!r} '
                    f'(expected p_controller | pn)')
            spec = ObservationSpec()
            ctl = (PController(spec) if baseline == 'p_controller'
                   else PNController(spec))
            self.pol = _BaselinePolicy(ctl, spec)
            self.checkpoint_id = f'baseline:{baseline}'
            self.get_logger().warn(
                f'flying the {baseline} BASELINE, not a learned policy '
                f'(obs_hash {self.pol.obs_spec_hash})')
        else:
            if not ckpt:
                raise SystemExit(
                    'chase_policy: set `checkpoint` (a trained bundle) or '
                    '`baseline` (p_controller | pn) -- there is no default')
            if not os.path.exists(ckpt):
                raise SystemExit(f'chase_policy: checkpoint not found: {ckpt}')
            try:
                self.pol = load_policy(ckpt, prefer_onnx=prefer_onnx,
                                       logger=self.get_logger())
            except ValueError as exc:
                # A refusal is a hard stop, never a warning: the
                # alternative is flying an unproven contract.
                raise SystemExit(
                    f'chase_policy REFUSED the checkpoint: {exc}')
            self.checkpoint_id = os.path.basename(ckpt)
            self.get_logger().info(
                f'policy {self.checkpoint_id} arm={self.pol.arm} '
                f'step={self.pol.env_step} backend={self.pol.backend} '
                f'obs_hash={self.pol.obs_spec_hash} '
                f'git={self.pol.git_sha[:8]}')

        self._prev = np.zeros(self.pol.spec.action_dim, dtype=np.float32)
        self._warned_hash = False

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(PolicyAction, 'chase/action', qos)
        self.create_subscription(TrackingState, 'chase/observation',
                                 self._on_observation, qos)

    # ------------------------------------------------------------------
    def _on_observation(self, msg: TrackingState):
        if msg.obs_spec_hash != self.pol.obs_spec_hash:
            if not self._warned_hash:
                self.get_logger().error(
                    f'observation spec hash {msg.obs_spec_hash!r} != policy '
                    f'{self.pol.obs_spec_hash!r} -- refusing to act. The '
                    f'state node and the checkpoint disagree about what the '
                    f'numbers mean.')
                self._warned_hash = True
            return

        obs = np.asarray(msg.observation, dtype=np.float32)
        if obs.shape != (self.pol.spec.dim,):
            self.get_logger().error(
                f'observation dim {obs.shape} != {self.pol.spec.dim}')
            return

        t0 = time.perf_counter()
        raw = self.pol.infer(obs)
        infer_ms = (time.perf_counter() - t0) * 1e3

        a = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
        if self.rate_limit > 0.0:
            a = np.clip(a, self._prev - self.rate_limit,
                        self._prev + self.rate_limit)
        self._prev = a

        out = PolicyAction()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id
        out.t_capture = msg.t_capture
        out.t_receive = msg.t_receive
        out.action = [float(x) for x in a]
        out.a_vertical = float(a[0])
        out.a_yaw = float(a[1])
        out.stale = bool(msg.staleness >= 1.0)
        out.checkpoint_id = self.checkpoint_id
        out.backend = self.pol.backend
        out.obs_spec_hash = self.pol.obs_spec_hash
        out.inference_ms = float(infer_ms)
        self.pub.publish(out)


def main(argv=None):
    rclpy.init(args=argv)
    node = ChasePolicyNode()
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
