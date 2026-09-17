"""ChaseEnv -- Tier A, the gradient mill (rl_training_guide.md 16).

One env step is EXACTLY one deployment control period (Delta_t = 0.1 s), and
the step order is the guide's 16.1 ten-liner: target advances, the follower's
lagged velocities respond, the TRUE geometry is projected and queued, the
observation is sampled Delta old and corrupted, the reward is computed from
truth. The policy owns [linear.z, angular.z]; the forward axis belongs to the
hand-coded law (metric standoff for FOLLOW, bounded closing law for
INTERCEPT) acting on the same delayed measurement stream the policy sees --
axis ownership exclusive, as deployed (implementation_plan.md 3).

Termination is task truth: target centre out of frame (loss, with penalty),
or INTERCEPT capture (range <= r_cap while in view, bonus +B). The step
budget is TRUNCATION, never terminal -- the strongest warning in the guide
(11.3): stored d flags must bootstrap through timeouts.
"""
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from . import constants as C
from .corruption import CorruptionConfig, Measurement
from .kinematics import FollowerKinematics, project, world_to_camera
from .observation import ObservationAssembler, ObservationSpec
from .pipeline import (MeasurementPipe, body_offset_from_frame_draw,
                       draw_reset_placement)
from .reward import RewardComputer, RewardConfig
from . import target_motion


@dataclass
class EnvConfig:
    task: str = 'follow'                     # 'follow' | 'intercept'
    dt: float = C.DT
    episode_steps: int = 300                 # guide 16.1 line 10: 200-500
    scenario: str = 'static'                 # family name or 'mix'
    scenario_mix: Sequence[str] = target_motion.FAMILIES
    target_speed_cap: float = 1.0            # m/s; a parameter of every result
    # Follower stick gains [D-impl 2].
    v_max: float = 1.5                       # m/s per unit stick (linear.z)
    omega_max: float = 1.5                   # rad/s per unit stick (angular.z)
    # First-order lag [D-impl 3]: PLACEHOLDER until the Phase-1 step-response
    # flight measures it; +/-30 % per-episode jitter per guide 16.1 line 2.
    t_lag: float = 0.25
    t_lag_jitter: float = 0.30
    # Latency injection (measured distribution [V]).
    latency: bool = True
    latency_range_s: Tuple[float, float] = C.LATENCY_RANGE_S
    latency_jitter_std_s: float = 0.020
    # Detector corruption.
    corruption: bool = True
    corruption_cfg: CorruptionConfig = field(default_factory=CorruptionConfig)
    # Observation contract.
    obs_mode: str = 'latency_aware'          # 'latency_aware' | 'raw_pixels'
    k: int = C.K_ACTION_HISTORY
    # Reward.
    reward_version: str = 'repaired'
    w_smooth: float = 0.05
    w_loss: float = 5.0
    gamma: float = 0.99                      # for INTERCEPT shaping only
    # Target reference width the oracle projects with (chase_detector
    # geometry: body 0.098 m vs prop-span 0.18 m -- printed, never implicit).
    ref_width_m: float = C.TELLO_BODY_W
    # FOLLOW standoff law [D-impl 9] (metric rule, NEVER the paper's ratios:
    # drl_ros2_reference_analysis.md 6.3 -- those command collision).
    standoff_m: float = 2.0
    standoff_deadband_m: float = 0.25
    standoff_gain: float = 0.8               # 1/s
    v_fwd_max: float = 0.8                   # m/s
    # INTERCEPT closing law [D-impl 10]. The floor speed keeps the approach
    # crossing r_cap instead of stalling asymptotically on it: "fast far,
    # slow near" -- but never zero while outside the capture radius.
    r_cap_m: float = 0.5
    close_gain: float = 1.0                  # 1/s
    v_close_max: float = 1.2                 # m/s
    v_close_min: float = 0.15                # m/s, while outside r_cap
    capture_bonus: float = 10.0
    shaping_lambda: float = 0.1
    # Reset draw: standoff band (FOLLOW) / approach band (INTERCEPT).
    reset_range_m: Optional[Tuple[float, float]] = None
    reset_frame_margin: float = 0.15         # uniform box-centre placement

    def resolved_reset_range(self) -> Tuple[float, float]:
        if self.reset_range_m is not None:
            return self.reset_range_m
        return (2.0, 6.0) if self.task == 'intercept' else (1.5, 2.5)


def reward_config_from(cfg: EnvConfig, trainer_gamma: Optional[float] = None
                       ) -> RewardConfig:
    """The ONE mapping EnvConfig -> RewardConfig, used by both envs and by
    the demo-reward recompute (so demos and on-policy transitions can never
    carry two different reward functions). When the trainer's gamma is
    known, it must equal the env's -- the shaping's policy invariance
    (guide 22) depends on it."""
    if trainer_gamma is not None and abs(trainer_gamma - cfg.gamma) > 1e-12:
        raise ValueError(
            f'trainer gamma {trainer_gamma} != env gamma {cfg.gamma}; '
            f'potential-based shaping is only policy-invariant at the '
            f'discount actually trained with')
    return RewardConfig(
        version=cfg.reward_version, w_smooth=cfg.w_smooth,
        w_loss=cfg.w_loss, intercept=(cfg.task == 'intercept'),
        gamma=cfg.gamma, shaping_lambda=cfg.shaping_lambda,
        capture_bonus=cfg.capture_bonus)


def forward_command(cfg: EnvConfig, meas) -> float:
    """The hand-coded forward-axis law, shared by every tier (Tier B's
    GzChaseEnv imports THIS function): metric standoff for FOLLOW, bounded
    closing law with a floor speed for INTERCEPT -- both on the DELAYED
    measurement, exactly like the deployed depth rule [D-impl 9/10]."""
    if meas is None or meas.w_px <= 0:
        return 0.0
    d_est = C.FX * cfg.ref_width_m / meas.w_px       # pinhole range
    if cfg.task == 'intercept':
        if d_est <= cfg.r_cap_m:
            return 0.0                   # inside: let the terminal fire
        return float(np.clip(cfg.close_gain * (d_est - cfg.r_cap_m),
                             cfg.v_close_min, cfg.v_close_max))
    err = d_est - cfg.standoff_m
    if abs(err) < cfg.standoff_deadband_m:
        return 0.0
    return float(np.clip(cfg.standoff_gain * err,
                         -cfg.v_fwd_max, cfg.v_fwd_max))


class ChaseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        if self.cfg.task not in ('follow', 'intercept'):
            raise ValueError(f'unknown task {self.cfg.task!r}')

        self.obs_spec = ObservationSpec(mode=self.cfg.obs_mode, k=self.cfg.k,
                                        control_dt_s=self.cfg.dt)
        self._assembler = ObservationAssembler(self.obs_spec)
        self._reward = RewardComputer(reward_config_from(self.cfg))
        self._pipe = MeasurementPipe.from_env_cfg(self.cfg)
        self._follower = FollowerKinematics(self.cfg.t_lag)

        if self.obs_spec.mode == 'raw_pixels':
            low = np.array([0.0, 0.0], dtype=np.float32)
            high = np.array([C.FRAME_W, C.FRAME_H], dtype=np.float32)
        else:
            low = -np.ones(self.obs_spec.dim, dtype=np.float32)
            high = np.ones(self.obs_spec.dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,),
                                           dtype=np.float32)

        # Curriculum hook (guide 20): the trainer narrows/widens these.
        self._stage_families: Tuple[str, ...] = self._initial_families()
        self._stage_cap = self.cfg.target_speed_cap
        self._target = None
        self._t = 0.0
        self._steps = 0

    # ---- curriculum interface -------------------------------------------
    def _initial_families(self) -> Tuple[str, ...]:
        if self.cfg.scenario == 'mix':
            return tuple(self.cfg.scenario_mix)
        return (self.cfg.scenario,)

    def set_stage(self, families: Sequence[str], speed_cap: float) -> None:
        for f in families:
            if f not in target_motion.FAMILIES:
                raise ValueError(f'unknown family {f!r}')
        self._stage_families = tuple(families)
        self._stage_cap = float(speed_cap)

    # ---- gym API ---------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        cfg = self.cfg

        # Steps 1-4 of the DRAW ORDER CONTRACT (pipeline.py): placement,
        # family, latency base, generator params -- shared with Tier B.
        z_cam, u0, v0, family = draw_reset_placement(
            rng, cfg.resolved_reset_range(), cfg.reset_frame_margin,
            self._stage_families)
        self._pipe.reset(rng)
        self._assembler.reset()

        follower_pos = np.array([0.0, 0.0, 1.0])
        target_pos = follower_pos + body_offset_from_frame_draw(z_cam, u0, v0)
        target_pos[2] = max(target_pos[2], 0.2)   # stay above the floor
        self._target = target_motion.make(family, rng, target_pos,
                                          self._stage_cap)
        self._family = family

        # Step 5: tier-specific draws LAST -- Tier A's lag jitter
        # (guide 16.1 line 2).
        jitter = rng.uniform(-cfg.t_lag_jitter, cfg.t_lag_jitter)
        self._t_lag_ep = cfg.t_lag * (1.0 + jitter)
        self._follower = FollowerKinematics(self._t_lag_ep)
        self._follower.reset(pos=follower_pos, yaw=0.0)

        self._t = 0.0
        self._steps = 0
        self._a_prev = np.zeros(2, dtype=np.float32)
        self._last_meas: Optional[Measurement] = None

        cam = world_to_camera(self._target.pos, self._follower.state)
        u, v, w_px, in_frame = project(cam, cfg.ref_width_m)
        self._pipe.preroll(0.0, cfg.dt,
                           Measurement(u, v, w_px) if in_frame else None,
                           cfg.latency_range_s[1], cfg.latency_jitter_std_s)
        self._prev_range = float(np.linalg.norm(
            self._target.pos - self._follower.state.pos))

        # First observation: sample the pre-rolled pipe at t = 0.
        meas = self._pipe.sample(self._t, rng)
        self._last_meas = meas
        obs = self._assembler.assemble(meas, cfg.dt)
        return obs, self._info(u, v, w_px, in_frame, meas is not None,
                               self._prev_range)

    def step(self, action):
        cfg = self.cfg
        rng = self.np_random
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(2),
                    -1.0, 1.0)

        # 1. Target moves (guide 16.1 line 1).
        self._target.advance(cfg.dt)

        # 2. Forward axis: the hand-coded law on the DELAYED measurement the
        #    policy also sees -- the deployed depth rule acts on detections,
        #    not on truth [D-impl 9/10].
        v_fwd_cmd = self._forward_command(self._last_meas)

        # 3. Follower first-order lag + integration (line 2).
        self._follower.step(v_fwd_cmd, float(a[0]) * cfg.v_max,
                            float(a[1]) * cfg.omega_max, cfg.dt)
        self._t += cfg.dt
        self._steps += 1

        # 4. TRUE geometry, projected with OUR calibration (line 3).
        cam = world_to_camera(self._target.pos, self._follower.state)
        u, v, w_px, in_frame = project(cam, cfg.ref_width_m)
        true_range = float(np.linalg.norm(
            self._target.pos - self._follower.state.pos))
        if cam[2] > 1e-3:
            true_dist = math.hypot(u - C.CX, v - C.CY)
        else:
            true_dist = C.MAX_DIST_PX      # behind the camera: worst case

        # 5. Truth enters the delay line only while the oracle would see it
        #    (FOV test -- sim_training_architecture.md 3.3 step 4).
        if in_frame:
            self._pipe.push(self._t, Measurement(u, v, w_px))

        # 6-7. Sample Delta old, corrupt (lines 5-6).
        meas = self._pipe.sample(self._t, rng)
        self._last_meas = meas

        # 8. Observation via THE shared module (line 7); history carries the
        #    action just sent.
        self._assembler.record_action(a)
        obs = self._assembler.assemble(meas, cfg.dt)

        # 9-10. Terminals from truth; budget is truncation (lines 9-10).
        captured = (cfg.task == 'intercept' and in_frame
                    and true_range <= cfg.r_cap_m)
        lost = not in_frame
        terminated = bool(lost or captured)
        truncated = bool(not terminated and self._steps >= cfg.episode_steps)

        r = self._reward.compute(
            dist_px=true_dist, action=a, prev_action=self._a_prev,
            lost=lost and not captured, captured=captured,
            range_m=true_range, prev_range_m=self._prev_range)

        self._a_prev = a
        self._prev_range = true_range
        info = self._info(u, v, w_px, in_frame, meas is not None, true_range)
        if captured:
            info['capture'] = True
        return obs, float(r), terminated, truncated, info

    # ---- internals -------------------------------------------------------
    def _forward_command(self, meas: Optional[Measurement]) -> float:
        return forward_command(self.cfg, meas)

    def _info(self, u, v, w_px, in_frame, detected, range_m) -> dict:
        return {
            'dist_px': float(math.hypot(u - C.CX, v - C.CY)) if in_frame else float('nan'),
            'in_frame': bool(in_frame),
            'detected': bool(detected),
            'range_m': float(range_m),
            'u': float(u), 'v': float(v), 'w_px': float(w_px),
            'family': getattr(self, '_family', ''),
            'latency_base_s': self._pipe.latency_base_s,
            't_lag_s': getattr(self, '_t_lag_ep', self.cfg.t_lag),
        }

    def env_config_dict(self) -> dict:
        d = asdict(self.cfg)
        d['obs_spec'] = self.obs_spec.to_dict()
        return d
