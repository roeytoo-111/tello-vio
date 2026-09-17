"""GzChaseEnv -- Tier B lockstep RL (sim_training_architecture.md 3.4).

The SAME trainer runs here unchanged: same observation assembler, same
reward module, same forward-axis law, same action convention -- all
imported from chase_gym, never re-implemented. What changes is only the
dynamics under the contract: rigid-body gz-sim physics (or the kinematic
fake in tests) instead of the first-order-lag point model.

One env step = publish commands, advance EXACTLY steps_per_control physics
iterations (100 x 1 ms = one 0.1 s control period [V]), read poses, run the
oracle -> latency -> corruption pipe on SIM time, assemble, reward from
truth. Determinism comes from lockstep + seeded generators; parallel
instances isolate by GZ_PARTITION (one backend per env).

Role discipline (3.4 throughput honesty): this env funds the FINE-TUNE arm
(~1e4 steps) and evaluations -- the 2e5-step primary training stays in
Tier A.
"""
import math
from typing import Optional

import gymnasium as gym
import numpy as np

from chase_gym import constants as C
from chase_gym.corruption import Measurement
from chase_gym.env import EnvConfig, forward_command, reward_config_from
from chase_gym.kinematics import FollowerState, project, world_to_camera
from chase_gym.observation import ObservationAssembler, ObservationSpec
from chase_gym.pipeline import (MeasurementPipe, body_offset_from_frame_draw,
                                draw_reset_placement)
from chase_gym.reward import RewardComputer
from chase_gym import target_motion

from .gz_iface import GzBackendBase


class GzChaseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, backend: GzBackendBase,
                 cfg: Optional[EnvConfig] = None,
                 follower: str = 'follower', target: str = 'target',
                 steps_per_control: int = 100):
        super().__init__()
        self.backend = backend
        self.cfg = cfg or EnvConfig()
        self.follower = follower
        self.target = target
        self.steps_per_control = int(steps_per_control)
        expected = self.cfg.dt / backend.physics_dt
        if abs(expected - self.steps_per_control) > 1e-6:
            raise ValueError(
                f'steps_per_control {self.steps_per_control} != '
                f'dt/physics_dt = {expected:.1f} -- the control period '
                f'contract (guide 16.1) would silently break')

        self.obs_spec = ObservationSpec(mode=self.cfg.obs_mode, k=self.cfg.k,
                                        control_dt_s=self.cfg.dt)
        self._assembler = ObservationAssembler(self.obs_spec)
        self._reward = RewardComputer(reward_config_from(self.cfg))
        self._pipe = MeasurementPipe.from_env_cfg(self.cfg)

        self.observation_space = gym.spaces.Box(
            -np.ones(self.obs_spec.dim, dtype=np.float32),
            np.ones(self.obs_spec.dim, dtype=np.float32), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,),
                                           dtype=np.float32)
        self._started = False
        self._stage_families = (self.cfg.scenario,) \
            if self.cfg.scenario != 'mix' else tuple(self.cfg.scenario_mix)
        self._stage_cap = self.cfg.target_speed_cap

    def set_stage(self, families, speed_cap: float) -> None:
        self._stage_families = tuple(families)
        self._stage_cap = float(speed_cap)

    # ------------------------------------------------------------------
    def _project_truth(self):
        od_f = self.backend.get_odom(self.follower)
        od_t = self.backend.get_odom(self.target)
        st = FollowerState(pos=od_f.pos, yaw=od_f.yaw)
        cam = world_to_camera(od_t.pos, st)
        u, v, w_px, in_frame = project(cam, self.cfg.ref_width_m)
        rng_m = float(np.linalg.norm(od_t.pos - od_f.pos))
        dist = math.hypot(u - C.CX, v - C.CY) if cam[2] > 1e-3 \
            else C.MAX_DIST_PX
        return u, v, w_px, in_frame, dist, rng_m, od_t.pos

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        cfg = self.cfg
        if not self._started:
            self.backend.start()
            self._started = True
        self.backend.reset_world()

        # Steps 1-4 of the DRAW ORDER CONTRACT (chase_gym.pipeline):
        # exactly ChaseEnv's shared draw prefix -- one seed, one geometry,
        # one latency base, one generator, in every tier.
        z_cam, u0, v0, family = draw_reset_placement(
            rng, cfg.resolved_reset_range(), cfg.reset_frame_margin,
            self._stage_families)
        self._pipe.reset(rng)
        self._assembler.reset()

        f_pos = np.array([0.0, 0.0, 1.0])
        t_pos = f_pos + body_offset_from_frame_draw(z_cam, u0, v0)
        t_pos[2] = max(t_pos[2], 0.2)
        self.backend.set_pose(self.follower, f_pos, 0.0)
        self.backend.set_pose(self.target, t_pos, 0.0)
        # set_pose is queued; hovering zero-commands also re-arm the
        # controllers after the world reset re-instantiated them.
        self.backend.send_twist(self.follower, (0, 0, 0), 0.0)
        self.backend.send_twist(self.target, (0, 0, 0), 0.0)
        # >= one odom publish period WITH margin (the backend refuses
        # sub-period steps; exactly-one-period would ride float equality).
        self.backend.step(12)

        self._family = family
        od_t = self.backend.get_odom(self.target)
        self._gen = target_motion.make(family, rng, od_t.pos, self._stage_cap)

        # Pre-roll the delay pipe with the settled initial view.
        u, v, w_px, in_frame, dist, rng_m, _ = self._project_truth()
        t0 = self.backend.sim_time()
        self._pipe.preroll(t0, cfg.dt,
                           Measurement(u, v, w_px) if in_frame else None,
                           cfg.latency_range_s[1], cfg.latency_jitter_std_s)
        self._prev_range = rng_m
        self._a_prev = np.zeros(2, dtype=np.float32)
        self._steps = 0
        meas = self._sample_pipe(t0, rng)
        self._last_meas = meas
        obs = self._assembler.assemble(meas, cfg.dt)
        return obs, {'family': family, 'in_frame': in_frame,
                     'range_m': rng_m,
                     'latency_base_s': self._pipe.latency_base_s}

    def _sample_pipe(self, t: float, rng) -> Optional[Measurement]:
        return self._pipe.sample(t, rng)

    def step(self, action):
        cfg = self.cfg
        rng = self.np_random
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(2),
                    -1.0, 1.0)

        # Follower command: policy sticks + the shared forward law on the
        # delayed measurement. MulticopterVelocityControl consumes body-
        # frame linear velocity + yaw rate -- the Tello stick contract.
        v_fwd = forward_command(cfg, self._last_meas)
        self.backend.send_twist(
            self.follower, (v_fwd, 0.0, float(a[0]) * cfg.v_max),
            float(a[1]) * cfg.omega_max)

        # Scripted target: velocity command tracking its generator's path
        # (the generator is intent; physics decides what actually happens).
        od_t = self.backend.get_odom(self.target)
        p_next = self._gen.advance(cfg.dt)
        v_world = (p_next - od_t.pos) / cfg.dt
        speed = float(np.linalg.norm(v_world))
        if self._stage_cap == 0.0:
            v_world = np.zeros(3)
        elif speed > self._stage_cap:
            v_world = v_world * (self._stage_cap / speed)
        cy, sy = math.cos(od_t.yaw), math.sin(od_t.yaw)
        v_body = (cy * v_world[0] + sy * v_world[1],
                  -sy * v_world[0] + cy * v_world[1], v_world[2])
        self.backend.send_twist(self.target, v_body, 0.0)

        # THE lockstep: exactly one control period of physics.
        self.backend.step(self.steps_per_control)
        t = self.backend.sim_time()
        self._steps += 1

        u, v, w_px, in_frame, dist, rng_m, _ = self._project_truth()
        if in_frame:
            self._pipe.push(t, Measurement(u, v, w_px))
        meas = self._sample_pipe(t, rng)
        self._last_meas = meas
        self._assembler.record_action(a)
        obs = self._assembler.assemble(meas, cfg.dt)

        captured = (cfg.task == 'intercept' and in_frame
                    and rng_m <= cfg.r_cap_m)
        lost = not in_frame
        terminated = bool(lost or captured)
        truncated = bool(not terminated and self._steps >= cfg.episode_steps)
        r = self._reward.compute(
            dist_px=dist, action=a, prev_action=self._a_prev,
            lost=lost and not captured, captured=captured,
            range_m=rng_m, prev_range_m=self._prev_range)
        self._a_prev = a
        self._prev_range = rng_m
        info = {'dist_px': dist if in_frame else float('nan'),
                'in_frame': in_frame, 'detected': meas is not None,
                'range_m': rng_m, 'u': u, 'v': v, 'w_px': w_px,
                'family': self._family, 'sim_time': t}
        if captured:
            info['capture'] = True
        return obs, float(r), terminated, truncated, info

    def close(self):
        if self._started:
            self.backend.stop()
            self._started = False
