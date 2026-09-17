"""FaithfulPointMassEnv -- the original training world, ported exactly.

The reproduction arm (arm F) carries a claim about the PUBLISHED system, so
it trains in the published environment: the point-mass image world of
`drone_sim_env.py` [C, rl_block_diagram.md section 3], not in ChaseEnv.

Verified semantics being ported (every line cited in rl_block_diagram.md):
  * observation: the box centre in RAW pixels, Box([0,0]..[960,720]);
  * action: Box(-60, +60)^2; a[0] = horizontal, a[1] = vertical
    (NOTE: the original's order -- opposite of ChaseEnv's [vertical, yaw]);
  * dynamics: position += 2.0 * action, clamped to the frame;
  * done: the clamp touched a frame edge ("target left the frame");
  * reward, threshold 100 [C drone_sim_env.py:70]:
        r = 100 - dist     if dist <= 100
        r = -0.25 * dist   if dist  > 100
  * reset: position ~ Uniform([0,960] x [0,720]);
  * the 10-step cutoff lives in the trainer (keras-rl forced done=True at
    nb_max_episode_steps -- reproduced there as truncation_as_terminal
    [D-impl 7]).

No latency, no noise, no target motion, no depth axis -- that absence IS the
finding (rl_block_diagram.md section 3); do not "improve" this file.
"""
from typing import Optional

import gymnasium as gym
import numpy as np

from . import constants as C
from .observation import ObservationSpec


class FaithfulPointMassEnv(gym.Env):
    metadata = {'render_modes': []}

    ACTION_SCALE = 60.0
    STEP_GAIN = 2.0            # position += 2 * action  [C drone_sim_env.py]

    # The same contract surface every env exposes (checkpoint bundles and
    # the trainer read these uniformly; no special-casing at call sites).
    obs_spec = ObservationSpec(mode='raw_pixels')

    def env_config_dict(self) -> dict:
        return {'env': 'FaithfulPointMassEnv',
                'action_scale': self.ACTION_SCALE,
                'step_gain': self.STEP_GAIN,
                'reward_version': 'faithful',
                'obs_spec': self.obs_spec.to_dict()}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([float(C.FRAME_W), float(C.FRAME_H)],
                          dtype=np.float32),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(
            -self.ACTION_SCALE, self.ACTION_SCALE, shape=(2,),
            dtype=np.float32)
        self._pos = np.array([C.CX, C.CY], dtype=np.float64)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._pos = np.array([
            self.np_random.uniform(0.0, float(C.FRAME_W)),
            self.np_random.uniform(0.0, float(C.FRAME_H)),
        ])
        return self._pos.astype(np.float32).copy(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(2)
        self._pos = self._pos + self.STEP_GAIN * a
        self._pos[0] = float(np.clip(self._pos[0], 0.0, float(C.FRAME_W)))
        self._pos[1] = float(np.clip(self._pos[1], 0.0, float(C.FRAME_H)))
        terminated = bool(
            self._pos[0] <= 0.0 or self._pos[0] >= float(C.FRAME_W)
            or self._pos[1] <= 0.0 or self._pos[1] >= float(C.FRAME_H))

        dist = float(np.hypot(self._pos[0] - C.CX, self._pos[1] - C.CY))
        if dist <= C.REWARD_THRESHOLD_PX:
            reward = C.REWARD_THRESHOLD_PX - dist
        else:
            reward = -0.25 * dist

        obs = self._pos.astype(np.float32).copy()
        return obs, reward, terminated, False, {'dist_px': dist}
