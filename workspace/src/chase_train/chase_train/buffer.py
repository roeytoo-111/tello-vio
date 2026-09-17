"""Preallocated ring replay buffer + demonstration prefill (guide 17).

Rules encoded here, each with its counterexample in the doc set:
  * preallocated numpy ring, not deque+random.sample
    (drl_ros2_reference_analysis.md 7.1);
  * d stores TERMINATED only -- a truncation bootstraps (guide 11.3, "the
    strongest warning in this guide"); the trainer decides what it passes;
  * every transition carries its real dt (diagnostics, guide 17);
  * demonstrations load from per-episode .npz files and their rewards are
    RECOMPUTED with the live reward function on load, so the reward can be
    revised without re-flying (guide 17);
  * a minimum-fill guard: sampling below min_fill or below batch size is an
    error, not a silent short batch (offline_training_recipe.md section 5).
"""
import os
from typing import Callable, Dict, Optional

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 device: str = 'cpu'):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.obs2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.dt = np.zeros((capacity, 1), dtype=np.float32)
        self._idx = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def add(self, obs, act, rew, obs2, terminated: bool, dt: float) -> None:
        i = self._idx
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.obs2[i] = obs2
        self.done[i] = float(terminated)
        self.dt[i] = dt
        self._idx = (i + 1) % self.capacity
        if self._idx == 0:
            self._full = True

    def sample(self, batch_size: int, rng: np.random.Generator,
               min_fill: int = 0) -> Dict[str, torch.Tensor]:
        n = len(self)
        if n < max(batch_size, min_fill):
            raise RuntimeError(
                f'buffer has {n} < required {max(batch_size, min_fill)} '
                f'transitions -- refusing a short batch')
        idx = rng.integers(0, n, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)
        return {'obs': to(self.obs), 'act': to(self.act), 'rew': to(self.rew),
                'obs2': to(self.obs2), 'done': to(self.done)}


def save_demo_episode(path: str, transitions: Dict[str, np.ndarray]) -> None:
    """One episode -> one .npz (never an appended text log --
    drl_ros2_reference_analysis.md 3.1). Required keys: obs, act, obs2,
    terminated, dt, plus the true-geometry fields reward recomputation
    needs: dist_px, lost, captured, range_m, prev_range_m, prev_act."""
    required = {'obs', 'act', 'obs2', 'terminated', 'dt', 'dist_px', 'lost',
                'captured', 'range_m', 'prev_range_m', 'prev_act'}
    missing = required - set(transitions)
    if missing:
        raise ValueError(f'demo episode missing fields {sorted(missing)}')
    np.savez_compressed(path, **transitions)


def load_demos(directory: str, buffer: ReplayBuffer,
               reward_fn: Callable[..., float],
               limit: Optional[int] = None) -> int:
    """Load every episode .npz under `directory` into `buffer`, recomputing
    each reward with `reward_fn` (signature of RewardComputer.compute).
    Returns the number of transitions loaded."""
    files = sorted(f for f in os.listdir(directory) if f.endswith('.npz'))
    loaded = 0
    for fname in files:
        data = np.load(os.path.join(directory, fname))
        n = data['obs'].shape[0]
        for i in range(n):
            r = reward_fn(
                dist_px=float(data['dist_px'][i]),
                action=data['act'][i],
                prev_action=data['prev_act'][i],
                lost=bool(data['lost'][i]),
                captured=bool(data['captured'][i]),
                range_m=float(data['range_m'][i]),
                prev_range_m=float(data['prev_range_m'][i]))
            buffer.add(data['obs'][i], data['act'][i], r, data['obs2'][i],
                       bool(data['terminated'][i]), float(data['dt'][i]))
            loaded += 1
            if limit is not None and loaded >= limit:
                return loaded
    return loaded
