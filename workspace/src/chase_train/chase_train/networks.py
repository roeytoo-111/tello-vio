"""Actor and critic networks -- the improved pair and the faithful pair.

Improved (guide 18): 256x256 ReLU MLPs, final layers uniform +/-3e-3 [L3]
so the initial policy is near-zero-action and initial values near zero
(guide 11.6); actions enter the critic at the input concat; NO
normalisation layers. ~90k parameters.

Faithful [C rl_drone.py:25-49]: actor Flatten -> 3x Dense(16) ReLU ->
Dense(2) tanh -> x60; critic concat(action, obs) -> 3x Dense(32) ReLU ->
Dense(1). Keras initialisers reproduced: Glorot-uniform hidden layers,
RandomUniform(+/-0.05) on the actor's output layer (the keras default the
original relied on -- guide 11.8 'final-layer init' row).

The critic container holds 1 or 2 critics: TD3 is DDPG with the twin
enabled, one codebase (guide 13.5).
"""
from typing import List

import torch
import torch.nn as nn


def _final_init(layer: nn.Linear, bound: float) -> None:
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


def _glorot(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    """mu(s) in [-1,1]^act_dim, scaled by action_scale at the output.

    action_scale is 1.0 for the deployment convention and 60.0 for the
    faithful arm [C]; it is part of the checkpoint's action map, never
    assumed (offline_training_recipe.md 6.4, the CleanRL trap).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int],
                 action_scale: float = 1.0, final_init: float = 3e-3,
                 faithful_init: bool = False):
        super().__init__()
        dims = [obs_dim] + list(hidden)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            if faithful_init:
                _glorot(lin)
            layers += [lin, nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], act_dim)
        if faithful_init:
            _final_init(self.head, 0.05)     # keras RandomUniform default
        else:
            _final_init(self.head, final_init)
        self.action_scale = float(action_scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.body(obs))) * self.action_scale


class _QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int],
                 final_init: float, faithful_init: bool):
        super().__init__()
        dims = [obs_dim + act_dim] + list(hidden)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            if faithful_init:
                _glorot(lin)
            layers += [lin, nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)
        if faithful_init:
            _glorot(self.head)
        else:
            _final_init(self.head, final_init)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # Standard nn.Sequential MLP with the action at the input concat --
        # NEVER the reference's detached-weight construction
        # (drl_ros2_reference_analysis.md 2.1: half its critic never trained).
        return self.head(self.body(torch.cat([obs, act], dim=-1)))


class CriticEnsemble(nn.Module):
    """1 critic = DDPG, 2 = TD3 clipped double-Q (guide 13.1)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int],
                 n_critics: int = 2, final_init: float = 3e-3,
                 faithful_init: bool = False):
        super().__init__()
        if n_critics not in (1, 2):
            raise ValueError(f'n_critics must be 1 or 2, got {n_critics}')
        self.nets = nn.ModuleList(
            _QNet(obs_dim, act_dim, hidden, final_init, faithful_init)
            for _ in range(n_critics))

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> List[torch.Tensor]:
        return [net(obs, act) for net in self.nets]

    def q1(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """The actor trains against Q1 only [L6] (guide 13.1)."""
        return self.nets[0](obs, act)

    def min_q(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        qs = self.forward(obs, act)
        if len(qs) == 1:
            return qs[0]
        return torch.min(qs[0], qs[1])
