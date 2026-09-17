"""ONE trainer, three arms (guide 13.5): TD3 is the class, DDPG is a
configuration of it -- never a second codebase
(drl_ros2_reference_analysis.md 5).

Arm flags:            T (TD3)         R (repaired DDPG)   F (faithful DDPG)
  n_critics           2               1                   1
  target smoothing    0.2 / 0.5       off                 off
  policy_delay        2               1                   1
  tau                 0.005           0.005               0.001
  lr (actor/critic)   3e-4 / 3e-4     3e-4 / 3e-4         1e-3 / 1e-3, clipnorm 1

Mechanics guaranteed here, each against a named failure:
  * targets initialised as EXACT copies (guide 11.5);
  * soft updates counted in GRADIENT steps and applied only on policy-delay
    steps -- never the reference's 500-at-once burst
    (drl_ros2_reference_analysis.md 2.2);
  * the TD label uses min over target critics with smoothing noise INSIDE
    the label only (guide 13.1-13.2), scaled by action_scale [L6];
  * the actor trains against Q1 only [L6];
  * (1-d) bootstraps through everything except true terminals (guide 11.3).
"""
import copy
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from .networks import Actor, CriticEnsemble


@dataclass
class TD3Config:
    n_critics: int = 2
    policy_delay: int = 2
    target_noise: float = 0.2         # sigma~, normalised units
    noise_clip: float = 0.5           # c
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    clipnorm: float = 0.0             # 0 = off; faithful arm uses 1.0 [C]
    action_scale: float = 1.0         # 60.0 for the faithful arm


class TD3:
    def __init__(self, actor: Actor, critic: CriticEnsemble, cfg: TD3Config,
                 device: str = 'cpu'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        if len(self.critic.nets) != cfg.n_critics:
            raise ValueError('critic ensemble size != cfg.n_critics')
        # Targets: exact copies, gradient-free.
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(),
                                          lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=cfg.critic_lr)
        self.grad_steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        return self.actor(t).squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        obs, act = batch['obs'], batch['act']
        rew, obs2, done = batch['rew'], batch['obs2'], batch['done']

        with torch.no_grad():
            a2 = self.actor_target(obs2)
            if cfg.target_noise > 0.0:
                # Trick 2: noise inside the label only, clipped, in action
                # units (sigma~ and c are normalised; scale like [L6]).
                s = cfg.action_scale
                eps = (torch.randn_like(a2) * (cfg.target_noise * s)
                       ).clamp(-cfg.noise_clip * s, cfg.noise_clip * s)
                a2 = (a2 + eps).clamp(-s, s)
            q_next = self.critic_target.min_q(obs2, a2)
            y = rew + cfg.gamma * (1.0 - done) * q_next

        qs = self.critic(obs, act)
        critic_loss = sum(torch.nn.functional.mse_loss(q, y) for q in qs)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.clipnorm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                           cfg.clipnorm)
        self.critic_opt.step()

        self.grad_steps += 1
        metrics = {
            'critic_loss': float(critic_loss.item()),
            'q1_mean': float(qs[0].mean().item()),
            'q_max': float(qs[0].max().item()),
            'td_abs_mean': float((y - qs[0]).abs().mean().item()),
        }
        if len(qs) == 2:
            metrics['q_gap'] = float((qs[0] - qs[1]).abs().mean().item())

        # Trick 3: delayed policy + target updates, counted in grad steps.
        if self.grad_steps % cfg.policy_delay == 0:
            for p in self.critic.parameters():
                p.requires_grad_(False)
            actor_loss = -self.critic.q1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if cfg.clipnorm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                               cfg.clipnorm)
            self.actor_opt.step()
            for p in self.critic.parameters():
                p.requires_grad_(True)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            metrics['actor_loss'] = float(actor_loss.item())
        return metrics

    def _soft_update(self, src: torch.nn.Module, dst: torch.nn.Module) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, pt in zip(src.parameters(), dst.parameters()):
                pt.mul_(1.0 - tau).add_(p, alpha=tau)

    # -- (de)serialisation for the checkpoint bundle ----------------------
    def state_dict(self) -> dict:
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'grad_steps': self.grad_steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])
        self.actor_target.load_state_dict(d['actor_target'])
        self.critic_target.load_state_dict(d['critic_target'])
        self.actor_opt.load_state_dict(d['actor_opt'])
        self.critic_opt.load_state_dict(d['critic_opt'])
        self.grad_steps = int(d['grad_steps'])
