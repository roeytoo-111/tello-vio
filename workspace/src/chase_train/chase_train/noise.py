"""Exploration noise: decayed Gaussian (improved arms) and the faithful OU.

Gaussian (guide 11.2): epsilon ~ N(0, sigma^2 I) in NORMALISED action units,
sigma linearly decayed 0.3 -> 0.05 over the run -- applied BEFORE the clip
(pre-training checklist item 8).

OU (faithful arm): keras-rl semantics reproduced exactly, because "fixing"
the noise makes it a different arm (guide 11.2 verified finding):
    x_{t+1} = x_t + theta*(mu - x_t)*dt + sigma*sqrt(dt)*N(0, I)
with theta=0.15, mu=0, sigma=0.3 and the library's NEVER-SET dt = 1e-2 --
per-step increments of std 0.3*sqrt(0.01) = 0.03 -- and the per-episode
reset drawing x ~ N(mu, sigma). The sample is added AFTER the x60 output
scaling (post-scale, [C]); the trainer owns that placement.
"""
import numpy as np


class GaussianNoise:
    def __init__(self, sigma0: float = 0.3, sigma_min: float = 0.05,
                 decay_steps: int = 200_000, act_dim: int = 2):
        self.sigma0 = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.decay_steps = int(decay_steps)
        self.act_dim = int(act_dim)
        self._step = 0

    def reset_episode(self, rng: np.random.Generator) -> None:
        pass                                  # memoryless between episodes

    def sigma(self) -> float:
        if self.decay_steps <= 0:
            return self.sigma_min
        f = min(self._step / self.decay_steps, 1.0)
        return self.sigma0 + f * (self.sigma_min - self.sigma0)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        eps = rng.normal(0.0, self.sigma(), size=self.act_dim)
        self._step += 1
        return eps.astype(np.float32)

    def state_dict(self) -> dict:
        # The schedule params travel with the state: a sanctioned
        # continuation (raised total_steps) must keep the ORIGINAL decay
        # horizon, not re-derive one from the new total.
        return {'type': 'gaussian', 'step': self._step,
                'sigma0': self.sigma0, 'sigma_min': self.sigma_min,
                'decay_steps': self.decay_steps}

    def load_state_dict(self, d: dict) -> None:
        if d.get('type', 'gaussian') != 'gaussian':
            raise ValueError(
                f"noise state is {d.get('type')!r}, this run builds "
                f"gaussian -- arm/config mismatch on resume")
        self._step = int(d['step'])
        self.sigma0 = float(d.get('sigma0', self.sigma0))
        self.sigma_min = float(d.get('sigma_min', self.sigma_min))
        self.decay_steps = int(d.get('decay_steps', self.decay_steps))


class OUNoise:
    def __init__(self, theta: float = 0.15, mu: float = 0.0,
                 sigma: float = 0.3, dt: float = 1e-2, act_dim: int = 2):
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)                   # EXPLICIT -- checklist item 8
        self.act_dim = int(act_dim)
        self.x = np.zeros(act_dim, dtype=np.float64)

    def reset_episode(self, rng: np.random.Generator) -> None:
        # keras-rl draws the episode's starting state from N(mu, sigma)
        # (rl/random.py, fetched for guide 11.2).
        self.x = rng.normal(self.mu, self.sigma, size=self.act_dim)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        dx = (self.theta * (self.mu - self.x) * self.dt
              + self.sigma * np.sqrt(self.dt)
              * rng.normal(size=self.act_dim))
        self.x = self.x + dx
        return self.x.astype(np.float32)

    def state_dict(self) -> dict:
        return {'type': 'ou', 'x': self.x.tolist()}

    def load_state_dict(self, d: dict) -> None:
        if d.get('type', 'ou') != 'ou':
            raise ValueError(
                f"noise state is {d.get('type')!r}, this run builds ou -- "
                f"arm/config mismatch on resume")
        self.x = np.asarray(d['x'], dtype=np.float64)
