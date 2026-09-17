"""The frozen evaluation suite (guide 19/24; recipe stage 3).

Rules: exploration OFF, scenario families and seeds FROZEN and disjoint from
anything tuned on, >= episodes_per_scenario per family, distributions
reported -- never best runs. Metrics are the eval trio adapted from the
reference workflow (offline_training_recipe.md section 5): mean return,
time-in-view % (the flight metric, truth-based), loss rate -- plus capture
rate / time-to-capture for INTERCEPT and the action histogram the guide's
dashboard reads (guide 23).

`policy` is any get_action(obs)->action callable in the DEPLOYMENT
convention (normalised [-1,1] actions): a trained actor, the P-controller,
or PN. The suite holds no privileged state; it reads env info only for
scoring.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from chase_gym import ChaseEnv, EnvConfig
from chase_gym.target_motion import FAMILIES

# Derived, never restated: the acceptance population of guide 28.4 (beat
# the P-controller on the MOVING families; match it on static).
MOVING_FAMILIES = tuple(f for f in FAMILIES if f != 'static')


def selection_tv(suite: dict) -> float:
    """The one selection metric train.py and report_matrix share: mean
    time-in-view over the moving families (what 28.4 acceptance judges),
    falling back to the aggregate when the suite has no families (the
    faithful arm's point-mass world)."""
    fams = suite.get('families') or {}
    vals = [fams[f]['time_in_view'] for f in MOVING_FAMILIES if f in fams]
    return float(np.mean(vals)) if vals else \
        float(suite['aggregate']['time_in_view'])


class ActorPolicy:
    """The one adapter that makes a trained actor look like a controller:
    get_action(obs) + reset(). Everything the suite runs -- checkpoints,
    P-controller, PN -- shares this surface, so no call site branches on
    policy shape (and none can forget the per-episode reset)."""

    def __init__(self, actor):
        import torch
        self._torch = torch
        self._actor = actor.eval()

    def reset(self) -> None:
        pass                       # a deterministic actor holds no state

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            t = self._torch.as_tensor(obs, dtype=self._torch.float32
                                      ).unsqueeze(0)
            return self._actor(t).squeeze(0).cpu().numpy()


@dataclass
class FamilyResult:
    family: str
    returns: List[float] = field(default_factory=list)
    time_in_view: List[float] = field(default_factory=list)
    losses: int = 0
    captures: int = 0
    steps_to_capture: List[int] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    mean_dist_px: List[float] = field(default_factory=list)

    def summary(self) -> dict:
        n = max(len(self.returns), 1)
        return {
            'family': self.family,
            # Per-episode scores travel with the summary: bootstrap CIs
            # need episode replication, not just family means.
            'tv_episodes': [float(x) for x in self.time_in_view],
            'return_episodes': [float(x) for x in self.returns],
            'episodes': len(self.returns),
            'return_mean': float(np.mean(self.returns)) if self.returns else float('nan'),
            'return_std': float(np.std(self.returns)) if self.returns else float('nan'),
            'time_in_view': float(np.mean(self.time_in_view)) if self.time_in_view else float('nan'),
            'loss_rate': self.losses / n,
            'capture_rate': self.captures / n,
            'mean_steps_to_capture': float(np.mean(self.steps_to_capture)) if self.steps_to_capture else float('nan'),
            'mean_episode_len': float(np.mean(self.episode_lengths)) if self.episode_lengths else float('nan'),
            'mean_dist_px': float(np.nanmean(self.mean_dist_px)) if self.mean_dist_px else float('nan'),
        }


def run_episode(env, policy, seed: int,
                action_bins: Optional[np.ndarray] = None,
                action_hist: Optional[np.ndarray] = None) -> dict:
    obs, info = env.reset(seed=seed)
    # One dispatch, hoisted; a bare callable (e.g. a lambda over
    # trainer.act) is wrapped on the fly.
    get = policy.get_action if hasattr(policy, 'get_action') else policy
    if hasattr(policy, 'reset'):
        policy.reset()
    total_r = 0.0
    in_view_steps = 0
    dists: List[float] = []
    steps = 0
    lost = False
    captured = False
    while True:
        # reshape(-1): a batched (1,2) action would otherwise broadcast
        # both components into histogram row 0.
        a = np.asarray(get(obs), dtype=np.float32).reshape(-1)
        if action_hist is not None:
            n_dims = min(len(a), action_hist.shape[0])
            b = np.clip(np.digitize(a[:n_dims], action_bins) - 1, 0,
                        action_hist.shape[1] - 1)
            action_hist[np.arange(n_dims), b] += 1
        obs, r, terminated, truncated, info = env.step(a)
        total_r += r
        steps += 1
        if info.get('in_frame', False):
            in_view_steps += 1
        d = info.get('dist_px', float('nan'))
        if np.isfinite(d):
            dists.append(d)
        if terminated:
            captured = bool(info.get('capture', False))
            lost = not captured
        if terminated or truncated:
            break
    return {
        'return': total_r,
        'time_in_view': in_view_steps / max(steps, 1),
        'lost': lost, 'captured': captured, 'steps': steps,
        'mean_dist_px': float(np.nanmean(dists)) if dists else float('nan'),
    }


def run_suite(policy, env_factory: Callable[[str], 'ChaseEnv'],
              families: Sequence[str] = FAMILIES,
              episodes_per_family: int = 10, eval_seed0: int = 10_000,
              n_action_bins: int = 21) -> dict:
    """Frozen suite: episode e of family f always uses the same seed."""
    bins = np.linspace(-1.0, 1.0, n_action_bins + 1)
    hist = np.zeros((2, n_action_bins), dtype=np.int64)
    results = {}
    for fi, fam in enumerate(families):
        fr = FamilyResult(fam)
        env = env_factory(fam)
        for e in range(episodes_per_family):
            ep = run_episode(env, policy, seed=eval_seed0 + 1000 * fi + e,
                             action_bins=bins, action_hist=hist)
            fr.returns.append(ep['return'])
            fr.time_in_view.append(ep['time_in_view'])
            fr.losses += int(ep['lost'])
            fr.captures += int(ep['captured'])
            if ep['captured']:
                fr.steps_to_capture.append(ep['steps'])
            fr.episode_lengths.append(ep['steps'])
            fr.mean_dist_px.append(ep['mean_dist_px'])
        results[fam] = fr

    summaries = {f: r.summary() for f, r in results.items()}
    all_returns = [x for r in results.values() for x in r.returns]
    all_tv = [x for r in results.values() for x in r.time_in_view]
    n_eps = max(len(all_returns), 1)
    total = hist.sum(axis=1, keepdims=True).clip(min=1)
    aggregate = {
        'return_mean': float(np.mean(all_returns)),
        'time_in_view': float(np.mean(all_tv)),
        'loss_rate': sum(r.losses for r in results.values()) / n_eps,
        'capture_rate': sum(r.captures for r in results.values()) / n_eps,
        'action_hist': (hist / total).tolist(),
        'action_sat_frac': float(
            (hist[:, 0].sum() + hist[:, -1].sum()) / hist.sum())
        if hist.sum() else 0.0,
    }
    return {'families': summaries, 'aggregate': aggregate}


def default_env_factory(base_env_kwargs: dict) -> Callable[[str], ChaseEnv]:
    """Eval envs mirror the training env config except the scenario, which
    is pinned per family; speed cap stays the config's full value so eval
    difficulty is constant across the curriculum (frozen suite)."""
    def factory(family: str) -> ChaseEnv:
        kwargs = dict(base_env_kwargs)
        kwargs['scenario'] = family
        return ChaseEnv(EnvConfig(**kwargs))
    return factory
