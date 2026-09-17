"""The training loop (guide 11.7 / 13.4), config-driven, seeds by CLI.

    python3 -m chase_train.train --config config/td3.yaml --seed 0
    python3 -m chase_train.train --config config/td3.yaml \
        --override env.scenario=static env.latency=false train.total_steps=8000

Every run writes to runs/{arm}_s{seed}_{stamp}/: the resolved config, a
metrics.jsonl (one line per eval: the guide 23 dashboard signals), versioned
checkpoints, and best.json. Stopping: budget exhausted or eval plateau after
the curriculum completes (recipe stage 3).
"""
import argparse
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import gymnasium

from chase_gym import ChaseEnv, EnvConfig, FaithfulPointMassEnv, PController
from chase_eval.evaluate import default_env_factory, run_suite

from .buffer import ReplayBuffer, load_demos
from .checkpoint import (export_onnx, load_bundle, save_bundle,
                         update_best_manifest, versioned_name)
from .config import RunConfig, load as load_config
from .networks import Actor, CriticEnsemble
from .noise import GaussianNoise, OUNoise
from .td3 import TD3, TD3Config

FAITHFUL_EPISODE_CAP = 10          # nb_max_episode_steps=10 [C train-rl-agent.py:7]


def seed_everything(seed: int) -> np.random.Generator:
    """python/numpy/torch, per run -- the reference seeded none of them
    (offline_training_recipe.md section 5)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def build_env(cfg: RunConfig):
    if cfg.arm == 'ddpg_faithful':
        env = gymnasium.wrappers.TimeLimit(
            FaithfulPointMassEnv(), max_episode_steps=FAITHFUL_EPISODE_CAP)
        return env
    return ChaseEnv(EnvConfig(**cfg.env))


def build_trainer(cfg: RunConfig, obs_dim: int, act_dim: int, device: str) -> TD3:
    actor = Actor(obs_dim, act_dim, cfg.nets.hidden_actor,
                  action_scale=cfg.train.action_scale,
                  final_init=cfg.nets.final_init,
                  faithful_init=cfg.nets.faithful_init)
    critic = CriticEnsemble(obs_dim, act_dim, cfg.nets.hidden_critic,
                            n_critics=cfg.n_critics,
                            final_init=cfg.nets.final_init,
                            faithful_init=cfg.nets.faithful_init)
    tcfg = TD3Config(
        n_critics=cfg.n_critics, policy_delay=cfg.train.policy_delay,
        target_noise=cfg.train.target_noise, noise_clip=cfg.train.noise_clip,
        gamma=cfg.train.gamma, tau=cfg.train.tau,
        actor_lr=cfg.train.actor_lr, critic_lr=cfg.train.critic_lr,
        clipnorm=cfg.train.clipnorm, action_scale=cfg.train.action_scale)
    return TD3(actor, critic, tcfg, device=device)


def build_noise(cfg: RunConfig):
    n = cfg.train.noise
    if n.type == 'gaussian':
        decay = n.decay_steps or cfg.train.total_steps
        return GaussianNoise(n.sigma0, n.sigma_min, decay, act_dim=2)
    if n.type == 'ou':
        return OUNoise(n.theta, 0.0, n.ou_sigma, n.ou_dt, act_dim=2)
    raise ValueError(f'unknown noise type {n.type!r}')


def action_map(cfg: RunConfig) -> dict:
    if cfg.arm == 'ddpg_faithful':
        return {'order': ['image_x', 'image_y'],
                'scale': cfg.train.action_scale,
                'note': "original convention [C]: a[0] horizontal, a[1] "
                        "vertical, position += 2*a; deployment mapped "
                        "yaw = -a[0], up = +a[1] [C main.py:225-227]"}
    return {'order': ['linear.z', 'angular.z'],
            'scale': 1.0, 'stick_range': [-1.0, 1.0],
            'signs': 'REP-103; +angular.z (CCW) moves box +u (right), '
                     '+linear.z (up) moves box +v (down) -- unit-tested in '
                     'chase_gym/test/test_kinematics.py'}


def prefill_demos(cfg: RunConfig, buf: ReplayBuffer, obs_spec) -> int:
    """P-controller episodes straight into the buffer (guide 17). Rewards
    come from the live reward function inside the env, which is exactly the
    recompute-on-load guarantee for in-process demos; .npz demo dirs (real
    flights) additionally go through load_demos()."""
    n_eps = cfg.train.demo_prefill_episodes
    added = 0
    if n_eps > 0:
        env = ChaseEnv(EnvConfig(**cfg.env))
        ctl = PController(env.obs_spec)
        for e in range(n_eps):
            obs, _ = env.reset(seed=1_000_000 + 997 * cfg.seed + e)
            ctl.reset()
            while True:
                a = ctl.get_action(obs)
                obs2, r, term, trunc, _ = env.step(a)
                buf.add(obs, a, r, obs2, term, env.cfg.dt)
                added += 1
                obs = obs2
                if term or trunc:
                    break
    if cfg.train.demo_dir:
        from chase_gym.reward import RewardComputer, RewardConfig
        rc = RewardComputer(RewardConfig(
            version=cfg.env.get('reward_version', 'repaired'),
            intercept=(cfg.env.get('task') == 'intercept'),
            gamma=cfg.train.gamma))
        added += load_demos(cfg.train.demo_dir, buf, rc.compute)
    return added


class Curriculum:
    def __init__(self, cfg: RunConfig, env):
        self.cfg = cfg
        self.env = env
        self.stage = 0
        self.enabled = (cfg.curriculum.enabled
                        and cfg.arm != 'ddpg_faithful')
        if self.enabled:
            self._apply()

    def _apply(self):
        s = self.cfg.curriculum.stages[self.stage]
        self.env.set_stage(s.families, s.speed_cap)

    @property
    def at_final(self) -> bool:
        return (not self.enabled
                or self.stage == len(self.cfg.curriculum.stages) - 1)

    def update(self, family_summaries: dict) -> None:
        """Advance/demote on the CURRENT stage families' eval time-in-view
        (rolling gate that can step back down, [D-impl 13])."""
        if not self.enabled:
            return
        s = self.cfg.curriculum.stages[self.stage]
        tvs = [family_summaries[f]['time_in_view'] for f in s.families
               if f in family_summaries]
        if not tvs:
            return
        tv = float(np.mean(tvs))
        if tv >= s.advance_tv and not self.at_final:
            self.stage += 1
            self._apply()
        elif tv <= s.demote_tv and self.stage > 0:
            self.stage -= 1
            self._apply()


def eval_faithful(trainer: TD3, episodes: int = 20,
                  eval_seed0: int = 10_000) -> dict:
    """Arm F is scored in its own world: return and survival, exploration
    off (its 'time in view' is the fraction of the 10-step budget survived
    before an edge exit)."""
    env = gymnasium.wrappers.TimeLimit(FaithfulPointMassEnv(),
                                       max_episode_steps=FAITHFUL_EPISODE_CAP)
    rets, lens = [], []
    for e in range(episodes):
        obs, _ = env.reset(seed=eval_seed0 + e)
        total, steps = 0.0, 0
        while True:
            a = trainer.act(np.asarray(obs, dtype=np.float32))
            obs, r, term, trunc, _ = env.step(a)
            total += r
            steps += 1
            if term or trunc:
                break
        rets.append(total)
        lens.append(steps)
    tv = float(np.mean(lens)) / FAITHFUL_EPISODE_CAP
    return {'families': {}, 'aggregate': {
        'return_mean': float(np.mean(rets)),
        'time_in_view': tv, 'loss_rate': float('nan'),
        'capture_rate': 0.0, 'mean_episode_len': float(np.mean(lens))}}


def train(cfg: RunConfig, resume: Optional[str] = None) -> str:
    rng = seed_everything(cfg.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if cfg.device == 'auto' else cfg.device

    env = build_env(cfg)
    faithful = cfg.arm == 'ddpg_faithful'
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    trainer = build_trainer(cfg, obs_dim, act_dim, device)
    noise = build_noise(cfg)
    buf = ReplayBuffer(cfg.train.buffer, obs_dim, act_dim, device=device)

    run_dir = os.path.join(
        cfg.out_root, f'{cfg.arm}_s{cfg.seed}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'config': cfg.to_dict(), 'hash': cfg.config_hash(),
                   'device': device}, f, indent=2)
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')

    obs_spec = env.obs_spec if not faithful else _FaithfulSpec()
    env_config = env.env_config_dict() if not faithful else {
        'env': 'FaithfulPointMassEnv', 'episode_cap': FAITHFUL_EPISODE_CAP,
        'reward_version': 'faithful'}

    start_step = 0
    if resume:
        bundle = load_bundle(resume)
        trainer.load_state_dict(bundle['trainer'])
        noise.load_state_dict(bundle['noise'])
        start_step = bundle['env_step']

    if not faithful and start_step == 0:
        n_demo = prefill_demos(cfg, buf, obs_spec)
        if n_demo:
            print(f'[demo] prefilled {n_demo} transitions from the '
                  f'P-controller / {cfg.train.demo_dir or "no npz dir"}')

    curriculum = Curriculum(cfg, env) if not faithful else None
    eval_env_factory = None
    if not faithful:
        eval_env_factory = default_env_factory(cfg.env)

    obs, _ = env.reset(seed=cfg.seed)
    noise.reset_episode(rng)
    ep_ret, ep_len = 0.0, 0
    recent_update_metrics: dict = {}
    best = {'tv': -1.0, 'ret': -np.inf}
    evals_since_best = 0
    t0 = time.time()

    for step in range(start_step + 1, cfg.train.total_steps + 1):
        # ---- act (guide 11.1-11.2) ----
        if step <= cfg.train.start_steps:
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32) \
                * cfg.train.action_scale
        else:
            mu = trainer.act(np.asarray(obs, dtype=np.float32))
            eps = noise.sample(rng)
            if cfg.train.noise.post_scale:
                a = (mu + eps).astype(np.float32)       # faithful [C]: no clip
            else:
                a = np.clip(mu + eps, -1.0, 1.0).astype(np.float32)

        obs2, r, terminated, truncated, _info = env.step(a)
        d_stored = terminated or (truncated and cfg.train.truncation_as_terminal)
        buf.add(obs, a, r, obs2, d_stored, getattr(env, 'cfg', None).dt
                if hasattr(env, 'cfg') and hasattr(env.cfg, 'dt') else 0.1)
        obs = obs2
        ep_ret += r
        ep_len += 1

        if terminated or truncated:
            obs, _ = env.reset()
            noise.reset_episode(rng)
            ep_ret, ep_len = 0.0, 0

        # ---- learn: one gradient step per env step (guide 19) ----
        if step >= cfg.train.update_after and len(buf) >= cfg.train.batch:
            batch = buf.sample(cfg.train.batch, rng)
            recent_update_metrics = trainer.update(batch)

        # ---- eval / checkpoint / curriculum / stopping ----
        if step % cfg.eval.every == 0 or step == cfg.train.total_steps:
            if faithful:
                suite = eval_faithful(trainer,
                                      episodes=cfg.eval.episodes_per_scenario,
                                      eval_seed0=cfg.eval.eval_seed0)
            else:
                suite = run_suite(
                    lambda o: trainer.act(np.asarray(o, dtype=np.float32)),
                    eval_env_factory,
                    episodes_per_family=cfg.eval.episodes_per_scenario,
                    eval_seed0=cfg.eval.eval_seed0)
                curriculum.update(suite['families'])

            agg = suite['aggregate']
            # avg_Q vs realised return -- the overestimation dashboard row.
            avg_q = float('nan')
            if len(buf) >= cfg.train.batch:
                b = buf.sample(cfg.train.batch, rng)
                with torch.no_grad():
                    avg_q = float(trainer.critic.q1(
                        b['obs'], trainer.actor(b['obs'])).mean().item())
            record = {
                'env_step': step,
                'wall_s': round(time.time() - t0, 1),
                'stage': curriculum.stage if curriculum else -1,
                'eval': suite,
                'avg_q': avg_q,
                'sigma': (noise.sigma() if callable(getattr(noise, 'sigma', None))
                          else getattr(noise, 'sigma', None)),
                'update': recent_update_metrics,
                'buffer_fill': len(buf),
            }
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
            print(f"[{cfg.arm} s{cfg.seed}] step {step}  "
                  f"ret {agg['return_mean']:.2f}  tv {agg['time_in_view']:.2%}  "
                  f"loss {agg['loss_rate'] if agg['loss_rate']==agg['loss_rate'] else float('nan'):.2f}  "
                  f"avgQ {avg_q:.2f}  stage {record['stage']}")

            improved = (agg['time_in_view'] > best['tv'] + cfg.eval.plateau_epsilon
                        or (abs(agg['time_in_view'] - best['tv']) <= cfg.eval.plateau_epsilon
                            and agg['return_mean'] > best['ret']))
            if improved:
                best = {'tv': agg['time_in_view'], 'ret': agg['return_mean']}
                evals_since_best = 0
                ckpt = os.path.join(run_dir, versioned_name(
                    cfg.arm, cfg.seed, step, agg['time_in_view']))
                save_bundle(
                    ckpt, trainer=trainer, noise=noise, cfg=cfg,
                    obs_spec=obs_spec, action_map=action_map(cfg),
                    env_config=env_config, env_step=step, eval_snapshot=agg,
                    curriculum_stage=record['stage'])
                update_best_manifest(run_dir, {
                    'arm': cfg.arm, 'seed': cfg.seed, 'path': ckpt,
                    'env_step': step, 'time_in_view': agg['time_in_view'],
                    'return_mean': agg['return_mean'],
                    'reason': 'best eval time-in-view'})
            else:
                evals_since_best += 1

            done_curriculum = curriculum.at_final if curriculum else True
            if done_curriculum and evals_since_best >= cfg.eval.plateau_evals:
                print(f'[stop] eval plateau: {evals_since_best} evals '
                      f'without improvement after curriculum completion')
                break

    # Final bundle + ONNX for the best checkpoint.
    final = os.path.join(run_dir, f'{cfg.arm}_s{cfg.seed}_final.pt')
    save_bundle(final, trainer=trainer, noise=noise, cfg=cfg,
                obs_spec=obs_spec, action_map=action_map(cfg),
                env_config=env_config, env_step=step,
                eval_snapshot=best,
                curriculum_stage=curriculum.stage if curriculum else -1)
    best_path = os.path.join(run_dir, 'best.json')
    if os.path.exists(best_path):
        with open(best_path) as f:
            best_entry = json.load(f)
        onnx_path = export_onnx(load_bundle(best_entry['path']),
                                best_entry['path'].replace('.pt', '.onnx'))
        if onnx_path:
            print(f'[export] ONNX actor: {onnx_path} (parity checked)')
    print(f'[done] run dir: {run_dir}')
    return run_dir


class _FaithfulSpec:
    """Minimal obs-spec stand-in for the faithful arm: raw pixels, no
    assembler. Carries the same contract surface the bundle needs."""

    mode = 'raw_pixels'
    control_dt_s = 0.1

    def to_dict(self) -> dict:
        return {'mode': 'raw_pixels', 'dim': 2, 'action_dim': 2,
                'note': 'original [C]: raw box-centre pixels, unnormalised'}

    def spec_hash(self) -> str:
        import hashlib, json as _json
        return hashlib.sha256(_json.dumps(
            self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--override', nargs='*', default=[],
                    help='dot-path overrides, e.g. env.latency=false')
    ap.add_argument('--resume', default=None,
                    help='checkpoint bundle to resume from')
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override, seed=args.seed)
    train(cfg, resume=args.resume)


if __name__ == '__main__':
    main()
