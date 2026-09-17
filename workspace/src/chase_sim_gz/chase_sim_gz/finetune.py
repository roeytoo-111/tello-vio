"""The Tier-B fine-tune arm (sim_training_architecture.md 3.5 item 3):
continue TD3 from a Tier-A checkpoint for ~1e4 LOCKSTEP steps against
rigid-body dynamics, reported separately. A large A->B performance gap
here is a red flag BEFORE any real flight.

    python3 -m chase_sim_gz.finetune --checkpoint runs/<...>/best.pt \
        [--fake] [--steps 10000]

The trainer, buffer, noise and checkpoint machinery are chase_train's --
nothing is reimplemented; only the env underneath changes, which is the
one-contract property doing its job.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from chase_gym.env import EnvConfig

from chase_train.buffer import ReplayBuffer
from chase_train.checkpoint import load_bundle, save_bundle, versioned_name
from chase_train.config import RunConfig, _build
from chase_train.noise import GaussianNoise
from chase_train.train import build_trainer, seed_everything

from .gz_iface import FakeGzBackend, GzTransportBackend
from .gz_chase_env import GzChaseEnv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True,
                    help='Tier-A bundle to continue from')
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--world', default=None)
    ap.add_argument('--steps', type=int, default=10_000)
    ap.add_argument('--sigma', type=float, default=0.05,
                    help='exploration noise for the fine-tune (small: the '
                         'policy is already trained)')
    ap.add_argument('--out-root', default='runs')
    args = ap.parse_args(argv)

    bundle = load_bundle(args.checkpoint)
    cfg = _build(dict(bundle['config']))
    if bundle['obs_spec']['mode'] != 'latency_aware':
        raise SystemExit('fine-tune needs a latency-aware Tier-A bundle')
    rng = seed_everything(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env_cfg = EnvConfig(**cfg.env)
    if args.fake:
        backend = FakeGzBackend()
    else:
        world = args.world or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'worlds', 'chase.sdf')
        backend = GzTransportBackend(world)
    env = GzChaseEnv(backend, env_cfg)

    # The bundle's spec must match what THIS env produces -- the refusal.
    load_bundle(args.checkpoint, expect_obs_spec=env.obs_spec)

    trainer = build_trainer(cfg, env.obs_spec.dim, 2, device)
    trainer.load_state_dict(bundle['trainer'])
    noise = GaussianNoise(args.sigma, args.sigma, decay_steps=0)
    buf = ReplayBuffer(cfg.train.buffer, env.obs_spec.dim, 2, device=device)

    run_dir = os.path.join(
        args.out_root,
        f'finetune_b_{cfg.arm}_s{cfg.seed}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')

    obs, _ = env.reset(seed=cfg.seed)
    ep_returns, ep_ret, tv_steps, steps_total = [], 0.0, 0, 0
    t0 = time.time()
    try:
        for step in range(1, args.steps + 1):
            mu = trainer.act(np.asarray(obs, dtype=np.float32))
            a = np.clip(mu + noise.sample(rng), -1.0, 1.0).astype(np.float32)
            obs2, r, term, trunc, info = env.step(a)
            buf.add(obs, a, r, obs2, term, env_cfg.dt)
            obs = obs2
            ep_ret += r
            tv_steps += int(info.get('in_frame', False))
            steps_total += 1
            if term or trunc:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()
            if len(buf) >= cfg.train.batch:
                trainer.update(buf.sample(cfg.train.batch, rng))
            if step % 1000 == 0:
                rec = {'step': step,
                       'mean_ep_return': float(np.mean(ep_returns[-20:]))
                       if ep_returns else float('nan'),
                       'time_in_view': tv_steps / steps_total,
                       'wall_s': round(time.time() - t0, 1)}
                with open(metrics_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                print(f"[finetune-b] step {step}  ret "
                      f"{rec['mean_ep_return']:.2f}  tv "
                      f"{rec['time_in_view']:.2%}")
    finally:
        env.close()

    out = os.path.join(run_dir, versioned_name(
        cfg.arm + '_ftB', cfg.seed, args.steps,
        tv_steps / max(steps_total, 1)))
    save_bundle(out, trainer=trainer, noise=noise, cfg=cfg,
                obs_spec=env.obs_spec, action_map=bundle['action_map'],
                env_config=dict(bundle['env_config'], tier='B'),
                env_step=bundle['env_step'] + args.steps,
                eval_snapshot={'time_in_view': tv_steps / max(steps_total, 1)})
    print(f'[done] fine-tuned bundle: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
