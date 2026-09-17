"""SB3 TD3 on the identical chase_gym env -- the external-implementation
check (offline_training_recipe.md 6.4): if the hand-rolled TD3 and SB3's
agree within seed variance on the frozen suite, implementation error is
largely excluded. Report both; tune neither on the other.

    python3 -m chase_train.sb3_check --config config/td3.yaml --seed 0 \
        --override train.total_steps=30000

SB3 2.9.0 facts relied on (verified in the recipe): DDPG is TD3 with the
tricks disabled; TD3 defaults differ from ours, so every shared knob is set
explicitly below. SB3 has no decaying NormalActionNoise: sigma is fixed at
the midpoint of our schedule -- a documented config difference, not a bug.
"""
import argparse
import json
import os
import time

import numpy as np

from chase_gym import ChaseEnv, EnvConfig
from chase_eval.evaluate import default_env_factory, run_suite

from .config import load as load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--override', nargs='*', default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override, seed=args.seed)
    if cfg.arm == 'ddpg_faithful':
        raise SystemExit('the SB3 cross-check runs on the ChaseEnv arms only')

    try:
        from stable_baselines3 import TD3 as SB3TD3
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as e:
        raise SystemExit(f'stable-baselines3 not installed: {e}')

    env = ChaseEnv(EnvConfig(**cfg.env))
    n = cfg.train.noise
    sigma_mid = 0.5 * (n.sigma0 + n.sigma_min)
    model = SB3TD3(
        'MlpPolicy', env, seed=cfg.seed,
        learning_rate=cfg.train.critic_lr,
        buffer_size=cfg.train.buffer, batch_size=cfg.train.batch,
        gamma=cfg.train.gamma, tau=cfg.train.tau,
        policy_delay=cfg.train.policy_delay,
        target_policy_noise=cfg.train.target_noise,
        target_noise_clip=cfg.train.noise_clip,
        learning_starts=cfg.train.update_after,
        train_freq=1, gradient_steps=1,
        action_noise=NormalActionNoise(np.zeros(2), sigma_mid * np.ones(2)),
        policy_kwargs={'net_arch': list(cfg.nets.hidden_actor)},
        verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=cfg.train.total_steps, progress_bar=False)
    wall = time.time() - t0

    suite = run_suite(
        lambda o: model.predict(o, deterministic=True)[0],
        default_env_factory(cfg.env),
        episodes_per_family=cfg.eval.episodes_per_scenario,
        eval_seed0=cfg.eval.eval_seed0)
    out_dir = os.path.join(cfg.out_root,
                           f'sb3_td3_s{cfg.seed}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'sb3_result.json'), 'w') as f:
        json.dump({'suite': suite, 'wall_s': wall,
                   'config': cfg.to_dict(), 'sb3_sigma': sigma_mid,
                   'config_differences': [
                       f'constant NormalActionNoise sigma={sigma_mid} '
                       f'(SB3 has no decay schedule; ours decays '
                       f'{n.sigma0}->{n.sigma_min})',
                       'single learning_rate for actor+critic (SB3 API)',
                       'no demonstration prefill', 'no curriculum',
                       'SB3 default final-layer init (not +/-3e-3)',
                   ]}, f, indent=2)
    agg = suite['aggregate']
    print(f"[sb3-check] steps {cfg.train.total_steps}  "
          f"ret {agg['return_mean']:.2f}  tv {agg['time_in_view']:.2%}  "
          f"({wall:.0f}s)  -> {out_dir}")


if __name__ == '__main__':
    main()
