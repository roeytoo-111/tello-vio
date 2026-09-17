"""Regression tests for the cross-tier contracts the review broke open.

The draw-order divergence (review finding 3) was invisible to the A<->B
gate because the gate runs static + latency-off; these tests pin the
contract in exactly the configurations that exposed it.
"""
import numpy as np
import pytest

from chase_gym import ChaseEnv, EnvConfig
from chase_gym.corruption import CorruptionConfig

from chase_sim_gz.gz_iface import FakeGzBackend
from chase_sim_gz.gz_chase_env import GzChaseEnv


def cfg(**kw):
    base = dict(scenario='constant_velocity', latency=True,
                corruption=False, target_speed_cap=0.5)
    base.update(kw)
    return EnvConfig(**base)


def test_same_seed_same_episode_draws_across_tiers():
    """Latency ON + a stochastic family: the same seed must give the same
    latency base, family, and initial geometry in Tier A and Tier B --
    the configuration where the jitter-draw divergence hid."""
    env_a = ChaseEnv(cfg())
    env_b = GzChaseEnv(FakeGzBackend(), cfg())
    for seed in range(5):
        _, ia = env_a.reset(seed=seed)
        _, ib = env_b.reset(seed=seed)
        assert ia['latency_base_s'] == pytest.approx(ib['latency_base_s']), \
            f'latency base diverged at seed {seed}'
        assert ia['family'] == ib['family']
        assert ia['range_m'] == pytest.approx(ib['range_m'], abs=0.05)


def test_mix_scenario_same_family_sequence():
    env_a = ChaseEnv(cfg(scenario='mix'))
    env_b = GzChaseEnv(FakeGzBackend(), cfg(scenario='mix'))
    fams_a = [env_a.reset(seed=s)[1]['family'] for s in range(8)]
    fams_b = [env_b.reset(seed=s)[1]['family'] for s in range(8)]
    assert fams_a == fams_b


def test_corruption_config_never_mutated():
    """One fitted CorruptionConfig shared across envs with different
    enable gates must come through untouched (review finding 6)."""
    shared = CorruptionConfig(sigma_px=3.5)
    assert shared.enabled
    ChaseEnv(cfg(corruption=False, corruption_cfg=shared))
    assert shared.enabled, 'env constructor mutated the caller config'
    env2 = ChaseEnv(cfg(corruption=True, corruption_cfg=shared))
    assert env2._pipe.corruption.cfg.enabled
    assert env2._pipe.corruption.cfg.sigma_px == 3.5


def test_corruption_config_accepts_yaml_dict():
    """The dot-override / YAML path delivers a plain mapping."""
    env = ChaseEnv(cfg(corruption=True,
                       corruption_cfg={'sigma_px': 4.0,
                                       'dropout_p_small': 0.5}))
    assert env._pipe.corruption.cfg.sigma_px == 4.0
    with pytest.raises(ValueError, match='unknown corruption keys'):
        ChaseEnv(cfg(corruption_cfg={'sigma_typo': 1.0}))
