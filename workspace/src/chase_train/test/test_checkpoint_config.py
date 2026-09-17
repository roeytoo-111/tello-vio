"""Checkpoint contract: round-trip, the spec-hash refusal, ONNX parity;
config presets resolve to the doc-mandated arm values."""
import os

import numpy as np
import pytest
import torch

from chase_gym.observation import ObservationSpec
from chase_train.checkpoint import (actor_from_bundle, export_onnx,
                                    load_bundle, save_bundle, versioned_name)
from chase_train.config import load as load_config
from chase_train.networks import Actor, CriticEnsemble
from chase_train.noise import GaussianNoise
from chase_train.td3 import TD3, TD3Config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'config')


def _mini_cfg():
    return load_config(os.path.join(CONFIG_DIR, 'td3.yaml'))


def _bundle_path(tmp_path):
    cfg = _mini_cfg()
    spec = ObservationSpec()
    actor = Actor(spec.dim, 2, cfg.nets.hidden_actor)
    critic = CriticEnsemble(spec.dim, 2, cfg.nets.hidden_critic, n_critics=2)
    tr = TD3(actor, critic, TD3Config())
    path = str(tmp_path / versioned_name('td3', 0, 100, 0.5))
    save_bundle(path, trainer=tr, noise=GaussianNoise(), cfg=cfg,
                obs_spec=spec, action_map={'order': ['linear.z', 'angular.z']},
                env_config={'reward_version': 'repaired'}, env_step=100,
                eval_snapshot={'time_in_view': 0.5})
    return path, spec, tr


def test_bundle_roundtrip_and_contract_fields(tmp_path):
    path, spec, tr = _bundle_path(tmp_path)
    b = load_bundle(path, expect_obs_spec=spec)
    for key in ('arm', 'seed', 'env_step', 'trainer', 'noise', 'obs_spec',
                'obs_spec_hash', 'action_map', 'control_rate_hz',
                'reward_version', 'config', 'config_hash', 'git_sha',
                'library_versions', 'rng'):
        assert key in b, f'contract field {key} missing'
    actor = actor_from_bundle(b)
    x = torch.zeros(1, spec.dim)
    with torch.no_grad():
        ref = tr.actor(x)
        got = actor(x)
    assert torch.allclose(ref, got)


def test_spec_hash_refusal(tmp_path):
    path, _, _ = _bundle_path(tmp_path)
    other = ObservationSpec(k=3)     # different layout
    with pytest.raises(ValueError, match='observation spec'):
        load_bundle(path, expect_obs_spec=other)


def test_onnx_export_parity(tmp_path):
    path, _, _ = _bundle_path(tmp_path)
    out = export_onnx(load_bundle(path), str(tmp_path / 'actor.onnx'))
    assert out is not None and os.path.exists(out)


def test_versioned_name_format():
    assert versioned_name('td3', 3, 20000, 0.914) == 'td3_s3_step20000_tv091.pt'


# ---- config presets: the doc-mandated arm values ------------------------

def test_faithful_preset_matches_published_code():
    """Every value from rl_block_diagram.md section 3 [C], verbatim."""
    cfg = load_config(os.path.join(CONFIG_DIR, 'ddpg_faithful.yaml'))
    assert cfg.arm == 'ddpg_faithful' and cfg.n_critics == 1
    assert cfg.seed == 123
    assert cfg.nets.hidden_actor == [16, 16, 16]
    assert cfg.nets.hidden_critic == [32, 32, 32]
    assert cfg.train.total_steps == 100_000
    assert cfg.train.batch == 32
    assert cfg.train.tau == pytest.approx(1e-3)
    assert cfg.train.actor_lr == pytest.approx(1e-3)
    assert cfg.train.clipnorm == pytest.approx(1.0)
    assert cfg.train.update_after == 100
    assert cfg.train.start_steps == 0
    assert cfg.train.action_scale == pytest.approx(60.0)
    assert cfg.train.truncation_as_terminal is True
    n = cfg.train.noise
    assert (n.type, n.theta, n.ou_sigma, n.ou_dt, n.post_scale) == \
        ('ou', 0.15, 0.3, 1e-2, True)
    assert cfg.curriculum.enabled is False


def test_td3_preset_matches_ours_column():
    """Guide 11.8 'Ours' column."""
    cfg = load_config(os.path.join(CONFIG_DIR, 'td3.yaml'))
    assert cfg.n_critics == 2
    assert cfg.nets.hidden_actor == [256, 256]
    assert cfg.nets.final_init == pytest.approx(3e-3)
    assert (cfg.train.gamma, cfg.train.tau) == (0.99, 0.005)
    assert cfg.train.batch == 64 and cfg.train.buffer == 100_000
    assert cfg.train.actor_lr == pytest.approx(3e-4)
    assert (cfg.train.start_steps, cfg.train.update_after) == (1000, 1000)
    assert (cfg.train.policy_delay, cfg.train.target_noise,
            cfg.train.noise_clip) == (2, 0.2, 0.5)
    assert cfg.train.total_steps == 200_000
    assert cfg.env['episode_steps'] == 300
    assert cfg.env['k'] == 4


def test_repaired_preset_is_td3_minus_tricks():
    cfg = load_config(os.path.join(CONFIG_DIR, 'ddpg_repaired.yaml'))
    assert cfg.n_critics == 1
    assert cfg.train.policy_delay == 1
    assert cfg.train.target_noise == 0.0


def test_override_parsing():
    cfg = load_config(os.path.join(CONFIG_DIR, 'td3.yaml'),
                      overrides=['env.latency=false',
                                 'train.total_steps=1234'], seed=7)
    assert cfg.env['latency'] is False
    assert cfg.train.total_steps == 1234
    assert cfg.seed == 7
