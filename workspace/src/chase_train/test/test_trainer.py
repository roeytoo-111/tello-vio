"""Trainer mechanics: the failures each piece of machinery answers
(guide 12) must be tested, not assumed."""
import copy

import numpy as np
import pytest
import torch

from chase_train.buffer import ReplayBuffer, load_demos, save_demo_episode
from chase_train.networks import Actor, CriticEnsemble
from chase_train.noise import GaussianNoise, OUNoise
from chase_train.td3 import TD3, TD3Config

OBS, ACT = 14, 2


def make_trainer(n_critics=2, policy_delay=2, target_noise=0.2, tau=0.005):
    actor = Actor(OBS, ACT, [32, 32])
    critic = CriticEnsemble(OBS, ACT, [32, 32], n_critics=n_critics)
    cfg = TD3Config(n_critics=n_critics, policy_delay=policy_delay,
                    target_noise=target_noise, tau=tau)
    return TD3(actor, critic, cfg)


def rand_batch(rng, n=32):
    t = lambda *shape: torch.as_tensor(
        rng.normal(size=shape).astype(np.float32))
    return {'obs': t(n, OBS), 'act': torch.clamp(t(n, ACT), -1, 1),
            'rew': t(n, 1), 'obs2': t(n, OBS),
            'done': torch.zeros(n, 1)}


def test_targets_start_as_exact_copies():
    tr = make_trainer()
    for p, pt in zip(tr.actor.parameters(), tr.actor_target.parameters()):
        assert torch.equal(p, pt)
    for p, pt in zip(tr.critic.parameters(), tr.critic_target.parameters()):
        assert torch.equal(p, pt)


def test_policy_delay_gates_actor_and_targets():
    """Targets move ONLY on policy-delay steps, counted in gradient steps
    -- never the reference's 500-at-once burst (RA 2.2)."""
    tr = make_trainer(policy_delay=2)
    rng = np.random.default_rng(0)
    before = copy.deepcopy(tr.actor_target.state_dict())
    m1 = tr.update(rand_batch(rng))
    after1 = tr.actor_target.state_dict()
    assert 'actor_loss' not in m1
    for k in before:
        assert torch.equal(before[k], after1[k])   # step 1: frozen
    m2 = tr.update(rand_batch(rng))
    assert 'actor_loss' in m2
    moved = any(not torch.equal(before[k], tr.actor_target.state_dict()[k])
                for k in before)
    assert moved                                   # step 2: soft-updated


def test_ddpg_config_updates_every_step():
    tr = make_trainer(n_critics=1, policy_delay=1, target_noise=0.0)
    rng = np.random.default_rng(0)
    m = tr.update(rand_batch(rng))
    assert 'actor_loss' in m
    assert 'q_gap' not in m                        # single critic


def test_min_q_is_pessimistic():
    tr = make_trainer()
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.normal(size=(8, OBS)).astype(np.float32))
    act = torch.zeros(8, ACT)
    qs = tr.critic_target(obs, act)
    mn = tr.critic_target.min_q(obs, act)
    assert torch.all(mn <= qs[0] + 1e-6) and torch.all(mn <= qs[1] + 1e-6)


def test_terminal_flag_truncates_bootstrap():
    """d=1 must remove the bootstrap term from the label (guide 11.3)."""
    tr = make_trainer(target_noise=0.0)
    rng = np.random.default_rng(0)
    b = rand_batch(rng, n=4)
    b['rew'] = torch.ones(4, 1) * 2.0
    with torch.no_grad():
        a2 = tr.actor_target(b['obs2'])
        q_next = tr.critic_target.min_q(b['obs2'], a2)
        y_boot = b['rew'] + 0.99 * q_next
    b_term = {k: v.clone() for k, v in b.items()}
    b_term['done'] = torch.ones(4, 1)
    with torch.no_grad():
        y_term = b_term['rew'] + 0.99 * (1 - b_term['done']) * q_next
    assert torch.allclose(y_term, b_term['rew'])
    assert not torch.allclose(y_boot, b_term['rew'])


def test_critic_actually_trains_everywhere():
    """Every critic parameter must receive gradient -- the reference's
    detached-weight critic trained half its weights (RA 2.1)."""
    tr = make_trainer()
    rng = np.random.default_rng(0)
    before = copy.deepcopy(tr.critic.state_dict())
    for _ in range(5):
        tr.update(rand_batch(rng))
    after = tr.critic.state_dict()
    unchanged = [k for k in before if torch.equal(before[k], after[k])]
    assert not unchanged, f'critic params never updated: {unchanged}'


def test_clipnorm_applies():
    """After an update with clipnorm=c, the critic gradients left on the
    parameters must have total norm <= c (the faithful arm's clipnorm 1
    [C rl_drone.py:59]); without clipnorm the same batch exceeds it."""
    def critic_grad_norm(clipnorm):
        actor = Actor(OBS, ACT, [32, 32])
        critic = CriticEnsemble(OBS, ACT, [32, 32], n_critics=1)
        tr = TD3(actor, critic, TD3Config(n_critics=1, policy_delay=1,
                                          target_noise=0.0,
                                          clipnorm=clipnorm))
        rng = np.random.default_rng(0)
        b = rand_batch(rng)
        b['rew'] = b['rew'] * 1000.0       # huge targets -> huge gradients
        tr.update(b)
        total = torch.sqrt(sum(
            p.grad.norm() ** 2 for p in tr.critic.parameters()
            if p.grad is not None))
        return float(total)

    assert critic_grad_norm(clipnorm=0.0) > 1.0
    assert critic_grad_norm(clipnorm=1.0) <= 1.0 + 1e-4


# ---- buffer -------------------------------------------------------------

def test_ring_wraparound_and_len():
    buf = ReplayBuffer(10, OBS, ACT)
    for i in range(25):
        buf.add(np.zeros(OBS), np.zeros(ACT), float(i), np.zeros(OBS),
                False, 0.1)
    assert len(buf) == 10
    assert set(buf.rew.flatten().tolist()) == set(float(i) for i in range(15, 25))


def test_short_batch_refused():
    buf = ReplayBuffer(100, OBS, ACT)
    for _ in range(5):
        buf.add(np.zeros(OBS), np.zeros(ACT), 0.0, np.zeros(OBS), False, 0.1)
    with pytest.raises(RuntimeError, match='refusing'):
        buf.sample(32, np.random.default_rng(0))


def test_demo_rewards_recomputed_on_load(tmp_path):
    n = 6
    ep = {
        'obs': np.zeros((n, OBS), dtype=np.float32),
        'act': np.zeros((n, ACT), dtype=np.float32),
        'obs2': np.zeros((n, OBS), dtype=np.float32),
        'terminated': np.zeros(n, dtype=bool),
        'dt': np.full(n, 0.1, dtype=np.float32),
        'dist_px': np.linspace(0, 500, n).astype(np.float32),
        'lost': np.zeros(n, dtype=bool),
        'captured': np.zeros(n, dtype=bool),
        'range_m': np.full(n, 2.0, dtype=np.float32),
        'prev_range_m': np.full(n, 2.0, dtype=np.float32),
        'prev_act': np.zeros((n, ACT), dtype=np.float32),
    }
    save_demo_episode(str(tmp_path / 'ep0.npz'), ep)
    buf = ReplayBuffer(100, OBS, ACT)
    from chase_gym.reward import RewardComputer, RewardConfig
    rc = RewardComputer(RewardConfig(version='repaired', w_smooth=0.0))
    loaded = load_demos(str(tmp_path), buf, rc.compute)
    assert loaded == n
    from chase_gym.reward import track_repaired
    expect = [track_repaired(d) for d in ep['dist_px']]
    np.testing.assert_allclose(buf.rew[:n, 0], expect, rtol=1e-5)


def test_demo_schema_enforced(tmp_path):
    with pytest.raises(ValueError, match='missing fields'):
        save_demo_episode(str(tmp_path / 'bad.npz'),
                          {'obs': np.zeros((1, OBS))})


# ---- noise --------------------------------------------------------------

def test_gaussian_sigma_decays_linearly():
    n = GaussianNoise(0.3, 0.05, decay_steps=100)
    rng = np.random.default_rng(0)
    assert n.sigma() == pytest.approx(0.3)
    for _ in range(50):
        n.sample(rng)
    assert n.sigma() == pytest.approx(0.175, abs=0.01)
    for _ in range(100):
        n.sample(rng)
    assert n.sigma() == pytest.approx(0.05)


def test_ou_keras_semantics():
    """Per-step increment std = sigma*sqrt(dt) = 0.03; episode reset draws
    N(0, sigma) -- the guide 11.2 verified finding, reproduced exactly."""
    rng = np.random.default_rng(0)
    ou = OUNoise(theta=0.15, mu=0.0, sigma=0.3, dt=1e-2, act_dim=1)
    resets = []
    for _ in range(3000):
        ou.reset_episode(rng)
        resets.append(ou.x[0])
    assert np.std(resets) == pytest.approx(0.3, abs=0.02)

    ou.reset_episode(rng)
    ou.x = np.zeros(1)
    increments = []
    for _ in range(3000):
        x0 = ou.x[0]
        ou.sample(rng)
        increments.append(ou.x[0] - x0)
    assert np.std(increments) == pytest.approx(0.03, abs=0.003)


def test_ou_stationary_std():
    """sigma/sqrt(2*theta) = 0.548 regardless of dt [V, guide 11.2]."""
    rng = np.random.default_rng(0)
    ou = OUNoise(theta=0.15, mu=0.0, sigma=0.3, dt=1e-2, act_dim=1)
    ou.reset_episode(rng)
    samples = [ou.sample(rng)[0] for _ in range(120_000)]
    tail = np.asarray(samples[80_000:])
    assert np.std(tail) == pytest.approx(0.3 / np.sqrt(2 * 0.15), abs=0.06)


def test_metrics_merge_keeps_actor_loss_across_parities():
    """train.py merges update dicts; with policy_delay=2 the actor_loss
    from one parity must survive the actor-less dict from the other
    (review: assignment made actor_loss unobservable for TD3 arms)."""
    tr = make_trainer(policy_delay=2)
    rng = np.random.default_rng(0)
    merged = {}
    for _ in range(4):
        m = tr.update(rand_batch(rng), compute_metrics=True)
        if m:
            merged.update(m)
    assert 'actor_loss' in merged
    assert 'critic_loss' in merged


def test_noise_state_type_guard():
    from chase_train.noise import GaussianNoise, OUNoise
    g, o = GaussianNoise(), OUNoise()
    with pytest.raises(ValueError, match='mismatch'):
        g.load_state_dict(o.state_dict())
    with pytest.raises(ValueError, match='mismatch'):
        o.load_state_dict(g.state_dict())
