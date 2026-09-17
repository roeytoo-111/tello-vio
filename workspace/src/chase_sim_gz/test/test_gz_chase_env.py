"""Tier B logic against the kinematic fake backend: every line of
GzChaseEnv (lockstep bookkeeping, oracle projection, latency on sim time,
shared reward/observation) runs without a gz install. The REAL backend is
exercised post-install by sign_test.py / calibrate_lag.py.
"""
import numpy as np
import pytest

from chase_gym import PController
from chase_gym.env import EnvConfig

from chase_sim_gz.gz_iface import FakeGzBackend
from chase_sim_gz.gz_chase_env import GzChaseEnv
from chase_sim_gz.sign_test import main as sign_test_main
from chase_sim_gz.calibrate_lag import fit_t_lag


def clean_env(**kw):
    base = dict(scenario='static', latency=False, corruption=False)
    base.update(kw)
    return GzChaseEnv(FakeGzBackend(), EnvConfig(**base))


def test_control_period_contract_enforced():
    with pytest.raises(ValueError, match='control period'):
        GzChaseEnv(FakeGzBackend(), EnvConfig(), steps_per_control=50)


def test_lockstep_advances_exactly_one_control_period():
    env = clean_env()
    env.reset(seed=0)
    t0 = env.backend.sim_time()
    env.step(np.zeros(2))
    assert env.backend.sim_time() - t0 == pytest.approx(0.1, abs=1e-9)


def test_reset_places_target_in_frame():
    env = clean_env()
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        assert info['in_frame']
        r_lo, r_hi = env.cfg.resolved_reset_range()
        assert r_lo * 0.8 <= info['range_m'] <= r_hi * 1.3


def test_p_controller_tracks_in_tier_b():
    """The same deployment-convention controller that works in Tier A must
    work here -- the one-contract property, live."""
    env = clean_env(episode_steps=120)
    obs, info = env.reset(seed=1)
    ctl = PController(env.obs_spec)
    dists = []
    for _ in range(120):
        obs, r, term, trunc, info = env.step(ctl.get_action(obs))
        if np.isfinite(info['dist_px']):
            dists.append(info['dist_px'])
        if term or trunc:
            break
    assert not term or info.get('capture')     # never lost the target
    assert np.mean(dists[-20:]) < 60.0         # centred by the end


def test_same_seed_same_trajectory():
    ea, eb = clean_env(scenario='constant_velocity'), \
        clean_env(scenario='constant_velocity')
    oa, _ = ea.reset(seed=9)
    ob, _ = eb.reset(seed=9)
    np.testing.assert_allclose(oa, ob)
    rng = np.random.default_rng(1)
    for _ in range(30):
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        ra, rb = ea.step(a), eb.step(a)
        np.testing.assert_allclose(ra[0], rb[0])
        assert ra[1] == rb[1]
        if ra[2] or ra[3]:
            break


def test_latency_delays_observation_on_sim_time():
    env = GzChaseEnv(FakeGzBackend(), EnvConfig(
        scenario='static', latency=True, latency_range_s=(0.3, 0.3),
        latency_jitter_std_s=0.0, corruption=False))
    obs, _ = env.reset(seed=2)
    ex0 = obs[0]
    o1, *_ = env.step(np.array([0.0, 1.0]))    # hard yaw: truth moves now
    o2, *_ = env.step(np.zeros(2))
    o3, *_ = env.step(np.zeros(2))
    o4, *_ = env.step(np.zeros(2))
    assert o1[0] == pytest.approx(ex0, abs=1e-6)   # 3-step-old pipe
    assert abs(o4[0] - ex0) > 0.02


def test_sign_test_passes_on_fake(tmp_path):
    out = tmp_path / 'sign.json'
    assert sign_test_main(['--fake', '--out', str(out)]) == 0


def test_fit_t_lag_recovers_time_constant():
    import math
    t_true = 0.15
    times = [0.1 * k for k in range(1, 20)]
    vels = [0.5 * (1 - math.exp(-t / t_true)) for t in times]
    assert fit_t_lag(times, vels, 0.5) == pytest.approx(t_true, rel=0.05)


def test_fake_backend_first_order_response():
    """The fake's velocity response is itself a first-order lag -- the
    calibrate_lag script must recover its tau."""
    b = FakeGzBackend(response_tau=0.15)
    b.start()
    b.set_pose('follower', (0, 0, 1), 0.0)
    b.send_twist('follower', (0.5, 0.0, 0.0), 0.0)
    times, vels = [], []
    for k in range(1, 16):
        b.step(100)
        times.append(0.1 * k)
        vels.append(float(b.get_odom('follower').lin_body[0]))
    assert fit_t_lag(times, vels, 0.5) == pytest.approx(0.15, rel=0.1)
