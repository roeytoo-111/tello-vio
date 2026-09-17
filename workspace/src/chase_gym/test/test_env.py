"""ChaseEnv contract tests: terminals, truncation, truth-based reward,
latency vintage in-loop, INTERCEPT capture, faithful-env port arithmetic."""
import numpy as np
import pytest

from chase_gym import ChaseEnv, EnvConfig, FaithfulPointMassEnv, PController
from chase_gym import constants as C


def clean_cfg(**kw):
    base = dict(scenario='static', latency=False, corruption=False)
    base.update(kw)
    return EnvConfig(**base)


def test_budget_is_truncation_not_termination():
    env = ChaseEnv(clean_cfg(episode_steps=20))
    obs, _ = env.reset(seed=0)
    ctl = PController(env.obs_spec)
    for i in range(20):
        obs, r, term, trunc, info = env.step(ctl.get_action(obs))
    assert trunc and not term     # guide 11.3: timeouts bootstrap


def test_hard_yaw_loses_target_and_terminates():
    env = ChaseEnv(clean_cfg(episode_steps=300))
    obs, _ = env.reset(seed=0)
    terminated = False
    for i in range(80):
        obs, r, term, trunc, info = env.step(np.array([0.0, 1.0]))
        if term:
            terminated = True
            break
    assert terminated             # full-stick yaw sweeps the 55 deg FOV
    assert not info['in_frame']
    assert r < -4.0               # the loss penalty w_loss=5 landed


def test_reward_uses_truth_under_total_dropout():
    """Detector fully dead -> observation blind, reward still finite and
    truth-based (guide 16.1 line 8)."""
    from chase_gym.corruption import CorruptionConfig
    cfg = clean_cfg(corruption=True,
                    corruption_cfg=CorruptionConfig(
                        dropout_p_large=1.0, dropout_p_small=1.0,
                        sigma_px=0.0, burst_enter_p=0.0))
    env = ChaseEnv(cfg)
    obs, info = env.reset(seed=1)
    obs, r, term, trunc, info = env.step(np.zeros(2))
    assert not info['detected']
    assert obs[4] == 0.0          # observation says blind
    assert np.isfinite(r)
    assert info['in_frame']       # truth: target still there


def test_observation_is_delayed():
    """With max latency, the first steps still see the pre-rolled static
    pipe: a fast-moving world change appears in the observation only
    ~Delta/dt steps later."""
    cfg = EnvConfig(scenario='static', latency=True,
                    latency_range_s=(0.30, 0.30), latency_jitter_std_s=0.0,
                    corruption=False)
    env = ChaseEnv(cfg)
    obs, _ = env.reset(seed=3)
    ex0 = obs[0]
    # Step with a hard yaw: truth moves immediately, observation later.
    obs1, *_ = env.step(np.array([0.0, 1.0]))
    obs2, *_ = env.step(np.array([0.0, 0.0]))
    obs3, *_ = env.step(np.array([0.0, 0.0]))
    obs4, *_ = env.step(np.array([0.0, 0.0]))
    # 0.30 s / 0.1 s = 3 steps of delay: obs1/obs2/obs3 still see the
    # pre-yaw pipe (unchanged ex), obs4 sees the yawed world.
    assert obs1[0] == pytest.approx(ex0, abs=1e-6)
    assert obs2[0] == pytest.approx(ex0, abs=1e-6)
    assert abs(obs4[0] - ex0) > 0.05


def test_intercept_capture_terminal_and_bonus():
    cfg = EnvConfig(task='intercept', scenario='static', latency=False,
                    corruption=False, reset_range_m=(0.9, 1.1),
                    episode_steps=300)
    env = ChaseEnv(cfg)
    obs, info = env.reset(seed=5)
    ctl = PController(env.obs_spec)
    captured = False
    for i in range(200):
        obs, r, term, trunc, info = env.step(ctl.get_action(obs))
        if term:
            captured = info.get('capture', False)
            break
    assert captured               # closing law + centring reaches r_cap
    assert r > 5.0                # capture bonus dominates
    assert info['range_m'] <= cfg.r_cap_m + 0.05


def test_follow_standoff_regulates_range():
    env = ChaseEnv(clean_cfg(episode_steps=300))
    obs, _ = env.reset(seed=7)
    ctl = PController(env.obs_spec)
    for _ in range(200):
        obs, r, term, trunc, info = env.step(ctl.get_action(obs))
        if term or trunc:
            break
    assert abs(info['range_m'] - env.cfg.standoff_m) \
        <= env.cfg.standoff_deadband_m + 0.15


def test_same_seed_same_trajectory():
    a, b = ChaseEnv(EnvConfig(scenario='aggressive')), \
        ChaseEnv(EnvConfig(scenario='aggressive'))
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    np.testing.assert_allclose(oa, ob)
    rng = np.random.default_rng(0)
    for _ in range(50):
        act = rng.uniform(-1, 1, 2).astype(np.float32)
        ra, rb = a.step(act), b.step(act)
        np.testing.assert_allclose(ra[0], rb[0])
        assert ra[1] == rb[1] and ra[2] == rb[2] and ra[3] == rb[3]
        if ra[2] or ra[3]:
            break


def test_info_carries_episode_draws():
    env = ChaseEnv(EnvConfig(scenario='static'))
    _, info = env.reset(seed=0)
    assert C.LATENCY_RANGE_S[0] <= info['latency_base_s'] <= C.LATENCY_RANGE_S[1]
    assert 0.7 * env.cfg.t_lag <= info['t_lag_s'] <= 1.3 * env.cfg.t_lag


# ---- the faithful port --------------------------------------------------

def test_faithful_dynamics_and_reward():
    env = FaithfulPointMassEnv()
    obs, _ = env.reset(seed=0)
    env._pos = np.array([480.0, 360.0])
    obs, r, term, trunc, info = env.step(np.array([10.0, -5.0]))
    # position += 2 * action  [C drone_sim_env.py]
    assert obs[0] == pytest.approx(500.0) and obs[1] == pytest.approx(350.0)
    d = np.hypot(20.0, 10.0)
    assert r == pytest.approx(100.0 - d)
    assert not term


def test_faithful_edge_exit_is_terminal():
    env = FaithfulPointMassEnv()
    env.reset(seed=0)
    env._pos = np.array([950.0, 360.0])
    obs, r, term, trunc, info = env.step(np.array([60.0, 0.0]))
    assert term                       # clamped at 960 -> edge -> done
    assert obs[0] == 960.0
    assert r == pytest.approx(-0.25 * info['dist_px'])


def test_faithful_reset_uniform_in_frame():
    env = FaithfulPointMassEnv()
    xs, ys = [], []
    for s in range(200):
        obs, _ = env.reset(seed=s)
        xs.append(obs[0])
        ys.append(obs[1])
    assert 0 <= min(xs) and max(xs) <= 960
    assert 0 <= min(ys) and max(ys) <= 720
    assert np.std(xs) > 200           # actually spread over the frame
