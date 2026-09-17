"""Reward anchors and properties -- the guide 28.2 pre-loop unit tests."""
import math

import numpy as np
import pytest

from chase_gym.reward import (RewardComputer, RewardConfig, track_faithful,
                              track_repaired)

Z2 = np.zeros(2)


def test_repaired_anchors_exact():
    # Guide 16.2 anchors [V]: r(0)=1.0, r(50)=0.5, r(100)=0, r(300)=-0.5,
    # r(600)=-1.25.
    assert track_repaired(0.0) == pytest.approx(1.0)
    assert track_repaired(50.0) == pytest.approx(0.5)
    assert track_repaired(100.0) == pytest.approx(0.0)
    assert track_repaired(300.0) == pytest.approx(-0.5)
    assert track_repaired(600.0) == pytest.approx(-1.25)


def test_repaired_is_continuous_at_threshold():
    eps = 1e-9
    assert abs(track_repaired(100.0 - eps) - track_repaired(100.0 + eps)) < 1e-6


def test_faithful_branch_values():
    # rl_block_diagram.md section 5: 0 just inside, -25 just outside --
    # the 25-point cliff is the point; defined everywhere (else catches =).
    assert track_faithful(100.0) == pytest.approx(0.0)
    assert track_faithful(100.0001) == pytest.approx(-25.000025)
    assert track_faithful(0.0) == pytest.approx(100.0)
    assert track_faithful(600.0) == pytest.approx(-150.0)


def test_faithful_computer_ignores_extras():
    rc = RewardComputer(RewardConfig(version='faithful'))
    r = rc.compute(dist_px=50.0, action=np.ones(2), prev_action=-np.ones(2),
                   lost=True, captured=False)
    assert r == pytest.approx(50.0)     # no smoothness, no loss term


def test_smoothness_and_loss_terms():
    rc = RewardComputer(RewardConfig(version='repaired', w_smooth=0.05,
                                     w_loss=5.0))
    base = rc.compute(dist_px=0.0, action=Z2, prev_action=Z2, lost=False)
    jerk = rc.compute(dist_px=0.0, action=np.array([1.0, -1.0]),
                      prev_action=np.array([-1.0, 1.0]), lost=False)
    assert base - jerk == pytest.approx(0.05 * 8.0)   # ||delta||^2 = 8
    lost = rc.compute(dist_px=0.0, action=Z2, prev_action=Z2, lost=True)
    assert base - lost == pytest.approx(5.0)


def test_intercept_shaping_telescopes():
    """Potential-based shaping must sum to gamma^T*Phi_T - Phi_0 along any
    trajectory (Ng et al. 1999, guide 22) -- with Phi(terminal)=0."""
    gamma, lam = 0.99, 0.1
    rc = RewardComputer(RewardConfig(version='repaired', w_smooth=0.0,
                                     w_loss=0.0, intercept=True, gamma=gamma,
                                     shaping_lambda=lam,
                                     capture_bonus=0.0))
    ranges = [5.0, 4.0, 3.5, 2.0, 1.0]
    shaping_sum, discount = 0.0, 1.0
    for i in range(1, len(ranges)):
        terminal = i == len(ranges) - 1
        r_with = rc.compute(dist_px=0.0, action=Z2, prev_action=Z2,
                            lost=False, captured=terminal,
                            range_m=ranges[i], prev_range_m=ranges[i - 1])
        r_without = track_repaired(0.0)
        shaping_sum += discount * (r_with - r_without)
        discount *= gamma
    phi0 = -lam * ranges[0]
    assert shaping_sum == pytest.approx(-phi0, abs=1e-9)


def test_capture_bonus_dominates():
    rc = RewardComputer(RewardConfig(version='repaired', intercept=True,
                                     capture_bonus=10.0))
    r = rc.compute(dist_px=0.0, action=Z2, prev_action=Z2, lost=False,
                   captured=True, range_m=0.4, prev_range_m=0.6)
    assert r > 10.0 - 1.0


def test_unknown_version_rejected():
    with pytest.raises(ValueError):
        RewardComputer(RewardConfig(version='bogus'))
