"""Sim(3) alignment tests, including the coplanar case that trips naive SVD."""
import numpy as np
import pytest

from tello_vio import lie
from tello_vio.sim3 import Sim3, ransac_sim3, umeyama_sim3


def test_recovers_a_known_similarity_exactly():
    rng = np.random.default_rng(0)
    R = lie.Exp(rng.normal(scale=0.7, size=3))
    s, t = 2.37, np.array([1.5, -0.4, 0.9])
    src = rng.normal(scale=3.0, size=(50, 3))
    dst = s * (src @ R.T) + t

    T = umeyama_sim3(src, dst)
    assert np.isclose(T.s, s, rtol=1e-10)
    assert np.allclose(T.R, R, atol=1e-10)
    assert np.allclose(T.t, t, atol=1e-9)
    assert np.allclose(T.apply(src), dst, atol=1e-9)


def test_inverse_round_trips():
    rng = np.random.default_rng(1)
    T = Sim3(0.4, lie.Exp(rng.normal(size=3)), rng.normal(size=3))
    x = rng.normal(size=(20, 3))
    assert np.allclose(T.inverse().apply(T.apply(x)), x, atol=1e-10)
    assert np.allclose(T.matrix() @ np.r_[x[0], 1.0], np.r_[T.apply(x[0]), 1.0], atol=1e-12)


def test_coplanar_points_do_not_produce_a_reflection():
    """A drone flying at constant height gives a nearly planar trajectory."""
    rng = np.random.default_rng(2)
    src = np.column_stack([rng.normal(size=60), rng.normal(size=60),
                           np.full(60, 1.2) + rng.normal(scale=1e-4, size=60)])
    R = lie.Exp([0.0, 0.0, 0.9])
    s, t = 1.7, np.array([0.3, -0.2, 0.1])
    dst = s * (src @ R.T) + t

    T = umeyama_sim3(src, dst)
    assert np.isclose(np.linalg.det(T.R), 1.0, atol=1e-9), "returned a reflection"
    assert np.isclose(T.s, s, rtol=1e-6)
    assert np.allclose(T.apply(src), dst, atol=1e-6)


def test_scale_can_be_locked_to_one():
    rng = np.random.default_rng(3)
    R = lie.Exp(rng.normal(scale=0.5, size=3))
    src = rng.normal(size=(30, 3))
    dst = (src @ R.T) + np.array([1.0, 2.0, 3.0])
    T = umeyama_sim3(src, dst, with_scale=False)
    assert np.isclose(T.s, 1.0)
    assert np.allclose(T.apply(src), dst, atol=1e-9)


def test_ransac_rejects_outliers_that_would_wreck_the_scale():
    rng = np.random.default_rng(4)
    R = lie.Exp([0.1, -0.2, 0.4])
    s, t = 3.1, np.array([0.5, 0.5, -1.0])
    src = rng.normal(scale=2.0, size=(80, 3))
    dst = s * (src @ R.T) + t + rng.normal(scale=0.01, size=(80, 3))
    bad = rng.choice(80, 16, replace=False)
    dst[bad] += rng.normal(scale=20.0, size=(16, 3))

    T_ls = umeyama_sim3(src, dst)
    T_rs, inl = ransac_sim3(src, dst, threshold=0.1, iterations=300)
    assert abs(T_rs.s - s) < 0.02, T_rs.s
    assert abs(T_rs.s - s) < 0.25 * abs(T_ls.s - s), (T_rs.s, T_ls.s)
    assert np.count_nonzero(inl[bad]) <= 2


def test_alignment_of_a_scale_free_slam_track_to_a_metric_one():
    """The actual use case: recover a monocular map's metric scale."""
    rng = np.random.default_rng(5)
    t = np.linspace(0, 12, 120)
    metric = np.column_stack([1.2 * np.sin(0.4 * t), 0.8 * t, 1.5 + 0.2 * np.sin(t)])
    true_scale = 0.137                       # SLAM's arbitrary internal units
    R_ms = lie.Exp([0.02, -0.05, 1.1])
    slam = (1.0 / true_scale) * (metric @ R_ms) + np.array([3.0, -2.0, 0.5])
    slam += rng.normal(scale=0.05, size=slam.shape)

    T, _ = ransac_sim3(slam, metric, threshold=0.1, iterations=200)
    assert abs(T.s - true_scale) / true_scale < 0.05, T.s
    resid = np.linalg.norm(T.apply(slam) - metric, axis=1)
    assert np.median(resid) < 0.05, np.median(resid)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        umeyama_sim3(np.zeros((2, 3)), np.zeros((2, 3)))
