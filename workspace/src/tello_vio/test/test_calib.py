"""Calibration tests with known ground truth injected into synthetic data."""
import numpy as np
import pytest

from tello_vio import lie
from tello_vio.calib import (allan_deviation, estimate_camera_imu_rotation,
                             estimate_time_offset, identify_imu_noise,
                             rotation_excitation)



# --------------------------------------------------------------------------- #
# Allan deviation / noise identification
# --------------------------------------------------------------------------- #

def test_allan_deviation_recovers_a_known_white_noise_density():
    """For pure white noise, ADEV(tau) = N / sqrt(tau), so ADEV(1) == N."""
    dt, N = 0.005, 0.04              # density in units/sqrt(Hz)
    n = 400_000
    x = np.random.default_rng(101).normal(scale=N / np.sqrt(dt), size=n)
    taus, adev = allan_deviation(x, dt)
    short = taus < 0.2
    pred = N / np.sqrt(taus[short])
    assert np.allclose(adev[short], pred, rtol=0.15), (adev[short][:5], pred[:5])


def test_allan_deviation_shows_the_random_walk_tail():
    """A random walk gives the +1/2 slope; the white-noise part must not hide it."""
    dt, K = 0.01, 0.02
    n = 200_000
    rw = np.cumsum(np.random.default_rng(102).normal(scale=K * np.sqrt(dt), size=n))
    taus, adev = allan_deviation(rw, dt)
    # Fit in a band that is long enough for the random walk to dominate but
    # not so long that only a handful of clusters remain -- the ADEV estimator's
    # own variance explodes at tau approaching the log length.
    band = (taus > 1.0) & (taus < taus.max() / 10)
    assert np.count_nonzero(band) >= 5
    slope = np.polyfit(np.log(taus[band]), np.log(adev[band]), 1)[0]
    assert 0.35 < slope < 0.65, slope


def test_identify_imu_noise_recovers_bias_and_density():
    dt = 0.01
    n = 120_000
    true_bias = np.array([0.08, -0.05, 0.03])
    nd = 0.05
    r = np.random.default_rng(103)
    accel = np.array([0.0, 0.0, 9.80665]) + true_bias \
        + r.normal(scale=nd / np.sqrt(dt), size=(n, 3))
    gyro_bias = np.array([0.01, -0.02, 0.005])
    gnd = 0.004
    gyro = gyro_bias + r.normal(scale=gnd / np.sqrt(dt), size=(n, 3))

    res = identify_imu_noise(accel, gyro, dt)
    # x/y bias is directly identifiable; z is confounded with |g| by construction.
    assert np.allclose(res.accel_bias[:2], true_bias[:2], atol=0.01), res.accel_bias
    assert np.allclose(res.gyro_bias, gyro_bias, atol=0.002), res.gyro_bias
    assert 0.7 * nd < res.accel_noise_density < 1.4 * nd, res.accel_noise_density
    assert 0.7 * gnd < res.gyro_noise_density < 1.4 * gnd, res.gyro_noise_density
    assert not res.warnings, res.warnings


def test_identify_imu_noise_warns_about_wrong_units():
    """Raw milli-g fed in by mistake must be caught, not silently fitted."""
    dt = 0.01
    accel = np.array([0.0, 0.0, -1000.0]) + np.random.default_rng(104).normal(scale=5.0, size=(5000, 3))
    res = identify_imu_noise(accel, None, dt)
    assert any("9.81" in w for w in res.warnings), res.warnings


def test_identify_imu_noise_warns_when_the_drone_moved():
    dt = 0.01
    n = 20000
    ramp = np.linspace(0, 0.6, n)[:, None] * np.array([1.0, 0.0, 0.0])
    accel = np.array([0.0, 0.0, 9.80665]) + ramp + np.random.default_rng(105).normal(scale=0.3, size=(n, 3))
    res = identify_imu_noise(accel, None, dt)
    assert any("drift" in w for w in res.warnings), res.warnings


# --------------------------------------------------------------------------- #
# Hand-eye
# --------------------------------------------------------------------------- #

def make_handeye_data(R_BC, n=60, noise_deg=0.0, seed=0, axes="xyz"):
    rng = np.random.default_rng(seed)
    Rc, Rb = [], []
    for _ in range(n):
        v = rng.normal(size=3)
        if axes == "z":
            v = np.array([0.0, 0.0, 1.0]) * rng.normal()
        v = v / (np.linalg.norm(v) + 1e-12) * rng.uniform(0.15, 0.6)
        R_b = lie.Exp(v)
        R_c = R_BC.T @ R_b @ R_BC
        if noise_deg:
            R_c = R_c @ lie.Exp(rng.normal(scale=np.deg2rad(noise_deg), size=3))
            R_b = R_b @ lie.Exp(rng.normal(scale=np.deg2rad(noise_deg), size=3))
        Rc.append(R_c)
        Rb.append(R_b)
    return Rc, Rb


def test_hand_eye_recovers_the_extrinsic_exactly_without_noise():
    # The standard FLU-body -> optical-camera rotation.
    R_BC = lie.euler_zyx_to_rot(-np.pi / 2, 0.0, -np.pi / 2)
    Rc, Rb = make_handeye_data(R_BC, n=40)
    res = estimate_camera_imu_rotation(Rc, Rb)
    err = np.degrees(np.linalg.norm(lie.Log(R_BC.T @ res.R_BC)))
    assert err < 1e-6, err
    assert res.residual_deg < 1e-6
    assert res.n_used == 40


@pytest.mark.parametrize("noise_deg", [0.5, 1.5, 3.0])
def test_hand_eye_degrades_gracefully_with_noise(noise_deg):
    R_BC = lie.euler_zyx_to_rot(-np.pi / 2, 0.0, -np.pi / 2)
    Rc, Rb = make_handeye_data(R_BC, n=200, noise_deg=noise_deg, seed=3)
    res = estimate_camera_imu_rotation(Rc, Rb, max_angle_mismatch_deg=15.0)
    err = np.degrees(np.linalg.norm(lie.Log(R_BC.T @ res.R_BC)))
    # Averaging over 200 pairs should beat the per-pair noise substantially.
    assert err < noise_deg, (err, noise_deg)


def test_hand_eye_rejects_single_axis_excitation():
    """Rotating only about z cannot determine R_BC about z. Say so."""
    R_BC = lie.euler_zyx_to_rot(-np.pi / 2, 0.0, -np.pi / 2)
    _, Rb = make_handeye_data(R_BC, n=60, axes="z", seed=4)
    assert rotation_excitation(Rb) < 0.05, rotation_excitation(Rb)
    _, good_b = make_handeye_data(R_BC, n=60, seed=5)
    assert rotation_excitation(good_b) > 0.3


def test_hand_eye_needs_enough_usable_pairs():
    tiny = [lie.Exp([0.001, 0, 0]) for _ in range(20)]
    with pytest.raises(ValueError, match="usable rotation pairs"):
        estimate_camera_imu_rotation(tiny, tiny)


def test_hand_eye_drops_pairs_whose_angles_disagree():
    """Blunders must be filtered -- including ones the angle screen cannot see.

    A quarter of the camera rotations are corrupted by an extra 25 deg twist.
    Most change the total rotation angle and are caught by the cheap screen;
    at least one does not (it lands at 0.1 deg of angle mismatch here) and is
    only removed by the residual-based second stage.
    """
    R_BC = lie.euler_zyx_to_rot(-np.pi / 2, 0.0, -np.pi / 2)
    Rc, Rb = make_handeye_data(R_BC, n=60, seed=6)
    # Corrupt a quarter of the camera rotations with a large angle change.
    for i in range(0, 60, 4):
        Rc[i] = Rc[i] @ lie.Exp([0.0, 0.0, np.deg2rad(25.0)])
    res = estimate_camera_imu_rotation(Rc, Rb, max_angle_mismatch_deg=4.0)
    assert res.n_rejected >= 10, res.n_rejected
    err = np.degrees(np.linalg.norm(lie.Log(R_BC.T @ res.R_BC)))
    assert err < 2.0, err


# --------------------------------------------------------------------------- #
# Time offset
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("true_offset", [0.0, 0.05, -0.08, 0.21, -0.30])
def test_time_offset_recovers_a_known_delay(true_offset):
    fs = 200.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    # A rate-magnitude-like signal: non-negative, broadband, non-periodic.
    rng = np.random.default_rng(7)
    base = np.abs(np.convolve(rng.normal(size=t.size), np.ones(40) / 40, "same"))

    t_a, s_a = t, base
    # Stream b observes the same physical signal but its timestamps are late
    # by `true_offset`, so b's sample at time t carries a(t - offset).
    t_b = t + true_offset
    s_b = base + rng.normal(scale=0.01, size=t.size)

    res = estimate_time_offset(t_a, s_a, t_b, s_b, max_offset_s=0.6)
    assert abs(res.offset_s - true_offset) < 0.01, (res.offset_s, true_offset)
    # Resampling a 200 Hz broadband signal onto the 50 Hz correlation grid
    # costs some peak height without moving the peak; 0.8 still indicates a
    # confidently locked correlation, which is what this number is for.
    assert res.correlation > 0.8, res.correlation


def test_time_offset_beats_the_resampling_grid():
    """Sub-sample refinement must do better than 1/resample_hz quantisation."""
    fs, offset = 200.0, 0.0137          # deliberately not a grid multiple
    t = np.arange(0.0, 20.0, 1.0 / fs)
    rng = np.random.default_rng(8)
    base = np.abs(np.convolve(rng.normal(size=t.size), np.ones(30) / 30, "same"))
    res = estimate_time_offset(t, base, t + offset, base, resample_hz=50.0)
    assert abs(res.offset_s - offset) < 0.5 / 50.0, res.offset_s


def test_time_offset_rejects_an_unexcited_log():
    t = np.arange(0.0, 10.0, 0.01)
    with pytest.raises(ValueError, match="constant"):
        estimate_time_offset(t, np.ones_like(t), t, np.ones_like(t))
