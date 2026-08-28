"""End-to-end simulation of a Tello flight through the full estimator.

Synthesises every stream the real drone produces, *with the real drone's
limitations* -- 10 Hz jittery telemetry, whole-degree attitude, milli-g
accelerometer quantisation, integer-decimetre velocity, a drifting barometer, a
free-running yaw, and monocular vision that supplies bearing but never scale --
then measures what comes out.

The point is not that a simulation proves flight performance; it does not. The
point is that these tests pin down the *structural* claims the design rests on,
each of which would otherwise be an assertion in a document:

* the estimator is metric, and stays metric, with no metric position input;
* vision measurably reduces drift over inertial-plus-flow dead reckoning;
* the reported covariance is honest (NEES within its chi-square band), which is
  what makes the innovation gates and any downstream fusion meaningful;
* removing the drone's optical-flow velocity destroys metric scale -- the
  dependency the whole architecture is built around.
"""
import numpy as np
import pytest

from tello_vio import lie
from tello_vio.eskf import ErrorStateKF, EskfConfig
from tello_vio.preintegration import GRAVITY_ENU
from tello_vio.tello_model import G0, StationarityDetector, TelloImuSurrogate, TelloNoise

R_BC = lie.euler_zyx_to_rot(-np.pi / 2, 0.0, -np.pi / 2)
P_BC = np.array([0.03, 0.0, -0.01])


# --------------------------------------------------------------------------- #
# Ground-truth flight
# --------------------------------------------------------------------------- #

def trajectory(t):
    """A figure-of-eight at ~1.5 m, ~0.5 m/s, with coordinated yaw.

    Chosen because it is the motion a Tello actually flies indoors and because
    it excites all three axes -- a straight line would leave the estimator's
    lateral states unobservable and flatter the result.
    """
    w = 2 * np.pi / 20.0                     # 20 s period
    A, B = 2.0, 1.2
    p = np.array([A * np.sin(w * t), B * np.sin(2 * w * t), 1.5 + 0.25 * np.sin(0.7 * w * t)])
    v = np.array([A * w * np.cos(w * t), 2 * B * w * np.cos(2 * w * t),
                  0.25 * 0.7 * w * np.cos(0.7 * w * t)])
    a = np.array([-A * w ** 2 * np.sin(w * t), -4 * B * w ** 2 * np.sin(2 * w * t),
                  -0.25 * (0.7 * w) ** 2 * np.sin(0.7 * w * t)])
    # Yaw follows the velocity heading (the drone flies where it looks); a small
    # bank and pitch keep the attitude realistic rather than level.
    yaw = np.arctan2(v[1], v[0])
    pitch = np.deg2rad(-6.0) * np.cos(w * t)
    roll = np.deg2rad(8.0) * np.sin(2 * w * t)
    R = lie.euler_zyx_to_rot(yaw, pitch, roll)
    return p, v, a, R


def simulate(duration=60.0, seed=0, telemetry_hz=10.0, image_hz=30.0,
             kf_period=0.30, with_vision=True, with_flow=True, flow_noise=0.08,
             attitude_quant_deg=1.0, accel_quant_mg=1.0, vel_quant_dm=1.0):
    """Run the ESKF over a synthesised flight. Returns (times, est, truth, kf)."""
    rng = np.random.default_rng(seed)
    noise = TelloNoise()

    ba_true = np.array([0.12, -0.08, 0.05])       # m/s^2
    baro_offset = 104.7                            # m of pressure altitude
    yaw_drift_rate = np.deg2rad(0.4)               # deg/s of free-running yaw

    kf = ErrorStateKF(EskfConfig(
        accel_noise_density=noise.accel_noise_density,
        gyro_noise_density=noise.gyro_noise_density,
        accel_bias_rw=noise.accel_bias_rw,
        gyro_bias_rw=noise.gyro_bias_rw,
    ))
    surrogate = TelloImuSurrogate()
    stationary = StationarityDetector(noise)

    # Telemetry epochs with realistic WiFi jitter.
    n_tel = int(duration * telemetry_hz)
    t_tel = np.cumsum(rng.normal(1.0 / telemetry_hz, 0.012, n_tel))
    t_tel = t_tel[t_tel < duration]

    kf_times = np.arange(0.0, duration, kf_period) if with_vision else np.array([])

    est_t, est_p, est_v, est_q, true_p, true_v, true_R = [], [], [], [], [], [], []
    nees = []
    last_kf_R = None
    last_kf_p = None
    next_kf = 0

    for k, t in enumerate(t_tel):
        p_t, v_t, a_t, R_t = trajectory(t)

        # ---- synthesise telemetry ------------------------------------- #
        f_body = R_t.T @ (a_t - GRAVITY_ENU)                 # specific force
        accel = f_body + ba_true + rng.normal(0, 0.06, 3)
        if accel_quant_mg:                                   # milli-g quantiser
            q = accel_quant_mg * G0 / 1000.0
            accel = np.round(accel / q) * q

        yaw, pitch, roll = lie.rot_to_euler_zyx(R_t)
        yaw_m = yaw + yaw_drift_rate * t                     # free-running yaw
        att = np.array([np.degrees(yaw_m), np.degrees(pitch), np.degrees(roll)])
        att = att + rng.normal(0, 0.4, 3)
        if attitude_quant_deg:                               # whole degrees
            att = np.round(att / attitude_quant_deg) * attitude_quant_deg
        q_meas = lie.euler_zyx_to_quat(*np.radians(att))

        v_body = R_t.T @ v_t + rng.normal(0, flow_noise, 3)
        if vel_quant_dm:                                     # integer dm/s
            q = vel_quant_dm * 0.1
            v_body = np.round(v_body / q) * q

        baro = p_t[2] + baro_offset + 0.02 * np.sin(0.05 * t) + rng.normal(0, 0.25)
        tof = p_t[2] / R_t[2, 2] + rng.normal(0, 0.02) if p_t[2] < 1.2 else None

        gyro = surrogate.update(t, q_meas)
        still = stationary.update(accel, v_body, gyro)

        # ---- estimator ------------------------------------------------ #
        if not kf.initialised:
            kf.initialise(t, q_meas, baro, p=np.zeros(3))
            # Anchor the world origin at the true starting point so the errors
            # below measure drift, not an arbitrary frame offset.
            kf.p = p_t.copy()
            kf.v = v_t.copy()
            kf.clone()
            last_kf_R, last_kf_p = R_t.copy(), p_t.copy()
            continue

        dt = t - kf.t
        kf.propagate(accel, gyro, dt)
        kf.update_attitude(q_meas, noise.att_rp_std, noise.att_yaw_std)
        if still:
            kf.update_zero_velocity(0.02)
        elif with_flow:
            kf.update_body_velocity(v_body, max(noise.vel_body_std, flow_noise))
        kf.update_barometer(baro, noise.baro_std)
        if tof is not None:
            kf.update_tof(tof, noise.tof_std)

        # ---- vision, at keyframe boundaries --------------------------- #
        while with_vision and next_kf < len(kf_times) and kf_times[next_kf] <= t:
            t_kf = kf_times[next_kf]
            next_kf += 1
            p_k, v_k, _, R_k = trajectory(t_kf)

            # Camera poses at the previous and current keyframe.
            R_c1 = last_kf_R @ R_BC
            R_c2 = R_k @ R_BC
            p_c1 = last_kf_p + last_kf_R @ P_BC
            p_c2 = p_k + R_k @ P_BC
            R_rel = R_c1.T @ R_c2
            t_rel = R_c1.T @ (p_c2 - p_c1)
            base = float(np.linalg.norm(t_rel))

            # Measurement noise matching what frontend.noise_std reports for a
            # healthy essential-matrix solve.
            rot_std, dir_std = 0.033 * np.sqrt(60.0 / 150.0), 0.128 * np.sqrt(60.0 / 150.0)
            R_meas = R_rel @ lie.Exp(rng.normal(0, rot_std, 3))
            if base > 0.03:
                d = t_rel / base + rng.normal(0, dir_std, 3)
                d /= np.linalg.norm(d)
                reliable = True
            else:
                d = np.zeros(3)
                reliable = False

            kf.update_visual_relative(R_meas, d, R_BC, P_BC, rot_std, dir_std,
                                      use_translation=reliable)
            kf.clone()
            last_kf_R, last_kf_p = R_k.copy(), p_k.copy()

        est_t.append(t)
        est_p.append(kf.p.copy())
        est_v.append(kf.v.copy())
        est_q.append(kf.q.copy())
        true_p.append(p_t)
        true_v.append(v_t)
        true_R.append(R_t)

        # Normalised estimation error squared on position: the standard
        # filter-consistency statistic.
        e = kf.p - p_t
        C = kf.P[0:3, 0:3] + np.eye(3) * 1e-9
        nees.append(float(e @ np.linalg.solve(C, e)))

    return (np.array(est_t), np.array(est_p), np.array(est_v), np.array(est_q),
            np.array(true_p), np.array(true_v), true_R, np.array(nees), kf)


def ate(est_p, true_p):
    """Absolute trajectory error: RMS position error, metres."""
    return float(np.sqrt(np.mean(np.sum((est_p - true_p) ** 2, axis=1))))


def path_length(true_p):
    return float(np.sum(np.linalg.norm(np.diff(true_p, axis=0), axis=1)))


# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_full_stack_tracks_a_60_second_flight():
    t, ep, ev, eq, tp, tv, tR, nees, kf = simulate(duration=60.0, seed=1)
    err = ate(ep, tp)
    dist = path_length(tp)
    final = float(np.linalg.norm(ep[-1] - tp[-1]))
    print(f"\n60 s flight, {dist:.1f} m flown:"
          f"\n  ATE (RMS)      = {err:.3f} m"
          f"\n  final error    = {final:.3f} m  ({100*final/dist:.2f} % of path)"
          f"\n  velocity RMS   = {ate(ev, tv):.3f} m/s"
          f"\n  accel bias est = {np.round(kf.ba, 3)} (true [0.12 -0.08 0.05])")
    assert err < 0.60, f"ATE {err:.3f} m"
    assert final / dist < 0.03, f"drift {100*final/dist:.2f} % of path"


@pytest.mark.slow
def test_vision_is_what_keeps_the_estimate_bounded_without_optical_flow():
    """The visual front-end's real job, measured.

    With the Tello's flow sensor working over textured ground its velocity
    output is excellent, and the estimator leans on it. The scenario that
    matters is when it is *not* working -- above roughly a metre of altitude,
    over a plain floor, in low light, or over a moving surface -- because there
    the flow estimate degrades or drops out entirely and inertial dead
    reckoning alone diverges within seconds.
    """
    _, ep_v, _, _, tp, _, _, _, _ = simulate(duration=40.0, seed=7,
                                             with_flow=False, with_vision=True)
    _, ep_n, _, _, tp2, _, _, _, _ = simulate(duration=40.0, seed=7,
                                              with_flow=False, with_vision=False)
    n = min(len(ep_v), len(ep_n))
    e_v, e_n = ate(ep_v[:n], tp[:n]), ate(ep_n[:n], tp2[:n])
    print(f"\n  no flow: with vision {e_v:.2f} m   vs   inertial only {e_n:.2f} m"
          f"   ({e_n / max(e_v, 1e-9):.0f}x better)")
    assert e_v < 0.15 * e_n, (e_v, e_n)


@pytest.mark.slow
def test_vision_helps_progressively_as_optical_flow_degrades():
    """Sweeps flow quality; vision's contribution grows as flow worsens.

    Averaged over several seeds on purpose. A single 40 s run has roughly +-30 %
    spread in ATE, so a one-seed A/B comparison at the healthy-flow end is pure
    noise and would make this test flap. Five seeds is enough to separate a
    ~30 % effect from that spread.
    """
    seeds = range(5)
    rows = []
    for fn in (0.08, 0.30, 0.80):
        ev, en = [], []
        for seed in seeds:
            _, a, _, _, tp, _, _, _, _ = simulate(duration=40.0, seed=seed,
                                                  flow_noise=fn, with_vision=True)
            _, b, _, _, tq, _, _, _, _ = simulate(duration=40.0, seed=seed,
                                                  flow_noise=fn, with_vision=False)
            n = min(len(a), len(b))
            ev.append(ate(a[:n], tp[:n]))
            en.append(ate(b[:n], tq[:n]))
        rows.append((fn, float(np.mean(ev)), float(np.mean(en))))

    print("\n  flow sigma   with vision   without vision   improvement")
    for fn, e_v, e_n in rows:
        print(f"    {fn:.2f} m/s      {e_v:6.3f} m       {e_n:6.3f} m        "
              f"{100*(1-e_v/e_n):+5.0f} %")

    # Healthy flow: vision is near-redundant for short-horizon odometry. That
    # is a real property of this airframe, not a defect -- the Tello's own
    # optical-flow velocity is genuinely good over textured ground -- so it is
    # asserted as "does no harm" rather than quietly hidden. Vision's payoff
    # here is redundancy and, with the ORB-SLAM2 backend, loop closure.
    assert rows[0][1] < 1.15 * rows[0][2], f"vision hurt with good flow: {rows[0]}"
    # Mildly degraded flow: a clear win.
    assert rows[1][1] < 0.85 * rows[1][2], rows[1]
    # Badly degraded flow: a large win.
    assert rows[2][1] < 0.60 * rows[2][2], rows[2]


@pytest.mark.slow
def test_scale_is_metric_without_any_metric_position_input():
    """No GPS, no motion capture, no known landmark -- and still in metres.

    Scale is checked as the ratio of estimated to true *path length*, which is
    the quantity a monocular system gets wrong. A pure mono pipeline would show
    an arbitrary, drifting ratio here.
    """
    _, ep, _, _, tp, _, _, _, _ = simulate(duration=60.0, seed=3)
    s = path_length(ep) / path_length(tp)
    print(f"\n  path-length ratio = {s:.4f} (1.0 is perfect metric scale)")
    assert 0.90 < s < 1.10, f"scale error {100*(s-1):.1f} %"


@pytest.mark.slow
def test_removing_optical_flow_destroys_metric_scale():
    """States the architectural dependency as a falsifiable test.

    Without the drone's own metric velocity the only scale sources left are the
    barometer (vertical only) and double-integrated accelerometer. Horizontal
    scale should degrade badly -- which is exactly why the design leans on the
    flow sensor rather than treating it as optional.
    """
    _, ep_on, _, _, tp, _, _, _, _ = simulate(duration=40.0, seed=4, with_flow=True)
    _, ep_off, _, _, tp2, _, _, _, _ = simulate(duration=40.0, seed=4, with_flow=False)
    n = min(len(ep_on), len(ep_off))
    e_on, e_off = ate(ep_on[:n], tp[:n]), ate(ep_off[:n], tp2[:n])
    print(f"\n  with flow {e_on:.3f} m   vs   without flow {e_off:.3f} m")
    assert e_off > 3.0 * e_on, (e_on, e_off)


@pytest.mark.slow
def test_filter_covariance_is_consistent():
    """NEES inside a sane band: the covariance is neither a lie nor useless.

    For a 3-D position error the expected NEES is 3.0. Measured over 8 seeds
    this filter averages ~3.8, i.e. it is **mildly over-confident, by roughly
    25 %**. That is an honest result worth stating rather than rounding away:
    a factor of ~1.3 is well inside the range where innovation gating and
    downstream fusion still behave, but it is not perfect calibration, and the
    direction of the error is the dangerous one.

    Over-confidence matters because it is what makes a chi-square gate reject
    good measurements and accept bad ones. Under-confidence only makes the
    filter sluggish.
    """
    means = []
    for seed in range(8):
        _, _, _, _, _, _, _, nees, _ = simulate(duration=40.0, seed=seed)
        means.append(float(np.mean(nees[len(nees) // 4:])))   # skip convergence
    m = float(np.mean(means))
    print(f"\n  mean position NEES over 8 seeds = {m:.2f} (ideal 3.0, "
          f"ratio {m/3:.2f}x)\n  per-seed {np.round(means, 2)}")
    assert 1.0 < m < 9.0, f"NEES {m:.2f} indicates an inconsistent filter"


@pytest.mark.slow
def test_attitude_stays_bounded_despite_free_running_yaw():
    """Roll/pitch are gravity-observable; yaw is not, and must not poison them."""
    _, _, _, eq, _, _, tR, _, _ = simulate(duration=60.0, seed=5)
    rp, yw = [], []
    for q, R in zip(eq, tR):
        e = lie.Log(R.T @ lie.quat_to_rot(q))
        rp.append(np.linalg.norm(e[:2]))
        yw.append(abs(e[2]))
    print(f"\n  roll/pitch RMS = {np.degrees(np.sqrt(np.mean(np.square(rp)))):.2f} deg"
          f"   yaw RMS = {np.degrees(np.sqrt(np.mean(np.square(yw)))):.2f} deg")
    assert np.degrees(np.sqrt(np.mean(np.square(rp)))) < 5.0
