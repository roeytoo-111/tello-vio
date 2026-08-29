"""Timing correctness of the telemetry/image fusion schedule.

Driven by the rates MEASURED on real hardware, not assumed ones:
telemetry 10.0 Hz, images 27.6 Hz. Because those are mutually prime-ish, an
image time almost never coincides with a telemetry sample, and what the filter
does with that residual interval decides whether every visual update is
compared against a state tens of milliseconds in its past.

These tests exercise the same replay-then-hold logic as
``vio_node._advance_to`` without needing ROS.
"""
import numpy as np
import pytest

from tello_vio import lie
from tello_vio.eskf import ErrorStateKF, EskfConfig

TELEM_HZ = 10.0        # measured
IMAGE_HZ = 27.6        # measured


class Sample:
    __slots__ = ("t", "accel", "gyro")

    def __init__(self, t, accel, gyro):
        self.t, self.accel, self.gyro = t, accel, gyro


def advance(kf, buf, last_fused, last_sample, t_target, hold: bool):
    """Mirror of vio_node._advance_to; `hold` toggles the residual propagation."""
    newest = buf[-1].t if buf else None
    for s in buf:
        if s.t <= last_fused or s.t > t_target:
            continue
        kf.propagate(s.accel, s.gyro, s.t - last_fused)
        last_fused, last_sample = s.t, s
    if hold and last_sample is not None and newest is not None:
        reach = min(t_target, newest)
        dt = reach - last_fused
        if 0.0 < dt <= kf.cfg.max_dt:
            kf.propagate(last_sample.accel, last_sample.gyro, dt)
            last_fused = reach
    return last_fused, last_sample


def run(hold, td=0.25, duration=20.0, speed=0.5):
    """Constant-velocity flight; measure how far the state lags each image."""
    kf = ErrorStateKF(EskfConfig(gating=False))
    kf.initialise(0.0, lie.quat_identity(), 100.0)
    kf.v = np.array([speed, 0.0, 0.0])
    accel = np.array([0.0, 0.0, 9.80665])          # level, no acceleration

    buf, last_fused, last_sample = [], 0.0, None
    lags = []
    t_tel = np.arange(0.0, duration, 1.0 / TELEM_HZ)
    # Images are stamped at ARRIVAL; subtracting td recovers capture time, so a
    # calibrated td puts image times BEHIND the newest telemetry.
    t_img = np.arange(1.0, duration - 1.0, 1.0 / IMAGE_HZ)

    ti = 0
    for t in t_img:
        while ti < len(t_tel) and t_tel[ti] <= t + td:
            buf.append(Sample(t_tel[ti], accel, np.zeros(3)))
            ti += 1
        last_fused, last_sample = advance(kf, buf, last_fused, last_sample,
                                          t, hold)
        lags.append(t - last_fused)
    return np.asarray(lags)


def test_without_hold_the_state_lags_the_image_by_tens_of_ms():
    """The defect: replaying buffered samples alone leaves a sawtooth lag."""
    lag = run(hold=False)
    assert lag.mean() > 0.030, f"expected a real lag, got {lag.mean()*1000:.1f} ms"
    assert lag.max() > 0.080, f"max lag only {lag.max()*1000:.1f} ms"


def test_zero_order_hold_closes_the_gap():
    """With the residual propagated, the state is at the image time."""
    lag = run(hold=True)
    assert lag.max() < 1e-9, f"still lagging by {lag.max()*1000:.3f} ms"


def test_hold_improves_position_accuracy_at_the_image_instant():
    """The lag is not cosmetic: it is un-modelled motion."""
    speed = 0.5
    lag_no = run(hold=False, speed=speed)
    lag_yes = run(hold=True, speed=speed)
    err_no = lag_no.mean() * speed
    err_yes = lag_yes.mean() * speed
    assert err_no > 0.015, f"{err_no*100:.1f} cm"
    assert err_yes < 1e-6
    assert err_no > 100 * max(err_yes, 1e-9)


def test_hold_is_disabled_when_the_offset_is_uncalibrated():
    """Guard against the regression the clamp exists to prevent.

    With td = 0 the image stamp is its ARRIVAL time, which runs ahead of the
    newest telemetry. Propagating to it would push the filter past the
    telemetry clock, and the next sample to arrive would be older than
    last_fused_t and be silently dropped. The clamp must prevent that.
    """
    kf = ErrorStateKF(EskfConfig(gating=False))
    kf.initialise(0.0, lie.quat_identity(), 100.0)
    accel = np.array([0.0, 0.0, 9.80665])
    buf = [Sample(0.10, accel, np.zeros(3)), Sample(0.20, accel, np.zeros(3))]
    last_fused, last_sample = 0.0, None

    # An image whose (uncorrected) stamp is ahead of all telemetry.
    last_fused, last_sample = advance(kf, buf, last_fused, last_sample,
                                      t_target=0.28, hold=True)
    assert last_fused == pytest.approx(0.20), \
        "filter ran ahead of telemetry; later samples would be dropped"

    # The next telemetry sample must still be consumable.
    buf.append(Sample(0.30, accel, np.zeros(3)))
    last_fused, _ = advance(kf, buf, last_fused, last_sample,
                            t_target=0.38, hold=True)
    assert last_fused == pytest.approx(0.30)


def test_measured_rates_are_what_the_design_assumes():
    """Pin the hardware rates the whole schedule is built around."""
    assert 9.0 < TELEM_HZ < 11.0
    assert 25.0 < IMAGE_HZ < 31.0
    # Images per telemetry sample -- why the residual interval exists at all.
    assert 2.5 < IMAGE_HZ / TELEM_HZ < 3.0
