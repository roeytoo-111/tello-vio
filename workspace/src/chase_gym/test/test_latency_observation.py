"""Delay-queue vintage, corruption behaviour, and the observation contract."""
import numpy as np
import pytest

from chase_gym import constants as C
from chase_gym.corruption import CorruptionConfig, CorruptionModel, Measurement
from chase_gym.latency import DelayQueue, LatencyModel
from chase_gym.observation import ObservationAssembler, ObservationSpec


# ---- latency ------------------------------------------------------------

def test_queue_returns_newest_not_after_t():
    q = DelayQueue()
    for i in range(10):
        q.push(i * 0.1, i)
    assert q.sample(0.55)[1] == 5          # newest with t <= 0.55
    assert q.sample(0.0)[1] == 0
    assert q.sample(-0.01) is None         # pipe empty before first frame


def test_queue_rejects_non_monotonic():
    q = DelayQueue()
    q.push(1.0, 'a')
    with pytest.raises(ValueError):
        q.push(0.5, 'b')


def test_latency_draw_in_measured_band():
    m = LatencyModel(enabled=True)
    rng = np.random.default_rng(0)
    for _ in range(100):
        base = m.reset(rng)
        assert C.LATENCY_RANGE_S[0] <= base <= C.LATENCY_RANGE_S[1]
        d = m.delay(rng)
        assert d >= 0.0


def test_latency_disabled_is_zero():
    m = LatencyModel(enabled=False)
    rng = np.random.default_rng(0)
    m.reset(rng)
    assert m.delay(rng) == 0.0


def test_delay_is_1p5_to_3p5_control_periods():
    """The doc-set arithmetic [V]: 150-350 ms at 0.1 s steps."""
    lo = C.LATENCY_RANGE_S[0] / C.DT
    hi = C.LATENCY_RANGE_S[1] / C.DT
    assert lo == pytest.approx(1.5)
    assert hi == pytest.approx(3.5)


# ---- corruption ---------------------------------------------------------

def test_dropout_rises_as_box_shrinks():
    cm = CorruptionModel(CorruptionConfig())
    assert cm.dropout_probability(120.0) < cm.dropout_probability(50.0)
    assert cm.dropout_probability(50.0) < cm.dropout_probability(10.0)
    assert cm.dropout_probability(10.0) == pytest.approx(0.30)


def test_corruption_disabled_passthrough():
    cm = CorruptionModel(CorruptionConfig(enabled=False))
    m = Measurement(100.0, 200.0, 50.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert cm.apply(m, rng) is m


def test_burst_produces_multiframe_gaps():
    """Burst loss must create consecutive misses (the CTS failure mode),
    which independent per-frame dropout at the same rate would rarely do."""
    cfg = CorruptionConfig(sigma_px=0.0, dropout_p_large=0.0,
                           dropout_p_small=0.0, burst_enter_p=0.05,
                           burst_mean_len=6.0)
    cm = CorruptionModel(cfg)
    rng = np.random.default_rng(1)
    m = Measurement(480.0, 360.0, 100.0)
    misses = [cm.apply(m, rng) is None for _ in range(3000)]
    # Longest run of consecutive misses.
    longest = cur = 0
    for x in misses:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    assert longest >= 4


def test_unknown_corruption_key_rejected(tmp_path):
    p = tmp_path / 'bad.yaml'
    p.write_text('sigma_px: 2.0\nsigma_typo: 1.0\n')
    with pytest.raises(ValueError, match='sigma_typo'):
        CorruptionConfig.from_file(str(p))


# ---- observation --------------------------------------------------------

def test_observation_hand_computed():
    spec = ObservationSpec()
    asm = ObservationAssembler(spec)
    asm.record_action(np.array([0.5, -0.25]))
    obs = asm.assemble(Measurement(u=720.0, v=180.0, w_px=50.0), dt_s=0.1)
    assert obs.shape == (14,)
    assert obs[0] == pytest.approx((720 - 480) / 480)   # ex = 0.5
    assert obs[1] == pytest.approx((180 - 360) / 360)   # ey = -0.5
    assert obs[2] == 0.0 and obs[3] == 0.0               # no previous yet
    assert obs[4] == 1.0                                 # visible
    assert obs[5] == 0.0                                 # fresh detection
    assert obs[6] == pytest.approx(0.5)                  # newest action first
    assert obs[7] == pytest.approx(-0.25)
    assert np.all(obs[8:] == 0.0)


def test_miss_zeroes_errors_never_holds():
    asm = ObservationAssembler(ObservationSpec())
    asm.assemble(Measurement(700.0, 500.0, 40.0), 0.1)
    obs = asm.assemble(None, 0.1)
    assert obs[0] == 0.0 and obs[1] == 0.0    # zeroed, not held
    assert obs[4] == 0.0
    assert obs[2] != 0.0                       # prev error still informative


def test_staleness_grows_and_clips():
    spec = ObservationSpec(loss_timeout_s=1.0)
    asm = ObservationAssembler(spec)
    asm.assemble(Measurement(480.0, 360.0, 40.0), 0.1)
    vals = [asm.assemble(None, 0.1)[5] for _ in range(15)]
    assert vals[0] == pytest.approx(0.1)
    assert vals[-1] == 1.0                     # clipped at the timeout


def test_history_stores_sent_post_clamp():
    asm = ObservationAssembler(ObservationSpec())
    asm.record_action(np.array([2.0, -3.0]))   # proposal beyond the range
    obs = asm.assemble(Measurement(480.0, 360.0, 40.0), 0.1)
    assert obs[6] == 1.0 and obs[7] == -1.0    # what was SENT


def test_spec_hash_changes_with_layout():
    a = ObservationSpec()
    b = ObservationSpec(k=3)
    c = ObservationSpec(loss_timeout_s=2.0)
    assert a.spec_hash() != b.spec_hash()
    assert a.spec_hash() != c.spec_hash()
    assert a.spec_hash() == ObservationSpec().spec_hash()


def test_errors_clip_to_unit_box():
    asm = ObservationAssembler(ObservationSpec())
    obs = asm.assemble(Measurement(u=5000.0, v=-900.0, w_px=10.0), 0.1)
    assert obs[0] == 1.0 and obs[1] == -1.0
