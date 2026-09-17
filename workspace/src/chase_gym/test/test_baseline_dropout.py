"""Regression: the PN baseline's reacquisition behaviour after a dropout
(review finding 5) -- the lead terms must not difference a real bearing
against the zeroed post-miss error."""
import numpy as np

from chase_gym import PNController
from chase_gym.corruption import Measurement
from chase_gym.observation import ObservationAssembler, ObservationSpec


def obs_sequence(*measurements):
    spec = ObservationSpec()
    asm = ObservationAssembler(spec)
    out = []
    for m in measurements:
        out.append(asm.assemble(m, spec.control_dt_s))
        asm.record_action(np.zeros(2))
    return spec, out


def test_no_fabricated_kick_on_reacquisition():
    off_centre = Measurement(u=720.0, v=360.0, w_px=50.0)   # ex = +0.5
    spec, seq = obs_sequence(off_centre, None, off_centre)
    pn = PNController(spec)
    pn.reset()
    a_first = pn.get_action(seq[0])          # first frame: centring only
    a_miss = pn.get_action(seq[1])
    a_reacq = pn.get_action(seq[2])          # reacquisition frame
    assert np.all(a_miss == 0.0)             # never chases a ghost
    # Reacquisition must equal a pure-centring response (no rate terms):
    # yaw toward the target (negative for ex > 0), NOT a saturated kick
    # from the fabricated (0.5 - 0.0)/dt bearing rate.
    np.testing.assert_allclose(a_reacq, a_first)
    assert -1.0 < a_reacq[1] < 0.0
    assert abs(a_reacq[1]) < 0.5             # far from saturation


def test_rates_active_when_continuously_visible():
    drift = [Measurement(u=480.0 + 20 * i, v=360.0, w_px=50.0)
             for i in range(4)]
    spec, seq = obs_sequence(*drift)
    pn = PNController(spec)
    pn.reset()
    acts = [pn.get_action(o) for o in seq]
    # From the second visible frame the PN term engages: a rightward-
    # drifting target (az_dot > 0, psi_dot 0) demands a clockwise turn,
    # more negative than centring alone would give for the same error.
    centring_only = -pn.k_center * seq[2][0]
    assert acts[2][1] < centring_only
