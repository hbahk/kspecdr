"""
Units of the make_im variance model (2dfdr FILL_VARIANCE).

RO_NOISE is in electrons, RO_GAIN in e-/ADU and the image in ADU, so the variance in ADU^2 is
(RO_NOISE / RO_GAIN)^2 + max(D, 0) / RO_GAIN.

Run from kspecdr/: python tests/test_make_im_variance.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from kspecdr.preproc.make_im import MakeIM


def test_read_noise_floor_in_adu():
    """A zero-signal pixel carries the read noise converted to ADU, not RO_NOISE^2."""
    rn_e, gain = 16.03, 4.09
    var = MakeIM(verbose=False)._calculate_variance(np.zeros((3, 4)), rn_e, gain)
    assert np.allclose(var, (rn_e / gain) ** 2)
    assert np.isclose(var[0, 0], 15.36, atol=0.01)


def test_negative_data_has_only_read_noise():
    rn_e, gain = 4.48, 1.12
    var = MakeIM(verbose=False)._calculate_variance(np.full((2, 2), -50.0), rn_e, gain)
    assert np.allclose(var, (rn_e / gain) ** 2)


def test_matches_simulated_detector():
    """Poisson electrons + Gaussian read noise (e-), digitized with gain g: var in ADU^2."""
    rng = np.random.default_rng(3)
    n = 400_000
    for rn_e, gain, signal_e in ((4.48, 1.12, 0.0), (4.48, 1.12, 50.0), (16.03, 4.09, 200.0)):
        electrons = rng.poisson(signal_e, n) + rng.normal(0.0, rn_e, n)
        adu = electrons / gain
        model = MakeIM(verbose=False)._calculate_variance(
            np.full((1, 1), signal_e / gain), rn_e, gain
        )[0, 0]
        measured = adu.var()
        # sampling error of a variance estimate: sqrt(2 / n) ~ 0.2 %
        assert abs(measured / model - 1.0) < 0.01, (rn_e, gain, signal_e, measured, model)


if __name__ == "__main__":
    test_read_noise_floor_in_adu()
    test_negative_data_has_only_read_noise()
    test_matches_simulated_detector()
    print("All make_im variance tests passed")
