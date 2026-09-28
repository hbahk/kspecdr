"""Check the arc-fit outlier rejection (2dfdr CALIBRATE_SPECTRAL_AXES step 8, L1 fit).

Run from the repository root: ``python tests/test_arc_fit_outliers.py``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))  # this checkout, not an installed kspecdr

from kspecdr.wavecal.calibrate import fit_calibration_model, l1_outliers

TRUE = np.array([4.0e-8, -1.5e-4, 2.0, 3600.0])  # wavelength (A) against pixel, ~2 A/px


def lines(pix, errors=None, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    wave = np.polyval(TRUE, pix) + rng.normal(0.0, noise, pix.size)
    if errors is not None:
        wave = wave + errors
    return pix.astype(float), wave


def test_misidentified_lines_only():
    # few lines in the blue, a cluster in the red, two misidentified (3.5 A and -1.2 A)
    pix = np.array([220, 450, 540, 600, 750, 915, 945, 1047, 1118, 1150, 1165, 1198, 1243,
                    1266, 1302, 1313])
    err = np.zeros(pix.size)
    err[1], err[7] = 3.5, -1.2
    x, y = lines(pix, err)
    coeffs, resid, out = fit_calibration_model(x, y, 3)
    assert np.array_equal(np.nonzero(out)[0], [1, 7]), np.nonzero(out)[0]
    dev = np.abs(np.polyval(coeffs, np.arange(1340.0)) - np.polyval(TRUE, np.arange(1340.0)))
    assert dev.max() < 0.5, dev.max()
    print(f"two misidentified lines rejected, others kept; solution within {dev.max():.2f} A")


def test_few_lines_kept():
    for n in (5, 6, 7):
        x, y = lines(np.linspace(150, 1250, n), seed=n)
        out = l1_outliers(x, y, 3)
        assert not out.any(), (n, np.nonzero(out)[0])
    x, y = lines(np.array([100.0, 500.0, 900.0, 1300.0]))
    assert not l1_outliers(x, y, 3).any()
    print("5-7 good lines: none rejected; 4 lines: no rejection")


if __name__ == "__main__":
    test_misidentified_lines_only()
    test_few_lines_kept()
    print("OK")
