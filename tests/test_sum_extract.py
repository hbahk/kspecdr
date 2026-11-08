"""
Tests for kspecdr.extract.make_ex.sum_extract (SUM extraction window and variance).

Run from kspecdr/: python tests/test_sum_extract.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from kspecdr.extract.make_ex import sum_extract
from kspecdr.tlm.sigma_profile import pixel_gaussian


def run(img, var, tlm, width):
    nspat, nspec = img.shape
    nfib = tlm.shape[1]
    out = np.zeros((nspec, nfib))
    outv = np.zeros((nspec, nfib))
    sum_extract(nspat, nspec, img, var, out, outv, nfib, tlm, width)
    return out, outv


def test_uniform_window():
    """A flat image of 1 sums to the window width; variance = full pixels + fractions^2."""
    img = np.ones((60, 4))
    var = np.ones((60, 4))
    tlm = np.array([[20.0], [20.3], [20.5], [31.77]])
    out, outv = run(img, var, tlm, 10.0)
    assert np.allclose(out[:, 0], 10.0), out
    # centre 20.3: window [15.3, 25.3]; pixels 16..24 full (9), 15 covers 0.8, 25 covers 0.2
    assert np.isclose(outv[1, 0], 9 + 0.8**2 + 0.2**2), outv[1, 0]
    print("uniform window ok:", out[:, 0], outv[:, 0])


def test_window_centred_on_trace():
    """A Gaussian fibre leaks equally into windows at +-pitch (no half-pixel offset).

    At a given sub-pixel phase the two leaks differ (a partial pixel is split linearly
    across a steep profile edge), so the symmetry is checked on the phase average; with
    pixel k taken as [k, k+1] (the old convention) the averages were 17.6 % and 4.4 %.
    """
    y = np.arange(80)[:, None]
    sigma, pitch = 1.29, 6.7
    leaks = []
    for c in 40.0 + np.arange(20) / 20:
        img = pixel_gaussian(y, c, sigma) * np.ones((1, 3))
        tlm = np.tile([c - pitch, c, c + pitch], (3, 1))
        out, _ = run(img, np.ones_like(img), tlm, 10.0)
        lower, mid, upper = out[0]
        assert mid > 0.999, mid
        leaks.append((lower, upper))
    lower, upper = np.mean(leaks, axis=0)
    # one-sided tail beyond pitch - width / 2 = 1.7 px: 9.4 % for the bare Gaussian, ~10 %
    # with pixel integration and the linear split of the edge pixels
    print(f"phase-averaged leak {lower:.4f} / {upper:.4f}")
    assert abs(lower - upper) < 2e-3, (lower, upper)
    assert 0.09 < lower < 0.115, lower
    # integer centre: exactly symmetric
    assert abs(leaks[0][0] - leaks[0][1]) < 1e-9, leaks[0]


def test_bad_pixels_and_edges():
    img = np.ones((30, 3))
    img[12, 1] = np.nan
    tlm = np.array([[15.0], [15.0], [-20.0]])
    out, outv = run(img, np.ones_like(img), tlm, 10.0)
    assert np.isfinite(out[0, 0]) and np.isnan(out[1, 0]) and np.isnan(out[2, 0])
    # window half off the detector: only the on-chip part is summed
    out, _ = run(np.ones((30, 1)), np.ones((30, 1)), np.array([[1.0]]), 10.0)
    # centre 1: window [-4, 6] -> rows 0-5 whole (row 0 spans [-0.5, 0.5]) + half of row 6
    assert np.isclose(out[0, 0], 6.5), out
    out, _ = run(np.ones((30, 1)), np.ones((30, 1)), np.array([[np.nan]]), 10.0)
    assert np.isnan(out[0, 0])
    print("bad pixels and edges ok")


if __name__ == "__main__":
    test_uniform_window()
    test_window_centred_on_trace()
    test_bad_pixels_and_edges()
    print("all passed")
