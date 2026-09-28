"""Check the cross-correlogram path search against a direct transcription of the 2dfdr loops.

Run from the repository root: ``python tests/test_crosscorr_path_search.py``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))  # this checkout, not an installed kspecdr

from kspecdr.wavecal.crosscorr import cross_corr_greedy_quad_path_search


def brute_force_path(crs_cgm, nrows, npix):
    """2dfdr CrossCorrGreedyQuadPathSearch loops (nodes on row centres), first maximum kept."""
    x = (np.arange(npix) + 0.5) / (npix - 0.5)
    x0, x1, x2 = x[0], (npix / 2.0 - 0.5) / (npix - 0.5), x[-1]
    a0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    a1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    a2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
    maxsum, best = 0.0, None
    for y0 in range(nrows):
        for y2 in range(nrows):
            for y1 in range(nrows):
                yv = a0 * y0 + a1 * y1 + a2 * y2
                rows = np.floor(yv + 0.5).astype(int)
                inside = (rows >= 0) & (rows < nrows)
                vals = crs_cgm[rows[inside], np.nonzero(inside)[0]]
                total = vals[vals >= 0.5].sum()
                if total > maxsum:
                    maxsum, best = total, yv
    return best


def test_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(20):
        nrows = int(rng.integers(1, 6)) * 2 + 1
        npix = int(rng.integers(10, 40))
        crs = rng.uniform(-0.3, 1.0, (nrows, npix)) * (rng.uniform(size=(nrows, npix)) < 0.4)
        expected = brute_force_path(crs, nrows, npix)
        got = cross_corr_greedy_quad_path_search(crs, nrows, npix)
        assert np.allclose(got, expected), (nrows, npix)
    print("random correlograms: path equals the brute-force search")


def test_recovers_quadratic_ridge():
    maxshift, npix = 20, 400
    nrows = 2 * maxshift + 1
    pix = np.arange(npix)
    true_shift = 3.0 + 6.0 * (pix / npix - 0.5) ** 2 - 4.0 * (pix / npix)
    crs = np.exp(-0.5 * ((np.arange(nrows)[:, None] - maxshift - true_shift) / 1.5) ** 2)
    path = cross_corr_greedy_quad_path_search(crs, nrows, npix)
    dev = np.abs(path - maxshift - true_shift).max()
    assert dev < 1.0, dev
    again = cross_corr_greedy_quad_path_search(crs, nrows, npix)
    assert np.array_equal(path, again)
    print(f"quadratic ridge: max deviation {dev:.2f} rows, repeat identical")


def test_empty_correlogram():
    path = cross_corr_greedy_quad_path_search(np.zeros((9, 30)), 9, 30)
    assert np.all(path == 4.0)
    print("empty correlogram: zero-shift path")


if __name__ == "__main__":
    test_matches_brute_force()
    test_recovers_quadratic_ridge()
    test_empty_correlogram()
    print("OK")
