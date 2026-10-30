"""
Recovery test for kspecdr.tlm.sigma_profile on synthetic fibre flats.

Run from kspecdr/: python tests/test_sigma_profile.py
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from kspecdr.tlm.sigma_profile import pixel_gaussian, sigma_profile

logging.basicConfig(level=logging.INFO)


def make_flat(pitch, nfib=20, nspat=300, nspec=600, seed=1, peak=5000.0, rdnoise=4.5):
    """Flat with tilted traces, sigma varying along x, a sloped background and bad pixels."""
    rng = np.random.default_rng(seed)
    x = np.arange(nspec)
    xn = (x - 0.5 * (nspec - 1)) / (0.5 * (nspec - 1))
    y0 = 0.5 * nspat - 0.5 * pitch * (nfib - 1)
    tlm = y0 + pitch * np.arange(nfib)[:, None] + 3.0 * xn[None, :] ** 2 + 0.01 * x[None, :]
    sig_true = (1.25 + 0.15 * xn**2)[None, :] * (1.0 + 0.03 * rng.standard_normal((nfib, 1)))
    flux = peak * np.sqrt(2 * np.pi) * sig_true * (0.6 + 0.4 * np.cos(xn)[None, :])
    y = np.arange(nspat)[:, None]
    img = 20.0 + 0.02 * y + 0.0 * x[None, :]
    for f in range(nfib):
        img = img + flux[f][None, :] * pixel_gaussian(y, tlm[f][None, :], sig_true[f][None, :])
    var = np.clip(img, 0, None) + rdnoise**2
    img = img + rng.standard_normal(img.shape) * np.sqrt(var)
    bad = rng.random(img.shape) < 1e-3
    img[bad] = np.nan
    var[bad] = np.nan
    return img, var, tlm, sig_true


def check(pitch, tol):
    img, var, tlm, sig_true = make_flat(pitch)
    sigma, info = sigma_profile(img, tlm, var, col_step=20)
    rel = sigma / sig_true - 1.0
    rms = np.sqrt(np.mean(rel**2))
    print(f"pitch {pitch}: median rel err {np.median(rel):+.4f}, rms {rms:.4f}, "
          f"max |rel| {np.abs(rel).max():.4f}, "
          f"MWIDTH {info['mwidth']:.3f} (true {2.3548 * np.median(sig_true):.3f})")
    assert info["fitted"].all()
    assert abs(np.median(rel)) < tol, np.median(rel)
    assert rms < tol, rms
    # The worst pixel sits at a frame edge, where the cubic is least constrained
    assert np.abs(rel).max() < 4 * tol, np.abs(rel).max()


def test_sigma_profile_isolated():
    check(pitch=12.0, tol=0.005)


def test_sigma_profile_dense():
    check(pitch=6.0, tol=0.01)


def test_unmeasured_fibre_gets_median():
    img, var, tlm, sig_true = make_flat(12.0)
    measured = np.ones(len(tlm), dtype=bool)
    measured[5] = False
    sigma, info = sigma_profile(img, tlm, var, measured=measured)
    assert not info["fitted"][5]
    assert np.allclose(sigma[5], np.median(sigma[info["fitted"]], axis=0))


if __name__ == "__main__":
    test_sigma_profile_isolated()
    test_sigma_profile_dense()
    test_unmeasured_fibre_gets_median()
    print("All sigma_profile tests passed")
