"""
Recovery tests for kspecdr.extract.optimal on synthetic multi-fibre frames.

Run from kspecdr/: python tests/test_optimal_extract.py
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from kspecdr.extract.optimal import optimal_extract
from kspecdr.tlm.sigma_profile import pixel_gaussian

logging.basicConfig(level=logging.INFO)

GAIN = 1.12
RDNOISE = 4.5  # ADU


def make_frame(pitch, nfib=14, nspat=260, nspec=400, seed=2, level=2000.0, bg=30.0,
               ncr=0, noise=True):
    """Frame with tilted traces, per-fibre sigma along x, a sloped background, NaN pixels."""
    rng = np.random.default_rng(seed)
    x = np.arange(nspec)
    xn = (x - 0.5 * (nspec - 1)) / (0.5 * (nspec - 1))
    y0 = 0.5 * nspat - 0.5 * pitch * (nfib - 1)
    tlm = y0 + pitch * np.arange(nfib)[:, None] + 2.0 * xn[None, :] ** 2 + 0.005 * x[None, :]
    sig = (1.2 + 0.15 * xn**2)[None, :] * (1.0 + 0.05 * rng.standard_normal((nfib, 1)))
    # Fluxes spanning bright to read-noise limited, varying along x
    amp = level * np.geomspace(1.0, 0.003, nfib)[:, None] * (0.7 + 0.3 * np.cos(3 * xn))[None, :]
    y = np.arange(nspat)[:, None]
    img = bg * (1.0 + 0.3 * (y - nspat / 2) / nspat) + 0.0 * x[None, :]
    for f in range(nfib):
        img = img + amp[f][None, :] * pixel_gaussian(y, tlm[f][None, :], sig[f][None, :])
    truth = img.copy()
    var = RDNOISE**2 + np.maximum(truth, 0) / GAIN
    if noise:
        img = img + rng.standard_normal(img.shape) * np.sqrt(var)
    var = RDNOISE**2 + np.maximum(img, 0) / GAIN  # as make_im: from the noisy data
    crs = []
    for _ in range(ncr):
        f, col = rng.integers(nfib), rng.integers(nspec)
        row = int(round(tlm[f, col] + rng.uniform(-1.5, 1.5)))
        img[row, col] += rng.uniform(300, 3000)
        crs.append((f, col))
    bad = rng.random(img.shape) < 1e-3
    img[bad] = np.nan
    var[bad] = np.nan
    return img, var, tlm, sig, amp, crs


def pulls(flux, fvar, amp):
    z = (flux - amp) / np.sqrt(fvar)
    return z[np.isfinite(z)]


def bias(flux, amp):
    """Mean error per fibre in units of its scatter (independent of the variance estimate)."""
    e = flux - amp
    return np.nanmean(e, axis=1) / np.nanstd(e, axis=1)


def test_noiseless_exact():
    """Without noise the model is exact: amplitudes to 1e-6 relative."""
    img, var, tlm, sig, amp, _ = make_frame(12.0, noise=False)
    flux, fvar, info = optimal_extract(img, var, tlm, sig, nslpars=2)
    good = np.isfinite(flux)
    rel = np.abs(flux[good] / amp[good] - 1.0)
    print(f"noiseless: max rel err {rel.max():.2e}, unsolved {np.sum(~good)}")
    assert good.mean() > 0.99
    assert rel.max() < 1e-6


def check_pulls(pitch, label):
    img, var, tlm, sig, amp, _ = make_frame(pitch)
    flux, fvar, info = optimal_extract(img, var, tlm, sig, nslpars=2, gain=GAIN, niter=2)
    z = pulls(flux, fvar, amp)
    b = bias(flux, amp)
    print(f"{label}: pull std {z.std():.3f}, bias/scatter mean {b.mean():+.3f} "
          f"(worst fibre {b[np.argmax(np.abs(b))]:+.3f}), median chi2 "
          f"{np.nanmedian(info['chi2']):.3f}")
    assert abs(z.std() - 1.0) < 0.05, z.std()
    assert abs(b.mean()) < 0.04, b.mean()


def test_isolated_pulls():
    check_pulls(12.0, "pitch 12")


def test_dense_pulls():
    """Fibres 3 sigma apart overlap strongly; the joint fit still has unit pulls."""
    check_pulls(3.6, "pitch 3.6")


def test_data_variance_bias():
    """One fit with the data variance (2dfdr) is biased high on faint fibres; a refit with
    the model variance is not."""
    bd, bm = [], []
    for seed in range(5):
        img, var, tlm, sig, amp, _ = make_frame(12.0, seed=seed)
        f1, _, _ = optimal_extract(img, var, tlm, sig, nslpars=2)
        f2, _, _ = optimal_extract(img, var, tlm, sig, nslpars=2, gain=GAIN, niter=2)
        bd.append(bias(f1, amp)[7:])
        bm.append(bias(f2, amp)[7:])
    bd, bm = np.mean(bd), np.mean(bm)
    print(f"faint fibres, bias/scatter: data variance {bd:+.3f}, model variance {bm:+.3f}")
    assert bd > 0.03
    assert abs(bm) < 0.02


def test_cosmic_rays():
    img, var, tlm, sig, amp, crs = make_frame(12.0, ncr=60)
    f0, v0, _ = optimal_extract(img, var, tlm, sig, nslpars=2, gain=GAIN, niter=2)
    f1, v1, info = optimal_extract(img, var, tlm, sig, nslpars=2, gain=GAIN, niter=4, clip=5.0)
    fib, col = np.array(crs).T
    z0 = np.abs((f0 - amp) / np.sqrt(v0))[fib, col]
    z1 = np.abs((f1 - amp) / np.sqrt(v1))[fib, col]
    zall = pulls(f1, v1, amp)
    print(f"cosmic rays: median |pull| at CRs {np.nanmedian(z0):.1f} -> {np.nanmedian(z1):.2f}; "
          f"{info['nrej'].sum()} pixels rejected for {len(crs)} CRs; "
          f"all pulls std {zall.std():.3f}")
    assert np.nanmedian(z1) < 1.0
    assert np.nanmedian(z0) > 3.0
    assert abs(zall.std() - 1.0) < 0.05
    # Clean pixels are hardly touched
    assert info["nrej"].sum() <= len(crs) + 5


def test_knots_background():
    img, var, tlm, sig, amp, _ = make_frame(12.0)
    flux, fvar, info = optimal_extract(img, var, tlm, sig, slmodel="KNOTS", gain=GAIN, niter=2)
    z = pulls(flux, fvar, amp)
    print(f"KNOTS background: pull mean {z.mean():+.3f}, std {z.std():.3f}")
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05


def test_model_image():
    img, var, tlm, sig, amp, _ = make_frame(12.0)
    flux, fvar, info = optimal_extract(img, var, tlm, sig, nslpars=2, gain=GAIN, niter=2,
                                       return_model=True)
    y0, y1 = info["rows"]
    model = info["model"]
    r = (img - model) / np.sqrt(RDNOISE**2 + np.maximum(model, 0) / GAIN)
    r = r[y0 : y1 + 1][np.isfinite(r[y0 : y1 + 1])]
    print(f"model image: residual/sigma mean {r.mean():+.3f}, std {r.std():.3f}")
    # 16 parameters on ~180 rows per column: std sqrt(1 - 16/180) = 0.95
    assert abs(r.mean()) < 0.02 and 0.92 < r.std() < 1.0


if __name__ == "__main__":
    test_noiseless_exact()
    test_isolated_pulls()
    test_dense_pulls()
    test_data_variance_bias()
    test_cosmic_rays()
    test_knots_background()
    test_model_image()
    print("All optimal_extract tests passed")
