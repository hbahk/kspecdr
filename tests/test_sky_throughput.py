"""
Tests for the sky-line throughput methods of kspecdr.reduce_object
(``SKYLINE(TEMPLATE)`` and the L1 fit of ``SKYLINE(KGB)``).

Run from kspecdr/: python tests/test_sky_throughput.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from kspecdr.reduce_object import _get_thput_kgb, _get_thput_template, _l1_line

STARS = [9, 26, 27, 29, 34]  # fibers with a bright star


def synthetic(seed=0, nabs=0, dmax=0.15):
    """43 fibers: sky lines on a moonlit continuum, stars in 5 fibers.

    The continuum (moonlight and starlight alike) has the telluric A band (7590-7700 Å) and
    ``nabs`` absorption lines of depth up to ``dmax``; the starlight is 5x the sky continuum,
    as for bright stars on IsoPlane tile frames. Returns spectra, wavelength, types and the true throughput
    (median 1).
    """
    rng = np.random.default_rng(seed)
    nfib, npix = 43, 1340
    wave = np.linspace(3600.0, 8900.0, npix)
    lines = rng.uniform(5000.0, 8850.0, 60)
    amp = rng.uniform(100.0, 600.0, 60)
    emis = sum(a * np.exp(-0.5 * ((wave - c) / 6.0) ** 2) for c, a in zip(lines, amp))
    # weak absorption lines (solar-type, also in the moonlight) and the A band
    absn = sum(
        d * np.exp(-0.5 * ((wave - c) / 8.0) ** 2)
        for c, d in zip(
            rng.uniform(3700.0, 8850.0, nabs), rng.uniform(0.02, dmax, nabs)
        )
    )
    cont = (1.0 - np.clip(absn, 0.0, 0.9)) * (
        1.0 - 0.8 * ((wave > 7590.0) & (wave < 7700.0))
    )
    t = rng.uniform(0.6, 1.2, nfib)
    t /= np.median(t)
    types = np.array(["P"] * nfib)
    types[[14, 21, 31, 40, 41, 42]] = "S"
    spec = t[:, None] * (emis + 300.0 * cont)[None, :]
    for i in STARS:
        spec[i] += t[i] * 1500.0 * cont
    spec += rng.normal(0.0, 5.0, spec.shape)
    return spec, np.tile(wave, (nfib, 1)), types, t


def test_l1_line():
    """The L1 line ignores 20 % gross outliers that pull a least-squares slope away."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 500)
    y = 0.3 + 1.7 * x + rng.laplace(0, 0.01, 500)
    y[:100] += 5.0 * x[:100] + 3.0
    assert abs(_l1_line(x, y) - 1.7) < 0.02


def test_template_recovers_throughput():
    """Template fit: 1 % on every fiber, star fibers (A band, no line forest) included."""
    spec, wave, types, t = synthetic()
    tp, err, npx = _get_thput_template(spec, wave, types)
    assert npx >= 20
    assert np.max(np.abs(tp / t - 1)) < 0.01, np.max(np.abs(tp / t - 1))
    assert np.all(err > 0) and np.all(err < 0.01)


def test_kgb_star_bias():
    """KGB fits every pixel: with the stars' A band and line forest, star fibers come out
    high; sky-only fibers agree."""
    spec, wave, types, t = synthetic(nabs=150)
    tp = _get_thput_kgb(spec, types)
    rest = np.setdiff1d(np.arange(len(t)), STARS)
    assert np.all(tp[STARS] / t[STARS] > 1.04), tp[STARS] / t[STARS]
    assert np.median(np.abs(tp[rest] / t[rest] - 1)) < 0.02


def test_template_nan_and_no_sky():
    """NaN pixels are tolerated; no sky fibers -> None."""
    spec, wave, types, t = synthetic()
    spec[3, 500:520] = np.nan
    tp, _, _ = _get_thput_template(spec, wave, types)
    assert abs(tp[3] / t[3] - 1) < 0.01
    assert _get_thput_template(spec, wave, np.array(["P"] * len(t))) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
