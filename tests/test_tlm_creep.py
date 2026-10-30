"""
Recovery tests for kspecdr.tlm.tlm_creep.measure_tlm_shift on synthetic frames.

Run from kspecdr/: python tests/test_tlm_creep.py
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("tests"))

from kspecdr.tlm.tlm_creep import measure_tlm_shift
from test_optimal_extract import make_frame

logging.basicConfig(level=logging.INFO)


def check(shift, level=2000.0, bg=30.0, pitch=12.0, init=0.0, tol=0.02, seed=3):
    """Frame drawn with traces at tlm + shift; the map handed to the measurement is tlm."""
    img, var, tlm, sig, amp, _ = make_frame(pitch, seed=seed, level=level, bg=bg)
    # make_frame draws at tlm: give the measurement a map displaced by -shift instead
    s, info = measure_tlm_shift(img, var, tlm - shift, sig, init_shift=init)
    print(f"shift {shift:+.2f} (level {level}, pitch {pitch}, init {init:+.1f}): measured "
          f"{s:+.4f} +- {info['err']:.4f}, rms {info['rms']:.3f}, "
          f"{info['nused']}/{info['nmeas']} blocks, slope {info['slope']:+.3f}")
    assert abs(s - shift) < tol, s
    assert abs(s - shift) < 5 * info["err"] + 0.005
    return s, info


def test_shifts():
    for shift in (0.0, 0.7, -1.3):
        check(shift)


def test_faint_frame():
    """Sky-like frame: peaks of 0.2-50 ADU on a 30 ADU background."""
    check(0.6, level=150.0, tol=0.05)


def test_dense_slit():
    check(0.5, pitch=6.0)


def test_large_shift_with_init():
    check(3.2, init=3.0)


def test_no_signal_fails():
    rng = np.random.default_rng(0)
    img, var, tlm, sig, amp, _ = make_frame(12.0)
    noise = rng.standard_normal(img.shape) * 4.5
    s, info = measure_tlm_shift(noise, np.full(img.shape, 4.5**2), tlm, sig)
    print(f"noise only: shift {s}, {info['nmeas']} block profiles detected")
    assert np.isnan(s)


if __name__ == "__main__":
    test_shifts()
    test_faint_frame()
    test_dense_slit()
    test_large_shift_with_init()
    test_no_signal_fails()
    print("All tlm_creep tests passed")
