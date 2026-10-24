import numpy as np
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath('src'))

from kspecdr.tlm.make_tlm import (
    _wavelet_peak_detection,
    _refine_peaks_gauss3,
    _fit_traces_joint,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIG = 3.5 / 2.355  # January flat profile, FWHM 3.5 px


def test_refinement_bias():
    # Centroid bias with and without a neighbour must stay well below the 0.2-0.5 px of the
    # wavelet midpoint
    y = np.arange(400, dtype=float)
    rng = np.random.default_rng(0)
    ok = True
    for pitch in (12, 10):
        for drop in (None, 1):
            bias = []
            for c0 in rng.uniform(200, 201, 20):
                cs = [c0 + k * pitch for k in range(-5, 6) if k != drop]
                col = sum(1000 * np.exp(-0.5 * ((y - c) / SIG) ** 2) for c in cs)
                col = col + rng.normal(0, 3, y.size) + 20
                p = _refine_peaks_gauss3(col, _wavelet_peak_detection(col, max_peaks=len(cs)))
                bias.append(p[np.argmin(np.abs(p - c0))] - c0)
            b = abs(np.mean(bias))
            print(f"pitch {pitch}, neighbour removed {drop is not None}: bias {b:.3f} px")
            if b > 0.02:
                print("FAIL: refined centroid biased")
                ok = False
    return ok


def test_joint_fit_extrapolation():
    # 14 parallel curved traces, points only at x >= 400: the joint fit must keep the
    # extrapolated x = 0 end parallel (pitch scatter << 0.1 px)
    rng = np.random.default_rng(1)
    x_steps = np.arange(1, 1341, 50) - 0.5
    x_fit = np.arange(1, 1341) - 0.5
    y0 = 573.0 + 12.0 * np.arange(14)
    curve = 2.0e-6 * (x_steps - 900.0) ** 2 - 1.0e-3 * x_steps
    tp = y0[:, None] + curve[None, :] + rng.normal(0, 0.03, (14, x_steps.size))
    tp[:, x_steps < 400] = 0.0
    out = _fit_traces_joint(x_steps, tp, 2, x_fit)
    if out is None:
        print("FAIL: joint fit returned None")
        return False
    sd = np.diff(out[0, :]).std()
    true0 = y0 + 2.0e-6 * (x_fit[0] - 900.0) ** 2 - 1.0e-3 * x_fit[0]
    err = np.abs(out[0, :] - true0).max()
    print(f"joint fit: pitch scatter at x=0 {sd:.3f} px, max error {err:.3f} px")
    if sd > 0.05:
        print("FAIL: extrapolated traces not parallel")
        return False
    print("SUCCESS")
    return True


if __name__ == "__main__":
    results = [test_refinement_bias(), test_joint_fit_extrapolation()]
    sys.exit(0 if all(results) else 1)
