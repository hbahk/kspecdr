import numpy as np
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath('src'))

from kspecdr.tlm.match_fibers import isoplane_nominal_fibpos, match_fibers_isoplane

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_traces(offsets, y0=900.0, scale=1.0, dead=(), spurious=(), noise=0.2, seed=1):
    """Detected trace positions (ascending) and the true fibre -> position map."""
    rng = np.random.default_rng(seed)
    nf = len(offsets)
    true_y = y0 - scale * offsets + rng.normal(0.0, noise, nf)
    lit = [i for i in range(nf) if (i + 1) not in dead]
    y = np.sort(np.concatenate([true_y[lit], np.asarray(spurious, float)]))
    return y, true_y, lit


def _check(match_vector, y, true_y, lit, nf, label):
    ok = True
    for i in range(nf):
        if i in lit:
            if match_vector[i] == 0 or abs(y[match_vector[i] - 1] - true_y[i]) > 1e-9:
                print(f"FAIL [{label}]: fibre {i + 1} matched to trace {match_vector[i]}")
                ok = False
        elif match_vector[i] != 0:
            print(f"FAIL [{label}]: dead fibre {i + 1} matched to trace {match_vector[i]}")
            ok = False
    if ok:
        print(f"SUCCESS [{label}]")
    return ok


def test_all_lit_january():
    # 14 fibres, all lit: must reproduce the old ordering (fibre 1 = highest-y trace)
    off = isoplane_nominal_fibpos(14, pitch=12.0)
    y, true_y, lit = _make_traces(off, y0=730.0)
    mv, _ = match_fibers_isoplane(14, y, off)
    ok = _check(mv, y, true_y, lit, 14, "january 14 fibres")
    if not np.array_equal(mv, np.arange(14, 0, -1)):
        print("FAIL [january 14 fibres]: differs from the 1-to-1 ordering")
        ok = False
    return ok


def test_dead_and_spurious_with_gap():
    # 42 fibres with a blank slot before fibre 22, 3% scale error, dead fibres, a stray peak
    off = isoplane_nominal_fibpos(42, pitch=10.0, gaps={22: 10.0})
    y, true_y, lit = _make_traces(
        off, y0=1000.0, scale=1.03, dead=(1, 7, 30), spurious=(805.0,)
    )
    mv, model = match_fibers_isoplane(42, y, off)
    ok = _check(mv, y, true_y, lit, 42, "42 fibres, gap, dead 1/7/30, spurious")
    # Dead fibres get a model position close to where they should be
    for fib in (1, 7, 30):
        if abs(model[fib - 1] - true_y[fib - 1]) > 1.0:
            print(f"FAIL: model position of dead fibre {fib} off by "
                  f"{model[fib - 1] - true_y[fib - 1]:.2f} px")
            ok = False
    return ok


def test_uniform_slit_needs_prior():
    # Uniform slit with fibre 1 dead is ambiguous by one pitch; the y prior resolves it
    off = isoplane_nominal_fibpos(40, pitch=12.0)
    y, true_y, lit = _make_traces(off, y0=1000.0, dead=(1,))
    mv, _ = match_fibers_isoplane(40, y, off, fib1_ypos=1003.0)
    return _check(mv, y, true_y, lit, 40, "uniform slit, fibre 1 dead, prior")


if __name__ == "__main__":
    results = [
        test_all_lit_january(),
        test_dead_and_spurious_with_gap(),
        test_uniform_slit_needs_prior(),
    ]
    sys.exit(0 if all(results) else 1)
