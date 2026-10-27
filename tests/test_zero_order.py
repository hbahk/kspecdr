"""Zero-order fibre photometry (extract/zero_order.py) on synthetic frames.

A zero-order flat (all fibres lit) is measured in detection mode and becomes the reference
for a star frame (three fibres lit, slit image shifted), measured in reference mode.
Run from kspecdr/: python tests/test_zero_order.py
"""

import logging
import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.append(os.path.abspath("src"))

from kspecdr.extract.zero_order import (  # noqa: E402
    ZOFLAG_NODETECT,
    make_zo,
    measure_zero_order,
)
from kspecdr.tlm.match_fibers import isoplane_nominal_fibpos  # noqa: E402

logging.basicConfig(level=logging.INFO)

NY, NX = 1300, 1340
NF = 20
PITCH = 12.0
GAPS = {11: 12.0}  # blank slit slot before fibre 11
SIGMA = 1.5  # spot Gaussian sigma (px)
BKG = 50.0  # sky / scattered light per pixel
RN2 = 25.0  # read-noise variance


def _fibre_xy(y1=900.0, x_at_y1=640.3, tilt=0.004, dx=0.0, dy=0.0):
    """Spot centres: fibre 1 at the highest y, the column slightly tilted."""
    off = isoplane_nominal_fibpos(NF, pitch=PITCH, gaps=GAPS)
    y = y1 - off + dy
    x = x_at_y1 + tilt * (y - y1) + dx
    return x, y


def _render(x, y, flux, seed):
    """Image and variance of Gaussian spots on a flat background, Gaussian noise."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    model = np.full((NY, NX), BKG)
    for xi, yi, fi in zip(x, y, flux):
        if fi <= 0:
            continue
        r0, r1 = int(yi) - 10, int(yi) + 11
        c0, c1 = int(xi) - 10, int(xi) + 11
        g = np.exp(-0.5 * (((xx[r0:r1, c0:c1] - xi) / SIGMA) ** 2
                           + ((yy[r0:r1, c0:c1] - yi) / SIGMA) ** 2))
        model[r0:r1, c0:c1] += fi * g / (2 * np.pi * SIGMA**2)
    var = model + RN2
    rng = np.random.default_rng(seed)
    data = model + rng.normal(0.0, np.sqrt(var))
    return data, var


def _write_im(path, data, var):
    fib = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TYPE", format="1A", array=np.array(["P"] * NF)),
            fits.Column(name="NAME", format="20A",
                        array=np.array([f"F{i + 1}" for i in range(NF)])),
        ],
        name="FIBRES",
    )
    hdr = fits.Header()
    hdr["LAMBDAC"] = 0.0
    hdr["EXPTIME"] = 30.0
    fits.HDUList(
        [fits.PrimaryHDU(data.astype(np.float32), header=hdr),
         fits.ImageHDU(var.astype(np.float32), name="VARIANCE"),
         fib]
    ).writeto(path, overwrite=True)


def _report(label, ok):
    print(f"{'SUCCESS' if ok else 'FAIL'} [{label}]")
    return ok


def test_flat_and_star_frames():
    tmp = tempfile.mkdtemp()
    ok = True

    # Zero-order flat: all fibres lit, fluxes 2e4..4e4
    x, y = _fibre_xy()
    flux = np.linspace(2e4, 4e4, NF)
    data, var = _render(x, y, flux, seed=1)
    flat_im = os.path.join(tmp, "flat_im.fits")
    _write_im(flat_im, data, var)
    args = {"IMAGE_FILENAME": flat_im, "ISO_FIB_PITCH": PITCH, "ISO_FIB_GAPS": GAPS}
    flat = make_zo(args)
    flat_zo = flat_im.replace("_im.fits", "_zo.fits")

    ok &= _report("flat: all fibres detected", bool(np.all(flat["DETECTED"])))
    ok &= _report("flat: y positions < 0.05 px",
                  bool(np.max(np.abs(flat["YPOS"] - y)) < 0.05))
    ok &= _report("flat: tilted column x < 0.05 px",
                  bool(np.max(np.abs(flat["XPOS"] - x)) < 0.05))
    # Box aperture +-6 px in x, +-6 px in y holds erf(4)^2 of a sigma-1.5 Gaussian = 1.0000
    pull = (flat["FLUX"] - flux) / np.sqrt(flat["FLUX_VAR"])
    ok &= _report(f"flat: flux pulls within 4 sigma (max {np.max(np.abs(pull)):.2f})",
                  bool(np.max(np.abs(pull)) < 4))

    hdr = fits.getheader(flat_zo, "ZOFLUX")
    ok &= _report("flat: product header", hdr["ZOMETHOD"] == "DETECT"
                  and fits.getheader(flat_zo)["EXPTIME"] == 30.0)

    # Star frame: three fibres lit, slit image shifted by (0.7, -1.3) px
    xs, ys = _fibre_xy(dx=0.7, dy=-1.3)
    sflux = np.zeros(NF)
    lit = [2, 9, 15]  # 0-based
    sflux[lit] = [8e3, 2e3, 4e2]  # the faintest (S/N ~ 4) is below the detection threshold
    data, var = _render(xs, ys, sflux, seed=2)
    star_im = os.path.join(tmp, "star_im.fits")
    _write_im(star_im, data, var)
    star = make_zo({"IMAGE_FILENAME": star_im, "ZO_REF_FILENAME": flat_zo})

    det = np.flatnonzero(star["DETECTED"]).tolist()
    ok &= _report(f"star: detected fibres {det} include {lit[:2]}, all lit",
                  set(lit[:2]) <= set(det) <= set(lit))
    meta = fits.getheader(star_im.replace("_im.fits", "_zo.fits"), "ZOFLUX")
    ok &= _report(f"star: shift ({meta['ZODX']:.2f}, {meta['ZODY']:.2f}) ~ (0.7, -1.3)",
                  abs(meta["ZODX"] - 0.7) < 0.1 and abs(meta["ZODY"] + 1.3) < 0.1)
    spull = (star["FLUX"] - sflux) / np.sqrt(star["FLUX_VAR"])
    ok &= _report(f"star: flux pulls within 4 sigma (max {np.max(np.abs(spull)):.2f})",
                  bool(np.max(np.abs(spull)) < 4))
    dark = np.setdiff1d(np.arange(NF), lit)
    ok &= _report("star: dark fibres flagged NODETECT",
                  bool(np.all(star["ZOFLAG"][dark] & ZOFLAG_NODETECT)))
    return ok


def test_variance_calibration():
    # Blank frames measured at fixed apertures: flux / sigma should have unit scatter
    x, y = _fibre_xy()
    ref = {"XPOS": x, "YPOS": y}
    from astropy.table import Table

    reference = Table(ref, meta={"ZOPITCH": PITCH, "FILENAME": "synthetic"})
    pulls = []
    for seed in range(20):
        data, var = _render(x, y, np.zeros(NF), seed=100 + seed)
        t = measure_zero_order(data, var, reference=reference,
                               args={"ZO_FIX_POSITIONS": True})
        pulls.append(np.asarray(t["FLUX"] / np.sqrt(t["FLUX_VAR"])))
    pulls = np.concatenate(pulls)
    std = float(np.std(pulls))
    return _report(f"blank: std of flux / sigma = {std:.2f} (expect ~1)", 0.85 < std < 1.15)


if __name__ == "__main__":
    results = [test_flat_and_star_frames(), test_variance_calibration()]
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
