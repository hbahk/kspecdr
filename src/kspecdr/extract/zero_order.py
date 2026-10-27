"""
Fibre photometry on zero-order Isoplane images.

With the grating at zero order (LightField centre wavelength 0 nm) the Isoplane images its
slit: every fibre is a spot on the zero-order column, at its slit position along y. This
module measures the light of each fibre in such an image (a ``*_im.fits`` from ``make_im``),
for focal-plane mapping: positioners step around a star and the fibre flux is compared
between positions, which locates the star in focal-plane x, y.

There is no 2dfdr counterpart. The step follows the 2dfdr conventions all the same:
``make_zo(args)`` with UPPERCASE keys, fibres identified as in ``make_tlm``
(``match_fibers_isoplane`` and the same ``ISO_FIB_*`` keys), variance propagated from the
``VARIANCE`` HDU.

The fibre positions on the detector come from one of two sources:

- **detection** (default): spots are found along the zero-order column and matched to the
  slit fibres. This needs most fibres lit, as in a zero-order flat (lamp or dome).
- **reference** (``ZO_REF_FILENAME``): the positions measured in an earlier ``*_zo.fits``
  (a zero-order flat), moved by the median shift of the spots found in this frame. Star
  frames, in which only a few fibres see light, should use a reference: a few spots cannot
  fix the slit.

Every fibre is measured in a box aperture at its position, lit or not, so a fibre that the
scan moved off its star still gets a flux (near zero) and a variance.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.signal import find_peaks

from ..io.image import ImageFile
from ..tlm.make_tlm import _refine_peaks_gauss3
from ..tlm.match_fibers import (
    ISOPLANE_FIB_PITCH,
    isoplane_nominal_fibpos,
    match_fibers_isoplane,
)
from ..utils.args import init_args
from ..utils.fiber import get_override_from_args

logger = logging.getLogger(__name__)

MAX_NFIBRES = 1000

# ZOFLAG bits
ZOFLAG_NODETECT = 1  # no spot found for this fibre: position from the slit model or reference
ZOFLAG_BADPIX = 2  # bad (non-finite) pixels inside the aperture, left out of the sum
ZOFLAG_EDGE = 4  # aperture cut by the detector edge
ZOFLAG_NOBKG = 8  # no valid background pixels: background taken as 0

# Fibre-table columns copied to the output, when present
FIBRE_COLUMNS = (
    "TYPE", "NAME", "SLIT", "HOLE", "LABEL", "FIBNUM",
    "MAGNITUDE", "RA", "DEC", "X", "Y", "CLASS",
)

# Header cards of the image not copied to the output primary header
_STRUCTURAL_CARDS = {
    "SIMPLE", "BITPIX", "EXTEND", "BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM",
}


def make_zo(args: Dict[str, Any]) -> Table:
    """
    Measure the fibre fluxes of a zero-order image and write them to a ``*_zo.fits`` file.

    Parameters
    ----------
    args : dict
        - 'IMAGE_FILENAME': zero-order image from ``make_im`` (``*_im.fits``), dispersion
          axis along x, so the slit runs along y
        - 'ZOFLUX_FILENAME': output file (default: ``_im.fits`` -> ``_zo.fits``)
        - 'ZO_REF_FILENAME': earlier ``*_zo.fits`` whose ``XPOS``/``YPOS`` fix the fibre
          positions (reference mode, see the module docstring)
        - 'ZO_FIX_POSITIONS': reference mode only; True uses the reference positions as
          they are, without the shift measured on this frame (default False)
        - 'ZO_NFIB': number of slit fibres when the image has no ``FIBRES`` table
        - 'ZO_XPOS': x of the zero-order column (pixels, 0-based; default: the brightest
          column after a row-median subtraction)
        - 'ZO_XRANGE': ``(x0, x1)`` columns searched for the zero order (default: all)
        - 'ZO_APHW_X': half-width in x of the flux aperture (pixels, default 6)
        - 'ZO_APHW_Y': largest half-height in y of the flux aperture (pixels, default half
          the fibre pitch); apertures of neighbouring fibres meet half-way between them
        - 'ZO_BKG_INNER', 'ZO_BKG_OUTER': background columns at ``ZO_BKG_INNER <= |x - XPOS|
          <= ZO_BKG_OUTER`` on both sides, in the aperture rows (pixels, default 12 and 30)
        - 'ZO_DETECT_NSIGMA': spot detection threshold in noise sigma (default 5)
        - 'ISO_FIB_PITCH', 'ISO_FIB_GAPS', 'ISO_FIB_POSITIONS', 'ISO_FIB1_YPOS',
          'ISO_MATCH_TOL': slit model and matching, as in ``make_tlm``. Here ``ISO_FIB1_YPOS``
          is the y of fibre 1 on the zero-order column.
        - 'FIB<no>': fibre type overrides, as in the other steps

    Returns
    -------
    astropy.table.Table
        The ``ZOFLUX`` table (see :func:`measure_zero_order`).
    """
    args = init_args(args)
    im_fname = args.get("IMAGE_FILENAME")
    if not im_fname:
        raise ValueError("IMAGE_FILENAME is required")
    zo_fname = args.get("ZOFLUX_FILENAME") or str(im_fname).replace("_im.fits", "_zo.fits")
    if zo_fname == im_fname:
        raise ValueError("ZOFLUX_FILENAME would overwrite IMAGE_FILENAME")

    reference = None
    ref_fname = args.get("ZO_REF_FILENAME")
    if ref_fname:
        reference = Table.read(ref_fname, hdu="ZOFLUX")
        reference.meta["FILENAME"] = Path(ref_fname).name

    with ImageFile(im_fname, mode="READ") as im_file:
        data = im_file.read_image_data()
        variance = im_file.read_variance_data() if im_file.has_variance() else None
        if variance is None:
            logger.warning("%s has no VARIANCE HDU: FLUX_VAR will be NaN", im_fname)

        lambdac = im_file.get_header_value("LAMBDAC", None)
        try:
            if lambdac is not None and float(lambdac) != 0.0:
                logger.warning(
                    "LAMBDAC = %s A: %s may not be a zero-order frame",
                    lambdac,
                    im_fname,
                )
        except (TypeError, ValueError):
            pass

        fibre_table = None
        if im_file.has_fiber_table():
            fibre_table = Table(im_file.read_fiber_table())
            overrides = get_override_from_args(args)
            if overrides and "TYPE" in fibre_table.colnames:
                types, _ = im_file.read_fiber_types(MAX_NFIBRES, overrides=overrides)
                fibre_table["TYPE"] = np.asarray(types[: len(fibre_table)])

        header = im_file.hdul[0].header.copy()

    table = measure_zero_order(
        data, variance, fibre_table=fibre_table, nfib=args.get("ZO_NFIB"), args=args,
        reference=reference,
    )
    write_zo(zo_fname, table, header, im_fname)
    logger.info("Zero-order fluxes of %d fibres -> %s", len(table), zo_fname)
    return table


def measure_zero_order(
    data: np.ndarray,
    variance: Optional[np.ndarray],
    fibre_table: Optional[Table] = None,
    nfib: Optional[int] = None,
    args: Optional[Dict[str, Any]] = None,
    reference: Optional[Table] = None,
) -> Table:
    """
    Locate the slit fibres on a zero-order image and measure their fluxes.

    Parameters
    ----------
    data : np.ndarray
        Image ``(nspat, nspec)``, bias (and dark) subtracted, bad pixels NaN.
    variance : np.ndarray or None
        Variance of ``data``; None gives ``FLUX_VAR = NaN`` and a noise estimate from the
        data for the detection.
    fibre_table : astropy.table.Table, optional
        ``FIBRES`` table in slit order; its rows set the number of fibres and the columns in
        ``FIBRE_COLUMNS`` are copied to the output.
    nfib : int, optional
        Number of slit fibres when there is neither a fibre table nor a reference.
    args : dict, optional
        ``ZO_*`` and ``ISO_FIB_*`` keys, see :func:`make_zo`.
    reference : astropy.table.Table, optional
        ``ZOFLUX`` table of a zero-order flat: reference mode.

    Returns
    -------
    astropy.table.Table
        One row per slit fibre: ``FIBNO`` (1-based slit order), the fibre-table columns,
        ``XPOS``, ``YPOS`` (aperture centre, 0-based detector pixels), ``XCEN``, ``YCEN``
        (flux-weighted centroid in the aperture), ``FLUX`` and ``FLUX_VAR`` (background
        subtracted, image units), ``SNR``, ``BKG`` (per pixel), ``NPIX`` (aperture area),
        ``PEAK`` (largest background-subtracted pixel), ``NBADPIX``, ``DETECTED`` (a spot was
        found at this fibre) and ``ZOFLAG`` (``ZOFLAG_*`` bits). ``meta`` holds the
        zero-order column, the measured shift and the aperture settings.
    """
    args = {} if args is None else args
    data = np.asarray(data, dtype=float)
    var = None if variance is None else np.asarray(variance, dtype=float)
    if var is not None and var.shape != data.shape:
        raise ValueError(f"variance shape {var.shape} differs from data shape {data.shape}")

    if reference is not None:
        nf = len(reference)
    elif fibre_table is not None:
        nf = len(fibre_table)
    elif nfib:
        nf = int(nfib)
    else:
        raise ValueError("Number of fibres unknown: give a fibre table, ZO_NFIB or a reference")
    if fibre_table is not None and len(fibre_table) != nf:
        raise ValueError(f"Fibre table has {len(fibre_table)} rows, reference has {nf}")

    hw_x = float(args.get("ZO_APHW_X", 6.0))
    bkg_in = float(args.get("ZO_BKG_INNER", 12.0))
    bkg_out = float(args.get("ZO_BKG_OUTER", 30.0))
    nsigma = float(args.get("ZO_DETECT_NSIGMA", 5.0))
    if not hw_x < bkg_in < bkg_out:
        raise ValueError("Need ZO_APHW_X < ZO_BKG_INNER < ZO_BKG_OUTER")

    if reference is not None:
        xpos, ypos, detected, meta = _positions_from_reference(
            data, var, reference, args, hw_x, nsigma
        )
    else:
        xpos, ypos, detected, meta = _positions_from_detection(
            data, var, nf, args, hw_x, nsigma
        )

    pitch = meta["ZOPITCH"]
    hw_y = float(args.get("ZO_APHW_Y", 0.5 * pitch))
    ylo, yhi = _aperture_rows(ypos, hw_y)
    phot = aperture_photometry(data, var, xpos, ylo, yhi, hw_x, bkg_in, bkg_out)

    out = Table()
    out["FIBNO"] = np.arange(1, nf + 1)
    if fibre_table is not None:
        for name in FIBRE_COLUMNS:
            if name in fibre_table.colnames:
                out[name] = fibre_table[name]
    out["XPOS"] = xpos
    out["YPOS"] = ypos
    for name in ("XCEN", "YCEN", "FLUX", "FLUX_VAR"):
        out[name] = phot[name]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["SNR"] = phot["FLUX"] / np.sqrt(phot["FLUX_VAR"])
    for name in ("BKG", "NPIX", "PEAK", "NBADPIX"):
        out[name] = phot[name]
    out["DETECTED"] = detected
    out["ZOFLAG"] = phot["ZOFLAG"] | np.where(detected, 0, ZOFLAG_NODETECT)

    meta.update(ZOAPHWX=hw_x, ZOAPHWY=hw_y, ZOBKGIN=bkg_in, ZOBKGOUT=bkg_out)
    out.meta.update(meta)
    logger.info(
        "Zero order at x = %.2f: %d/%d fibres detected (%s), %d with flags",
        meta["ZOXPOS"], int(np.sum(detected)), nf, meta["ZOMETHOD"],
        int(np.sum(out["ZOFLAG"] & ~ZOFLAG_NODETECT != 0)),
    )
    return out


def find_zero_order_column(
    data: np.ndarray, xrange: Optional[Tuple[int, int]] = None
) -> float:
    """
    x of the zero-order column: the peak of the row-median-subtracted column sums.

    Parameters
    ----------
    data : np.ndarray
        Image ``(nspat, nspec)``.
    xrange : (int, int), optional
        Columns ``[x0, x1)`` searched (default: all).

    Returns
    -------
    float
        Column position (0-based pixels), refined to sub-pixel by a 3-point Gaussian.
    """
    xprof = _column_profile(data)
    x0, x1 = (0, len(xprof)) if xrange is None else (int(xrange[0]), int(xrange[1]))
    if not np.any(np.isfinite(xprof[x0:x1])):
        raise ValueError("No valid pixels in the zero-order search range")
    peak = x0 + int(np.nanargmax(xprof[x0:x1]))
    return float(_refine_peaks_gauss3(xprof, np.array([peak]))[0])


def find_spots(
    data: np.ndarray,
    variance: Optional[np.ndarray],
    xpos: float,
    halfwidth: float,
    min_separation: float,
    nsigma: float = 5.0,
) -> np.ndarray:
    """
    y positions of the fibre spots along the zero-order column.

    Parameters
    ----------
    data, variance : np.ndarray
        Image and its variance (``variance`` may be None: noise from the profile MAD).
    xpos : float
        Zero-order column.
    halfwidth : float
        Columns ``|x - xpos| <= halfwidth`` are summed into the spatial profile.
    min_separation : float
        Smallest distance between two spots (pixels).
    nsigma : float, optional
        Detection threshold in noise sigma of the profile.

    Returns
    -------
    np.ndarray
        Spot positions (0-based pixels), ascending.
    """
    profile, prof_var = _spatial_profile(data, variance, xpos, halfwidth)
    if prof_var is not None and np.any(np.isfinite(prof_var)):
        sigma = np.sqrt(np.where(np.isfinite(prof_var) & (prof_var > 0), prof_var, np.inf))
    else:
        resid = profile[np.isfinite(profile)]
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) if resid.size else np.inf
        sigma = np.full_like(profile, sigma)
    prof = np.where(np.isfinite(profile), profile, 0.0)
    peaks, _ = find_peaks(
        prof, height=nsigma * sigma, distance=max(1, int(round(min_separation)))
    )
    if peaks.size == 0:
        return np.array([], dtype=float)
    return np.sort(_refine_peaks_gauss3(prof, peaks.astype(float)))


def aperture_photometry(
    data: np.ndarray,
    variance: Optional[np.ndarray],
    xpos: np.ndarray,
    ylo: np.ndarray,
    yhi: np.ndarray,
    hw_x: float,
    bkg_inner: float,
    bkg_outer: float,
) -> Dict[str, np.ndarray]:
    """
    Background-subtracted box-aperture sums with fractional pixel weights.

    Pixel ``p`` covers ``[p - 0.5, p + 0.5]``; its weight is the fraction of it inside the
    box ``[xpos - hw_x, xpos + hw_x] x [ylo, yhi]``. The background per pixel is the
    3-sigma-clipped median of the pixels in the aperture rows at ``bkg_inner <= |x - xpos|
    <= bkg_outer``.

    ``FLUX = sum w (d - B)`` and ``FLUX_VAR = sum w^2 v + (sum w)^2 var(B)``, with
    ``var(B) = (pi / 2) <v> / n`` for the median of ``n`` background pixels. Non-finite
    pixels are left out of the sums and counted in ``NBADPIX``.

    Parameters
    ----------
    data, variance : np.ndarray
        Image and variance (``variance`` None gives ``FLUX_VAR = NaN``).
    xpos, ylo, yhi : np.ndarray
        Per-fibre aperture centre in x and row limits in y (0-based pixels).
    hw_x, bkg_inner, bkg_outer : float
        Aperture half-width and background annulus in x (pixels).

    Returns
    -------
    dict
        Arrays ``FLUX``, ``FLUX_VAR``, ``BKG``, ``NPIX``, ``PEAK``, ``XCEN``, ``YCEN``,
        ``NBADPIX`` and ``ZOFLAG`` (``ZOFLAG_BADPIX | ZOFLAG_EDGE | ZOFLAG_NOBKG`` bits).
    """
    ny, nx = data.shape
    nf = len(xpos)
    out = {k: np.full(nf, np.nan) for k in ("FLUX", "FLUX_VAR", "BKG", "NPIX", "PEAK",
                                            "XCEN", "YCEN")}
    out["NBADPIX"] = np.zeros(nf, dtype=int)
    out["ZOFLAG"] = np.zeros(nf, dtype=int)

    for i in range(nf):
        flag = 0
        x0, y0, y1 = float(xpos[i]), float(ylo[i]), float(yhi[i])
        if x0 - hw_x < -0.5 or x0 + hw_x > nx - 0.5 or y0 < -0.5 or y1 > ny - 0.5:
            flag |= ZOFLAG_EDGE
        r0 = max(0, int(np.floor(y0 + 0.5)))
        r1 = min(ny, int(np.ceil(y1 + 0.5)))
        c0 = max(0, int(np.floor(x0 - hw_x + 0.5)))
        c1 = min(nx, int(np.ceil(x0 + hw_x + 0.5)))
        if r1 <= r0 or c1 <= c0:
            out["ZOFLAG"][i] = flag | ZOFLAG_EDGE
            continue
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        wy = _overlap(rows, y0, y1)
        wx = _overlap(cols, x0 - hw_x, x0 + hw_x)
        w = wy[:, None] * wx[None, :]
        d = data[r0:r1, c0:c1]
        v = None if variance is None else variance[r0:r1, c0:c1]

        # Background from the same rows, both sides of the aperture
        dist = np.abs(np.arange(nx) - x0)
        bcols = np.flatnonzero((dist >= bkg_inner) & (dist <= bkg_outer))
        bkg_d = data[r0:r1][:, bcols][wy > 0].ravel()
        good = np.isfinite(bkg_d)
        if variance is not None:
            bkg_v = variance[r0:r1][:, bcols][wy > 0].ravel()
            good &= np.isfinite(bkg_v)
        bkg, keep = _clipped_median(bkg_d[good])
        if keep.sum() == 0:
            bkg, bkg_var = 0.0, 0.0
            flag |= ZOFLAG_NOBKG
        elif variance is None:
            bkg_var = np.nan
        else:
            bkg_var = 0.5 * np.pi * np.mean(bkg_v[good][keep]) / keep.sum()

        ok = np.isfinite(d) & (w > 0)
        if v is not None:
            ok &= np.isfinite(v)
        nbad = int(np.sum(~ok & (w > 0)))
        if nbad:
            flag |= ZOFLAG_BADPIX
        ds = np.where(ok, d - bkg, 0.0)
        wsum = float(np.sum(w))
        flux = float(np.sum(w * ds))
        if v is None:
            fvar = np.nan
        else:
            fvar = float(np.sum(np.where(ok, w**2 * v, 0.0))) + wsum**2 * bkg_var

        out["FLUX"][i] = flux
        out["FLUX_VAR"][i] = fvar
        out["BKG"][i] = bkg
        out["NPIX"][i] = wsum
        out["NBADPIX"][i] = nbad
        if np.any(ok):
            out["PEAK"][i] = float(np.max(ds[ok]))
        if flux > 0:
            out["XCEN"][i] = float(np.sum(w * ds * cols[None, :]) / flux)
            out["YCEN"][i] = float(np.sum(w * ds * rows[:, None]) / flux)
        out["ZOFLAG"][i] = flag
    return out


def write_zo(
    zo_fname: str, table: Table, im_header: fits.Header, im_fname: Optional[str] = None
) -> None:
    """
    Write a ``ZOFLUX`` table with the image header cards in an empty primary HDU.

    Parameters
    ----------
    zo_fname : str
        Output file (overwritten).
    table : astropy.table.Table
        From :func:`measure_zero_order`; its ``meta`` cards go to the table header.
    im_header : astropy.io.fits.Header
        Primary header of the image (observation cards: exposure time, ICS cards, ...).
    im_fname : str, optional
        Image file name, recorded as ``ZOIMAGE``.
    """
    primary = fits.PrimaryHDU()
    for card in im_header.cards:
        key = card.keyword
        if key in _STRUCTURAL_CARDS or key.startswith("NAXIS") or key in ("", "END"):
            continue
        if key == "HISTORY":
            primary.header.add_history(str(card.value))
        elif key == "COMMENT":
            primary.header.add_comment(str(card.value))
        else:
            primary.header.append(card, end=True)
    primary.header["CLASS"] = ("ZOFLUX", "Zero-order fibre fluxes")
    if im_fname:
        primary.header["ZOIMAGE"] = (Path(im_fname).name, "Zero-order image measured")
    primary.header.add_history("kspecdr make_zo: zero-order fibre photometry")

    tab = Table(table, copy=True)
    tab.meta.clear()
    hdu = fits.table_to_hdu(tab)
    hdu.name = "ZOFLUX"
    comments = {
        "ZOMETHOD": "Fibre positions: DETECT or REFERENCE",
        "ZOXPOS": "Zero-order column (0-based px)",
        "ZOPITCH": "Fibre pitch along the slit (px)",
        "ZODX": "Shift from the reference in x (px)",
        "ZODY": "Shift from the reference in y (px)",
        "ZONDET": "Fibres with a detected spot",
        "ZOREF": "Reference zero-order flux file",
        "ZOAPHWX": "Aperture half-width in x (px)",
        "ZOAPHWY": "Largest aperture half-height in y (px)",
        "ZOBKGIN": "Background inner distance in x (px)",
        "ZOBKGOUT": "Background outer distance in x (px)",
    }
    for key, value in table.meta.items():
        if value is None:
            continue
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        hdu.header[key] = (value, comments.get(key, ""))
    fits.HDUList([primary, hdu]).writeto(zo_fname, overwrite=True)


def _positions_from_detection(data, var, nf, args, hw_x, nsigma):
    """Fibre positions from spots matched to the slit model (``make_tlm`` keys)."""
    xpos0 = args.get("ZO_XPOS")
    if xpos0 is None:
        xpos0 = find_zero_order_column(data, args.get("ZO_XRANGE"))
    xpos0 = float(xpos0)

    pitch = args.get("ISO_FIB_PITCH")
    min_sep = 0.5 * (ISOPLANE_FIB_PITCH if pitch is None else float(pitch))
    spots = find_spots(data, var, xpos0, hw_x, min_sep, nsigma)
    if pitch is None:
        pitch = float(np.median(np.diff(spots))) if len(spots) >= 2 else ISOPLANE_FIB_PITCH
        logger.info("Fibre pitch from the spots: %.2f px", pitch)
    pitch = float(pitch)

    offsets = isoplane_nominal_fibpos(
        nf, pitch=pitch, gaps=args.get("ISO_FIB_GAPS"), positions=args.get("ISO_FIB_POSITIONS")
    )
    fib1 = args.get("ISO_FIB1_YPOS")
    tol = args.get("ISO_MATCH_TOL")
    match_vector, model = match_fibers_isoplane(
        nf, spots, offsets,
        max_del=None if tol is None else float(tol),
        fib1_ypos=None if fib1 is None else float(fib1),
    )
    detected = match_vector > 0
    ypos = np.where(detected, spots[np.maximum(match_vector, 1) - 1], model)

    # The zero-order image of the slit may be tilted: fit x(y) to the spot centroids
    xpos = _fit_column_tilt(data, var, xpos0, ypos, detected, pitch, hw_x, args)
    meta = dict(ZOMETHOD="DETECT", ZOXPOS=xpos0, ZOPITCH=pitch, ZONDET=int(detected.sum()))
    return xpos, ypos, detected, meta


def _positions_from_reference(data, var, reference, args, hw_x, nsigma):
    """Reference positions moved by the median shift of the spots found in this frame."""
    xref = np.asarray(reference["XPOS"], dtype=float)
    yref = np.asarray(reference["YPOS"], dtype=float)
    meta = dict(ZOMETHOD="REFERENCE", ZOREF=reference.meta.get("FILENAME", ""))
    pitch = reference.meta.get("ZOPITCH")
    if pitch is None:
        pitch = float(np.median(np.diff(np.sort(yref)))) if len(yref) >= 2 else ISOPLANE_FIB_PITCH
    pitch = float(pitch)
    xpos0 = float(args.get("ZO_XPOS", np.median(xref)))
    meta.update(ZOXPOS=xpos0, ZOPITCH=pitch)

    spots = find_spots(data, var, xpos0, hw_x, 0.5 * pitch, nsigma)
    tol = float(args.get("ISO_MATCH_TOL", 0.35 * pitch))
    detected = np.zeros(len(yref), dtype=bool)
    dy_each = []
    for y in spots:
        j = int(np.argmin(np.abs(yref - y)))
        if abs(yref[j] - y) < tol and not detected[j]:
            detected[j] = True
            dy_each.append(y - yref[j])
    meta["ZONDET"] = int(detected.sum())

    dx = dy = 0.0
    if args.get("ZO_FIX_POSITIONS", False):
        pass
    elif dy_each:
        dy_each = np.asarray(dy_each)
        ylo, yhi = _aperture_rows(
            yref + np.median(dy_each), float(args.get("ZO_APHW_Y", 0.5 * pitch))
        )
        phot = aperture_photometry(
            data, var, xref, ylo, yhi, hw_x,
            float(args.get("ZO_BKG_INNER", 12.0)), float(args.get("ZO_BKG_OUTER", 30.0)),
        )
        # Brighter spots weigh more: weights SNR^2 (1 without a variance)
        snr = phot["FLUX"][detected] / np.sqrt(phot["FLUX_VAR"][detected])
        weight = np.where(np.isfinite(snr), np.clip(snr, 0.0, None) ** 2, 1.0)
        dy = _weighted_shift(dy_each, weight)
        dx_each = _spot_x(data, xref, ylo, yhi, hw_x)[detected] - xref[detected]
        dx = _weighted_shift(dx_each, weight)
    else:
        logger.warning("No spots near the reference positions: using them unshifted")
    if max(abs(dx), abs(dy)) > 0.5 * pitch:
        logger.warning("Shift from the reference (%.2f, %.2f) px exceeds half a pitch", dx, dy)
    meta.update(ZODX=dx, ZODY=dy)
    logger.info("Reference %s: %d spots, shift dx %.2f, dy %.2f px",
                meta["ZOREF"], meta["ZONDET"], dx, dy)
    return xref + dx, yref + dy, detected, meta


def _fit_column_tilt(data, var, xpos0, ypos, detected, pitch, hw_x, args):
    """Per-fibre x from a straight-line fit of x(y) to the centroids of detected spots."""
    xpos = np.full(len(ypos), xpos0)
    if detected.sum() < 3:
        return xpos
    ylo, yhi = _aperture_rows(ypos, float(args.get("ZO_APHW_Y", 0.5 * pitch)))
    phot = aperture_photometry(
        data, var, xpos, ylo, yhi, hw_x,
        float(args.get("ZO_BKG_INNER", 12.0)), float(args.get("ZO_BKG_OUTER", 30.0)),
    )
    xspot = _spot_x(data, xpos, ylo, yhi, hw_x)
    use = detected & np.isfinite(xspot) & (phot["FLUX"] > 0)
    if use.sum() < 3:
        return xpos
    coef = np.polyfit(ypos[use], xspot[use], 1)
    resid = xspot[use] - np.polyval(coef, ypos[use])
    keep = np.abs(resid) < max(3.0 * 1.4826 * np.median(np.abs(resid)), 0.1)
    if keep.sum() >= 3:
        coef = np.polyfit(ypos[use][keep], xspot[use][keep], 1)
    logger.info("Zero-order column x(y) = %.3f + %.5f y", coef[1], coef[0])
    return np.polyval(coef, ypos)


def _spot_x(data, xpos, ylo, yhi, hw_x):
    """Spot x per fibre: 3-point Gaussian peak of the row-summed profile in the aperture."""
    ny, nx = data.shape
    out = np.full(len(xpos), np.nan)
    with np.errstate(invalid="ignore"):
        rowbkg = np.nanmedian(data, axis=1)
    for i, (x0, y0, y1) in enumerate(zip(xpos, ylo, yhi)):
        r0 = max(0, int(np.floor(y0 + 0.5)))
        r1 = min(ny, int(np.ceil(y1 + 0.5)))
        c0 = max(0, int(np.floor(x0 - hw_x + 0.5)))
        c1 = min(nx, int(np.ceil(x0 + hw_x + 0.5)))
        if r1 <= r0 or c1 - c0 < 3:
            continue
        prof = np.nansum(data[r0:r1, c0:c1] - rowbkg[r0:r1, None], axis=0)
        peak = int(np.argmax(prof))
        if 0 < peak < len(prof) - 1:
            out[i] = c0 + float(_refine_peaks_gauss3(prof, np.array([float(peak)]))[0])
    return out


def _weighted_shift(values: np.ndarray, weight: np.ndarray, clip: float = 0.5) -> float:
    """Weighted mean of per-spot shifts within ``clip`` px of their median (0 if none)."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weight) & (weight > 0)
    if not np.any(ok):
        return 0.0
    ok &= np.abs(values - np.median(values[ok])) <= clip
    return float(np.sum(weight[ok] * values[ok]) / np.sum(weight[ok]))


def _aperture_rows(ypos: np.ndarray, hw_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """Row limits: half-way to the neighbouring fibres, at most ``hw_y`` from the centre."""
    ypos = np.asarray(ypos, dtype=float)
    order = np.argsort(ypos)
    ys = ypos[order]
    mid = 0.5 * (ys[1:] + ys[:-1])
    lo = np.concatenate([[-np.inf], mid])
    hi = np.concatenate([mid, [np.inf]])
    ylo = np.empty_like(ypos)
    yhi = np.empty_like(ypos)
    ylo[order] = np.maximum(lo, ys - hw_y)
    yhi[order] = np.minimum(hi, ys + hw_y)
    return ylo, yhi


def _overlap(pix: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fraction of each pixel ``[p - 0.5, p + 0.5]`` inside ``[lo, hi]``."""
    return np.clip(np.minimum(hi, pix + 0.5) - np.maximum(lo, pix - 0.5), 0.0, 1.0)


def _clipped_median(x: np.ndarray, nsig: float = 3.0, niter: int = 3):
    """Median after iterative clipping at ``nsig`` MAD-sigma; returns (median, kept mask)."""
    keep = np.ones(x.size, dtype=bool)
    if x.size == 0:
        return 0.0, keep
    for _ in range(niter):
        med = np.median(x[keep])
        sig = 1.4826 * np.median(np.abs(x[keep] - med))
        if sig == 0:
            break
        new = np.abs(x - med) <= nsig * sig
        if np.array_equal(new, keep) or new.sum() == 0:
            break
        keep = new
    return float(np.median(x[keep])), keep


def _column_profile(data: np.ndarray) -> np.ndarray:
    """Column sums after subtracting each row's median."""
    with np.errstate(invalid="ignore"):
        rowbkg = np.nanmedian(data, axis=1, keepdims=True)
        return np.nansum(data - rowbkg, axis=0)


def _spatial_profile(data, variance, xpos, halfwidth):
    """Row profile summed over ``|x - xpos| <= halfwidth``, row median subtracted."""
    nx = data.shape[1]
    c0 = max(0, int(np.floor(xpos - halfwidth + 0.5)))
    c1 = min(nx, int(np.ceil(xpos + halfwidth + 0.5)))
    if c1 <= c0:
        raise ValueError(f"Zero-order column x = {xpos:.1f} is off the detector")
    with np.errstate(invalid="ignore"):
        rowbkg = np.nanmedian(data, axis=1, keepdims=True)
        profile = np.nansum(data[:, c0:c1] - rowbkg, axis=1)
        prof_var = None if variance is None else np.nansum(variance[:, c0:c1], axis=1)
    return profile, prof_var
