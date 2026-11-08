"""
Extraction Routines for KSPEC.

This module implements the extraction of spectra from image data using tramline maps,
converting the 2dfdr `MAKE_EX` and related subroutines.
"""

import logging
import sys
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

from ..io.image import ImageFile
from ..tlm.make_tlm import TLM_SUPPORT_KEYS, predict_wavelength
from ..tlm.sigma_profile import FWHM_PER_SIGMA
from ..tlm.tlm_creep import measure_tlm_shift
from ..utils.fiber import get_override_from_args
from .optimal import optimal_extract

logger = logging.getLogger(__name__)

# Frame classes whose trace shift is measured with MSR_TLM_CREEP (2dfdr: MFOBJECT, MFSKY,
# MFARC; the Isoplane converter writes OBJECT for science frames)
CREEP_CLASSES = ("MFOBJECT", "MFSKY", "MFARC", "OBJECT", "SKY", "ARC")

# Constants
MAX_NFIBRES = 1000
VAL__BADR = np.nan  # Using NaN for bad values in Python


def make_ex(args: Dict[str, Any]) -> None:
    """
    Main driver for extraction process.
    Replaces 2dfdr SUBROUTINE MAKE_EX.

    Parameters
    ----------
    args : dict
        Dictionary containing arguments:
        - IMAGE_FILENAME: Input image file
        - EXTRAC_FILENAME: Output extracted file
        - TLMAP_FILENAME: Tramline map file
        - WTSCHEME: Weighting scheme (optional): 'STND' (1/variance, default) or 'NONE'
        - EXTR_OPERATION: 'SUM' (default 'TRAM' runs SUM) or 'OPTEX'
        - SUM_WIDTH: SUM window width in pixels (default 10)
        - TLM_SHIFT: shift added to the tramline map, in pixels (default 0)
        - MSR_TLM_CREEP: measure the shift of the frame's traces against the tramline map
          (2dfdr ``MEASURE_TLM_SHIFT``, :func:`kspecdr.tlm.tlm_creep.measure_tlm_shift`) on
          object, sky and arc frames; it overrides TLM_SHIFT, which remains the fallback when
          the measurement fails (default False)
        - TLM_INIT_SHIFT: centre of the searched shift range, in pixels (default 0; the
          search covers +-2.5 px)

        OPTEX options (2dfdr names unless noted; see :func:`kspecdr.extract.optimal`):

        - SIGPROF: use the ``SIGMAPRF`` profile widths of the tramline map (default False:
          ``MWIDTH / 2.3548`` for every fibre and column)
        - PSF_TYPE: profile model, only 'GAUSS'
        - OPTEX_NSLPARS: scattered-light terms per column (default 1, a constant)
        - OPTEX_SLMODEL: 'POLYN' (Chebyshev, default) or 'KNOTS' (piecewise linear with
          knots between the fibres; kspecdr)
        - OPTEX_NITER: number of fits per column (kspecdr; default 2 when the image has
          ``RO_GAIN``: the second fit uses the model variance)
        - OPTEX_CLIP: cosmic-ray rejection threshold in sigma (kspecdr; default 0 = off)
        - OPTEX_PRFERR: fractional profile error in the rejection threshold (kspecdr;
          default 0.02)
        - OPTEX_MKRES: also write ``<EXTRAC_FILENAME stem>_res.fits`` with the residual,
          model and scattered-light images (default False)
    """
    im_fname = args.get("IMAGE_FILENAME")
    ex_fname = args.get("EXTRAC_FILENAME")
    tlm_fname = args.get("TLMAP_FILENAME")
    wtscheme = args.get("WTSCHEME", "STND")

    if not im_fname or not ex_fname or not tlm_fname:
        raise ValueError("Missing required filenames (IMAGE, EXTRAC, or TLMAP)")

    # Check if TLM exists
    if not Path(tlm_fname).exists():
        raise FileNotFoundError(f"Tramline map file not found: {tlm_fname}")

    logger.info(f"Extracting {im_fname} -> {ex_fname} using TLM {tlm_fname}")

    # Call the main extraction routine
    make_ex_from_im(im_fname, tlm_fname, ex_fname, wtscheme, args)

    # TODO: Handle Stochastic copies if needed (NSTOCHIM) - skipping for now as it's advanced usage


def make_ex_from_im(
    im_fname: str, tlm_fname: str, ex_fname: str, wtscheme: str, args: Dict[str, Any]
) -> None:
    """
    Process image file to produce extracted spectra.
    Replaces 2dfdr SUBROUTINE MAKE_EX_FROM_IM.

    Parameters
    ----------
    im_fname : str
        Input image filename
    tlm_fname : str
        Tramline map filename
    ex_fname : str
        Output extracted filename
    wtscheme : str
        Weighting scheme
    args : dict
        Additional arguments
    """
    # 1. Get Extraction Method
    operat = args.get("EXTR_OPERATION", "TRAM").upper()
    logger.info(f"Extraction Method: {operat}")

    valid_methods = [
        "FIT",
        "TRAM",
        "NEWTRAM",
        "GAUSS",
        "OPTEX",
        "CLOPTEX",
        "SMCOPTEX",
        "SUM",
    ]
    if operat not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    if operat == "FIT":
        raise NotImplementedError("FIT method does NOT work (legacy status)")

    # 2. Open Image File
    with ImageFile(im_fname, mode="READ") as im_file:
        im_class = str(
            im_file.get_header_value("NDFCLASS", "") or im_file.get_header_value("CLASS", "")
        ).strip().upper()
        instrument_code = im_file.get_instrument_code()
        ro_gain = im_file.get_header_value("RO_GAIN", None)

        img_data = im_file.read_image_data()
        var_data = im_file.read_variance_data()
        fib_tabl = im_file.read_fiber_table()

        # 3. Read Tramline Map
        with ImageFile(tlm_fname, mode="READ") as tlm_file:
            tlm_data = tlm_file.read_image_data()
            # TLM is (NFIB, NSPEC) - Horizontal
            nfib, nspec_tlm = tlm_data.shape

            # Read fiber types
            overrides = get_override_from_args(args)
            fiber_types, _ = im_file.read_fiber_types(MAX_NFIBRES, overrides=overrides)
            # Note: 2dfdr reads fiber types from IM file usually, as TLM might not have them updated?

            # Get MWIDTH from TLM header (Median FWHM)
            mwidth = float(tlm_file.get_header_value("MWIDTH", 2.0))
            tlm_hdr = tlm_file.hdul[0].header
            mwidth_measured = "MWIDTH" in tlm_hdr and "measured" in tlm_hdr.comments["MWIDTH"]

            # Read Wavelength data if available
            wave_data = tlm_file.read_wave_data()

            # Verify TLM header matches IM header; recompute WAVELA if mismatch.
            def _get_hdr_float(img, key):
                value = img.get_header_value(key, None)
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            im_lambdac = _get_hdr_float(im_file, "LAMBDAC")
            im_dispers = _get_hdr_float(im_file, "DISPERS")
            tlm_lambdac = _get_hdr_float(tlm_file, "LAMBDAC")
            tlm_dispers = _get_hdr_float(tlm_file, "DISPERS")

            im_gratid = im_file.get_header_value("GRATID", None)
            im_gratl = im_file.get_header_value("GRATLPMM", None)
            tlm_gratid = tlm_file.get_header_value("GRATID", None)
            tlm_gratl = tlm_file.get_header_value("GRATLPMM", None)

            def _float_mismatch(a, b, tol=1e-3):
                if a is None or b is None:
                    return True
                return abs(a - b) > tol

            grat_mismatch = (
                im_gratid is not None
                and tlm_gratid is not None
                and im_gratid != tlm_gratid
            ) or (
                im_gratl is not None and tlm_gratl is not None and im_gratl != tlm_gratl
            )
            wave_mismatch = _float_mismatch(im_lambdac, tlm_lambdac) or _float_mismatch(
                im_dispers, tlm_dispers
            )

            if (
                grat_mismatch
                or wave_mismatch
                or (tlm_lambdac is None and tlm_dispers is None)
            ):
                logger.warning(
                    "TLM header mismatch with IM header; recomputing WAVELA from IM."
                )
                wave_data = predict_wavelength(im_file, tlm_data.T, args).T

    # 4. Apply TLM Shift (Shift-Rotate-Tweak), measured with MSR_TLM_CREEP
    tlm_shift = float(args.get("TLM_SHIFT", 0.0))
    creep = None
    if bool(args.get("MSR_TLM_CREEP", False)):
        if im_class in CREEP_CLASSES:
            sigma_creep, _ = read_profile_sigma(
                tlm_fname, nfib, tlm_data.shape[1], mwidth, bool(args.get("SIGPROF", False))
            )
            measured, creep = measure_tlm_shift(
                img_data, var_data, tlm_data, sigma_creep,
                init_shift=float(args.get("TLM_INIT_SHIFT", 0.0)),
            )
            if np.isfinite(measured):
                tlm_shift = measured
            else:
                logger.warning(f"Keeping TLM_SHIFT = {tlm_shift} after the failed measurement")
        else:
            logger.info(f"Not measuring the TLM shift on a frame of class {im_class!r}")
    if tlm_shift != 0.0:
        logger.info(f"Shifting Tramline Map by {tlm_shift} pixels")
        tlm_data += tlm_shift

    # 5. Background / Scattered Light Subtraction
    # (Placeholder: SCATSUB)
    scatsub = args.get("SCATSUB", "NONE")
    if scatsub != "NONE":
        # TODO: implement scattered light subtraction
        logger.warning(
            f"Scattered light subtraction '{scatsub}' requested but not implemented yet."
        )

    # 6. Standardize dimensions
    nspat, nspec = img_data.shape

    tlm_data_T = tlm_data.T  # Now (NSPEC, NFIB)

    # Initialize Output Arrays
    # Output extracted data: (NSPEC, NFIB)
    ex_img = np.zeros((nspec, nfib), dtype=np.float32)
    ex_var = np.zeros((nspec, nfib), dtype=np.float32)

    # 7. Perform Extraction
    if operat in ["TRAM", "SUM", "NEWTRAM"]:
        # Simple Summing Extraction
        # Get width from args or use default
        width = float(args.get("SUM_WIDTH", 10.0))  # Default from SUMEXTR
        # Or from MWIDTH if TRAM?
        if operat == "TRAM":
            # TRAM usually uses tramlines. SUMEXTR uses WIDTH.
            # 2dfdr: IF (OPERAT=='TRAM') CALL UMFIM_TRMEXTR...
            # ELSE CALL SUMEXTR...
            # We will implement SUMEXTR logic here as requested "simple summing"
            # TODO: Implement TRAM extraction
            pass

        logger.info(f"Performing SUM extraction with width={width}")

        sum_extract(
            nspat, nspec, img_data, var_data, ex_img, ex_var, nfib, tlm_data_T, width
        )

    elif operat == "GAUSS":
        # Placeholder for GAUSS
        logger.warning(
            "GAUSS extraction not fully implemented. Using simplified fallback or raising error."
        )
        raise NotImplementedError("GAUSS extraction not implemented")

    elif operat == "OPTEX":
        psf_type = str(args.get("PSF_TYPE", "GAUSS")).upper()
        if psf_type != "GAUSS":
            raise NotImplementedError(f"PSF_TYPE {psf_type!r} not implemented (only GAUSS)")
        sigma, sig_source = read_profile_sigma(
            tlm_fname, nfib, nspec, mwidth, bool(args.get("SIGPROF", False))
        )
        if sig_source == "MWIDTH" and not mwidth_measured:
            logger.warning(
                f"OPTEX: MWIDTH = {mwidth} in {tlm_fname} was not measured on a flat; "
                "remake the tramline map from the fibre flat (make_tlm, SIGPROF=True)"
            )
        gain = float(ro_gain) if ro_gain is not None else None
        niter = int(args.get("OPTEX_NITER", 2 if gain is not None else 1))
        mkres = bool(args.get("OPTEX_MKRES", False))
        optex_pars = dict(
            nslpars=int(args.get("OPTEX_NSLPARS", 1)),
            slmodel=str(args.get("OPTEX_SLMODEL", "POLYN")).upper(),
            wtscheme=wtscheme,
            gain=gain,
            niter=niter,
            clip=float(args.get("OPTEX_CLIP", 0.0)),
            prferr=float(args.get("OPTEX_PRFERR", 0.02)),
        )
        logger.info(f"Performing OPTEX extraction, sigma from {sig_source}: {optex_pars}")
        flux, fvar, optex_info = optimal_extract(
            img_data, var_data, tlm_data, sigma, return_model=mkres, **optex_pars
        )
        ex_img[:] = flux.T
        ex_var[:] = fvar.T

    elif operat == "SMCOPTEX":
        raise NotImplementedError("SMCOPTEX not implemented; use OPTEX")

    else:
        raise ValueError(f"Unknown operation: {operat}")

    # 8. Post-Processing

    # Handle Bad Fibers (Zero them out)
    # 2dfdr: UMFIM_ZERO(..., TYP)
    # Zero out 'F' (Guide), 'N' (Not used), 'U' (Unallocated)
    for fib in range(nfib):
        ftype = fiber_types[fib]
        if ftype in ["F", "N", "U"]:
            ex_img[:, fib] = 0.0
            ex_var[:, fib] = 0.0

    # 9. Write Output
    # Create output file from copy of image file (to preserve headers)
    # In kspecdr, we might create a new file or copy.
    # Using ImageFile's save_as or creating new HDUList.

    # We need to reshape/transpose back to FITS convention (NFIB, NSPEC) for writing?
    # If ex_img is (NSPEC, NFIB), FITS expects (NAXIS2, NAXIS1) -> (NFIB, NSPEC).
    ex_img_out = ex_img.T
    ex_var_out = ex_var.T

    from astropy.io import fits

    # Read original header
    with fits.open(im_fname) as hdul_src:
        header = hdul_src[0].header.copy()

    # Update Header
    header["HISTORY"] = f"Extracted using {operat}"
    header["TLMSHIFT"] = (round(tlm_shift, 4), "Shift applied to the tramline map [px]")
    if creep is not None:
        header["TLMSHERR"] = (
            round(creep["err"], 4) if np.isfinite(creep["err"]) else -1.0,
            "Error of the measured TLM shift [px]",
        )
        header["TLMSHN"] = (creep["nused"], "Block profiles used for the TLM shift")
    if operat == "OPTEX":
        header["OPTSIGSR"] = (sig_source, "OPTEX profile sigma source")
        header["OPTSLMOD"] = (optex_pars["slmodel"], "OPTEX scattered-light model")
        header["OPTNSLP"] = (optex_pars["nslpars"], "OPTEX scattered-light terms")
        header["OPTNITER"] = (niter, "OPTEX fits per column")
        header["OPTCLIP"] = (optex_pars["clip"], "OPTEX CR clip [sigma], 0 = off")
        header["OPTNREJ"] = (int(optex_info["nrej"].sum()), "OPTEX rejected pixels")
        header["OPTCHI2"] = (
            round(float(np.nanmedian(optex_info["chi2"])), 4),
            "OPTEX median reduced chi2 per column",
        )

    # Carry the measured trace range of the tramline map (columns outside are extrapolated)
    with fits.open(tlm_fname) as hdul_tlm:
        for key in TLM_SUPPORT_KEYS:
            if key in hdul_tlm[0].header:
                header[key] = (hdul_tlm[0].header[key], hdul_tlm[0].header.comments[key])

    # Set Axes Labels
    header["CTYPE1"] = "Wavelength"
    header["CUNIT1"] = "Angstroms"
    header["CTYPE2"] = "Fibre Number"

    # Create HDUs
    hdu_data = fits.PrimaryHDU(data=ex_img_out, header=header)
    hdu_var = fits.ImageHDU(data=ex_var_out, name="VARIANCE")

    hdul_out = fits.HDUList([hdu_data, hdu_var])

    # Copy WAVELA if available (from TLM usually)
    if wave_data is not None:
        # wave_data is (NFIB, NSPEC) if read from standard TLM.
        # Output is (NFIB, NSPEC).
        # So no transpose needed if TLM is already in output format.
        # However, check internal consistency.
        # If read_wave_data returned data.shape == (ny, nx) == (nfib, nx_tlm)
        # And output is (nfib, nspec).
        # It matches.
        hdu_wave = fits.ImageHDU(data=wave_data, name="WAVELA")
        hdul_out.append(hdu_wave)

    # Copy FIBRES if available (from IM file)
    if fib_tabl is not None:
        hdu_fibres = fits.BinTableHDU(data=fib_tabl, name="FIBRES")
        hdul_out.append(hdu_fibres)

    hdul_out.writeto(ex_fname, overwrite=True)
    hdul_out.close()
    logger.info(f"Written extracted file: {ex_fname}")

    if operat == "OPTEX" and mkres:
        write_optex_residuals(ex_fname, img_data, optex_info, header)


def read_profile_sigma(
    tlm_fname: str, nfib: int, nspec: int, mwidth: float, sigprof: bool
) -> Tuple[np.ndarray, str]:
    """
    Profile sigma per fibre and column for OPTEX, as 2dfdr ``MAKE_EX``.

    With ``sigprof`` and a ``SIGMAPRF`` HDU in the tramline map, its values are used and
    those below 0.5 px (or non-finite) are replaced by ``MWIDTH / 2.3548``; otherwise that
    constant is used everywhere.

    Returns
    -------
    sigma : np.ndarray
        Shape (nfib, nspec).
    source : str
        'SIGMAPRF' or 'MWIDTH'.
    """
    from astropy.io import fits

    sigma0 = mwidth / FWHM_PER_SIGMA
    if sigprof:
        with fits.open(tlm_fname) as hdul:
            if "SIGMAPRF" in hdul:
                sigma = np.asarray(hdul["SIGMAPRF"].data, dtype=float)
                if sigma.shape != (nfib, nspec):
                    raise ValueError(
                        f"SIGMAPRF shape {sigma.shape} does not match ({nfib}, {nspec})"
                    )
                return np.where(np.isfinite(sigma) & (sigma >= 0.5), sigma, sigma0), "SIGMAPRF"
        logger.warning(f"SIGPROF set but {tlm_fname} has no SIGMAPRF; using MWIDTH = {mwidth}")
    return np.full((nfib, nspec), sigma0), "MWIDTH"


def write_optex_residuals(
    ex_fname: str, img_data: np.ndarray, info: Dict[str, Any], header: Any
) -> None:
    """
    Write the OPTEX residual diagnostics (2dfdr ``OPTEX_MKRES``) next to the EX file.

    ``<stem>_res.fits``: PRIMARY = image - model, ``MODEL`` (fibre profiles plus scattered
    light) and ``SCATLITE`` (scattered-light model), all (nspat, nspec) and NaN outside the
    fitted rows, and ``REJECTED`` (1 = pixel rejected by ``OPTEX_CLIP``).
    """
    from astropy.io import fits

    stem = str(ex_fname)
    stem = stem[: -len(".fits")] if stem.endswith(".fits") else stem
    stem = stem[: -len("_ex")] if stem.endswith("_ex") else stem
    res_fname = f"{stem}_res.fits"
    model = info["model"]
    hdr = header.copy()
    for key in ("CTYPE1", "CUNIT1", "CTYPE2"):
        hdr.remove(key, ignore_missing=True)
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=(img_data - model).astype(np.float32), header=hdr),
            fits.ImageHDU(data=model.astype(np.float32), name="MODEL"),
            fits.ImageHDU(data=info["background"].astype(np.float32), name="SCATLITE"),
            fits.ImageHDU(data=info["rejected"].astype(np.uint8), name="REJECTED"),
        ]
    )
    hdul.writeto(res_fname, overwrite=True)
    logger.info(f"Written OPTEX residual file: {res_fname}")


def sum_extract(
    nspat: int,
    nspec: int,
    indat: np.ndarray,
    invar: np.ndarray,
    outdat: np.ndarray,
    outvar: np.ndarray,
    nfib: int,
    tlmap: np.ndarray,
    width: float,
) -> None:
    """
    Perform simple summing extraction.
    Replaces 2dfdr SUBROUTINE SUMEXTR.

    Sums the image over a window of ``width`` pixels centred on each tramline, with
    fractional weights for the two partial pixels at the window edges. Pixel ``k`` spans
    ``[k - 0.5, k + 0.5]``, the convention of the tramline map (0-based pixel centres) and of
    the OPTEX profiles. The variance of a partial pixel is weighted by the square of its
    fraction (2dfdr ``SUMEXTR`` adds ``INVAR * PART``). A window touching a bad (NaN) pixel,
    or lying entirely off the detector, gives NaN.

    Parameters
    ----------
    nspat : int
        Number of spatial pixels (rows in indat)
    nspec : int
        Number of spectral pixels (cols in indat)
    indat : np.ndarray
        Input image (NSPAT, NSPEC) - Horizontal Dispersion
    invar : np.ndarray
        Input variance (NSPAT, NSPEC) - Horizontal Dispersion
    outdat : np.ndarray
        Output spectra (NSPEC, NFIB) - Updated in place
    outvar : np.ndarray
        Output variance (NSPEC, NFIB) - Updated in place
    nfib : int
        Number of fibers
    tlmap : np.ndarray
        Tramline map (NSPEC, NFIB), 0-based pixel centres
    width : float
        Width of extraction window in pixels
    """
    cols = np.arange(nspec)
    half = width / 2.0
    # Row offsets around each centre that can overlap the window
    offsets = np.arange(-int(np.ceil(half)) - 1, int(np.ceil(half)) + 2)
    for fibre in range(nfib):
        centre = tlmap[:, fibre].astype(float)
        ok = np.isfinite(centre)
        base = np.floor(np.where(ok, centre, 0.0)).astype(int)
        rows = base[None, :] + offsets[:, None]  # (noff, nspec)
        lo = centre[None, :] - half
        hi = centre[None, :] + half
        frac = np.clip(np.minimum(rows + 0.5, hi) - np.maximum(rows - 0.5, lo), 0.0, 1.0)
        on_chip = (rows >= 0) & (rows < nspat)
        frac = np.where(on_chip & ok[None, :], frac, 0.0)
        r = np.clip(rows, 0, nspat - 1)
        val = indat[r, cols[None, :]]
        var = invar[r, cols[None, :]]
        used = frac > 0
        bad = np.any(used & ~(np.isfinite(val) & np.isfinite(var)), axis=0)
        bad |= ~np.any(used, axis=0)
        flux = np.sum(np.where(used, val * frac, 0.0), axis=0)
        fvar = np.sum(np.where(used, var * frac**2, 0.0), axis=0)
        outdat[:, fibre] = np.where(bad, VAL__BADR, flux)
        outvar[:, fibre] = np.where(bad, VAL__BADR, fvar)
