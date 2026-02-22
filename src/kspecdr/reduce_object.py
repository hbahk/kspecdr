"""
Reduce Object Module

This module implements the top-level ``reduce_object`` routine for the KSPEC
pipeline, orchestrating the reduction of a raw science file to produce im(age),
ex(tracted), and red(uced) science files.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .preproc.make_im import make_im
from .extract.make_ex import make_ex
from .io.image import ImageFile
from .wavecal.scrunch import scrunch_from_arc_id
from .utils.args import init_args, validate_reduce_object_args

logger = logging.getLogger(__name__)


def reduce_object(args: Dict[str, Any]) -> None:
    """
    Reduce a raw science file to produce im, ex, and red science files.

    Parameters
    ----------
    args : dict
        Dictionary containing reduction arguments:

        Required keys:
        - ``RAW_FILENAME``: Input raw filename
        - ``IMAGE_FILENAME``: Output IM filename
        - ``EXTRAC_FILENAME``: Output extracted filename
        - ``OUTPUT_FILENAME``: Output reduced filename
        - ``TLMAP_FILENAME``: Tramline map filename
        - ``WAVEL_FILENAME``: Wavelength calibration (arc RED) filename

        Optional keys:
        - ``FFLAT_FILENAME``: Fiber flat filename
        - ``OUT_DIRNAME``: Output directory
        - ``DPCRREX``: Double pass cosmic ray rejection (bool)
        - ``EXTR_OPERATION``: Extraction method (default ``"SUM"``)
        - ``OPTEX_MKRES``: Make residual map for optimal extraction (bool)
        - ``VERBOSE``: Verbosity (bool, default True)
        - ``USE_GENCAL``: Use skyline recalibration (bool)
        - ``TST_SKYCAL``: Test skyline calibration (bool)
        - ``INC_RWSS``: Include Reduced Without Sky Subtraction copy (bool)
        - ``SKYSPRSMP``: Super sky subtraction (bool)
        - ``SKYSUB``: Enable sky subtraction (bool, default True)
        - ``SKYCOMBINE``: Sky fiber combination method
          (``'MEAN'`` | ``'MEDIAN'`` | ``'SIGCLIP'``, default ``'MEAN'``)
        - ``SKYCOMBINE_SIGMA``: Sigma-clipping threshold for ``SIGCLIP``
          (float, default 3.0)
        - ``SKYCOMBINE_ITERS``: Max iterations for ``SIGCLIP``
          (int, default 5)
        - ``SKYSUB_PCA``: Enable PCA sky subtraction (bool)
        - ``CALIBFLUX``: Enable flux calibration (bool)
        - ``TELCOR``: Enable telluric correction (bool)
        - ``VELCOR``: Enable velocity correction (bool)
        - ``TRANSFUNC``: Transfer function correction (bool)
        - ``DEWIGGLE``: De-wiggle (bool)
    """
    # --- Initialisation ---
    init_args(args)
    validate_reduce_object_args(args)

    verbose = args.get('VERBOSE', True)

    # --- Create IM frame from raw ---
    raw_fname = args.get('RAW_FILENAME')
    im_fname = args.get('IMAGE_FILENAME')

    if raw_fname:
        make_im(raw_fname, im_fname, **args)
    else:
        logger.warning(
            "RAW_FILENAME not in args; skipping MAKE_IM "
            "(assuming IM file already exists)"
        )

    # --- Double-pass cosmic ray rejection (requires OPTEX + residual map) ---
    dbl_pass_crr_extr = args.get('DPCRREX', False)
    operat = args.get('EXTR_OPERATION', '')
    make_res = args.get('OPTEX_MKRES', False)
    is_optex_based = operat in ('OPTEX', 'SCMOPTEX', 'SMCOPTEX')

    if dbl_pass_crr_extr and is_optex_based and make_res:
        logger.info("Performing double-pass cosmic ray rejection extraction")
        make_ex(args)
        _clean_im(args)
    elif dbl_pass_crr_extr:
        logger.warning(
            "DPCRREX requested but OPTEX or OPTEX_MKRES not selected — ignoring"
        )

    # --- Create EX frame from IM ---
    make_ex(args)

    ex_filename = args.get('EXTRAC_FILENAME')
    red_filename = args.get('OUTPUT_FILENAME')
    if not ex_filename or not red_filename:
        raise ValueError("EXTRAC_FILENAME and OUTPUT_FILENAME must be specified.")

    if verbose:
        logger.info("=" * 50)
        logger.info("Reducing object spectra from extraction file")
        logger.info("=" * 50)
        logger.info("Extraction file = %s", ex_filename)

    # --- Skyline recalibration (if requested) ---
    if args.get('USE_GENCAL', False):
        _skylines_recalibration(ex_filename, args)

    # --- Create RED frame by copying EX ---
    logger.info("Creating RED file %s from %s", red_filename, ex_filename)
    shutil.copyfile(ex_filename, red_filename)

    # --- Skyline calibration test ---
    if args.get('USE_GENCAL', False) and args.get('TST_SKYCAL', False):
        _skycalib_test(red_filename, args)

    # --- Divide by fiber flat-field (if flat file provided) ---
    _flatfield(red_filename, args)

    # --- Scrunch (rebin to linear wavelength grid) ---
    _scrunch(red_filename, args)

    # --- Fiber throughput calibration and sky subtraction ---
    # (not applicable for Nod & Shuffle data)
    is_nod_shuffle = _check_nod_shuffle(red_filename)

    if not is_nod_shuffle:
        _throughput_calibrate(red_filename, args)

    if args.get('INC_RWSS', False):
        _make_rwss(red_filename)

    if not is_nod_shuffle:
        if args.get('SKYSUB', True):
            _skysub(red_filename, args)
        if args.get('SKYSPRSMP', False):
            _super_skysub(red_filename, ex_filename, args)

    # --- Clean up intermediate PIXCAL HDU ---
    _delete_pixcal(red_filename)

    # --- Telluric correction ---
    if args.get('TELCOR', False):
        _telluric_correct(red_filename, args)

    # --- Velocity correction ---
    if args.get('VELCOR', False):
        _velocity_correct(red_filename, args)

    # --- PCA sky subtraction ---
    if not is_nod_shuffle and args.get('SKYSUB_PCA', False):
        _skysub_pca(red_filename, args)

    # --- Flux calibration ---
    if args.get('CALIBFLUX', False):
        _apply_fluxcal(red_filename, args)

    # --- Transfer function correction ---
    if args.get('TRANSFUNC', False):
        _apply_transfer_function(red_filename, args)

    # --- De-wiggle ---
    if args.get('DEWIGGLE', False):
        _dewiggle(red_filename, args)

    # --- Finalize: write metadata and mark as reduced ---
    _write_reduction_args(red_filename, args)
    _set_reduced_status(red_filename)
    _stamp_pipeline_version(red_filename)

    logger.info("Object frame reduced")
    if verbose:
        logger.info("Reduction file %s created.", red_filename)


# ---------------------------------------------------------------------
# Flux Calibration
# ---------------------------------------------------------------------

def _apply_fluxcal(red_filename: str, args: Dict[str, Any]) -> None:
    """Apply spectrophotometric flux calibration to a reduced frame.

    Identifies standard-star fibers (TYPE='C'), matches to BOSZ templates,
    derives per-star calibration vectors, combines them, and applies the
    result to all fibers.  Writes back to the RED file in place.

    Parameters
    ----------
    red_filename : str
        Path to the reduced FITS file (modified in place).
    args : dict
        Reduction arguments.  Relevant keys:

        - ``CALIBFLUX_CATALOG`` : str — path to standard-star CSV catalog
        - ``CALIBFLUX_FWHM`` : float — instrument FWHM in Å (default: from header SPECFWHM)
        - ``CALIBFLUX_METRIC`` : str — scoring metric (default: ``"chi2"``)
        - ``CALIBFLUX_SMOOTH`` : bool — smooth combined vector (default: False)
    """
    from .constants import FIBER_TYPE_CALIBRATION
    from .io.image import ImageFile
    from .fluxcal.calibration import (
        compute_calibration_vector_for_star,
        combine_calibration_vectors,
        apply_flux_calibration,
    )
    from .fluxcal.photometry import (
        load_filter_curves,
        load_standard_star_catalog,
        photometry_from_catalog_row,
        DEFAULT_BANDS,
    )
    from .fluxcal.templates import TemplateLibrary
    from .fluxcal.masks import load_mask_regions
    from .fluxcal.containers import Spectrum1D

    import numpy as np

    # --- Load the RED file ---
    with ImageFile(red_filename, mode='UPDATE') as red_file:
        spectra = red_file.read_image_data()     # (NFIB, NPIX) or (NPIX, NFIB)
        variance = red_file.read_variance_data()
        fiber_types, nf = red_file.read_fiber_types(1000)
        wave_data = red_file.read_wave_data()     # wavelength solution

        # Determine wavelength axis (1-D common grid for scrunched data)
        if wave_data is not None and wave_data.ndim == 1:
            wavelength = wave_data
        elif wave_data is not None and wave_data.ndim == 2:
            wavelength = wave_data[0]  # use first fiber's wavelength
        else:
            nx, _ = red_file.get_size()
            wavelength = np.arange(nx, dtype=float)
            logger.warning("No wavelength solution found; using pixel indices")

        # Determine data layout
        if spectra.shape[0] == len(wavelength):
            # (NPIX, NFIB) — transpose to (NFIB, NPIX) for processing
            spectra = spectra.T
            variance = variance.T
            layout = "npix_nfib"
        else:
            layout = "nfib_npix"

        nfib, npix = spectra.shape

        # --- Identify standard-star fibers ---
        std_indices = [
            i for i in range(min(nfib, len(fiber_types)))
            if fiber_types[i] == FIBER_TYPE_CALIBRATION
        ]

        if not std_indices:
            logger.warning(
                "CALIBFLUX requested but no fibers with TYPE='C' found. "
                "Skipping flux calibration."
            )
            return

        logger.info(
            "Flux calibration: %d standard-star fibers (TYPE='C'): %s",
            len(std_indices), std_indices,
        )

        # --- Load resources ---
        catalog_path = args.get('CALIBFLUX_CATALOG')
        if not catalog_path:
            logger.error(
                "CALIBFLUX_CATALOG not set. "
                "Provide the path to the standard-star photometry CSV."
            )
            return

        catalog = load_standard_star_catalog(catalog_path)
        if len(catalog) == 0:
            logger.error("Standard-star catalog is empty: %s", catalog_path)
            return

        library = TemplateLibrary()
        filter_curves = load_filter_curves(DEFAULT_BANDS)
        mask_regions = load_mask_regions("telluric_default")

        instrument_fwhm = args.get('CALIBFLUX_FWHM')
        if instrument_fwhm is None:
            instrument_fwhm = red_file.get_header_value('SPECFWHM')
            if instrument_fwhm is None:
                instrument_fwhm = 3.0
                logger.warning("Using default FWHM=%.1f Å", instrument_fwhm)
            else:
                instrument_fwhm = float(instrument_fwhm)

        metric = args.get('CALIBFLUX_METRIC', 'chi2')
        smooth = args.get('CALIBFLUX_SMOOTH', False)

        # --- Compute per-star calibration vectors ---
        cal_vectors = []
        fiber_table = red_file.read_fiber_table() if red_file.has_fiber_table() else None

        for idx, fib_idx in enumerate(std_indices):
            # Extract observed spectrum for this fiber
            obs_flux = spectra[fib_idx, :]
            obs_var = variance[fib_idx, :]
            obs_mask = np.isfinite(obs_flux) & (obs_var >= 0) & np.isfinite(obs_var)

            obs_spec = Spectrum1D(
                wavelength=wavelength.copy(),
                flux=obs_flux.copy(),
                variance=np.where(obs_var > 0, obs_var, 0.0),
                mask=obs_mask,
                meta={"fiber_id": fib_idx},
            )

            # Get star name from fiber table
            star_name = ""
            if fiber_table is not None:
                try:
                    star_name = str(fiber_table["NAME"][fib_idx]).strip()
                except (KeyError, IndexError):
                    pass

            # Match to catalog row (by index for now; positional matching is TODO)
            if idx < len(catalog):
                row = catalog[idx]
            else:
                logger.warning(
                    "More standard fibers (%d) than catalog rows (%d); "
                    "skipping fiber %d",
                    len(std_indices), len(catalog), fib_idx,
                )
                continue

            phot = photometry_from_catalog_row(row)

            try:
                cal_vec = compute_calibration_vector_for_star(
                    obs_spec, phot, library, filter_curves,
                    instrument_fwhm_angstrom=instrument_fwhm,
                    mask_regions=mask_regions,
                    metric=metric,
                    star_name=star_name,
                    fiber_id=fib_idx,
                )
                cal_vectors.append(cal_vec)
            except Exception as exc:
                logger.warning(
                    "Calibration failed for fiber %d (%s): %s",
                    fib_idx, star_name, exc,
                )
                continue

        if not cal_vectors:
            logger.error(
                "All standard-star calibrations failed. "
                "Skipping flux calibration."
            )
            return

        # --- Combine and apply ---
        result = combine_calibration_vectors(
            cal_vectors, method="weighted_mean", smooth=smooth,
        )

        cal_spectra, cal_variance, header_updates = apply_flux_calibration(
            spectra, variance, result,
        )

        # --- Write back ---
        if layout == "npix_nfib":
            red_file.write_image_data(cal_spectra.T)
            red_file.write_variance_data(cal_variance.T)
        else:
            red_file.write_image_data(cal_spectra)
            red_file.write_variance_data(cal_variance)

        # Update header
        for key, val in header_updates.items():
            if key == "HISTORY":
                for h in val:
                    red_file.set_header_value("HISTORY", h)
            else:
                value, comment = val
                red_file.set_header_value(key, value, comment=comment)

        logger.info(
            "Flux calibration complete: %d standards, RMS=%.4f",
            result.summary["n_stars_used"],
            result.summary["rms_scatter"],
        )


# =====================================================================
# P0 — Implemented Functions
# =====================================================================

def _scrunch(red_filename: str, args: Dict[str, Any]) -> None:
    """Rebin the object frame to a linear wavelength grid using the arc
    wavelength solution.

    Reads ``WAVEL_FILENAME`` from *args* to locate the calibrated arc
    RED file and delegates to :func:`wavecal.scrunch.scrunch_from_arc_id`.
    """
    arc_filename = args.get('WAVEL_FILENAME')
    if not arc_filename:
        logger.warning("WAVEL_FILENAME not set — skipping scrunch")
        return
    if not Path(arc_filename).exists():
        logger.warning("Arc file %s not found — skipping scrunch", arc_filename)
        return

    scrunch_from_arc_id(red_filename, arc_filename, args, reverse=False)
    logger.info("Scrunched %s using arc %s", red_filename, arc_filename)


def _check_nod_shuffle(red_filename: str) -> bool:
    """Return True if the observation used Nod & Shuffle mode.

    Checks the ``UTNODSFL`` header keyword.  Falls back to False (standard
    mode) when the keyword is absent.
    """
    with ImageFile(red_filename, mode='READ') as f:
        flag = f.get_header_value('UTNODSFL', None)
    if flag is not None:
        return str(flag).strip().upper() in ('T', 'TRUE', '1', 'Y')
    return False


def _delete_pixcal(red_filename: str) -> None:
    """Remove the intermediate PIXCAL HDU if present."""
    with ImageFile(red_filename, mode='UPDATE') as f:
        if f.delete_hdu('PIXCAL'):
            logger.info("Deleted PIXCAL HDU from %s", red_filename)


def _write_reduction_args(red_filename: str, args: Dict[str, Any]) -> None:
    """Persist selected reduction arguments as FITS header keywords.

    Writes each arg as ``HIERARCH DRARG <KEY> = <value>`` so the
    provenance of the reduction is recorded in the file.
    """
    _SKIP_KEYS = {'RAW_FILENAME', 'IMAGE_FILENAME', 'EXTRAC_FILENAME',
                   'OUTPUT_FILENAME'}
    with ImageFile(red_filename, mode='UPDATE') as f:
        for key, value in args.items():
            if key in _SKIP_KEYS:
                continue
            hdr_key = f"HIERARCH DRARG {key}"
            try:
                f.set_header_value(hdr_key, value)
            except (ValueError, TypeError):
                f.set_header_value(hdr_key, str(value))


def _set_reduced_status(red_filename: str) -> None:
    """Mark the output file as reduced by setting the DRSTATUS keyword."""
    with ImageFile(red_filename, mode='UPDATE') as f:
        f.set_header_value('DRSTATUS', 'REDUCED', comment='Reduction status')


def _stamp_pipeline_version(red_filename: str) -> None:
    """Write the kspecdr pipeline version into the FITS header."""
    from . import __version__
    with ImageFile(red_filename, mode='UPDATE') as f:
        f.set_header_value('DRPIPVER', __version__,
                           comment='kspecdr pipeline version')
        f.add_history(f"Reduced with kspecdr {__version__}")


# =====================================================================
# P1 — Implemented Functions
# =====================================================================

def _flatfield(red_filename: str, args: Dict[str, Any]) -> None:
    """Divide by fiber flat-field response (P1-1: cmfspec_flatfield).

    Reads the master FFLAT RED file and divides the science spectra and
    variance in place.  Implements the full Taylor-expansion variance
    propagation from 2dfdr ``CMFSPEC_FLATFIELD``.

    The flat must be in pixel space (un-scrunched); flat-fielding happens
    **before** scrunching in the reduction order.
    """

    if not args.get('USEFFLAT', True):
        with ImageFile(red_filename, mode='UPDATE') as f:
            f.add_history('Not divided by fibre flat field')
        return

    fflat_fname = args.get('FFLAT_FILENAME')
    if not fflat_fname or not str(fflat_fname).strip():
        logger.error("FFLAT_FILENAME not set — skipping flat-field division")
        return

    truncflat    = args.get('TRUNCFLAT', False)
    useflatstart = int(args.get('USEFLATSTART', 1))
    useflatend   = int(args.get('USEFLATEND',   2048))

    with ImageFile(red_filename, mode='UPDATE') as red_f, \
         ImageFile(fflat_fname,  mode='READ')   as flt_f:

        obj_img = red_f.read_image_data()    # (NFIB, NPIX)
        obj_var = red_f.read_variance_data()
        flt_img = flt_f.read_image_data().copy()
        flt_var = flt_f.read_variance_data().copy()

        if obj_img.shape != flt_img.shape:
            raise ValueError(
                f"Flat shape {flt_img.shape} != object shape {obj_img.shape}. "
                "Flat must be reduced with the same TLM as the science frame."
            )

        # Apply truncation: pixels outside the trusted range divide by 1.0
        if truncflat:
            p0 = useflatstart - 1   # 1-based → 0-based lower bound (inclusive)
            p1 = useflatend          # 1-based → 0-based upper bound (exclusive)
            if p0 > 0:
                flt_img[:, :p0] = 1.0
                flt_var[:, :p0] = 0.0
            if p1 < obj_img.shape[1]:
                flt_img[:, p1:] = 1.0
                flt_var[:, p1:] = 0.0

        good = (flt_img != 0.0) & np.isfinite(flt_img) & np.isfinite(obj_img)

        # Divide image
        out_img = np.where(good, obj_img / flt_img, np.nan).astype(np.float32)

        # Full Taylor-expansion variance propagation (2dfdr formula):
        #   Var_out = (1/flat)^2 * Var_obj + (obj_divided/flat^2)^2 * Var_flat
        # Note: obj_divided = out_img = obj_img/flat, so second term becomes
        #       (obj_orig/flat^3)^2 * Var_flat
        good_var = good & np.isfinite(obj_var) & np.isfinite(flt_var)
        out_var = np.where(
            good_var,
            (1.0 / flt_img) ** 2 * obj_var
            + (out_img / flt_img ** 2) ** 2 * flt_var,
            np.nan,
        ).astype(np.float32)

        red_f.write_image_data(out_img)
        red_f.write_variance_data(out_var)
        red_f.add_history(f'Divided by fibre flat field {fflat_fname}')
        if truncflat:
            red_f.add_history(
                f'Flat truncated to pixel range {useflatstart}:{useflatend}'
            )

    logger.info("Applied flat-field from %s to %s", fflat_fname, red_filename)


def _throughput_calibrate(red_filename: str, args: Dict[str, Any]) -> None:
    """Per-fiber throughput correction (P1-2: cmfspec_ftpcal).

    Calculates relative fiber throughputs and divides each fiber spectrum
    (and variance) by its throughput.  Writes a ``THPUT`` ImageHDU (1-D,
    length NFIB) to the RED file for downstream diagnostics.

    Methods (``TPMETH`` arg, default ``'OFFSKY'``):
    - ``'OFF'``: fix all fiber throughputs to 1.0 (no correction; THPUT HDU still written)
    - ``'OFFSKY'``: read pre-computed throughputs from ``THPUT_FILENAME``
    - ``'SKYLINE(KGB)'``: Karl Glazebrook's sky-line robust-fit algorithm
    - ``'MEDIAN'``: simple per-fiber mean, normalized by median
      (equivalent to 2dfdr ``UMFSPEC_FTPC``)

    Bad throughputs (NaN / outside 0.01–100) are stored as 0.0 in the
    THPUT extension and their spectra are set to zero (2dfdr convention).
    """
    from astropy.io import fits
    from .utils.fiber import get_override_from_args

    if not args.get('THRUPUT', True):
        with ImageFile(red_filename, mode='UPDATE') as f:
            f.add_history('No throughput calibration performed')
        return

    tpmeth = str(args.get('TPMETH', 'OFFSKY')).upper().strip()

    with ImageFile(red_filename, mode='UPDATE') as red_f:
        spec = red_f.read_image_data()    # (NFIB, NPIX)
        var  = red_f.read_variance_data()
        overrides = get_override_from_args(args)
        fiber_types, _ = red_f.read_fiber_types(1000, overrides=overrides)
        nfib = spec.shape[0]
        fiber_types_arr = np.array(
            list(fiber_types[:nfib]), dtype='U1'
        )

        thput_vec = np.full(nfib, np.nan, dtype=np.float64)

        # --- optional external override file ---
        use_external = args.get('USETHPTFILE', False)
        loaded_external = False
        if use_external:
            import os
            if os.path.exists('THROUGHPUT.fits'):
                with fits.open('THROUGHPUT.fits') as hdul:
                    for hdu in hdul:
                        if 'THPUT' in hdu.name.upper() and hdu.data is not None:
                            n = min(len(hdu.data), nfib)
                            thput_vec[:n] = hdu.data[:n]
                            loaded_external = True
                            break
                if loaded_external:
                    logger.info("Throughput: loaded from THROUGHPUT.fits")
                    tpmeth = '_EXTERNAL'

        # --- compute throughputs if not loaded from external file ---
        if not loaded_external:
            if tpmeth == 'OFFSKY':
                thput_fname = str(args.get('THPUT_FILENAME', '')).strip()
                if not thput_fname:
                    logger.warning(
                        "TPMETH=OFFSKY but THPUT_FILENAME not set — "
                        "skipping throughput calibration"
                    )
                    red_f.add_history('No throughput calibration performed')
                    return
                with fits.open(thput_fname) as hdul:
                    thput_hdu = None
                    for hdu in hdul:
                        if 'THPUT' in hdu.name.upper() and hdu.data is not None:
                            thput_hdu = hdu
                            break
                    if thput_hdu is None:
                        logger.warning(
                            "No THPUT extension found in %s — "
                            "skipping throughput calibration", thput_fname
                        )
                        red_f.add_history('No throughput calibration performed')
                        return
                    n = min(len(thput_hdu.data), nfib)
                    thput_vec[:n] = thput_hdu.data[:n]
                logger.info("Throughput: loaded from %s (OFFSKY)", thput_fname)

            elif tpmeth == 'SKYLINE(KGB)':
                thput_vec = _get_thput_kgb(spec, fiber_types_arr)
                logger.info("Throughput: KGB sky-line algorithm")

            elif tpmeth in ('MEDIAN', 'UMFSPEC'):
                thput_vec = _umfspec_ftpc(spec)
                logger.info("Throughput: per-fiber median (UMFSPEC)")

            elif tpmeth == 'OFF':
                thput_vec[:] = 1.0
                logger.info("Throughput: OFF (all fibers fixed to 1.0)")

            else:
                logger.warning(
                    "Unknown TPMETH '%s' — skipping throughput calibration", tpmeth
                )
                red_f.add_history('No throughput calibration performed')
                return

        # --- sanity check ---
        bad = ~np.isfinite(thput_vec) | (thput_vec < 0.01) | (thput_vec > 100.0)
        thput_vec[bad] = np.nan
        n_bad = int(np.sum(bad))
        if n_bad:
            logger.warning("Throughput: %d fibers have bad/out-of-range values", n_bad)

        # --- divide spectra by throughput ---
        for i in range(nfib):
            if np.isfinite(thput_vec[i]) and thput_vec[i] > 0:
                scale = 1.0 / thput_vec[i]
            else:
                scale = 0.0    # 2dfdr: bad throughput → zero spectrum
            spec[i] = np.where(np.isfinite(spec[i]), spec[i] * scale, np.nan)
            var[i]  = np.where(np.isfinite(var[i]),  var[i]  * scale**2, np.nan)

        red_f.write_image_data(spec.astype(np.float32))
        red_f.write_variance_data(var.astype(np.float32))

        # Write THPUT extension (bad → 0.0 per 2dfdr convention)
        thput_out = np.where(np.isfinite(thput_vec), thput_vec, 0.0).astype(np.float32)
        _write_image_hdu(red_f, 'THPUT', thput_out)

        hist_map = {
            '_EXTERNAL':    'Throughput calibration using THROUGHPUT.fits',
            'OFF':          'Throughput calibration disabled (all fibers = 1.0)',
            'OFFSKY':       f'Throughput calibration using OFFSKY from '
                            f'{args.get("THPUT_FILENAME", "")}',
            'MEDIAN':       'Throughput calibration using per-fiber median (UMFSPEC)',
            'UMFSPEC':      'Throughput calibration using per-fiber median (UMFSPEC)',
            'SKYLINE(KGB)': 'Throughput calibration using sky lines (KGB method)',
        }
        red_f.add_history(hist_map.get(tpmeth, f'Throughput calibration ({tpmeth})'))

    logger.info("Throughput calibrated %s (method=%s)", red_filename, tpmeth)


def _make_rwss(red_filename: str) -> None:
    """Copy spectra to RWSS HDU before sky subtraction (P1-4).

    Saves a snapshot of the current (throughput-corrected, pre-sky) spectra
    so the before/after sky subtraction comparison is available in the same
    RED file.  Only executed when ``INC_RWSS=True`` (default: False).
    """
    with ImageFile(red_filename, mode='UPDATE') as f:
        data = f.read_image_data()    # (NFIB, NPIX)
        _write_image_hdu(f, 'RWSS', data.astype('float32'))
    logger.info("Saved RWSS (pre-sky) snapshot in %s", red_filename)


def _skysub(red_filename: str, args: Dict[str, Any]) -> None:
    """Sky subtraction using sky fibers (P1-3: SKYSUB).

    Identifies sky fibers (``TYPE='S'`` in the FIBRES table), rejects those
    with more than 1/8 bad pixels, combines them into a master sky spectrum,
    and subtracts it from all fibers.  The combined sky is written to a
    ``SKY`` ImageHDU.

    Combination method is controlled by ``SKYCOMBINE`` arg:

    - ``'MEAN'`` (default): straight mean of sky fibers.
    - ``'MEDIAN'``: median combination.
    - ``'SIGCLIP'``: iterative sigma-clipping mean.  Clipping threshold and
      maximum iterations are controlled by ``SKYCOMBINE_SIGMA`` (default 3.0)
      and ``SKYCOMBINE_ITERS`` (default 5).

    Variance propagation: ``Var_out = Var_fib + Var_sky`` where ``Var_sky``
    already accounts for the combination of N_sky fibers.
    """
    import numpy as np
    from pathlib import Path
    from .utils.fiber import get_override_from_args

    if not args.get('SKYSUB', True):
        with ImageFile(red_filename, mode='UPDATE') as f:
            f.add_history('No sky subtraction performed')
        return

    combine_method = str(args.get('SKYCOMBINE', 'MEAN')).upper().strip()
    if combine_method not in ('MEAN', 'MEDIAN', 'SIGCLIP'):
        logger.warning(
            "Unknown SKYCOMBINE '%s' — defaulting to MEAN", combine_method
        )
        combine_method = 'MEAN'

    with ImageFile(red_filename, mode='UPDATE') as red_f:
        spec = red_f.read_image_data()    # (NFIB, NPIX)
        var  = red_f.read_variance_data()
        overrides = get_override_from_args(args)
        fiber_types, _ = red_f.read_fiber_types(1000, overrides=overrides)
        nfib = spec.shape[0]

        # Identify sky fibers from the FIBRES table
        sky_fibs = [
            i for i in range(nfib)
            if i < len(fiber_types) and fiber_types[i] == 'S'
        ]

        # Fallback: read from skyfibres.dat if no FIBRES table
        if not sky_fibs and not red_f.has_fiber_table():
            dat = Path('skyfibres.dat')
            if dat.exists():
                lines = dat.read_text().splitlines()
                sky_fibs = [
                    int(line.strip()) - 1   # 1-based → 0-based
                    for line in lines
                    if line.strip().isdigit()
                ]
                logger.info(
                    "Sky fibers: loaded %d from skyfibres.dat", len(sky_fibs)
                )

        if not sky_fibs:
            logger.warning(
                "No sky fibers found — skipping sky subtraction"
            )
            red_f.add_history('No sky subtraction: no sky fibers found')
            return

        # FCHECK: reject fibers with > 1/8 bad pixels (2dfdr criterion)
        good_sky = [
            i for i in sky_fibs
            if np.sum(~np.isfinite(spec[i])) < spec.shape[1] / 8
        ]
        if not good_sky:
            logger.warning(
                "All %d sky fibers have too many bad pixels — "
                "skipping sky subtraction", len(sky_fibs)
            )
            red_f.add_history(
                'No sky subtraction: all sky fibers have too many bad pixels'
            )
            return

        logger.info(
            "Sky subtraction: %d/%d sky fibers pass quality check (%s)",
            len(good_sky), len(sky_fibs), combine_method,
        )

        # Combine sky fibers into a single sky spectrum and sky variance
        sky_stack = spec[good_sky, :]    # (N_sky, NPIX)
        var_stack  = var[good_sky, :]
        n_sky = len(good_sky)

        if combine_method == 'MEDIAN':
            sky_spec = np.nanmedian(sky_stack, axis=0)
            # Variance of median ≈ (π/2) * mean_var / N
            sky_var  = (np.pi / 2.0) * np.nanmean(var_stack, axis=0) / n_sky
        elif combine_method == 'SIGCLIP':
            sigma     = float(args.get('SKYCOMBINE_SIGMA', 3.0))
            max_iters = int(args.get('SKYCOMBINE_ITERS', 5))
            sky_spec, sky_var = _sigclip_combine(
                sky_stack, var_stack, sigma=sigma, max_iters=max_iters
            )
            logger.info(
                "Sky sigma-clipping: sigma=%.1f, max_iters=%d", sigma, max_iters
            )
        else:  # MEAN
            sky_spec = np.nanmean(sky_stack, axis=0)
            # Error propagation for mean of N independent measurements
            sky_var  = np.nansum(var_stack, axis=0) / n_sky ** 2

        # Subtract sky from every fiber and propagate variance
        for fib in range(nfib):
            bad = ~np.isfinite(spec[fib]) | ~np.isfinite(sky_spec)
            spec[fib, ~bad] -= sky_spec[~bad]
            var[fib, ~bad]  += sky_var[~bad]
            spec[fib, bad]   = np.nan
            var[fib, bad]    = np.nan

        red_f.write_image_data(spec.astype(np.float32))
        red_f.write_variance_data(var.astype(np.float32))

        # Write combined sky to SKY extension (bad pixels → 0.0)
        sky_out = np.where(np.isfinite(sky_spec), sky_spec, 0.0).astype(np.float32)
        _write_image_hdu(red_f, 'SKY', sky_out)

        red_f.add_history(
            f'Sky subtracted using {n_sky} sky fibers ({combine_method})'
        )

    logger.info(
        "Sky subtracted %s using %d fibers", red_filename, len(good_sky)
    )


# =====================================================================
# P1 — Private Helpers
# =====================================================================

def _sigclip_combine(
    sky_stack: 'np.ndarray',
    var_stack: 'np.ndarray',
    sigma: float = 3.0,
    max_iters: int = 5,
) -> 'tuple[np.ndarray, np.ndarray]':
    """Iterative sigma-clipping mean combination of sky fibers.

    For each wavelength pixel, computes the mean and standard deviation of the
    contributing sky fibers and masks values that deviate more than ``sigma``
    standard deviations from the mean.  Iteration continues until no new
    pixels are clipped or ``max_iters`` is reached.

    Parameters
    ----------
    sky_stack : ndarray, shape (N_sky, NPIX)
        Sky fiber spectra.
    var_stack : ndarray, shape (N_sky, NPIX)
        Corresponding variance arrays.
    sigma : float
        Clipping threshold in units of standard deviations.
    max_iters : int
        Maximum number of clipping iterations.

    Returns
    -------
    sky_spec : ndarray, shape (NPIX,)
        Sigma-clipped mean sky spectrum.
    sky_var : ndarray, shape (NPIX,)
        Propagated variance of the combined sky spectrum.
    """
    mask = ~np.isfinite(sky_stack)   # True = bad/clipped

    for _ in range(max_iters):
        work = np.where(mask, np.nan, sky_stack)
        mean = np.nanmean(work, axis=0)           # (NPIX,)
        std  = np.nanstd(work, axis=0)            # (NPIX,)
        new_mask = mask | (np.abs(sky_stack - mean[np.newaxis, :]) > sigma * std[np.newaxis, :])
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    work     = np.where(mask, np.nan, sky_stack)
    var_work = np.where(mask, np.nan, var_stack)
    n_eff    = np.sum(~mask, axis=0).clip(min=1).astype(float)   # (NPIX,)

    sky_spec = np.nanmean(work, axis=0)
    sky_var  = np.nansum(var_work, axis=0) / n_eff ** 2

    return sky_spec, sky_var


def _write_image_hdu(f: ImageFile, name: str, data: 'np.ndarray') -> None:
    """Add or overwrite a named ImageHDU inside an open ImageFile context."""
    from astropy.io import fits
    name_upper = name.upper()
    for idx, hdu in enumerate(f.hdul):
        if hdu.name.upper() == name_upper and idx > 0:
            hdu.data = data
            return
    f.hdul.append(fits.ImageHDU(data=data, name=name_upper))


def _umfspec_ftpc(spec: 'np.ndarray') -> 'np.ndarray':
    """Per-fiber mean normalized by the global median (UMFSPEC_FTPC).

    This is the 2dfdr ``UMFSPEC_FTPC`` algorithm used as a first-pass
    throughput estimate.  Values ≤ 0.05 (parked fibers) are set to NaN.
    The returned vector is normalized so that the median of good values is 1.
    """
    nfib = spec.shape[0]
    ftpc = np.full(nfib, np.nan, dtype=np.float64)
    for j in range(nfib):
        row  = spec[j]
        good = np.isfinite(row) & (row == row)
        if np.sum(good) == 0:
            continue
        mean_val = float(np.mean(row[good]))
        if mean_val <= 0.05:    # parked / unused fiber
            continue
        ftpc[j] = mean_val
    valid = np.isfinite(ftpc)
    if np.sum(valid) > 0:
        med = float(np.median(ftpc[valid]))
        if med > 0:
            ftpc[valid] /= med
    return ftpc


def _subtract_continuum(spec: 'np.ndarray', hw: int = 100) -> 'np.ndarray':
    """Subtract a local median continuum (running window ±hw pixels).

    NaN pixels are filled with the global median before filtering, then
    restored.  Mirrors 2dfdr ``SUBTRACT_MED_FILT`` (box = ±100 pixels).
    """
    from scipy.signal import medfilt
    nan_mask = ~np.isfinite(spec)
    tmp = spec.copy()
    if nan_mask.any():
        fill = float(np.nanmedian(spec)) if np.any(~nan_mask) else 0.0
        tmp[nan_mask] = fill
    ksize = 2 * hw + 1
    # kernel must be odd and ≤ spectrum length
    if ksize > len(tmp):
        ksize = len(tmp) if len(tmp) % 2 == 1 else max(1, len(tmp) - 1)
    continuum = medfilt(tmp, ksize)
    result = spec - continuum
    result[nan_mask] = np.nan
    return result


def _boxcar1d(spec: 'np.ndarray', width: int = 5) -> 'np.ndarray':
    """Boxcar (top-hat) smoothing of a 1-D spectrum, NaN-aware."""
    from scipy.ndimage import uniform_filter1d
    nan_mask = ~np.isfinite(spec)
    tmp = np.where(nan_mask, 0.0, spec)
    cnt = np.where(nan_mask, 0.0, 1.0)
    sm_sum = uniform_filter1d(tmp, size=width, mode='nearest') * width
    sm_cnt = uniform_filter1d(cnt, size=width, mode='nearest') * width
    result = np.where(sm_cnt > 0, sm_sum / sm_cnt, np.nan)
    result[nan_mask] = np.nan
    return result


def _get_thput_kgb(
    spec: 'np.ndarray', fiber_types: 'np.ndarray'
) -> 'np.ndarray':
    """Karl Glazebrook's sky-line throughput algorithm (SKYLINE(KGB)).

    Algorithm (mirrors 2dfdr ``GET_THPUT_KGB``):
    1. First-pass throughput estimate via ``_umfspec_ftpc``.
    2. Build a median sky spectrum from sky fibers, each normalized by the
       first-pass estimate.
    3. Subtract continuum (±100-px running median) from the sky and each
       fiber spectrum, then apply a 5-pixel boxcar smooth.
    4. For each P/S fiber, fit a robust line (Siegel slope) of the
       continuum-subtracted fiber spectrum vs. the median sky.  The slope B
       is the throughput.
    5. Validate 0.01 < B < 100; normalize the whole vector by its median.
    """
    nfib, npix = spec.shape

    # Step 1: first-pass estimate
    thput_init = _umfspec_ftpc(spec)

    # Step 2: identify sky fibers and build median sky
    sky_fibs = [
        i for i in range(nfib)
        if i < len(fiber_types) and fiber_types[i] == 'S'
    ]
    if not sky_fibs:
        logger.warning(
            "KGB throughput: no sky fibers — falling back to MEDIAN"
        )
        return thput_init

    sky_rows = []
    for sf in sky_fibs:
        tp = float(thput_init[sf]) if np.isfinite(thput_init[sf]) and thput_init[sf] > 0 else 1.0
        sky_rows.append(spec[sf] / tp)
    sky_med = np.nanmedian(np.array(sky_rows), axis=0)  # (NPIX,)

    # Step 3: continuum-subtract and smooth the median sky
    sky_cs = _subtract_continuum(sky_med, hw=100)
    sky_sm = _boxcar1d(sky_cs, width=5)

    # Step 4: fit each P/S fiber
    thput_out = np.full(nfib, np.nan, dtype=np.float64)
    for i in range(nfib):
        ft = fiber_types[i] if i < len(fiber_types) else 'N'
        if ft not in ('P', 'S'):
            continue
        fib_cs = _subtract_continuum(spec[i].copy(), hw=100)
        fib_sm = _boxcar1d(fib_cs, width=5)
        ok = np.isfinite(fib_sm) & np.isfinite(sky_sm)
        if np.sum(ok) < 20:
            logger.debug(
                "KGB: fiber %d has only %d good pixels — skipping", i, int(np.sum(ok))
            )
            continue
        x = sky_sm[ok]
        y = fib_sm[ok]
        try:
            from scipy.stats import siegelslopes
            B = float(siegelslopes(y, x).slope)
        except Exception:
            # Fallback: OLS slope (no intercept) through origin
            denom = float(np.dot(x, x))
            B = float(np.dot(x, y) / denom) if denom != 0 else 0.0
        if 0.01 < B < 100.0:
            thput_out[i] = B

    # Step 5: normalize by median
    valid = np.isfinite(thput_out)
    if np.sum(valid) > 0:
        med = float(np.median(thput_out[valid]))
        if med > 0:
            normed = thput_out[valid] / med
            thput_out[valid] = np.where(
                (normed > 0.01) & (normed < 100.0), normed, np.nan
            )

    return thput_out


# =====================================================================
# P2+ — Not Yet Implemented (safe no-ops)
# =====================================================================

def _clean_im(args: Dict[str, Any]) -> None:
    """Clean the IM frame using the OPTEX residual map (not yet implemented)."""
    logger.warning("Double-pass CR cleaning not yet implemented — skipping")


def _skylines_recalibration(filename: str, args: Dict[str, Any]) -> None:
    """Fine-tune wavelength solution using sky emission lines (not yet implemented)."""
    logger.warning("Skyline recalibration not yet implemented — skipping")


def _skycalib_test(filename: str, args: Dict[str, Any]) -> None:
    """QC test for skyline wavelength calibration (not yet implemented)."""
    logger.warning("Skyline calibration test not yet implemented — skipping")


def _super_skysub(red_filename: str, ex_filename: str, args: Dict[str, Any]) -> None:
    """Super-sampled sky subtraction (not yet implemented)."""
    logger.warning("Super sky subtraction not yet implemented — skipping")


def _telluric_correct(red_filename: str, args: Dict[str, Any]) -> None:
    """Telluric absorption correction (not yet implemented)."""
    logger.warning("Telluric correction not yet implemented — skipping")


def _velocity_correct(red_filename: str, args: Dict[str, Any]) -> None:
    """Heliocentric/barycentric velocity correction (not yet implemented)."""
    logger.warning("Velocity correction not yet implemented — skipping")


def _skysub_pca(red_filename: str, args: Dict[str, Any]) -> None:
    """PCA-based sky subtraction (not yet implemented)."""
    logger.warning("PCA sky subtraction not yet implemented — skipping")


def _apply_transfer_function(red_filename: str, args: Dict[str, Any]) -> None:
    """Apply an associated transfer function (not yet implemented)."""
    logger.warning("Transfer function correction not yet implemented — skipping")


def _dewiggle(red_filename: str, args: Dict[str, Any]) -> None:
    """Remove sinusoidal fringing artifacts (not yet implemented)."""
    logger.warning("De-wiggle not yet implemented — skipping")
