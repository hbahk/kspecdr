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
# P1+ — Not Yet Implemented (safe no-ops)
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


def _flatfield(red_filename: str, args: Dict[str, Any]) -> None:
    """Divide by fiber flat-field response (not yet implemented)."""
    fflat_fname = args.get('FFLAT_FILENAME')
    if not fflat_fname:
        logger.info("No FFLAT_FILENAME provided — skipping flat-field division")
        return
    logger.warning(
        "Fiber flat-field division not yet implemented — skipping "
        "(FFLAT_FILENAME=%s)", fflat_fname
    )


def _throughput_calibrate(red_filename: str, args: Dict[str, Any]) -> None:
    """Per-fiber throughput correction (not yet implemented)."""
    logger.warning("Fiber throughput calibration not yet implemented — skipping")


def _make_rwss(red_filename: str) -> None:
    """Copy spectra to RWSS HDU before sky subtraction (not yet implemented)."""
    logger.warning("RWSS snapshot not yet implemented — skipping")


def _skysub(red_filename: str, args: Dict[str, Any]) -> None:
    """Median sky subtraction from sky fibers (not yet implemented)."""
    logger.warning("Sky subtraction not yet implemented — skipping")


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
