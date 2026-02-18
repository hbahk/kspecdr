"""
Calibration vector computation and application for kspecdr flux calibration.

Provides:
- :func:`scale_template_to_photometry` — absolute flux anchor via synthetic photometry
- :func:`compute_calibration_vector_for_star` — full per-star calibration pipeline
- :func:`combine_calibration_vectors` — robust combination of per-star vectors
- :func:`apply_flux_calibration` — apply calibration to all fibers with variance propagation

This is the top-level orchestration module (P2) that ties together P0 and P1.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .containers import (
    CalibrationVector,
    FilterCurve,
    FluxCalibrationResult,
    Photometry,
    Spectrum1D,
)
from .masks import load_mask_regions
from .matching import select_best_template
from .photometry import (
    DEFAULT_BANDS,
    load_filter_curves,
    synthetic_photometry,
)
from .templates import TemplateLibrary, prepare_template

logger = logging.getLogger(__name__)

__all__ = [
    "scale_template_to_photometry",
    "compute_calibration_vector_for_star",
    "combine_calibration_vectors",
    "apply_flux_calibration",
]


# ---------------------------------------------------------------------------
# Photometric scaling
# ---------------------------------------------------------------------------

def scale_template_to_photometry(
    template_spec: Spectrum1D,
    photometry: Photometry,
    filter_curves: Dict[str, FilterCurve],
) -> Tuple[float, float, Dict[str, float]]:
    """Scale a template so its synthetic photometry matches observed magnitudes.

    The scale factor is determined by minimising the weighted mean squared
    difference between synthetic and observed magnitudes across all valid
    bands.  Because magnitudes are logarithmic, the scale factor is computed
    in flux space:

    .. math::

        s = 10^{-0.4 \\, \\Delta m}

    where ``Δm`` is the weighted-mean offset (observed − synthetic).

    Parameters
    ----------
    template_spec : Spectrum1D
        Template spectrum on the observed wavelength grid (output of
        :func:`~.templates.prepare_template`).  Flux in erg/s/cm²/Å
        (surface flux).
    photometry : Photometry
        Observed AB magnitudes with errors.
    filter_curves : dict of FilterCurve
        Loaded filter curves keyed by filter name.

    Returns
    -------
    scale_factor : float
        Multiplicative factor to apply to the template flux.
    scale_error : float
        1-σ uncertainty on *scale_factor* (propagated from magnitude errors).
    band_residuals : dict
        ``{filter_name: synth_mag − obs_mag}`` *after* scaling.
    """
    valid_bands = photometry.valid_bands()
    offsets = []
    weights = []

    for band in valid_bands:
        if band not in filter_curves:
            continue
        fc = filter_curves[band]
        synth_mag = synthetic_photometry(template_spec, fc)
        if not np.isfinite(synth_mag):
            continue
        obs_mag, obs_err = photometry.get_band(band)
        dm = obs_mag - synth_mag   # offset: positive means template is brighter
        w = 1.0 / obs_err ** 2 if obs_err > 0 else 0.0
        offsets.append(dm)
        weights.append(w)

    if not offsets:
        logger.warning("No valid bands for photometric scaling; returning scale=1")
        return 1.0, 0.0, {}

    offsets = np.array(offsets)
    weights = np.array(weights)

    # Weighted mean offset
    dm_mean = np.average(offsets, weights=weights)
    scale_factor = 10.0 ** (-0.4 * dm_mean)

    # Propagate error: σ(s) = s × 0.4 × ln(10) × σ(Δm)
    if weights.sum() > 0:
        dm_err = 1.0 / np.sqrt(weights.sum())
    else:
        dm_err = np.std(offsets)
    scale_error = scale_factor * 0.4 * np.log(10.0) * dm_err

    # Compute per-band residuals after scaling
    band_residuals = {}
    for band in valid_bands:
        if band not in filter_curves:
            continue
        fc = filter_curves[band]
        scaled_spec = Spectrum1D(
            wavelength=template_spec.wavelength,
            flux=template_spec.flux * scale_factor,
            variance=template_spec.variance,
            mask=template_spec.mask,
        )
        synth_scaled = synthetic_photometry(scaled_spec, fc)
        obs_mag, _ = photometry.get_band(band)
        band_residuals[band] = synth_scaled - obs_mag

    n_bands = len(offsets)
    rms_resid = float(np.sqrt(np.mean(np.array(list(band_residuals.values())) ** 2))) if band_residuals else 0.0
    logger.debug(
        "Photometric scale: factor=%.4e (±%.2e), Δm_mean=%.4f, "
        "%d bands, RMS residual=%.4f mag",
        scale_factor, scale_error, dm_mean, n_bands, rms_resid,
    )

    return float(scale_factor), float(scale_error), band_residuals


# ---------------------------------------------------------------------------
# Per-star calibration vector
# ---------------------------------------------------------------------------

def compute_calibration_vector_for_star(
    observed: Spectrum1D,
    photometry: Photometry,
    library: TemplateLibrary,
    filter_curves: Dict[str, FilterCurve],
    instrument_fwhm_angstrom: float,
    fwhm_poly_coeffs: Optional[Sequence[float]] = None,
    mask_regions: Optional[List[Tuple[float, float]]] = None,
    metric: str = "chi2",
    star_name: str = "",
    fiber_id: int = -1,
) -> CalibrationVector:
    """Full per-star calibration pipeline.

    Orchestrates:

    1. Select best-matching template (:func:`~.matching.select_best_template`).
    2. Prepare template at instrument resolution.
    3. Scale template to match observed photometry.
    4. Compute ``Cal(λ) = scaled_template(λ) / observed(λ)``.
    5. Propagate variance and apply mask.

    Parameters
    ----------
    observed : Spectrum1D
        Extracted, wavelength-calibrated, sky-subtracted observed spectrum
        (in counts or counts/s).
    photometry : Photometry
        Broadband AB magnitudes for this standard star.
    library : TemplateLibrary
        Loaded BOSZ template library.
    filter_curves : dict of FilterCurve
        Loaded filter curves.
    instrument_fwhm_angstrom : float
        Instrument FWHM in Å.
    fwhm_poly_coeffs : sequence of float, optional
        Wavelength-dependent FWHM polynomial.
    mask_regions : list of (lo, hi), optional
        Wavelength regions to exclude.
    metric : str
        Template scoring metric (``"chi2"`` or ``"huber"``).
    star_name : str
        Identifier for this star (for logging / metadata).
    fiber_id : int
        Fiber index.

    Returns
    -------
    CalibrationVector
        The per-star calibration factor ``Cal(λ)`` such that
        ``flux_phys = counts × Cal(λ)``.
    """
    if mask_regions is None:
        mask_regions = load_mask_regions("telluric_default")

    # 1. Template selection (includes RV measurement and scoring)
    best_template, best_rv, fit_stats = select_best_template(
        observed, photometry, library,
        instrument_fwhm_angstrom=instrument_fwhm_angstrom,
        fwhm_poly_coeffs=fwhm_poly_coeffs,
        mask_regions=mask_regions,
        metric=metric,
    )

    # 2. Prepare template on observed grid (convolved + resampled)
    template_spec = prepare_template(
        best_template, observed.wavelength, instrument_fwhm_angstrom,
        fwhm_poly_coeffs=fwhm_poly_coeffs,
    )

    # 3. Scale template to observed photometry
    scale_factor, scale_error, band_residuals = scale_template_to_photometry(
        template_spec, photometry, filter_curves,
    )
    template_scaled_flux = template_spec.flux * scale_factor

    # 4. Compute Cal(λ) = scaled_template / observed
    obs_flux = observed.flux
    good = (
        observed.mask
        & (obs_flux > 0)
        & np.isfinite(obs_flux)
        & np.isfinite(template_scaled_flux)
        & (template_scaled_flux > 0)
    )

    # Apply telluric / bad-region mask
    for lo, hi in mask_regions:
        good &= ~((observed.wavelength >= lo) & (observed.wavelength <= hi))

    cal_factor = np.zeros_like(obs_flux)
    cal_variance = np.zeros_like(obs_flux)

    cal_factor[good] = template_scaled_flux[good] / obs_flux[good]

    # 5. Variance propagation
    # Cal = F_model / C_obs
    # σ(Cal)² = Cal² × [σ(F_model)²/F_model² + σ(C_obs)²/C_obs²]
    # σ(F_model) comes from the scale factor uncertainty
    # σ(C_obs) comes from the observed variance
    model_frac_var = (scale_error / scale_factor) ** 2 if scale_factor > 0 else 0.0
    obs_var = observed.variance
    obs_frac_var = np.zeros_like(obs_flux)
    obs_frac_var[good] = obs_var[good] / obs_flux[good] ** 2

    cal_variance[good] = cal_factor[good] ** 2 * (model_frac_var + obs_frac_var[good])

    meta = {
        "star_name": star_name,
        "fiber_id": fiber_id,
        "teff": best_template.teff,
        "logg": best_template.logg,
        "feh": best_template.feh,
        "alpha_m": best_template.alpha_m,
        "rv_kms": best_rv,
        "scale_factor": scale_factor,
        "scale_error": scale_error,
        "band_residuals": band_residuals,
        "chi2": fit_stats.get("best_score", np.nan),
        "ndof": fit_stats.get("ndof", 0),
        "n_candidates": fit_stats.get("n_candidates", 0),
    }

    logger.info(
        "Calibration vector for '%s' (fiber %d): %s, RV=%.1f km/s, "
        "scale=%.3e, %d/%d good pixels",
        star_name, fiber_id, best_template.label, best_rv,
        scale_factor, good.sum(), len(good),
    )

    return CalibrationVector(
        wavelength=observed.wavelength.copy(),
        cal_factor=cal_factor,
        cal_variance=cal_variance,
        mask=good,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

def combine_calibration_vectors(
    vectors: List[CalibrationVector],
    method: str = "weighted_mean",
    sigma_clip: float = 3.0,
    smooth: bool = False,
    smooth_window: int = 51,
) -> FluxCalibrationResult:
    """Combine per-star calibration vectors into a single curve.

    Gracefully handles N = 1 (single star: pass-through, no combination).

    Parameters
    ----------
    vectors : list of CalibrationVector
        Per-star calibration vectors (must share the same wavelength grid).
    method : ``"weighted_mean"`` | ``"median"``
        Combination method.
    sigma_clip : float
        Rejection threshold in sigma for iterative clipping.
    smooth : bool
        If True, apply Savitzky–Golay smoothing to the combined curve.
    smooth_window : int
        Window length for smoothing (must be odd).

    Returns
    -------
    FluxCalibrationResult
    """
    if not vectors:
        raise ValueError("No calibration vectors to combine")

    n_stars = len(vectors)
    wavelength = vectors[0].wavelength
    n_pix = len(wavelength)

    # Single-star pass-through
    if n_stars == 1:
        logger.info("Single standard star — using pass-through (no combination)")
        v = vectors[0]
        residuals = [np.zeros(n_pix)]
        summary = {
            "n_stars_used": 1,
            "n_stars_rejected": 0,
            "rms_scatter": 0.0,
            "wavelength_range": (float(wavelength[0]), float(wavelength[-1])),
            "per_star_metrics": [v.meta],
        }
        return FluxCalibrationResult(
            combined_vector=v,
            per_star_vectors=vectors,
            per_star_residuals=residuals,
            summary=summary,
        )

    # Stack cal_factor arrays: (n_stars, n_pix)
    stack = np.array([v.cal_factor for v in vectors])
    var_stack = np.array([v.cal_variance for v in vectors])
    mask_stack = np.array([v.mask for v in vectors])

    # Compute weights (inverse variance)
    ivar_stack = np.zeros_like(var_stack)
    pos = var_stack > 0
    ivar_stack[pos] = 1.0 / var_stack[pos]
    ivar_stack[~mask_stack] = 0.0

    if method == "weighted_mean":
        combined, combined_var, n_used = _weighted_mean_with_clipping(
            stack, ivar_stack, mask_stack, sigma_clip,
        )
    elif method == "median":
        combined, combined_var, n_used = _median_combination(
            stack, mask_stack,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'weighted_mean' or 'median'.")

    # Optional smoothing
    if smooth:
        combined = _savgol_smooth(combined, combined > 0, smooth_window)

    # Build combined mask: at least one star contributed
    combined_mask = combined > 0

    # Per-star residuals: (Cal_star − Cal_combined) / Cal_combined
    per_star_residuals = []
    for v in vectors:
        safe = np.where(combined > 0, combined, 1.0)
        resid = np.where(
            v.mask & combined_mask,
            (v.cal_factor - combined) / safe,
            0.0,
        )
        per_star_residuals.append(resid)

    # RMS scatter across stars at good pixels
    if n_stars > 1:
        resid_stack = np.array(per_star_residuals)
        good_any = combined_mask
        rms_per_pix = np.sqrt(np.mean(resid_stack[:, good_any] ** 2, axis=0))
        rms_scatter = float(np.median(rms_per_pix)) if len(rms_per_pix) > 0 else 0.0
    else:
        rms_scatter = 0.0

    summary = {
        "n_stars_used": n_stars,
        "n_stars_rejected": 0,
        "rms_scatter": rms_scatter,
        "wavelength_range": (float(wavelength[0]), float(wavelength[-1])),
        "per_star_metrics": [v.meta for v in vectors],
    }

    combined_vector = CalibrationVector(
        wavelength=wavelength.copy(),
        cal_factor=combined,
        cal_variance=combined_var,
        mask=combined_mask,
        meta={"method": method, "n_stars": n_stars, "sigma_clip": sigma_clip},
    )

    logger.info(
        "Combined %d calibration vectors (method=%s): "
        "%d/%d good pixels, RMS scatter=%.4f",
        n_stars, method, combined_mask.sum(), n_pix, rms_scatter,
    )

    return FluxCalibrationResult(
        combined_vector=combined_vector,
        per_star_vectors=vectors,
        per_star_residuals=per_star_residuals,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_flux_calibration(
    spectra: np.ndarray,
    variance: np.ndarray,
    calibration: FluxCalibrationResult,
    update_header: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Apply the combined calibration vector to all fibers.

    .. math::

        F_{\\rm cal}(\\lambda) = C_{\\rm obs}(\\lambda) \\times \\mathrm{Cal}(\\lambda)

        \\sigma^2_{\\rm cal} = \\sigma^2_{\\rm obs} \\times \\mathrm{Cal}^2
                             + C_{\\rm obs}^2 \\times \\sigma^2_{\\rm Cal}

    Parameters
    ----------
    spectra : ndarray, shape ``(NFIB, NPIX)`` or ``(NPIX, NFIB)``
        Observed spectra in counts (or counts/s).  The calibration vector
        length must match one of the two axes.
    variance : ndarray, same shape as *spectra*
        Variance array.
    calibration : FluxCalibrationResult
        Output of :func:`combine_calibration_vectors`.
    update_header : dict, optional
        If provided, FITS header keywords are added/updated in place.

    Returns
    -------
    cal_spectra : ndarray, same shape as *spectra*
        Flux-calibrated spectra in erg/s/cm²/Å.
    cal_variance : ndarray, same shape as *spectra*
        Propagated variance.
    header_updates : dict
        Suggested FITS header updates (BUNIT, HISTORY lines).
    """
    cv = calibration.combined_vector
    cal = cv.cal_factor
    cal_var = cv.cal_variance
    n_cal = len(cal)

    # Determine axis: match calibration length to one axis of spectra
    transposed = False
    if spectra.shape[0] == n_cal:
        # (NPIX, NFIB) — calibration along axis 0
        transposed = False
    elif spectra.shape[1] == n_cal:
        # (NFIB, NPIX) — calibration along axis 1
        transposed = False
    else:
        raise ValueError(
            f"Calibration vector length ({n_cal}) does not match "
            f"either axis of spectra shape {spectra.shape}"
        )

    # Broadcast calibration to spectra shape
    if spectra.ndim == 2:
        if spectra.shape[1] == n_cal:
            # (NFIB, NPIX): broadcast cal along axis 1
            cal_2d = cal[np.newaxis, :]
            calvar_2d = cal_var[np.newaxis, :]
        else:
            # (NPIX, NFIB): broadcast cal along axis 0
            cal_2d = cal[:, np.newaxis]
            calvar_2d = cal_var[:, np.newaxis]
    else:
        cal_2d = cal
        calvar_2d = cal_var

    # Apply calibration
    cal_spectra = spectra * cal_2d

    # Variance propagation: σ²_cal = σ²_obs × Cal² + C_obs² × σ²_Cal
    cal_variance_out = variance * cal_2d ** 2 + spectra ** 2 * calvar_2d

    # Header updates
    n_stars = calibration.summary.get("n_stars_used", 0)
    rms = calibration.summary.get("rms_scatter", 0.0)
    header_updates = {
        "BUNIT": ("erg/cm2/s/A", "Flux-calibrated units"),
        "FLUXCAL": (True, "Flux calibration applied"),
        "FCALNSTR": (n_stars, "Number of standard stars used"),
        "FCALRMS": (round(rms, 6), "RMS fractional scatter across standards"),
    }

    # Per-star HISTORY lines
    histories = []
    for v in calibration.per_star_vectors:
        m = v.meta
        histories.append(
            f"FLUXCAL: star={m.get('star_name', '?')} fiber={m.get('fiber_id', -1)} "
            f"Teff={m.get('teff', 0):.0f} logg={m.get('logg', 0):.1f} "
            f"[M/H]={m.get('feh', 0):+.2f} RV={m.get('rv_kms', 0):.1f}km/s "
            f"scale={m.get('scale_factor', 0):.3e}"
        )
    header_updates["HISTORY"] = histories

    if update_header is not None:
        for key, val in header_updates.items():
            if key == "HISTORY":
                for h in val:
                    update_header[key] = h
            else:
                update_header[key] = val

    logger.info(
        "Applied flux calibration to %s spectra (%d stars, RMS=%.4f)",
        spectra.shape, n_stars, rms,
    )

    return cal_spectra, cal_variance_out, header_updates


# ---------------------------------------------------------------------------
# Combination internals
# ---------------------------------------------------------------------------

def _weighted_mean_with_clipping(
    stack: np.ndarray,
    ivar_stack: np.ndarray,
    mask_stack: np.ndarray,
    sigma_clip: float,
    n_iter: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted mean with iterative sigma-clipping, pixel by pixel.

    Parameters
    ----------
    stack : (n_stars, n_pix)
    ivar_stack : (n_stars, n_pix)
    mask_stack : (n_stars, n_pix) bool
    sigma_clip : float
    n_iter : int

    Returns
    -------
    combined : (n_pix,)
    combined_var : (n_pix,)
    n_used : (n_pix,) int — number of stars contributing per pixel
    """
    n_stars, n_pix = stack.shape
    clip_mask = mask_stack.copy()

    for _ in range(n_iter):
        w = ivar_stack * clip_mask
        w_sum = w.sum(axis=0)
        safe_w = np.where(w_sum > 0, w_sum, 1.0)

        wmean = (stack * w).sum(axis=0) / safe_w

        # Residuals
        resid = stack - wmean[np.newaxis, :]
        # Weighted std per pixel
        var_pix = (w * resid ** 2).sum(axis=0) / safe_w
        std_pix = np.sqrt(var_pix)
        std_pix = np.where(std_pix > 0, std_pix, 1.0)

        # Clip
        new_clip = mask_stack & (np.abs(resid) < sigma_clip * std_pix[np.newaxis, :])
        if np.array_equal(new_clip, clip_mask):
            break
        clip_mask = new_clip

    # Final weighted mean
    w = ivar_stack * clip_mask
    w_sum = w.sum(axis=0)
    safe_w = np.where(w_sum > 0, w_sum, 1.0)
    combined = (stack * w).sum(axis=0) / safe_w
    combined_var = np.where(w_sum > 0, 1.0 / w_sum, 0.0)
    n_used = clip_mask.sum(axis=0)

    # Zero out pixels with no contributing stars
    no_data = w_sum == 0
    combined[no_data] = 0.0
    combined_var[no_data] = 0.0

    return combined, combined_var, n_used


def _median_combination(
    stack: np.ndarray,
    mask_stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-wise masked median combination."""
    n_stars, n_pix = stack.shape
    combined = np.zeros(n_pix)
    combined_var = np.zeros(n_pix)
    n_used = np.zeros(n_pix, dtype=int)

    for j in range(n_pix):
        vals = stack[mask_stack[:, j], j]
        n = len(vals)
        n_used[j] = n
        if n == 0:
            continue
        combined[j] = np.median(vals)
        if n > 1:
            # σ of median ≈ 1.253 × σ / √n
            combined_var[j] = (1.253 * np.std(vals) / np.sqrt(n)) ** 2
        else:
            combined_var[j] = 0.0

    return combined, combined_var, n_used


def _savgol_smooth(
    data: np.ndarray,
    mask: np.ndarray,
    window: int,
) -> np.ndarray:
    """Savitzky–Golay smoothing of the combined calibration curve."""
    from scipy.signal import savgol_filter

    if window % 2 == 0:
        window += 1
    window = min(window, mask.sum() - 1)
    if window < 5:
        return data

    filled = data.copy()
    if (~mask).any():
        good_idx = np.where(mask)[0]
        filled[~mask] = np.interp(np.where(~mask)[0], good_idx, data[mask])

    smoothed = savgol_filter(filled, window, polyorder=3)
    return smoothed
