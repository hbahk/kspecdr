"""
Template selection and radial-velocity measurement for kspecdr flux calibration.

Workflow for each standard star:

1. Narrow the template search range using a photometric Teff estimate.
2. For each candidate template:
   a. Prepare (convolve + resample) to the observed grid.
   b. Cross-correlate to measure/correct radial velocity.
   c. Continuum-normalise both observed and template.
   d. Score the fit on line features (χ² or Huber metric).
3. Return the best-matching template, RV, and fit statistics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .containers import Photometry, Spectrum1D, StellarTemplate
from .continuum import normalize_continuum, normalize_with_model_continuum
from .photometry import estimate_teff_from_color
from .templates import TemplateLibrary, prepare_template

logger = logging.getLogger(__name__)

__all__ = [
    "select_best_template",
    "cross_correlate_rv",
    "score_template_fit",
]

# Speed of light in km/s
_C_KMS = 2.99792458e5


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def select_best_template(
    observed: Spectrum1D,
    photometry: Photometry,
    library: TemplateLibrary,
    instrument_fwhm_angstrom: float,
    fwhm_poly_coeffs: Optional[Sequence[float]] = None,
    rv_guess: float = 0.0,
    mask_regions: Optional[List[Tuple[float, float]]] = None,
    teff_range: Optional[Tuple[float, float]] = None,
    logg_range: Optional[Tuple[float, float]] = None,
    feh_range: Optional[Tuple[float, float]] = None,
    metric: str = "chi2",
    max_rv_shift: float = 500.0,
    continuum_method: str = "bspline",
    continuum_n_knots: int = 20,
) -> Tuple[StellarTemplate, float, Dict]:
    """Find the best-matching template for an observed standard star.

    Parameters
    ----------
    observed : Spectrum1D
        Extracted, wavelength-calibrated observed spectrum.
    photometry : Photometry
        Broadband magnitudes for the star (AB system).
    library : TemplateLibrary
        Loaded BOSZ template library.
    instrument_fwhm_angstrom : float
        Instrument spectral FWHM in Å.
    fwhm_poly_coeffs : sequence of float, optional
        Wavelength-dependent FWHM polynomial.
    rv_guess : float
        Initial RV guess in km/s.
    mask_regions : list of (lo, hi), optional
        Wavelength regions to exclude from scoring.
    teff_range : (float, float), optional
        Explicit Teff search range.  If None, estimated from *photometry*.
    logg_range, feh_range : (float, float), optional
        Explicit search ranges.
    metric : ``"chi2"`` | ``"huber"``
        Scoring metric.
    max_rv_shift : float
        Maximum allowed RV shift in km/s for cross-correlation.
    continuum_method : str
        Continuum fitting method for observed spectrum.
    continuum_n_knots : int
        Number of knots for B-spline continuum fit.

    Returns
    -------
    best_template : StellarTemplate
        The best-matching template (un-convolved, native BOSZ grid).
    best_rv : float
        Measured radial velocity in km/s.
    fit_stats : dict
        Keys: ``"best_score"``, ``"ndof"``, ``"runner_up_score"``,
        ``"n_candidates"``, ``"teff_range"``, ``"best_params"``,
        ``"all_scores"`` (list of dicts).
    """
    # 1. Determine Teff search range
    if teff_range is None:
        _, teff_range = estimate_teff_from_color(photometry)
    if logg_range is None:
        logg_range = (3.0, 5.5)
    if feh_range is None:
        feh_range = (-1.25, 0.75)

    # 2. Query candidate templates
    candidates = library.query(
        teff_range=teff_range,
        logg_range=logg_range,
        feh_range=feh_range,
    )
    if not candidates:
        logger.warning(
            "No templates in range Teff=%s logg=%s [M/H]=%s; "
            "expanding to full grid",
            teff_range, logg_range, feh_range,
        )
        candidates = library.query()

    logger.info(
        "Template matching: %d candidates in Teff=[%.0f, %.0f]",
        len(candidates), teff_range[0], teff_range[1],
    )

    # 3. Normalise observed spectrum once
    obs_norm, obs_cont = normalize_continuum(
        observed,
        method=continuum_method,
        n_knots=continuum_n_knots,
        mask_regions=mask_regions,
    )

    # 4. Score each candidate
    all_scores: List[Dict] = []
    for entry in candidates:
        tmpl = library.load_template(entry)
        tmpl_spec = prepare_template(
            tmpl, observed.wavelength, instrument_fwhm_angstrom,
            fwhm_poly_coeffs=fwhm_poly_coeffs,
        )

        # Measure RV via cross-correlation
        rv = cross_correlate_rv(
            obs_norm, tmpl_spec, max_shift_kms=max_rv_shift, rv_guess=rv_guess,
        )

        # Apply RV shift to template
        tmpl_shifted = _apply_rv_shift(tmpl_spec, rv)

        # Normalise template using its BOSZ continuum column
        if "continuum" in tmpl_shifted.meta and tmpl_shifted.meta["continuum"] is not None:
            tmpl_norm, _ = normalize_with_model_continuum(
                tmpl_shifted, tmpl_shifted.meta["continuum"],
            )
        else:
            tmpl_norm, _ = normalize_continuum(
                tmpl_shifted,
                method=continuum_method,
                n_knots=continuum_n_knots,
                mask_regions=mask_regions,
            )

        # Score
        score, ndof = score_template_fit(
            obs_norm, tmpl_norm,
            metric=metric,
            mask_regions=mask_regions,
        )

        all_scores.append({
            "teff": tmpl.teff,
            "logg": tmpl.logg,
            "feh": tmpl.feh,
            "alpha_m": tmpl.alpha_m,
            "rv_kms": rv,
            "score": score,
            "ndof": ndof,
            "source": tmpl.source,
        })

    # 5. Select best
    all_scores.sort(key=lambda d: d["score"])
    best = all_scores[0]
    runner_up_score = all_scores[1]["score"] if len(all_scores) > 1 else np.nan

    best_template = library.get_template(
        best["teff"], best["logg"], best["feh"], best.get("alpha_m"),
    )

    fit_stats = {
        "best_score": best["score"],
        "ndof": best["ndof"],
        "runner_up_score": runner_up_score,
        "n_candidates": len(candidates),
        "teff_range": teff_range,
        "best_params": best,
        "all_scores": all_scores,
    }

    logger.info(
        "Best template: %s (score=%.2f, RV=%.1f km/s, %d candidates)",
        best_template.label, best["score"], best["rv_kms"], len(candidates),
    )

    return best_template, best["rv_kms"], fit_stats


# ---------------------------------------------------------------------------
# Cross-correlation RV measurement
# ---------------------------------------------------------------------------

def cross_correlate_rv(
    observed: Spectrum1D,
    template: Spectrum1D,
    max_shift_kms: float = 500.0,
    rv_guess: float = 0.0,
) -> float:
    """Measure radial velocity by cross-correlating observed and template.

    Both spectra are resampled onto a uniform log-λ grid (= uniform velocity
    bins) before computing the FFT-based cross-correlation.  The peak is
    refined by quadratic interpolation for sub-pixel precision.

    Parameters
    ----------
    observed : Spectrum1D
        Continuum-normalised observed spectrum.
    template : Spectrum1D
        Continuum-normalised (and convolved) template on the same grid.
    max_shift_kms : float
        Maximum allowed shift in km/s.
    rv_guess : float
        Initial guess (shifts the search window centre).

    Returns
    -------
    float
        Best-fit radial velocity in km/s (positive = receding).
    """
    wave = observed.wavelength
    good = (
        observed.mask & template.mask
        & np.isfinite(observed.flux) & np.isfinite(template.flux)
        & (wave > 0)
    )

    if good.sum() < 50:
        logger.warning("Too few good pixels (%d) for cross-correlation", good.sum())
        return rv_guess

    w_good = wave[good]
    obs_good = observed.flux[good]
    tmpl_good = template.flux[good]

    # Resample onto a uniform log-λ grid (uniform velocity bins)
    ln_lo = np.log(w_good[0])
    ln_hi = np.log(w_good[-1])
    n_logpix = len(w_good)
    ln_wave_uniform = np.linspace(ln_lo, ln_hi, n_logpix)
    d_ln = ln_wave_uniform[1] - ln_wave_uniform[0]
    v_per_pix = _C_KMS * d_ln  # km/s per log-pixel

    wave_uniform = np.exp(ln_wave_uniform)
    obs_f = np.interp(wave_uniform, w_good, obs_good)
    tmpl_f = np.interp(wave_uniform, w_good, tmpl_good)

    # Zero-centre (remove mean)
    obs_f -= np.mean(obs_f)
    tmpl_f -= np.mean(tmpl_f)

    # FFT cross-correlation — zero-pad to avoid circular wrap-around
    nfft = int(2 ** np.ceil(np.log2(2 * n_logpix)))
    ccf = np.fft.irfft(
        np.fft.rfft(obs_f, n=nfft) * np.conj(np.fft.rfft(tmpl_f, n=nfft)),
        n=nfft,
    )

    # Build velocity axis: each lag unit = one log-pixel = v_per_pix km/s
    shifts = np.arange(nfft)
    shifts[shifts > nfft // 2] -= nfft
    velocities = shifts * v_per_pix

    # Restrict to search window around rv_guess
    window = np.abs(velocities - rv_guess) <= max_shift_kms
    if window.sum() == 0:
        return rv_guess

    ccf_win = ccf[window]
    vel_win = velocities[window]

    # Find peak
    peak_idx = np.argmax(ccf_win)

    # Quadratic interpolation for sub-pixel refinement
    if 0 < peak_idx < len(ccf_win) - 1:
        y0, y1, y2 = ccf_win[peak_idx - 1], ccf_win[peak_idx], ccf_win[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 0:
            delta = 0.5 * (y0 - y2) / denom
        else:
            delta = 0.0
        rv = vel_win[peak_idx] + delta * v_per_pix
    else:
        rv = vel_win[peak_idx]

    return float(rv)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_template_fit(
    observed: Spectrum1D,
    template: Spectrum1D,
    metric: str = "chi2",
    mask_regions: Optional[List[Tuple[float, float]]] = None,
    huber_delta: float = 1.5,
) -> Tuple[float, int]:
    """Score the agreement between normalised observed and template spectra.

    Parameters
    ----------
    observed, template : Spectrum1D
        Continuum-normalised spectra on the same wavelength grid.
    metric : ``"chi2"`` | ``"huber"``
    mask_regions : list of (lo, hi), optional
        Additional regions to exclude from the score.
    huber_delta : float
        Transition point for the Huber loss function.

    Returns
    -------
    score : float
        Reduced chi² or mean Huber loss.
    ndof : int
        Number of pixels used.
    """
    good = observed.mask & template.mask
    good &= np.isfinite(observed.flux) & np.isfinite(template.flux)

    if mask_regions is not None:
        for lo, hi in mask_regions:
            good &= ~(
                (observed.wavelength >= lo) & (observed.wavelength <= hi)
            )

    ndof = int(good.sum())
    if ndof < 10:
        return np.inf, 0

    residual = observed.flux[good] - template.flux[good]

    # Weights from observed variance (template variance = 0)
    ivar = observed.ivar[good]
    has_weight = ivar > 0
    if has_weight.sum() < 10:
        # Fall back to unit weights
        ivar = np.ones_like(residual)
        has_weight = np.ones(len(residual), dtype=bool)

    if metric == "chi2":
        chi2 = np.sum(residual[has_weight] ** 2 * ivar[has_weight])
        score = float(chi2 / max(has_weight.sum() - 1, 1))

    elif metric == "huber":
        # Huber loss: quadratic for |r| < delta, linear for |r| >= delta
        abs_r = np.abs(residual[has_weight]) * np.sqrt(ivar[has_weight])
        loss = np.where(
            abs_r <= huber_delta,
            0.5 * abs_r ** 2,
            huber_delta * abs_r - 0.5 * huber_delta ** 2,
        )
        score = float(np.mean(loss))
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'chi2' or 'huber'.")

    return score, ndof


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_rv_shift(spectrum: Spectrum1D, rv_kms: float) -> Spectrum1D:
    """Doppler-shift a spectrum by *rv_kms* km/s (positive = redshift).

    Resamples flux and continuum onto the *original* wavelength grid so the
    output shares the same sampling as the input.
    """
    if abs(rv_kms) < 0.01:
        return spectrum

    factor = 1.0 + rv_kms / _C_KMS
    shifted_wave = spectrum.wavelength * factor

    flux_new = np.interp(spectrum.wavelength, shifted_wave, spectrum.flux,
                         left=0.0, right=0.0)

    meta = dict(spectrum.meta)
    meta["rv_kms"] = rv_kms

    if "continuum" in meta and meta["continuum"] is not None:
        cont_new = np.interp(spectrum.wavelength, shifted_wave, meta["continuum"],
                             left=0.0, right=0.0)
        meta["continuum"] = cont_new

    return Spectrum1D(
        wavelength=spectrum.wavelength.copy(),
        flux=flux_new,
        variance=spectrum.variance.copy(),
        mask=spectrum.mask.copy(),
        meta=meta,
    )
