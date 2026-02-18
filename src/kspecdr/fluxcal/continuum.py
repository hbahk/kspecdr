"""
Continuum normalization for kspecdr flux calibration.

Provides iterative pseudo-continuum fitting with lower-sigma clipping to
avoid absorption lines biasing the fit downward.  Three methods:

- ``"bspline"``  — LSQ B-spline (default; robust for broad coverage)
- ``"polynomial"`` — Legendre polynomial (simpler, good for narrow ranges)
- ``"running_median"`` — Smoothed running median (fast, non-parametric)

All methods operate on :class:`~.containers.Spectrum1D` objects.

For BOSZ templates, the theoretical continuum column
(``StellarTemplate.continuum``) can be used directly, bypassing the fit
entirely — see :func:`normalize_with_model_continuum`.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.ndimage import median_filter

from .containers import Spectrum1D

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_continuum",
    "normalize_with_model_continuum",
    "fit_continuum",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_continuum(
    spectrum: Spectrum1D,
    method: str = "bspline",
    order: int = 3,
    n_knots: int = 20,
    sigma_lo: float = 2.0,
    sigma_hi: float = 4.0,
    n_iter: int = 5,
    mask_regions: Optional[List[Tuple[float, float]]] = None,
    median_window: int = 151,
) -> Tuple[Spectrum1D, np.ndarray]:
    """Fit and divide out the pseudo-continuum.

    Parameters
    ----------
    spectrum : Spectrum1D
        Input spectrum (flux in any units).
    method : ``"bspline"`` | ``"polynomial"`` | ``"running_median"``
        Fitting method.
    order : int
        B-spline degree (for ``"bspline"``) or polynomial degree
        (for ``"polynomial"``).  Ignored for ``"running_median"``.
    n_knots : int
        Number of interior knots for B-spline.  Ignored for other methods.
    sigma_lo : float
        Lower rejection threshold in sigma units.  Pixels more than
        *sigma_lo* × σ **below** the current continuum fit are clipped
        (targeting absorption lines).
    sigma_hi : float
        Upper rejection threshold.  Pixels more than *sigma_hi* × σ
        **above** the continuum are clipped (targeting emission artefacts
        or cosmic-ray residuals).
    n_iter : int
        Number of iterative sigma-clipping passes.
    mask_regions : list of (lam_lo, lam_hi), optional
        Wavelength regions to exclude from the fit entirely (e.g. tellurics).
        Pixels in these regions still receive interpolated continuum values.
    median_window : int
        Window size in pixels for ``"running_median"``.  Must be odd.

    Returns
    -------
    normalized : Spectrum1D
        Continuum-normalised spectrum (``flux / continuum``).  Variance is
        propagated.  ``meta["continuum_fit"]`` stores the fitted continuum.
    continuum : ndarray, shape (N,)
        The fitted continuum array.
    """
    continuum = fit_continuum(
        spectrum,
        method=method,
        order=order,
        n_knots=n_knots,
        sigma_lo=sigma_lo,
        sigma_hi=sigma_hi,
        n_iter=n_iter,
        mask_regions=mask_regions,
        median_window=median_window,
    )

    return _divide_by_continuum(spectrum, continuum), continuum


def normalize_with_model_continuum(
    spectrum: Spectrum1D,
    model_continuum: np.ndarray,
) -> Tuple[Spectrum1D, np.ndarray]:
    """Normalise using an externally supplied continuum (e.g. BOSZ ``C`` column).

    Parameters
    ----------
    spectrum : Spectrum1D
    model_continuum : ndarray, shape (N,)

    Returns
    -------
    normalized : Spectrum1D
    continuum : ndarray
    """
    return _divide_by_continuum(spectrum, model_continuum), model_continuum


def fit_continuum(
    spectrum: Spectrum1D,
    method: str = "bspline",
    order: int = 3,
    n_knots: int = 20,
    sigma_lo: float = 2.0,
    sigma_hi: float = 4.0,
    n_iter: int = 5,
    mask_regions: Optional[List[Tuple[float, float]]] = None,
    median_window: int = 151,
) -> np.ndarray:
    """Fit the pseudo-continuum and return it as an array.

    See :func:`normalize_continuum` for parameter descriptions.

    Returns
    -------
    ndarray, shape (N,)
        Fitted continuum evaluated at every pixel.
    """
    wave = spectrum.wavelength
    flux = spectrum.flux
    n = len(wave)

    # Build fitting mask: good pixels AND outside excluded regions
    fit_mask = spectrum.mask.copy() & np.isfinite(flux)
    if mask_regions is not None:
        for lo, hi in mask_regions:
            fit_mask &= ~((wave >= lo) & (wave <= hi))

    # Reject non-positive flux from fit (but still evaluate continuum there)
    fit_mask &= flux > 0

    if fit_mask.sum() < max(n_knots + order + 1, order + 1, 10):
        logger.warning(
            "Too few valid pixels (%d) for continuum fitting; "
            "returning flux as continuum", fit_mask.sum(),
        )
        return flux.copy()

    # Build weights from variance
    weights = np.ones(n, dtype=float)
    has_var = spectrum.variance > 0
    weights[has_var] = 1.0 / np.sqrt(spectrum.variance[has_var])
    weights[~fit_mask] = 0.0

    if method == "bspline":
        continuum = _fit_bspline(wave, flux, weights, fit_mask, order, n_knots,
                                 sigma_lo, sigma_hi, n_iter)
    elif method == "polynomial":
        continuum = _fit_polynomial(wave, flux, weights, fit_mask, order,
                                    sigma_lo, sigma_hi, n_iter)
    elif method == "running_median":
        continuum = _fit_running_median(flux, fit_mask, median_window)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'bspline', 'polynomial', or 'running_median'."
        )

    return continuum


# ---------------------------------------------------------------------------
# Fitting backends
# ---------------------------------------------------------------------------

def _fit_bspline(
    wave: np.ndarray,
    flux: np.ndarray,
    weights: np.ndarray,
    fit_mask: np.ndarray,
    order: int,
    n_knots: int,
    sigma_lo: float,
    sigma_hi: float,
    n_iter: int,
) -> np.ndarray:
    """LSQ B-spline with iterative asymmetric sigma clipping."""
    clip_mask = fit_mask.copy()

    for iteration in range(n_iter):
        idx = np.where(clip_mask)[0]
        if len(idx) < n_knots + order + 1:
            logger.warning("B-spline: too few points at iteration %d", iteration)
            break

        x_fit = wave[idx]
        y_fit = flux[idx]
        w_fit = weights[idx]

        # Uniform interior knots (excluding boundary knots handled by scipy)
        t = np.linspace(x_fit[0], x_fit[-1], n_knots + 2)[1:-1]

        try:
            spl = LSQUnivariateSpline(x_fit, y_fit, t, w=w_fit, k=order)
        except Exception as exc:
            logger.warning("B-spline fit failed at iteration %d: %s", iteration, exc)
            break

        cont_fit = spl(wave[fit_mask])
        residual = flux[fit_mask] - cont_fit
        sigma = _robust_sigma(residual)
        if sigma <= 0:
            break

        # Asymmetric clipping: tighter below (absorption), looser above
        new_clip = fit_mask.copy()
        res_full = flux - spl(wave)
        new_clip &= res_full > -sigma_lo * sigma
        new_clip &= res_full < sigma_hi * sigma

        if np.array_equal(new_clip, clip_mask):
            break
        clip_mask = new_clip

    # Evaluate final spline everywhere
    try:
        return spl(wave)
    except UnboundLocalError:
        # spl was never successfully created
        return flux.copy()


def _fit_polynomial(
    wave: np.ndarray,
    flux: np.ndarray,
    weights: np.ndarray,
    fit_mask: np.ndarray,
    order: int,
    sigma_lo: float,
    sigma_hi: float,
    n_iter: int,
) -> np.ndarray:
    """Weighted polynomial with iterative asymmetric sigma clipping.

    Wavelength is normalised to [−1, 1] before fitting for numerical stability.
    """
    # Normalise to [-1, 1]
    w_lo, w_hi = wave[0], wave[-1]
    w_norm = 2.0 * (wave - w_lo) / (w_hi - w_lo) - 1.0

    clip_mask = fit_mask.copy()

    for iteration in range(n_iter):
        idx = np.where(clip_mask)[0]
        if len(idx) < order + 1:
            logger.warning("Polynomial: too few points at iteration %d", iteration)
            break

        coeffs = np.polynomial.legendre.legfit(
            w_norm[idx], flux[idx], order, w=weights[idx],
        )
        cont_all = np.polynomial.legendre.legval(w_norm, coeffs)

        residual = flux[fit_mask] - cont_all[fit_mask]
        sigma = _robust_sigma(residual)
        if sigma <= 0:
            break

        res_full = flux - cont_all
        new_clip = fit_mask.copy()
        new_clip &= res_full > -sigma_lo * sigma
        new_clip &= res_full < sigma_hi * sigma

        if np.array_equal(new_clip, clip_mask):
            break
        clip_mask = new_clip

    return cont_all


def _fit_running_median(
    flux: np.ndarray,
    fit_mask: np.ndarray,
    window: int,
) -> np.ndarray:
    """Running median smoothing (non-parametric, no iteration)."""
    if window % 2 == 0:
        window += 1

    # Replace masked pixels with local median before filtering
    filled = flux.copy()
    bad = ~fit_mask
    if bad.any():
        filled[bad] = np.interp(
            np.where(bad)[0],
            np.where(fit_mask)[0],
            flux[fit_mask],
        )

    return median_filter(filled, size=window, mode="nearest").astype(float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _divide_by_continuum(
    spectrum: Spectrum1D,
    continuum: np.ndarray,
) -> Spectrum1D:
    """Divide spectrum by continuum, propagating variance."""
    safe = np.where(continuum > 0, continuum, 1.0)
    norm_flux = spectrum.flux / safe
    # σ(f/c)² = σ_f² / c²
    norm_var = spectrum.variance / safe ** 2

    # Mask pixels where continuum is non-positive
    new_mask = spectrum.mask.copy() & (continuum > 0)

    meta = dict(spectrum.meta)
    meta["continuum_fit"] = continuum

    return Spectrum1D(
        wavelength=spectrum.wavelength.copy(),
        flux=norm_flux,
        variance=norm_var,
        mask=new_mask,
        meta=meta,
    )


def _robust_sigma(residual: np.ndarray) -> float:
    """MAD-based robust sigma estimate."""
    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    return float(1.4826 * mad)  # σ ≈ 1.4826 × MAD for Gaussian
