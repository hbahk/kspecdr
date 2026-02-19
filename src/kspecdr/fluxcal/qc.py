"""
Quality-control plotting utilities for kspecdr flux calibration.

All functions return ``matplotlib.figure.Figure`` objects and are designed
for interactive use in Jupyter notebooks.  They are **not** called by the
pipeline itself.

Typical usage::

    from kspecdr.fluxcal.qc import plot_calibration_summary
    fig = plot_calibration_summary(result)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from .containers import (
    CalibrationVector,
    FluxCalibrationResult,
    Photometry,
    Spectrum1D,
    StellarTemplate,
)

logger = logging.getLogger(__name__)

__all__ = [
    "plot_calibration_summary",
    "plot_per_star_vectors",
    "plot_calibration_residuals",
    "plot_photometric_residuals",
    "plot_template_match",
    "plot_calibrated_spectrum",
    "summarize_calibration",
]


def _import_plt():
    """Lazy matplotlib import to avoid hard dependency at module level."""
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Summary overview
# ---------------------------------------------------------------------------

def plot_calibration_summary(
    result: FluxCalibrationResult,
    title: str = "Flux Calibration Summary",
    figsize: tuple = (14, 10),
):
    """Four-panel summary of the flux calibration result.

    Panels:
    1. Combined calibration vector Cal(λ)
    2. Per-star calibration vectors (overlaid)
    3. Per-star fractional residuals
    4. Histogram of fractional residuals

    Parameters
    ----------
    result : FluxCalibrationResult
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, y=0.98)

    cv = result.combined_vector
    wave = cv.wavelength

    # Panel 1: combined vector
    ax = axes[0, 0]
    good = cv.mask
    ax.plot(wave[good], cv.cal_factor[good], "k-", lw=0.8)
    _shade_masked(ax, wave, cv.mask)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Cal(λ)")
    ax.set_title("Combined calibration vector")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    # Panel 2: per-star vectors
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(result.n_stars, 1)))
    for i, v in enumerate(result.per_star_vectors):
        label = v.meta.get("star_name", f"star {i}")
        ax.plot(wave[v.mask], v.cal_factor[v.mask],
                lw=0.6, alpha=0.7, color=colors[i], label=label)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Cal(λ)")
    ax.set_title("Per-star vectors")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    if result.n_stars <= 8:
        ax.legend(fontsize=7, loc="best")

    # Panel 3: fractional residuals
    ax = axes[1, 0]
    for i, resid in enumerate(result.per_star_residuals):
        label = result.per_star_vectors[i].meta.get("star_name", f"star {i}")
        good_r = result.per_star_vectors[i].mask & cv.mask
        ax.plot(wave[good_r], resid[good_r] * 100,
                lw=0.5, alpha=0.7, color=colors[i], label=label)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Fractional residuals (star − combined)")
    rms = result.summary.get("rms_scatter", 0)
    ax.set_ylim(-max(10 * rms * 100, 5), max(10 * rms * 100, 5))

    # Panel 4: residual histogram
    ax = axes[1, 1]
    all_resid = []
    for i, resid in enumerate(result.per_star_residuals):
        good_r = result.per_star_vectors[i].mask & cv.mask
        all_resid.extend(resid[good_r] * 100)
    if all_resid:
        ax.hist(all_resid, bins=50, color="steelblue", edgecolor="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Residual (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution (RMS = {rms*100:.2f}%)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Individual per-star vector plot
# ---------------------------------------------------------------------------

def plot_per_star_vectors(
    result: FluxCalibrationResult,
    log_scale: bool = False,
    figsize: tuple = (12, 5),
):
    """Plot each per-star calibration vector on its own axis.

    Parameters
    ----------
    result : FluxCalibrationResult
    log_scale : bool
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    n = result.n_stars
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]),
                              sharey=True, squeeze=False)
    for i, v in enumerate(result.per_star_vectors):
        ax = axes[0, i]
        m = v.meta
        wave = v.wavelength
        good = v.mask
        ax.plot(wave[good], v.cal_factor[good], "k-", lw=0.6)
        err = v.cal_error
        ax.fill_between(wave[good],
                         v.cal_factor[good] - err[good],
                         v.cal_factor[good] + err[good],
                         alpha=0.2, color="steelblue")
        _shade_masked(ax, wave, v.mask)
        ax.set_xlabel("Wavelength (Å)")
        if i == 0:
            ax.set_ylabel("Cal(λ)")
        name = m.get("star_name", f"star {i}")
        teff = m.get("teff", 0)
        ax.set_title(f"{name}\nTeff={teff:.0f} K", fontsize=9)
        if log_scale:
            ax.set_yscale("log")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibration residuals
# ---------------------------------------------------------------------------

def plot_calibration_residuals(
    result: FluxCalibrationResult,
    figsize: tuple = (12, 4),
):
    """Fractional residuals of each per-star vector relative to the combined.

    Parameters
    ----------
    result : FluxCalibrationResult
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=figsize)
    cv = result.combined_vector
    wave = cv.wavelength
    colors = plt.cm.tab10(np.linspace(0, 1, max(result.n_stars, 1)))

    for i, resid in enumerate(result.per_star_residuals):
        v = result.per_star_vectors[i]
        good = v.mask & cv.mask
        label = v.meta.get("star_name", f"star {i}")
        ax.plot(wave[good], resid[good] * 100,
                lw=0.6, alpha=0.7, color=colors[i], label=label)

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Residual (%)")
    rms = result.summary.get("rms_scatter", 0)
    ax.set_title(f"Per-star residuals (RMS = {rms*100:.2f}%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Photometric residuals
# ---------------------------------------------------------------------------

def plot_photometric_residuals(
    cal_vectors: List[CalibrationVector],
    figsize: tuple = (8, 5),
):
    """Bar chart of per-band photometric residuals (synth − obs) for each star.

    Parameters
    ----------
    cal_vectors : list of CalibrationVector
        Each must have ``meta["band_residuals"]`` (from
        :func:`~.calibration.scale_template_to_photometry`).
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=figsize)

    n_stars = len(cal_vectors)
    all_bands = set()
    for v in cal_vectors:
        br = v.meta.get("band_residuals", {})
        all_bands.update(br.keys())
    bands = sorted(all_bands)
    if not bands:
        ax.text(0.5, 0.5, "No band residuals available",
                transform=ax.transAxes, ha="center")
        return fig

    x = np.arange(len(bands))
    width = 0.8 / max(n_stars, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_stars, 1)))

    for i, v in enumerate(cal_vectors):
        br = v.meta.get("band_residuals", {})
        vals = [br.get(b, 0.0) for b in bands]
        label = v.meta.get("star_name", f"star {i}")
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               color=colors[i], edgecolor="k", lw=0.3, label=label, alpha=0.8)

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Synth − Obs (mag)")
    ax.set_title("Photometric residuals after scaling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Template match inspection
# ---------------------------------------------------------------------------

def plot_template_match(
    observed: Spectrum1D,
    template: Spectrum1D,
    rv_kms: float = 0.0,
    title: str = "",
    figsize: tuple = (12, 6),
):
    """Compare observed and best-matching template spectra.

    Shows raw flux (top) and continuum-normalised (bottom, if continuum
    is available in template meta).

    Parameters
    ----------
    observed : Spectrum1D
        Observed standard-star spectrum.
    template : Spectrum1D
        Prepared (convolved + resampled) template spectrum.
    rv_kms : float
        Measured RV in km/s (annotated on the plot).
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    has_cont = "continuum" in template.meta and template.meta["continuum"] is not None

    n_panels = 2 if has_cont else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    wave = observed.wavelength
    good = observed.mask

    # Top: raw flux
    ax = axes[0]
    ax.plot(wave[good], observed.flux[good], "k-", lw=0.5, label="Observed", alpha=0.8)
    ax.plot(wave, template.flux, "r-", lw=0.5, label="Template", alpha=0.7)
    ax.set_ylabel("Flux")
    teff = template.meta.get("teff", "?")
    logg = template.meta.get("logg", "?")
    feh = template.meta.get("feh", "?")
    ax.legend(fontsize=8, title=f"Teff={teff} logg={logg} [M/H]={feh}  RV={rv_kms:.1f} km/s")
    if title:
        ax.set_title(title)

    # Bottom: normalised
    if has_cont:
        cont = template.meta["continuum"]
        safe_cont = np.where(cont > 0, cont, 1.0)
        ax = axes[1]
        # Normalise observed by fitting or by template continuum
        obs_norm = observed.flux / safe_cont
        tmpl_norm = template.flux / safe_cont
        ax.plot(wave[good], obs_norm[good], "k-", lw=0.5, alpha=0.8, label="Obs / cont")
        ax.plot(wave, tmpl_norm, "r-", lw=0.5, alpha=0.7, label="Tmpl / cont")
        ax.axhline(1.0, color="gray", lw=0.3, ls="--")
        ax.set_ylabel("Normalised flux")
        ax.set_ylim(0.3, 1.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Wavelength (Å)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibrated spectrum
# ---------------------------------------------------------------------------

def plot_calibrated_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    variance: np.ndarray = None,
    fiber_id: int = -1,
    title: str = "",
    figsize: tuple = (12, 4),
):
    """Quick plot of a flux-calibrated spectrum with optional error band.

    Parameters
    ----------
    wavelength : ndarray
        In Angstrom.
    flux : ndarray
        In erg/s/cm²/Å.
    variance : ndarray, optional
        If provided, ±1σ band is shown.
    fiber_id : int
        Annotated on the plot.
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=figsize)

    good = np.isfinite(flux) & (flux > 0)
    ax.plot(wavelength[good], flux[good], "k-", lw=0.5)

    if variance is not None:
        err = np.sqrt(np.where(variance > 0, variance, 0.0))
        ax.fill_between(
            wavelength[good],
            flux[good] - err[good],
            flux[good] + err[good],
            alpha=0.2, color="steelblue",
        )

    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel(r"$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    t = title or f"Fiber {fiber_id}" if fiber_id >= 0 else "Calibrated spectrum"
    ax.set_title(t)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Text summary table
# ---------------------------------------------------------------------------

def summarize_calibration(result: FluxCalibrationResult) -> str:
    """Return a formatted text summary of the calibration result.

    Suitable for printing in a notebook cell or logging.

    Parameters
    ----------
    result : FluxCalibrationResult

    Returns
    -------
    str
    """
    lines = []
    s = result.summary
    lines.append("=" * 60)
    lines.append("  Flux Calibration Summary")
    lines.append("=" * 60)
    lines.append(f"  Stars used       : {s.get('n_stars_used', '?')}")
    lines.append(f"  Stars rejected   : {s.get('n_stars_rejected', 0)}")
    rms = s.get("rms_scatter", 0)
    lines.append(f"  RMS scatter      : {rms*100:.2f}%")
    wr = s.get("wavelength_range", (0, 0))
    lines.append(f"  Wavelength range : {wr[0]:.0f} – {wr[1]:.0f} Å")
    lines.append(f"  Good pixels      : {result.combined_vector.n_good}")
    lines.append("-" * 60)

    for i, v in enumerate(result.per_star_vectors):
        m = v.meta
        lines.append(
            f"  Star {i}: {m.get('star_name', '?'):>12s}  "
            f"fiber={m.get('fiber_id', -1):>3d}  "
            f"Teff={m.get('teff', 0):5.0f}  "
            f"logg={m.get('logg', 0):.1f}  "
            f"[M/H]={m.get('feh', 0):+.2f}  "
            f"RV={m.get('rv_kms', 0):+6.1f} km/s  "
            f"scale={m.get('scale_factor', 0):.3e}"
        )
        br = m.get("band_residuals", {})
        if br:
            resid_str = "  ".join(f"{b}={r:+.3f}" for b, r in sorted(br.items()))
            lines.append(f"         resid(mag): {resid_str}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shade_masked(ax, wavelength, mask):
    """Add light red shading for masked (False) regions."""
    bad = ~mask
    if not bad.any():
        return
    # Find contiguous bad regions
    diff = np.diff(bad.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if bad[0]:
        starts = np.concatenate([[0], starts])
    if bad[-1]:
        ends = np.concatenate([ends, [len(wavelength) - 1]])
    for s, e in zip(starts, ends):
        ax.axvspan(wavelength[s], wavelength[min(e, len(wavelength) - 1)],
                   alpha=0.08, color="red", zorder=0)
