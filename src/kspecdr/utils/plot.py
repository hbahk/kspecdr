"""
Spectral plotting utilities for KSPEC reduced data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from astropy.io import fits


# ---------------------------------------------------------------------------
# Spectral-feature catalogues
# ---------------------------------------------------------------------------

EMISSION_LINES = [
    (3729.875, "O II"),
    (3889.0,   "He I"),
    (3970.1,   "Hε"),
    (4072.3,   "S II"),
    (4102.89,  "Hδ"),
    (4341.68,  "Hγ"),
    (4364.436, "O III"),
    (4862.68,  "Hβ"),
    (4932.603, "O III"),
    (4960.295, "O III"),
    (5008.240, "O III"),
    (6302.046, "O I"),
    (6365.536, "O I"),
    (6529.03,  "N I"),
    (6549.86,  "N II"),
    (6564.61,  "Hα"),
    (6585.27,  "N II"),
    (6718.29,  "S II"),
    (6732.67,  "S II"),
]

ABSORPTION_LINES = [
    (3934.777, "K"),
    (3969.588, "H"),
    (4305.61,  "G"),
    (5176.7,   "Mg"),
    (5895.6,   "Na"),
    (8500.36,  "CaII"),
    (8544.44,  "CaII"),
    (8664.52,  "CaII"),
]

SKY_LINES = [
    (5578.5, "Sky"),
    (5894.6, "Sky"),
    (6301.7, "Sky"),
    (7246.0, "Sky"),
]

TELLURIC_BANDS = [
    (6860,  6960,  "O$_2$ B"),
    (7160,  7340,  "H$_2$O"),
    (7580,  7700,  "O$_2$ A"),
    (8120,  8400,  "H$_2$O"),
    (8920,  9800,  "H$_2$O"),
]


# ---------------------------------------------------------------------------
# Core single-panel plotting function
# ---------------------------------------------------------------------------

def plot_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    target_z: float = 0.0,
    ref_wave: Optional[np.ndarray] = None,
    ref_flux: Optional[np.ndarray] = None,
    ref_label: str = "ref",
    ref_color: str = "tab:red",
    ylim: Optional[tuple] = None,
    title: Optional[str] = None,
    xlabel: str = r"Wavelength ($\AA$)",
    ylabel: str = r"Flux (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)",
    show_emission: bool = True,
    show_absorption: bool = True,
    show_sky: bool = True,
    show_telluric: bool = True,
    figsize: tuple = (10, 4),
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """
    Single-panel spectral plot with emission, absorption, sky, and telluric
    feature annotations.

    Parameters
    ----------
    wave : array-like
        Observed-frame wavelength array [Å].
    flux : array-like
        Flux array (same length as *wave*).
    ax : Axes, optional
        Existing axes to plot into.  A new figure is created if None.
    target_z : float
        Object redshift applied to rest-frame line wavelengths.
    ref_wave, ref_flux : array-like, optional
        Optional reference spectrum (e.g. SDSS) to overplot in *ref_color*.
    ref_label : str
        Legend label for the reference spectrum.
    ref_color : str
        Colour for the reference spectrum.
    ylim : (ymin, ymax), optional
        Y-axis limits.  If None, set from the 99th percentile of |flux|.
    title : str, optional
        Axes title.
    xlabel, ylabel : str
        Axis labels.
    show_emission, show_absorption, show_sky, show_telluric : bool
        Toggle individual annotation layers.
    figsize : (width, height)
        Figure size when a new figure is created.

    Returns
    -------
    fig, ax
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    # Main spectrum
    ax.plot(wave, flux, lw=0.5, color="k", zorder=3)

    # Reference spectrum
    if ref_wave is not None and ref_flux is not None:
        ax.plot(ref_wave, ref_flux, lw=0.5, color=ref_color,
                label=ref_label, zorder=2)

    ax.set_xlim(wave[0], wave[-1])

    # Y limits
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        finite = flux[np.isfinite(flux)]
        if finite.size:
            lo = np.nanpercentile(finite, 1)
            hi = np.nanpercentile(finite, 99)
            margin = 0.1 * (hi - lo) if hi > lo else 50
            ax.set_ylim(lo - margin, hi + margin)

    ymin, ymax = ax.get_ylim()

    # --- Emission lines ---
    if show_emission:
        for rest_wave, name in EMISSION_LINES:
            w_obs = rest_wave * (1 + target_z)
            if wave[0] < w_obs < wave[-1]:
                ax.axvline(w_obs, color="tab:blue", lw=0.5, alpha=0.7, ls="--", zorder=1)
                ax.text(w_obs, ymax * 0.97, name, color="tab:blue",
                        rotation=90, ha="center", va="top", fontsize=7)

    # --- Absorption lines ---
    if show_absorption:
        for rest_wave, name in ABSORPTION_LINES:
            w_obs = rest_wave * (1 + target_z)
            if wave[0] < w_obs < wave[-1]:
                ax.axvline(w_obs, color="tab:red", lw=0.5, alpha=0.7, ls="--", zorder=1)
                ax.text(w_obs, ymin * 0.97, name, color="tab:red",
                        rotation=90, ha="center", va="bottom", fontsize=7)

    # --- Sky lines ---
    if show_sky:
        for w_sky, name in SKY_LINES:
            if wave[0] < w_sky < wave[-1]:
                ax.axvline(w_sky, color="tab:green", lw=0.5, alpha=0.7, ls=":", zorder=1)
                ax.text(w_sky, ymax * 0.75, name, color="tab:green",
                        rotation=90, ha="center", va="top", fontsize=7)

    # --- Telluric bands ---
    if show_telluric:
        for w1, w2, name in TELLURIC_BANDS:
            if w1 < wave[-1] and w2 > wave[0]:
                w1c = max(w1, wave[0])
                w2c = min(w2, wave[-1])
                ax.axvspan(w1c, w2c, color="tab:green", alpha=0.1, lw=0, zorder=0)
                ax.text((w1c + w2c) / 2, ymax * 0.10, "Sky",
                        color="tab:green", rotation=90, ha="center", va="top", fontsize=7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return fig, ax


# ---------------------------------------------------------------------------
# Convenience: read a _red.fits file and plot a single fiber
# ---------------------------------------------------------------------------

def plot_red_fiber(
    red_path: str | Path,
    fiber_idx: int,
    *,
    ext_flux: str = "PRIMARY",
    ext_wave: str = "WAVELA",
    **kwargs,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """
    Read a reduced FITS file and plot a single fiber spectrum.

    Parameters
    ----------
    red_path : str or Path
        Path to the ``_red.fits`` file.
    fiber_idx : int
        Zero-based fiber index (row in the PRIMARY/WAVELA arrays).
    ext_flux : str
        FITS extension name for flux (default ``"PRIMARY"``).
    ext_wave : str
        FITS extension name for wavelength (default ``"WAVELA"``).
    **kwargs
        Forwarded to :func:`plot_spectrum`.

    Returns
    -------
    fig, ax
    """
    red_path = Path(red_path)
    with fits.open(red_path) as hdul:
        flux = hdul[ext_flux].data[fiber_idx].astype(float)
        wave_data = hdul[ext_wave].data
        wave = (wave_data[fiber_idx] if wave_data.ndim == 2 else wave_data).astype(float)

        # Try to pull a useful title from the header
        hdr = hdul[0].header
        obj_name = hdr.get("OBJECT", "")
        fibname = hdr.get(f"FBRNAME{fiber_idx}", f"fiber {fiber_idx}")

    title = kwargs.pop("title", f"{red_path.name}  –  {fibname}" + (f"  ({obj_name})" if obj_name else ""))

    return plot_spectrum(wave, flux, title=title, **kwargs)


# ---------------------------------------------------------------------------
# Convenience: plot all (or selected) fibers from a _red.fits, one panel each
# ---------------------------------------------------------------------------

def plot_red_file(
    red_path: str | Path,
    fiber_indices: Optional[Sequence[int]] = None,
    *,
    ncols: int = 2,
    ext_flux: str = "PRIMARY",
    ext_wave: str = "WAVELA",
    target_z: float = 0.0,
    ylim: Optional[tuple] = None,
    figsize_per_panel: tuple = (10, 3),
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot spectra for multiple fibers from a single reduced file, one panel each.

    Parameters
    ----------
    red_path : str or Path
        Path to the ``_red.fits`` file.
    fiber_indices : sequence of int, optional
        Which fibers to plot.  Defaults to all fibers.
    ncols : int
        Number of columns in the subplot grid.
    ext_flux, ext_wave : str
        FITS extension names.
    target_z : float
        Redshift applied to line annotations.
    ylim : (ymin, ymax), optional
        Shared Y limits for all panels.
    figsize_per_panel : (w, h)
        Size of each individual panel.
    **kwargs
        Forwarded to :func:`plot_spectrum`.

    Returns
    -------
    fig, axes  (axes is a 2-D ndarray of Axes)
    """
    red_path = Path(red_path)
    with fits.open(red_path) as hdul:
        flux_all = hdul[ext_flux].data.astype(float)
        wave_data = hdul[ext_wave].data.astype(float)
        hdr = hdul[0].header

    if wave_data.ndim == 1:
        wave_all = np.tile(wave_data, (flux_all.shape[0], 1))
    else:
        wave_all = wave_data

    n_fibers = flux_all.shape[0]
    if fiber_indices is None:
        fiber_indices = list(range(n_fibers))

    nrows = int(np.ceil(len(fiber_indices) / ncols))
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for plot_idx, fib_idx in enumerate(fiber_indices):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row, col]
        wave = wave_all[fib_idx]
        flux = flux_all[fib_idx]
        fibname = hdr.get(f"FBRNAME{fib_idx}", f"fiber {fib_idx}")
        plot_spectrum(
            wave, flux,
            ax=ax,
            target_z=target_z,
            ylim=ylim,
            title=fibname,
            **kwargs,
        )

    # Hide unused axes
    for plot_idx in range(len(fiber_indices), nrows * ncols):
        row, col = divmod(plot_idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(red_path.name, fontsize=10)
    return fig, axes
