"""
Photometric utilities for kspecdr flux calibration.

Provides:
- AB magnitude ↔ flux density conversions
- Filter curve loading (``data/filters/*.dat``)
- Synthetic photometry through a bandpass
- ATLAS Refcat2 catalog loading and extraction into :class:`Photometry`

All magnitudes *inside this module* and in :class:`~.containers.Photometry`
objects are in the **AB system**.  Vega-to-AB corrections for Gaia and 2MASS
bands are applied automatically when reading from a catalogue row.

Filter file format
------------------
Two-column ASCII, whitespace-separated, no header::

    wavelength(µm)   transmission(0-1)

Internal representation uses Angstrom throughout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .containers import FilterCurve, Photometry, Spectrum1D

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/kspecdr/fluxcal/photometry.py  →  repo root  (4 × .parent)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_FILTER_DIR = _REPO_ROOT / "data" / "filters"

# ---------------------------------------------------------------------------
# Filter metadata registry
# ---------------------------------------------------------------------------

# For each filter:
#   "system"     : "AB" or "Vega"
#   "vega_to_ab" : scalar added to m_Vega to obtain m_AB (0.0 for AB filters)
#
# Vega-to-AB corrections:
#   Gaia DR2  — Maíz Apellániz & Weiler 2018 / Evans+ 2018
#   2MASS     — Blanton & Roweis 2007, Table 1
FILTER_INFO: Dict[str, Dict] = {
    # Pan-STARRS1 — AB system
    "ps1_g":    {"system": "AB", "vega_to_ab": 0.000},
    "ps1_r":    {"system": "AB", "vega_to_ab": 0.000},
    "ps1_i":    {"system": "AB", "vega_to_ab": 0.000},
    "ps1_z":    {"system": "AB", "vega_to_ab": 0.000},
    "ps1_y":    {"system": "AB", "vega_to_ab": 0.000},
    # Gaia DR2 — Vega-like system
    "gaia_g":   {"system": "Vega", "vega_to_ab": +0.1069},
    "gaia_bp":  {"system": "Vega", "vega_to_ab": +0.0676},
    "gaia_rp":  {"system": "Vega", "vega_to_ab": -0.3958},
    # 2MASS — Vega system
    "2mass_j":  {"system": "Vega", "vega_to_ab": +0.91},
    "2mass_h":  {"system": "Vega", "vega_to_ab": +1.39},
    "2mass_k":  {"system": "Vega", "vega_to_ab": +1.85},
}

# Mapping: filter name → (magnitude column, error column) in Refcat2 CSV
_REFCAT2_COLUMNS: Dict[str, Optional[tuple[str, str]]] = {
    "ps1_g":   ("g",    "dg"),
    "ps1_r":   ("r",    "dr"),
    "ps1_i":   ("i",    "di"),
    "ps1_z":   ("z",    "dz"),
    "ps1_y":   None,               # not present in Refcat2
    "gaia_g":  ("Gaia", "dGaia"),
    "gaia_bp": ("BP",   "dBP"),
    "gaia_rp": ("RP",   "dRP"),
    "2mass_j": ("J",    "dJ"),
    "2mass_h": ("H",    "dH"),
    "2mass_k": ("K",    "dK"),
}

# Default band set for photometric calibration
DEFAULT_BANDS: List[str] = [
    "gaia_g", "gaia_bp", "gaia_rp",
    "ps1_g", "ps1_r", "ps1_i", "ps1_z",
    "2mass_j", "2mass_h", "2mass_k",
]

# Sentinel values used in Refcat2 to indicate missing measurements
_REFCAT2_SENTINEL_MAG = 99.0   # magnitudes ≥ this → missing
_REFCAT2_SENTINEL_ERR = 9.0    # errors    ≥ this → missing

# Speed of light in Å/s  (used for f_λ ↔ f_ν conversion)
_C_ANG_PER_S = 2.99792458e18

# ---------------------------------------------------------------------------
# In-memory cache for filter curves
# ---------------------------------------------------------------------------
_filter_cache: Dict[str, FilterCurve] = {}


# ---------------------------------------------------------------------------
# AB magnitude ↔ flux density
# ---------------------------------------------------------------------------

def ab_mag_to_fnu(mag: float | np.ndarray,
                  mag_err: float | np.ndarray | None = None,
                  ) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert AB magnitude(s) to f_ν in erg/s/cm²/Hz.

    ``m_AB = −2.5 log₁₀(f_ν) − 48.60``

    Parameters
    ----------
    mag : float or ndarray
        AB magnitude(s).
    mag_err : float or ndarray, optional
        1-sigma magnitude error(s).

    Returns
    -------
    fnu : ndarray
        Flux density in erg/s/cm²/Hz.
    fnu_err : ndarray or None
        Propagated 1-sigma uncertainty on *fnu*, or None if *mag_err* is None.
    """
    mag = np.asarray(mag, dtype=float)
    fnu = 10.0 ** (-(mag + 48.60) / 2.5)
    if mag_err is None:
        return fnu, None
    mag_err = np.asarray(mag_err, dtype=float)
    fnu_err = fnu * mag_err * np.log(10.0) / 2.5
    return fnu, fnu_err


def fnu_to_ab_mag(fnu: float | np.ndarray,
                  fnu_err: float | np.ndarray | None = None,
                  ) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert f_ν (erg/s/cm²/Hz) to AB magnitude(s).

    Parameters
    ----------
    fnu : float or ndarray
        Flux density in erg/s/cm²/Hz.  Must be positive.
    fnu_err : float or ndarray, optional
        1-sigma uncertainty.

    Returns
    -------
    mag : ndarray
    mag_err : ndarray or None
    """
    fnu = np.asarray(fnu, dtype=float)
    mag = -2.5 * np.log10(fnu) - 48.60
    if fnu_err is None:
        return mag, None
    fnu_err = np.asarray(fnu_err, dtype=float)
    mag_err = 2.5 / np.log(10.0) * np.abs(fnu_err / fnu)
    return mag, mag_err


def ab_mag_to_flam(mag: float | np.ndarray,
                   wave_eff_ang: float,
                   mag_err: float | np.ndarray | None = None,
                   ) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert AB magnitude to f_λ (erg/s/cm²/Å) at effective wavelength.

    ``f_λ = f_ν × c / λ²``

    Parameters
    ----------
    mag : float or ndarray
        AB magnitude(s).
    wave_eff_ang : float
        Effective wavelength of the filter in Angstrom.
    mag_err : float or ndarray, optional

    Returns
    -------
    flam : ndarray
    flam_err : ndarray or None
    """
    fnu, fnu_err = ab_mag_to_fnu(mag, mag_err)
    flam = fnu * _C_ANG_PER_S / wave_eff_ang ** 2
    if fnu_err is None:
        return flam, None
    flam_err = fnu_err * _C_ANG_PER_S / wave_eff_ang ** 2
    return flam, flam_err


# ---------------------------------------------------------------------------
# Filter curve I/O
# ---------------------------------------------------------------------------

def load_filter_curve(filter_name: str) -> FilterCurve:
    """Load a filter transmission curve from ``data/filters/{filter_name}.dat``.

    Results are cached in memory after the first load.

    Parameters
    ----------
    filter_name : str
        One of the keys in :data:`FILTER_INFO`, e.g. ``"ps1_g"``.

    Returns
    -------
    FilterCurve
        Wavelength axis in Angstrom, transmission in [0, 1].

    Raises
    ------
    KeyError
        If *filter_name* is not in :data:`FILTER_INFO`.
    FileNotFoundError
        If the corresponding ``.dat`` file is missing.
    """
    if filter_name in _filter_cache:
        return _filter_cache[filter_name]

    if filter_name not in FILTER_INFO:
        raise KeyError(
            f"Unknown filter '{filter_name}'. "
            f"Known filters: {sorted(FILTER_INFO)}"
        )

    path = _FILTER_DIR / f"{filter_name}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"Filter file not found: {path}\n"
            "Run the filter extraction script or check data/filters/."
        )

    info = FILTER_INFO[filter_name]
    data = np.loadtxt(path)
    wave_um = data[:, 0]
    trans   = data[:, 1]

    # Convert µm → Å
    wave_ang = wave_um * 1e4

    fc = FilterCurve(
        name=filter_name,
        wavelength=wave_ang,
        transmission=trans,
        system=info["system"],
        vega_to_ab=info["vega_to_ab"],
    )
    _filter_cache[filter_name] = fc
    return fc


def load_filter_curves(filter_names: Sequence[str]) -> Dict[str, FilterCurve]:
    """Load multiple filters at once.

    Returns
    -------
    dict mapping filter name → :class:`FilterCurve`
    """
    return {name: load_filter_curve(name) for name in filter_names}


# ---------------------------------------------------------------------------
# Synthetic photometry
# ---------------------------------------------------------------------------

def synthetic_photometry(spectrum: Spectrum1D,
                          filter_curve: FilterCurve,
                          ) -> float:
    """Compute synthetic AB magnitude of *spectrum* through *filter_curve*.

    Uses the photon-counting (energy-per-photon weighted) convention:

    .. math::

        \\langle f_\\nu \\rangle =
            \\frac{\\int f_\\nu(\\lambda)\\, T(\\lambda)\\, \\lambda\\, d\\lambda}
                  {c \\int T(\\lambda) / \\lambda\\, d\\lambda}

        m_{\\rm AB} = -2.5 \\log_{10}(\\langle f_\\nu \\rangle) - 48.60

    where :math:`f_\\nu = f_\\lambda \\, \\lambda^2 / c`.

    Parameters
    ----------
    spectrum : Spectrum1D
        Observed or model spectrum with ``flux`` in erg/s/cm²/Å and
        ``wavelength`` in Angstrom.  Masked pixels are excluded.
    filter_curve : FilterCurve
        Bandpass transmission.

    Returns
    -------
    float
        Synthetic AB magnitude.  Returns ``np.nan`` if the filter is not
        covered by the spectrum or if the result is non-physical.

    Raises
    ------
    ValueError
        If *spectrum* flux and wavelength have mismatched shapes.
    """
    wave_spec = spectrum.wavelength
    flux_spec = spectrum.flux
    good_mask = spectrum.mask & np.isfinite(flux_spec) & (flux_spec > 0)

    wave_spec = wave_spec[good_mask]
    flux_spec = flux_spec[good_mask]

    # Check overlap
    w_lo, w_hi = filter_curve.wave_range
    if wave_spec[-1] < w_lo or wave_spec[0] > w_hi:
        logger.warning(
            "Spectrum (%.0f–%.0f Å) does not overlap filter '%s' (%.0f–%.0f Å)",
            wave_spec[0], wave_spec[-1],
            filter_curve.name, w_lo, w_hi,
        )
        return np.nan

    # Resample filter onto spectrum wavelength grid
    trans_on_spec = np.interp(
        wave_spec, filter_curve.wavelength, filter_curve.transmission,
        left=0.0, right=0.0,
    )

    # f_ν = f_λ × λ² / c  (both in Å units)
    fnu_spec = flux_spec * wave_spec ** 2 / _C_ANG_PER_S

    # Photon-count weighted average: ∫ f_ν T λ dλ / (c ∫ T/λ dλ)
    # Equivalently: numerator = ∫ f_λ T λ² dλ / c,
    #               denominator = c × ∫ T/λ dλ  → combine:
    # <f_ν> = ∫ fnu T λ dλ / (c ∫ T/λ dλ)
    # Simpler: integrate directly in log-λ space (= integrate fnu * T d[ln λ])
    ln_wave = np.log(wave_spec)
    numerator   = np.trapz(fnu_spec * trans_on_spec, ln_wave)
    denominator = np.trapz(trans_on_spec, ln_wave)

    if denominator <= 0 or numerator <= 0:
        return np.nan

    fnu_mean = numerator / denominator
    mag_ab = -2.5 * np.log10(fnu_mean) - 48.60
    return float(mag_ab)


def synthetic_color(spectrum: Spectrum1D,
                    filter1: FilterCurve,
                    filter2: FilterCurve,
                    ) -> float:
    """Return synthetic color ``m_filter1 − m_filter2`` (AB mags)."""
    m1 = synthetic_photometry(spectrum, filter1)
    m2 = synthetic_photometry(spectrum, filter2)
    return m1 - m2


# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------

def load_standard_star_catalog(catalog_path: str | Path):
    """Load an ATLAS Refcat2-format standard-star CSV catalog.

    Parameters
    ----------
    catalog_path : str or Path
        Path to the CSV file.  Must have a header row matching the Refcat2
        column names (see ``resources/comm/.../standard_star_atlas_refcat2.csv``).

    Returns
    -------
    astropy.table.Table
        Full catalog table with all photometric columns.  Columns with
        ``99.9`` sentinel values are *not* filtered here; use
        :func:`photometry_from_catalog_row` to obtain validated
        :class:`Photometry` objects.
    """
    from astropy.table import Table

    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)

    tbl = Table.read(str(catalog_path), format="ascii.csv")
    logger.debug(
        "Loaded standard-star catalog: %d rows from %s", len(tbl), catalog_path
    )
    return tbl


def photometry_from_catalog_row(row,
                                 bands: Optional[List[str]] = None,
                                 ) -> Photometry:
    """Extract a :class:`Photometry` object from one Refcat2 catalogue row.

    Vega-to-AB corrections for Gaia and 2MASS bands are applied automatically.
    Bands with sentinel values (magnitude ≥ 99 or error ≥ 9.9, or missing)
    are stored as ``nan`` and excluded by :meth:`~Photometry.valid_bands`.

    Parameters
    ----------
    row : astropy.table.Row or dict-like
        One row from a table returned by :func:`load_standard_star_catalog`.
    bands : list of str, optional
        Which filter names to extract.  Defaults to :data:`DEFAULT_BANDS`.

    Returns
    -------
    Photometry
        Magnitudes in the AB system.
    """
    if bands is None:
        bands = DEFAULT_BANDS

    mags = np.full(len(bands), np.nan)
    errs = np.full(len(bands), np.nan)

    for i, fname in enumerate(bands):
        cols = _REFCAT2_COLUMNS.get(fname)
        if cols is None:
            continue  # filter not in Refcat2

        mag_col, err_col = cols
        try:
            mag = float(row[mag_col])
            err = float(row[err_col])
        except (KeyError, TypeError, ValueError):
            continue

        # Sentinel value check
        if mag >= _REFCAT2_SENTINEL_MAG or err >= _REFCAT2_SENTINEL_ERR or err <= 0:
            continue

        # Convert Vega → AB
        vega_to_ab = FILTER_INFO[fname]["vega_to_ab"]
        mags[i] = mag + vega_to_ab
        errs[i] = err  # error is the same in both systems

    # Build metadata from available Refcat2 columns
    meta: Dict = {}
    for key in ("RA", "Dec", "objid", "Teff", "AGaia", "Ag"):
        try:
            meta[key.lower()] = float(row[key])
        except (KeyError, TypeError, ValueError):
            pass

    return Photometry(
        filter_names=list(bands),
        magnitudes=mags,
        mag_errors=errs,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Photometric color → Teff estimate
# ---------------------------------------------------------------------------

def estimate_teff_from_color(photometry: Photometry,
                              method: str = "bp_rp",
                              ) -> tuple[float, tuple[float, float]]:
    """Rough photometric Teff estimate for F/G/K stars from broadband color.

    Parameters
    ----------
    photometry : Photometry
    method : str
        Color index to use.  Supported:

        * ``"bp_rp"``  — Gaia BP − RP (recommended; ~100 K precision for F stars)
        * ``"g_r"``    — PS1 g − r

    Returns
    -------
    teff : float
        Estimated effective temperature in K.
    teff_range : (float, float)
        Conservative ±2-sigma search window (teff_lo, teff_hi).
    """
    if method == "bp_rp":
        try:
            m_bp, e_bp = photometry.get_band("gaia_bp")
            m_rp, e_rp = photometry.get_band("gaia_rp")
        except KeyError:
            return _fallback_teff()
        if not (np.isfinite(m_bp) and np.isfinite(m_rp)):
            return _fallback_teff()
        color = m_bp - m_rp   # AB system (corrections cancel for color)
        # Polynomial fit to Casagrande+ 2010 dwarf calibration (BP-RP → Teff)
        # Valid roughly for 0.2 < BP-RP < 1.5, [Fe/H] ~ 0
        teff = _bprp_to_teff(color)
    elif method == "g_r":
        try:
            m_g, e_g = photometry.get_band("ps1_g")
            m_r, e_r = photometry.get_band("ps1_r")
        except KeyError:
            return _fallback_teff()
        if not (np.isfinite(m_g) and np.isfinite(m_r)):
            return _fallback_teff()
        color = m_g - m_r
        teff = _gr_to_teff(color)
    else:
        raise ValueError(f"Unsupported method: '{method}'")

    # ±600 K window covers ~2 steps of the BOSZ subgrid (step = 250 K)
    teff_range = (max(4500.0, teff - 600.0), min(9000.0, teff + 600.0))
    return teff, teff_range


def _bprp_to_teff(bprp: float) -> float:
    """BP−RP (AB, Gaia DR2) → Teff (K) polynomial.

    Calibrated against Casagrande et al. 2010 dwarf effective temperatures
    for 0.3 ≤ BP−RP ≤ 1.6 (roughly G0–K5).  Extrapolated outside this range.
    """
    # Coefficients from a degree-3 polynomial fit to Table 3 of
    # Casagrande+ 2010 (IRFM, solar metallicity, dwarfs)
    # BP-RP here is in the Gaia DR2 photometric system (Vega-corrected AB)
    # NOTE: replace with a more precise calibration when available.
    c = np.array([8899.3, -4291.5, 1991.8, -470.9])
    x = np.clip(bprp, 0.0, 2.5)
    return float(np.polyval(c[::-1], x))    # polyval expects highest-degree first


def _gr_to_teff(gr: float) -> float:
    """PS1 g−r (AB) → Teff (K) rough calibration."""
    # Simple linear approximation (g−r ≈ 0 → 6500 K; g−r ≈ 0.5 → 5000 K)
    return float(np.clip(6500.0 - 3000.0 * gr, 4000.0, 9000.0))


def _fallback_teff() -> tuple[float, tuple[float, float]]:
    """Return a wide default window when color is unavailable."""
    logger.warning(
        "Cannot estimate Teff from photometry; using full F-star range 5000–8000 K"
    )
    return 6500.0, (5000.0, 8000.0)


# ---------------------------------------------------------------------------
# Placeholder for future catalog queries
# ---------------------------------------------------------------------------

def query_photometry(ra: float, dec: float,  # noqa: ARG001
                     radius_arcsec: float = 2.0,
                     catalog: str = "refcat2",
                     ) -> Photometry:
    """Query an external catalogue for standard-star photometry.

    .. note::
        **Not yet implemented.**  Use :func:`load_standard_star_catalog` and
        :func:`photometry_from_catalog_row` with a pre-downloaded CSV file.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "Online catalogue queries are not yet implemented.  "
        "Use load_standard_star_catalog() with a local Refcat2 CSV file."
    )
