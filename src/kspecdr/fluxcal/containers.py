"""
Data containers for kspecdr flux calibration.

All containers are plain dataclasses with light validation in ``__post_init__``.
No heavy dependencies — only numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


__all__ = [
    "Spectrum1D",
    "Photometry",
    "FilterCurve",
    "StellarTemplate",
    "CalibrationVector",
    "FluxCalibrationResult",
]


@dataclass
class Spectrum1D:
    """Wavelength-calibrated 1D spectrum with uncertainty and pixel mask.

    Parameters
    ----------
    wavelength : ndarray, shape (N,)
        Wavelength axis in Angstrom.
    flux : ndarray, shape (N,)
        Flux values (counts, counts/s, or physical flux density depending on
        pipeline stage).
    variance : ndarray, shape (N,)
        Per-pixel variance, same units as ``flux**2``.
    mask : ndarray of bool, shape (N,)
        True for *good* (usable) pixels.
    meta : dict
        Freeform metadata.  Recognised keys:

        ``fiber_id``, ``fiber_name``, ``exptime``, ``bunit``,
        ``continuum`` (ndarray) — theoretical continuum carried from BOSZ,
        ``rv_kms`` — applied radial-velocity correction.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.variance = np.asarray(self.variance, dtype=float)
        self.mask = np.asarray(self.mask, dtype=bool)
        n = len(self.wavelength)
        if not (len(self.flux) == len(self.variance) == len(self.mask) == n):
            raise ValueError(
                "wavelength, flux, variance, and mask must all have the same length"
            )

    @property
    def ivar(self) -> np.ndarray:
        """Inverse variance; zero for masked or non-positive variance pixels."""
        good = self.mask & (self.variance > 0)
        iv = np.zeros_like(self.variance)
        iv[good] = 1.0 / self.variance[good]
        return iv

    @property
    def n_pixels(self) -> int:
        return len(self.wavelength)

    @property
    def wave_range(self) -> tuple[float, float]:
        """(wave_min, wave_max) in Angstrom."""
        return float(self.wavelength[0]), float(self.wavelength[-1])

    @property
    def n_good(self) -> int:
        """Number of unmasked pixels."""
        return int(self.mask.sum())


@dataclass
class Photometry:
    """Broadband photometric measurements for a single standard star.

    All magnitudes are stored in the **AB system** regardless of the native
    photometric system of the catalogue.  The conversion from Vega to AB is
    applied when the object is constructed (see
    :func:`~kspecdr.fluxcal.photometry.photometry_from_catalog_row`).

    Parameters
    ----------
    filter_names : list of str
        Names matching the keys in :data:`~kspecdr.fluxcal.photometry.FILTER_INFO`
        (e.g. ``"ps1_g"``, ``"gaia_g"``).
    magnitudes : ndarray, shape (N_bands,)
        AB magnitudes.  ``nan`` marks unavailable bands.
    mag_errors : ndarray, shape (N_bands,)
        Magnitude errors (1-sigma).  ``nan`` or non-positive marks unavailable.
    meta : dict
        Optional metadata: ``ra``, ``dec``, ``objid``, ``teff_catalog``,
        ``a_gaia`` (extinction), ``a_g`` (extinction in g-band).
    """

    filter_names: List[str]
    magnitudes: np.ndarray
    mag_errors: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.magnitudes = np.asarray(self.magnitudes, dtype=float)
        self.mag_errors = np.asarray(self.mag_errors, dtype=float)
        if len(self.filter_names) != len(self.magnitudes):
            raise ValueError("filter_names and magnitudes must have the same length")

    def get_band(self, name: str) -> tuple[float, float]:
        """Return ``(mag, mag_err)`` for the named filter.

        Raises ``KeyError`` if *name* is not in this object's filter list.
        """
        try:
            idx = self.filter_names.index(name)
        except ValueError:
            raise KeyError(name) from None
        return float(self.magnitudes[idx]), float(self.mag_errors[idx])

    def valid_bands(self) -> List[str]:
        """Filter names with finite magnitude *and* positive, finite error."""
        return [
            n
            for n, m, e in zip(self.filter_names, self.magnitudes, self.mag_errors)
            if np.isfinite(m) and np.isfinite(e) and e > 0.0
        ]

    def __len__(self) -> int:
        return len(self.filter_names)


@dataclass
class FilterCurve:
    """Transmission curve for one photometric bandpass.

    Parameters
    ----------
    name : str
        Identifier matching a file in ``data/filters/`` (without ``.dat``).
    wavelength : ndarray, shape (M,)
        Wavelength in **Angstrom** (converted from µm on load).
    transmission : ndarray, shape (M,)
        Fractional transmission, 0–1.
    system : str
        Photometric system: ``"AB"`` or ``"Vega"``.
    vega_to_ab : float
        Offset to add to a Vega magnitude to obtain an AB magnitude:
        ``m_AB = m_Vega + vega_to_ab``.  Zero for AB-system filters.
    """

    name: str
    wavelength: np.ndarray
    transmission: np.ndarray
    system: str = "AB"
    vega_to_ab: float = 0.0

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.transmission = np.asarray(self.transmission, dtype=float)

    @property
    def wave_eff(self) -> float:
        """Transmission-weighted mean wavelength in Angstrom."""
        return float(
            np.trapz(self.wavelength * self.transmission, self.wavelength)
            / np.trapz(self.transmission, self.wavelength)
        )

    @property
    def wave_range(self) -> tuple[float, float]:
        """Wavelength range where transmission > 1 % of peak (Angstrom)."""
        thresh = 0.01 * self.transmission.max()
        above = self.wavelength[self.transmission > thresh]
        return float(above[0]), float(above[-1])


@dataclass
class StellarTemplate:
    """A single BOSZ 2024 stellar model spectrum.

    Parameters
    ----------
    wavelength : ndarray, shape (K,)
        Wavelength in Angstrom (native BOSZ log-λ grid at R = 10,000).
    flux : ndarray, shape (K,)
        Surface flux density in erg/s/cm²/Å  (= 4π × H, column 3 of file).
    continuum : ndarray, shape (K,)
        Theoretical continuum in the same units (= 4π × C, column 4 of file).
        Useful as a first-pass normalization without fitting a spline.
    teff : float
        Effective temperature in K.
    logg : float
        Surface gravity log(g) in cgs.
    feh : float
        Overall metallicity [M/H].
    alpha_m : float
        Alpha-element enhancement [α/M].
    carbon_m : float
        Carbon enhancement [C/M].
    vmicro : float
        Microturbulent velocity in km/s.
    atmos_model : str
        Atmosphere model code: ``"mp"`` (MARCS plane-parallel) or
        ``"ap"`` (ATLAS9 plane-parallel).
    source : str
        Provenance string — typically the original filename.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    continuum: np.ndarray
    teff: float
    logg: float
    feh: float
    alpha_m: float = 0.0
    carbon_m: float = 0.0
    vmicro: float = 1.0
    atmos_model: str = ""
    source: str = "BOSZ2024"

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.continuum = np.asarray(self.continuum, dtype=float)

    def to_spectrum1d(self) -> Spectrum1D:
        """Wrap as :class:`Spectrum1D` (variance = 0, mask = all good)."""
        return Spectrum1D(
            wavelength=self.wavelength.copy(),
            flux=self.flux.copy(),
            variance=np.zeros_like(self.flux),
            mask=np.ones(len(self.flux), dtype=bool),
            meta={
                "teff": self.teff,
                "logg": self.logg,
                "feh": self.feh,
                "alpha_m": self.alpha_m,
                "source": self.source,
                "continuum": self.continuum.copy(),
            },
        )

    @property
    def label(self) -> str:
        """Short human-readable parameter label."""
        return (
            f"Teff={self.teff:.0f} logg={self.logg:.1f} "
            f"[M/H]={self.feh:+.2f} [α/M]={self.alpha_m:+.2f}"
        )


@dataclass
class CalibrationVector:
    """Wavelength-dependent factor converting observed counts to physical flux.

    ``flux_calibrated[i] = counts[i] * cal_factor[i]``

    Parameters
    ----------
    wavelength : ndarray, shape (N,)
        Wavelength axis in Angstrom.
    cal_factor : ndarray, shape (N,)
        Calibration factor in erg/s/cm²/Å per count (or per count/s).
    cal_variance : ndarray, shape (N,)
        Variance on ``cal_factor``.
    mask : ndarray of bool, shape (N,)
        True for *reliable* (unmasked) pixels.  Telluric bands and other
        problematic regions are set to False.
    meta : dict
        Per-star metadata.  Recognised keys:

        ``star_name``, ``fiber_id``,
        ``teff``, ``logg``, ``feh``, ``alpha_m``,
        ``scale_factor`` (float) — photometric normalisation,
        ``band_residuals`` (dict) — synth − obs per filter,
        ``chi2``, ``ndof``, ``rv_kms``.
    """

    wavelength: np.ndarray
    cal_factor: np.ndarray
    cal_variance: np.ndarray
    mask: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.cal_factor = np.asarray(self.cal_factor, dtype=float)
        self.cal_variance = np.asarray(self.cal_variance, dtype=float)
        self.mask = np.asarray(self.mask, dtype=bool)

    @property
    def cal_error(self) -> np.ndarray:
        """1-sigma uncertainty on ``cal_factor``."""
        return np.sqrt(np.where(self.cal_variance >= 0, self.cal_variance, 0.0))

    @property
    def n_good(self) -> int:
        return int(self.mask.sum())


@dataclass
class FluxCalibrationResult:
    """Complete output of the flux calibration procedure for one exposure.

    Parameters
    ----------
    combined_vector : CalibrationVector
        Final combined calibration curve to apply to all science fibers.
    per_star_vectors : list of CalibrationVector
        Individual per-star calibration vectors before combination.
    per_star_residuals : list of ndarray
        Fractional residual arrays ``(Cal_star − Cal_combined) / Cal_combined``
        for each star.  Shape (N,) each.
    summary : dict
        Exposure-level summary.  Keys:

        ``n_stars_used`` (int), ``n_stars_rejected`` (int),
        ``rms_scatter`` (float, fractional), ``wavelength_range`` (tuple),
        ``per_star_metrics`` (list of dict).
    """

    combined_vector: CalibrationVector
    per_star_vectors: List[CalibrationVector]
    per_star_residuals: List[np.ndarray]
    summary: Dict = field(default_factory=dict)

    @property
    def n_stars(self) -> int:
        return len(self.per_star_vectors)
