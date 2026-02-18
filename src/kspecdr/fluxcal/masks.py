"""
Wavelength mask utilities for kspecdr flux calibration.

Masks are lists of ``(lam_lo, lam_hi)`` pairs in Angstrom that mark regions
to *exclude* from calibration fits, continuum normalisation, and template
matching (e.g. telluric absorption bands, detector bad regions).

Mask file format
----------------
Two-column ASCII, whitespace-separated.  Lines starting with ``#`` are
comments and are ignored::

    6860.0    6960.0    # O2 B-band
    7150.0    7340.0    # H2O

The default mask is loaded from ``data/masks/telluric_default.dat``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .containers import Spectrum1D

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/kspecdr/fluxcal/masks.py  →  repo root (4 × .parent)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_MASK_DIR  = _REPO_ROOT / "data" / "masks"

# In-memory cache for loaded mask files
_mask_cache: dict[str, List[tuple[float, float]]] = {}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_mask_regions(
    mask_name: str = "telluric_default",
) -> List[tuple[float, float]]:
    """Load wavelength exclusion regions from ``data/masks/{mask_name}.dat``.

    Results are cached after the first load.

    Parameters
    ----------
    mask_name : str
        Base name of the mask file (without ``.dat``).
        Default: ``"telluric_default"``.

    Returns
    -------
    list of (lam_lo, lam_hi)
        Wavelength intervals in Angstrom to be masked (excluded).

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist.
    """
    if mask_name in _mask_cache:
        return _mask_cache[mask_name]

    path = _MASK_DIR / f"{mask_name}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"Mask file not found: {path}\n"
            "Available masks: "
            + ", ".join(p.stem for p in _MASK_DIR.glob("*.dat"))
        )

    regions: List[tuple[float, float]] = []
    with open(path) as fh:
        for line in fh:
            stripped = line.split("#")[0].strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            lo, hi = float(parts[0]), float(parts[1])
            if lo >= hi:
                logger.warning(
                    "Skipping degenerate mask region in '%s': %.1f >= %.1f",
                    mask_name, lo, hi,
                )
                continue
            regions.append((lo, hi))

    logger.debug(
        "Loaded %d mask regions from '%s'", len(regions), mask_name
    )
    _mask_cache[mask_name] = regions
    return regions


def load_mask_array(
    wavelength: np.ndarray,
    mask_name: str = "telluric_default",
) -> np.ndarray:
    """Return a boolean mask array for a given wavelength axis.

    Parameters
    ----------
    wavelength : ndarray, shape (N,)
        Wavelength axis in Angstrom.
    mask_name : str
        Mask file to load (see :func:`load_mask_regions`).

    Returns
    -------
    ndarray of bool, shape (N,)
        True for *good* (unmasked) pixels; False inside any exclusion region.
    """
    regions = load_mask_regions(mask_name)
    return _regions_to_array(wavelength, regions)


# ---------------------------------------------------------------------------
# Application to spectra
# ---------------------------------------------------------------------------

def apply_mask_regions(
    spectrum: Spectrum1D,
    regions: List[tuple[float, float]],
) -> Spectrum1D:
    """Set ``mask=False`` for all pixels inside any exclusion region.

    Returns a *new* :class:`~.containers.Spectrum1D` with an updated mask;
    the original is not modified.

    Parameters
    ----------
    spectrum : Spectrum1D
    regions : list of (lam_lo, lam_hi)
        Wavelength intervals to exclude (Angstrom).

    Returns
    -------
    Spectrum1D
        New spectrum with pixels inside *regions* masked out.
    """
    new_mask = spectrum.mask.copy()
    bad = _regions_to_array(spectrum.wavelength, regions)
    new_mask &= bad   # bad is True for GOOD pixels; AND with existing mask

    return Spectrum1D(
        wavelength=spectrum.wavelength.copy(),
        flux=spectrum.flux.copy(),
        variance=spectrum.variance.copy(),
        mask=new_mask,
        meta=dict(spectrum.meta),
    )


def apply_named_mask(
    spectrum: Spectrum1D,
    mask_name: str = "telluric_default",
) -> Spectrum1D:
    """Convenience wrapper: load mask by name and apply to *spectrum*.

    Parameters
    ----------
    spectrum : Spectrum1D
    mask_name : str
        Mask file base name (see :func:`load_mask_regions`).

    Returns
    -------
    Spectrum1D
    """
    regions = load_mask_regions(mask_name)
    return apply_mask_regions(spectrum, regions)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _regions_to_array(
    wavelength: np.ndarray,
    regions: List[tuple[float, float]],
) -> np.ndarray:
    """Build a boolean array: True = pixel is *outside* all exclusion regions.

    Parameters
    ----------
    wavelength : ndarray, shape (N,)
    regions : list of (lam_lo, lam_hi)

    Returns
    -------
    ndarray of bool, shape (N,)
    """
    good = np.ones(len(wavelength), dtype=bool)
    for lo, hi in regions:
        good &= ~((wavelength >= lo) & (wavelength <= hi))
    return good


def mask_coverage_fraction(
    wavelength: np.ndarray,
    mask_name: str = "telluric_default",
) -> float:
    """Return the fraction of wavelength pixels inside the named mask.

    Useful for QC: large fractions (> 0.3) may indicate a wavelength unit
    mismatch.

    Parameters
    ----------
    wavelength : ndarray
        In Angstrom.
    mask_name : str

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    good = load_mask_array(wavelength, mask_name)
    return float((~good).sum() / len(wavelength))


def combine_regions(
    *region_lists: List[tuple[float, float]],
) -> List[tuple[float, float]]:
    """Merge multiple region lists and sort by start wavelength.

    Overlapping or adjacent regions are *not* merged — they are stacked and
    applied independently, which produces the same result.

    Parameters
    ----------
    *region_lists : list of (lam_lo, lam_hi)

    Returns
    -------
    list of (lam_lo, lam_hi)
    """
    combined = []
    for rl in region_lists:
        combined.extend(rl)
    return sorted(combined, key=lambda r: r[0])
