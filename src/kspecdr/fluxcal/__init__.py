"""
kspecdr.fluxcal — Spectrophotometric flux calibration subpackage.

See docs/planning/fluxcal_design.md for the full implementation plan.

Implemented (P0)
----------------
containers
    :class:`~.containers.Spectrum1D`, :class:`~.containers.Photometry`,
    :class:`~.containers.FilterCurve`, :class:`~.containers.StellarTemplate`,
    :class:`~.containers.CalibrationVector`,
    :class:`~.containers.FluxCalibrationResult`
photometry
    AB mag ↔ flux density, filter curve loading, synthetic photometry,
    ATLAS Refcat2 catalog I/O, photometric Teff estimate
masks
    Telluric / bad-region mask I/O and application

Planned (P1 / P2)
-----------------
templates    — TemplateLibrary (BOSZ 2024), resolution matching, resampling
continuum    — Continuum normalization (B-spline, polynomial, running-median)
matching     — Template selection, RV cross-correlation
calibration  — Per-star and combined calibration vectors, application

Utilities
---------
download_bosz — Download BOSZ 2024 template subgrid from MAST
"""

from .containers import (
    CalibrationVector,
    FilterCurve,
    FluxCalibrationResult,
    Photometry,
    Spectrum1D,
    StellarTemplate,
)
from .masks import (
    apply_mask_regions,
    apply_named_mask,
    load_mask_regions,
)
from .photometry import (
    DEFAULT_BANDS,
    FILTER_INFO,
    ab_mag_to_flam,
    ab_mag_to_fnu,
    estimate_teff_from_color,
    fnu_to_ab_mag,
    load_filter_curve,
    load_filter_curves,
    load_standard_star_catalog,
    photometry_from_catalog_row,
    synthetic_photometry,
)

__all__ = [
    # containers
    "Spectrum1D",
    "Photometry",
    "FilterCurve",
    "StellarTemplate",
    "CalibrationVector",
    "FluxCalibrationResult",
    # photometry
    "FILTER_INFO",
    "DEFAULT_BANDS",
    "ab_mag_to_fnu",
    "ab_mag_to_flam",
    "fnu_to_ab_mag",
    "load_filter_curve",
    "load_filter_curves",
    "synthetic_photometry",
    "estimate_teff_from_color",
    "load_standard_star_catalog",
    "photometry_from_catalog_row",
    # masks
    "load_mask_regions",
    "apply_mask_regions",
    "apply_named_mask",
]
