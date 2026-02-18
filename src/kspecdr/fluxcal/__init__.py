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

Implemented (P1)
----------------
templates
    :class:`~.templates.TemplateLibrary` (BOSZ 2024 index, lazy load),
    :func:`~.templates.prepare_template`, :func:`~.templates.resample_spectrum`
continuum
    :func:`~.continuum.normalize_continuum` (B-spline / polynomial /
    running-median with iterative sigma-clipping),
    :func:`~.continuum.normalize_with_model_continuum`
matching
    :func:`~.matching.select_best_template`,
    :func:`~.matching.cross_correlate_rv`,
    :func:`~.matching.score_template_fit`

Planned (P2)
------------
calibration  — Per-star and combined calibration vectors, application to science spectra

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
from .continuum import (
    fit_continuum,
    normalize_continuum,
    normalize_with_model_continuum,
)
from .masks import (
    apply_mask_regions,
    apply_named_mask,
    load_mask_regions,
)
from .matching import (
    cross_correlate_rv,
    score_template_fit,
    select_best_template,
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
from .templates import (
    TemplateLibrary,
    parse_bosz_filename,
    prepare_template,
    resample_spectrum,
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
    # templates
    "TemplateLibrary",
    "parse_bosz_filename",
    "prepare_template",
    "resample_spectrum",
    # continuum
    "normalize_continuum",
    "normalize_with_model_continuum",
    "fit_continuum",
    # matching
    "select_best_template",
    "cross_correlate_rv",
    "score_template_fit",
]
