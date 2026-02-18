"""
kspecdr.fluxcal — Spectrophotometric flux calibration subpackage.

This subpackage is under active development. See docs/planning/fluxcal_design.md
for the implementation plan.

Modules (planned):
    containers   — Spectrum1D, Photometry, StellarTemplate, CalibrationVector, ...
    photometry   — AB mag ↔ flux density, filter curves, synthetic photometry
    templates    — TemplateLibrary (BOSZ 2024), resolution matching, resampling
    continuum    — Continuum normalization (B-spline, polynomial, running-median)
    matching     — Template selection, RV cross-correlation
    calibration  — Per-star and combined calibration vectors, application
    masks        — Telluric / bad-region mask I/O

Utilities:
    download_bosz — Download BOSZ 2024 template subgrid from MAST
"""
