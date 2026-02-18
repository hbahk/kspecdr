# Flux Calibration Design Plan for `kspecdr`

> **Status**: Implementation plan — not yet implemented.
> **Last updated**: 2026-02-17

---

## 0. Scope

Add SDSS-style spectrophotometric flux calibration to `kspecdr`. The procedure
uses spectrophotometric standard stars (typically F-type) observed on the same
plate/exposure to derive a wavelength-dependent calibration vector that converts
extracted counts to physical flux density (erg/s/cm²/Å).

---

## 1. Current State

| Area | Status |
|---|---|
| FITS I/O (`ImageFile`) | Done — Primary + VARIANCE + WAVELA + FIBRES extensions |
| Spectrum container | **None** — data flows as bare `(flux, variance, wavelength)` numpy arrays |
| Wavelength calibration | Done — arc line matching, polynomial fitting, scrunching |
| Extraction with variance propagation | Done — `sum_extract` |
| Fiber metadata | Done — FIBRES BinTableHDU with TYPE/NAME columns |
| Flux calibration | **Stub only** — `reduce_object.py:232-241` raises `NotImplementedError` |
| Photometry / template / flux-density utilities | **None** |

---

## 2. Design Decisions (Resolved)

### 2.1 Template Library

Use the **PHOENIX/BT-Settl** grid as the starting point. These are freely
available, cover F-star parameter space well, and come as FITS files with
well-documented wavelength grids.

### 2.2 Fiber Type Codes

The assign-file `target_class` column maps to fiber TYPE codes as follows:

| `target_class` | TYPE code | Meaning |
|---|---|---|
| 0 | `U` | Unallocated fiber |
| 1 | `P` | Program (science) target — star |
| 2 | `P` | Program (science) target — galaxy |
| 8 | `C` | Standard star (**new type**) |
| 9 | `S` | Sky fiber |

The new `C` type must be added to any code that currently handles fiber types
(`read_fiber_types`, extraction logic, sky subtraction, etc.) so that standard
star fibers are recognized and routed to the calibration pathway.

### 2.3 Photometry Input

Photometry for standard stars comes from a **separate catalog table file** in
CSV format. The reference format is ATLAS Refcat2, as used in the commissioning
data. Columns (from `standard_star_atlas_refcat2.csv`):

```
dstDegrees, objid, RA, Dec, plx, dplx, pmra, dpmra, pmdec, dpmdec,
Gaia, dGaia, BP, dBP, RP, dRP, Teff, AGaia, dupvar, Ag,
rp1, r1, r10,
g, dg, gchi, gcontrib,
r, dr, rchi, rcontrib,
i, di, ichi, icontrib,
z, dz, zchi, zcontrib,
nstat,
J, dJ, H, dH, K, dK
```

Available photometric bands from this catalog:
- **Gaia**: G, BP, RP (with errors)
- **Pan-STARRS/SkyMapper** (via Refcat2): g, r, i, z (with errors and chi/contrib)
- **2MASS**: J, H, K (with errors)
- **Derived**: Teff, A_Gaia, A_g (extinction estimates)

A catalog-query pathway (e.g., Vizier/TAP) should be left as a **placeholder**
for future use.

### 2.4 Combination with Few Standards

With only 14 ISOPLANE fibers, there may be very few (1–3) standard stars per
exposure. `combine_calibration_vectors` must degrade gracefully to the
single-star case (no combination, pass-through with appropriate warnings and
reduced QC).

### 2.5 Custom `Spectrum1D`

Use a lightweight custom dataclass rather than `specutils.Spectrum1D`. This keeps
the package dependency-light. A `.to_specutils()` method can be added later if
needed.

---

## 3. Calibration Workflow

```
For each standard star on the plate:
  1. Load observed spectrum from RED file (fiber with TYPE='C')
  2. Load broadband photometry from catalog CSV
  3. Estimate Teff from broadband colors → narrow template search range
  4. For each candidate template in the (narrowed) PHOENIX grid:
     a. Convolve to instrument resolution (FWHM from DISPERS header)
     b. Resample to observed wavelength grid
     c. Cross-correlate to measure/correct RV shift
     d. Continuum-normalize both observed and template
     e. Score fit on line features (chi² or Huber metric)
  5. Select best-matching template
  6. Scale template so synthetic photometry matches observed mags (absolute flux anchor)
  7. Compute Cal_star(λ) = F_model_scaled(λ) / C_obs(λ), masking bad regions

Combine per-star vectors:
  8. Robust weighted mean with sigma clipping across all Cal_star(λ)
  9. Optional smoothing (Savitzky-Golay or B-spline)

Apply:
  10. Multiply all science spectra by combined Cal(λ)
  11. Propagate variance
  12. Update FITS header (BUNIT, HISTORY)
```

---

## 4. New Module Layout

```
src/kspecdr/
├── fluxcal/                          # NEW top-level subpackage
│   ├── __init__.py
│   ├── containers.py                 # Dataclass definitions (§5)
│   ├── photometry.py                 # AB mag ↔ flux, filter curves, synthetic phot (§6)
│   ├── templates.py                  # TemplateLibrary, resolution matching (§7)
│   ├── continuum.py                  # Continuum normalization utilities (§8)
│   ├── matching.py                   # Template selection, RV handling (§9)
│   ├── calibration.py                # Per-star cal vector, combination, application (§10)
│   └── masks.py                      # Telluric/bad-region mask I/O (§11)
│
├── data/
│   ├── filters/                      # NEW: filter transmission curves
│   │   ├── sdss_u.dat                #   (or use SVO FPS naming convention)
│   │   ├── sdss_g.dat
│   │   ├── sdss_r.dat
│   │   ├── sdss_i.dat
│   │   ├── sdss_z.dat
│   │   ├── ps1_g.dat                 #   Pan-STARRS bands (Refcat2 uses these)
│   │   ├── ps1_r.dat
│   │   ├── ps1_i.dat
│   │   ├── ps1_z.dat
│   │   ├── gaia_g.dat
│   │   ├── gaia_bp.dat
│   │   ├── gaia_rp.dat
│   │   ├── 2mass_j.dat
│   │   ├── 2mass_h.dat
│   │   └── 2mass_k.dat
│   └── masks/                        # NEW: telluric/bad-region definitions
│       └── telluric_default.dat      #   list of (lam_lo, lam_hi) in Angstrom
```

---

## 5. Data Containers (`fluxcal/containers.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Spectrum1D:
    """Wavelength-calibrated 1D spectrum with uncertainty and mask."""
    wavelength: np.ndarray            # (N,) in Angstrom
    flux: np.ndarray                  # (N,) in counts or flux-density units
    variance: np.ndarray              # (N,)
    mask: np.ndarray                  # (N,) bool — True = good pixel
    meta: Dict = field(default_factory=dict)

    @property
    def ivar(self) -> np.ndarray:
        good = self.mask & (self.variance > 0)
        iv = np.zeros_like(self.variance)
        iv[good] = 1.0 / self.variance[good]
        return iv


@dataclass
class Photometry:
    """Broadband photometric measurements for a single source."""
    filter_names: List[str]           # e.g. ["ps1_g", "ps1_r", "ps1_i"]
    magnitudes: np.ndarray            # (Nbands,)
    mag_errors: np.ndarray            # (Nbands,)
    system: str = "AB"                # "AB" or "Vega"


@dataclass
class FilterCurve:
    """Transmission curve for one photometric filter."""
    name: str
    wavelength: np.ndarray            # (M,) in Angstrom
    transmission: np.ndarray          # (M,) dimensionless, 0–1
    ab_zeropoint_fnu: float = 3.631e-20  # erg/s/cm²/Hz (AB system default)


@dataclass
class StellarTemplate:
    """A single stellar model spectrum with its grid parameters."""
    wavelength: np.ndarray            # (K,) in Angstrom (native resolution)
    flux: np.ndarray                  # (K,) in erg/s/cm²/Å (surface flux)
    teff: float
    logg: float
    feh: float                        # [Fe/H]
    source: str = ""                  # e.g. "BT-Settl", filename


@dataclass
class CalibrationVector:
    """Calibration curve mapping counts → physical flux density."""
    wavelength: np.ndarray            # (N,)
    cal_factor: np.ndarray            # (N,) multiply counts by this → flux
    cal_variance: np.ndarray          # (N,) variance on cal_factor
    mask: np.ndarray                  # (N,) bool — True = reliable
    meta: Dict = field(default_factory=dict)
    # meta keys: star_name, teff, logg, feh, chi2, scale_factor, band_residuals


@dataclass
class FluxCalibrationResult:
    """Complete output of the flux calibration procedure."""
    combined_vector: CalibrationVector
    per_star_vectors: List[CalibrationVector]
    per_star_residuals: List[np.ndarray]      # (N,) fractional residuals
    summary: Dict = field(default_factory=dict)
    # summary keys: n_stars_used, rms_scatter, wavelength_range,
    #               rejected_fraction, per_star_metrics
```

---

## 6. Photometry Utilities (`fluxcal/photometry.py`)

### Functions

```python
def ab_mag_to_flux_density(mag, mag_err=None, unit="f_lambda",
                           wavelength_eff=None):
    """AB mag → flux density (f_lambda or f_nu) with error propagation."""

def flux_density_to_ab_mag(flux, flux_err=None, unit="f_lambda",
                           wavelength_eff=None):
    """Inverse of ab_mag_to_flux_density."""

def load_filter_curve(filter_name: str) -> FilterCurve:
    """Load filter from data/filters/{filter_name}.dat."""

def synthetic_photometry(spectrum, filter_curve) -> float:
    """Compute synthetic AB mag of a spectrum through a filter.

    <f_nu> = ∫ f_nu(ν) T(ν) d(ln ν) / ∫ T(ν) d(ln ν)
    mag_AB = -2.5 log10(<f_nu>) - 48.60
    """

def load_standard_star_catalog(catalog_path: str) -> Table:
    """Load an ATLAS Refcat2–format CSV catalog.

    Returns an astropy Table with all photometric columns.
    """

def photometry_from_catalog_row(row, bands=None) -> Photometry:
    """Extract a Photometry object from one row of the catalog table.

    Parameters
    ----------
    row : astropy.table.Row
        One row from the standard star catalog.
    bands : list of str, optional
        Which bands to extract. Default: ["g", "r", "i", "z", "Gaia",
        "BP", "RP", "J", "H", "K"].
    """

# --- Placeholder for future catalog queries ---
def query_photometry(ra, dec, radius_arcsec=2.0, catalog="refcat2"):
    """Query an external catalog for photometry (NOT YET IMPLEMENTED).

    Raises NotImplementedError with a message pointing to
    load_standard_star_catalog as the current alternative.
    """
```

### Notes

- The Refcat2 g, r, i, z bands are on the **Pan-STARRS** system, not SDSS.
  Filter curves and zero-points must match. The AB zero-point is the same, but
  the bandpasses differ.
- Gaia G, BP, RP are on the Vega-like system natively but have well-known AB
  offsets. Handle in `synthetic_photometry` by checking `FilterCurve` metadata.
- 2MASS J, H, K are Vega-system. Conversion constants to AB should be stored
  with the filter curves or in a lookup table.

---

## 7. Stellar Template Handling (`fluxcal/templates.py`)

### TemplateLibrary class

```python
class TemplateLibrary:
    """Indexed library of PHOENIX/BT-Settl stellar template spectra.

    Parameters
    ----------
    library_dir : str or Path
        Directory containing template FITS files.
    index_file : str or Path, optional
        Pre-built CSV index (filename, Teff, logg, [Fe/H]).
        If None, scans library_dir on first load and caches.
    """

    def __init__(self, library_dir, index_file=None): ...

    def get_template(self, teff, logg, feh) -> StellarTemplate:
        """Nearest-grid-point lookup."""

    def query(self, teff_range=None, logg_range=None,
              feh_range=None) -> List[StellarTemplate]:
        """Return templates within parameter box."""

    @property
    def grid_params(self) -> np.ndarray:
        """(N_templates, 3) array of [Teff, logg, [Fe/H]]."""
```

### Utility functions

```python
def prepare_template(template, target_wavelength,
                     instrument_fwhm_angstrom) -> Spectrum1D:
    """Convolve to instrument resolution, resample to observed grid.

    Uses scipy.ndimage.gaussian_filter1d for convolution and
    scipy.interpolate.interp1d for resampling.
    Returns Spectrum1D with variance=0, mask=True everywhere.
    """

def resample_spectrum(wave_in, flux_in, wave_out) -> np.ndarray:
    """General resampling utility (extracted from wavecal/scrunch.py)."""
```

### PHOENIX/BT-Settl file convention

BT-Settl FITS filenames encode parameters:
`lte{Teff/100}-{logg}{+/-feh}.BT-Settl.spec.fits`

Example: `lte060-4.5-0.0.BT-Settl.spec.fits` → Teff=6000, logg=4.5, [Fe/H]=0.0

The loader should parse these filenames to build the index automatically.

---

## 8. Continuum Normalization (`fluxcal/continuum.py`)

```python
def normalize_continuum(spectrum, method="bspline", order=3, n_knots=20,
                        sigma_clip=3.0, n_iter=3,
                        mask_regions=None):
    """Fit and divide out the pseudo-continuum.

    Parameters
    ----------
    spectrum : Spectrum1D
    method : str
        "bspline", "polynomial", or "running_median"
    order : int
        Polynomial order or spline degree.
    n_knots : int
        Number of interior knots for B-spline.
    sigma_clip : float
        Lower rejection threshold (masks absorption lines).
    n_iter : int
        Number of iterative rejection passes.
    mask_regions : list of (float, float), optional
        Wavelength ranges to exclude (e.g. tellurics).

    Returns
    -------
    normalized : Spectrum1D
    continuum : np.ndarray
    """
```

### Implementation notes

- Reuse B-spline smoothing logic from `reduce_fflat.py:bs_smooth_redflat`.
- Normalize wavelength to [0, 1] before fitting (pattern from
  `landmarks.py:robust_polyfit`).
- Iterative lower-sigma clipping to avoid absorption lines biasing the
  continuum downward.
- Handle zero/negative flux by masking, not by dividing.

---

## 9. Template Selection (`fluxcal/matching.py`)

```python
def select_best_template(observed, photometry, library,
                         instrument_fwhm_angstrom,
                         rv_guess=0.0, mask_regions=None,
                         teff_range=None, metric="chi2"):
    """Find best-matching template for a standard star.

    Returns
    -------
    best_template : StellarTemplate
    best_rv : float           # km/s
    fit_stats : dict          # chi2, ndof, runner_up_chi2, ...
    """

def estimate_teff_from_colors(photometry, filter_curves=None):
    """Rough Teff from broadband color (e.g. BP-RP or g-r).

    Returns
    -------
    teff_estimate : float
    teff_range : (float, float)
    """
```

### Implementation notes

- Pre-filter templates using `estimate_teff_from_colors` to avoid exhaustive
  grid search.
- Cross-correlate to measure RV before scoring. Adapt
  `wavecal/crosscorr.py:crosscorr_analysis` or use a simpler version.
- Score on **continuum-normalized** spectra so the metric reflects line
  features, not SED shape.
- Support both chi² and Huber-loss metrics (Huber is more robust to outlying
  pixels from sky-subtraction residuals).

---

## 10. Calibration Vector Computation (`fluxcal/calibration.py`)

### Core functions

```python
def scale_template_to_photometry(template, photometry, filter_curves):
    """Scale template so synthetic photometry matches observed mags.

    Returns
    -------
    scale_factor : float
    scale_error : float
    band_residuals : dict     # {filter_name: synth_mag - obs_mag}
    """

def compute_calibration_vector_for_star(observed, photometry, library,
                                        filter_curves,
                                        instrument_fwhm_angstrom,
                                        mask_regions=None):
    """Full per-star calibration pipeline.

    Orchestrates:
      1. select_best_template
      2. prepare_template
      3. scale_template_to_photometry
      4. Cal(λ) = scaled_template / observed
      5. Propagate variance, apply mask

    Returns
    -------
    CalibrationVector
    """

def combine_calibration_vectors(vectors, method="weighted_mean",
                                sigma_clip=3.0, smooth=False,
                                smooth_window=51):
    """Combine per-star vectors into a single calibration curve.

    Gracefully handles N=1 (single star, no combination).

    Returns
    -------
    FluxCalibrationResult
    """

def apply_flux_calibration(spectra, variance, calibration,
                           update_header=None):
    """Apply calibration to all fibers.

    flux_cal = counts × Cal(λ)
    var_cal  = var_obs × Cal² + counts² × var_Cal

    Returns
    -------
    cal_spectra : np.ndarray   # (NFIB, NSPEC)
    cal_variance : np.ndarray  # (NFIB, NSPEC)
    header_updates : dict      # BUNIT, HISTORY entries
    """
```

---

## 11. Masks (`fluxcal/masks.py`)

```python
def load_mask_regions(mask_name="telluric_default"):
    """Load wavelength regions from data/masks/{mask_name}.dat.

    File format: two-column ASCII, lam_lo lam_hi (Angstrom).

    Returns
    -------
    list of (float, float)
    """

def apply_mask_regions(spectrum, regions):
    """Set mask=False for pixels in any region. Returns new Spectrum1D."""
```

### Default telluric mask regions

| Region | λ range (Å) | Source |
|---|---|---|
| O₂ B-band | 6860–6960 | telluric |
| O₂ A-band | 7590–7700 | telluric |
| H₂O | 7150–7340 | telluric |
| H₂O | 8100–8350 | telluric |
| H₂O | 8925–9200 | telluric |

---

## 12. Integration with Existing Pipeline

### Entry point: `reduce_object.py`

Replace lines 232–241 (the `CALIBFLUX` block):

```python
if calflx:
    from .fluxcal.calibration import (
        compute_calibration_vector_for_star,
        combine_calibration_vectors,
        apply_flux_calibration,
    )
    from .fluxcal.photometry import (
        load_standard_star_catalog,
        photometry_from_catalog_row,
    )
    # 1. Identify standard-star fibers (TYPE == 'C') from FIBRES table
    # 2. Load catalog, match by fiber NAME or position
    # 3. For each standard: compute_calibration_vector_for_star(...)
    # 4. combine_calibration_vectors(...)
    # 5. apply_flux_calibration(...) to all fibers
    # 6. Write back to RED file, update header
```

### Changes to existing code

| File | Change |
|---|---|
| `constants.py` | Add `FIBER_TYPE_CALIBRATION = 'C'` |
| `inst/isoplane.py` | Update `target_class → TYPE` mapping to include `8 → 'C'` |
| `io/image.py` : `read_fiber_types` | Recognize `'C'` as a valid type |
| `extract/make_ex.py` | Ensure `'C'` fibers are extracted (not skipped) |
| `__init__.py` | Add `fluxcal` to imports and `__all__` |

---

## 13. New Dependencies

| Package | Purpose | Status |
|---|---|---|
| `scipy>=1.7.0` | Convolution, spline fitting, interpolation | Already used; **add to `pyproject.toml`** |
| `astropy.stats` | `sigma_clip`, `sigma_clipped_stats` | Available (astropy already a dep) |

No new external dependencies required.

---

## 14. Implementation Priority

| Phase | Modules | Notes |
|---|---|---|
| **P0** | `containers.py`, `photometry.py`, `masks.py` | Foundational types and utilities |
| **P1** | `templates.py`, `continuum.py` | Template loading, resolution matching, continuum fitting |
| **P1** | `matching.py` | Template selection and RV handling |
| **P2** | `calibration.py` | Orchestration: per-star vectors, combination, application |
| **P2** | Integration into `reduce_object.py` | Wire into existing pipeline, update fiber type handling |
| **P3** | QC notebook helpers | Plotting utilities, summary tables (notebook-level, not in pipeline) |

---

## 15. Data Files Needed

| File | Source | Notes |
|---|---|---|
| PHOENIX/BT-Settl grid | [PHOENIX website](https://phoenix.astro.physik.uni-goettingen.de/) | Download F-star parameter range first: Teff 5500–7500 K, logg 3.0–5.0, [Fe/H] -1.0 to +0.5 |
| Filter curves | [SVO FPS](http://svo2.cab.inta-csic.es/theory/fps/) | Pan-STARRS g/r/i/z, Gaia G/BP/RP, 2MASS J/H/K |
| Telluric mask | Define manually | Table of (λ_lo, λ_hi) pairs |
| Standard star catalog | ATLAS Refcat2 query | Already have example: `standard_star_atlas_refcat2.csv` |
