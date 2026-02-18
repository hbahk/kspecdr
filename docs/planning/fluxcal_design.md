# Flux Calibration Design Plan for `kspecdr`

> **Status**: Implementation plan — not yet implemented.
> **Last updated**: 2026-02-18

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

Use the **BOSZ 2024** synthetic stellar spectral library (Mészáros et al. 2024,
MAST HLSP), based on ATLAS9 and MARCS model atmospheres. BOSZ is preferred over
PHOENIX/BT-Settl for this use case because:

- Calibrated against HST/STIS CALSPEC flux standards — directly relevant for
  spectrophotometric calibration.
- Each file contains a **continuum column** (theoretical continuum), usable as
  a first-pass normalization without fitting a spline.
- Pre-computed at multiple resolutions (R = 500–50,000) — download at
  R = 10,000, convolve down to the instrument LSF, avoiding full-resolution
  computation.
- Filenames encode all parameters — trivial to build an index without reading
  file contents.
- Updated Sept 2025 (OH+ and H-line bug fixes); actively maintained.

**Downloaded subgrid** (`data/templates/bosz2024/`, gitignored):

| Parameter | Values |
|---|---|
| Resolution | R = 10,000 (`r10000`) |
| Teff | 5000–8000 K, step 250 K (13 values) |
| log(g) | 3.5, 4.0, 4.5, 5.0 |
| [M/H] | −1.00 to +0.50, step 0.25 (7 values) |
| [α/M] | 0.00 for all; +0.25 additionally for [M/H] ≤ −0.50 |
| [C/M] | 0.00 |
| vmicro | 1 km/s |
| atmos | `mp` (MARCS plane-parallel) for Teff 5000–7250 K |
|        | `ap` (ATLAS9 plane-parallel) for Teff 7500–8000 K |
| Total | **520 model files** + 1 shared wavelength grid |

**Re-download**: `python -m kspecdr.fluxcal.download_bosz` (see §16).

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
  4. For each candidate template in the (narrowed) BOSZ subgrid:
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
│   ├── __init__.py                   # ✓ created
│   ├── download_bosz.py              # ✓ created — BOSZ 2024 subgrid downloader (§16)
│   ├── containers.py                 # Dataclass definitions (§5)
│   ├── photometry.py                 # AB mag ↔ flux, filter curves, synthetic phot (§6)
│   ├── templates.py                  # TemplateLibrary (BOSZ 2024), resolution matching (§7)
│   ├── continuum.py                  # Continuum normalization utilities (§8)
│   ├── matching.py                   # Template selection, RV handling (§9)
│   ├── calibration.py                # Per-star cal vector, combination, application (§10)
│   └── masks.py                      # Telluric/bad-region mask I/O (§11)
│
├── data/
│   ├── filters/                      # ✓ populated — filter transmission curves
│   │   ├── ps1_g.dat                 #   Pan-STARRS1 g (Tonry+ 2012)
│   │   ├── ps1_r.dat
│   │   ├── ps1_i.dat
│   │   ├── ps1_z.dat
│   │   ├── ps1_y.dat
│   │   ├── gaia_g.dat                #   Gaia DR2 G/BP/RP (Evans+ 2018)
│   │   ├── gaia_bp.dat
│   │   ├── gaia_rp.dat
│   │   ├── 2mass_j.dat               #   2MASS J/H/Ks
│   │   ├── 2mass_h.dat
│   │   └── 2mass_k.dat
│   ├── masks/                        # NEW: telluric/bad-region definitions
│   │   └── telluric_default.dat      #   list of (lam_lo, lam_hi) in Angstrom
│   └── templates/                    # gitignored — large downloaded grids
│       └── bosz2024/                 # ✓ downloaded (520 files + wave grid)
│           ├── bosz2024_wave_r10000.txt   # shared wavelength grid (Å)
│           └── r10000/
│               ├── m-1.00/
│               ├── m-0.75/
│               ├── ...
│               └── m+0.50/
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
    """A single BOSZ 2024 stellar model spectrum with its grid parameters."""
    wavelength: np.ndarray            # (K,) in Angstrom (on the r10000 log-λ grid)
    flux: np.ndarray                  # (K,) in erg/s/cm²/Å  (= 4π × H from file)
    continuum: np.ndarray             # (K,) theoretical continuum — provided by BOSZ
    teff: float
    logg: float
    feh: float                        # [M/H]
    alpha_m: float = 0.0              # [α/M]
    carbon_m: float = 0.0             # [C/M]
    vmicro: float = 1.0               # km/s
    atmos_model: str = ""             # "ap" (ATLAS9) or "mp"/"ms" (MARCS)
    source: str = "BOSZ2024"          # filename for provenance


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
    """Indexed library of BOSZ 2024 stellar template spectra.

    Parameters
    ----------
    library_dir : str or Path
        Root directory of the template subgrid
        (e.g. data/templates/bosz2024/).
        Must contain bosz2024_wave_r10000.txt and r10000/<feh>/*.txt.gz files.
    resolution : str
        Subdirectory / resolution tag to use. Default: "r10000".
    index_file : str or Path, optional
        Pre-built CSV index (filepath, teff, logg, feh, alpha_m, atmos).
        If None, scans library_dir on first load and caches to
        library_dir/index_r10000.csv.
    """

    def __init__(self, library_dir, resolution="r10000",
                 index_file=None): ...

    def get_template(self, teff, logg, feh,
                     alpha_m=None) -> StellarTemplate:
        """Nearest-grid-point lookup (5-D: Teff, logg, [M/H], [α/M], atmos).

        alpha_m defaults to 0.00 (or 0.25 for [M/H] ≤ −0.50 if 0.00
        is not present in the subgrid).
        """

    def query(self, teff_range=None, logg_range=None,
              feh_range=None) -> List[StellarTemplate]:
        """Return templates within parameter box (metadata only, lazy load)."""

    @property
    def grid_params(self) -> np.ndarray:
        """(N_templates, 4) array of [Teff, logg, [M/H], [α/M]]."""
```

### BOSZ file format

Each `*.txt.gz` file is whitespace-separated ASCII with **no header**. The
columns are:

| Col | Content | Unit |
|---|---|---|
| 1 | Wavelength | Å |
| 2 | Normalized flux (F/C) | dimensionless |
| 3 | Flux density (H) | erg/s/cm²/Å/steradian |
| 4 | Continuum (C) | erg/s/cm²/Å/steradian |

Use columns 1, 3, and 4 (wavelength, flux, continuum). Convert H → F via
`F = 4π × H` to get surface flux in erg/s/cm²/Å.

The shared wavelength grid is in `bosz2024_wave_r10000.txt` (single column, Å).
Files are read with `numpy.loadtxt` on the decompressed stream:

```python
import gzip
import numpy as np

with gzip.open(filepath, "rt") as f:
    data = np.loadtxt(f, usecols=(0, 2, 3))   # wave, H, C
wave  = data[:, 0]
flux  = 4 * np.pi * data[:, 1]
cont  = 4 * np.pi * data[:, 2]
```

### BOSZ filename convention

```
bosz2024_{atmos}_t{Teff}_g+{logg}_m{feh:+.2f}_a{alpha_m:+.2f}_c{carbon_m:+.2f}_v{vmicro}_r10000_resam.txt.gz
```

Example: `bosz2024_mp_t6000_g+4.0_m+0.00_a+0.00_c+0.00_v1_r10000_resam.txt.gz`

The index builder should parse these fields from the filename with a regex to
avoid reading file contents.

### Utility functions

```python
def parse_bosz_filename(filename: str) -> dict:
    """Parse BOSZ filename → {atmos, teff, logg, feh, alpha_m, carbon_m, vmicro}."""

def prepare_template(template, target_wavelength,
                     instrument_fwhm_angstrom) -> Spectrum1D:
    """Convolve to instrument resolution, resample to observed grid.

    Uses scipy.ndimage.gaussian_filter1d for convolution and
    scipy.interpolate.interp1d for resampling.
    Returns Spectrum1D with variance=0, mask=True everywhere.
    The BOSZ continuum column is carried through as meta['continuum'].
    """

def resample_spectrum(wave_in, flux_in, wave_out) -> np.ndarray:
    """General resampling utility (extracted from wavecal/scrunch.py)."""
```

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

## 15. Data Files Status

| File / Set | Source | Status |
|---|---|---|
| BOSZ 2024 template subgrid (520 files) | [MAST HLSP BOSZ](https://archive.stsci.edu/hlsp/bosz) | ✓ Downloaded — `data/templates/bosz2024/` |
| BOSZ shared wavelength grid | MAST HLSP BOSZ | ✓ Downloaded — `bosz2024_wave_r10000.txt` |
| Pan-STARRS1 filter curves (g/r/i/z/y) | Tonry et al. 2012 (via SVO FPS) | ✓ `data/filters/ps1_*.dat` |
| Gaia DR2 passbands (G/BP/RP) | Evans et al. 2018 (via ESA) | ✓ `data/filters/gaia_*.dat` |
| 2MASS filter curves (J/H/Ks) | Cohen et al. 2003 | ✓ `data/filters/2mass_*.dat` |
| Telluric mask | Define manually | Pending — `data/masks/telluric_default.dat` |
| Standard star catalog | ATLAS Refcat2 | ✓ Example: `resources/comm/20260129/calib/standard_star_atlas_refcat2.csv` |

---

## 16. Template Download Utility (`fluxcal/download_bosz.py`)

The subgrid download is implemented as a runnable module:

```
python -m kspecdr.fluxcal.download_bosz            # full download
python -m kspecdr.fluxcal.download_bosz --dry-run  # preview URLs only
python -m kspecdr.fluxcal.download_bosz --force    # re-download all files
```

Run from the repository root with the package installed (or with
`PYTHONPATH=src` set).

**Source**: `src/kspecdr/fluxcal/download_bosz.py`

Key design points:
- `ThreadPoolExecutor(max_workers=8)` — 8 concurrent downloads from MAST.
- 3-retry loop with 2 s back-off on transient HTTP errors.
- Skips existing non-empty files by default (`skip_existing=True`).
- Logs per-file status and a final summary (ok / skipped / not_found / error).
- `not_found` entries are expected for (Teff, logg) pairs outside the BOSZ
  physical grid boundaries; they do not indicate a bug.
