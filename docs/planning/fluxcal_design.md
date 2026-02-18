# Flux Calibration Design Plan for `kspecdr`

> **Status**: P0 + P1 + P2 implemented — calibration pipeline complete.
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
| Spectrum container | ✓ `fluxcal/containers.py` — `Spectrum1D`, `Photometry`, `FilterCurve`, `StellarTemplate`, `CalibrationVector`, `FluxCalibrationResult` |
| Wavelength calibration | Done — arc line matching, polynomial fitting, scrunching |
| Extraction with variance propagation | Done — `sum_extract` |
| Fiber metadata | Done — FIBRES BinTableHDU with TYPE/NAME columns |
| Flux calibration | **Stub only** — `reduce_object.py:232-241` raises `NotImplementedError` |
| Photometry / flux-density utilities | ✓ `fluxcal/photometry.py` — AB mag conversions, filter loading, synthetic photometry, Refcat2 catalog I/O |
| Filter curves | ✓ `data/filters/` — PS1 g/r/i/z/y, Gaia G/BP/RP, 2MASS J/H/K |
| Telluric / region masks | ✓ `fluxcal/masks.py` + `data/masks/telluric_default.dat` |
| Stellar template library | ✓ `fluxcal/templates.py` — `TemplateLibrary` (520 BOSZ models, auto-index), `prepare_template`, `resample_spectrum` |
| Continuum normalization | ✓ `fluxcal/continuum.py` — B-spline / polynomial / running-median with iterative σ-clipping; model-continuum passthrough |
| Template matching | ✓ `fluxcal/matching.py` — `select_best_template`, FFT cross-correlation RV (log-λ), χ² / Huber scoring |
| Calibration vector computation | ✓ `fluxcal/calibration.py` — per-star vectors, robust combination, application with variance propagation |
| Pipeline integration | ✓ `reduce_object.py:_apply_fluxcal` — reads TYPE='C' fibers, catalog, runs full pipeline |
| Fiber type constant | ✓ `constants.py:FIBER_TYPE_CALIBRATION = 'C'` |

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
     a. Convolve to instrument resolution (SPECFWHM / FWHMPCn headers from wavecal)
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
│   ├── __init__.py                   # ✓ created — exposes full P0 public API
│   ├── download_bosz.py              # ✓ created — BOSZ 2024 subgrid downloader (§16)
│   ├── containers.py                 # ✓ implemented (P0) — dataclass definitions (§5)
│   ├── photometry.py                 # ✓ implemented (P0) — AB mag utils, filter loading, synthetic phot (§6)
│   ├── masks.py                      # ✓ implemented (P0) — telluric/bad-region mask I/O (§11)
│   ├── templates.py                  # ✓ implemented (P1) — TemplateLibrary, prepare_template (§7)
│   ├── continuum.py                  # ✓ implemented (P1) — normalize_continuum, fit_continuum (§8)
│   ├── matching.py                   # ✓ implemented (P1) — select_best_template, cross_correlate_rv (§9)
│   └── calibration.py                # ✓ implemented (P2) — per-star cal vector, combination, application (§10)
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
│   ├── masks/                        # ✓ populated
│   │   └── telluric_default.dat      #   ✓ 5 regions: O₂ B, H₂O, O₂ A, H₂O×2
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

## 5. Data Containers (`fluxcal/containers.py`) ✓ implemented

```python
@dataclass
class Spectrum1D:
    wavelength: np.ndarray   # (N,) Angstrom
    flux: np.ndarray         # (N,) counts or flux-density
    variance: np.ndarray     # (N,)
    mask: np.ndarray         # (N,) bool — True = good pixel
    meta: Dict = field(default_factory=dict)
    # meta keys: fiber_id, fiber_name, exptime, bunit,
    #            continuum (ndarray), rv_kms

    # properties: .ivar, .n_pixels, .wave_range, .n_good


@dataclass
class Photometry:
    filter_names: List[str]  # e.g. ["ps1_g", "gaia_g"]
    magnitudes: np.ndarray   # (Nbands,) — all AB; nan = unavailable band
    mag_errors: np.ndarray   # (Nbands,)
    meta: Dict = field(default_factory=dict)
    # meta keys: ra, dec, objid, teff_catalog, a_gaia, a_g

    # methods: .get_band(name) → (mag, err), .valid_bands() → List[str]


@dataclass
class FilterCurve:
    name: str
    wavelength: np.ndarray    # (M,) Angstrom (converted from µm on load)
    transmission: np.ndarray  # (M,) 0–1
    system: str = "AB"        # "AB" or "Vega"
    vega_to_ab: float = 0.0   # m_AB = m_Vega + vega_to_ab

    # properties: .wave_eff (Å), .wave_range (lo, hi at 1% of peak)


@dataclass
class StellarTemplate:
    wavelength: np.ndarray    # (K,) Angstrom (BOSZ r10000 log-λ grid)
    flux: np.ndarray          # (K,) erg/s/cm²/Å  (= 4π × H, col 3)
    continuum: np.ndarray     # (K,) theoretical continuum (= 4π × C, col 4)
    teff: float
    logg: float
    feh: float                # [M/H]
    alpha_m: float = 0.0      # [α/M]
    carbon_m: float = 0.0     # [C/M]
    vmicro: float = 1.0       # km/s
    atmos_model: str = ""     # "ap" (ATLAS9) or "mp" (MARCS)
    source: str = "BOSZ2024"

    # methods: .to_spectrum1d() → Spectrum1D, .label → str


@dataclass
class CalibrationVector:
    wavelength: np.ndarray    # (N,) Angstrom
    cal_factor: np.ndarray    # (N,) erg/s/cm²/Å per count
    cal_variance: np.ndarray  # (N,)
    mask: np.ndarray          # (N,) bool — True = reliable
    meta: Dict = field(default_factory=dict)
    # meta keys: star_name, fiber_id, teff, logg, feh, alpha_m,
    #            scale_factor, band_residuals, chi2, ndof, rv_kms

    # properties: .cal_error, .n_good


@dataclass
class FluxCalibrationResult:
    combined_vector: CalibrationVector
    per_star_vectors: List[CalibrationVector]
    per_star_residuals: List[np.ndarray]   # fractional residuals per star
    summary: Dict = field(default_factory=dict)
    # summary keys: n_stars_used, n_stars_rejected, rms_scatter,
    #               wavelength_range, per_star_metrics

    # properties: .n_stars
```

---

## 6. Photometry Utilities (`fluxcal/photometry.py`) ✓ implemented

### Filter registry

`FILTER_INFO` dict maps each filter name to `{"system", "vega_to_ab"}`.
All 11 filters are registered (PS1 g/r/i/z/y, Gaia G/BP/RP, 2MASS J/H/K).

Vega-to-AB offsets used:

| Filter | vega_to_ab | Reference |
|---|---|---|
| Gaia G | +0.1069 | Maíz Apellániz & Weiler 2018 |
| Gaia BP | +0.0676 | Evans+ 2018 |
| Gaia RP | −0.3958 | Evans+ 2018 |
| 2MASS J | +0.91 | Blanton & Roweis 2007 |
| 2MASS H | +1.39 | Blanton & Roweis 2007 |
| 2MASS K | +1.85 | Blanton & Roweis 2007 |

### Functions

```python
# --- Unit conversions ---
def ab_mag_to_fnu(mag, mag_err=None) -> (fnu, fnu_err):
    """AB mag → f_ν (erg/s/cm²/Hz) with error propagation."""

def fnu_to_ab_mag(fnu, fnu_err=None) -> (mag, mag_err):
    """f_ν (erg/s/cm²/Hz) → AB mag."""

def ab_mag_to_flam(mag, wave_eff_ang, mag_err=None) -> (flam, flam_err):
    """AB mag → f_λ (erg/s/cm²/Å) at effective wavelength.
    f_λ = f_ν × c / λ²
    """

# --- Filter I/O ---
def load_filter_curve(filter_name: str) -> FilterCurve:
    """Load from data/filters/{filter_name}.dat; convert µm → Å; cache."""

def load_filter_curves(filter_names) -> Dict[str, FilterCurve]:
    """Load multiple filters at once."""

# --- Synthetic photometry ---
def synthetic_photometry(spectrum: Spectrum1D,
                          filter_curve: FilterCurve) -> float:
    """Photon-count weighted synthetic AB mag.

    <f_ν> = ∫ f_ν T d(ln λ) / ∫ T d(ln λ)
    m_AB  = -2.5 log10(<f_ν>) - 48.60
    Returns nan if filter not covered or flux non-physical.
    Verified: exact recovery for flat f_ν spectra.
    """

def synthetic_color(spectrum, filter1, filter2) -> float:
    """Synthetic color m_filter1 − m_filter2 (AB mags)."""

# --- Catalog I/O ---
def load_standard_star_catalog(catalog_path) -> astropy.table.Table:
    """Load ATLAS Refcat2-format CSV; returns full astropy Table."""

def photometry_from_catalog_row(row, bands=DEFAULT_BANDS) -> Photometry:
    """Extract Photometry from one Refcat2 row.

    - Applies Vega→AB corrections automatically.
    - Sentinel values (mag ≥ 99 or err ≥ 9.9) stored as nan.
    - Populates meta with ra, dec, objid, teff_catalog, a_gaia, a_g.
    """

# --- Color → Teff ---
def estimate_teff_from_color(photometry, method="bp_rp") -> (teff, teff_range):
    """Rough photometric Teff estimate.

    method="bp_rp" : Gaia BP−RP (recommended; Casagrande+ 2010 polynomial)
    method="g_r"   : PS1 g−r (fallback)
    Returns (teff_K, (teff_lo, teff_hi)) with ±600 K search window.
    """

# --- Placeholder ---
def query_photometry(ra, dec, radius_arcsec=2.0, catalog="refcat2"):
    """NOT YET IMPLEMENTED. Raises NotImplementedError."""
```

---

## 7. Stellar Template Handling (`fluxcal/templates.py`) ✓ implemented

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

Each ``_resam.txt.gz`` file is whitespace-separated ASCII with **no header**
and **2 columns** (150,000 rows).  The wavelength grid is shared and stored
separately in ``bosz2024_wave_r10000.txt``.

| Col | Content | Unit |
|---|---|---|
| 0 | H (Eddington flux) | erg/s/cm²/Å/steradian |
| 1 | C (continuum) | erg/s/cm²/Å/steradian |

Convert to surface flux: ``F = 4π × H``.  Normalised flux: ``H / C``.

The shared wavelength grid has 150,000 points spanning 500–320,000 Å.
``TemplateLibrary`` trims to the optical range (default 3000–11,000 Å →
30,163 pixels) on load:

```python
import gzip
import numpy as np

wave = np.loadtxt("bosz2024_wave_r10000.txt")  # shared grid
with gzip.open(filepath, "rt") as f:
    data = np.loadtxt(f)   # (150000, 2)
flux = 4 * np.pi * data[:, 0]   # surface flux
cont = 4 * np.pi * data[:, 1]   # continuum
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
                     instrument_fwhm_angstrom,
                     fwhm_poly_coeffs=None) -> Spectrum1D:
    """Convolve to instrument resolution, resample to observed grid.

    If fwhm_poly_coeffs is provided, applies wavelength-dependent
    convolution using FWHM(lambda) polynomial from wavecal (FWHMPCn
    header keywords).  Otherwise uses a constant instrument_fwhm_angstrom
    (SPECFWHM header keyword).

    Uses scipy.ndimage.gaussian_filter1d for convolution and
    scipy.interpolate.interp1d for resampling.
    Returns Spectrum1D with variance=0, mask=True everywhere.
    The BOSZ continuum column is carried through as meta['continuum'].
    """

def resample_spectrum(wave_in, flux_in, wave_out) -> np.ndarray:
    """General resampling utility (extracted from wavecal/scrunch.py)."""
```

---

## 8. Continuum Normalization (`fluxcal/continuum.py`) ✓ implemented

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

## 9. Template Selection (`fluxcal/matching.py`) ✓ implemented

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

## 10. Calibration Vector Computation (`fluxcal/calibration.py`) ✓ implemented

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

## 11. Masks (`fluxcal/masks.py`) ✓ implemented

### Default telluric mask (`data/masks/telluric_default.dat`)

| Region | λ range (Å) |
|---|---|
| O₂ B-band | 6860–6960 |
| H₂O | 7150–7340 |
| O₂ A-band | 7590–7700 |
| H₂O | 8100–8360 |
| H₂O | 8920–9200 |

### Functions

```python
def load_mask_regions(mask_name="telluric_default") -> List[tuple]:
    """Load (lam_lo, lam_hi) pairs from data/masks/{mask_name}.dat.
    Cached after first load. Comment lines (#) ignored.
    """

def load_mask_array(wavelength, mask_name="telluric_default") -> np.ndarray:
    """Boolean array for a given wavelength axis (True = good pixel)."""

def apply_mask_regions(spectrum: Spectrum1D,
                        regions: List[tuple]) -> Spectrum1D:
    """Return new Spectrum1D with mask=False inside each region."""

def apply_named_mask(spectrum: Spectrum1D,
                      mask_name="telluric_default") -> Spectrum1D:
    """Convenience: load mask by name and apply in one call."""

def mask_coverage_fraction(wavelength, mask_name="telluric_default") -> float:
    """Fraction of pixels inside the mask; large values suggest unit error."""

def combine_regions(*region_lists) -> List[tuple]:
    """Merge multiple region lists, sorted by start wavelength."""
```

---

## 12. Integration with Existing Pipeline ✓ implemented

### Entry point: `reduce_object.py`

The `CALIBFLUX` block now calls `_apply_fluxcal(red_filename, args)` which:

1. Opens the RED file, reads spectra/variance/wavelength/fiber table.
2. Identifies standard-star fibers (``TYPE='C'``).
3. Loads the standard-star catalog (``CALIBFLUX_CATALOG``).
4. Loads the BOSZ template library and filter curves.
5. For each standard: ``compute_calibration_vector_for_star(...)``
6. ``combine_calibration_vectors(...)`` with sigma-clipping.
7. ``apply_flux_calibration(...)`` to all fibers with variance propagation.
8. Writes calibrated data and header keywords (``FLUXCAL``, ``BUNIT``,
   ``FCALNSTR``, ``FCALRMS``, ``HISTORY``) back to the RED file.

### Relevant ``args`` keys

| Key | Type | Description |
|---|---|---|
| ``CALIBFLUX`` | bool | Enable flux calibration |
| ``CALIBFLUX_CATALOG`` | str | Path to standard-star CSV catalog |
| ``CALIBFLUX_FWHM`` | float | Instrument FWHM in Å (default: ``SPECFWHM`` header) |
| ``CALIBFLUX_METRIC`` | str | ``"chi2"`` or ``"huber"`` (default: ``"chi2"``) |
| ``CALIBFLUX_SMOOTH`` | bool | Smooth combined cal vector (default: False) |

### Changes to existing code

| File | Change | Status |
|---|---|---|
| `constants.py` | Added `FIBER_TYPE_CALIBRATION = 'C'` and all fiber type codes | ✓ |
| `reduce_object.py` | Replaced `NotImplementedError` with `_apply_fluxcal` call | ✓ |
| `inst/isoplane.py` | Update `target_class → TYPE` mapping to include `8 → 'C'` | Pending (assign-file parser) |
| `io/image.py` : `read_fiber_types` | Already accepts any single-char type from FIBRES table | No change needed |
| `extract/make_ex.py` | Ensure `'C'` fibers are extracted (not skipped) | ✓ No change needed — zeroing loop only targets `'F'`, `'N'`, `'U'` |

---

## 13. New Dependencies

| Package | Purpose | Status |
|---|---|---|
| `scipy>=1.7.0` | Convolution, spline fitting, interpolation | Already used; **add to `pyproject.toml`** |
| `astropy.stats` | `sigma_clip`, `sigma_clipped_stats` | Available (astropy already a dep) |

No new external dependencies required.

---

## 14. Implementation Priority

| Phase | Modules | Status |
|---|---|---|
| **P0** | `containers.py`, `photometry.py`, `masks.py` | ✓ **Done** |
| **P1** | `templates.py` | ✓ **Done** — TemplateLibrary (auto-index, lazy load, wave trim), prepare_template |
| **P1** | `continuum.py` | ✓ **Done** — B-spline / polynomial / running-median + model-continuum passthrough |
| **P1** | `matching.py` | ✓ **Done** — select_best_template, FFT cross-correlation RV, χ² / Huber scoring |
| **P2** | `calibration.py` | ✓ **Done** — scale_template_to_photometry, per-star vectors, weighted-mean/median combination, apply with variance propagation |
| **P2** | Integration into `reduce_object.py` | ✓ **Done** — `_apply_fluxcal` reads TYPE='C' fibers, catalog, runs full pipeline; `FIBER_TYPE_CALIBRATION` added to constants |
| **P3** | QC notebook helpers | Plotting utilities, summary tables (notebook-level) |

---

## 15. Data Files Status

| File / Set | Source | Status |
|---|---|---|
| BOSZ 2024 template subgrid (520 files) | [MAST HLSP BOSZ](https://archive.stsci.edu/hlsp/bosz) | ✓ Downloaded — `data/templates/bosz2024/` |
| BOSZ shared wavelength grid | MAST HLSP BOSZ | ✓ Downloaded — `bosz2024_wave_r10000.txt` |
| Pan-STARRS1 filter curves (g/r/i/z/y) | Tonry et al. 2012 (via SVO FPS) | ✓ `data/filters/ps1_*.dat` |
| Gaia DR2 passbands (G/BP/RP) | Evans et al. 2018 (via ESA) | ✓ `data/filters/gaia_*.dat` |
| 2MASS filter curves (J/H/Ks) | Cohen et al. 2003 | ✓ `data/filters/2mass_*.dat` |
| Telluric mask | Defined manually (5 regions) | ✓ `data/masks/telluric_default.dat` |
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
