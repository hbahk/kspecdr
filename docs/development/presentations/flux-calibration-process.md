# KSPECDR Flux Calibration Pipeline

## End-to-End Process Overview

- Project: `kspecdr`
- Goal: convert extracted counts to physical flux density (`erg/cm2/s/A`)
- Entry point: `reduce_object(args)`
- Flux calibration stage: `_apply_fluxcal(red_filename, args)`

---

# Where Flux Calibration Happens

## Reduction Order in `reduce_object`

1. Raw -> IM (`make_im`)
2. IM -> EX (`make_ex`)
3. EX -> RED copy
4. Flat-field, scrunch, throughput correction, sky subtraction
5. **Flux calibration (`CALIBFLUX=True`)**
6. Final metadata + reduced status

Key point: flux calibration runs after wavelength calibration and sky subtraction.

---

# Inputs Required for Flux Calibration

## Science Data + Metadata

- RED spectra + variance + wavelength solution (`WAVELA`)
- Fiber table with fiber types and names (`FIBRES`)
- Standard fibers identified by `TYPE='C'`

## External Resources

- Standard-star photometry CSV (`CALIBFLUX_CATALOG`)
- BOSZ template library (`data/templates/bosz2024/`)
- Filter curves (`data/filters/*.dat`)
- Mask regions (`data/masks/telluric_default.dat`)

---

# Standard-Star Selection

## Fiber and Catalog Mapping

- Pipeline scans fibers and keeps only calibration fibers (`TYPE='C'`)
- For each selected fiber:
  - Build observed `Spectrum1D` (wavelength, flux, variance, mask)
  - Read star name from fiber table (if present)
  - Load photometry row from catalog

Current implementation detail:
- Catalog matching is by row index order (not RA/Dec positional matching yet).

---

# Template Matching (Per Standard Star)

## Best-Fit Stellar Model Selection

1. Estimate Teff search range from broadband colors
2. Query BOSZ candidates in parameter box (Teff/logg/[M/H])
3. For each candidate:
   - Convolve to instrument resolution (`CALIBFLUX_FWHM` or `SPECFWHM`)
   - Resample to observed wavelength grid
   - Measure RV by log-lambda cross-correlation
   - Continuum-normalize observed and template spectra
   - Score fit (`chi2` or `huber`)
4. Select best template + RV

---

# Photometric Scaling

## Set the Absolute Flux Level

- Compute synthetic photometry of the best template through loaded filters
- Compare to observed catalog magnitudes (internally converted to AB)
- Derive a weighted scale factor from magnitude offsets

Formula used:
- `scale = 10^(-0.4 * mean_delta_mag)`

Output:
- Scaled template spectrum in physical flux units
- Scale uncertainty from photometric errors

---

# Per-Star Calibration Vector

## Build `Cal_star(lambda)`

- Core relation:
  - `Cal_star(lambda) = F_model_scaled(lambda) / C_obs(lambda)`
- Mask invalid pixels:
  - non-finite values
  - non-positive flux
  - telluric/bad wavelength regions

Variance model:
- `Var(Cal) = Cal^2 * [Var(model)/model^2 + Var(obs)/obs^2]`

Each star yields one `CalibrationVector` with metadata (Teff, logg, [M/H], RV, scale).

---

# Combine Multiple Standards

## Robust Combined Calibration Curve

- Input: per-star calibration vectors
- Default method: inverse-variance weighted mean
- Iterative sigma clipping (default 3 sigma)
- Optional smoothing: Savitzky-Golay (`CALIBFLUX_SMOOTH=True`)

Special case:
- If only one standard star is available, pass-through without combination.

QC metric:
- RMS fractional scatter across stars vs wavelength

---

# Apply Calibration to All Fibers

## Final Flux and Variance Update

For every fiber spectrum:
- `F_cal = Counts * Cal_combined`
- `Var_cal = Var_counts * Cal^2 + Counts^2 * Var_calibration`

Write back to RED file:
- Calibrated science image
- Propagated variance
- Header keywords: `BUNIT`, `FLUXCAL`, `FCALNSTR`, `FCALRMS`
- HISTORY lines per standard star (template parameters, RV, scale)

---

# Runtime Controls (`args`)

## Main FluxCal Arguments

- `CALIBFLUX` (bool): enable flux calibration
- `CALIBFLUX_CATALOG` (str): standard-star catalog CSV path
- `CALIBFLUX_FWHM` (float): instrument FWHM in Angstrom
- `CALIBFLUX_METRIC` (`chi2` or `huber`)
- `CALIBFLUX_SMOOTH` (bool): smooth combined curve

Recommended minimum:
- set `CALIBFLUX=True`
- provide `CALIBFLUX_CATALOG`
- ensure fibers include at least one `TYPE='C'`

---

# Practical Example (Commissioning Data)

## Example Resource Paths

- Assign file: `resources/comm/assign/tile_058.assign.txt`
- Standard-star catalog:
  - `resources/comm/20260129/calib/standard_star_atlas_refcat2.csv`

Example fiber classes from assign table:
- `target_class=8` -> standard star -> `TYPE='C'`
- `target_class=9` -> sky fiber -> `TYPE='S'`

---

# QC Checklist

## Validate Calibration Quality

- Confirm `FLUXCAL=True` in final RED header
- Check number of standards used (`FCALNSTR`)
- Inspect scatter (`FCALRMS`) and per-star HISTORY entries
- Plot combined and per-star vectors (`fluxcal/qc.py` utilities)
- Verify calibrated spectra units are `erg/cm2/s/A`

---

# Known Gaps and Next Improvements

## Current Limitations

- Catalog-to-fiber matching is index-based; positional matching is TODO
- Calibration quality depends on available standard stars and photometric bands
- Telluric correction stage is separate and optional (`TELCOR`)

## Planned Enhancements

- RA/Dec-based catalog association
- tighter QC gating and automated outlier rejection policies
- richer nightly transfer-function handling

---

# Summary

## Flux Calibration in This Pipeline

- Uses in-field standard stars (`TYPE='C'`)
- Fits BOSZ templates with RV + continuum normalization
- Anchors absolute scale with catalog photometry
- Combines per-star vectors robustly
- Applies calibrated flux conversion to every fiber with full variance propagation

Deliverable: physically calibrated RED spectra ready for science analysis.
