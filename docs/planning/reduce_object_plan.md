# Object (Science) Reduction Planning Document

> **Status**: P0 complete — pipeline runs end-to-end; P1 core (flatfield, throughput, sky, RWSS) implemented; P2+ in progress.
> **Last updated**: 2026-02-19

---

## 0. Scope

Complete the `reduce_object` pipeline so that a raw science frame can be
reduced end-to-end to a wavelength-calibrated, flat-fielded,
sky-subtracted, flux-calibrated RED file. This document covers every
placeholder function in `reduce_object.py` and related gaps in
`make_ex.py` and `make_red.py`.

### What already works

| Component | Module | Status |
|---|---|---|
| Preprocessing (IM creation) | `preproc/make_im.py` | ~90% |
| Tramline map generation | `tlm/make_tlm.py` | ~90% |
| SUM extraction | `extract/make_ex.py` | Done |
| Arc wavelength calibration | `extract/reduce_arc.py` | ~90% |
| Scrunching (standalone) | `wavecal/scrunch.py` | Done |
| Fiber flat reduction | `reduce_fflat.py` | ~50% |
| Object core (flatfield, throughput, sky, RWSS) | `reduce_object.py` | P1 core implemented |
| Flux calibration | `reduce_object.py:_apply_fluxcal` | Done |

### What is missing

The P1 core of `reduce_object()` (flat-fielding, fiber-throughput
correction, RWSS snapshot, and basic sky subtraction) is now implemented
in `reduce_object.py`.  Remaining gaps are the P2/P3 features
(`skylines_recalibration`, `skycalib_test`, `super_skysub`,
`telcor`, `velcor_update_fibre_table`, `skysubpca`,
`correct_frame_by_assoc_transfer_function`, `propagate_badthput`,
`de_wiggle`) which are still no-ops guarded by flags.  The current
commissioning workflow can now produce wavelength-calibrated,
flat-fielded, sky-subtracted, flux-calibrated RED files; what remains is
to add the advanced corrections and diagnostics.

---

## 1. Architecture Overview

```
Raw science frame
  │
  ▼
make_im          ──► IM file  (bias/dark/flat corrected image + variance)
  │
  ▼
make_ex          ──► EX file  (extracted 1D spectra per fiber + variance)
  │
  ▼
reduce_object    ──► RED file (calibrated, sky-subtracted spectra)
  │
  ├─ skylines_recalibration     [P3]  fine-tune wavelength via sky lines
  ├─ copy EX → RED
  ├─ skycalib_test              [P3]  QC check on skyline wavelengths
  ├─ cmfspec_flatfield          [P1]  divide by fiber flat response
  ├─ scrunch_object_frame       [P0]  rebin to linear wavelength grid
  ├─ tdfio_nod_shuffle          [P0]  trivial flag check
  ├─ cmfspec_ftpcal             [P1]  fiber throughput correction
  ├─ make_rwss                  [P3]  snapshot before sky subtraction
  ├─ skysub                     [P1]  median sky subtraction
  ├─ super_skysub               [P2]  super-sampled sky subtraction
  ├─ tdfio_pixcal_delete        [P3]  housekeeping
  ├─ telcor                     [P2]  telluric absorption correction
  ├─ velcor_update_fibre_table  [P2]  heliocentric velocity correction
  ├─ skysubpca                  [P2]  PCA sky subtraction (alternative)
  ├─ _apply_fluxcal             [Done] flux calibration
  ├─ correct_frame_by_assoc_transfer_function  [P3]
  ├─ propagate_badthput         [P3]  flag bad-throughput fibers
  ├─ de_wiggle                  [P3]  remove sinusoidal artifacts
  ├─ tdfio_sds_write            [P3]  write args to FITS header
  ├─ tdfio_setred               [P3]  set REDUCED status flag
  └─ stamp_2dfdrver             [P3]  write pipeline version
```

---

## 2. Priority Definitions

| Priority | Meaning | Goal |
|---|---|---|
| **P0** | Critical — pipeline cannot produce useful output without this | First usable end-to-end reduction |
| **P1** | High — needed for science-quality reductions | Commissioning-grade spectra |
| **P2** | Medium — improves accuracy and enables advanced use cases | Publication-quality spectra |
| **P3** | Low — housekeeping, diagnostics, edge cases | Polish and completeness |

---

## 3. Implementation Plan

### Phase 0 — Minimum Viable Pipeline

Goal: run `reduce_object` end-to-end on commissioning data without errors
and produce wavelength-calibrated extracted spectra.

#### P0-1. `tdfio_nod_shuffle` — Nod & Shuffle flag check

- **Effort**: trivial (< 10 lines)
- **Action**: return `0` for standard (non-N&S) mode. Read `UTNODSFL`
  header keyword if present; otherwise assume standard mode.
- **Why P0**: this function gates access to `cmfspec_ftpcal`, `skysub`,
  and `skysubpca` — without it, the pipeline crashes immediately.

#### P0-2. `scrunch_object_frame` — Wavelength rebinning

- **Effort**: small (~30 lines, wiring)
- **Action**: wrap the existing `scrunch_from_arc_id` from
  `wavecal/scrunch.py`.  Read `WAVEL_FILENAME` from `args`, call
  `scrunch_from_arc_id(red_filename, arc_filename, args)`.
- **Depends on**: the arc RED file existing with a calibrated `WAVELA`
  extension.
- **Why P0**: without scrunching, spectra remain on pixel coordinates —
  flat-fielding, sky subtraction, and flux calibration all assume a
  common wavelength grid.

#### P0-3. I/O utilities — `tdfio_sds_write`, `tdfio_setred`, `stamp_2dfdrver`, `tdfio_pixcal_delete`

- **Effort**: small (~50 lines total)
- **Action**:
  - `tdfio_sds_write`: write selected `args` keys as FITS header cards
    (e.g., `HIERARCH KSPECDR_<KEY> = <VALUE>`), or as a `DRARGS`
    BinTableHDU.
  - `tdfio_setred`: set `DRSTATUS = 'REDUCED'` header keyword.
  - `stamp_2dfdrver`: set `DRPIPVER = kspecdr.__version__` header
    keyword.
  - `tdfio_pixcal_delete`: delete `PIXCAL` HDU if present (try/except).
- **Why P0**: these are required for the function to complete without
  raising `NotImplementedError`, even though their scientific impact is
  minimal.

#### P0-4. Remaining stubs — make non-critical functions conditional no-ops

- **Effort**: trivial
- **Action**: for functions not yet implemented that are behind
  `args.get(FLAG, False)` guards (e.g., `skylines_recalibration`,
  `skycalib_test`, `super_skysub`, `telcor`, `velcor_update_fibre_table`,
  `skysubpca`, `correct_frame_by_assoc_transfer_function`,
  `propagate_badthput`, `de_wiggle`), change `raise NotImplementedError`
  to `logger.warning("... not implemented, skipping")` and return early.
  This allows the pipeline to run when these features are not requested.
- For functions that are unconditionally called
  (`skylines_recalibration`, `propagate_badthput`, `de_wiggle`), add a
  check for an enable flag or make them no-ops with a warning.
- **Why P0**: the pipeline must not crash on any code path during normal
  operation.

**P0 deliverable**: `reduce_object` runs end-to-end on commissioning
data and produces a scrunched RED file. No flat-fielding or sky
subtraction yet, but the spectra are wavelength-calibrated and
inspectable.

---

### Phase 1 — Science-Quality Core

Goal: produce flat-fielded, sky-subtracted, wavelength-calibrated spectra
suitable for commissioning analysis.

#### P1-1. `cmfspec_flatfield` — Fiber flat-field division

- **Effort**: medium (~70 lines)
- **Called at**: after EX→RED copy, **before** scrunching (pixel space)
- **Action**:
  1. Check `args.get('USEFFLAT', True)`.  If `False`, write history
     `'Not divided by fibre flat field'` and return immediately.
  2. Read `FFLAT_FILENAME` from `args`.  If missing or empty, raise
     (or log error and return); 2dfdr sets status `DRS__NOFFLAT`.
  3. Open the FFLAT RED FITS file and read the primary image array
     `FLTIMG (NPIX, NFIB)` and variance array `FLTVAR (NPIX, NFIB)`.
  4. Read the science RED primary image `OBJIMG` and variance `OBJVAR`.
     **Assert** `OBJIMG.shape == FLTIMG.shape`; raise a clear error if
     dimensions do not match (this is the most common failure mode).
  5. Check for optional truncation: if `TRUNCFLAT` is `True`, read
     `USEFLATSTART` (default 1) and `USEFLATEND` (default 2048) and set
     `FLTIMG[:USEFLATSTART-1, :] = 1.0`, `FLTVAR[:USEFLATSTART-1, :] = 0.0`
     (likewise for pixels beyond `USEFLATEND`) so those regions are
     divided by unity (no correction applied outside the trusted range).
  6. Division (element-wise, handling bad pixels / zeros):
     ```python
     good = (FLTIMG != 0) & np.isfinite(FLTIMG) & np.isfinite(OBJIMG)
     OBJIMG_out = np.where(good, OBJIMG / FLTIMG, np.nan)
     ```
  7. Full variance propagation (Taylor expansion, both terms):
     ```
     Var_out = (1/flat)^2 * Var_obj + (obj/flat^2)^2 * Var_flat
     ```
     Set `OBJVAR_out = np.nan` wherever `OBJIMG_out` is NaN.
  8. Write updated image and variance back to the RED file.
  9. Write FITS history: `f'Divided by fibre flat field {FFLAT_FILENAME}'`.
     If `TRUNCFLAT`, append a second history record noting the pixel
     range used.
- **What the FFLAT RED file contains**: `reduce_fflat` produces a
  pixel-space (un-scrunched) flat where each fiber spectrum is
  normalized to ≈1.0 (the per-fiber median is used for normalization,
  then the global averaged flat is removed). Values should be close to
  1.0 ± a few per cent. No explicit `THPUT` extension is present —
  that is the job of `cmfspec_ftpcal`.
- **Confirmed**: flat-fielding happens in **pixel space**, before
  scrunching. `reduce_fflat` scrunch→average→unscrunch→extrapolate→
  normalize so the output is always pixel-space. This matches the
  2dfdr call order in `reduce_object.F95` (line 183, before
  `SCRUNCH_OBJECT_FRAME` at line 190).

#### P1-2. `cmfspec_ftpcal` — Fiber throughput correction

- **Effort**: medium (~90 lines)
- **Called at**: after scrunching, **before** sky subtraction; skipped
  entirely if `TDFIO_NOD_SHUFFLE` returns non-zero (Nod & Shuffle mode)
- **Action**:
  1. Check `args.get('THRUPUT', True)`.  If `False`, write history
     `'No throughput calibration performed'` and return.
  2. Check `args.get('USETHPTFILE', False)`.  If `True` and a
     `THROUGHPUT.fits` file exists in the working directory, read its
     `THPUT` extension (1-D vector of length NFIB) and use those values
     directly (skipping all calculation below).
  3. Otherwise, select the calculation method from
     `args.get('TPMETH', 'OFFSKY')`:
     - **`'OFFSKY'`** *(default)*: read `THPUT_FILENAME` from args;
       open that file and read its `THPUT` extension.  If filename is
       empty, log a warning `'No throughput map available — continuing'`
       and return without modifying data.
     - **`'SKYLINE(KGB)'`**: Karl Glazebrook's sky-line algorithm —
       (a) call `umfspec_ftpc` (mean-per-fiber, median-normalized) for a
       first-pass estimate; (b) build a median sky spectrum from sky
       fibers (`TYPE='S'`) normalized by that estimate; (c) subtract
       continuum via a ±100-pixel median filter; (d) for each P/S fiber,
       do a robust linear fit (slope B) of the continuum-subtracted fiber
       spectrum against the median sky spectrum; (e) `thput = B` if
       `0.01 < B < 100`, else `NaN`; (f) normalize the whole vector by
       its median.
     - **`'SKYFLUX(COR)'`** / **`'SKYFLUX(MED)'`**: sky-line flux
       integration methods — read `skylines.dat` for line positions,
       integrate continuum-subtracted flux in ±`2×FWHM`-pixel windows,
       normalize per line and across lines.  These are more robust when
       many sky lines are present but require a `skylines.dat` data file.
     - **Recommendation for KSPEC commissioning**: start with
       `'SKYLINE(KGB)'` (self-contained, no external file); fall back to
       `'OFFSKY'` if a dedicated sky frame is available.
  4. `thput_vec` is a 1-D float array of length NFIB.  Sanity bounds:
     values outside `(0.01, 100.0)` are set to `NaN`.
  5. Divide each fiber's spectrum and variance by its throughput:
     ```python
     scale = np.where(np.isfinite(thput_vec) & (thput_vec > 0),
                      1.0 / thput_vec, 0.0)
     spec[:, i] *= scale[i]
     var[:, i]  *= scale[i]**2
     ```
     (2dfdr sets scale=0 for bad fibers so that their spectra become
     identically zero, not NaN — this is intentional to prevent sky
     contamination.)
  6. Write the throughput vector to a `THPUT` extension in the RED file
     (1-D BinTable or ImageHDU of length NFIB). Replace any remaining
     NaN values with 0.0 before writing (per 2dfdr convention).
  7. Write FITS history noting the method used.
- **Note**: `THPUT` is written as a file extension, not as a column in
  the `FIBRES` table. The `FIBRES` table `THPUT` column is a separate
  (optional) annotation.

#### P1-3. `skysub` — Sky subtraction

- **Effort**: large (~100 lines)
- **Called at**: after throughput correction, before telluric; skipped
  for Nod & Shuffle data
- **Action**:
  1. Check `args.get('SKYSUB', True)`.  If `False`, write history
     `'No sky subtraction performed'` and return.
  2. **Identify sky fibers** via `getskyfibres()`:
     - Read `FIBRES` table `TYPE` column; collect indices where
       `TYPE == 'S'`.
     - Fallback if no FIBRES table: read from a `skyfibres.dat` ASCII
       file (one fiber index per line).
     - If no sky fibers are identified at all, log a warning
       `'No sky fibers found'` and skip sky subtraction (do not crash).
     - (`AUTO_SKYFIBRE_DECLARATION` in 2dfdr handles instrument-specific
       auto-assignment of sky fibers when the table has none; for KSPEC
       this path should not be needed if the assign file correctly
       populates the FIBRES table.)
  3. **Reject bad sky fibers**: a sky fiber is bad if more than 1/8 of
     its pixels are `NaN`/bad (the `FCHECK` criterion from 2dfdr).
     Keep a list of good sky fibers.
  4. **Combine sky fibers** into a single sky spectrum and sky variance:
     - Method selected by `args.get('SKYCOMBINE', 'MEAN')`;
       supported values: `'MEAN'` or `'MEDIAN'`.
     - For MEAN: `sky = np.nanmean(spectra[:, good_sky_fibs], axis=1)`.
       Variance: `sky_var = np.nansum(var[:, good_sky_fibs], axis=1) / N_good^2`
       (error propagation for mean of N independent spectra).
     - For MEDIAN: use `np.nanmedian`; sky variance estimate should use
       the `π/2 · MEAN_VAR / N` scaling factor or fall back to the mean
       variance.
  5. **Normal sky subtraction** (default, `ITERSKY = False`):
     ```python
     for fib in range(NFIB):
         good = ~(np.isnan(spec[:,fib]) | np.isnan(sky))
         spec[good, fib] -= sky[good]
         var[good, fib]  += sky_var[good]
         spec[~good, fib] = np.nan
         var[~good, fib]  = np.nan
     ```
     The variance addition `Var_fib += Var_sky` is the correct
     propagation (not `Var_sky / N_sky` — the division by N_sky is
     already baked into the sky variance from step 4).
  6. **Iterative sky subtraction** (optional, `ITERSKY = True`): not
     required for P1; add a `NotImplementedError`-as-warning stub.
  7. Write sky-subtracted spectra and updated variances back to the
     RED file's PRIMARY and `VARIANCE` extensions.
  8. Write the combined sky spectrum to a `SKY` extension (1-D
     ImageHDU, NPIX elements).  Replace bad pixels with 0.0 before
     writing (2dfdr uses `CMFSPEC_COPY` for this).
  9. Write FITS history: `f'Sky subtracted using {N_good} sky fibers'`.
- **Design notes for ISOPLANE (~2 sky fibers)**:
  - With only 2 sky fibers, MEAN and MEDIAN produce the same result;
    use MEAN as the default.
  - Sky subtraction quality will be limited by small-number statistics
    and spatial sky gradients across the field; document this.
  - Consider exposing `SKYCOMBINE` in the commissioning notebook args
    so it can be toggled easily.

#### P1-4. `make_rwss` — Pre-sky-subtraction snapshot (optional)

- **Effort**: small (~15 lines)
- **Called at**: after `cmfspec_ftpcal`, immediately before `skysub`
- **Action**: if `args.get('INC_RWSS', False)`, copy the current
  PRIMARY image array to a new `RWSS` ImageHDU in the RED file. No
  header modification needed beyond the extension name.
- **Why P1**: the RWSS ("Reduced Without Sky Subtraction") HDU lets the
  commissioning analyst compare pre/post sky-subtraction spectra in the
  same file.  Also required by the planned `MEASURE_SKY_RESIDUALS`
  diagnostic (P3).
- **Note**: `INC_RWSS` defaults to `False` in 2dfdr (`reduce_object.F95`
  line 211); we should follow this default.

**P1 deliverable**: `reduce_object` produces flat-fielded,
sky-subtracted, wavelength-calibrated spectra. Combined with the
already-implemented flux calibration, this is a complete basic science
reduction.

---

### Phase 2 — Advanced Corrections

Goal: publication-quality spectra with atmospheric, velocity, and
advanced sky corrections.

#### P2-1. `telcor` — Telluric absorption correction

- **Effort**: medium (~80 lines)
- **Design options** (decide before implementing):
  - **Option A**: Empirical — use a hot star (TYPE='C' or dedicated
    telluric standard) observed at similar airmass. Divide science by
    the (continuum-normalized) telluric star spectrum. Simplest; requires
    a suitable star on the plate.
  - **Option B**: Model-based — use a pre-computed atmospheric
    transmission model (e.g., Molecfit output or a lookup table by
    airmass + PWV). More general but requires external data.
  - **Option C**: Combined — use the telluric mask regions from
    `data/masks/telluric_default.dat` to flag (not correct) affected
    wavelengths. Quickest to implement as a first pass.
- **Recommendation**: implement Option C first (flag-only), then Option A
  for correction.
- **Depends on**: scrunched data on a common wavelength grid.

#### P2-2. `velcor_update_fibre_table` — Velocity corrections

- **Effort**: medium (~60 lines)
- **Action**:
  1. Compute heliocentric/barycentric velocity correction from the
     observation date, RA/Dec, and observatory coordinates using
     `astropy.coordinates.SkyCoord.radial_velocity_correction`.
  2. Store `VHELIO` (km/s) in the `FIBRES` table.
  3. Optionally store `VLSR` (Local Standard of Rest).
  4. Do NOT apply the correction to the wavelength grid (let the user
     decide); only store the values.
- **Depends on**: `astropy.coordinates`, valid `RA`, `DEC`, `DATE-OBS`,
  `OBSERVAT` (or `LONG-OBS`, `LAT-OBS`, `ALT-OBS`) header keywords.

#### P2-3. `super_skysub` — Super-sampled sky subtraction

- **Effort**: large (~150 lines)
- **Action**: uses the original EX file (pixel-space) and the RED file
  (scrunched) to perform sky subtraction at higher spectral resolution
  than the output grid. Primarily relevant when the number of sky fibers
  is large enough to build a spatially-varying sky model. Low priority
  for ISOPLANE (14 fibers).
- **Note**: can be deferred until full KSPEC is operational.

#### P2-4. `skysubpca` — PCA sky subtraction

- **Effort**: large (~150 lines)
- **Action**: PCA decomposition of sky-fiber spectra to build an
  eigenspectrum basis, then project and subtract from all fibers. More
  robust for structured sky residuals (e.g., OH line variation across the
  field).
- **Note**: most valuable for the full KSPEC instrument with many sky
  fibers. Defer for ISOPLANE commissioning.

#### P2-5. `skylines_recalibration` — Wavelength fine-tuning from sky lines

- **Effort**: medium (~80 lines)
- **Action**:
  1. After extraction (on the EX file), identify known sky emission lines
     (e.g., OI 5577, OI 6300, OH lines).
  2. Measure centroids in each fiber.
  3. Compute per-fiber wavelength offsets/shifts.
  4. Apply as corrections to the `WAVELA` extension before scrunching.
- **Why P2**: the arc calibration is usually sufficient for commissioning,
  but sky-line recalibration corrects for flexure between arc and science
  exposures.

**P2 deliverable**: telluric-corrected, velocity-annotated spectra with
optional advanced sky subtraction.

---

### Phase 3 — Polish and Completeness

Goal: complete feature parity with the 2dfdr `REDUCE_OBJECT` flow and
production-ready diagnostics.

#### P3-1. `skycalib_test`

- Test function that verifies the skyline recalibration worked correctly.
  Log statistics (mean offset, scatter) and optionally raise a warning
  if residuals exceed a threshold.

#### P3-2. `correct_frame_by_assoc_transfer_function`

- Apply a transfer function from an associated observation (e.g., a
  spectrophotometric standard at different airmass). Primarily relevant
  for multi-visit survey operations.

#### P3-3. `propagate_badthput`

- If a fiber's throughput was flagged as bad during `cmfspec_ftpcal`,
  propagate NaN to all wavelength pixels for that fiber so downstream
  analysis doesn't use unreliable data.

#### P3-4. `de_wiggle`

- Remove sinusoidal fringing artifacts. Implementation depends on
  characterizing the KSPEC detector's fringe pattern. Requires
  commissioning data showing the artifact.

#### P3-5. `clean_im` — Double-pass cosmic ray rejection

- Use the residual map from optimal extraction to identify cosmic rays
  in the IM frame, clean them, then re-extract. Only meaningful once
  optimal extraction is implemented.

---

## 4. Extraction Improvements (Parallel Track)

These are independent of `reduce_object` but improve the EX files it
consumes.

| Item | Effort | Priority | Notes |
|---|---|---|---|
| GAUSS extraction | Large | P2 | Gaussian profile fit per fiber per column |
| Optimal extraction (OPTEX) | Large | P2 | Horne (1986) variance-weighted extraction; maximizes S/N |
| Scattered light subtraction | Medium | P2 | Fit and subtract inter-fiber background before extraction |

---

## 5. Fiber Flat Completion (Dependency)

`reduce_fflat.py` is at ~50% completion. For `cmfspec_flatfield` (P1-1)
to work, the following must be verified:

1. `reduce_fflat` produces a valid master flat RED file with correct
   array dimensions matching science EX files (`NPIX × NFIB`).
2. **The flat is in pixel space (not scrunched)**: `reduce_fflat`
   scrunch→averages→unscrunch→extrapolate→normalize, so the
   output RED flat is always in pixel space with values ≈ 1.0.
   This is confirmed by `SCRUNCH_FLAT_FRAME` in `reduce_fflat.F95`.
3. Fiber types (`P`, `S`) in the flat FIBRES table must be populated
   to correctly exclude non-illuminated fibers (`U`, `N`, `G`, etc.)
   from the averaging step.
4. `CMFFF_EXTRAP` performs a linear least-squares extrapolation into
   the first/last 5 pixels of each fiber spectrum to fill edge bad
   pixels created by scrunching; verify that the Python equivalent
   does the same (or replaces it with a constant edge-fill).
5. Optional B-spline post-smoothing (`BSSMOOTH` flag) is available in
   2dfdr but not required for P1.

**Action**: test `reduce_fflat` on commissioning flat data, inspect
the output (values near 1.0, no NaN-dominated fibers), and confirm
shape matches science EX files before implementing `cmfspec_flatfield`.

---

## 6. Implementation Order Summary

```
Phase 0 (MVP — pipeline runs without crashing)
  ├── P0-1  tdfio_nod_shuffle            trivial
  ├── P0-2  scrunch_object_frame         small (wire existing code)
  ├── P0-3  I/O utilities                small (FITS header writes)
  └── P0-4  Remaining stubs → no-ops     trivial

Phase 1 (Science-quality core)
  ├── P1-1  cmfspec_flatfield            medium  (~70 lines; pixel-space division, full Var propagation)
  ├── P1-2  cmfspec_ftpcal               medium  (~90 lines; OFFSKY default, KGB optional)
  ├── P1-3  skysub                       large   (~100 lines; MEAN/MEDIAN combine, bad-fib filter)
  └── P1-4  make_rwss                    small   (~15 lines; copy PRIMARY → RWSS HDU)

Phase 2 (Advanced corrections)
  ├── P2-1  telcor                       medium
  ├── P2-2  velcor_update_fibre_table    medium
  ├── P2-3  super_skysub                 large (defer for ISOPLANE)
  ├── P2-4  skysubpca                    large (defer for ISOPLANE)
  └── P2-5  skylines_recalibration       medium

Phase 3 (Polish)
  ├── P3-1  skycalib_test                small
  ├── P3-2  transfer function            medium
  ├── P3-3  propagate_badthput           small
  ├── P3-4  de_wiggle                    unknown
  └── P3-5  clean_im                     medium (needs OPTEX first)
```

---

## 7. Testing Strategy

### Unit tests

Each implemented function should have a corresponding test in
`tests/test_reduce_object.py` covering:

- Normal operation with valid input.
- Graceful handling of missing optional files (e.g., no flat → skip).
- Edge cases: all-NaN spectra, single-fiber data, zero sky fibers.

### Integration test

A notebook-based end-to-end test using commissioning data
(`20260129` dataset):

1. Convert raw files (bias, flat, arc, science).
2. Build master bias.
3. Preprocess flat → TLM → extract flat → `reduce_fflat`.
4. Preprocess arc → extract → `reduce_arc`.
5. Preprocess science → extract → `reduce_object` (full pipeline).
6. Inspect output: plot wavelength-calibrated, sky-subtracted spectra.

### Regression test

Compare output RED files against a reference set (to be generated once
Phase 1 is complete) to catch accidental changes.

---

## 8. Open Questions

1. **Flat-field order**: confirmed — 2dfdr applies flat-fielding
   **before** scrunching (`reduce_object.F95` lines 183→190). The
   FFLAT RED file is in pixel space by design. No change needed.

2. **Sky fibers in ISOPLANE**: with only ~2 sky fibers out of 14,
   MEAN and MEDIAN sky combination are equivalent. Simple subtraction
   is the only practical option; iterative sky subtraction (`ITERSKY`)
   requires more sky fibers and can be deferred. Document the
   limitation in the commissioning notebook.

3. **Throughput method for KSPEC**: `'OFFSKY'` requires a dedicated
   sky frame with a pre-computed `THPUT` extension. `'SKYLINE(KGB)'`
   computes throughput from the science frame itself using sky fibers,
   making it the most self-contained option for commissioning. Decide
   which method to use based on whether an offset sky frame is
   available. Note that `SKYFLUX(*)` methods require `skylines.dat`.

4. **`THPUT` extension vs. `FIBRES` table column**: 2dfdr writes
   throughputs as a standalone `THPUT` ImageHDU (1-D, NFIB elements),
   not as a column in `FIBRES`. Our Python implementation should
   follow this. The `FIBRES.THPUT` column is a separate annotation
   written by `velcor_update_fibre_table`.

5. **Bad sky fiber handling**: `FCHECK` (< 1/8 bad pixels) is defined
   in `skysub.F95` but is not called in the `SKYSUB` flow visible in
   the source — bad pixel masking is delegated to `COMBINE_STACK`.
   In Python, explicitly filter sky fibers with a NaN-fraction check
   before combining.

6. **Telluric standard**: is a dedicated telluric standard observed
   during commissioning, or should we rely on flux calibration stars
   (`TYPE='C'`)?

7. **Velocity correction source**: do science targets have RA/Dec
   stored in the `FIBRES` table, or only in the assign file? Need to
   decide where `velcor_update_fibre_table` reads coordinates from.

8. **`TDFIO_WAVE_DELETE`**: 2dfdr deletes the `WAVELA` extension from
   the RED file after scrunching (line 201 of `reduce_object.F95`).
   Verify whether `kspecdr` needs to do the same, or whether keeping
   `WAVELA` for diagnostics is preferable.

---

## 9. Dependencies and Blockers

| Blocker | Affects | Resolution |
|---|---|---|
| `reduce_fflat` output quality unverified | P1-1 `cmfspec_flatfield` | Run `reduce_fflat` on commissioning data, inspect output |
| Assign file → FIBRES table integration | P1-3 `skysub` (needs TYPE='S') | Verify `write_isoplane_converted_image` propagates TYPE from assign table |
| Observatory coordinates in headers | P2-2 `velcor_update_fibre_table` | Check if LONG-OBS/LAT-OBS/ALT-OBS are set during conversion |
| Telluric standard identification | P2-1 `telcor` | Determine if TYPE='C' stars can serve as telluric standards |
