# Object (Science) Reduction Planning Document

> **Status**: P0 complete — pipeline runs end-to-end; P1 in progress.
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
| Flux calibration | `reduce_object.py:_apply_fluxcal` | Done |

### What is missing

Every step in `reduce_object()` between `make_ex(args)` and
`_apply_fluxcal` is a `NotImplementedError` stub. The current
commissioning workflow (notebook) stops after arc calibration and manual
scrunching; no object frame has been reduced end-to-end yet.

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

- **Effort**: medium (~60 lines)
- **Action**:
  1. Read `FFLAT_FILENAME` from `args`.  If not set, log warning and
     skip.
  2. Open the master FFLAT RED file (produced by `reduce_fflat`).
  3. Read the flat-field data array (NFIB, NPIX).
  4. Divide the RED file's spectra and variance by the flat-field
     response.  Handle division by zero / NaN: set to NaN and propagate
     to the mask.
  5. Variance propagation: `Var_out = Var_in / flat²` (if flat has
     negligible variance) or the full expression if flat variance is
     tracked.
  6. Update FITS header: `FLATCOR = True`, `FLATFILE = <filename>`,
     `HISTORY`.
- **Depends on**: `reduce_fflat` producing a valid master flat. Currently
  `reduce_fflat` is ~50% complete. Need to verify its output is usable
  and matches the EX/RED array dimensions.
- **Open question**: should flat-fielding happen before or after
  scrunching? The 2dfdr order is flat-field first, then scrunch. This
  means the flat must be in pixel space (un-scrunched). Verify that
  `reduce_fflat` produces pixel-space output.

#### P1-2. `cmfspec_ftpcal` — Fiber throughput correction

- **Effort**: medium (~50 lines)
- **Action**:
  1. For each fiber, compute the median flux in a clean spectral region
     (avoiding ends and sky lines).
  2. Normalize each fiber by its median so all fibers have comparable
     flux levels.
  3. Store per-fiber throughput values in the `FIBRES` table as a
     `THPUT` column.
  4. This is distinct from flat-fielding (which corrects the wavelength-
     dependent response); throughput correction handles the overall
     efficiency of each fiber.
- **Why P1**: sky subtraction requires all sky fibers to be on a
  comparable flux scale. Without throughput correction, the median sky
  will be biased toward brighter sky fibers.

#### P1-3. `skysub` — Sky subtraction

- **Effort**: large (~120 lines)
- **Action**:
  1. Identify sky fibers from the `FIBRES` table (`TYPE = 'S'`).
  2. Reject bad sky fibers (NaN-dominated, outlier flux).
  3. Compute median (or sigma-clipped mean) sky spectrum from good sky
     fibers.
  4. Optionally: iterative sky subtraction (subtract sky, re-estimate
     residual, subtract again).
  5. Subtract sky from all fibers (including sky fibers themselves for
     QC).
  6. Propagate variance: `Var_out = Var_fiber + Var_sky / N_sky`.
  7. Store the master sky spectrum as a `SKY` HDU in the RED file for
     diagnostics.
  8. Update header: `SKYSUB = True`, `NSKYFIBS = N`, `HISTORY`.
- **Design choices**:
  - For the ISOPLANE (14 fibers, ~2 sky), a simple median of sky fibers
    is appropriate. With more fibers (full KSPEC), consider iterative
    B-spline sky models.
  - The 2dfdr order is: throughput → sky sub.  Follow this.
- **Open question**: with only ~2 sky fibers in ISOPLANE commissioning
  data, sky subtraction quality will be limited. Document this
  limitation.

#### P1-4. `make_rwss` — Pre-sky-subtraction snapshot (optional)

- **Effort**: small (~15 lines)
- **Action**: before sky subtraction, if `INC_RWSS = True`, copy current
  `PRIMARY` data to a new `RWSS` ImageHDU. This allows comparing
  before/after sky subtraction.
- **Why P1**: useful diagnostic for verifying sky subtraction quality,
  especially during commissioning.

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
   array dimensions matching science EX files.
2. The flat is in pixel space (not scrunched), since flat-fielding
   happens before scrunching in `reduce_object`.
3. Fiber types in the flat match those in the science frame.

**Action**: test `reduce_fflat` on commissioning flat data and verify the
output before implementing `cmfspec_flatfield`.

---

## 6. Implementation Order Summary

```
Phase 0 (MVP — pipeline runs without crashing)
  ├── P0-1  tdfio_nod_shuffle            trivial
  ├── P0-2  scrunch_object_frame         small (wire existing code)
  ├── P0-3  I/O utilities                small (FITS header writes)
  └── P0-4  Remaining stubs → no-ops     trivial

Phase 1 (Science-quality core)
  ├── P1-1  cmfspec_flatfield            medium
  ├── P1-2  cmfspec_ftpcal               medium
  ├── P1-3  skysub                       large
  └── P1-4  make_rwss                    small

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

1. **Flat-field order**: 2dfdr applies flat-fielding before scrunching.
   Confirm this is correct for KSPEC / Isoplane, or whether scrunching
   first and then dividing by a scrunched flat is preferable.

2. **Sky fibers in ISOPLANE**: with only ~2 sky fibers out of 14, is
   median sky subtraction sufficient? Should we explore fitting sky
   models using spatial information?

3. **Telluric standard**: is a dedicated telluric standard observed during
   commissioning, or should we rely on the flux calibration stars
   (TYPE='C')?

4. **Velocity correction source**: do science targets have RA/Dec stored
   in the `FIBRES` table, or only in the assign file? Need to decide
   where `velcor_update_fibre_table` reads coordinates from.

5. **Throughput normalization reference**: should throughput be normalized
   to the median fiber, the brightest fiber, or unity? 2dfdr uses
   median.

---

## 9. Dependencies and Blockers

| Blocker | Affects | Resolution |
|---|---|---|
| `reduce_fflat` output quality unverified | P1-1 `cmfspec_flatfield` | Run `reduce_fflat` on commissioning data, inspect output |
| Assign file → FIBRES table integration | P1-3 `skysub` (needs TYPE='S') | Verify `write_isoplane_converted_image` propagates TYPE from assign table |
| Observatory coordinates in headers | P2-2 `velcor_update_fibre_table` | Check if LONG-OBS/LAT-OBS/ALT-OBS are set during conversion |
| Telluric standard identification | P2-1 `telcor` | Determine if TYPE='C' stars can serve as telluric standards |
