# Data Reduction Steps

This document summarizes how `kspecdr` converts raw FITS images into calibrated spectra.
The current pipeline now includes four major stages:

1. Tramline map generation
2. Extraction
3. Wavelength calibration
4. Science-frame calibration (flat-field/throughput/sky/flux)

## Status Snapshot (as of 2026-02-19)

Based on the development documents:

- `reduce_object` P0 is complete (end-to-end run)
- P1 core is implemented (flat-fielding, throughput correction, RWSS snapshot, sky subtraction)
- Flux calibration is implemented and integrated (`_apply_fluxcal`)
- Advanced P2/P3 corrections are still in progress and currently safe no-ops behind flags

---

## 1. Tramline Map Generation (Tramline Fitting)

Tramline fitting identifies the precise location of each fiber's spectrum across the CCD image.

### Concept and Importance

Spectra recorded on a CCD are rarely perfectly aligned with the pixel grid. Optical distortions and mechanical alignments cause the spectra to appear as curved "traces" across the detector.

**Tramline fitting** determines the centroid of these traces at every column (spectral pixel). This map is essential for the subsequent extraction step; if the tramlines are inaccurate, the signal from one fiber may bleed into another (crosstalk), or flux may be lost, degrading S/N.

### Mechanism

The process in `kspecdr` involves several algorithmic stages:

1. **Peak detection**
   - **Standard method**: `scipy.signal.find_peaks` with dynamic thresholding
   - **Wavelet method**: Mexican-hat wavelet logic (ported from 2dfdr)
2. **Trace linking**
   - Peaks from different steps are linked into continuous fiber traces
3. **Polynomial fitting**
   - Trace points are fitted (typically 2nd-6th order) into smooth trajectories
4. **Fiber matching**
   - Detected traces are matched to physical fiber IDs

---

## 2. Extraction

Extraction converts the 2D CCD image into 1D per-fiber spectra (intensity vs. pixel) using the tramline map.

### Concept and Importance

This is the transition from image processing to spectroscopic analysis. The goal is to integrate fiber flux while handling noise and contamination.

### Mechanism

`kspecdr` currently provides:

1. **SUM extraction (implemented)**
   - Sums pixel values in a fixed aperture centered on each tramline
2. **Optimal extraction (planned)**
   - Uses profile-weighted extraction for better S/N and crosstalk handling

Variance is propagated during extraction so downstream steps can carry uncertainty consistently.

---

## 3. Wavelength Calibration

Wavelength calibration maps spectral pixels to physical wavelengths (Angstrom or nm).

### Concept and Importance

Science analysis requires wavelength coordinates rather than detector pixels. Arc-lamp lines with known wavelengths provide this mapping.

### Mechanism

1. **Arc spectrum analysis**
   - Detect emission lines (including wavelet-based line finding)
2. **Pattern matching and cross-correlation**
   - Match observed line patterns to reference line lists and predicted models
3. **Landmark registration**
   - Anchor the fit with strong isolated features
4. **Polynomial wavelength solution**
   - Fit robust polynomial to pixel-wavelength pairs

---

## 4. Science-Frame Calibration (Recently Added)

After extraction and wavelength calibration, science spectra still include
instrument-response differences between fibers and additive night-sky
background. The science-frame-calibration stage addresses this and produces
commissioning-grade RED products.

### Concept and Importance

The earlier stages define "where the spectrum is" (trace) and "which
wavelength each pixel corresponds to" (wavelength solution). This stage
focuses on "how the measured counts should be corrected physically."

In practice, the pipeline applies multiplicative corrections first (flat-field,
throughput, optional flux calibration) and additive correction next (sky
subtraction), while tracking variance through each step.

### Mechanism

The overall flow is:

1. **Flat-field correction**
   - Correct relative fiber/pixel response using fiber-flat products.
2. **Scrunching to linear wavelength grid**
   - Rebin all fibers to a common wavelength basis.
3. **Throughput correction**
   - Normalize fiber-to-fiber transmission differences.
4. **Sky subtraction**
   - Build a sky model from designated sky fibers and subtract it from all
     spectra.
5. **Flux calibration (optional)**
   - Use standard stars to convert counts to physical flux scale.
6. **Metadata/finalization**
   - Record reduction status and provenance in FITS headers/extensions.

An optional pre-sky snapshot can also be saved for diagnostics.

---

## 5. Planned / In-Progress Steps

The following items are planned for higher-accuracy (P2/P3) reductions:

1. **Skyline-based wavelength refinement**
   - Fine-tune arc-based calibration using sky emission features and provide QC
     diagnostics for residual offsets.
2. **Advanced sky modeling**
   - Improve subtraction with super-sampled or PCA-based sky models for complex
     sky variability.
3. **Atmospheric and velocity corrections**
   - Add telluric absorption correction and heliocentric/barycentric velocity
     handling.
4. **Transfer-function and artifact corrections**
   - Apply associated transfer-function updates and mitigate residual artifacts
     such as wiggles/fringing.
5. **Extraction upgrades**
   - Extend beyond SUM extraction toward profile-aware methods for improved S/N
     and crosstalk control.

These steps are currently staged as future development and are intended to move
the pipeline from commissioning-grade outputs to publication-grade calibration
quality.
