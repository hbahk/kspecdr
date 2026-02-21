# Data Reduction Steps

This document details the key processes involved in converting raw FITS images into calibrated spectra using the `kspecdr` pipeline. The pipeline, inspired by 2dfdr but optimized for a Python environment, performs three major steps: **Tramline Map Generation**, **Extraction**, and **Wavelength Calibration**.

Each step is critical for ensuring data quality and scientific accuracy.

## 1. Tramline Map Generation (Tramline Fitting)

Tramline fitting identifies the precise location of each fiber's spectrum across the CCD image.

### Concept and Importance

Spectra recorded on a CCD are rarely perfectly aligned with the pixel grid. Optical distortions and mechanical alignments cause the spectra to appear as curved "traces" across the detector.

**Tramline Fitting** determines the centroid of these traces at every column (spectral pixel). This map is essential for the subsequent extraction step; if the tramlines are inaccurate, the signal from one fiber may bleed into another (crosstalk), or flux may be lost, degrading the signal-to-noise ratio (S/N).

### Mechanism

The process in `kspecdr` involves several algorithmic stages:

1.  **Peak Detection**:
    The pipeline scans the image in the spatial direction (typically the Y-axis) at regular intervals (steps) to locate intensity peaks corresponding to fibers.
    *   **Standard Method**: Uses `scipy.signal.find_peaks` to identify peaks above a dynamic threshold.
    *   **Wavelet Method**: Implements a Mexican Hat Wavelet transform (ported from 2dfdr logic) to detect peaks robustly, even in the presence of noise or background variation.

2.  **Trace Linking**:
    Peaks detected at different steps are linked together to form continuous traces. A clustering algorithm (based on Euclidean distance) groups peaks that belong to the same fiber trajectory.

3.  **Polynomial Fitting**:
    The linked points are fitted with a polynomial (typically 2nd to 6th order) to create a smooth model of the trace. This results in a "Tramline Map" that defines the Y-position of every fiber for every X-pixel.

4.  **Fiber Matching**:
    The detected traces are matched to physical fiber IDs using a pattern matching algorithm that compares the observed trace positions with the instrument's nominal fiber positions.

---

## 2. Extraction

Extraction converts the 2D CCD image into 1D spectra (Intensity vs. Pixel) using the generated Tramline Map.

### Concept and Importance

This is the transition from "image processing" to "data analysis." The goal is to integrate the flux falling onto each fiber while rejecting background noise, cosmic rays, and contamination from scattered light or adjacent fibers.

### Mechanism

`kspecdr` supports different extraction methods:

1.  **Sum Extraction (TRAM/SUM)**:
    *   **How it works**: Simply sums the pixel values within a fixed width (aperture) centered on the tramline.
    *   **Pros/Cons**: Fast and robust for high S/N data. However, it adds noise from the wings of the profile where little signal exists, and it handles fiber crosstalk poorly.

2.  **Optimal Extraction (Work in Progress)**:
    *   **How it works**: Assumes a spatial profile (e.g., Gaussian) for the fiber. It weights pixels by their expected signal fraction, giving higher weight to the bright center and lower weight to noisy wings.
    *   **Pros/Cons**: Maximizes S/N and can mathematically solve for crosstalk between overlapping profiles. It is computationally more intensive but provides superior results for faint targets.

During extraction, a **Variance** array is also propagated, ensuring that uncertainties are tracked correctly for subsequent analysis.

---

## 3. Wavelength Calibration

Wavelength calibration maps the spectral pixel coordinates (X-axis) to physical wavelengths (Angstroms or Nanometers).

### Concept and Importance

Raw spectra are measured in pixels. To analyze the physical properties of astronomical objects (e.g., redshift, chemical composition), these pixels must be converted to wavelength. This is achieved by comparing an observed "Arc" lamp spectrum (with known emission lines like CuAr or FeAr) to a reference table.

### Mechanism

1.  **Arc Spectrum Analysis**:
    The pipeline extracts the spectrum of an arc lamp. It uses wavelet transforms to detect emission lines and determine their centroids with sub-pixel precision.

2.  **Pattern Matching & Cross-Correlation**:
    The observed pattern of lines is matched against a reference list of known wavelengths.
    *   A **theoretical spectrum** is generated based on the optical model (prediction).
    *   **Cross-correlation** is performed between the observed arc spectrum and the theoretical model to find the global shift and align the spectra roughly.

3.  **Landmark Registering**:
    Strong, isolated lines are identified as "landmarks." These are used to anchor the solution and correct for non-linear distortions across the detector.

4.  **Polynomial Fitting**:
    Finally, a robust high-order polynomial is fitted to the pairs of (Pixel Position, Reference Wavelength). This polynomial provides the Wavelength Solution, allowing every pixel in the science frames to be assigned a precise wavelength.
