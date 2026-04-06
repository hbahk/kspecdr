# Package Architecture

`kspecdr` is a Python-based data reduction pipeline for K-SPEC, designed to be modular and extensible. It separates instrument definitions, file I/O, core algorithms, and orchestration logic.

## Project Structure

The package is organized as follows:

```text
src/kspecdr/
├── constants.py          # Global constants and definitions
├── tracking.py           # Multi-target tracking for trace linking
├── reduce_object.py      # Top-level science frame reduction
├── reduce_fflat.py       # Top-level fibre-flat reduction
├── extract/              # Spectral extraction and reduction orchestration
│   ├── make_ex.py        # Extraction routines (Sum, Tramline)
│   ├── make_red.py       # Final reduction steps
│   └── reduce_arc.py     # Arc reduction workflow
├── fluxcal/              # Flux calibration pipeline
│   ├── containers.py     # Data containers (Spectrum1D, Photometry, etc.)
│   ├── photometry.py     # Magnitude conversions and filter handling
│   ├── templates.py      # BOSZ template library access
│   ├── continuum.py      # Continuum normalization
│   ├── matching.py       # Template matching and RV fitting
│   ├── calibration.py    # Calibration vector computation
│   ├── masks.py          # Telluric and bad-region masks
│   ├── qc.py             # Quality-control plots
│   └── download_bosz.py  # BOSZ 2024 subgrid downloader
├── inst/                 # Instrument-specific definitions
│   └── isoplane.py       # ISOPLANE instrument configuration
├── io/                   # File Input/Output
│   └── image.py          # FITS handling and ImageFile class
├── preproc/              # Preprocessing algorithms
│   ├── make_im.py        # Image combination (Bias, Dark, LFlat)
│   └── preproc.py        # Basic CCD processing
├── tlm/                  # Tramline Map generation
│   ├── make_tlm.py       # Trace fitting and map creation
│   └── match_fibers.py   # Fiber matching logic
├── utils/                # Shared utilities
│   ├── args.py           # Argument parsing and validation
│   ├── fiber.py          # Fiber type overrides
│   └── plot.py           # Plotting helpers and emission-line data
└── wavecal/              # Wavelength Calibration core logic
    ├── arc_io.py         # Arc line list I/O
    ├── calibrate.py      # Polynomial fitting and calibration models
    ├── crosscorr.py      # Cross-correlation analysis
    ├── landmarks.py      # Landmark detection (peaks)
    └── wavelets.py       # Wavelet transforms for feature detection
```

## Module Details

### `kspecdr.io`
Handles all interactions with FITS files. It standardizes header information and manages data reading/writing, ensuring that other modules can work with consistent data structures.

### `kspecdr.preproc`
Responsible for the initial processing of raw CCD images.
*   **`make_im`**: Combines multiple frames into master calibration frames (Master Bias, Dark, Flat).
*   **`preproc`**: Applies bias subtraction, dark current correction, and flat-fielding to science or arc frames.

### `kspecdr.tlm`
Generates the Tramline Map (TLM), which defines the position of each fiber trace on the detector.
*   **`make_tlm`**: Fits Gaussian profiles to fiber traces to determine their centers and widths across the spectral axis.
*   **`match_fibers`**: Matches detected traces to physical fiber IDs, handling missing or broken fibers.

### `kspecdr.extract`
Performs spectral extraction and orchestrates high-level reduction steps.
*   **`make_ex`**: Extracts 1D spectra from 2D images using the TLM. Supports simple summation and will support optimal extraction.
*   **`reduce_arc`**: Manages the wavelength calibration workflow, calling `wavecal` functions to derive and apply the wavelength solution.
*   **`make_red`**: Handles final reduction steps like sky subtraction and flux calibration.

### `kspecdr.wavecal`
Contains the mathematical engines for wavelength calibration.
*   **`calibrate`**: Fits polynomial models to pixel-wavelength pairs.
*   **`landmarks`** & **`wavelets`**: Detect spectral features (arc lines) with sub-pixel precision.
*   **`crosscorr`**: Matches detected lines to reference templates using cross-correlation.

### `kspecdr.fluxcal`
Implements the full flux calibration pipeline. It provides data containers (`Spectrum1D`, `Photometry`, `FilterCurve`, `StellarTemplate`, `CalibrationVector`), photometric utilities for AB/Vega magnitude systems, BOSZ 2024 template library access, continuum normalization, template matching with radial-velocity fitting, calibration vector computation and combination, telluric/bad-region masks, and QC plotting. See the [Flux Calibration API](../reference/flux-calibration.md) and [design document](../development/fluxcal-design.md) for details.

### `kspecdr.utils`
Shared utilities used across the pipeline.
*   **`args`**: Normalizes and validates the dictionary-based configuration (`args`) used by all reduction entry points.
*   **`fiber`**: Parses and applies per-fiber type overrides (e.g., marking specific fibers as sky or calibration).
*   **`plot`**: Plotting helpers and an emission-line registry for annotating spectra.

### Top-level orchestration

*   **`reduce_object`**: End-to-end reduction of a raw science frame — calls preprocessing, extraction, wavelength calibration, sky subtraction, and (optionally) flux calibration.
*   **`reduce_fflat`**: Reduces a raw fibre-flat frame to produce a normalised flat-field used for throughput correction.
*   **`tracking`**: Multi-target tracking algorithm (LAP/Hungarian assignment) used by `make_tlm` to link per-column peak detections into continuous fibre traces.

### `kspecdr.inst`
Stores instrument-specific configurations. New instruments can be added here (e.g., `isoplane.py`), defining parameters like readout noise, gain, and detector format.

## Data Directory

The `data/` directory located at the root of the repository contains static resources required for reduction and documentation.

```text
data/
├── arc_tables/       # Reference arc line lists (e.g., ThAr, CuAr) used for wavelength calibration.
└── images/           # Images and logos used in the documentation.
```

## Related Reference Pages

Detailed API docs are organized under [API Reference](../reference/index.md):

- [Preprocessing](../reference/preprocessing.md)
- [Tramline](../reference/tramline.md)
- [Extraction](../reference/extraction.md)
- [Wavelength Calibration](../reference/wavelength-calibration.md)
- [Flux Calibration](../reference/flux-calibration.md)
- [Instrument](../reference/instrument.md)
- [Utilities](../reference/utilities.md)
