# Package Overview

`kspecdr` is a Python-based data reduction pipeline for K-SPEC, designed to be modular and extensible. It separates instrument definitions, file I/O, core algorithms, and orchestration logic.

## Project Structure

The package is organized as follows:

```text
src/kspecdr/
├── constants.py          # Global constants and definitions
├── extract/              # Spectral extraction and reduction orchestration
│   ├── make_ex.py        # Extraction routines (Sum, Tramline)
│   ├── make_red.py       # Final reduction steps
│   └── reduce_arc.py     # Arc reduction workflow
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

### `kspecdr.inst`
Stores instrument-specific configurations. New instruments can be added here (e.g., `isoplane.py`), defining parameters like readout noise, gain, and detector format.

## Data Directory

The `data/` directory located at the root of the repository contains static resources required for reduction and documentation.

```text
data/
├── arc_tables/       # Reference arc line lists (e.g., ThAr, CuAr) used for wavelength calibration.
└── images/           # Images and logos used in the documentation.
```

## API Reference

### Preprocessing

```{eval-rst}
.. automodule:: kspecdr.preproc.make_im
   :members:

.. automodule:: kspecdr.preproc.preproc
   :members:
```

### Tramline Map

```{eval-rst}
.. automodule:: kspecdr.tlm.make_tlm
   :members:
```

### Extraction & Reduction

```{eval-rst}
.. automodule:: kspecdr.extract.make_ex
   :members:

.. automodule:: kspecdr.extract.make_red
   :members:
```

### Wavelength Calibration

The high-level orchestration for arc reduction is located in `kspecdr.extract`, while the core calibration logic resides in `kspecdr.wavecal`.

```{eval-rst}
.. automodule:: kspecdr.extract.reduce_arc
   :members:
```
