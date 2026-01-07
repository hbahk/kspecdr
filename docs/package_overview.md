# Package Overview

`kspecdr` is designed to be modular, separating instrument definitions, file I/O, and reduction algorithms.

## Module Structure

*   `kspecdr.io`: Handles FITS file input/output, including header parsing and standardization.
*   `kspecdr.preproc`: Preprocessing steps (bias, dark, flat handling, image combination).
*   `kspecdr.tlm`: Tramline map generation (trace identification and fitting).
*   `kspecdr.extract`: Spectral extraction routines (Sum, Optimal) and orchestration of reduction steps (including arc reduction).
*   `kspecdr.wavecal`: Underlying wavelength calibration algorithms and utilities (used by `reduce_arc`).
*   `kspecdr.inst`: Instrument-specific definitions and constants.

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
