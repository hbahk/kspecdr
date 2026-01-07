# Quick Start

This guide provides a quick introduction to running the `kspecdr` pipeline.

```{warning}
The pipeline is currently in development. Not all reduction steps are fully implemented.
Currently, **Preprocessing**, **Tramline Map Generation**, **Extraction (Sum)**, and **Wavelength Calibration** are functional.
```

## Basic Usage

The pipeline is typically run via Python scripts or interactive sessions.

### 1. Preprocessing

Convert raw FITS files to `_im.fits` format, applying bias and dark subtraction if available.

```python
from kspecdr.preproc import make_im, reduce_bias, reduce_dark

# Reduce Bias Frames
master_bias = reduce_bias(['bias1.fits', 'bias2.fits'], output_file='BIAScombined.fits')

# Reduce Dark Frames
master_dark = reduce_dark(['dark1.fits', 'dark2.fits'], bias_file=master_bias, output_file='DARKcombined.fits')

# Process Science Frame
im_file = make_im(
    raw_filename='science_raw.fits',
    im_filename='science_im.fits',
    bias_filename=master_bias,
    dark_filename=master_dark,
    use_bias=True,
    use_dark=True
)
```

### 2. Tramline Map Generation

Generate a tramline map (`_tlm.fits`) identifying fiber positions.

```python
from kspecdr.tlm import make_tlm

args = {
    'IMAGE_FILENAME': 'science_im.fits',
    'TLMAP_FILENAME': 'science_tlm.fits',
    'INST_CODE': 99 # Example code for ISOPLANE
}

make_tlm(args)
```

### 3. Extraction

Extract spectra from the image using the tramline map.

```python
from kspecdr.extract import make_ex

args = {
    'IMAGE_FILENAME': 'science_im.fits',
    'TLMAP_FILENAME': 'science_tlm.fits',
    'EXTRAC_FILENAME': 'science_ex.fits',
    'EXTR_OPERATION': 'SUM',  # Only SUM is fully supported currently
    'SUM_WIDTH': 5.0
}

make_ex(args)
```

### 4. Wavelength Calibration

Reduce an arc frame to determine the wavelength solution.

```python
from kspecdr.extract import reduce_arc

# Assuming you have an extracted arc file or run full reduction
reduce_args = {
    'RAW_FILENAME': 'arc_raw.fits',
    'IMAGE_FILENAME': 'arc_im.fits',
    'TLMAP_FILENAME': 'science_tlm.fits', # Use science TLM or generate new
    'EXTRAC_FILENAME': 'arc_ex.fits',
    'OUTPUT_FILENAME': 'arc_red.fits',
    'USE_GENCAL': True
}

reduce_arc(reduce_args)
```

```{note}
For full pipeline execution, typically a higher-level script (like `reduce_run` in 2dfdr) would orchestrate these calls. This wrapper is currently under development.
```
