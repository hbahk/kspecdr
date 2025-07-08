# Preprocessing Module

This module provides preprocessing functionality for astronomical data reduction, specifically for converting raw instrument data into calibrated IM (Image) files.

## Overview

The preprocessing module implements the equivalent of the Fortran `MAKE_IM` routine from 2dfdr, performing all necessary preprocessing steps to convert raw instrument data into calibrated image files with proper variance estimates.

## Features

- **Raw to IM conversion**: Copy raw files to IM files with proper FITS structure
- **Bad pixel handling**: Mark bad pixels using masks or saturation detection
- **Bias subtraction**: Process overscan regions and subtract bias frames
- **Dark frame subtraction**: Optional dark frame removal with proper scaling
- **Variance calculation**: Automatic variance HDU creation with proper error propagation
- **Flat fielding**: Optional long-slit flat field application
- **Cosmic ray removal**: Multiple algorithms for cosmic ray detection and removal

## Usage

### Basic Usage

```python
from kspecdr.preproc import make_im

# Basic processing - just create IM file with variance
im_filename = make_im(
    raw_filename="example_raw.fits",
    verbose=True
)
```

### Full Processing

```python
# Full processing with all options
im_filename = make_im(
    raw_filename="example_raw.fits",
    use_bias=True,
    use_dark=True,
    dark_filename="master_dark.fits",
    use_flat=True,
    flat_filename="master_flat.fits",
    bad_pixel_mask="bad_pixels.fits",
    mark_saturated=True,
    cosmic_ray_method='LACOSMIC',
    verbose=True
)
```

### Using the Class Directly

```python
from kspecdr.preproc import MakeIM

# Create processor instance
processor = MakeIM(verbose=True)

# Process with custom parameters
im_filename = processor.process_raw_to_im(
    raw_filename="example_raw.fits",
    use_bias=True,
    mark_saturated=True,
    cosmic_ray_method='NONE'
)
```

## Processing Steps

The module performs the following steps in order:

1. **Copy raw file to IM file**: Creates the output IM file by copying the raw file
2. **Mark bad pixels**: Applies bad pixel masks and marks saturated pixels
3. **Process overscan and bias**: Subtracts bias and processes overscan regions
4. **Subtract dark frame**: Optionally removes dark current (with proper scaling)
5. **Create variance HDU**: Calculates and initializes variance estimates
6. **Apply flat field**: Optionally divides by long-slit flat field
7. **Remove cosmic rays**: Optionally removes cosmic rays using various algorithms

## Variance Calculation

The variance is calculated using the formula:

```
σ² = (readout_noise)² + (signal/gain)
```

The module supports different amplifier configurations:
- **Single amplifier**: Same noise/gain for entire image
- **Two amplifiers**: Different noise/gain for left/right halves
- **Four amplifiers**: Different noise/gain for each quadrant

## File Naming Convention

- **Input**: `raw_file.fits` (original raw data)
- **Output**: `raw_file_im.fits` (processed image file)
- **Variance**: Stored as VARIANCE HDU within the IM file

## Parameters

### Main Parameters

- `raw_filename` (str): Path to the raw input file
- `im_filename` (str, optional): Path for output IM file (default: derived from raw_filename)
- `use_bias` (bool): Whether to subtract bias frame (default: False)
- `use_dark` (bool): Whether to subtract dark frame (default: False)
- `dark_filename` (str): Path to dark frame file
- `use_flat` (bool): Whether to divide by long-slit flat field (default: False)
- `flat_filename` (str): Path to flat field file
- `bad_pixel_mask` (str): Path to bad pixel mask file
- `bad_pixel_mask2` (str): Path to second bad pixel mask file (e.g., cosmic ray mask)
- `mark_saturated` (bool): Whether to mark saturated pixels as bad (default: True)
- `cosmic_ray_method` (str): Method for cosmic ray removal:
  - 'NONE': No cosmic ray removal
  - 'LACOSMIC': Use LACosmic algorithm
  - 'BCLEAN': Use BCLEAN algorithm
  - 'PYCOSMIC': Use Python cosmic ray removal
- `verbose` (bool): Whether to print verbose output (default: True)

## Error Handling

The module includes comprehensive error handling:
- File existence checks
- Dimension validation
- Proper error propagation in variance calculations
- Warning messages for potential issues

## Dependencies

- `numpy`: Numerical computations
- `astropy`: FITS file handling
- `pathlib`: File path operations
- `logging`: Logging functionality

## Example

See `example_usage.py` for complete usage examples.

## Notes

- Based on the Fortran `MAKE_IM` routine from 2dfdr
- All processing steps are optional and can be controlled via parameters
- Variance HDU is automatically created and initialized
- Error handling and logging are built into the module
- Supports multiple amplifier configurations
- Includes proper error propagation for all operations 