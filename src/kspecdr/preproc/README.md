# MakeIM - Raw to IM File Conversion

This module provides functionality to convert raw astronomical data files to IM (Intermediate) format, based on the Fortran `MAKE_IM` routine from 2dfdr.

## Features

- **Raw to IM conversion**: Convert raw data files to IM format with proper FITS structure
- **Bias subtraction**: Subtract bias frames from raw data
- **Dark subtraction**: Subtract dark frames from raw data  
- **Flat fielding**: Divide by flat field frames
- **Bad pixel masking**: Apply bad pixel masks and mark saturated pixels
- **Cosmic ray removal**: Remove cosmic rays using astroscrappy (LACosmic algorithm)
- **Fiber table handling**: Copy and process fiber tables from raw files
- **Instrument-specific processing**: Handle different instruments (TAIPAN, 6DF, KOALA, etc.)

## Installation

### Dependencies

The module requires the following packages:
- `astropy` - For FITS file handling
- `numpy` - For numerical operations
- `astroscrappy` - For cosmic ray removal (optional)

### Installing astroscrappy

For cosmic ray removal functionality, install astroscrappy:

```bash
pip install astroscrappy
```

## Usage

### Basic Usage

```python
from kspecdr.preproc.make_im import make_im

# Basic raw to IM conversion
im_file = make_im(
    raw_filename="data/raw_taipan.fits",
    im_filename="data/taipan_im.fits"
)
```

### Complete Reduction Pipeline

```python
# Full reduction with all steps
im_file = make_im(
    raw_filename="data/raw_taipan.fits",
    im_filename="data/taipan_complete_im.fits",
    use_bias=True,
    bias_filename="data/bias_taipan.fits",
    use_dark=True,
    dark_filename="data/dark_taipan.fits", 
    use_flat=True,
    flat_filename="data/flat_taipan.fits",
    bad_pixel_mask="data/bad_pixels.fits",
    mark_saturated=True,
    cosmic_ray_method="LACOSMIC"
)
```

### Using the MakeIM Class

```python
from kspecdr.preproc.make_im import MakeIM

# Create processor instance
processor = MakeIM(verbose=True)

# Process with custom parameters
im_file = processor.process_raw_to_im(
    raw_filename="data/raw_taipan.fits",
    im_filename="data/taipan_im.fits",
    cosmic_ray_method="LACOSMIC",
    sigclip=4.0,
    sigfrac=0.25
)
```

## Parameters

### Main Parameters

- `raw_filename` (str): Path to the raw input file
- `im_filename` (str, optional): Path for output IM file (default: derived from raw_filename)
- `use_bias` (bool): Whether to subtract bias frame (default: False)
- `use_dark` (bool): Whether to subtract dark frame (default: False)
- `dark_filename` (str): Path to dark frame file
- `use_flat` (bool): Whether to divide by flat field (default: False)
- `flat_filename` (str): Path to flat field file
- `bad_pixel_mask` (str): Path to bad pixel mask file
- `bad_pixel_mask2` (str): Path to second bad pixel mask file
- `mark_saturated` (bool): Whether to mark saturated pixels as bad (default: True)
- `cosmic_ray_method` (str): Method for cosmic ray removal (default: 'NONE')
- `verbose` (bool): Whether to print verbose output (default: True)

### Cosmic Ray Removal Parameters

When using `cosmic_ray_method="LACOSMIC"`:

- `sigclip` (float): Sigma clipping threshold (default: 4.5)
- `sigfrac` (float): Fraction of sigma clipping (default: 0.3)
- `objlim` (float): Object limit (default: 5.0)
- `gain` (float): Gain value (default: auto-detect)
- `readnoise` (float): Read noise (default: auto-detect)
- `satlevel` (float): Saturation level (default: auto-detect)
- `niter` (int): Number of iterations (default: 4)
- `sepmed` (bool): Use median for separation (default: True)
- `cleantype` (str): Clean type (default: "meanmask")
- `fsmode` (str): FSM mode (default: "median")
- `psfmodel` (str): PSF model (default: "gauss")
- `psffwhm` (float): PSF FWHM (default: 2.5)
- `psfsize` (int): PSF size (default: 7)
- `psfbeta` (float): PSF beta (default: 4.765)

## Fiber Table Handling

The module automatically handles fiber tables during conversion:

### Automatic Fiber Table Copying

When converting raw files that contain fiber tables (FIBRES or FIBRES_IFU HDUs), the fiber table is automatically copied to the IM file:

```python
# Fiber table is copied automatically if present in raw file
im_file = make_im(
    raw_filename="data/raw_with_fibers.fits",
    im_filename="data/im_with_fibers.fits"
)
```

### TAIPAN Fiber Table Processing

For TAIPAN instruments, the module automatically:
- Limits fiber tables to 150 fibers (removes fibers beyond 150)
- Removes TAIPAN-specific keywords (N_FIBRES, FIBRES_IFU)

```python
# TAIPAN fiber table processing is automatic
im_file = make_im(
    raw_filename="data/raw_taipan.fits",
    im_filename="data/taipan_im.fits"
)
```

### Manual Fiber Table Handling

You can also handle fiber tables manually using the ImageFile class:

```python
from kspecdr.io.image import ImageFile

# Check for fiber table
with ImageFile("data/im_file.fits", mode='READ') as im:
    if im.has_fiber_table():
        fiber_data = im.read_fiber_table()
        table_name = im.get_fiber_table_name()
        print(f"Found fiber table '{table_name}' with {len(fiber_data)} fibers")

# Copy fiber table from another file
with ImageFile("source.fits", mode='READ') as source:
    with ImageFile("dest.fits", mode='UPDATE') as dest:
        dest.copy_fiber_table_from(source)

# Remove fibers beyond a limit (for TAIPAN)
with ImageFile("taipan_im.fits", mode='UPDATE') as im:
    im.remove_fibers_beyond(150)
```

## Instrument-Specific Processing

The module handles different instruments automatically:

### TAIPAN
- Transposes image axes
- Processes fiber tables (limits to 150 fibers)
- Removes TAIPAN-specific keywords

### 6DF  
- Transposes and reverses spectral axis
- Handles 6DF-specific transformations

### KOALA
- Flips spatial axis
- Handles KOALA-specific transformations

## Output Files

IM files are created with the following structure:

- **Primary HDU**: Image data (32-bit float)
- **VARIANCE HDU**: Variance data (if created from raw)
- **FIBRES/FIBRES_IFU HDU**: Fiber table (if copied from raw)

Key header modifications:
- `BITPIX = -32` (32-bit float)
- `CLASS` keyword set to frame class
- `BSCALE`, `BZERO`, `AVGVALUE` keywords removed
- Processing history added

## Error Handling

The module includes comprehensive error handling:
- File existence checks
- Format validation
- Processing step validation
- Detailed logging and error messages

## Examples

See `example_usage.py` for complete examples demonstrating:
- Basic conversion
- Bias/dark subtraction
- Flat fielding
- Bad pixel masking
- Cosmic ray removal
- Fiber table handling
- Complete reduction pipeline

## Based On

This module is based on the Fortran `MAKE_IM` routine from the 2dfdr data reduction package, modernized for Python with additional features and improved error handling. 