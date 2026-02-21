# Isoplane Data Reduction Workflow

This document describes the practical, code-level reduction flow in `kspecdr` for PIXIS/Isoplane data.
It is focused on what is implemented in the current codebase, with explicit notes where wrappers are still incomplete.

## 1. Scope and Current Status

Implemented and usable now:

- Isoplane header conversion and orientation normalization (`write_isoplane_converted_image`).
- IM preprocessing (`make_im`): bad/saturated pixel masking, bias/dark correction, variance initialization, optional lflat and cosmic-ray steps.
- Calibration frame combination (`reduce_bias`, `reduce_dark`, `reduce_lflat`, `reduce_fflat`, `reduce_arc` in `preproc.preproc`).
- Tramline map generation (`make_tlm`) for `INSTRUME=ISOPLANE`.
- Spectral extraction (`make_ex`) through the `SUM` path (`EXTR_OPERATION="SUM"` or `"TRAM"` currently use the sum extraction routine).
- Arc wavelength calibration (`extract.reduce_arc.reduce_arc` and `reduce_arcs`).
- Rebinning onto an arc wavelength solution (`wavecal.scrunch.scrunch_from_arc_id`).

Not fully implemented yet:

- Full science wrapper pipeline in `reduce_object` and several `make_red`/sky/flux stages.
- `GAUSS` and `OPTEX` extraction modes in `make_ex`.

## 2. Input Requirements

For each raw Isoplane FITS file:

- Header must contain enough Isoplane metadata for conversion (`DATE-OBS`, spectrometer/grating keys, orientation keys).
- A fiber table must exist after conversion (either generated with `n_fibers` or supplied via `fiber_table`).

Typical raw categories:

- `BIAS`
- `DARK`
- Fiber flat (`MFFFF`)
- Arc (`MFARC`)
- Science/object (`MFOBJECT`)

## 3. Output Products (by stage)

The standard progression in this repository is:

- `*_converted.fits`: Isoplane raw file converted to `kspecdr`-compatible headers and orientation.
- `*_im.fits`: preprocessed image file (float image + `VARIANCE` HDU).
- `*_tlm.fits`: tramline map from a fiber flat (`PRIMARY` tramlines + predicted `WAVELA`).
- `*_ex.fits`: extracted spectra (`PRIMARY`, `VARIANCE`, optional `WAVELA`, `FIBRES`).
- `*_red.fits`: reduced arc output (copy of `EX` with calibrated `WAVELA` and `SHIFTS` written by `reduce_arc`).

## 4. Detailed Step-by-Step Workflow

### Step 1. Convert Isoplane raw files

Use `write_isoplane_converted_image` for every raw file before reduction.
This function:

- Standardizes Isoplane header keys (`INSTRUME`, `RO_GAIN`, `RO_NOISE`, `LAMBDAC`, `DISPERS`, `DISPAXIS`, etc.).
- Adds a `FIBRES` table.
- Handles 2D/3D input and optional frame splitting.
- Applies orientation flips and writes normalized 2D output.

```python
from pathlib import Path
from kspecdr.inst.isoplane import write_isoplane_converted_image

raw_file = Path("raw/flat_001.fits")
converted_file = Path("work/flat_001_converted.fits")

write_isoplane_converted_image(
    fpath=str(raw_file),
    output_fpath=str(converted_file),
    ndfclass="MFFFF",
    n_fibers=14,   # or use fiber_table=...
)
```

For multi-frame cubes, use `split_frames=True` and combine later if needed.

### Step 2. Build master bias and dark

Run `make_im` on converted calibration files and combine:

- Bias: `reduce_bias`
- Dark: `reduce_dark` (optionally grouped by exposure time)

```python
from kspecdr.preproc.preproc import reduce_bias, reduce_dark

master_bias = reduce_bias(
    raw_files=[
        "work/bias_001_converted.fits",
        "work/bias_002_converted.fits",
        "work/bias_003_converted.fits",
    ],
    output_file="work/mbias.fits",
)

master_dark = reduce_dark(
    raw_files=[
        "work/dark_300s_001_converted.fits",
        "work/dark_300s_002_converted.fits",
    ],
    output_file="work/mdark.fits",
    bias_filename=master_bias,
)
```

### Step 3. Preprocess a fiber flat and generate TLM

Create an IM from a converted fiber flat, then derive tramlines.

```python
from kspecdr.preproc.make_im import make_im
from kspecdr.tlm.make_tlm import make_tlm

flat_im = make_im(
    raw_filename="work/flat_001_converted.fits",
    im_filename="work/flat_001_im.fits",
    use_bias=True,
    bias_filename=master_bias,
    use_dark=True,
    dark_filename=master_dark,
    cosmic_ray_method="NONE",
)

make_tlm(
    {
        "IMAGE_FILENAME": flat_im,
        "TLMAP_FILENAME": "work/flat_001_tlm.fits",
    }
)
```

Isoplane tramline behavior in current code:

- Polynomial trace order defaults to `2`.
- Peak search method defaults to wavelet mode.
- Fiber matching is currently simple 1-to-1 (`match_fibers_isoplane`).
- Missing fibers are interpolated/extrapolated.
- Predicted `WAVELA` is written from `LAMBDAC` + `DISPERS`.

### Step 4. Preprocess and extract arc spectra

Each arc frame should be converted, preprocessed, and extracted with the same TLM.

```python
from kspecdr.preproc.make_im import make_im
from kspecdr.extract.make_ex import make_ex

arc_im = make_im(
    raw_filename="work/hgar_001_converted.fits",
    im_filename="work/hgar_001_im.fits",
    use_bias=True,
    bias_filename=master_bias,
    use_dark=True,
    dark_filename=master_dark,
)

make_ex(
    {
        "IMAGE_FILENAME": arc_im,
        "TLMAP_FILENAME": "work/flat_001_tlm.fits",
        "EXTRAC_FILENAME": "work/hgar_001_ex.fits",
        "EXTR_OPERATION": "SUM",   # "TRAM" currently goes through same SUM routine
        "SUM_WIDTH": 5.0,
    }
)
```

### Step 5. Calibrate wavelength from arcs

Use `reduce_arc` for one lamp, or `reduce_arcs` for multi-lamp/global fitting.
Arc line tables are read from `<ARCDIR>/<LAMPNAME>.arc` (for example `hgar.arc`, `ne.arc`, `kr.arc`, `cd.arc`).

```python
from pathlib import Path
from kspecdr.extract.reduce_arc import reduce_arc

reduce_arc(
    {
        "RAW_FILENAME": "work/hgar_001_converted.fits",
        "IMAGE_FILENAME": "work/hgar_001_im.fits",
        "TLMAP_FILENAME": "work/flat_001_tlm.fits",
        "EXTRAC_FILENAME": "work/hgar_001_ex.fits",
        "OUTPUT_FILENAME": "work/hgar_001_red.fits",
        "LAMPNAME": "hgar",
        "ARCDIR": str(Path("data/arc_tables")),
        "USE_GENCAL": True,
    },
    get_diagnostic=True,
    diagnostic_dir=Path("work/diagnostic"),
)
```

What `reduce_arc` currently does:

- Ensures IM and EX exist (creates if missing).
- Copies EX to RED.
- Runs generic calibration for Isoplane (`INST_ISOPLANE` path).
- Chooses a reference fiber, builds landmark shifts, cross-correlates against lamp tables, fits a robust polynomial model, propagates to all fibers.
- Writes calibrated `WAVELA` and `SHIFTS`.

### Step 6. Preprocess and extract science/object data

```python
from kspecdr.preproc.make_im import make_im
from kspecdr.extract.make_ex import make_ex

sci_im = make_im(
    raw_filename="work/object_001_converted.fits",
    im_filename="work/object_001_im.fits",
    use_bias=True,
    bias_filename=master_bias,
    use_dark=True,
    dark_filename=master_dark,
)

make_ex(
    {
        "IMAGE_FILENAME": sci_im,
        "TLMAP_FILENAME": "work/flat_001_tlm.fits",
        "EXTRAC_FILENAME": "work/object_001_ex.fits",
        "EXTR_OPERATION": "SUM",
        "SUM_WIDTH": 5.0,
    }
)
```

### Step 7. Apply arc wavelength solution to science spectra

Because the full `reduce_object` flow is not complete, the practical current step is to rebin science EX spectra using an arc RED file:

```python
import shutil
from kspecdr.wavecal.scrunch import scrunch_from_arc_id

shutil.copy2("work/object_001_ex.fits", "work/object_001_red.fits")

scrunch_from_arc_id(
    obj_filename="work/object_001_red.fits",
    arc_filename="work/hgar_001_red.fits",   # or a multi-lamp calibrated arc RED
    args={"WAVEL_FILENAME": "work/hgar_001_red.fits"},
    reverse=False,
)
```

This modifies the object file in place and sets `SCRUNCH=True`.

## 5. Quality Control Checklist

Before trusting extracted data, check:

- Converted files contain expected Isoplane keys (`INSTRUME`, `RO_GAIN`, `RO_NOISE`, `LAMBDAC`, `DISPERS`, `DISPAXIS`).
- `FIBRES` table exists and row count matches expected fiber count.
- TLM file has reasonable traces across all fibers and includes `WAVELA`.
- Arc RED file has calibrated `WAVELA` and `SHIFTS` HDUs after `reduce_arc`/`reduce_arcs`.
- Science extraction uses the same TLM and reduction settings as arcs.

## 6. Practical Notes

- `make_im` supports optional cosmic-ray cleaning with `LACOSMIC` when `astroscrappy` is installed.
- `reduce_dark` may return a list of files when input darks contain multiple exposure-time groups.
- Lamp names are file-name driven in `read_arc_file`; pass the base name matching your `.arc` files.
- For Isoplane commissioning runs, this manual staged workflow is currently the reliable path.
