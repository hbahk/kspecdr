# Extract and Calibrate Spectra

Use this guide to extract spectra and compute wavelength calibration from arcs.

## 1. Preprocess and extract an arc

```python
from kspecdr.preproc.make_im import make_im
from kspecdr.extract.make_ex import make_ex

arc_im = make_im(
    raw_filename="work/hgar_001_converted.fits",
    im_filename="work/hgar_001_im.fits",
    use_bias=True,
    bias_filename="work/mbias.fits",
    use_dark=True,
    dark_filename="work/mdark.fits",
)

make_ex(
    {
        "IMAGE_FILENAME": arc_im,
        "TLMAP_FILENAME": "work/flat_001_tlm.fits",
        "EXTRAC_FILENAME": "work/hgar_001_ex.fits",
        "EXTR_OPERATION": "SUM",
        "SUM_WIDTH": 5.0,
    }
)
```

## 2. Run arc wavelength calibration

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
    }
)
```

For multi-lamp fitting, use `reduce_arcs`.

## 3. Apply arc solution to science spectra

```python
import shutil
from kspecdr.wavecal.scrunch import scrunch_from_arc_id

shutil.copy2("work/object_001_ex.fits", "work/object_001_red.fits")
scrunch_from_arc_id(
    obj_filename="work/object_001_red.fits",
    arc_filename="work/hgar_001_red.fits",
    args={"WAVEL_FILENAME": "work/hgar_001_red.fits"},
    reverse=False,
)
```

## 4. Output checks

- arc RED file includes `WAVELA` and `SHIFTS`
- science red file has `SCRUNCH=True` after rebinning
