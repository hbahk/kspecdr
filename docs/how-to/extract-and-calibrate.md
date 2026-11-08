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

### Optimal extraction (OPTEX)

`EXTR_OPERATION="OPTEX"` fits the Gaussian profiles of all fibres plus a scattered-light
term column by column (port of 2dfdr OPTEX). It needs the profile widths of the fibre flat:
make the tramline map from the flat with `SIGPROF=True` (writes the `SIGMAPRF` HDU and the
measured `MWIDTH`) and pass `SIGPROF=True` to `make_ex` as well. OPTEX is sensitive to
trace-centre errors: a trace off by a fraction of a pixel already loses flux. Object, sky
and arc frames whose traces have moved relative to the flat should be extracted with
`MSR_TLM_CREEP=True`, which measures the shift against the tramline map (written to the
`TLMSHIFT` header keyword); `TLM_SHIFT` sets it by hand and is the fallback when the
measurement fails, `TLM_INIT_SHIFT` centres the ±2.5 px search.

```python
make_tlm({"IMAGE_FILENAME": "work/flat_001_im.fits",
          "TLMAP_FILENAME": "work/flat_001_tlm.fits", "SIGPROF": True})

make_ex(
    {
        "IMAGE_FILENAME": "work/object_001_im.fits",
        "TLMAP_FILENAME": "work/flat_001_tlm.fits",
        "EXTRAC_FILENAME": "work/object_001_ex.fits",
        "EXTR_OPERATION": "OPTEX",
        "SIGPROF": True,
        "MSR_TLM_CREEP": True,  # measure the trace shift against the flat's TLM
        "OPTEX_NSLPARS": 1,  # constant scattered-light term per column
        "OPTEX_CLIP": 5.0,  # reject cosmic rays and hot pixels (object frames only)
        "OPTEX_NITER": 4,
    }
)
```

Keep `OPTEX_CLIP` off for flats and arcs: their profiles differ from the Gaussian by more
than the noise, so the rejection would remove real light. `OPTEX_MKRES=True` writes
`work/object_001_res.fits` with the residual, model and rejected-pixel images.

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

## See also

- [Extraction API reference](../reference/extraction.md)
- [Wavelength Calibration API reference](../reference/wavelength-calibration.md)
- [Arc calibration theory](../explanations/isoplane-theory.md)
