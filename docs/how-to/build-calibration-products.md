# Build Calibration Products

Use this guide to create master calibration files and a tramline map.

## 1. Build master bias and dark

```python
from kspecdr.preproc.preproc import reduce_bias, reduce_dark

master_bias = reduce_bias(
    raw_files=["work/bias_001_converted.fits", "work/bias_002_converted.fits"],
    output_file="work/mbias.fits",
)

master_dark = reduce_dark(
    raw_files=["work/dark_300s_001_converted.fits", "work/dark_300s_002_converted.fits"],
    output_file="work/mdark.fits",
    bias_filename=master_bias,
)
```

## 2. Preprocess a fiber flat

```python
from kspecdr.preproc.make_im import make_im

flat_im = make_im(
    raw_filename="work/flat_001_converted.fits",
    im_filename="work/flat_001_im.fits",
    use_bias=True,
    bias_filename=master_bias,
    use_dark=True,
    dark_filename=master_dark,
)
```

## 3. Generate a tramline map

```python
from kspecdr.tlm.make_tlm import make_tlm

make_tlm(
    {
        "IMAGE_FILENAME": flat_im,
        "TLMAP_FILENAME": "work/flat_001_tlm.fits",
    }
)
```

## 4. Quick validation

- `*_tlm.fits` contains plausible traces for all expected fibers
- `WAVELA` extension is present
- no obvious crossing/jumping traces in detector edges
