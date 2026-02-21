# Quick Start

This page gives the shortest path to produce:

- preprocessed IM files (`*_im.fits`)
- a tramline map (`*_tlm.fits`)
- extracted spectra (`*_ex.fits`)
- an arc-calibrated product (`*_red.fits`)

```{warning}
`kspecdr` is in active development. SUM extraction and arc wavelength calibration are usable now, while full science wrapper stages are still evolving.
```

## Minimal End-to-End Example

```python
from pathlib import Path
from kspecdr.inst.isoplane import write_isoplane_converted_image
from kspecdr.preproc.preproc import reduce_bias, reduce_dark
from kspecdr.preproc.make_im import make_im
from kspecdr.tlm.make_tlm import make_tlm
from kspecdr.extract.make_ex import make_ex
from kspecdr.extract.reduce_arc import reduce_arc

work = Path("work")
work.mkdir(exist_ok=True)

# 1) Convert one fiber-flat and one arc file (repeat similarly for other files)
write_isoplane_converted_image("raw/flat_raw.fits", work / "flat_conv.fits", ndfclass="MFFFF", n_fibers=14)
write_isoplane_converted_image("raw/arc_raw.fits", work / "arc_conv.fits", ndfclass="MFARC", n_fibers=14)
write_isoplane_converted_image("raw/bias1.fits", work / "bias1_conv.fits", ndfclass="BIAS", n_fibers=14)
write_isoplane_converted_image("raw/bias2.fits", work / "bias2_conv.fits", ndfclass="BIAS", n_fibers=14)
write_isoplane_converted_image("raw/dark1.fits", work / "dark1_conv.fits", ndfclass="DARK", n_fibers=14)
write_isoplane_converted_image("raw/dark2.fits", work / "dark2_conv.fits", ndfclass="DARK", n_fibers=14)

# 2) Build master bias/dark
mbias = reduce_bias([str(work / "bias1_conv.fits"), str(work / "bias2_conv.fits")], output_file=str(work / "mbias.fits"))
mdark = reduce_dark([str(work / "dark1_conv.fits"), str(work / "dark2_conv.fits")], output_file=str(work / "mdark.fits"), bias_filename=mbias)

# 3) Preprocess flat and build TLM
flat_im = make_im(raw_filename=str(work / "flat_conv.fits"), im_filename=str(work / "flat_im.fits"),
                  use_bias=True, bias_filename=mbias, use_dark=True, dark_filename=mdark)
make_tlm({"IMAGE_FILENAME": flat_im, "TLMAP_FILENAME": str(work / "flat_tlm.fits")})

# 4) Preprocess/extract arc and calibrate wavelength
arc_im = make_im(raw_filename=str(work / "arc_conv.fits"), im_filename=str(work / "arc_im.fits"),
                 use_bias=True, bias_filename=mbias, use_dark=True, dark_filename=mdark)
make_ex({"IMAGE_FILENAME": arc_im, "TLMAP_FILENAME": str(work / "flat_tlm.fits"),
         "EXTRAC_FILENAME": str(work / "arc_ex.fits"), "EXTR_OPERATION": "SUM", "SUM_WIDTH": 5.0})
reduce_arc({"RAW_FILENAME": str(work / "arc_conv.fits"), "IMAGE_FILENAME": str(work / "arc_im.fits"),
            "TLMAP_FILENAME": str(work / "flat_tlm.fits"), "EXTRAC_FILENAME": str(work / "arc_ex.fits"),
            "OUTPUT_FILENAME": str(work / "arc_red.fits"), "LAMPNAME": "hgar",
            "ARCDIR": str(Path("data/arc_tables")), "USE_GENCAL": True})
```

## Next Steps

- End-to-end workflow tutorial: [Isoplane end-to-end](../tutorials/isoplane-end-to-end.md)
- Task-focused recipes: [How-to guides](../how-to/index.md)
- Module-level API details: [API reference](../reference/index.md)
