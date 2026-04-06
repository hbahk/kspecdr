# Prepare Isoplane Inputs

Use this guide to convert raw Isoplane FITS files into `kspecdr`-ready inputs.

## 1. Convert raw headers and orientation

```python
from kspecdr.inst.isoplane import write_isoplane_converted_image

write_isoplane_converted_image(
    fpath="raw/flat_001.fits",
    output_fpath="work/flat_001_converted.fits",
    ndfclass="MFFFF",
    n_fibers=14,
)
```

For multi-frame cubes, use `split_frames=True` when needed.

## 2. Set frame class (`ndfclass`) consistently

Common classes in current workflows:

- `MFFFF` for fiber flats
- `MFARC` for arc lamps
- `MFOBJECT` for science/object
- `BIAS` and `DARK` for calibrations

## 3. Validate converted products

Before reduction, check:

- required keys exist (`INSTRUME`, `RO_GAIN`, `RO_NOISE`, `LAMBDAC`, `DISPERS`, `DISPAXIS`)
- a `FIBRES` table exists
- output is 2D and orientation is correct for your setup

## See also

- [Instrument API reference](../reference/instrument.md)
- [Isoplane theory](../explanations/isoplane-theory.md)
