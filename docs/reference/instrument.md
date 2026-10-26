# Instrument API

Instrument-specific configuration: header conversion, readout settings, and fibre-table management. Currently supports the ISOPLANE spectrograph. See the [Isoplane theory](../explanations/isoplane-theory.md) for instrument details.

```{eval-rst}
.. automodule:: kspecdr.inst.isoplane
   :members:
```

## Fibre map

The fibre map in `data/fibers/` links the 73 positioner holes (number and label) and the fixed
sky-fibre holes to the Isoplane slit. `make_fiber_table` joins it to an ICS `.assign.txt` file
and returns the `FIBRES` table in slit order for `write_isoplane_converted_image`.

```{eval-rst}
.. automodule:: kspecdr.inst.fibermap
   :members:
```
