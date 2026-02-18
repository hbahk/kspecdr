## Filter Response Curves

All single-band files use the same two-column format (whitespace-separated, no header):

    wavelength(µm)   transmission

### Files

| File | Band | Source |
|---|---|---|
| `ps1_g.dat` | Pan-STARRS1 g | Tonry+ 2012, VizieR J/ApJ/750/99/table3 |
| `ps1_r.dat` | Pan-STARRS1 r | Tonry+ 2012 |
| `ps1_i.dat` | Pan-STARRS1 i | Tonry+ 2012 |
| `ps1_z.dat` | Pan-STARRS1 z | Tonry+ 2012 |
| `ps1_y.dat` | Pan-STARRS1 y | Tonry+ 2012 |
| `2mass_j.dat` | 2MASS J | 2MASS All-Sky Release |
| `2mass_h.dat` | 2MASS H | 2MASS All-Sky Release |
| `2mass_k.dat` | 2MASS Ks | 2MASS All-Sky Release |
| `gaia_g.dat` | Gaia DR2 G | Evans+ 2018 |
| `gaia_bp.dat` | Gaia DR2 BP | Evans+ 2018 |
| `gaia_rp.dat` | Gaia DR2 RP | Evans+ 2018 |

### References

* [2MASS Photometric system](https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html)
* [Pan-STARRS1 bandpasses](https://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/ApJ/750/99/table3)
* [Gaia DR2 passbands](https://www.cosmos.esa.int/web/gaia/iow_20180316) (Evans+ 2018)

### Raw source data

* `ps1.dat` — Original multi-band VizieR table from Tonry+ 2012. The
  `ps1_*.dat` files above were extracted by splitting each band column and
  converting wavelength from nm to µm. Only the wavelength range with non-zero
  transmission is kept (plus one zero-padding row on each end).
* `GaiaDR2_Passbands_ZeroPoints/` — Original Gaia DR2 passband and zero-point
  files from Evans+ 2018. The `gaia_*.dat` files above were extracted from
  `GaiaDR2_Passbands.dat` by selecting rows where the transmissivity is defined
  (sentinel value 99.99 excluded) and converting wavelength from nm to µm.
