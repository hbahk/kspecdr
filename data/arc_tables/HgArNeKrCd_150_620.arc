# K-SPEC arc line list v2, setup 150_620 (Isoplane SCT-320, 150 g/mm, centre 620 nm)
# Lamps on together: Newport-Oriel Hg(Ar) 6035, Ne 6032, Kr 6031; UVP Cd 90-0071-03
# Wavelengths: NIST ASD air wavelengths (observed, else Ritz), retrieved 2026-09-28
# Intensities: measured on the stacked commissioning arc (12 frames, nights 20260124-20260205), brightest unsaturated line = 1000;
#   blend components share the measured flux by NIST-predicted ratios
# Selection: >=5 sigma features, S/N >= 10, unique or blend components, second-order images at 2x lambda
#   lines that reduce_arc placed consistently off on the commissioning frames removed
# label=XX is the element (kspecdr read_arc_file); then species, class, S/N
# K-SPEC arc atlas v2 (2026-09-28): commissioning arcs of 2026 Jan-Feb stacked per setup;
#   lines identified against NIST ASD, checked with single-lamp lab frames and reduce_arc

* Hg lines
4046.5650        3.233   label=Hg  HgI unique S/N=161
4077.8370        0.231   label=Hg  HgI blend S/N=15
4358.3350       48.492   label=Hg  HgI blend S/N=308
5460.7500      192.695   label=Hg  HgI unique S/N=326
5769.6100        6.775   label=Hg  HgI blend S/N=300
5790.6700       31.507   label=Hg  HgI unique S/N=267
6149.4750      160.492   label=Hg  HgII unique S/N=319
6716.3400       58.165   label=Hg  HgI blend S/N=241
7944.5550        2.625   label=Hg  HgII unique S/N=138
8716.6700        1.620   label=Hg  HgI blend S/N=47 2nd-order image of 4358.3350

* Ar lines
8006.1570        0.775   label=Ar  ArI blend S/N=62
8014.7860        0.971   label=Ar  ArI blend S/N=62
8264.5220        2.500   label=Ar  ArI blend S/N=125
8424.6480        1.081   label=Ar  ArI blend S/N=83

* Ne lines
4080.1480        0.058   label=Ne  NeI blend S/N=15
4203.2700        0.245   label=Ne  NeI unique S/N=12
4422.5205        0.745   label=Ne  NeI blend S/N=13
5330.7775        0.282   label=Ne  NeI blend S/N=58
5341.0938        0.470   label=Ne  NeI blend S/N=58
5343.2834        0.282   label=Ne  NeI blend S/N=58
5656.6588        0.143   label=Ne  NeI blend S/N=13
5852.4878       56.231   label=Ne  NeI unique S/N=315
5872.8275        8.185   label=Ne  NeI blend S/N=252
5881.8950       16.370   label=Ne  NeI blend S/N=252
5944.8340       65.077   label=Ne  NeI unique S/N=314
5974.6273        8.347   label=Ne  NeI blend S/N=104
5975.5343        9.998   label=Ne  NeI blend S/N=104
6029.9968       22.930   label=Ne  NeI unique S/N=138
6074.3376       58.687   label=Ne  NeI unique S/N=197
6163.5937       53.833   label=Ne  NeI unique S/N=113
6217.2812       50.485   label=Ne  NeI unique S/N=201
6266.4952      101.044   label=Ne  NeI unique S/N=302
6304.7893       14.420   label=Ne  NeI blend S/N=97
6313.6855       21.630   label=Ne  NeI blend S/N=97
6334.4276       29.833   label=Ne  NeI blend S/N=232
6402.2480      235.975   label=Ne  NeI unique S/N=139
6506.5277      181.792   label=Ne  NeI unique S/N=240
6532.8824       88.221   label=Ne  NeI unique S/N=162
6598.9528      100.903   label=Ne  NeI unique S/N=310
6678.2766      161.413   label=Ne  NeI unique S/N=299
6717.0430       34.339   label=Ne  NeI blend S/N=241
6929.4672      162.280   label=Ne  NeI unique S/N=299
7024.0500      195.051   label=Ne  NeI blend S/N=311
7032.4128      486.947   label=Ne  NeI blend S/N=311
7173.9380       30.676   label=Ne  NeI unique S/N=116
7245.1665      308.457   label=Ne  NeI unique S/N=303
7438.8981       83.364   label=Ne  NeI unique S/N=318
7488.8712       12.470   label=Ne  NeI unique S/N=72
7535.7739       11.646   label=Ne  NeI blend S/N=93
7544.0439        5.405   label=Ne  NeI blend S/N=93
8259.3795        2.927   label=Ne  NeI blend S/N=125
8266.0769        6.403   label=Ne  NeI blend S/N=125
8300.3248       36.001   label=Ne  NeI blend S/N=271
8377.6070       29.833   label=Ne  NeI unique S/N=298
8418.4265        5.027   label=Ne  NeI blend S/N=83
8591.2583        3.118   label=Ne  NeI unique S/N=141
8634.6472        4.404   label=Ne  NeI unique S/N=163
8654.3828        5.727   label=Ne  NeI unique S/N=208
8679.4936        0.846   label=Ne  NeI blend S/N=73
8681.9216        0.978   label=Ne  NeI blend S/N=73
8780.6223       21.436   label=Ne  NeI blend S/N=291
8783.7539       16.171   label=Ne  NeI blend S/N=291

* Kr lines
4362.6416       24.428   label=Kr  KrI blend S/N=308
4410.3681        0.298   label=Kr  KrI blend S/N=13
4418.7613        0.298   label=Kr  KrI blend S/N=13
5649.5618        0.069   label=Kr  KrI blend S/N=13
6813.1088        0.619   label=Kr  KrI unique S/N=35
7601.5457      128.318   label=Kr  KrI unique S/N=316
7685.2459       11.032   label=Kr  KrI blend S/N=389
7694.5401       13.814   label=Kr  KrI blend S/N=389
7854.8233       15.391   label=Kr  KrI unique S/N=283
7913.4251        0.260   label=Kr  KrI blend S/N=15
7920.4700        0.208   label=Kr  KrI blend S/N=15
8059.5048        9.600   label=Kr  KrI unique S/N=34
8190.0566       31.358   label=Kr  KrI unique S/N=298
8263.2426        8.497   label=Kr  KrI blend S/N=125
8298.1099       14.848   label=Kr  KrI blend S/N=271
8725.2832        0.816   label=Kr  KrI blend S/N=47 2nd-order image of 4362.6416
8726.5400        0.367   label=Kr  KrI blend S/N=47

* Cd lines
4412.9894        0.186   label=Cd  CdI blend S/N=13
4415.6859        0.196   label=Cd  CdII blend S/N=13
4678.1493      344.493   label=Cd  CdI unique S/N=309
4799.9123      671.067   label=Cd  CdI unique S/N=322
5085.8217     1000.000   label=Cd  CdI unique S/N=332
6438.4695      598.323   label=Cd  CdI unique S/N=296
7345.6704        6.469   label=Cd  CdI unique S/N=224

