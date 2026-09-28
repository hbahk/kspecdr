# K-SPEC arc line list v2, setup 600_680 (Isoplane SCT-320, 600 g/mm, centre 680 nm)
# Lamps on together: Newport-Oriel Hg(Ar) 6035, Ne 6032, Kr 6031; UVP Cd 90-0071-03
# Wavelengths: NIST ASD air wavelengths (observed, else Ritz), retrieved 2026-09-28
# Intensities: measured on the stacked commissioning arc (14 frames, nights 20260124-20260205), brightest unsaturated line = 1000;
#   blend components share the measured flux by NIST-predicted ratios
# Selection: >=5 sigma features, S/N >= 10, unique or blend components, second-order images at 2x lambda
#   lines that reduce_arc placed consistently off on the commissioning frames removed
# label=XX is the element (kspecdr read_arc_file); then species, class, S/N
# K-SPEC arc atlas v2 (2026-09-28): commissioning arcs of 2026 Jan-Feb stacked per setup;
#   lines identified against NIST ASD, checked with single-lamp lab frames and reduce_arc

* Hg lines
6716.3400      108.050   label=Hg  HgI blend S/N=314
6907.4600        0.594   label=Hg  HgI unique S/N=31

* Ar lines
6965.4310        2.518   label=Ar  ArI unique S/N=118
7428.5740        2.397   label=Ar  ArII unique S/N=126

* Ne lines
6217.2812       46.244   label=Ne  NeI unique S/N=319
6266.4952      102.626   label=Ne  NeI unique S/N=316
6304.7893       41.431   label=Ne  NeI unique S/N=317
6334.4276      138.940   label=Ne  NeI unique S/N=311
6382.9914      237.637   label=Ne  NeI unique S/N=315
6402.2480      386.290   label=Ne  NeI unique S/N=322
6506.5277      261.723   label=Ne  NeI unique S/N=322
6532.8824      121.778   label=Ne  NeI unique S/N=314
6598.9528      150.363   label=Ne  NeI unique S/N=314
6652.0925        0.426   label=Ne  NeI blend S/N=36
6678.2766      254.141   label=Ne  NeI unique S/N=326
6717.0430       63.730   label=Ne  NeI blend S/N=314
6929.4672      244.382   label=Ne  NeI unique S/N=292
7032.4128     1000.000   label=Ne  NeI unique S/N=284
7173.9380       40.792   label=Ne  NeI unique S/N=270
7245.1665      370.896   label=Ne  NeI unique S/N=255

* Kr lines
6652.2347        0.271   label=Kr  KrI blend S/N=36
7224.1030        0.359   label=Kr  KrI unique S/N=18

* Cd lines
6438.4695      887.477   label=Cd  CdI unique S/N=335
6778.1157        0.445   label=Cd  CdI unique S/N=24
7345.6704        4.521   label=Cd  CdI unique S/N=158

