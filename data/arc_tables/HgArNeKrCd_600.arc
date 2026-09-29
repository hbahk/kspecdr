# K-SPEC arc line list v2, Isoplane setups 600_430, 600_450, 600_680
# Lamps on together: Newport-Oriel Hg(Ar) 6035, Ne 6032, Kr 6031; UVP Cd 90-0071-03
# Wavelengths: NIST ASD air wavelengths (observed, else Ritz), retrieved 2026-09-28
# Intensities: measured on the stacked commissioning arcs (a line seen in several setups
#   from the one with the most grooves/mm, then the highest S/N), on the 150_620 flux
#   scale; brightest unsaturated line = 1000; blend components share the measured flux by
#   NIST-predicted ratios
# Selection: lines kept in a setup's v2 list (S/N >= 10, unique or blend components,
#   second-order images at 2x lambda), none that reduce_arc placed consistently off
# label=XX is the element (kspecdr read_arc_file); then species, class, S/N, setup
# K-SPEC arc atlas v2 (2026-09-28): commissioning arcs of 2026 Jan-Feb stacked per setup;
#   lines identified against NIST ASD, checked with single-lamp lab frames and reduce_arc

* Hg lines
4046.5650        5.514   label=Hg  HgI unique S/N=220 600_430
4077.8370        0.612   label=Hg  HgI unique S/N=38 600_430
4358.3350      124.786   label=Hg  HgI unique S/N=291 600_430
4916.0680        0.330   label=Hg  HgI unique S/N=19 600_450
6716.3400      108.050   label=Hg  HgI blend S/N=314 600_680
6907.4600        0.594   label=Hg  HgI unique S/N=31 600_680

* Ar lines
6965.4310        2.518   label=Ar  ArI unique S/N=118 600_680
7428.5740        2.397   label=Ar  ArII unique S/N=126 600_680

* Ne lines
4306.2508        0.130   label=Ne  NeI blend S/N=26 600_430
4412.2850        0.475   label=Ne  NeI blend S/N=115 600_430
4413.5610        0.358   label=Ne  NeI blend S/N=115 600_430
4565.8880        0.104   label=Ne  NeI blend S/N=10 600_450
4566.8300        0.070   label=Ne  NeI blend S/N=10 600_450
4567.1390        0.026   label=Ne  NeI blend S/N=10 600_450
4604.0950        0.196   label=Ne  NeI blend S/N=14 600_450
4604.9380        0.065   label=Ne  NeI blend S/N=14 600_450
4640.4430        0.283   label=Ne  NeI unique S/N=14 600_430
4661.1054        3.180   label=Ne  NeI blend S/N=20 600_450
4663.0920        0.851   label=Ne  NeI blend S/N=20 600_450
5037.7512        0.198   label=Ne  NeI unique S/N=12 600_450
6217.2812       46.244   label=Ne  NeI unique S/N=319 600_680
6266.4952      102.626   label=Ne  NeI unique S/N=316 600_680
6304.7893       41.431   label=Ne  NeI unique S/N=317 600_680
6334.4276      138.940   label=Ne  NeI unique S/N=311 600_680
6382.9914      237.637   label=Ne  NeI unique S/N=315 600_680
6402.2480      386.290   label=Ne  NeI unique S/N=322 600_680
6506.5277      261.723   label=Ne  NeI unique S/N=322 600_680
6532.8824      121.778   label=Ne  NeI unique S/N=314 600_680
6598.9528      150.363   label=Ne  NeI unique S/N=314 600_680
6652.0925        0.426   label=Ne  NeI blend S/N=36 600_680
6678.2766      254.141   label=Ne  NeI unique S/N=326 600_680
6717.0430       63.730   label=Ne  NeI blend S/N=314 600_680
6929.4672      244.382   label=Ne  NeI unique S/N=292 600_680
7032.4128     1000.000   label=Ne  NeI unique S/N=284 600_680
7173.9380       40.792   label=Ne  NeI unique S/N=270 600_680
7245.1665      370.896   label=Ne  NeI unique S/N=255 600_680

* Kr lines
4273.9694        0.313   label=Kr  KrI unique S/N=17 600_430
4318.5524        0.155   label=Kr  KrI blend S/N=28 600_430
4319.5795        0.388   label=Kr  KrI blend S/N=28 600_430
4463.6900        0.250   label=Kr  KrI unique S/N=13 600_430
6652.2347        0.271   label=Kr  KrI blend S/N=36 600_680
7224.1030        0.359   label=Kr  KrI unique S/N=18 600_680

* Cd lines
4306.6718        0.371   label=Cd  CdI blend S/N=26 600_430
4412.9894        1.778   label=Cd  CdI blend S/N=115 600_430
4662.3520        4.228   label=Cd  CdI blend S/N=20 600_450
4678.1493      510.797   label=Cd  CdI unique S/N=291 600_450
4799.9123      901.227   label=Cd  CdI unique S/N=288 600_450
5085.8217      598.251   label=Cd  CdI unique S/N=305 600_450
6438.4695      887.477   label=Cd  CdI unique S/N=335 600_680
6778.1157        0.445   label=Cd  CdI unique S/N=24 600_680
7345.6704        4.521   label=Cd  CdI unique S/N=158 600_680

