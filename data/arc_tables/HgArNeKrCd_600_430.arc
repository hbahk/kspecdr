# K-SPEC arc line list v2, setup 600_430 (Isoplane SCT-320, 600 g/mm, centre 430 nm)
# Lamps on together: Newport-Oriel Hg(Ar) 6035, Ne 6032, Kr 6031; UVP Cd 90-0071-03
# Wavelengths: NIST ASD air wavelengths (observed, else Ritz), retrieved 2026-09-28
# Intensities: measured on the stacked commissioning arc (9 frames, nights 20260124-20260128), brightest unsaturated line = 1000;
#   blend components share the measured flux by NIST-predicted ratios
# Selection: >=5 sigma features, S/N >= 10, unique or blend components, second-order images at 2x lambda
#   lines that reduce_arc placed consistently off on the commissioning frames removed
# label=XX is the element (kspecdr read_arc_file); then species, class, S/N
# K-SPEC arc atlas v2 (2026-09-28): commissioning arcs of 2026 Jan-Feb stacked per setup;
#   lines identified against NIST ASD, checked with single-lamp lab frames and reduce_arc

* Hg lines
4046.5650        9.442   label=Hg  HgI unique S/N=220
4077.8370        1.048   label=Hg  HgI unique S/N=38
4358.3350      213.657   label=Hg  HgI unique S/N=291
4916.0680        0.537   label=Hg  HgI unique S/N=17

* Ne lines
4306.2508        0.223   label=Ne  NeI blend S/N=26
4412.2850        0.813   label=Ne  NeI blend S/N=115
4413.5610        0.612   label=Ne  NeI blend S/N=115
4640.4430        0.485   label=Ne  NeI unique S/N=14
4661.1054        5.637   label=Ne  NeI blend S/N=15
4663.0920        1.508   label=Ne  NeI blend S/N=15

* Kr lines
4273.9694        0.536   label=Kr  KrI unique S/N=17
4318.5524        0.266   label=Kr  KrI blend S/N=28
4319.5795        0.664   label=Kr  KrI blend S/N=28
4463.6900        0.428   label=Kr  KrI unique S/N=13

* Cd lines
4306.6718        0.635   label=Cd  CdI blend S/N=26
4412.9894        3.044   label=Cd  CdI blend S/N=115
4662.3520        7.496   label=Cd  CdI blend S/N=15
4678.1493      831.318   label=Cd  CdI unique S/N=278
4799.9123     1000.000   label=Cd  CdI unique S/N=279

