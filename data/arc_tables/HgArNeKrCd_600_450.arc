# K-SPEC arc line list v2, setup 600_450 (Isoplane SCT-320, 600 g/mm, centre 450 nm)
# Lamps on together: Newport-Oriel Hg(Ar) 6035, Ne 6032, Kr 6031; UVP Cd 90-0071-03
# Wavelengths: NIST ASD air wavelengths (observed, else Ritz), retrieved 2026-09-28
# Intensities: measured on the stacked commissioning arc (11 frames, nights 20260129-20260205), brightest unsaturated line = 1000;
#   blend components share the measured flux by NIST-predicted ratios
# Selection: >=5 sigma features, S/N >= 10, unique or blend components, second-order images at 2x lambda
#   lines that reduce_arc placed consistently off on the commissioning frames removed
# label=XX is the element (kspecdr read_arc_file); then species, class, S/N
# K-SPEC arc atlas v2 (2026-09-28): commissioning arcs of 2026 Jan-Feb stacked per setup;
#   lines identified against NIST ASD, checked with single-lamp lab frames and reduce_arc

* Hg lines
4046.5650        5.508   label=Hg  HgI unique S/N=212
4077.8370        0.606   label=Hg  HgI unique S/N=34
4358.3350      127.872   label=Hg  HgI unique S/N=285
4916.0680        0.367   label=Hg  HgI unique S/N=19

* Ne lines
4306.2508        0.122   label=Ne  NeI blend S/N=22
4412.2850        0.500   label=Ne  NeI blend S/N=111
4413.5610        0.376   label=Ne  NeI blend S/N=111
4565.8880        0.116   label=Ne  NeI blend S/N=10
4566.8300        0.077   label=Ne  NeI blend S/N=10
4567.1390        0.029   label=Ne  NeI blend S/N=10
4604.0950        0.218   label=Ne  NeI blend S/N=14
4604.9380        0.073   label=Ne  NeI blend S/N=14
4661.1054        3.528   label=Ne  NeI blend S/N=20
4663.0920        0.944   label=Ne  NeI blend S/N=20
5037.7512        0.220   label=Ne  NeI unique S/N=12

* Kr lines
4273.9694        0.253   label=Kr  KrI unique S/N=12
4318.5524        0.158   label=Kr  KrI blend S/N=26
4319.5795        0.395   label=Kr  KrI blend S/N=26
4463.6900        0.225   label=Kr  KrI unique S/N=10

* Cd lines
4306.6718        0.347   label=Cd  CdI blend S/N=22
4412.9894        1.871   label=Cd  CdI blend S/N=111
4662.3520        4.692   label=Cd  CdI blend S/N=20
4678.1493      566.779   label=Cd  CdI unique S/N=291
4799.9123     1000.000   label=Cd  CdI unique S/N=288
5085.8217      663.818   label=Cd  CdI unique S/N=305

