# K-SPEC arc line list v2, Isoplane setups 300_490
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
4046.5650        4.611   label=Hg  HgI unique S/N=187 300_490
4077.8370        0.464   label=Hg  HgI blend S/N=30 300_490
4358.3350       75.782   label=Hg  HgI blend S/N=313 300_490
4916.0680        0.274   label=Hg  HgI blend S/N=19 300_490
5460.7500      249.894   label=Hg  HgI unique S/N=318 300_490
5769.6100       30.352   label=Hg  HgI blend S/N=298 300_490
5790.6700       39.059   label=Hg  HgI unique S/N=300 300_490

* Ar lines
5603.9355        0.504   label=Ar  ArII unique S/N=26 300_490

* Ne lines
4080.1480        0.117   label=Ne  NeI blend S/N=30 300_490
4306.2508        0.116   label=Ne  NeI blend S/N=19 300_490
4416.8170        0.384   label=Ne  NeI blend S/N=106 300_490
5150.0842        0.738   label=Ne  NeI blend S/N=241 300_490
5151.9610        1.589   label=Ne  NeI blend S/N=241 300_490
5154.4271        1.060   label=Ne  NeI blend S/N=241 300_490
5156.6672        1.060   label=Ne  NeI blend S/N=241 300_490
5158.9018        1.060   label=Ne  NeI blend S/N=241 300_490
5330.7775        0.507   label=Ne  NeI unique S/N=26 300_490
5341.0938        0.453   label=Ne  NeI blend S/N=37 300_490
5343.2834        0.272   label=Ne  NeI blend S/N=37 300_490
5400.5616        1.044   label=Ne  NeI unique S/N=54 300_490
5656.6588        0.243   label=Ne  NeI unique S/N=13 300_490
5770.3067        7.683   label=Ne  NeI blend S/N=298 300_490
5852.4878       70.678   label=Ne  NeI unique S/N=316 300_490
5881.8950       52.407   label=Ne  NeI unique S/N=278 300_490
5944.8340       79.906   label=Ne  NeI unique S/N=316 300_490
5974.6273        9.691   label=Ne  NeI blend S/N=164 300_490
5975.5343       11.607   label=Ne  NeI blend S/N=164 300_490
5987.9074        0.290   label=Ne  NeI blend S/N=11 300_490
5991.6477        0.145   label=Ne  NeI blend S/N=11 300_490
6029.9968       25.423   label=Ne  NeI unique S/N=296 300_490
6074.3376       61.143   label=Ne  NeI unique S/N=241 300_490
6143.0627       73.173   label=Ne  NeI blend S/N=290 300_490
6163.5937       47.304   label=Ne  NeI unique S/N=129 300_490

* Kr lines
4273.9694        0.284   label=Kr  KrI unique S/N=12 300_490
4318.5524        0.157   label=Kr  KrI blend S/N=21 300_490
4319.5795        0.392   label=Kr  KrI blend S/N=21 300_490
4362.6416       38.176   label=Kr  KrI blend S/N=313 300_490
4410.3681        0.917   label=Kr  KrI blend S/N=106 300_490
4416.8838        0.367   label=Kr  KrI blend S/N=106 300_490
4502.3543        0.316   label=Kr  KrI unique S/N=13 300_490
5570.2894        3.323   label=Kr  KrI unique S/N=144 300_490
5993.8502        0.277   label=Kr  KrI blend S/N=11 300_490

* Cd lines
4306.6718        0.329   label=Cd  CdI blend S/N=19 300_490
4412.9894        0.572   label=Cd  CdI blend S/N=106 300_490
4415.6859        0.604   label=Cd  CdII blend S/N=106 300_490
4678.1493      520.173   label=Cd  CdI unique S/N=319 300_490
4799.9123     1000.000   label=Cd  CdI unique S/N=313 300_490
4918.8500        0.109   label=Cd  CdII blend S/N=19 300_490
5085.8217      915.406   label=Cd  CdI unique S/N=411 300_490
5154.6605        3.170   label=Cd  CdI blend S/N=241 300_490

