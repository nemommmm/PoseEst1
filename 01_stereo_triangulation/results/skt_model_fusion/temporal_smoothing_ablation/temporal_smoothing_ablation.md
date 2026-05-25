# Temporal Smoothing Ablation

- High-delta threshold: `35.0 deg`
- Reference-free diagnostic: Xsens is not used here.

| Variant | Angle | K | Elbow valid | Mean delta | P95 delta | High-delta rate |
|---|---|---|---|---|---|---|
| raw | LeftElbow | 1 | 0.492 | 4.439 | 11.524 | 0.017 |
| raw | LeftElbow | 6 | 0.492 | 8.100 | 20.112 | 0.019 |
| raw | RightElbow | 1 | 0.567 | 4.510 | 14.036 | 0 |
| raw | RightElbow | 6 | 0.567 | 11.451 | 37.155 | 0.066 |
| bone_only | LeftElbow | 1 | 1 | 2.440 | 8.070 | 0.008 |
| bone_only | LeftElbow | 6 | 1 | 7.475 | 44.240 | 0.053 |
| bone_only | RightElbow | 1 | 1 | 3.928 | 14.339 | 0.017 |
| bone_only | RightElbow | 6 | 1 | 10.360 | 44.025 | 0.105 |
| one_euro_only | LeftElbow | 1 | 0.492 | 1.902 | 5.405 | 0 |
| one_euro_only | LeftElbow | 6 | 0.492 | 6.569 | 17.072 | 0 |
| one_euro_only | RightElbow | 1 | 0.567 | 2.638 | 10.558 | 0 |
| one_euro_only | RightElbow | 6 | 0.567 | 9.950 | 33.470 | 0.049 |
| bone_plus_one_euro | LeftElbow | 1 | 1 | 1.506 | 5.717 | 0.008 |
| bone_plus_one_euro | LeftElbow | 6 | 1 | 6.780 | 33.951 | 0.053 |
| bone_plus_one_euro | RightElbow | 1 | 1 | 2.208 | 9.141 | 0 |
| bone_plus_one_euro | RightElbow | 6 | 1 | 7.852 | 29.687 | 0.035 |
| kalman_only | LeftElbow | 1 | 1 | 1.547 | 9.635 | 0 |
| kalman_only | LeftElbow | 6 | 1 | 8.307 | 41.177 | 0.061 |
| kalman_only | RightElbow | 1 | 1 | 2.931 | 7.177 | 0.017 |
| kalman_only | RightElbow | 6 | 1 | 16.480 | 100.703 | 0.123 |
