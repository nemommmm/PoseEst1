# Temporal Smoothing Ablation

- High-delta threshold: `35.0 deg`
- Reference-free diagnostic: Xsens is not used here.

- Timeline: `corrected` (original non-positive timestamp diffs: `241`).

| Variant | Angle | K | Elbow valid | Mean delta | P95 delta | High-delta rate |
|---|---|---|---|---|---|---|
| raw | LeftElbow | 1 | 0.728 | 17.543 | 56.438 | 0.146 |
| raw | LeftElbow | 6 | 0.728 | 24.508 | 71.197 | 0.241 |
| raw | RightElbow | 1 | 0.848 | 16.763 | 60.906 | 0.138 |
| raw | RightElbow | 6 | 0.848 | 25.703 | 78.363 | 0.255 |
| bone_only | LeftElbow | 1 | 0.926 | 14.782 | 53.635 | 0.109 |
| bone_only | LeftElbow | 6 | 0.926 | 23.706 | 75.317 | 0.234 |
| bone_only | RightElbow | 1 | 0.938 | 15.596 | 52.796 | 0.125 |
| bone_only | RightElbow | 6 | 0.938 | 25.068 | 75.777 | 0.248 |
| one_euro_only | LeftElbow | 1 | 0.728 | 11.213 | 41.101 | 0.073 |
| one_euro_only | LeftElbow | 6 | 0.728 | 21.804 | 64.409 | 0.198 |
| one_euro_only | RightElbow | 1 | 0.848 | 11.073 | 41.441 | 0.070 |
| one_euro_only | RightElbow | 6 | 0.848 | 23.847 | 75.316 | 0.228 |
| bone_plus_one_euro | LeftElbow | 1 | 0.926 | 9.741 | 36.984 | 0.057 |
| bone_plus_one_euro | LeftElbow | 6 | 0.926 | 20.223 | 64.427 | 0.185 |
| bone_plus_one_euro | RightElbow | 1 | 0.938 | 9.696 | 35.478 | 0.051 |
| bone_plus_one_euro | RightElbow | 6 | 0.938 | 21.497 | 68.852 | 0.192 |
| kalman_only | LeftElbow | 1 | 1 | 3.789 | 14.428 | 0.013 |
| kalman_only | LeftElbow | 6 | 1 | 14.576 | 52.422 | 0.101 |
| kalman_only | RightElbow | 1 | 1 | 3.355 | 10.933 | 0.009 |
| kalman_only | RightElbow | 6 | 1 | 13.615 | 48.102 | 0.091 |
