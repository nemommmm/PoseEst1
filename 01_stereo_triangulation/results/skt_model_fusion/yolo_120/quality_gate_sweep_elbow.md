# SKT Quality-Gate Sweep for Elbow High-Delta Outliers

- Input: `01_stereo_triangulation/results/skt_model_fusion/yolo_120/yolo_3d_optimized.npz`
- High-delta threshold: `35.0 deg`
- Scope: LeftElbow and RightElbow only; no Xsens reference is used here.

## Best candidate gates

| Angle | K | Signal | Bad if | Threshold | Retained | Captured high-delta | High-delta retained | Precision lift |
|---|---:|---|---|---:|---:|---:|---:|---:|
| LeftElbow | 1 | reprojection_error_max | > | 60.00 | 0.966 | 1.000 | 0.000 | 29.750 |
| LeftElbow | 1 | reprojection_error_max | > | 40.00 | 0.882 | 1.000 | 0.000 | 8.500 |
| LeftElbow | 1 | reprojection_error_max | > | 30.00 | 0.773 | 1.000 | 0.000 | 4.407 |
| LeftElbow | 6 | reprojection_error_max | > | 30.00 | 0.719 | 1.000 | 0.000 | 3.562 |
| LeftElbow | 1 | reprojection_error_max | > | 20.00 | 0.529 | 1.000 | 0.000 | 2.125 |
| RightElbow | 6 | detect_conf_min | < | 0.40 | 0.500 | 0.750 | 0.018 | 1.500 |
| RightElbow | 6 | pair_conf_min | < | 0.50 | 0.500 | 0.750 | 0.018 | 1.500 |
| LeftElbow | 6 | reprojection_error_max | > | 40.00 | 0.877 | 0.500 | 0.030 | 4.071 |
| RightElbow | 6 | detect_conf_min | < | 0.30 | 0.509 | 0.500 | 0.034 | 1.018 |
| RightElbow | 6 | pair_conf_min | < | 0.30 | 0.509 | 0.500 | 0.034 | 1.018 |
| RightElbow | 6 | pair_conf_min | < | 0.40 | 0.509 | 0.500 | 0.034 | 1.018 |
| RightElbow | 6 | reprojection_error_max | > | 40.00 | 0.921 | 0.250 | 0.029 | 3.167 |

## How to read this

- `Captured high-delta` means how many suspicious large angle jumps would be flagged by this gate.
- `Retained` means how much data remains if flagged frame pairs are removed or sent to a prior-assisted repair path.
- A useful gate should capture many high-delta pairs without discarding most of the sequence.
- This sweep is a diagnostic layer for SKT jitter; it should be paired with the frame-delta comparison report later.
