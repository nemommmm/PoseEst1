# SKT Quality-Gate Sweep for Elbow High-Delta Outliers

- Input: `01_stereo_triangulation/results/skt_model_fusion/rtmlib_body_light_120/yolo_3d_optimized.npz`
- High-delta threshold: `35.0 deg`
- Scope: LeftElbow and RightElbow only; no Xsens reference is used here.

## Best candidate gates

| Angle | K | Signal | Bad if | Threshold | Retained | Captured high-delta | High-delta retained | Precision lift |
|---|---:|---|---|---:|---:|---:|---:|---:|
| RightElbow | 6 | detect_conf_min | < | 0.50 | 0.588 | 1.000 | 0.000 | 2.425 |
| RightElbow | 6 | pair_conf_min | < | 0.60 | 0.577 | 1.000 | 0.000 | 2.366 |
| RightElbow | 6 | detect_conf_min | < | 0.60 | 0.557 | 1.000 | 0.000 | 2.256 |
| LeftElbow | 1 | pair_conf_min | < | 0.40 | 0.554 | 1.000 | 0.000 | 2.244 |
| LeftElbow | 1 | detect_conf_min | < | 0.40 | 0.545 | 1.000 | 0.000 | 2.196 |
| LeftElbow | 1 | pair_conf_min | < | 0.50 | 0.545 | 1.000 | 0.000 | 2.196 |
| LeftElbow | 1 | pair_conf_min | < | 0.60 | 0.535 | 1.000 | 0.000 | 2.149 |
| LeftElbow | 6 | pair_conf_min | < | 0.40 | 0.531 | 1.000 | 0.000 | 2.133 |
| LeftElbow | 1 | detect_conf_min | < | 0.50 | 0.525 | 1.000 | 0.000 | 2.104 |
| LeftElbow | 1 | reprojection_error_max | > | 30.00 | 0.525 | 1.000 | 0.000 | 2.104 |
| LeftElbow | 6 | detect_conf_min | < | 0.40 | 0.521 | 1.000 | 0.000 | 2.087 |
| LeftElbow | 6 | pair_conf_min | < | 0.50 | 0.521 | 1.000 | 0.000 | 2.087 |

## How to read this

- `Captured high-delta` means how many suspicious large angle jumps would be flagged by this gate.
- `Retained` means how much data remains if flagged frame pairs are removed or sent to a prior-assisted repair path.
- A useful gate should capture many high-delta pairs without discarding most of the sequence.
- This sweep is a diagnostic layer for SKT jitter; it should be paired with the frame-delta comparison report later.
