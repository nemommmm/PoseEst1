# SKT Quality-Gate Sweep for Elbow High-Delta Outliers

- Input: `01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
- High-delta threshold: `35.0 deg`
- Scope: LeftElbow and RightElbow only; no Xsens reference is used here.

## Best candidate gates

| Angle | K | Signal | Bad if | Threshold | Retained | Captured high-delta | High-delta retained | Precision lift |
|---|---:|---|---|---:|---:|---:|---:|---:|
| RightElbow | 1 | stereo_quality_min | < | 0.50 | 0.515 | 0.766 | 0.025 | 1.580 |
| LeftElbow | 1 | stereo_quality_min | < | 0.40 | 0.600 | 0.677 | 0.034 | 1.692 |
| RightElbow | 1 | pair_conf_min | < | 0.60 | 0.689 | 0.660 | 0.027 | 2.119 |
| LeftElbow | 6 | pair_conf_min | < | 0.60 | 0.561 | 0.649 | 0.118 | 1.478 |
| LeftElbow | 1 | pair_conf_min | < | 0.60 | 0.586 | 0.646 | 0.038 | 1.562 |
| LeftElbow | 6 | stereo_quality_min | < | 0.40 | 0.568 | 0.635 | 0.121 | 1.469 |
| RightElbow | 1 | stereo_quality_min | < | 0.40 | 0.667 | 0.596 | 0.033 | 1.787 |
| LeftElbow | 1 | epipolar_error_max | > | 6.00 | 0.692 | 0.571 | 0.039 | 1.854 |
| RightElbow | 6 | stereo_quality_min | < | 0.40 | 0.627 | 0.570 | 0.136 | 1.529 |
| LeftElbow | 1 | pair_conf_min | < | 0.50 | 0.667 | 0.565 | 0.041 | 1.695 |
| LeftElbow | 6 | pair_conf_min | < | 0.50 | 0.634 | 0.562 | 0.130 | 1.533 |
| RightElbow | 1 | epipolar_error_max | > | 6.00 | 0.695 | 0.560 | 0.034 | 1.839 |

## How to read this

- `Captured high-delta` means how many suspicious large angle jumps would be flagged by this gate.
- `Retained` means how much data remains if flagged frame pairs are removed or sent to a prior-assisted repair path.
- A useful gate should capture many high-delta pairs without discarding most of the sequence.
- This sweep is a diagnostic layer for SKT jitter; it should be paired with the frame-delta comparison report later.
