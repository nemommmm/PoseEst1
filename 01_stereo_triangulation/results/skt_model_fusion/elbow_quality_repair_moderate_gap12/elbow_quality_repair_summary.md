# SKT Elbow Quality Repair Experiment

- Input: `01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
- Scope: reference-free internal SKT stability; Xsens is not used.
- Gate: pair/detect conf < `0.5`, stereo quality < `0.3`, epipolar > `10.0` px, reprojection > `40.0` px.
- Repair: linear interpolation for bad segments up to `12` frames.

## Repair coverage

| Side | Bad frames | Bad ratio | Repaired frames | Repaired/bad |
|---|---:|---:|---:|---:|
| Left | 1043 | 0.372 | 279 | 0.267 |
| Right | 773 | 0.276 | 301 | 0.389 |

## Before vs after elbow deltas

| Angle | K | Before high-rate | After high-rate | Before p95 | After p95 | Before valid | After valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeftElbow | 1 | 0.062 | 0.046 | 38.390 | 33.571 | 0.926 | 0.927 |
| LeftElbow | 6 | 0.188 | 0.172 | 64.594 | 59.924 | 0.926 | 0.927 |
| RightElbow | 1 | 0.054 | 0.046 | 36.752 | 33.950 | 0.938 | 0.938 |
| RightElbow | 6 | 0.199 | 0.197 | 68.933 | 69.165 | 0.938 | 0.938 |

## Interpretation guardrails

- A lower high-delta rate is useful only if valid-angle coverage does not collapse.
- This is not yet an accuracy result against Xsens-derived reference or FastSAM3D.
- If this improves internal stability, the repaired NPZ should be rerun through frame-delta evaluation.
