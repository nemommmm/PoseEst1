# SKT Elbow Quality Repair Experiment

- Input: `01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
- Scope: reference-free internal SKT stability; Xsens is not used.
- Gate: pair/detect conf < `0.6`, stereo quality < `0.4`, epipolar > `6.0` px, reprojection > `30.0` px.
- Repair: linear interpolation for bad segments up to `20` frames.

## Repair coverage

| Side | Bad frames | Bad ratio | Repaired frames | Repaired/bad |
|---|---:|---:|---:|---:|
| Left | 1268 | 0.453 | 384 | 0.303 |
| Right | 1018 | 0.363 | 442 | 0.434 |

## Before vs after elbow deltas

| Angle | K | Before high-rate | After high-rate | Before p95 | After p95 | Before valid | After valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeftElbow | 1 | 0.062 | 0.046 | 38.390 | 33.326 | 0.926 | 0.927 |
| LeftElbow | 6 | 0.188 | 0.157 | 64.594 | 59.924 | 0.926 | 0.927 |
| RightElbow | 1 | 0.054 | 0.037 | 36.752 | 30.963 | 0.938 | 0.938 |
| RightElbow | 6 | 0.199 | 0.169 | 68.933 | 58.963 | 0.938 | 0.938 |

## Interpretation guardrails

- A lower high-delta rate is useful only if valid-angle coverage does not collapse.
- This is not yet an accuracy result against Xsens-derived reference or FastSAM3D.
- If this improves internal stability, the repaired NPZ should be rerun through frame-delta evaluation.
