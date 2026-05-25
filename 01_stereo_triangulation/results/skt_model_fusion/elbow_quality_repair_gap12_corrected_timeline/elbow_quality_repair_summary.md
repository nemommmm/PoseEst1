# SKT Elbow Quality Repair Experiment

- Input: `/Users/laomeng/Desktop/MVE386 Project Course/PoseEst1/01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
- Scope: reference-free internal SKT stability; Xsens is not used.
- Timeline: `corrected` (original non-positive timestamp diffs: `241`).
- Gate: pair/detect conf < `0.6`, stereo quality < `0.4`, epipolar > `6.0` px, reprojection > `30.0` px.
- Repair: linear interpolation for bad segments up to `12` frames.

## Repair coverage

| Side | Bad frames | Bad ratio | Repaired frames | Repaired/bad |
|---|---:|---:|---:|---:|
| Left | 1268 | 0.453 | 238 | 0.188 |
| Right | 1018 | 0.363 | 375 | 0.368 |

## Before vs after elbow deltas

| Angle | K | Before high-rate | After high-rate | Before p95 | After p95 | Before valid | After valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeftElbow | 1 | 0.062 | 0.047 | 38.390 | 33.814 | 0.926 | 0.927 |
| LeftElbow | 6 | 0.188 | 0.171 | 64.594 | 62.396 | 0.926 | 0.927 |
| RightElbow | 1 | 0.054 | 0.034 | 36.752 | 29.581 | 0.938 | 0.938 |
| RightElbow | 6 | 0.199 | 0.174 | 68.933 | 60.247 | 0.938 | 0.938 |

## Interpretation guardrails

- A lower high-delta rate is useful only if valid-angle coverage does not collapse.
- This is not yet an accuracy result against Xsens-derived reference or FastSAM3D.
- If this improves internal stability, the repaired NPZ should be rerun through frame-delta evaluation.
