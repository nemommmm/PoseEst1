# Quality vs Angle Error Diagnostic Summary

- Pose file: `/Users/laomeng/Desktop/MVE386 Project Course/PoseEst1/01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
- Alignment offset: `17.25 s`
- Angle smoothing radius: `8`

## Highest-MAE angles

| Angle | MAE (deg) | N |
|-------|-----------|---|
| LeftElbow | 22.21 | 2500 |
| RightElbow | 21.71 | 2525 |
| LeftShoulder | 19.54 | 2498 |
| LeftKnee | 18.04 | 2200 |
| RightKnee | 16.80 | 2155 |
| RightHip | 16.47 | 2202 |
| LeftHip | 15.99 | 2200 |
| RightShoulder | 15.37 | 2512 |

## Best predictive quality signal per angle

| Angle | Signal | Spearman rho | Bad quartile MAE | Good quartile MAE | Gap |
|-------|--------|--------------|------------------|-------------------|-----|
| LeftElbow | detect_conf_min | -0.398 | 36.43 | 14.43 | +22.00 |
| LeftKnee | detect_conf_min | -0.229 | 28.80 | 13.80 | +15.01 |
| RightKnee | stereo_quality_min | -0.191 | 25.38 | 13.41 | +11.97 |
| RightElbow | detect_conf_min | -0.180 | 28.47 | 18.42 | +10.04 |
| LeftHip | detect_conf_min | -0.195 | 22.06 | 12.20 | +9.86 |
| RightShoulder | pair_conf_min | -0.230 | 20.21 | 11.29 | +8.93 |
| LeftShoulder | epipolar_error_max | +0.095 | 25.14 | 16.67 | +8.47 |
| RightHip | stereo_quality_min | -0.170 | 20.90 | 13.36 | +7.54 |

## Interpretation

- The strongest error-predictive combinations are `LeftElbow` via `detect_conf_min`, `LeftKnee` via `detect_conf_min`, `RightKnee` via `stereo_quality_min`. These are the best candidates for selective correction.
- If bad-quality quartiles show clearly higher MAE than good-quality quartiles, we have room for quality-aware gating instead of replacing the full pipeline.
- Signals tied to `reprojection_error` / `epipolar_error` point to stereo geometry issues; signals tied to `detect_conf` / `temporal_rescue` point to 2D detector instability or rescue-side effects.
