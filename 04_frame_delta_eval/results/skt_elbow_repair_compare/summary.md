# SKT Elbow Quality Repair Frame-Delta Check

## Purpose

This check tests whether the reference-free SKT elbow quality repair also
improves the existing frame-delta evaluation against the Xsens-derived
reference. It is not a new claim that Xsens is ground truth; the comparison is
only agreement with the Xsens-derived reference.

## Setup

- Original SKT:
  - `01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
- Repaired SKT candidates:
  - `elbow_quality_repair_gap12/skt_elbow_quality_repaired.npz`
  - `elbow_quality_repair_gap20/skt_elbow_quality_repaired.npz`
- FastSAM3D TRC:
  - `../10 Aitor/fastsam3d_2.trc`
- Merge TRC:
  - `../10 Aitor/merged_output_2.trc`
- Smoothing:
  - camera systems: moving average, 200 ms nominal window
  - Xsens systems: no extra smoothing
- Important:
  - `--enable-quality-filter` was intentionally not used here, because it would re-mask the same low-quality frames that the repair step tries to fill.

## SKT vs Xsens-Derived Reference

| Run | Angle | K | Pearson | RMSE | Path ratio | Quiet MAE |
|---|---|---:|---:|---:|---:|---:|
| original | LeftElbow | 1 | 0.177 | 8.96 | 2.630 | 4.69 |
| original | LeftElbow | 6 | 0.310 | 24.94 | 1.912 | 14.17 |
| original | RightElbow | 1 | 0.172 | 9.24 | 2.618 | 4.97 |
| original | RightElbow | 6 | 0.282 | 27.39 | 1.899 | 13.88 |
| gap12 | LeftElbow | 1 | 0.182 | 8.65 | 2.501 | 4.70 |
| gap12 | LeftElbow | 6 | 0.314 | 23.98 | 1.843 | 13.31 |
| gap12 | RightElbow | 1 | 0.188 | 8.64 | 2.408 | 4.61 |
| gap12 | RightElbow | 6 | 0.311 | 25.76 | 1.793 | 12.97 |
| gap20 | LeftElbow | 1 | 0.169 | 8.26 | 2.351 | 4.69 |
| gap20 | LeftElbow | 6 | 0.285 | 23.17 | 1.722 | 13.04 |
| gap20 | RightElbow | 1 | 0.207 | 8.39 | 2.325 | 4.47 |
| gap20 | RightElbow | 6 | 0.350 | 24.74 | 1.753 | 12.78 |

## Interpretation

- `gap12` improves all four SKT-vs-XsensFair dynamic comparisons modestly:
  - Pearson increases for both elbows and both K values.
  - RMSE decreases.
  - Path ratio decreases, meaning SKT contains less excess motion path relative to the reference.
  - Quiet-frame MAE improves except for LeftElbow K=1, where it is essentially unchanged.
- `gap20` gives the best path-ratio and RMSE values, but it worsens LeftElbow Pearson. It is also much less conservative because 20 frames is about 1.6 s at 12.5 fps.
- Current best defensible candidate:
  - `gap12`
  - It gives consistent, modest improvement without using the most aggressive repair window.

## Conclusion

This pilot supports the idea that SKT jitter is partly quality-signal
explainable and can be reduced with targeted elbow-chain repair. The effect is
not large enough to close the gap to FastSAM3D, but it is a real improvement
over the original SKT output and is more explainable than direct raw-coordinate
fusion with FastSAM3D.
