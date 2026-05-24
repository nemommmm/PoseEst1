# SKT Elbow Quality Repair Ablation

## Purpose

This experiment tests whether SKT elbow high-delta events can be reduced by
using internal quality signals to repair short unreliable elbow-chain segments.
It is reference-free: Xsens and FastSAM3D are not used here.

## Input

- Pose file: `01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
- Angles: `LeftElbow`, `RightElbow`
- Delta windows: `K=1`, `K=6`
- High-delta threshold: `35 deg`

## Ablation Summary

| Run | Angle | K | High before | High after | P95 before | P95 after | Valid after |
|---|---|---:|---:|---:|---:|---:|---:|
| default gap=5 | LeftElbow | 1 | 0.062 | 0.056 | 38.4 | 37.5 | 0.927 |
| default gap=5 | LeftElbow | 6 | 0.188 | 0.186 | 64.6 | 63.5 | 0.927 |
| default gap=5 | RightElbow | 1 | 0.054 | 0.048 | 36.8 | 34.3 | 0.938 |
| default gap=5 | RightElbow | 6 | 0.199 | 0.198 | 68.9 | 67.5 | 0.938 |
| default gap=12 | LeftElbow | 1 | 0.062 | 0.051 | 38.4 | 35.0 | 0.927 |
| default gap=12 | LeftElbow | 6 | 0.188 | 0.175 | 64.6 | 62.8 | 0.927 |
| default gap=12 | RightElbow | 1 | 0.054 | 0.040 | 36.8 | 32.3 | 0.938 |
| default gap=12 | RightElbow | 6 | 0.199 | 0.177 | 68.9 | 60.0 | 0.938 |
| default gap=20 | LeftElbow | 1 | 0.062 | 0.046 | 38.4 | 33.3 | 0.927 |
| default gap=20 | LeftElbow | 6 | 0.188 | 0.157 | 64.6 | 59.9 | 0.927 |
| default gap=20 | RightElbow | 1 | 0.054 | 0.037 | 36.8 | 31.0 | 0.938 |
| default gap=20 | RightElbow | 6 | 0.199 | 0.169 | 68.9 | 59.0 | 0.938 |
| moderate gap=12 | LeftElbow | 1 | 0.062 | 0.046 | 38.4 | 33.6 | 0.927 |
| moderate gap=12 | LeftElbow | 6 | 0.188 | 0.172 | 64.6 | 59.9 | 0.927 |
| moderate gap=12 | RightElbow | 1 | 0.054 | 0.046 | 36.8 | 33.9 | 0.938 |
| moderate gap=12 | RightElbow | 6 | 0.199 | 0.197 | 68.9 | 69.2 | 0.938 |
| moderate gap=20 | LeftElbow | 1 | 0.062 | 0.044 | 38.4 | 32.8 | 0.927 |
| moderate gap=20 | LeftElbow | 6 | 0.188 | 0.163 | 64.6 | 58.4 | 0.927 |
| moderate gap=20 | RightElbow | 1 | 0.054 | 0.044 | 36.8 | 33.2 | 0.938 |
| moderate gap=20 | RightElbow | 6 | 0.199 | 0.189 | 68.9 | 68.9 | 0.938 |

## Interpretation

- Quality-aware repair does reduce elbow high-delta events, especially for `K=1`.
- Longer repair windows also reduce `K=6` high-delta events, but the interpretation becomes less conservative:
  - `gap=12` is approximately `0.96 s` at 12.5 fps.
  - `gap=20` is approximately `1.6 s` at 12.5 fps.
- The best numerical result is `default gap=20`, but this may be too aggressive for motion-risk evaluation because it can smooth over real motion.
- The most defensible next candidate is `default gap=12`: it improves both elbows while remaining less aggressive than 20-frame interpolation.

## Next Step

Run the frame-delta evaluation on the repaired output, preferably starting with:

```bash
01_stereo_triangulation/results/skt_model_fusion/elbow_quality_repair_gap12/skt_elbow_quality_repaired.npz
```

Then compare against the original SKT, FastSAM3D unfiltered, and the
Xsens-derived reference using the same K=1/K=6 evaluation pipeline.
