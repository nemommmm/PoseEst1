# SKT-Only Enhancement Deep Dive

## Why FastSAM3D Prior Fusion Was Removed From the Main Ranking

FastSAM3D prior fusion is useful as a diagnostic experiment, but it is not a
clean SKT improvement. If FastSAM3D already performs better, blending it into
SKT can trivially improve SKT-like output while weakening the methodological
argument. Therefore, the main SKT enhancement ranking now excludes FastSAM3D
fusion.

FastSAM3D should remain a strong external comparison method / candidate method,
not a hidden correction source inside SKT.

## Methodology Fix

Two SKT-only scripts now use the corrected stereo-video timeline by default:

- `22_elbow_quality_repair.py`
- `24_temporal_smoothing_ablation.py`

This matters because the original SKT NPZ timestamps contain `241` non-positive
frame-to-frame differences. The corrected timeline is built from synchronized
left/right camera metadata and is strictly monotonic.

## Quality-Aware Repair

Setting:

- `max_gap_frames = 12`
- `pair/detect confidence < 0.6`
- `stereo quality < 0.4`
- `epipolar error > 6 px`
- `reprojection error > 30 px`

Reference-free internal result:

| Angle | K | Before high-rate | After high-rate | Before p95 | After p95 |
|---|---:|---:|---:|---:|---:|
| LeftElbow | 1 | 0.062 | 0.047 | 38.390 | 33.814 |
| LeftElbow | 6 | 0.188 | 0.171 | 64.594 | 62.396 |
| RightElbow | 1 | 0.054 | 0.034 | 36.752 | 29.581 |
| RightElbow | 6 | 0.199 | 0.174 | 68.933 | 60.247 |

05 frame-delta result vs Xsens-derived reference:

| Run | Side | K | Pearson | RMSE deg | Path ratio |
|---|---|---:|---:|---:|---:|
| Original SKT | Left | 1 | 0.177 | 8.96 | 2.630 |
| Gap12 repair | Left | 1 | 0.191 | 8.43 | 2.451 |
| Original SKT | Left | 6 | 0.310 | 24.94 | 1.912 |
| Gap12 repair | Left | 6 | 0.327 | 23.88 | 1.849 |
| Original SKT | Right | 1 | 0.172 | 9.24 | 2.618 |
| Gap12 repair | Right | 1 | 0.198 | 8.43 | 2.343 |
| Original SKT | Right | 6 | 0.282 | 27.39 | 1.899 |
| Gap12 repair | Right | 6 | 0.319 | 25.48 | 1.788 |

Interpretation: this is the cleanest current SKT-only improvement. It reduces
short outliers, improves path ratio, and slightly improves dynamic correlation.

## OneEuro / Bone Prior / Kalman

Full raw-sequence reference-free smoothing ablation:

| Variant | Angle | K | P95 delta | High-delta rate |
|---|---|---:|---:|---:|
| raw | LeftElbow | 1 | 56.438 | 0.146 |
| bone_plus_one_euro | LeftElbow | 1 | 36.984 | 0.057 |
| raw | RightElbow | 1 | 60.906 | 0.138 |
| bone_plus_one_euro | RightElbow | 1 | 35.478 | 0.051 |
| kalman_only | LeftElbow | 1 | 14.428 | 0.013 |
| kalman_only | RightElbow | 1 | 10.933 | 0.009 |

At first glance, Kalman looks strongest because it greatly reduces high-delta
rates. However, downstream frame-delta evaluation shows that it destroys motion
shape agreement:

| Run | Side | K | Pearson vs XsensFair | Path ratio |
|---|---|---:|---:|---:|
| Original SKT | Left | 6 | 0.310 | 1.912 |
| bone_plus_one_euro | Left | 6 | 0.305 | 1.869 |
| kalman_only | Left | 6 | 0.067 | 1.547 |
| Original SKT | Right | 6 | 0.282 | 1.899 |
| one_euro_only | Right | 6 | 0.349 | 2.195 |
| kalman_only | Right | 6 | 0.038 | 1.309 |

Interpretation:

- OneEuro / bone+OneEuro is useful as a basic stabilization layer.
- Kalman is not recommended in its current form: it smooths too aggressively and
  can preserve amplitude/path while losing temporal shape.
- Quality-aware repair is more defensible than aggressive predictive filtering.

## Detector Replacement

Short-run detector smoke result:

| Run | Valid joints | L elbow chain | R elbow chain | Epi p90 px | Reproj p90 px | L K6 high | R K6 high |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8m | 0.784 | 1.000 | 1.000 | 133.153 | 16.617 | 0.053 | 0.035 |
| YOLO11m | 0.791 | 1.000 | 1.000 | 76.138 | 18.854 | 0.053 | 0.053 |
| RTMPoseS | 0.748 | 0.850 | 0.892 | 153.597 | 22.692 | 0.208 | 0.144 |

Interpretation:

- YOLO11m improves epipolar p90 in the short test, but it does not reduce elbow
  high-delta rates.
- RTMPoseS is not recommended as a direct replacement under the current gates.
- YOLO11m full-sequence testing is optional, but not higher priority than
  quality-aware repair.

## Updated Ranking

1. Quality-aware repair with corrected timeline: current best SKT-only route.
2. Existing OneEuro / bone+OneEuro: keep as baseline stabilization, but do not overclaim.
3. YOLO11m full-sequence test: optional, mainly to verify whether the epipolar gain holds.
4. RTMPoseS direct replacement: not recommended without detector-specific gate tuning.
5. Kalman-only smoothing: not recommended in current form.
