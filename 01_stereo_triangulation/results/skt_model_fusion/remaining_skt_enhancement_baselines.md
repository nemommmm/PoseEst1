# Remaining SKT Enhancement Baselines

## Scope

This note summarizes the last three quick SKT enhancement routes:

- Quality-aware elbow repair
- OneEuro / Kalman temporal smoothing
- Detector replacement with YOLO11m / RTMPoseS

All results here are reference-free smoke diagnostics unless explicitly stated.
Xsens is not used as ground truth in these checks.

## Quality-Aware Repair

The existing quality-gated repair remains useful. The most defensible setting is
`default gap=12`, because it reduces elbow high-delta rates without smoothing as
aggressively as `gap=20`.

| Angle | K | High before | High after | P95 before | P95 after |
|---|---:|---:|---:|---:|---:|
| LeftElbow | 1 | 0.062 | 0.051 | 38.4 | 35.0 |
| LeftElbow | 6 | 0.188 | 0.175 | 64.6 | 62.8 |
| RightElbow | 1 | 0.054 | 0.040 | 36.8 | 32.3 |
| RightElbow | 6 | 0.199 | 0.177 | 68.9 | 60.0 |

## Temporal Smoothing

3D OneEuro smoothing is a useful simple baseline. A simple Kalman filter is not
clearly safe without additional tuning, because it worsened the right-elbow K=6
tail in this short diagnostic.

| Variant | Angle | K | P95 delta | High-delta rate |
|---|---|---:|---:|---:|
| raw | LeftElbow | 1 | 11.524 | 0.017 |
| one_euro_only | LeftElbow | 1 | 5.405 | 0.000 |
| bone_plus_one_euro | RightElbow | 6 | 29.687 | 0.035 |
| kalman_only | RightElbow | 6 | 100.703 | 0.123 |

The 2D OneEuro on/off test produced almost identical first-120-frame high-delta
rates, so the remaining short-run elbow outliers are not solved by 2D smoothing
alone.

## Detector Replacement

| Run | Valid joints | L elbow chain | R elbow chain | Epi p90 px | Reproj p90 px | L K6 high | R K6 high |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8m | 0.784 | 1.000 | 1.000 | 133.153 | 16.617 | 0.053 | 0.035 |
| YOLO11m | 0.791 | 1.000 | 1.000 | 76.138 | 18.854 | 0.053 | 0.053 |
| RTMPoseS | 0.748 | 0.850 | 0.892 | 153.597 | 22.692 | 0.208 | 0.144 |

YOLO11m improves epipolar p90 in the short test, but it does not clearly reduce
elbow high-delta rates. RTMPoseS should not be used as a direct drop-in with the
current gates.

## Working Recommendation

The most practical short-term stack is:

1. Keep the current YOLOv8m SKT baseline unless YOLO11m full-sequence testing shows a clear gain.
2. Add conservative quality-aware repair for elbow-chain outliers.
3. Keep OneEuro / bone+OneEuro as the simple smoothing baseline.
4. Treat FastSAM3D prior fusion as the most promising auxiliary route, but not a final replacement for SKT.
