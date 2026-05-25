# SKT Detector Smoke Comparison

- High-delta threshold: `35.0 deg`
- Compared first `120` frames from each run.
- Reference-free summary: Xsens is not used in this diagnostic.

| Run | Frames | Valid joints | L elbow chain | R elbow chain | Epi p90 px | Reproj p90 px | Stereo quality | Pair conf | L K1 high | R K1 high | L K6 high | R K6 high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8m | 120 | 0.784 | 1.000 | 1.000 | 133.153 | 16.617 | 0.235 | 0.571 | 0.008 | 0.000 | 0.053 | 0.035 |
| YOLO11m | 120 | 0.791 | 1.000 | 1.000 | 76.138 | 18.854 | 0.232 | 0.587 | 0.008 | 0.000 | 0.053 | 0.053 |
| RTMPoseS | 120 | 0.748 | 0.850 | 0.892 | 153.597 | 22.692 | 0.220 | 0.628 | 0.089 | 0.000 | 0.208 | 0.144 |

## Notes

- Lower epipolar/reprojection percentiles and higher stereo-quality values generally indicate cleaner stereo geometry.
- High-delta rates are only a quick smoke-test signal on short clips; full-sequence frame-delta evaluation is still required.
