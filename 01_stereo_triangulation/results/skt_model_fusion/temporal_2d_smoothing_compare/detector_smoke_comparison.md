# SKT Detector Smoke Comparison

- High-delta threshold: `35.0 deg`
- Compared first `120` frames from each run.
- Reference-free summary: Xsens is not used in this diagnostic.

| Run | Frames | Valid joints | L elbow chain | R elbow chain | Epi p90 px | Reproj p90 px | Stereo quality | Pair conf | L K1 high | R K1 high | L K6 high | R K6 high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8m_2DOneEuro | 120 | 0.784 | 1.000 | 1.000 | 133.153 | 16.617 | 0.235 | 0.571 | 0.008 | 0.000 | 0.053 | 0.035 |
| YOLOv8m_no2DOneEuro | 120 | 0.784 | 1.000 | 1.000 | 135.311 | 16.608 | 0.234 | 0.571 | 0.008 | 0.000 | 0.053 | 0.035 |

## Notes

- Lower epipolar/reprojection percentiles and higher stereo-quality values generally indicate cleaner stereo geometry.
- High-delta rates are only a quick smoke-test signal on short clips; full-sequence frame-delta evaluation is still required.
