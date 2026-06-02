# Dataset Parameter Notes for `00_pose_pipeline`

This document explains which parameters are specific to the current dataset and
which settings must be checked when a new dataset is added.

The goal of `00_pose_pipeline` is to keep the complete workflow in one place:
**SKT inference + time synchronization + angle evaluation + K-frame motion
evaluation**. For a new dataset, start by copying
`configs/template_new_dataset.yaml` and updating only the required paths and
dataset-specific settings.

## Current Dataset Defaults

| Parameter | Current Value | Reason |
|---|---:|---|
| `dataset.rotate_180` | `true` | The current stereo videos are upside down when loaded, so frames are rotated 180 degrees before SKT/YOLO inference. |
| `dataset.sync_by` | `hardware_frame_id` | Left and right camera streams are paired by the hardware frame ID in the metadata. When frames are missing, the lower-ID side is skipped until both sides match. |
| `dataset.timestamp_format` | `seconds_microseconds_columns` | The metadata stores seconds and microseconds in separate columns, parsed as `seconds + microseconds * 1e-6`. |
| `offset.initial_reference_seconds` | `17.25` | Previous manually guided reference value. It is only used as a sanity-check reference and is not forced by the new pipeline. |
| `evaluation.camera_smooth_window_ms` | `200` | Camera/vision methods are smoothed with a 200 ms moving average before motion-delta calculation. |
| `evaluation.xsens_extra_smoothing` | `false` | Xsens already contains internal filtering, so no extra project smoothing is applied. |
| `evaluation.skt_quality_filter` | `enabled` | Matches the 05-18 evaluation policy by masking poor SKT upper-body keypoints using triangulation confidence, epipolar error, and reprojection error. |

## Checklist for a New Dataset

1. Confirm whether the videos need rotation before inference and set `rotate_180`.
2. Confirm the metadata column meanings: frame ID, seconds, and microseconds.
3. Confirm whether left/right frames can be paired directly by frame ID.
4. Run `validate` and inspect synchronized frame count, dropped/skipped rows, duration, and FPS.
5. Confirm whether the existing camera calibration is still valid. If the camera setup changed, use a new `camera_params` file.
6. Re-run automatic Xsens offset search. Do not reuse the current dataset's `17.25s` offset.
7. Confirm how FastSAM3D / Merge TRC files should align: synced-frame index, left-camera frame index, or TRC time column.

## Recommended Run Order

For the current dataset:

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/run_pipeline.py \
  --config 00_pose_pipeline/configs/current_2025_ergonomics.yaml \
  --stages validate,offset,angle,motion,segment,scatter
```

For a new dataset without an SKT NPZ:

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/run_pipeline.py \
  --config 00_pose_pipeline/configs/template_new_dataset.yaml \
  --stages validate,skt,offset,angle,motion
```

## Output Files

- `alignment_summary.json`: automatic offset-search result. Downstream angle and motion evaluations read `selected_offset_seconds`.
- `angle_summary.csv/json`: traditional angle agreement metrics.
- `motion_delta_summary.json`: K-frame motion-delta agreement metrics.
- `segment_summary.csv/json`: segment-level ROM, DTW, and RULA-like agreement metrics.
- `scatter/`: K-frame delta scatter plots.

## Wording Policy

Xsens is used as an external comparison/reference system, not as absolute ground
truth. Recommended report wording:

- `agreement with Xsens-derived reference`
- `comparison against the Xsens comparison system`
- `Xsens-derived geometric reference`

Avoid:

- `ground truth error`
- `validated against ground truth`
