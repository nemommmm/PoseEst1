# 00_pose_pipeline

`00_pose_pipeline` is the standalone end-to-end workflow for this project. It is
intended to make a new dataset runnable after updating a small number of dataset
paths and dataset-specific parameters.

The pipeline covers stereo SKT inference, automatic time-offset estimation,
traditional angle evaluation, K-frame motion-delta evaluation, segment-level
analysis, scatter plots, and comparison videos.

## Quick Start

Reproduce the current dataset evaluation:

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/run_pipeline.py \
  --config 00_pose_pipeline/configs/current_2025_ergonomics.yaml \
  --stages validate,offset,angle,motion,segment,scatter
```

For a new dataset without an existing SKT NPZ:

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/run_pipeline.py \
  --config 00_pose_pipeline/configs/template_new_dataset.yaml \
  --stages validate,skt,offset,angle,motion
```

## Pipeline Stages

| Stage | Purpose |
|---|---|
| `validate` | Check file paths, stereo metadata, synchronized frame count, FPS, and duration. |
| `skt` | Run standalone sparse keypoint triangulation and export an SKT NPZ. |
| `offset` | Automatically estimate the temporal offset between the video timeline and Xsens timeline. |
| `angle` | Run traditional angle evaluation with MAE, bias, and RULA-like agreement. |
| `motion` | Run K-frame motion-delta evaluation with motion agreement, high-delta counts, and path ratios. |
| `segment` | Run activity-segment ROM, DTW, and RULA-like agreement analysis. |
| `scatter` | Generate K-frame delta scatter plots. |
| `video` | Generate raw-video plus skeleton comparison videos. |

## Important Files

- `configs/current_2025_ergonomics.yaml`: reproducible configuration for the current dataset.
- `configs/template_new_dataset.yaml`: template for future datasets.
- `docs/dataset_parameters.md`: checklist of dataset-specific parameters to verify before running a new dataset.
- `runs/<dataset>/alignment_summary.json`: automatic time-offset search output.
