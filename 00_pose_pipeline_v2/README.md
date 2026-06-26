# 00_pose_pipeline_v2

`00_pose_pipeline_v2` is an isolated experimental copy of the standalone pose
pipeline. The original `00_pose_pipeline` remains frozen as the stable
reference workflow, while v2 is used for Fanbo7-focused SKT jitter reduction and
FastSAM3D-referenced ablation experiments.

The pipeline covers stereo SKT inference, automatic time-offset estimation,
traditional angle evaluation, K-frame motion-delta evaluation, segment-level
analysis, scatter plots, and comparison videos.

## Quick Start

Reproduce the current dataset evaluation:

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline_v2/src/run_pipeline.py \
  --config 00_pose_pipeline_v2/configs/current_2025_ergonomics.yaml \
  --stages validate,offset,angle,motion,segment,scatter
```

Run the Fanbo7 right-elbow baseline and FastSAM3D-referenced evaluation:

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline_v2/src/run_pipeline.py \
  --config 00_pose_pipeline_v2/configs/assar2026_fanbo7_a257.yaml \
  --stages validate,offset,angle,fasteval
```

Export the current best YOLO11l full-replacement candidate for review:

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline_v2/src/export_candidate_timeseries.py \
  --config 00_pose_pipeline_v2/configs/assar2026_fanbo7_a257.yaml \
  --run-dir 00_pose_pipeline_v2/runs/assar2026_fanbo7_a257_yolo11l_full_skt \
  --variant-name hard_filter_keypoint_savgol \
  --candidate-label YOLO11l_SKT_keypoint_savgol
```

## Pipeline Stages

| Stage | Purpose |
|---|---|
| `validate` | Check file paths, stereo metadata, synchronized frame count, FPS, and duration. |
| `skt` | Run standalone sparse keypoint triangulation and export an SKT NPZ. |
| `offset` | Automatically estimate the temporal offset between the video timeline and Xsens timeline. |
| `angle` | Run traditional angle evaluation with MAE, bias, and RULA-like agreement. |
| `fasteval` | Evaluate selected systems against FastSAM3D as the right-elbow reference. |
| `modeldiag` | Compare 2D pose detectors on sampled stereo frames without replacing SKT. |
| `filter_ablation` | Compare filter, angle-postprocess, keypoint smoothing, and right-arm bone-constraint variants. |
| `motion` | Run K-frame motion-delta evaluation with motion agreement, high-delta counts, and path ratios. |
| `segment` | Run activity-segment ROM, DTW, and RULA-like agreement analysis. |
| `scatter` | Generate K-frame delta scatter plots. |
| `video` | Generate raw-video plus skeleton comparison videos. |

`export_candidate_timeseries.py` is a standalone review helper rather than a
pipeline stage. It is intended for finalized candidate runs where the SKT NPZ
already exists and only the selected postprocess variant needs to be exported as
a figure and summary.

## Important Files

- `configs/current_2025_ergonomics.yaml`: reproducible configuration for the current dataset.
- `configs/assar2026_fanbo7_a257.yaml`: Fanbo7 right-elbow baseline configuration.
- `configs/template_new_dataset.yaml`: template for future datasets.
- `docs/dataset_parameters.md`: checklist of dataset-specific parameters to verify before running a new dataset.
- `runs/<dataset>/alignment_summary.json`: automatic time-offset search output.
- `runs/<dataset>/eval_vs_fastsam/summary.json`: FastSAM3D-referenced SKT comparison summary.
- `runs/<dataset>/model_diagnostic/summary.json`: Stage 1 2D detector consistency diagnostic.
- `runs/<dataset>/filter_ablation/summary.json`: Stage 2-4 lightweight postprocess ablation.
- `runs/<dataset>/candidate_eval/`: exported angle trace and summary for the selected candidate model/postprocess combination.
