# PoseEst1

Chalmers master's thesis project on 3D-vision and AI-based human pose
estimation for ergonomic assessment in manufacturing environments.

## Pipeline Naming

Use the following names consistently in the thesis, reports, and code comments:

| Directory | Formal Name | Abbreviation | Purpose |
|---|---|---|---|
| `00_pose_pipeline/` | Standalone Pose Pipeline | Pipeline | Reusable end-to-end workflow for new datasets |
| `01_stereo_triangulation/` | Sparse Keypoint Triangulation | SKT | Stereo keypoint detection and triangulation |
| `02_dense_stereo_sgbm/` | Dense Disparity Mapping | DDM | Dense stereo disparity baseline |
| `03_FastSAM3D/` | FastSAM3D / EasyErgo Hybrid Pose | FastSAM3D | External single-view / hybrid pose comparison |
| `04_frame_delta_eval/` | Motion-Level Elbow Evaluation | Frame Delta | Motion-level elbow evaluation experiments |

## Structure

- `00_pose_pipeline/`: standalone workflow for dataset validation, SKT inference,
  automatic time-offset estimation, angle evaluation, motion-delta evaluation,
  segment analysis, scatter plots, and comparison videos.
- `shared/`: shared utilities for Xsens parsing, angle semantics,
  post-processing, and calibration assets.
- `01_stereo_triangulation/`: original SKT development branch.
- `02_dense_stereo_sgbm/`: dense SGBM disparity baseline.
- `03_FastSAM3D/`: FastSAM3D / EasyErgo hybrid-pose branch.
- `04_frame_delta_eval/`: motion-level evaluation experiments against external
  reference and comparison methods.

## Core Evaluation

The core evaluation metrics are prioritized as follows:

1. Joint Angle MAE
2. RULA scoring
3. MPJPE as a supporting spatial diagnostic
