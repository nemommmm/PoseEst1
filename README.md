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

## Datasets

| Dataset | Folder | Subject | Capture date | Stereo cameras | Xsens file |
|---|---|---|---|---|---|
| 2025 Ergonomics | `2025_Ergonomics_Data/` | Aitor | 2025 | One stereo pair (`0_video_left/right.avi`) | `../Xsens_ground_truth/Aitor-001.mvnx` |
| 2026 Assar | `2026_Assar_Data/` | Fanbo | 2026-06-16 | Two locations: **A255** and **A257** (9 sessions each, `cap_{n}_0/1.avi`) | `Xsens MVNX/Fanbo-00{n}.mvnx` |

### Calibration files (in `shared/`, git-tracked)

| Camera | File | Notes |
|---|---|---|
| 2025 single stereo pair | `shared/camera_params_2025.npz` | Used by `current_2025_ergonomics.yaml` |
| 2026 A255 | `shared/camera_params_A255.npz` | Calibrated from `A255/SensorCalibration/` |
| 2026 A257 | `shared/camera_params_A257.npz` | Calibrated from `A257/SensorCalibration/` + `SiteCalibration/` (supplemented due to thin Sensor data) |

All three cameras are the same Viscando model (fx ≈ 1128 px, 2048×1536, baseline ≈ 41 cm).
Convention: `cap_*_0.avi` = left camera, `cap_*_1.avi` = right camera (consistent with 2025 naming).

### 2026 Assar dataset — capture modalities per session

Each session `Fanbo{n}` (n = 1–9) contains data from three capture modalities:

| Modality | Files | Location |
|---|---|---|
| Stereo A255 | `A255/Video/cap_{n}_0.avi` + `cap_{n}_1.avi` + `.txt` timestamps | Angle 1 |
| Stereo A257 | `A257/Video/cap_{n}_0.avi` + `cap_{n}_1.avi` + `.txt` timestamps | Angle 2 |
| Monocular webcam | `Webcam videos/Fanbo{n}_*.mp4` | Single view |
| FastSAM3D TRC | `TRC FastSAM3D/markers_Fanbo{n}_*.trc` | From webcam video |
| Xsens MVNX | `Xsens MVNX/Fanbo-00{n}.mvnx` | IMU motion capture |

Session-to-stereo-recording index and per-session action notes are stored in `2026_Assar_Data/video_note.md` (local only, not tracked in git).

## Structure

- `00_pose_pipeline/`: standalone workflow for dataset validation, SKT inference,
  automatic time-offset estimation, angle evaluation, motion-delta evaluation,
  segment analysis, scatter plots, and comparison videos.
- `shared/`: shared utilities for Xsens parsing, angle semantics,
  post-processing, and calibration assets (`camera_params_*.npz`).
- `01_stereo_triangulation/`: original SKT development branch (2025 dataset).
- `02_dense_stereo_sgbm/`: dense SGBM disparity baseline.
- `03_FastSAM3D/`: FastSAM3D / EasyErgo hybrid-pose branch.
- `04_frame_delta_eval/`: motion-level evaluation experiments against external
  reference and comparison methods.

## Core Evaluation

The core evaluation metrics are prioritized as follows:

1. Joint Angle MAE
2. RULA scoring
3. MPJPE as a supporting spatial diagnostic
