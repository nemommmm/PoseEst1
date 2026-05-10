# PoseEst1

MVE386 / Chalmers 硕士论文项目：基于 3D 视觉与 AI 的制造环境人体姿态与人体工程学评估。

## Xsens Reference Policy

Xsens MVN data should be treated as an external comparison/reference system, not
as absolute ground truth. The suit allows sensors to be mounted in arbitrary
orientations and then relies heavily on its calibration routine; this can
neutralize or override the real global orientation and introduce systematic
initial offsets, especially visible in elbow-angle data.

For thesis/report wording, use terms such as "Xsens-derived reference",
"comparison against Xsens", or "agreement with the Xsens comparison system".
Avoid calibrating the vision pipeline directly to Xsens initial frames as if
they were physically exact, because that can transfer Xsens calibration bias into
our method. True ground truth would require an independent measurement source
such as controlled goniometer measurements or a higher-grade optical mocap setup.

## Pipeline Naming

后续论文、报告和代码说明统一使用以下命名：

| Directory | Formal Name | Abbreviation | Chinese |
|---|---|---|---|
| `01_stereo_triangulation/` | Sparse Keypoint Triangulation | SKT | 稀疏关键点三角化 |
| `02_dense_stereo_sgbm/` | Dense Disparity Mapping | DDM | 密集视差映射 |
| `03_mono_motionbert/` | Monocular Temporal Lifting | MTL | 单目时序提升 |

## Structure

- `shared/`: 共享工具库（Xsens reference 解析、角度语义、后处理、标定参数）
- `01_stereo_triangulation/`: 双目关键点三角化主线（SKT）
- `02_dense_stereo_sgbm/`: SGBM 密集视差对照方向（DDM）
- `03_mono_motionbert/`: 单目 MotionBERT 对照方向（MTL）

## Core Evaluation

项目核心指标按优先级统一为：

1. Joint Angle MAE
2. RULA scoring
3. MPJPE（支持性空间指标）

When these metrics are computed against Xsens, interpret them as agreement with
the Xsens-derived reference rather than true physical error against absolute GT.
