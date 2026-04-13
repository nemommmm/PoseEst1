# PoseEst1

MVE386 / Chalmers 硕士论文项目：基于 3D 视觉与 AI 的制造环境人体姿态与人体工程学评估。

## Pipeline Naming

后续论文、报告和代码说明统一使用以下命名：

| Directory | Formal Name | Abbreviation | Chinese |
|---|---|---|---|
| `01_stereo_triangulation/` | Sparse Keypoint Triangulation | SKT | 稀疏关键点三角化 |
| `02_dense_stereo_sgbm/` | Dense Disparity Mapping | DDM | 密集视差映射 |
| `03_mono_motionbert/` | Monocular Temporal Lifting | MTL | 单目时序提升 |

## Structure

- `shared/`: 共享工具库（Xsens 解析、角度语义、后处理、标定参数）
- `01_stereo_triangulation/`: 双目关键点三角化主线（SKT）
- `02_dense_stereo_sgbm/`: SGBM 密集视差对照方向（DDM）
- `03_mono_motionbert/`: 单目 MotionBERT 对照方向（MTL）

## Core Evaluation

项目核心指标按优先级统一为：

1. Joint Angle MAE
2. RULA scoring
3. MPJPE（支持性空间指标）
