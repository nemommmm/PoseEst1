# Master Thesis Outline (Draft)

**Title**: Posture and Activity Detection in Manufacturing Environment Using 3D Vision and AI

**Author**: Fanbo Meng
**Institution**: Chalmers University of Technology, in collaboration with Viscando AB

---

## Chapter 1. Introduction

Introduces the background and motivation for ergonomic monitoring in industrial manufacturing environments, the limitations of current manual assessment practices, and the potential of automated vision-based solutions. Defines the research questions, scope, and main contributions, and outlines the structure of the thesis.

---

## Chapter 2. Background and Related Work

### 2.1 Overview of 3D Human Pose Estimation
Surveys monocular methods (2D detection + 3D lifting, end-to-end estimation) and stereo / multi-view methods (triangulation, dense disparity), discussing their principles, prior work, applicable scenarios, and limitations. Includes the theoretical background of stereo calibration (epipolar geometry, intrinsic and extrinsic parameters).

### 2.2 Pose Evaluation Methodology
Introduces the evaluation dimensions commonly used in the pose estimation literature: joint angle accuracy (angle-space), temporal motion consistency (motion-space), and 3D joint position accuracy (position-space). Also introduces RULA as a standard industrial ergonomic assessment tool and its dependence on joint angles; this thesis uses RULA bin classification accuracy as an application-level metric but does not implement full automated RULA scoring.

### 2.3 Reference Systems
Describes the working principles of IMU-based motion capture (Xsens) and its calibration limitations, explaining why it cannot be treated as absolute ground truth, and compares it with higher-grade reference systems such as optical motion capture.

### 2.4 Related Work Summary
Compares this thesis with prior research and clarifies its novelty.

---

## Chapter 3. Data and Experimental Setup

### 3.1 Stereo Capture System
Describes the hardware parameters, calibration procedure, and calibration accuracy (reprojection error). Detailed parameter matrices are provided in the Appendix.

### 3.2 2025 Ergonomics Dataset
Describes the recording environment, operational tasks, subject information, and data characteristics, including dataset scale and known limitations (single subject, controlled scenario).

### 3.3 Xsens Reference Data
Describes the acquisition parameters; introduces the two angle representations (Native vs. Fair) and their origin; quantitatively analyzes the calibration limitation through a self-comparison experiment (motion-space MAE ≈ 2°); justifies the choice of Xsens-Fair as the evaluation reference adopted in this thesis.

### 3.4 Cross-system Time Synchronization
Describes the timeline alignment method across systems, including reconstruction of a common timeline from hardware frame IDs and correction of the Xsens timing offset.

---

## Chapter 4. Methods

This chapter introduces the three pose estimation methods used in the thesis. The scope covers all 17 full-body keypoints for angle and position evaluation; in-depth motion-space evaluation focuses on the elbow joint.

### 4.1 SKT — Stereo Keypoint Triangulation (Main Method)
Performs stereo 2D keypoint detection using YOLOv8m-pose, combines it with DLT triangulation and quality filtering (epipolar error, reprojection error) to obtain 3D keypoints, and computes joint angles using a three-point geometric formula.

### 4.2 FastSAM3D — Monocular Machine Learning Baseline
End-to-end monocular 3D pose estimation, with outputs provided by Aitor. The core idea is a deep learning model that directly regresses 3D keypoint coordinates from a single image, without stereo geometry. This section describes how it is used in this study and the format of its outputs.

### 4.3 SGBM — Dense Stereo Disparity (Alternative Stereo Path)
Extracts 3D depth from dense disparity maps computed via Semi-Global Block Matching, then back-projects 3D keypoint coordinates using the 2D detections. Discusses limitations in texture-poor regions and the scope of applicability relative to SKT.

---

## Chapter 5. Evaluation Framework

### 5.1 Angle Dimension (Full-body 17 Keypoints)
Uses frame-by-frame absolute angle MAE as the core metric, directly comparing each method against Xsens-Fair joint angles; also reports RULA bin agreement as an application-level classification metric. Explicitly acknowledges that this dimension is affected by the Xsens calibration limitation — absolute angles carry a systematic offset, and results should be interpreted alongside the calibration analysis in Chapter 3 — which motivates the motion-space evaluation that follows.

### 5.2 Motion Dimension (Focused on the Elbow Joint)
Addresses the limitation of absolute error by switching to relative-change evaluation: K-frame deltas (angle[i] − angle[i−K], K = 1, 6, 12, 25) capture motion consistency across different time scales, with the Xsens calibration offset cancelled by differencing. Metrics include K-frame delta Pearson, DTW (mean-subtracted, L2-normalized), and path ratio. A unified filtering policy is specified here (no extra smoothing on Xsens, unified 200 ms moving average on camera-based methods) to ensure fair cross-method comparison.

### 5.3 Position Dimension (Full-body 17 Keypoints)
3D joint-position accuracy evaluation: bone-length consistency, pelvis-relative MPJPE, and anthropometric scale comparison (including the note that FastSAM3D requires an external scale correction).

---

## Chapter 6. Results

### 6.1 Angle-Dimension Results
Comparison of frame-by-frame absolute angle MAE and RULA bin agreement across methods, including a left/right elbow asymmetry analysis; results are interpreted in light of the Xsens calibration limitation.

### 6.2 Motion-Dimension Results
K-frame delta Pearson across time scales (K = 1 → 25), and comparison of DTW and path ratio.

### 6.3 Position-Dimension Results
Bone-length consistency, pelvis-relative MPJPE, and anthropometric scale (including the note that FastSAM3D requires a scale correction factor of 0.786).

### 6.4 Cross-method Synthesis
Summary table of headline metrics across the three dimensions, analyzing the strengths and applicability boundaries of each method.

---

## Chapter 7. Discussion

### 7.1 Results Analysis and Applicability Boundaries
Based on the three-dimension results in Chapter 6, compares the performance of stereo and monocular methods across the angle, motion, and position dimensions. Analyzes the competitiveness of monocular ML in the angle and motion dimensions, and the irreplaceable value of stereo vision in the position dimension (absolute scale), geometric quality signals, and out-of-distribution robustness. Identifies the applicability boundaries — in which industrial scenarios each method type should be preferred.

### 7.2 Study Limitations
Honestly states the scope limitations of this study: single subject, single controlled scenario, motion-dimension evaluation limited to the elbow joint; the non-absolute-GT nature of Xsens as a reference system; sensitivity to segment-detection parameters and filter-window choices. Discusses how these limitations affect the generalizability of the conclusions and which can be addressed in future work.

---

## Chapter 8. Conclusion and Future Work

### 8.1 Main Conclusions
Summarizes the three-dimension evaluation results and the conclusions on the applicability of stereo vision for industrial ergonomic monitoring.

### 8.2 Future Work
- Validation and evaluation on a new dataset
- Real-time GPU deployment
- Exploration of Transformer-based model fusion

---

## Appendix

- A. Stereo calibration parameters (intrinsic and extrinsic matrices)
- B. Full per-segment result tables and sensitivity analysis

---

*Draft version, pending revision. Last updated: 2026-06-02.*
