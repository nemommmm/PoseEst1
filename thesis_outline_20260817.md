# Master Thesis Outline

**Title**: Posture and Activity Detection in Manufacturing Environment Using 3D Vision and AI

**Author**: Fanbo Meng
**Institution**: Chalmers University of Technology, in collaboration with Viscando AB

---

## Chapter 1: Introduction

- Introduce ergonomic assessment in manufacturing and the role and limitations of manual methods such as RULA and REBA.
- Explain the value of camera-based analysis of posture, joint angles, and critical hazardous moments.
- Present the practical aim: rather than medical- or laboratory-grade precision, the system should provide stable and understandable results that are sufficient for ergonomic decisions.
- Introduce occlusion, camera distance, viewpoint, and computational speed as the main challenges.
- Define the research questions, scope, and contributions; activity recognition is not evaluated as a separate task.

## Chapter 2: Background and Related Work

- Explain why ergonomic assessment depends on joint angles and clarify that complete automatic RULA/REBA scoring is outside the implemented scope.
- Introduce YOLO for human-keypoint detection, focusing on the evaluated YOLOv8m and YOLO11l models.
- Introduce stereo vision, camera calibration, and triangulation, leading to SKT.
- Introduce monocular 3D human-pose estimation and human priors, leading to FastSAM3D.
- Explain the operation and limitations of Xsens, treating it as an external reference rather than absolute ground truth.

## Chapter 3: Data and Experimental Environment

- Describe the 2025 dataset and the 2026 Assar recordings, including tasks, subjects, and acquisition conditions.
- Explain the relationship among the stereo-camera, webcam, and Xsens recordings.
- Describe camera positions, recording distances, calibration, and video properties.
- State the data scope, availability, and known limitations.

## Chapter 4: System Methods

- Use one overview diagram to connect video input, YOLO keypoint detection, 3D-pose calculation, and joint-angle output.
- Describe the main SKT steps: stereo rectification, keypoint detection, left-right matching, triangulation, and output processing.
- Explain how FastSAM3D produces 3D pose from monocular video and how its joints are compared with SKT.
- Describe joint mapping, angle definitions, temporal alignment, common-valid-frame handling, and smoothing.
- Smooth camera-derived angles over approximately 200 ms before motion differences are calculated; use DTW for segment-shape comparison rather than time-offset correction.

## Chapter 5: Evaluation Framework

This chapter answers one question: **How are SKT and FastSAM3D evaluated fairly and comprehensively?**

### 5.1 Angle Dimension

- Compare frame-level absolute joint angles using MAE, median, percentiles, and box plots.
- Use agreement between ergonomic angle categories as an application-level metric.
- Interpret results in light of Xsens calibration uncertainty rather than calling every disagreement absolute physical error.

### 5.2 Motion Dimension

- Use K-frame Delta Angle to compare motion trends and peaks over several time scales.
- Delta Angle reduces sensitivity to constant angular offsets but does not remove all systematic differences.
- Use Pearson correlation, segment ROM, and DTW to evaluate motion trends and segment shape.

### 5.3 Position and 3D-Structure Dimension

- Compare bone-length stability, body proportions, and metric-scale provenance.
- Use pelvis-relative position measures as 3D-structure diagnostics without treating Xsens as absolute position ground truth.
- Explain the differences between FastSAM3D and SKT in scale and coordinate definitions.

### 5.4 Output Reliability and Fair Comparison

- Report valid-frame coverage, missing keypoints, discontinuities, and geometric quality information.
- Use common time intervals, common valid frames, and explicit filtering rules.
- State the dataset, joint, reference, and configuration behind every result instead of mixing numbers from different conditions into one ranking.

## Chapter 6: Experimental Results


### 6.1 Angle Results

- Present the absolute-angle distributions and ergonomic-category agreement for SKT and FastSAM3D.

### 6.2 Motion Results

- Present Delta Angle, correlation, ROM, and DTW results and analyze motion-tracking behavior.

### 6.3 Position and 3D-Structure Results

- Present bone-length stability, body proportions, scale, and position diagnostics.

### 6.4 V1/V2 and YOLO Ablation

- Use SKT V1/V2 and YOLOv8m/YOLO11l combinations to distinguish pipeline effects from detector effects.

### 6.5 Distance, Viewpoint, and Occlusion

- Analyze error against camera distance while separating image-resolution, viewpoint, and occlusion effects.
- Retain the source data behind distance plots, box plots, and main tables.

### 6.6 Overall Comparison and Failure Cases

- Use one summary table to compare angle, motion, 3D structure, and output reliability.
- Analyze missed keypoints, stereo-matching errors, scale problems, and temporal misalignment.
- Explain unsuccessful approaches as negative results, including their causes, boundaries, and research value.
- Retain FastSAM3D as a main comparison method; only the route that attempted to use its intermediate output for SKT stereo fusion was discontinued.

## Chapter 7: GPU Acceleration and Real-Time Implementation Evaluation

- Define the implementation goal and processing path while distinguishing offline stereo-video throughput from a complete live system.
- Report model initialization, first-frame time, post-warm-up steady performance, and p50/p95 latency separately, using repeated measurements.
- Compare CPU and GPU throughput while noting the limitations caused by different test environments.
- Report the PyTorch FP32 GPU result and discuss TensorRT FP32/FP16 as deployment negative results because they did not pass the output-equivalence gate.
- Document codec, pixel format, compression settings, file size, and effects on 2D, 3D, angle, and ergonomic categories.
- Summarize current real-time feasibility, acceptable input formats, and the work remaining for a complete live system.

## Chapter 8: Discussion

- Discuss the trade-offs between SKT and FastSAM3D in stability, metric scale, interpretability, operating conditions, and computational cost.
- Explain under which conditions the system is sufficiently reliable for industrial ergonomic analysis.
- Discuss relationships among YOLO model size, speed, pose quality, working distance, hardware cost, and system configuration.
- Label incompletely validated model-selection explanations as hypotheses or engineering judgments.
- State the limitations of data scale, subject count, Xsens uncertainty, and incomplete control of occlusion and distance.

## Chapter 9: Conclusions and Future Work

- Answer the research questions directly and summarize the most reliable findings.
- State the value of the work for industrial pose analysis and subsequent system development.
- Propose stronger independent references, controlled multi-subject studies, a complete live system, full RULA/REBA scoring, and activity recognition as future work.

## Suggested Appendices

- Camera calibration parameters, joint mappings, and angle formulas.
- Full configurations, source statistics, supplementary results, and failure cases.
- Detailed GPU, video-codec, and deployment settings and commands.
- Software versions, code repositories, licenses, and reproducibility information.

## Writing and Experiment Checklist (Not Part of the Formal Contents)

- Both positive and negative results are useful when their causes and scope are explained.
- Call experimentally supported statements results; label incomplete explanations as hypotheses or possible causes.
- Always describe Xsens as an external reference rather than absolute ground truth.
- Report both absolute angles and Delta Angle so that motion results do not hide absolute-angle differences.
- Complement averages with medians, p95 values, and box plots so that a few outliers do not dominate the message.
- Preserve video parameters, calibration, model versions, run configurations, code entry points, and plot source data for reproducibility.
