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

### Research Questions

- **RQ1 — Accuracy:** *How closely do stereo-based SKT and monocular FastSAM3D agree with an Xsens-derived reference in estimating ergonomically relevant human posture?*
- **RQ2 — Robustness:** *How does the performance of the two approaches vary across different distances, viewpoints, and occlusion conditions?*
- **RQ3 — Deployment feasibility:** *How feasible is the real-time deployment of the selected SKT pipeline for industrial ergonomic assessment?*

In RQ1, accuracy refers to agreement with an Xsens-derived reference rather than physical accuracy established using absolute ground truth.

- Conclude the chapter by defining the scope and contributions; activity recognition is not evaluated as a separate task.

## Chapter 2: Background and Related Work

- Explain why ergonomic assessment depends on joint angles and clarify that complete automatic RULA/REBA scoring is outside the implemented scope.
- Explain why the core accuracy metric is joint angle rather than 3D joint-position error (MPJPE): occupational ergonomic risk is determined by posture -- RULA/REBA bin risk directly by whether a joint angle falls into a hazardous range, so angle error maps directly onto risk-category agreement, while a joint's raw 3D displacement in centimeters does not.
- Note the supporting technical reason: MPJPE would first require aligning SKT (stereo metric geometry), FastSAM3D (monocular plus human-prior scale), and Xsens (IMU plus kinematic model) to one shared coordinate frame and scale -- a harder, arguably ill-posed problem that angle avoids, since it is computed from three local joint points and is inherently invariant to translation, rotation, and scale differences.
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
- Treat bone-length coefficient of variation as a per-result diagnostic reported alongside angle results (Chapter 6) rather than as an independent results subsection; it is most informative as a geometric-failure indicator at long range.
- State explicitly that MPJPE and other 3D joint-position error metrics are not used as formal accuracy metrics in this thesis; see Chapter 2 for the rationale (risk-category relevance and coordinate-frame/scale invariance).

### 5.4 Output Reliability and Fair Comparison

- Report valid-frame coverage, missing keypoints, discontinuities, and geometric quality information.
- Use common time intervals, common valid frames, and explicit filtering rules.
- State the dataset, joint, reference, and configuration behind every result instead of mixing numbers from different conditions into one ranking.

## Chapter 6: Experimental Results

Five subsections rather than six: the former "Output Reliability and Failure Cases" subsection has no dataset of its own -- its content is redistributed into 6.1 (quality-signal explanation, walking misdetection case) and 6.4 (long-range geometric failure mechanism).

### 6.1 Angle Results: A Distance-by-Complexity Ladder

- Organize the main results along one table spanning distance and action complexity: single-joint right-elbow flexion near and far (Fanbo7 ~180 cm, Fanbo4 ~410 cm), eight-joint single-DOF motion near and far on the same session (Fanbo9 A257/A255, same subject and calibration, two stereo cameras), full-body walking (Fanbo3), and the longest, distance-varying sequence (2025 Aitor, ~249 s, eight ergonomically relevant joints). Error rises with both distance and action complexity.
- Tag every number with dataset, joint scope, reference (FastSAM3D comparison trajectory or Xsens-derived reference), and post-processing chain (pipeline version, detector, position-level configuration); never mix YOLOv8m and YOLO11l numbers in the same column, and re-export the formal table under one locked protocol rather than pooling numbers from different historical summaries.
- Use 2025 Aitor as the main RQ1 result: MAE, median, p95, bias, ergonomic angle-category agreement, and per-method coverage against the Xsens-derived reference on common valid frames, with FastSAM3D agreement reported alongside for consistency with the other datasets.
- Use Fanbo3 walking as the complex-action check spanning the same eight joints; attribute the elbow error mainly to wrist mislocalisation during arm swing (YOLO places the wrist keypoint near the hip rather than the hand, inflating the computed elbow flexion), corroborated by FastSAM3D's substantially lower error on the same recording.
- Use Fanbo7 and Fanbo4 as directional right-elbow cases with full angle time series; tabulate separately from the eight-joint averages.
- Close the subsection with the quality-signal explanation for why coverage differs across rows: detection confidence, stereo quality, epipolar error, and reprojection error each show a clear quartile gap against angle error, without repeating the condition-level ranking that belongs to 6.4.

### 6.2 Motion Results: Read Together with 6.1

- Use the same 2025 Aitor recording that carries the main 6.1 result as the headline motion experiment: K-frame Delta Angle (K = 1, 6, 12, 25) time series and scatter plots, Delta MAE, Pearson correlation, segment ROM MAE, DTW, path ratio, and stationary-segment jitter.
- State the paired-reading rule at the start of the subsection: 6.1 is more exposed to Xsens's initial calibration offset, while 6.2's differencing amplifies visual-method frame-to-frame jitter, so poor 6.1 with good 6.2 points to reference bias, good 6.1 with poor 6.2 points to smoothing or quality issues, and both poor indicates a genuine failure. This rule requires 6.1 and 6.2 to share the same recording, which is why 2025 Aitor anchors both.
- Report the XsensNative-versus-XsensFair gap as the reference system's own resolution floor (roughly 6-8 deg at the shoulder, 1-1.5 deg at the elbow, well under 1 deg at the knee): agreement near this magnitude should not be read as a further correctable SKT error.
- Use Fanbo9 A255/A257 as exploratory, camera-configuration motion evidence only, given weak SKT-Xsens offset identification (peak significance about 0.8-1.0 sigma).
- Fanbo4/7 have no retained formal motion summary; do not generate one purely for symmetry unless a specific conclusion requires it.

### 6.3 Ablation: Detector, Pipeline, and Position-Level Post-Processing

- Use the existing 2x2x5-scenario ablation (V1/V2 pipeline times YOLOv8m/YOLO11l, on Fanbo7, Fanbo4, Fanbo3, Fanbo9 A255, Fanbo9 A257) to separate pipeline-change from detector-upgrade contribution: pipeline upgrade improves all five scenarios (up to roughly -53%), while the detector upgrade regresses on three scenarios and only marginally helps on the remaining two.
- Explain the far-distance detector regression using 2D detection-quality signals: at range, YOLO11l shows higher detection confidence but worse left-right (epipolar) and reprojection consistency than YOLOv8m -- confident but not geometrically accurate -- which is especially damaging because the pipeline weights triangulation by confidence.
- State the model-selection history as evidence-appropriate at each point in time: the original YOLOv8m-to-YOLO11l switch was justified under the older pipeline and a single near-distance scenario; later pipeline changes and additional far-distance/multi-camera data shifted the conclusion back to YOLOv8m. Label this trajectory explicitly rather than presenting only the final choice.
- Use the cross-run ranking of the 17 filter-ablation results to show that the only robust post-processing conclusion is "quality filter present versus absent," with no crossover between the two groups; the specific choice among filtered variants is not distinguishable from run-to-run variation and should be reported as such.
- Use the Fanbo4 progression (raw to soft bone-constrained to adaptive bone-plus-KF/RTS) as the strongest position-level post-processing evidence, confirmed on Fanbo3/Fanbo7/Fanbo9; state its boundary explicitly -- the generic adaptive default does not recover the 2025 long-range sequence to its dataset-tuned historical baseline, with the geometric mechanism given in 6.4.

### 6.4 Distance, Viewpoint, and Occlusion: Physical Mechanism

- Use the direct Fanbo7-versus-Fanbo4 pair (same action, same joint, distance as the only varied factor) as the clearest single-variable distance evidence.
- Use the within-recording depth sweep in 2025 Aitor (subject depth spans roughly 2.9-6.5 m within one continuous recording) to explain why far-range error grows: triangulation depth uncertainty grows with the square of distance for a fixed disparity uncertainty; a small fraction of far-range observations fall into near-degenerate disparity and produce implausible depths, visible directly in bone-length-stability diagnostics. This is the project's only within-recording distance sweep and the most direct evidence for the RQ2 distance question.
- State that such observations are physically implausible rather than merely noisy, so confidence-based down-weighting alone cannot correct them; a physical-validity gate upstream of position-level smoothing is required at this range, with formal integration alongside the existing quality filter left for confirmation.
- Use Fanbo9 A255/A257 as a camera-configuration comparison; because distance and viewpoint change together, attribute results to "camera configuration" rather than distance alone.
- Downgrade the existing cross-recording distance-binning analysis (Fanbo7/Fanbo4/Fanbo9 A255/A257) to observational evidence: each contributing recording has almost no internal distance variation (spans under about 1 m), so the resulting "distance curve" mostly reflects between-session differences rather than a controlled within-subject distance response.
- Use the 2025 Baseline/Occlusion/Environmental-Interference scene tags to discuss occlusion; Fanbo6 (upper-body occlusion) and Fanbo8 (black-background recording) remain qualitative only, without a formal quantitative result under a common reference.

### 6.5 Overall Comparison and Failure Cases

- Use one layered summary table rather than a single average rank, keeping the distance-complexity ladder, the ablation findings, and the distance/occlusion evidence separately labeled.
- Summarize SKT-versus-FastSAM3D trade-offs in angle consistency, motion stability, coverage, and interpretable quality signals, and map them back to RQ1 and RQ2; state the position-level post-processing's boundary on the 2025 long-range sequence as an explained, mechanism-backed limitation rather than an unqualified failure.
- Analyze missed keypoints, stereo mismatching, long-range depth divergence, and temporal misalignment as representative failure cases.
- Merge the unsuccessful human-prior and fusion routes (Pose2Sim/OpenSim, MeTRAbs, EasyMocap/SMPL, the geometry-conditioned kinematic prior, and the discontinued FastSAM3D-intermediate-output stereo-fusion route) into one negative-result paragraph: their failure mode is consistent -- anatomically plausible but unfaithful to the recorded motion -- so state it once rather than case by case. Retain FastSAM3D as a main comparison method throughout; only the fusion route was discontinued. GPU candidates and deployment failures stay in Chapter 7.

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
