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
- Define the project boundary: this thesis implements and evaluates the vision-processing core of a broader industrial ergonomic-assessment system, including pose estimation, stereo reconstruction, common angle evaluation, and computational validation. Live acquisition, transport, user interfaces, risk communication, and the complete ergonomic decision chain remain outside the implemented scope.

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
- Introduce stereo vision in enough mathematical detail to make the SKT implementation locally understandable: the pinhole projection $s\tilde{x}=K[R\mid t]\tilde{X}$, intrinsic/extrinsic calibration, lens distortion, epipolar geometry, rectification, DLT triangulation, reprojection error and positive-depth checks. Derive $Z=fB/d$ and the first-order sensitivity $|\delta Z|\approx Z^2|\delta d|/(fB)$ to establish the distance limitation used later. Include one compact rectified-stereo geometry figure.
- Introduce monocular 3D human-pose estimation and human priors, leading to FastSAM3D.
- Explain the operation and limitations of Xsens, treating it as an external reference rather than absolute ground truth.

## Chapter 3: Experimental Data and System Methods

This chapter answers one continuous question: **How is the same physical activity recorded by three systems and converted through their respective processing paths into synchronized joint-angle sequences that can be compared within one framework?** It connects data acquisition to system processing, but does not define evaluation metrics or report experimental results.

### 3.1 Study Design, System Overview and Roles

- Treat each recording as the basic experimental unit. Describe how stereo video, monocular video and Xsens data observe the same activity or a corresponding scene, while identifying which streams were genuinely captured simultaneously and which were only matched or aligned afterwards.
- Use one shared--separate--shared overview diagram that mirrors the later subsections: the top contains only the activity or scene observed by the available systems; acquisition then separates into Viscando stereo-SKT, monocular FastSAM3D and body-worn Xsens paths because their devices, calibration and source timing differ. Sections 3.4--3.7 converge through common joint mapping, validity handling, temporal alignment and configuration provenance into evaluation-ready angle sequences. Chapter 4 then branches those sequences into absolute-angle and motion evaluation.
- Define the systems by measurement principle: SKT is visual observation plus stereo geometry; FastSAM3D is visual observation plus a learned human-body prior; Xsens is body-worn sensing plus a kinematic model. SKT and FastSAM3D are the evaluated vision methods; Xsens is a synchronized external reference rather than a third candidate vision method or physical ground truth.
- Distinguish acquired or received source files from project-generated outputs. Camera videos and exported Xsens MVNX files are project inputs, although MVNX already contains vendor sensor fusion and is not raw IMU data. SKT NPZ files, FastSAM3D/EasyErgo TRC trajectories and derived angle files are system outputs rather than new raw datasets.

### 3.2 Datasets, Acquisition Environment and Experimental Use

#### 3.2.1 Recordings and Scenes

- Describe the 2025 dataset and the 2026 Assar recordings, including subjects, tasks, actions, duration, distance, viewpoint and occlusion conditions.
- Summarize the 2026 recording log without expanding every session into a case study: Fanbo1--3 cover walking, Fanbo4 and Fanbo7 right-elbow poses, Fanbo5 box carrying, Fanbo6 upper-body occlusion, Fanbo8 a black background, and Fanbo9 one continuous movement. Note unusable, missing or occluded A255/A257 streams where relevant, and keep detailed ID/offset mapping in the reproducibility material.
- Define A255 and A257 at first use as identifiers for two complete Viscando stereo sensors installed in 2026. Each sensor contains its own left--right camera pair and is processed independently; they are not four views of one SKT reconstruction.
- State the actual relationship among the stereo cameras, webcam and Xsens in each recording, and identify the role of each dataset in the thesis.
- Separate recordings used for the principal comparison, ablation, distance/viewpoint/occlusion analysis and deployment tests so that measurements from different conditions are not silently pooled.

#### 3.2.2 Acquisition Devices and Source Files

- Stereo input: one Viscando stereo sensor per SKT stream, its left and right videos, resolution, nominal frame rate, hardware frame identifiers and timestamps. Monocular input: webcam video supplied to FastSAM3D, including its resolution, frame rate and recording relationship to the other streams. Xsens input: the received MVNX file, nominal sampling rate and calibration record. Include one compact two-panel hardware figure showing a Viscando sensor and the participant wearing Xsens.
- Use one acquisition table to summarize recording, devices, source files, resolution, frame rate, duration, synchronization and missing modalities. Identify MVNX as a received Xsens export and distinguish the project-generated NPZ, TRC and angle outputs.

#### 3.2.3 Stereo-Camera Calibration and Validation

- Describe camera placement, the 41~cm physical stereo baseline, intrinsic and extrinsic parameters, resolution, recording settings, when each calibration was obtained and which recordings it supports. Explain that the recovered baselines of approximately 41.00--41.27~cm are physically reasonable consistency checks, not independent human-joint accuracy evidence.
- Fix the three archived calibrations used in the study: `camera_params_2025.npz` for the 2025 pair; `camera_params_A255.npz`, obtained from the A255 SensorCalibration recording; and `camera_params_A257.npz`, obtained from SensorCalibration supplemented by SiteCalibration because the sensor-only recording contained too few accepted frames. Each experiment uses the corresponding frozen file; calibration is not adjusted against the pose results.
- Use one compact table to report image size, accepted calibration frames, baseline, rectified vertical disparity and rigid-alignment RMSE. Define these values precisely as internal target-consistency checks over the accepted calibration frames, not independent position ground truth in the human recordings. Move the complete $K,D,R,T,E,F$ matrices and configuration search to an appendix.
- Keep calibration results in this subsection; Section 3.3.1 explains how the fixed parameters enter rectification, projection matrices and triangulation.

#### 3.2.4 Activity-Segment Annotation for the 2025 Recording

- Define the four manual scenario groups used in Section 5.1: Baseline (normal walking at 17--32 s and 220--240 s); Dynamic Action (the short squat at 66--69 s); Occlusion (32--62 s, 87--97 s, 130--140 s and 164--170 s); and Environmental Interference (chair interaction at 140--160 s and box lifting at 214--218 s). Report the boundaries as rounded recording times.
- An historical label also marked 156--160 s as `Squatting (Check)`, overlapping the 140--160 s chair-interaction interval. Preserve the precedence used by the actual analysis: assign this overlap only to Environmental Interference and do not count it again as Dynamic Action.
- Describe these groups as manual visual annotations introduced for scenario-level analysis, not automatically generated labels or ground truth.

#### 3.2.5 Experimental Use and Evidence Boundaries

- Use a compact table to map every recording to angle comparison, motion comparison, ablation, robustness analysis or deployment testing and to the corresponding research question.
- Separate controlled variations from confounded conditions. Mark the recordings used for controlled one-degree-of-freedom elbow flexion; Chapter 4.2 explains why this action is selected for motion evaluation.
- State limitations in subject count, action coverage, missing frames, duration and available recordings.
- Explain the confounding among distance, viewpoint, activity and recording year, distinguishing controlled comparisons from observational evidence.
- State that no independent optical ground truth was recorded; Xsens is a synchronized external reference rather than absolute truth.

### 3.3 Three System Processing Paths and Outputs

Use the same narrative order for all three paths: **input $\rightarrow$ processing $\rightarrow$ output $\rightarrow$ observable quality information or method limitation.**

#### 3.3.1 SKT: Stereo Visual Observation and Geometric Measurement

- Describe the canonical pipeline in six numbered stages that correspond exactly to the pipeline figure. For every stage state its input, operation, purpose and output: (1) pair frames using hardware IDs; (2) detect COCO-17 keypoints and track one person in each view; (3) rectify 2D points and check the stereo pair; (4) apply soft epipolar adjustment and confidence-weighted DLT, then check disparity, positive depth and reprojection; (5) recover short 2D failures from neighbouring observations and re-triangulate; (6) store accepted 3D joints together with their quality and provenance fields.
- Add one real 2025 frame example showing synchronized left/right skeleton overlays and the corresponding epipolar error, reprojection error and recovered depth. Present it as a processing example, not a result figure.
- Explain confidence-weighted triangulation, the soft epipolar constraint, and the confidence, epipolar-error, reprojection-error and disparity signals produced by reconstruction. Keep inference-time acceptance rules distinct from the evaluation masks in Section 3.5.
- Define the positive-disparity requirement and the minimum disparity of 45 px as a physical validity boundary. Relate it to $Z=fB/d$ and the approximately 10 m maximum depth implied by the calibrations used in this study, without presenting the earlier 1.5 px development setting as part of the final method.
- Explain how small and inconsistent 2D keypoint errors can be amplified in stereo depth and then propagate to joint angles.
- Refer to this configuration as the **canonical SKT pipeline**. Do not use the historical label V2 as a synonym for it.

#### 3.3.2 FastSAM3D: Monocular Visual Observation and a Human-Body Prior

- Organize the subsection into three explicit layers: (1) the published Fast SAM 3D Body monocular reconstruction, (2) Aitor Iriondo Pascual's EasyErgo/OpenSim conversion, and (3) the TRC loading and joint mapping performed in this thesis. State unambiguously that the upstream model and biomechanical exporter are not contributions of this thesis.
- Explain the upstream path at a readable system level: one RGB webcam stream, person and coarse 2D-pose localization, camera/FOV cues, image encoding, MHR body decoding, and 3D mesh/joint output. Explain that its output combines image evidence with a learned body representation; do not present it as measured stereo depth.
- Redraw the supplied workflow as a method-boundary figure. Show manual participant height entering subject/model scaling, followed by skeleton/marker translation, practical corrections and the OpenSim/IK/export branch. Keep the TRC trajectory used by this thesis on a solid path and show downstream products not evaluated here separately.
- Attribute the biomechanical conversion to Aitor and cite both the Fast SAM 3D Body paper and Aitor's public FastSAM3DToOpenSim repository. Explain that manual height supports subject scale but is not an additional visual measurement. Record smoothing/grounding as external upstream provenance whose exact settings are unavailable; do not turn it into a common evaluation filter.
- Identify the FastSAM3D/EasyErgo TRC trajectories actually used by the project: 2025 files are 12.5 Hz, 42 markers and millimetres; 2026 files are 30 Hz, 46 markers and millimetres. The TRC headers do not declare semantic axes, so preserve the supplied axes and rely on rotation/translation/scale-invariant local angles after named-marker mapping.
- State the version boundary: Aitor's implementation has continued to evolve and the current public repository documents a 24-marker, metre-based Y-up export. Where that differs from the archived 42/46-marker EasyErgo files, the archived files and analysis code used in this project take precedence; the repository establishes attribution and general conversion purpose rather than exact historical settings.
- Explain that a strong prior may allow an anatomically plausible estimate when image evidence is weak and that FastSAM3D does not apply SKT's explicit stereo-consistency rejection. Treat high coverage as availability, not as proof of correctness; do not claim that the method always produces an output.
- State that the available TRC trajectory is not equivalent to SKT output and does not contain the per-view 2D intermediate evidence required for direct insertion into the stereo triangulation path.

#### 3.3.3 Xsens: Body-Worn Sensing and a Kinematic Model

- Follow the same input--processing--output logic used for the two vision systems, while keeping the proprietary method description concise: (1) body-worn inertial signals plus subject setup, (2) MVN sensor fusion and biomechanical reconstruction, and (3) MVNX output plus thesis-side parsing.
- Show one compact method-boundary figure. Inputs are the segment-mounted sensors, body dimensions, sensor assignment and calibration posture; MVN processing combines inertial propagation with sensor-to-segment calibration, a linked biomechanical model and constraints; output is a timestamped MVNX stream. State that only MVNX parsing and common-angle construction belong to this thesis.
- Explain that sensors measure their own motion rather than anatomical joint centres. Subject dimensions and calibration map those measurements to segment frames; incorrect dimensions, shifted placement or an imperfect calibration posture can therefore create a persistent offset.
- Describe the practical benefit and corresponding limitation together: Xsens does not depend on camera visibility and supplies a dense trajectory, but positions and angles are already model-based, fused and vendor-processed rather than raw IMU measurements or independent optical truth.
- Record the verified file properties: the 2025 Aitor file is MVN 2024.0 and the 2026 Fanbo files are MVN 2024.2; both are 60 Hz, 23-segment, 22-joint FullBody/HD exports. Treat file metadata as authoritative over current product defaults.
- State how the parser uses the files: retain normal motion frames, convert milliseconds to seconds and metres to centimetres, and map upper arm--forearm--hand origins to shoulder--elbow--wrist and upper leg--lower leg--foot origins to hip--knee--ankle.
- Explain why the principal reference is recomputed from segment positions using the shared geometric angle definition. Retain vendor-native joint and ergonomic angles only to explain that choice and audit provenance; they are not a second result stream or a reference-resolution experiment.
- Conclude explicitly that Xsens is a synchronized external reference, not physical ground truth, because calibration, assumed body geometry and internal fusion remain part of every reported value.

### 3.4 Common Joint Representation and Angles

- Explain first why the systems' own angle outputs cannot be compared directly. SKT produces triangulated COCO-17 points and the evaluated FastSAM3D input is a TRC marker trajectory, whereas Xsens additionally provides vendor-native anatomical and ergonomic angle fields. The historical Xsens extraction also used different field conventions for shoulder and hinge joints. Direct comparison would therefore mix pose-estimation disagreement with angle-definition disagreement.
- Add a compact system-boundary table: available output from each path, whether a native angle exists, and the common comparison quantity constructed in this thesis.
- Map SKT COCO-17 keypoints, FastSAM3D/EasyErgo TRC markers and Xsens segment origins to common shoulder, elbow, wrist, hip, knee and ankle chains. State units, coordinate conventions and which anatomical locations can be matched reliably.
- Apply one vector-angle implementation to all systems. Elbow, hip and knee flexion use the shoulder--elbow--wrist, shoulder--hip--knee and hip--knee--ankle chains, respectively, and are reported as the supplement of each interior angle. Shoulder elevation is the angle between the upper-arm vector and the downward torso vector. Consequently, reported differences do not arise from different angle implementations.
- Explain the joint-specific conventions: elbow/knee are hinge-like flexion measures, hip is a shoulder--hip--knee flexion proxy, and shoulder is elevation relative to the downward torso. These rules differ by the motion represented, but never by measurement system.
- Use angles geometrically recomputed from Xsens segment positions as the principal Xsens-derived reference rather than vendor-native angles. Fix this as the meaning of joint angle for the remainder of the thesis instead of redefining it in later chapters.
- Fix the notation used later: $m\in\{\mathrm{SKT},\mathrm{FS}\}$ denotes a vision method, $\theta_{m,j,t}$ its angle, and $\theta_{X,j,t}$ the geometrically recomputed Xsens-derived reference angle. Use native/geometric qualifiers only where the two Xsens outputs must explicitly be distinguished.
- Justify $\theta_X$ as the reference because it shares the same geometric angle definition as the vision methods. Record the 2026 reference audit and state that the affected results were recomputed using this definition while the saved vision outputs and time offsets were unchanged.

### 3.5 Validity Checks and Missing-Data Handling

- Separate three layers clearly: inference-time triangulation checks; the evaluation-time SKT quality mask; and the additional upper-limb depth-consistency mask examined in Section 5.1.
- Define the evaluation quality mask as it is implemented: for the shoulder, elbow and wrist keypoints, reject estimates with insufficient paired-view confidence or excessive epipolar error. Do not claim that the evaluation mask uses reprojection error; reprojection is an inference-stage quality signal in the current implementation.
- Define the upper-limb depth-consistency rule for the shoulder--elbow--wrist chains and explain that it trades coverage for protection against implausible arm depth. State its role before the on/off result is presented in Chapter 5.
- State that FastSAM3D missing values originate in the supplied TRC trajectory, whereas Xsens availability is determined by temporal overlap between MVNX and the camera sequence. Do not invent stereo-confidence or depth-consistency rules for either system.
- After angle calculation, fill only finite-bounded gaps of at most five camera samples. Longer gaps remain missing. Define common valid timestamps and method-specific coverage in Chapter 4 rather than silently deleting invalid rows here.

### 3.6 Temporal Alignment and Evaluation-Ready Angle Sequences

- Establish the stereo camera timeline from hardware frame identifiers, not assumed video indices or a perfectly uniform frame rate. Match left and right metadata rows by frame ID, retain the left timestamp of each physical pair and preserve missing IDs.
- Use the 2025 recording as a worked example: 3015 left metadata rows and 2882 right rows become 2801 synchronized pairs with a median interval of about 80 ms. Map the 3015-row FastSAM3D trajectory by the saved left-frame indices rather than truncation.
- Map FastSAM3D by synchronized frame index or TRC timestamp. For each recording use $t_X=t_C-\delta_r$ and interpolate the 60 Hz Xsens signal at those query times rather than discarding every $n$th IMU sample; do not extrapolate outside temporal overlap.
- Explain recording-specific camera-to-Xsens offset estimation through a coarse-to-fine search. The primary score is the across-joint median Pearson correlation of short-interval angular changes between SKT and Xsens; angle correlation and rigidly aligned position agreement serve as secondary diagnostics. Save candidate scores and inspect angle curves, recording boundaries and durations for weak peaks, residual misalignment or clock drift.
- State the implemented search resolution (0.10 s coarse, 0.01 s fine) and record the 17.25 s 2025 offset as a concrete example. Clarify that alignment uses one recording-level mapping and does not locally warp motions to improve agreement.
- Record the source and necessary preprocessing of each system output. Differences between recordings are not interpreted as isolated filtering effects when their upstream outputs differ.
- Do not introduce a common moving-average claim across recordings. Retain the provenance of the actual FastSAM3D, SKT and Xsens inputs used for each experiment, and do not interpret cross-recording differences as isolated filtering effects when upstream processing is not uniform or fully observable.
- Produce traceable evaluation-ready angle sequences after joint mapping, temporal alignment, validity checks and explicit missing-value handling. Absolute-angle and motion evaluation begin from the corresponding processed angle series, after which the motion branch applies the $K$-sample difference. The two evaluation paths retain their own validity masks, so their final sample counts need not be identical.

### 3.7 Experimental Configurations and Reproducibility

- Introduce the development background: SKT was revised repeatedly to improve agreement and robustness across distance, viewpoint and difficult detections, and changes that helped one recording did not always generalise. V1 and V2 are two stable historical snapshots retained for comparison, not the full chronological history.
- Distinguish the **canonical SKT pipeline** used in Sections 5.1--5.3 from the **historical V1/V2 configurations** in the archived 2026-07-07 ablation. Historical labels identify configurations evaluated at that time and must not be reused as names for the final pipeline.
- Explain the difference qualitatively. V1 is the earlier frame-wise baseline, relying more on full-frame detections, hard acceptance and gap-based post-processing. V2 is the bundled revision, adding target-region tracking, confidence-aware weighted triangulation and softer epipolar handling, neighbouring-frame recovery, and temporal/depth/bone-length processing. These features explain the intended direction but are not independently isolated by the ablation.
- State the provenance limitation. Some V1 cells read outputs saved by an earlier implementation, while others were generated from historical configurations in the current repository; the historical V2 study coupled several reconstruction and post-processing choices. The ablation can therefore evaluate the pipeline revision as a bundle but cannot identify the contribution of each component separately. Treat YOLOv8m versus YOLO11l as a separate detector factor in the $2\times2$ design.
- Keep mechanisms, processing order and parameters that directly determine physical validity or sample inclusion in the main text. Move exact joint indices, recording-specific offsets, full configuration identifiers, software versions and the wider model search to a reproducibility appendix. Include filter trigger rates only after calculation and verification.

## Chapter 4: Evaluation Framework

This chapter answers one question: **How are SKT and FastSAM3D evaluated fairly and comprehensively?**

Assume that methods and abbreviations are already defined. Open with the evaluation flow figure: processed angles branch into absolute-angle and Delta comparisons, each with its own common valid samples, interpreted alongside method-specific coverage. Do not delete missing rows before differencing or repeat descriptions of the systems and later chapters.

**Two comparison roles, kept distinct.** The main results compare each vision method with $\theta_X$; only this comparison supports statements about agreement with the external reference. The archived ablation compares alternative SKT configurations with one unchanged FastSAM3D trajectory. That method-to-method quantity can rank configurations within the same recording, but it is not an accuracy measure and must never be described as error against truth.

### 4.1 Angle Dimension

- Compare frame-level joint-angle levels to answer how closely each vision method agrees with the Xsens-derived reference.
- Use median absolute error and MAE for typical and overall disagreement, p95 and box plots for the error tail, and bias for persistent over- or underestimation. A box plot is a presentation method rather than a separate metric.
- Retain the angle offset in time-series plots rather than zeroing each curve before comparison.
- Use agreement between predefined ergonomic angle intervals as an application-level metric without presenting it as complete automated RULA/REBA scoring.
- Interpret all differences in light of Xsens calibration uncertainty rather than calling them absolute physical errors.

### 4.2 Motion Dimension

- Explain why absolute-angle error alone cannot describe motion agreement: Xsens may contain an initial angular offset, while the systems can still follow a similar movement over time.
- Focus the main analysis on elbow flexion/extension as a clear one-degree-of-freedom motion that all three systems can calculate from the shoulder--elbow--wrist chain.
- Use one primary K-frame Delta Angle interval, with $K=6$ corresponding to approximately 0.48 s on the main 12.5 FPS camera timeline.
- Use Delta MAE to quantify disagreement in movement magnitude and Pearson correlation to quantify similarity in direction and temporal trend. Interpret them together because high correlation does not imply accurate magnitude.
- State that this controlled single-degree-of-freedom analysis improves interpretability but does not by itself establish performance for complex multi-joint motion.

### 4.3 Fair Comparison and Reliability

- State the central comparison limitation: absolute angles expose Xsens calibration offsets, whereas Delta Angle can amplify frame-level jitter in vision outputs. Report both views together rather than using either one to hide the other system's weakness.
- For recordings supporting the main conclusion, present absolute-angle and motion evidence from the same data. Diagnostic, ablation, and robustness recordings do not need to repeat the full analysis when it does not serve a clear purpose.
- Restrict direct head-to-head metrics to the same temporal overlap and common valid timestamps, while also reporting each method's own valid-frame coverage so that missing difficult frames cannot improve an apparent accuracy result.
- Report missing keypoints, discontinuities, and geometric quality signals as reliability evidence and as aids for explaining failure cases.
- Attach the dataset, joint scope, reference definition, and pipeline configuration to every reported result instead of mixing values produced under different conditions.
- Treat bone-length stability, metric scale, and coordinate-system differences as diagnostics and discussion material, not as a separate position-accuracy dimension. MPJPE and related 3D position-error metrics are outside the formal evaluation because no independent position ground truth is available and the application is angle-centred.

**Writing note for Chapter 5 (not formal Chapter 4 content):** The motion-results section may use a three-panel scatter plot for Delta Xsens--Delta SKT, Delta Xsens--Delta FastSAM3D, and Delta SKT--Delta FastSAM3D. Use common axes, the $y=x$ line, a fitted line, and Pearson correlation. The first two panels show agreement with the reference; the third shows agreement between the two vision methods and must not be used as an accuracy ranking.

## Chapter 5: Experimental Results

This chapter follows one compact evidence chain: absolute-angle and motion agreement are first evaluated on the same 2025 long recording; the distance dependence of both methods is then examined *within* that same recording and cross-checked on two 2026 right-elbow recordings; finally, the archived ablation explains the historical pipeline and detector decision. Xsens remains an external reference system, so the chapter reports agreement rather than known physical error. All principal agreement results compare each vision method with $\theta_X$; the archived method-to-method ablation is identified explicitly in Section 5.4.

### 5.1 Absolute-Angle Agreement: Main 2025 Experiment

- Use only the frozen canonical SKT pipeline with YOLOv8m on the 2025 Aitor recording, covering eight ergonomically relevant joints.
- Evaluate each vision method against $\theta_X$ separately. Restrict the head-to-head table to timestamps where both methods are valid so that neither benefits from dropping difficult frames, and report each method's own coverage in the same table. This is not a direct SKT-versus-FastSAM3D difference metric.
- Use one per-joint MAE table for exact values and a paired error-distribution/coverage figure for typical differences, tails and availability; do not make a separate figure for every metric.
- Report ergonomic angle-interval agreement as one supporting result rather than expanding it into a full RULA/REBA score.
- Interpret all values as agreement with the Xsens reference, not known physical error.

### 5.2 Motion Results: Read Together with 5.1

- Use the same 2025 Aitor recording, restrict the analysis to the left and right elbows, and use only the primary K=6 interval defined in Chapter 4.
- Use one three-panel correlation figure: Delta Xsens versus Delta SKT, Delta Xsens versus Delta FastSAM3D, and Delta SKT versus Delta FastSAM3D. Apply common axes and show the $y=x$ line, a fitted line, and Pearson correlation.
- Use one compact table for Delta MAE and Pearson correlation of SKT and FastSAM3D against the Xsens-derived reference. The third method-to-method correlation is explanatory only.
- Pair actual angle and Delta curves. Retain the previously inspected May 11 right-elbow window (22.5–43.8 s), redrawn from current outputs, with full-sequence context. The window illustrates behaviour and does not determine the quantitative sample selection.
- Limit the conclusion to movement magnitude and trend agreement. Correlation is not interpreted alone as accuracy and does not replace the absolute-angle evidence in Section 5.1.

### 5.3 Distance Dependence and Recording Conditions

This section answers RQ2. Its primary evidence is a within-recording distance sweep, because that is the only place in this study where distance varies while subject, session, camera, and calibration are all held fixed.

- **Primary evidence -- the 2025 recording's own depth sweep.** The subject's torso depth spans roughly 2.8 m to 6.4 m (p10--p90) *within the single recording already used in Sections 5.1 and 5.2*, so no additional data is introduced. Bin each method's per-frame disagreement with $\theta_X$ by estimated torso depth and report the eight-joint mean per bin alongside frame count and coverage.
- The result separates the two methods by mechanism: SKT disagreement grows steeply with distance (roughly 16° in the 3--4 m band to roughly 28° beyond 7 m), whereas FastSAM3D stays approximately flat across the same range (roughly 13° in both). The gap between the two methods therefore widens from about 5° at close range to about 14° at the far end.
- Interpret this as the expected consequence of the two measurement principles, not as a generic quality ranking: stereo depth uncertainty grows with the square of distance and enters SKT's angle only after propagating through a 3D position estimate, while a prior-constrained model fit with an externally supplied body scale is largely distance-independent. This is the mechanism behind the overall ordering reported in Section 5.5.
- State the limitation of the horizontal axis honestly: depth is estimated from the same SKT reconstruction being evaluated, so the axis is not an independent measurement. It is nevertheless the only available option, because FastSAM3D's scale derives from a manually supplied subject height and Xsens carries no position relative to the camera. Use the torso centroid rather than a single joint, and note that the physical-validity bound applied in Chapter 3 removes the implausible far-depth outliers that would otherwise distort the binning.
- **Cross-recording check -- Fanbo7 (approximately 1.8 m) and Fanbo4 (approximately 4.1 m).** Both contain right-elbow flexion and both are evaluated with the canonical SKT pipeline and YOLOv8m on common valid timestamps. Combine the June 29 scene photographs, paired elbow time series, and one compact table of MAE, median, common sample count and method-specific coverage. Add no other recordings.
- Corrected values after the reference audit defined in Chapter 3: SKT/FastSAM3D MAE is 6.68°/5.01° for Fanbo7 and 11.49°/7.36° for Fanbo4. Source hashes and calculations are in MasterThesis/figure/chapter6/evidence_manifest.json; experimental files are unchanged.
- Both methods disagree more on Fanbo4, consistent with the within-recording sweep, but movement amplitude, duration, viewpoint and image scale differ between the two recordings as well. Treat this pair as corroboration of the direction rather than as an independent measurement of the distance effect.
- Viewpoint and occlusion lack controlled quantitative results with a common reference and are treated as limitations in Chapter 7.

### 5.4 Pipeline and Detector Ablation

- Open with the purpose: determine whether the observed SKT improvement comes
  from the V1--V2 pipeline revision or merely from changing to a larger pose
  detector. The two factors must therefore be changed one at a time.
- Introduce the five 2026 conditions before presenting the figure: Fanbo7 and
  Fanbo4 evaluate the right elbow at two distances, Fanbo3 contains walking,
  and Fanbo9 supplies two simultaneous stereo-camera views. Identify V1 and V2
  explicitly as the historical configurations defined in Section 3.7, not as
  names for the canonical pipeline used earlier in this chapter.
- Use the July 7 2x2x5 matrix in two fixed-factor plots: pipeline change with YOLOv8m fixed, and detector change with V2 fixed. Retain the full matrix in the supporting data rather than duplicating the figure in a table.
- This section alone uses SKT--FastSAM3D disagreement. It is admissible here because the comparison target is identical across all four configurations, making the within-recording ranking interpretable even though the level is not an accuracy figure. Do not carry these values into Sections 5.1--5.3.
- Pre-empt the circularity objection explicitly: selecting SKT's configuration against FastSAM3D and then comparing SKT with FastSAM3D would be circular if the ranking depended on that choice of target. It does not -- on the Fanbo9 dual-camera session, where a reference comparison is also available, the detector ordering is the same under both comparison targets in all eight configuration cells. Say this in one sentence rather than leaving it for the examiner to raise.
- Retain two robust findings: with YOLOv8m fixed, V2 reduces or approximately preserves disagreement in all five scenarios; under V2, YOLOv8m is better in three scenarios, while YOLO11l leads the other two by only 0.05 and 0.78 degrees.
- State the trade-off explicitly. The historical V2 configuration adds processing and configuration
  complexity but improves agreement consistently across the tested scenarios;
  YOLO11l increases detector-side computational demand without a consistent
  accuracy benefit. Treat historical V2 + YOLOv8m as the preferred combination
  within the archived study and as evidence for retaining YOLOv8m, without
  equating historical V2 with the canonical pipeline or claiming that YOLOv8m
  is universally more accurate.
- Many other pose detectors, stereo models, filter settings and fusion routes
  were explored during development. Do not turn 5.4 into an inventory: keep
  the controlled V1/V2--YOLO comparison in the main text and move the wider
  model search and failed routes to an appendix or the limitations discussion.

### 5.5 Summary of Findings

- Open with the direct chapter-level conclusion: FastSAM3D is the stronger
  method for the angle-centred ergonomic assessment under the tested
  conditions. V2 improves SKT but does not close the main-result gap.
- Answer RQ1 from the paired 2025 evidence. FastSAM3D is closer to Xsens
  overall and has shorter error tails; SKT's small absolute elbow advantage
  reverses in the Delta comparison because FastSAM3D's elbow difference is
  largely a persistent offset. Do not call Xsens physical ground truth.
- Keep the reference boundary concise: Xsens is a comparison system rather
  than physical ground truth, so the result concerns relative agreement.
- Answer RQ2 within the available evidence. Coverage and disagreement are
  independent; the filter explicitly trades one for the other. The 2025
  scenario groups and the two 2026 distance conditions show sensitivity to
  recording conditions, but the latter do not isolate distance, viewpoint or
  occlusion because the recordings differ in several ways.
- Interpret the methods without balancing measured evidence against untested
  potential: FastSAM3D gives more stable angles under the tested conditions.
  SKT provides metric geometric output, but the present study does not validate
  its position accuracy, and that capability does not offset the angle result.
- Close with the ablation decision: the archived experiment supports retaining
  YOLOv8m and shows a contribution from the historical pipeline revision; the
  final pipeline used in Sections 5.1--5.3 is defined separately in Chapter 3.
  Neither result establishes a universally best configuration. Leave RQ3
  computational feasibility to the real-time deployment chapter.

## Chapter 6: GPU Acceleration and Real-Time Implementation Evaluation

- Define the implementation goal and processing path while distinguishing offline stereo-video throughput from a complete live system.
- Report model initialization, first-frame time, post-warm-up steady performance, and p50/p95 latency separately, using repeated measurements.
- Compare CPU and GPU throughput while noting the limitations caused by different test environments.
- Report the PyTorch FP32 GPU result and discuss TensorRT FP32/FP16 as deployment negative results because they did not pass the output-equivalence gate.
- Document codec, pixel format, compression settings, file size, and effects on 2D, 3D, angle, and ergonomic categories.
- Summarize current real-time feasibility, acceptable input formats, and the work remaining for a complete live system.

## Chapter 7: Discussion

### 7.1 Interpretation of the Method Comparison

- Interpret the principal results from Chapter 5 without repeating its tables. FastSAM3D produces the more stable trajectories for the present angle-centred task, whereas SKT first reconstructs metric 3D positions and then derives angles from local joints.
- Distinguish metric-scale output capability from validated position accuracy. No independent position ground truth is available, so metric coordinates must not be presented as a demonstrated accuracy advantage of SKT.
- Explain the methodological contrast: SKT depends on 2D keypoints, cross-view correspondence, and triangulation, whereas FastSAM3D regularizes the body with a learned human prior. The prior can suppress independent-joint depth variation but may also produce a plausible pose when image evidence is weak.
- Reserve a focused analysis of why SKT underperforms, organized as 2D localization or occlusion problems, inconsistent cross-view observations, depth and limb-vector errors, and finally angle jitter or rejected output. Mechanisms not directly verified on the canonical run must remain possible explanations rather than established causes.

### 7.2 Implications for Industrial Ergonomic Assessment

- Return to the practical goal: determine whether the methods can assist in locating critical or high-risk postures rather than replace professional ergonomic judgement or provide medical-grade measurement.
- Interpret absolute-angle agreement, motion agreement, and coverage together; a valid output, a small angular difference, and a stable movement trend are separate requirements.
- Position the current system as an aid for posture screening and human review. Long-range, occluded, cluttered, or discontinuous outputs still require inspection.
- Clarify that complete RULA/REBA assessment is not implemented because load, muscle use, hand coupling, and other worksheet decisions are not inferred by the current pipeline.

### 7.3 Engineering and Deployment Trade-offs

- Connect the algorithmic results in Chapter 5 with the implementation results in Chapter 6. A larger detector does not consistently improve stereo angles; triangulation itself is inexpensive, while two-view detection and decoding dominate computational cost.
- Discuss pose quality, working distance, coverage, speed, GPU requirements, hardware cost, and video-input quality as a coupled product-design problem rather than selecting a configuration from one accuracy or FPS value.
- Interpret the PyTorch FP32, TensorRT, and compression experiments through the rule that speed is useful only when output equivalence is preserved.
- Separate two conclusions explicitly: the offline computational core is real-time capable on the tested GPU, but a complete live ergonomic-assessment system has not been demonstrated.

### 7.4 Limitations and Threats to Validity

- Reference limitation: Xsens is an external comparison system rather than physical ground truth, and its calibration, body model, and registration affect absolute-angle comparisons.
- Data limitation: the number of participants and recordings is limited, and action coverage, duration, and available modalities are uneven.
- Experimental-control limitation: distance, viewpoint, occlusion, activity, and recording year are partly confounded, preventing single-factor causal interpretation across recordings.
- Method and traceability limitation: parts of the historical V1/V2 study use archived configurations or FastSAM3D as the comparison trajectory and therefore support configuration choice rather than independent physical-accuracy ranking.
- System-scope limitation: the evaluated path processes stored videos and excludes live capture, stream transport, user interfaces, risk communication, downstream interfaces, and long-duration field stability.
- Label incompletely validated mechanisms and model-selection explanations as hypotheses, possible causes, or engineering judgments.

## Chapter 8: Conclusions and Future Work

### 8.1 Conclusions

- Answer the three research questions concisely without repeating every experimental value. RQ1 summarizes relative agreement with the Xsens-derived reference; RQ2 summarizes observed condition sensitivity within the available evidence; RQ3 establishes only the real-time computational feasibility of the SKT core, not completion of a live system.
- State the project value: rather than proving one method universally superior, the thesis establishes a common comparison framework and identifies the measured behaviour, limitations, engineering costs, and development implications of two vision routes for industrial ergonomics.
- Reaffirm the thesis's place in the larger project: it implements the processing core from recorded inputs through pose, angles, evaluation, and GPU validation, while the complete field system still requires integration.

### 8.2 Future Work

#### A. Further Development and Validation of the SKT Pipeline

- Trace how 2D keypoint errors propagate through cross-view correspondence and triangulation into 3D positions and joint angles, using reproducible canonical-pipeline failure cases.
- Investigate more reliable cross-view person/joint association, uncertainty-aware triangulation, quality-adaptive processing, and human or bone-length constraints validated across datasets; do not assume that one filter or prior will transfer universally.
- Use controlled experiments to vary distance, viewpoint, and occlusion independently, validate positions and angles against a stronger external reference, and extend the study to more participants, activities, and real industrial tasks.

#### B. Completion of the End-to-End Real-Time System

- Extend the stored-video processing path into a complete chain: live stereo capture, hardware synchronization, validated lossless transport, online detection and triangulation, ergonomic analysis, and visualization, warning, or data export.
- Measure end-to-end latency, long-duration synchronization and stability, dropped-frame recovery, and field maintainability rather than only short-window algorithm throughput.
- Re-evaluate the pipeline on the intended GPU, edge, or embedded hardware. Adopt TensorRT, FP16, hardware decoding, or other accelerations only after they pass the same output-equivalence checks.
- Integrate the load, muscle-use, interaction, and contextual information required for complete RULA/REBA assessment and, where useful, activity recognition, so that pose estimation becomes one component of a full industrial decision workflow.

## Suggested Appendices

### Appendix A: Calibration, System Mapping, and Common Definitions

- Complete intrinsic and extrinsic parameters, calibration-search settings, and calibration-validation data.
- Joint mappings, coordinate conventions, and full angle definitions not expanded in the main text.
- Detailed mappings among recordings, devices, activity segments, time offsets, and experimental use.

### Appendix B: Pipeline Configurations and Supplementary Experiments

- Identify the single canonical SKT configuration, run identifier, and reproduction entry point used by the principal results, clearly separating it from legacy or similarly named configurations.
- Preserve complete V1/V2, YOLOv8m/YOLO11l, filter, bone-constraint, Stage C, and other detector/pipeline settings and results.
- Include supplementary success and failure cases and SKT error diagnostics, explicitly separating historical outputs from canonical-pipeline evidence.

### Appendix C: Complete Numerical Results

- Full joint-level MAE, median, percentiles, bias, coverage, Delta Angle, and correlation tables.
- Distance and scenario groups, supplementary scatter plots, ablations, and figures omitted from the main text to preserve its argument.

### Appendix D: Deployment and Reproducibility

- Detailed CPU/GPU repeats, initialization and steady-state timing, TensorRT equivalence checks, and outputs.
- Video codecs, pixel formats, compression parameters, FFmpeg commands, and input-equivalence results.
- Hardware and software versions, configuration files, input/output files, source hashes, repositories, licences, and complete reproduction instructions.
- Reference every appendix component used by an argument in the main text so that the appendices do not become an unstructured data archive.

## Writing and Experiment Checklist (Not Part of the Formal Contents)

- Both positive and negative results are useful when their causes and scope are explained.
- Call experimentally supported statements results; label incomplete explanations as hypotheses or possible causes.
- Always describe Xsens as an external reference rather than absolute ground truth.
- Report both absolute angles and Delta Angle so that motion results do not hide absolute-angle differences.
- Complement averages with medians, p95 values, and box plots so that a few outliers do not dominate the message.
- Preserve video parameters, calibration, model versions, run configurations, code entry points, and plot source data for reproducibility.
