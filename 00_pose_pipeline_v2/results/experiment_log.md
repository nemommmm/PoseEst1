# Experiment Log

## 2026-07-12 — Runpod A6000 Migration and Fanbo7 Reproduction

Comparison system: Xsens-derived reference. Xsens is not treated as absolute
ground truth.

### Video compression gate

- Dataset: Fanbo7 A257, `cap_5_0/1`, 2048x1536 at 12.5 fps.
- H.265 CRF 18 reduced the left stream from about 1.3 GB to 13 MB but failed
  the accuracy gate: raw-vs-compressed RightElbow angle MAE was 2.28 degrees,
  and SKT agreement MAE changed by +0.88 degrees.
- H.264 QP 0 produced pixel-identical decoded grayscale frames in the sampled
  pixel audit. Full SKT output exactly reproduced the raw keypoints and angle
  metrics. The two Fanbo7 streams total about 938 MB after compression.
- Decision: use H.264 QP 0; reject CRF 18 for formal experiments.

### Remote environment

- GPU: NVIDIA RTX A6000 48 GB.
- PyTorch: 2.8.0 + CUDA 12.8.
- Ultralytics pinned to 8.3.235 to match the canonical local baseline.
- ONNX Runtime GPU pinned to 1.22.0; providers include TensorRT, CUDA, and CPU.
- RTMLib is installed with `--no-deps` to prevent CPU ONNX Runtime and
  conflicting OpenCV wheels from replacing the GPU runtime.

### Deterministic CUDA diagnosis

- Default CUDA inference changed the crop-tracked sequence: the first-100-frame
  RightElbow angle MAE relative to the local CPU run was 0.791 degrees.
- Remote CPU reproduced the local CPU result within 0.00037 degrees.
- Disabling TF32/cuDNN benchmark and enabling deterministic algorithms reduced
  the GPU-vs-CPU first-100-frame angle MAE to 0.00038 degrees.
- Deterministic CUDA is therefore enabled by default in `skt_inference.py`.

### Accepted Fanbo7 result

| Metric | Local CPU | A6000 deterministic GPU |
|---|---:|---:|
| RightElbow MAE | 10.4307738 deg | 10.4307745 deg |
| Valid ratio | 0.4272517 | 0.4272517 |
| RULA-like agreement | 0.9405405 | 0.9405405 |
| Warm-excluded YOLO throughput | n/a | 41.90 fps |
| Warm-excluded end-to-end throughput | n/a | 29.21 fps |

Result: the GPU reproduction passed the 0.1-degree alignment tolerance and the
12.5 fps target on Fanbo7. Detailed real-time backend benchmarking remains a
separate workstream because the current frame timing combines decode,
inference, triangulation, and tracking.

## 2026-07-12 — A6000 Deterministic GPU Reproduction Across Fanbo4/7/9

All runs use V2 + YOLOv8m with deterministic CUDA and H.264 QP 0 compressed
stereo streams. FastSAM3D and Xsens are external comparison/reference systems.

| Dataset | Frames | FastSAM3D MAE | Mean valid ratio | GPU warm FPS | Frame speedup | YOLO speedup |
|---|---:|---:|---:|---:|---:|---:|
| Fanbo7 A257 | 433 | n/a (RightElbow Xsens-derived MAE 10.4308 deg) | 0.4273 | 28.17 | 9.65x | 12.26x |
| Fanbo4 A257 | 390 | 9.8202 deg (RightElbow) | 0.6026 | 29.68 | 10.63x | 14.61x |
| Fanbo9 A255 | 897 | 10.7004 deg (8-joint mean) | 0.6810 | 29.92 | 18.13x | 24.83x |
| Fanbo9 A257 | 837 | 6.4200 deg (8-joint mean) | 0.8571 | 29.92 | 13.10x | 17.88x |

`Frame speedup` uses the mean saved `frame_time_ms`; `YOLO speedup` uses the
saved `yolo_time_ms`. `GPU warm FPS` excludes the first ten frames. GPU angle
summaries matched the local CPU summaries to approximately 1e-4 degrees or
better, with identical valid counts and RULA-like agreement values.

Conclusion: the deterministic A6000 pipeline reproduces the local reference
results and exceeds the 12.5 fps camera rate for all evaluated distances and
camera views. Fanbo9 again confirms that A257 has stronger agreement with the
FastSAM3D comparison trajectory than A255; this is not a ground-truth claim.

## 2026-07-12 — PyTorch, ONNX, and TensorRT Real-time Backend Benchmark

The benchmark runs the real crop-tracked SKT path and separates video decode,
stereo pose inference, per-frame geometry, and sequence post-processing.
Reported FPS includes online decode, inference, tracking, and geometry.

| Backend | Dataset | Online FPS | Pose ms | Decode median ms | Online p95 ms | RightElbow MAE |
|---|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | Fanbo7 | 28.88 | 23.44 | 9.41 | 43.76 | 10.43 deg |
| PyTorch FP32 | Fanbo4 | 29.23 | 23.71 | 9.22 | 46.30 | 13.66 deg |
| ONNX Runtime CUDA | Fanbo7 | 17.02 | 25.53 | 9.07 | 245.88 | 12.36 deg |
| ONNX Runtime CUDA | Fanbo4 | 25.70 | 25.10 | 9.06 | 40.87 | 13.65 deg |
| TensorRT FP16 | Fanbo7 | 30.21 | 17.49 | 8.22 | 56.50 | 12.81 deg |
| TensorRT FP16 | Fanbo4 | 26.80 | 16.46 | 8.45 | 58.36 | 15.62 deg |

TensorRT reduced stereo pose inference latency by about 26-31%, but ONNX and
TensorRT changed first-frame full-image keypoints by roughly 0.8 pixels and
the crop tracker amplified the difference. PT-vs-ONNX and PT-vs-TRT
RightElbow trajectory MAE on the first 200 Fanbo7 frames was 3.72 and 3.44
degrees, respectively. Both deployment backends therefore failed the accuracy
gate despite exceeding 12.5 fps.

The Fanbo7 ONNX mean was additionally affected by network-volume decode
outliers (9.07 ms median, 221 ms p95). Decode and model latency must remain
separate in deployment claims. Accepted current backend: deterministic
PyTorch FP32. TensorRT FP16 remains a speed result, not an accuracy-preserving
replacement.

### Model-size and stereo-batch timing

All PyTorch model sizes exceeded 12.5 fps on the 200-frame Fanbo7 timing run.
Network-volume decode outliers make mean FPS less stable than model latency.

| Model | Stereo pose mean ms | Online FPS |
|---|---:|---:|
| YOLOv8m-pose | 23.44 | 28.88 |
| YOLO11n-pose | 23.97 | 24.87 |
| YOLO11s-pose | 21.96 | 22.64 |
| YOLO11m-pose | 25.29 | 28.58 |
| YOLO11l-pose | 33.23 | 23.19 |

The non-monotonic online FPS values reflect decode jitter and tracking branch
behavior; model latency is the appropriate column for model-size comparison.
These runs are timing results only and do not replace the existing canonical
accuracy ablation.

For YOLOv8m full-frame model-only inference, one batch=2 call processed a
stereo pair in 21.36 ms versus 23.79 ms for two serial calls, a 1.11x speedup.
The production crop tracker uses different left/right ROIs and therefore still
runs serially. Batch=2 is recorded as an optimization opportunity, not as an
equivalent end-to-end result.

## 2026-07-12 — New Pose Model Stereo Gate on A6000

Recent model families were screened in two directions: RTMPose-X as an
accuracy-oriented two-stage candidate, and RTMPose-S / RTMO-S/M as real-time
candidates. ViTPose-L/H and RTMW-L were retained as literature candidates but
not run after the lighter RTMPose-X already exceeded the latency gate.

The diagnostic uses identical sampled stereo frames and reports real latency
per stereo pair plus right-arm rectified epipolar consistency. A candidate must
work on both Fanbo7 near-view and Fanbo4 far-view data.

| Candidate | Fanbo7 ms | Fanbo7 epi median/p95 px | Fanbo4 ms | Fanbo4 epi median/p95 px | Decision |
|---|---:|---:|---:|---:|---|
| RTMPose-S | 86.3 | 121.3 / 289.1 | 75.3 | 0.58 / 3.87 | reject: near-view geometry |
| RTMPose-M | 92.6 | 4.83 / 323.6 | 118.9 | 0.41 / 94.7 | reject: latency and outliers |
| RTMPose-X | 126.2 | 2.99 / 123.0 | 180.8 | 0.53 / 121.1 | reject: latency and outliers |
| RTMO-S | 65.9 | 196.2 / 357.0 | 44.4 | 0.87 / 385.9 | reject: stereo inconsistency |
| RTMO-M | 45.8 | 305.6 / 426.5 | 41.9 | 1.16 / 406.2 | reject: stereo inconsistency |

RTMO met the model-level speed target but produced catastrophic left/right
semantic inconsistency on some frames. RTMPose-X improved median epipolar
agreement but could not meet the 80 ms stereo-pair budget and retained large
outliers. No candidate advanced to full 3D angle evaluation. This gate avoids
reporting misleading 3D errors from models that are confident monocularly but
not stereo-consistent.

## 2026-07-12 — Pose2Sim / OpenSim Kinematic-Prior Gate

Pose2Sim 0.10.48 (upstream commit `14d101c`) and OpenSim 4.6 were installed in
an isolated remote environment. Deterministic YOLOv8m/SKT COCO-17 trajectories
were converted from camera coordinates to OpenSim Y-up meters. Missing marker
coordinates were interpolated on the full sequence before selecting a
continuous gate window. The simple OpenSim model was scaled per sequence and
then used for inverse kinematics. Xsens remained an external comparison signal
and was not used to fit the model or select parameters.

| Dataset (first 200 frames) | System | RightElbow MAE | Valid ratio | RULA-like agreement | >10 deg jumps | Total prior stage |
|---|---|---:|---:|---:|---:|---:|
| Fanbo7 A257 | SKT geometric | 10.47 deg | 0.655 | 0.939 | 0 | n/a |
| Fanbo7 A257 | Pose2Sim/OpenSim | 52.36 deg | 0.655 | 0.344 | 0 | 102.6 ms/frame |
| Fanbo4 A257 | SKT geometric | 15.47 deg | 0.765 | 0.948 | 7 | n/a |
| Fanbo4 A257 | Pose2Sim/OpenSim | 24.98 deg | 0.765 | 0.895 | 19 | 101.4 ms/frame |

Decision: reject before full-sequence evaluation. Sparse COCO-17 observations,
especially the extensively missing left arm in Fanbo7, allowed the kinematic
solver to find anatomically feasible but motion-inconsistent solutions. The
candidate failed both the angle-improvement gate and the 12.5 fps real-time
gate. Per the project experiment rule, the Pose2Sim adapter was rolled back to
the pre-route snapshot; ignored run evidence and this negative-result log were
retained.

## 2026-07-12 — MeTRAbs Calibrated Stereo Gate on A6000

The official experimental PyTorch MeTRAbs repository (commit `8b2b116`) was
tested with its EfficientNetV2-S 256 px checkpoint. This is the only supported
lightweight PyTorch checkpoint available alongside the EfficientNetV2-L model;
the advertised MobileNetV3 weights are available only through the older
TensorFlow export. Two official Torch 2.x compatibility fixes were required
after the network forward (`torch.split` sizes converted to Python lists).

The two images were undistorted, rotated consistently with the dataset, and
processed as one stereo batch using the existing deterministic YOLO boxes.
Camera extrinsics converted both metric 3D predictions into the left-camera
world frame. Fusion weights used YOLO confidence and 2D agreement; no Xsens
signal was used in inference, fusion, or parameter selection.

| Dataset (40-frame gate) | End-to-end FPS | Model inference ms/pair | Right-arm stereo 3D median/p95 | 2D disagreement median/p95 | Finite keypoint ratio |
|---|---:|---:|---:|---:|---:|
| Fanbo7 A257 | 4.76 | 86.8 | 69.45 / 75.68 cm | 564.8 / 735.1 px | 0.404 |
| Fanbo4 A257 | 4.96 | 82.1 | 66.38 / 109.02 cm | 854.6 / 1028.0 px | 0.134 |

Decision: reject before the 200-frame and full-sequence stages. The
lightweight model failed both the 12.5 fps requirement and the calibrated
cross-view geometry gate. The larger EfficientNetV2-L model and additional
test-time augmentation cannot recover the real-time requirement and were not
run after this failure. Per the project rule, the MeTRAbs integration was
rolled back; the isolated external environment, ignored run evidence, and this
negative-result log were retained.

## 2026-07-12 — EasyMocap / SMPL Readiness Gate

EasyMocap commit `e681319` was cloned under the persistent external directory.
Its CUDA SMPL module imported successfully with PyTorch 2.8 / CUDA 12.8. A
reproducible dependency setup, licensed-asset validator, private-path ignore
rules, and bilingual download instructions were added.

The formal SMPL fitting experiment did not start because neither the local
workspace nor `/workspace/model_assets/smpl/` contains a licensed neutral SMPL
model. The setup script exits with code 2 at this expected asset gate. No model
parameters were downloaded from unofficial sources, and no Xsens-derived
signal was used. Once the user supplies `SMPL_NEUTRAL.pkl` under the documented
persistent path, the CUDA zero-pose validation must pass before any 40-frame
fitting gate is allowed to run.

## 2026-07-13 — Geometry-conditioned Kinematic Prior Gate

A lightweight calibrated-stereo kinematic optimizer was evaluated on a
continuous 40-frame Fanbo7 A257 interval. The fixed grid covered 3D-anchor
weights `[0.25, 1.0]`, bone weights `[1.0, 3.0]`, and temporal weights
`[0.05, 0.2]`. Candidate selection used only reprojection, finite coverage,
bone stability, and temporal jumps. Xsens-derived data were not used for
fitting or hyperparameter selection.

Hardware and runtime: NVIDIA RTX A6000 48 GB (UUID
`GPU-9677f9a4-5d00-2cb2-fac6-d50eef706f30`), driver 580.95.05, PyTorch 2.8.0,
CUDA 12.8, and cuDNN 9.1. The experiment commit was `d9d29fe`; the frozen A257
calibration and source NPZ hashes are recorded in the downloaded candidate
metadata and checksummed artifact manifest.

| Metric | Raw deterministic SKT | Selected kinematic prior |
|---|---:|---:|
| Finite core-joint ratio | 0.8375 | 0.9167 |
| Reprojection median | 1.3660 px | 1.6110 px |
| Reprojection p95 | 7.7878 px | 13.5446 px |
| Mean bone-length CV | 0.1116 | 0.0971 |
| Angle jumps above 10 degrees | 53 | 42 |
| High-quality correction median / p95 | -- | 0.2061 / 0.6308 cm |
| Prior time | -- | 5028.9 ms/frame |
| Estimated end-to-end throughput | existing baseline >12.5 fps | 0.1976 fps |

The selected setting was anchor 1.0, bone 3.0, and temporal 0.2. The prior
filled missing joints and improved bone stability, while its small correction
on high-quality observations confirmed that it did not overpower reliable
image evidence. However, reprojection p95 exceeded the predefined 10 px gate
and the implementation was far below the real-time target. Decision: reject
before Fanbo4, 200-frame, angle-agreement, and full-sequence evaluation. The
standard negative-result bundle, including the reconstruction NPZ, timing,
GPU metadata, 12 diagnostic figures, preview video, and SHA256 manifest, was
downloaded locally. Per the experiment rule, the failed fitting adapter is
rolled back rather than patched further.

## 2026-07-13 — SMPL Licensed-asset Gate Recheck

The new A6000 Pod was checked at the expected persistent path
`/workspace/model_assets/smpl/SMPL_NEUTRAL.pkl`. The asset is still absent, so
the run was recorded as `asset_blocked` and synchronized as a standard result
bundle. No unofficial model download was attempted. This is an external asset
block, not a model-accuracy rejection; EasyMocap/SMPL fitting remains pending
the official licensed file and its CUDA forward validation.

## 2026-07-13 — EasyMocap / SMPL Calibrated-stereo Feasibility Gate

The official SMPL for Python v1.1.0 neutral model was supplied by the user and
stored only in the private persistent path
`/workspace/model_assets/smpl/SMPL_NEUTRAL.pkl`. Its size was 247,186,228
bytes and its SHA256 was
`4924f235e63f7c5d5b690acedf736419c2edb846a2d69fc0956169615fa75688`.
EasyMocap produced a finite CUDA zero-pose forward with shape `[1, 25, 3]`.

The fitting gate used a continuous centered 40-frame Fanbo7 A257 interval,
the frozen tracked stereo calibration, and the deterministic YOLOv8m/SKT
source. EasyMocap BODY-25 joints were explicitly mapped to COCO-17. One shape
was estimated for the session and frozen for final refinement; per-frame pose,
global rotation, and translation were fitted with two-view Huber reprojection,
a quality-weighted SKT anchor, the official EasyMocap CMU GMM pose prior,
shape regularization, and temporal second-order continuity. No Xsens-derived
signal was used for initialization, fitting, or parameter selection.

| Metric | EasyMocap / SMPL |
|---|---:|
| Finite core-joint ratio | 1.0000 |
| Reprojection median / p95 | 10.7509 / 35.3356 px |
| Mean bone-length CV | 0.0121 |
| Angle jumps above 10 degrees | 34 |
| High-quality correction median / p95 | 6.1537 / 28.6947 cm |
| Prior time | 404.22 ms/frame |
| Estimated end-to-end throughput | 2.2966 fps |

Decision: reject at the first feasibility gate. The body prior produced a
complete and temporally rigid skeleton, but it displaced reliable image
observations by several centimeters and failed the 10 px reprojection gate by
a large margin. This is the same important failure mode observed with
Pose2Sim: an anatomically plausible solution was not sufficiently faithful to
the recorded motion. Fanbo4, 200-frame, full-sequence, angle-agreement, and
RULA comparisons were not run. The checksummed local result bundle contains
the canonical NPZ, SMPL pose/shape/translation arrays, 12 diagnostic figures,
an H.264 preview, timing, GPU metadata, and the artifact manifest. The fitting
adapter is rolled back to snapshot `275b727`; licensed-asset validation and
the reproducible EasyMocap dependency fix are retained.

## 2026-07-20 — Distance-stratified YOLOv8m / YOLO11L Reanalysis

Existing SKT V2 reconstructions were reanalysed locally for Fanbo7 A257,
Fanbo9 A257, Fanbo9 A255, and Fanbo4 A257. The comparison used one fixed
Xsens offset per recording, the same common valid frames for both detectors,
and the YOLOv8m torso optical-depth estimate as a shared horizontal
coordinate. Xsens remained an external, Xsens-derived comparison reference;
no model parameter or time offset was tuned separately to improve agreement.

Command:

```bash
/opt/anaconda3/envs/pose/bin/python \
  00_pose_pipeline_v2/src/analyze_error_vs_distance.py \
  --config 00_pose_pipeline_v2/configs/distance_error_analysis.yaml
```

The analysis retained 1219 unique common valid frames. Pooled right-elbow
median absolute disagreement was 17.58 degrees for YOLOv8m and 18.49 degrees
for YOLO11L. Among five 0.5 m bins with at least 20 common frames, YOLO11L
had the lower paired median in only one. Fanbo7 illustrates why means alone
are insufficient: YOLO11L had the lower mean (9.18 vs 10.43 degrees), while
YOLOv8m had the lower median (6.66 vs 7.19 degrees).

Decision: retain YOLOv8m as the current default. A near/far dynamic switch is
recorded only as an engineering hypothesis, not a validated result. The
present sessions confound distance with action, viewpoint, and occlusion, and
the horizontal axis is estimated optical depth rather than an independent
physical measurement. The next admissible test is a controlled 2.0--4.5 m
recording at 0.5 m intervals with repeated actions and held-out validation.
The local result package contains per-frame CSV data, session/bin/paired
summaries, SHA256 source metadata, five bilingual figures, and bilingual HTML
reports under `00_pose_pipeline_v2/runs/distance_error_analysis_20260720/`.

## 2026-07-25 — Rebuilt A6000 PyTorch, TensorRT, and NVDEC Gates

The expired Pod was replaced and the required repository, environments,
weights, compressed stereo inputs, historical references, and evaluation
artifacts were restored. The new Pod used an NVIDIA RTX A6000 with 49,140 MiB
VRAM (UUID `GPU-6211cba3-6c72-4f80-70d5-1f7c7af7da9a`), driver 580.159.03,
PyTorch 2.8.0 with CUDA 12.8, and cuDNN 9.1. All formal results and their
SHA256 manifests were downloaded under
`00_pose_pipeline_v2/runs/gpu_rebuild_20260725/`.

### Three-repeat deterministic PyTorch baseline

Fanbo7 and Fanbo4 A257 were each evaluated for 200 frames after a ten-frame
warm-up, with three independent repeats. Online-stage timing includes decode,
crop-tracked stereo pose inference, and per-frame geometry; sequence
post-processing is recorded separately and is not included in the FPS value.

| Dataset | Median online FPS (min--max) | Online median / p95 | Stereo pose median / p95 | Repeat and historical gates |
|---|---:|---:|---:|---|
| Fanbo7 A257 | 29.744 (29.191--30.588) | 30.151 / 42.250 ms | 20.268 / 31.135 ms | pass / pass |
| Fanbo4 A257 | 30.309 (28.656--30.348) | 30.151 / 45.329 ms | 19.281 / 31.355 ms | pass / pass |

All three repeats were exactly deterministic. Both datasets also exactly
reproduced the downloaded deterministic PyTorch reference: the RightElbow
trajectory difference was zero and RULA-bin agreement was 1.000. Decision:
retain deterministic PyTorch as the accepted deployment baseline. The run
manifest records project commit `b12ae95` and marks that working tree as dirty;
the exact code, configuration, model, calibration, video, and reference hashes
are therefore the authoritative reproduction identifiers. Formal source:
`pytorch_formal/suite_summary.json`.

### TensorRT FP32 and FP16

Static batch-one TensorRT engines were exported from the same YOLOv8m-pose
weight. The FP32 and FP16 engine SHA256 values were respectively
`a17d0037f22dc44ac666fce4c1892a8ccb3cfe43448d3b96d381d414875531f9`
and
`46dd4adfe835d04a3856c5c315016c3be25ba529f2b65c84e61cfe0ec7ba5154`.
Each route used the same 200-frame, ten-frame-warm-up, three-repeat protocol.
The angle differences below compare TensorRT output with the historical
deterministic PyTorch result; they are not errors against the Xsens-derived
reference.

| Dataset | Engine | Median online FPS | Stereo pose median / p95 | RightElbow difference median / p95 | RULA-bin agreement | Decision |
|---|---|---:|---:|---:|---:|---|
| Fanbo7 A257 | FP32 | 27.554 | 17.671 / 45.174 ms | 2.803 / 9.490 deg | 0.985 | reject |
| Fanbo4 A257 | FP32 | 27.799 | 17.612 / 46.832 ms | 5.754 / 24.727 deg | 0.960 | reject |
| Fanbo7 A257 | FP16 | 27.999 | 12.378 / 50.461 ms | 2.775 / 8.984 deg | 0.985 | reject |
| Fanbo4 A257 | FP16 | 29.715 | 11.855 / 53.120 ms | 7.396 / 27.665 deg | 0.970 | reject |

Each TensorRT route was internally deterministic across its three repeats, but
all four failed the historical-output gate. Reduced median inference time,
especially with FP16, did not provide a reliable end-to-end speed advantage
and did not preserve the accepted trajectory. Decision: reject both TensorRT
precisions as replacements for the deterministic baseline. The formal export
and evaluation commit was `95dc739`; sources:
`tensorrt_exports/export_manifest.json` and
`tensorrt_formal/suite_summary.json`.

### FFmpeg NVDEC decode-only measurement

A controlled decode-only comparison used two 200-frame Fanbo7 H.264 High
`yuv420p` proxies, one 30-frame warm-up, and three repeats. The proxies were
generated with QP 0 and are classified as near-lossless, not pixel-identical
to the source.

| Backend | Paired-stream decode FPS, median | p95 |
|---|---:|---:|
| CPU software decode | 74.659 | 82.624 |
| NVIDIA NVDEC | 112.907 | 118.507 |

NVDEC provided a 1.512x median decode-only speedup. A three-frame-per-view
luma audit found exact CPU/NVDEC decoded-pixel agreement. This benchmark
includes FFmpeg startup, demux, decode, and a null output sink; it excludes
timestamp synchronization, RGB tensor preparation, pose inference,
triangulation, angles, and RULA. It must not be reported as end-to-end stereo
pose throughput. Decision: retain NVDEC as a deployment optimization candidate
that still requires integration into the real pipeline. The benchmark commit
was `29ab460`; formal source:
`decode/fanbo7_proxy_cpu_nvdec_v2.json`.

## 2026-07-25 — NVIDIA BodyPose3DNet Calibrated-stereo Gate

The official NVIDIA `deepstream-bodypose-3d` reference application
(upstream commit `a6488b5`) was evaluated with the BodyPose3DNet Accuracy and
Performance variants. The project evaluation commit was `d03aa85`. The test
used 433 synchronized Fanbo7 A257 frame pairs from upright near-lossless
proxies, the fixed tracked A257 calibration
(`2306d08b68621c31141a92320c369a44ac8d6aa139c6179d4d8e0ccab6eb495c`),
and the existing fixed alignment to the Xsens-derived reference. No
candidate-specific time offset or angle offset was fitted. The YOLO control
was rerun on the same proxy inputs, and the comparison below uses only frames
where BodyPose3DNet, YOLO, and the Xsens-derived reference were all finite.

| Variant | Matched frames | BodyPose3DNet / YOLO RightElbow MAE | MAE change | BodyPose3DNet / YOLO RULA agreement | Epipolar median / p95 | Decision |
|---|---:|---:|---:|---:|---:|---|
| Accuracy | 147 | 10.726 / 12.384 deg | 13.389% better | 0.932 / 0.891 | 3.474 / 25.908 px | angle signal retained; geometry gate failed |
| Performance | 134 | 13.082 / 13.149 deg | 0.509% better | 0.925 / 0.881 | 4.653 / 40.531 px | reject |

The Accuracy model passed the preliminary matched-angle gate, but it did not
pass the predefined stereo geometry limits of 3 px median and 10 px p95.
Because only 147 matched frames supported the apparent improvement, this is a
promising signal rather than evidence of full-sequence superiority; it does
not advance to the far-view gate in its present form. The Performance model
failed both the matched-angle improvement requirement and the stereo geometry
gate, so it is rejected.

DeepStream throughput measurements are retained only as internal feasibility
evidence. This public experiment log intentionally omits all DeepStream FPS
and competitive performance figures pending a check of the applicable NVIDIA
DeepStream EULA and any required NVIDIA written permission. Formal sources:
`nvidia_bodypose3d_feasibility/nvidia_environment_manifest.json`,
`nvidia_bodypose3d_feasibility/accuracy_full433_formal_fixed_reference/metrics.json`,
and
`nvidia_bodypose3d_feasibility/performance_full433_formal_fixed_reference/metrics.json`.
