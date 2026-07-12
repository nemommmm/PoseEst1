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
