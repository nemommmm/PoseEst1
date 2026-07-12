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
