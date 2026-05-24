# SKT + Mature Pose Model Fusion Notes

## Goal

This experiment track keeps SKT as the metric stereo anchor and uses mature pose
models only as auxiliary sources:

- stronger 2D keypoint detector before triangulation;
- quality-aware gate for unstable SKT joints;
- temporal 3D pose prior for low-quality frames or joints.

It intentionally avoids direct raw-coordinate averaging between SKT and
FastSAM3D / monocular models, because their coordinate frames, scale behaviour,
and keypoint semantics are not guaranteed to match.

## Direction A: RTMPose / RTMO as 2D detector

`02_batch_inference.py` now supports a detector backend switch:

```bash
POSE_2D_DETECTOR=rtmlib \
POSE_RTMLIB_MODEL=Body \
POSE_RTMLIB_BACKEND=onnxruntime \
POSE_RTMLIB_DEVICE=cpu \
POSE_RTMLIB_MODE=lightweight \
POSE_MAX_FRAMES=120 \
/opt/anaconda3/envs/pose/bin/python 01_stereo_triangulation/src/02_batch_inference.py
```

Recommended first-pass modes:

- `POSE_RTMLIB_MODE=lightweight`: fastest RTMPose-s path; useful for smoke tests and deployment-oriented checks.
- `POSE_RTMLIB_MODE=balanced`: medium RTMPose-m path.
- `POSE_RTMLIB_MODE=performance`: RTMPose-x path; it works but is too slow on CPU for long runs.

The default remains the historical YOLO path:

```bash
POSE_2D_DETECTOR=yolo \
POSE_MODEL_NAME=yolov8m-pose.pt \
/opt/anaconda3/envs/pose/bin/python 01_stereo_triangulation/src/02_batch_inference.py
```

Important: RTMPose / RTMO confidence scores should not reuse YOLO thresholds
blindly. After each detector run, scan the gate thresholds again.

RTMO note: direct `POSE_RTMLIB_MODEL=RTMO` requires an explicit ONNX model path:

```bash
POSE_2D_DETECTOR=rtmlib \
POSE_RTMLIB_MODEL=RTMO \
POSE_RTMLIB_ONNX_MODEL=/path/to/rtmo.onnx \
/opt/anaconda3/envs/pose/bin/python 01_stereo_triangulation/src/02_batch_inference.py
```

For most first-pass tests, `POSE_RTMLIB_MODEL=Body` is easier because RTMLib
auto-downloads the detector and RTMPose model.

## Direction B: quality-gate sweep

Run this on an existing SKT output to see which internal quality signal catches
large elbow deltas:

```bash
/opt/anaconda3/envs/pose/bin/python \
  01_stereo_triangulation/src/19_quality_gate_sweep.py \
  --input 01_stereo_triangulation/results/yolo_3d_optimized.npz \
  --k-values 1 6 \
  --high-delta-deg 35
```

Outputs:

- `quality_gate_sweep_elbow.csv`
- `quality_gate_sweep_elbow.md`

Use these outputs to decide whether high-delta outliers are mainly associated
with low pair confidence, low stereo quality, high epipolar error, or high
reprojection error.

## Detector smoke comparison

Compare short detector-backend runs before committing to full-sequence inference:

```bash
/opt/anaconda3/envs/pose/bin/python \
  01_stereo_triangulation/src/21_compare_skt_detector_runs.py \
  --limit-frames 120 \
  --run YOLOv8m=01_stereo_triangulation/results/skt_model_fusion/yolo_120/yolo_3d_optimized.npz \
  --run RTMPoseS=01_stereo_triangulation/results/skt_model_fusion/rtmlib_body_light_120/yolo_3d_optimized.npz
```

Initial 120-frame result:

- RTMLib / RTMPose-s lightweight successfully runs through the SKT pipeline.
- In this early segment, RTMPose-s does **not** outperform YOLOv8m internally:
  - epipolar p90 is higher;
  - reprojection p90 is higher;
  - elbow-chain valid ratios are lower;
  - left-elbow high-delta rates are higher.
- Therefore, simply replacing YOLO with RTMPose is not enough. The next useful
  step is quality-aware gating or temporal-prior repair, not a blind detector swap.

## Direction C: temporal prior fusion

External temporal models should first be converted to:

```text
timestamps: (N,)
keypoints:  (N, 17, 3)
```

Then fuse them with SKT using:

```bash
/opt/anaconda3/envs/pose/bin/python \
  01_stereo_triangulation/src/20_temporal_prior_fusion.py \
  --skt 01_stereo_triangulation/results/yolo_3d_optimized.npz \
  --prior path/to/motionbert_or_videopose3d_prior.npz \
  --output 01_stereo_triangulation/results/skt_model_fusion/skt_temporal_prior_fused.npz
```

The prior is interpolated to the SKT timeline, similarity-aligned to reliable
SKT joints per frame, and blended only where SKT quality is poor. This makes the
fusion explainable: SKT provides metric scale; the mature temporal model provides
motion plausibility.

## Evaluation

Compare candidate outputs using the frame-delta evaluation pipeline:

- K=1 and K=6 elbow delta scatter;
- Spearman / DTW / high-delta outlier count;
- RULA agreement as the ergonomic-level check;
- report as agreement with Xsens-derived reference, not absolute ground truth.
