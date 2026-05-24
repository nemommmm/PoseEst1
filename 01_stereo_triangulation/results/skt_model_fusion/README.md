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
/opt/anaconda3/envs/pose/bin/python 01_stereo_triangulation/src/02_batch_inference.py
```

The default remains the historical YOLO path:

```bash
POSE_2D_DETECTOR=yolo \
POSE_MODEL_NAME=yolov8m-pose.pt \
/opt/anaconda3/envs/pose/bin/python 01_stereo_triangulation/src/02_batch_inference.py
```

Important: RTMPose / RTMO confidence scores should not reuse YOLO thresholds
blindly. After each detector run, scan the gate thresholds again.

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
