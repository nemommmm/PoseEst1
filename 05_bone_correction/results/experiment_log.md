# Bone Correction Experiment Log

## 2026-04-19 - Direction-Preserving Rescale Sanity Check

- Script:
  - `05_bone_correction/src/00_direction_preserving_rescale_eval.py`
- Inputs:
  - stereo pose: `01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
  - target lengths: medians from `04_hybrid_afh1/results/easyergo_normalized.npz`
- Method:
  - keep local bone directions from stereo
  - replace only segment lengths (`hip_width`, `shoulder_width`, `torso`, `upper_arm`, `forearm`, `thigh`, `shank`)
  - recompute semantic angles and compare against Xsens

### Result

- Original end-to-end MAE: `18.36°`
- Rescaled end-to-end MAE: `18.59°`
- Delta end-to-end MAE: `+0.23°`
- Original fair MAE: `17.47°`
- Rescaled fair MAE: `17.48°`
- Delta fair MAE: `+0.02°`

### Angle sensitivity

- `LeftShoulder`, `RightShoulder`, `LeftElbow`, `RightElbow`, `LeftKnee`, `RightKnee`:
  - mean absolute angle change ≈ `0.00°`
- `LeftHip`:
  - mean absolute change `3.07°`
- `RightHip`:
  - mean absolute change `2.91°`

### Interpretation

- A pure direction-preserving bone-length rescale does **not** materially improve angle MAE.
- This means the original P1 formulation is not strong enough as a main experiment.
- If bone-length correction is pursued further, it must allow **local direction changes** or a constrained kinematic projection, not only segment rescaling.

## 2026-04-19 - Temporal Smoothing Benchmark

- Best variant: `raw`
- `raw`: MAE=18.36°, fair=17.47°, elbow RULA=0.6581, mean bone std=2.53 cm
- `median_r5`: MAE=19.11°, fair=18.42°, elbow RULA=0.6837, mean bone std=10.63 cm
- `median_r3`: MAE=19.22°, fair=18.53°, elbow RULA=0.6588, mean bone std=24.16 cm
- `spike3_then_median_r3`: MAE=19.48°, fair=18.82°, elbow RULA=0.6423, mean bone std=8.41 cm
- `spike3_then_median_r5`: MAE=21.33°, fair=20.73°, elbow RULA=0.6322, mean bone std=11.90 cm

## 2026-04-19 - Xsens Three-Way Bone-Length Reference

- Script:
  - `05_bone_correction/src/01_xsens_bone_length_reference.py`
- Outputs:
  - `05_bone_correction/results/bone_length_three_way_comparison.csv`
  - `05_bone_correction/results/bone_length_three_way_comparison.md`

### Key result

- The three-way verdict is currently `mixed_or_inconclusive`.
- Stereo is **not uniformly compressed** against Xsens:
  - shoulder width `0.90x`
  - torso `1.09x`
  - forearm `1.01x`
  - shank `1.08x`
  - but upper arm `0.76x`, thigh `0.77x`
- EasyErgo is **not uniformly closer to Xsens** either:
  - every tracked segment is longer than Xsens under the current proxy definition
  - especially torso `1.53x`, shoulder width `1.39x`, thigh `1.34x`

### Interpretation

- The old story "stereo is globally compressed while EasyErgo is anatomically correct" is **not supported** by this Xsens-based check.
- At minimum, the problem is more mixed:
  - some stereo segments are short,
  - some are close,
  - some are long,
  - and EasyErgo appears systematically overlong under the current segment-origin proxy.
- Therefore P1b should not be justified as "repairing a confirmed stereo scale-compression bug" without further evidence.
