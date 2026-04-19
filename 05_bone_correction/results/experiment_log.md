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

## 2026-04-20 - S1 Calibration Sanity Check

- Script:
  - `01_stereo_triangulation/src/15_calibration_sanity_check.py`
- Outputs:
  - `01_stereo_triangulation/results/calibration_sanity_check.md`
  - `01_stereo_triangulation/results/calibration_sanity_check.json`

### Key result

- Calibration-board scale error: `0.03%`
- Mean board edge absolute error: `0.022 cm`
- Mean rigid alignment RMSE: `0.127 cm`
- Decision: `Calibration scale looks OK`

### Interpretation

- The active stereo calibration is not suffering from a simple global scale mistake.
- This weakens the hypothesis that the upper-arm / thigh shortening is mainly caused by a bad baseline or focal length.
- The secondary body-height proxy is still short (`148.39 cm` vs `169 cm`),
  but the board-based check is much more trustworthy than the body-height proxy because COCO keypoints do not encode anatomical height exactly.

## 2026-04-20 - S2 Constrained Triangulation Pilot (limb-only priors)

- Script:
  - `01_stereo_triangulation/src/16_constrained_triangulation.py`
- Inputs:
  - stereo pose: `01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/raw_pose.npz`
  - target lengths: Xsens column from `05_bone_correction/results/bone_length_three_way_comparison.csv`
- Scope:
  - constrained only `upper_arm`, `forearm`, `thigh`, `shank`
  - intentionally excluded `shoulder_width`, `hip_width`, `torso`
  - solver setting used in the pilot: `MAX_NFEV=20`
- Outputs:
  - `01_stereo_triangulation/results/constrained_triangulation_v1/constrained_triangulation_pose_lambda_0p001.npz`
  - `01_stereo_triangulation/results/constrained_triangulation_v1/constrained_triangulation_pose_lambda_0p01.npz`
  - `01_stereo_triangulation/results/constrained_triangulation_v1/pilot_summary.md`

### Bone-length trend

- Raw ratios:
  - `upper_arm=0.886`, `forearm=0.920`, `thigh=0.906`, `shank=0.982`
- `lambda=0.001`:
  - `upper_arm=0.894`, `forearm=0.941`, `thigh=0.906`, `shank=0.984`
- `lambda=0.01`:
  - `upper_arm=0.911`, `forearm=0.953`, `thigh=0.925`, `shank=0.988`

### Angle evaluation

- Raw pose reference:
  - MAE `19.80°`
  - fair `18.49°`
  - elbow RULA `60.68%`
  - trunk `11.80°`
  - MPJPE `31.13 cm`
- `lambda=0.001`:
  - MAE `20.82°`
  - fair `19.56°`
  - elbow RULA `58.53%`
  - trunk `13.02°`
  - MPJPE `34.81 cm`
- `lambda=0.01`:
  - MAE `21.00°`
  - fair `19.67°`
  - elbow RULA `54.76%`
  - trunk `13.19°`
  - MPJPE `35.76 cm`

### Interpretation

- The limb-only constrained triangulation pilot does change geometry in the intended direction:
  limb lengths move closer to Xsens as `lambda` increases.
- However, that geometric correction does **not** translate into better semantic angles.
- In the current v1 formulation, the method is an informative negative pilot:
  - bone ratios improve,
  - but angle MAE, fair MAE, elbow RULA, trunk MAE, and MPJPE all get worse.
- Therefore simply injecting limb-length priors into the triangulation objective is not sufficient by itself.
