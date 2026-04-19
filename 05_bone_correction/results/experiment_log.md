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
