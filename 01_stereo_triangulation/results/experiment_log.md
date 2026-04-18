# Experiment Log

## 2026-04-13 - Historical Best (13.21 deg) Reproduction Check

- Goal: assess whether the historical best Direction A result (`13.21°` calibrated MAE, reported on 2026-03-24) can be reproduced from files still present in the reorganized repository.
- Historical evidence package created at:
  - `01_stereo_triangulation/results/historical_best_20260324/README.md`
  - `01_stereo_triangulation/results/historical_best_20260324/manifest.json`

### Evidence confirmed

- Historical reports still record:
  - calibrated angle MAE `13.21°`
  - uncalibrated angle MAE `18.59°`
  - elbow RULA accuracy `76.3%`
  - RULA within ±1 `84.9%`
  - MPJPE `26.02 cm`

### Reproduction checks performed

1. Current script + current optimized pose + current calibration file
   - Command pattern:
     - `POSE_ANGLE_CALIBRATION=1`
     - `POSE_CALIBRATION_LOAD=01_stereo_triangulation/results/angle_calibration.npz`
     - `/opt/anaconda3/envs/pose/bin/python 05_detailed_evaluation.py`
   - Result:
     - overall calibrated angle MAE `14.89°`
     - MPJPE `26.38 cm`

2. Old pre-reorganization script (`86c9b7b`) + current optimized pose + same calibration file
   - Result:
     - overall calibrated angle MAE `14.89°`
     - MPJPE `26.38 cm`

3. Candidate pose aliases tested
   - `yolo_3d_optimized.npz`
   - `yolo_3d_optimized_yolov8m_best.npz`
   - `yolo_3d_optimized_yolov8m_pose.npz`
   - `cmp -s` showed these files are byte-identical.

### Conclusion

- The historical `13.21°` result is a confirmed historical record, but it is **not reproducible from the currently preserved optimized pose files plus the currently preserved calibration file**.
- The missing piece is most likely the historical pose output used on 2026-03-24, not the evaluation-script version.
- The pre-reorganization worktree (`86c9b7b`) contains the old scripts, but the old `results/` directory was not versioned, so raw result files still need to be recovered or re-generated.

### Next useful actions

- Search for any archived / copied historical Direction A result directories outside git.
- If unavailable, try to identify the historical 02 inference configuration from report notes and re-run the old pipeline in the isolated worktree.

## 2026-04-13 - Filesystem Search for Lost Historical Outputs

- Search scope:
  - `~/Desktop/MVE386 Project Course`
  - `~/Documents`
  - `~/Downloads`
  - `~/.Trash`

### Findings

- No archived copy of the lost historical Direction A raw results was found.
- No backup `results/` directory containing the historical `13.21° / 18.59°` raw CSV/JSON/NPZ files was found outside the current repository.
- The isolated historical worktree `PoseEst1_hist_86c9b7b/` contains an empty `results/` directory only.
- `~/Documents/src` still contains older scripts (including `02_batch_inference.py`, `05_detailed_evaluation.py`, `04_skeleton_compare.py`), but no historical Direction A result files.

### Implication

- The historical best result is still evidenced by reports and report-generation scripts, but the corresponding raw result directory appears genuinely lost from disk.
- The only robust path to full reproducibility now is to reconstruct the old pipeline configuration and re-run it in the historical worktree.

## 2026-04-13 - Historical Reproduction Sweep on Preserved Pose Files

- Purpose: test whether the historical `13.21°` result can be recovered from pose files that are still present in the reorganized repository, by varying:
  - pose input (`yolo_3d_optimized.npz` vs `yolo_3d_optimized_quality_blend.npz`)
  - temporal offset source (`17.40` from current `alignment_summary.json` vs forced legacy default `17.20`)
  - calibration mode (uncalibrated vs in-sample piecewise fit)

- Sweep summary saved to:
  - `01_stereo_triangulation/results/historical_repro_sweep_20260413.csv`

### Best result in sweep

- Best calibrated MAE found:
  - `14.3277°`
  - configuration:
    - input: `yolo_3d_optimized_quality_blend.npz`
    - offset: `17.40`
    - calibration: in-sample fit

### Key conclusions

- None of the preserved optimized pose files can reproduce the historical `13.21°` result.
- Changing the offset from `17.40` to legacy `17.20` does not recover the lost performance.
- In-sample piecewise calibration consistently improves the preserved pose files from roughly `20.5–22.2°` down to `14.3–14.4°`, but still stops well above `13.21°`.
- Therefore, the missing ingredient is almost certainly the historical optimized pose output itself, not just the evaluation script version, calibration file, or alignment offset.

## 2026-04-13 - Old Pipeline Re-run Started

- Started a fresh re-run in isolated worktree:
  - worktree: `PoseEst1_hist_86c9b7b`
  - script: `src/02_batch_inference.py`
  - model: `yolov8m-pose.pt`
- Added a symlink so the historical worktree can access the current dataset:
  - `PoseEst1_hist_86c9b7b/2025_Ergonomics_Data -> PoseEst1/2025_Ergonomics_Data`
- Confirmed old loader still applies the 180° video rotation internally.

### Rationale

- Historical report text explicitly describes the calibrated result as:
  - `18.59°` after bone constraint + smoothing
  - `13.21°` after piecewise calibration
- Since preserved current pose files cannot recreate that path, the next step is to regenerate the old optimized pose directly from the historical code path.

## 2026-04-13 - Old Pipeline Rebuilt Results (Historical Worktree)

- Re-ran the historical pre-reorganization pipeline in isolated worktree `PoseEst1_hist_86c9b7b` using:
  - old `02_batch_inference.py`
  - model `yolov8m-pose.pt`
  - old `03_auto_optimizer.py`
  - old `05_detailed_evaluation.py`

### Rebuilt old-run metrics

- Old pipeline, default smoothing, default offset path:
  - uncalibrated MAE: `20.59°`
  - elbow RULA: `63.91%`
  - MPJPE: `26.99 cm`
- Old pipeline, `POSE_ANGLE_SMOOTH_RADIUS=4`:
  - uncalibrated MAE @ `17.20s`: `19.92°`
  - uncalibrated MAE @ `17.30s`: `19.79°`
- Old pipeline, `POSE_ANGLE_SMOOTH_RADIUS=4` + full in-sample piecewise calibration (10 bins):
  - calibrated MAE @ `17.20s`: `14.01°`
  - calibrated MAE @ `17.30s`: `13.91°`
  - elbow RULA @ `17.30s`: `76.34%`
  - MPJPE @ `17.30s`: `26.32 cm`

### Interpretation

- This is the closest reconstruction so far to the historical report path:
  - historical report: `18.59° -> 13.21°`
  - rebuilt old-run path: `19.79° -> 13.91°`
- The rebuilt result strongly supports the report notes that the historical best used:
  - `yolov8m`
  - `POSE_ANGLE_SMOOTH_RADIUS=4`
  - full piecewise calibration (`10 bins`)
  - temporal offset around `17.30s`
- The remaining gap to the exact historical `13.21°` is now only about `0.7°`, suggesting that the missing ingredient is likely a small difference in the historical optimized pose output or a minor configuration detail, not a fundamentally different evaluation pipeline.

## 2026-04-13 - Cross-check: Historical Calibration on Rebuilt Old Pose

- Applied the preserved historical calibration file:
  - `01_stereo_triangulation/results/angle_calibration.npz`
- Tested on rebuilt old-run pose with:
  - `POSE_ANGLE_SMOOTH_RADIUS=4`
  - forced old evaluation offset target around `17.30s`

### Result

- Rebuilt old-run + historical calibration:
  - `14.44°`
  - elbow RULA `73.90%`

### Interpretation

- The preserved historical calibration file does **not** improve the rebuilt old-run pose to the historical `13.21°`; it performs worse than the rebuilt in-sample `r=4` calibration (`13.91°`).
- This indicates the remaining gap is **not** caused by using the wrong calibration file.
- The most likely missing ingredient is therefore the historical optimized pose output itself (or a small upstream inference configuration detail), not the downstream calibration curve.

## 2026-04-13 - Cross-check: Current Main Branch with Historical Best Settings

- Tested current main-branch evaluation on preserved `yolov8m` optimized pose aliases with:
  - `POSE_INPUT_FILENAME=yolo_3d_optimized_yolov8m_pose.npz`
  - `POSE_ANGLE_SMOOTH_RADIUS=4`
  - `POSE_ANGLE_CALIBRATION=1`
  - `POSE_CALIBRATION_LOAD=../results/angle_calibration.npz`

### Result

- Current main branch result:
  - `14.45°`
  - elbow RULA `74.04%`

### Interpretation

- Even when using the preserved historical calibration file and the current preserved `yolov8m` optimized pose, the main branch still does not recover the historical `13.21°`.
- Combined with the rebuilt old-run result (`13.91°`), this strengthens the conclusion that the missing component is the exact historical optimized pose output (or a small inference-side configuration difference), not just evaluation-side settings.

## 2026-04-13 - Targeted Reproduction: Historical `phase2d_yolov8m_pose`

- Started a new reproduction run in:
  - `PoseEst1_hist_9a9eaa9`
- Using historically recovered `Phase 2d` inference settings:
  - `POSE_MODEL_NAME=yolov8m-pose.pt`
  - `POSE_REPROJECTION_BASE_PX=12`
  - `POSE_REPROJECTION_CONF_GAIN_PX=15`
  - `POSE_REPROJECTION_MAX_PX=30`
  - `POSE_MIN_PAIR_CONF=0.40`
  - `POSE_MIN_DISPARITY_PX=4.0`
  - `POSE_ENFORCE_EPIPOLAR_CONSTRAINT=1`
  - `POSE_ENABLE_TEMPORAL_WINDOW_TRIANGULATION=1`
  - `POSE_ENABLE_2D_TEMPORAL_SMOOTHING=1`
  - `POSE_OUTPUT_TAG=phase2d_yolov8m_pose`
  - `POSE_RESULTS_DIR=results_phase2d_repro`
- Historical motivation:
  - session logs reference a missing upstream file named `yolo_3d_optimized_phase2d_yolov8m_pose.npz`
  - its uncalibrated downstream metrics matched the remembered historical chain (`18.59°`, elbow RULA `65.75%`, MPJPE `25.86 cm`)

### Goal

- Rebuild the missing historical upstream pose output as faithfully as possible before rerunning old `03_auto_optimizer.py`, `05_detailed_evaluation.py`, and `08_ergonomic_scoring.py`.

### Result

- `02_batch_inference.py` completed successfully in `PoseEst1_hist_9a9eaa9` and recreated the previously missing upstream file:
  - `results_phase2d_repro/yolo_3d_optimized_phase2d_yolov8m_pose.npz`
- Recomputed old downstream chain:
  - old `03_auto_optimizer.py`: best offset `17.25s`
  - old `05_detailed_evaluation.py` with `POSE_ANGLE_SMOOTH_RADIUS=4`: `18.63°`
  - old `05_detailed_evaluation.py` with `POSE_ANGLE_SMOOTH_RADIUS=4`, `POSE_ANGLE_CALIBRATION=1`, `POSE_CALIBRATION_BINS=10`: `13.26°`

## 2026-04-19 - Quality Signal vs Angle Error Diagnostics

- Added diagnostic script:
  - `01_stereo_triangulation/src/12_angle_error_quality_diagnostics.py`
- Outputs saved to:
  - `01_stereo_triangulation/results/quality_error_diagnostics/`

### Goal

- Test whether current SKT angle errors are predictable from already available quality signals.
- Decide whether the next improvement should target:
  - confidence-aware gating,
  - selective joint correction,
  - scenario-aware fallback,
  rather than another full-pipeline replacement.

### Main findings

- Highest-MAE semantic angles in the historical best pose are still:
  - `LeftElbow 22.21°`
  - `RightElbow 21.71°`
  - `LeftShoulder 19.54°`
  - `LeftKnee 18.04°`
- These errors are not random. Strongest bad-vs-good quartile gaps:
  - `LeftElbow`: low `detect_conf_min` -> `36.43°` vs `14.43°` (`+22.00°`)
  - `LeftKnee`: low `detect_conf_min` -> `28.80°` vs `13.80°` (`+15.01°`)
  - `RightKnee`: low `stereo_quality_min` -> `25.38°` vs `13.41°` (`+11.97°`)
  - `RightElbow`: low `detect_conf_min` -> `28.47°` vs `18.42°` (`+10.04°`)
- Scenario summary shows the worst remaining concentration is still in occlusion-heavy lower-limb frames:
  - `LeftKnee` in `Occlusion`: `32.45°`
  - `RightKnee` in `Occlusion`: `25.79°`

### Interpretation

- There is still improvement space, but it is likely in **quality-aware selective correction** rather than another global hybridization strategy.
- The best next candidates are:
  - elbow correction triggered by low 2D detector confidence,
  - knee correction / stronger fallback triggered by low stereo quality in occlusion frames,
  - scenario-aware selective smoothing or interpolation instead of full-sequence replacement.
  - old `08_ergonomic_scoring.py` on the same chain: Grand Score within ±1 `84.7%`

### Interpretation

- This is effectively a recovery of the historical best result chain.
- Recovered values are nearly identical to the historical report:
  - historical: `13.21°`, elbow RULA `76.3%`, RULA within ±1 `84.9%`
  - recovered: `13.26°`, elbow RULA `76.11%`, RULA within ±1 `84.7%`
- The remaining difference is only about `0.05°`, small enough to treat the historical best as successfully reproduced for practical purposes.

## 2026-04-13 — Direction B/C Audit and Visualization

### Direction C (MTL) evaluation-layer correction

- Updated `03_mono_motionbert/evaluate_vs_gt.py` so that:
  - elbow angles use flexion semantics (`180 - interior`)
  - hip angles are included, matching the 8-joint evaluation scope used by Direction A
- Re-ran:
  - `/opt/anaconda3/envs/pose/bin/python 03_mono_motionbert/evaluate_vs_gt.py`

## 2026-04-13 — Direction A/C Overlay Upgrade (Angles + Distance + Snapshots)

- Goal:
  - update the comparison videos so the per-frame text focuses on angle and distance rather than only raw overlay distance
  - automatically export representative good / bad static frames
  - add an explicit depth-vs-planar diagnosis for Direction C

### Direction A (SKT)

- Re-rendered:
  - `01_stereo_triangulation/results/skeleton_comparison_dirA.mp4`
  - `01_stereo_triangulation/results/skeleton_comparison_dirA.json`
- New overlay content per frame:
  - timestamp pair (`t`, `gt`)
  - 8-joint angle MAE
  - worst joint + worst-joint error
  - anchor distance and pelvis distance
- Exported representative snapshots:
  - `01_stereo_triangulation/results/skeleton_snapshots_dirA/`
- Summary:
  - mean overlay joint distance: `21.97 cm`
  - mean overlay pelvis distance: `17.82 cm`
  - mean 8-joint angle MAE shown in video: `17.85°`
  - good snapshot examples:
    - `good_01_t56.72_score7.19.png`
    - `good_02_t199.84_score8.16.png`
  - bad snapshot examples:
    - `bad_01_t94.40_score83.04.png`
    - `bad_02_t60.24_score69.28.png`

### Direction C (MTL)

- Re-rendered:
  - `03_mono_motionbert/results_mono/skeleton_comparison_dirC.mp4`
  - `03_mono_motionbert/results_mono/skeleton_comparison_dirC.json`
- New overlay content per frame:
  - timestamp pair (`t`, `gt`)
  - 8-joint angle MAE
  - worst joint + worst-joint error
  - anchor distance and pelvis distance
  - depth component vs non-depth component of the anchor error
- Exported representative snapshots:
  - `03_mono_motionbert/results_mono/skeleton_snapshots_dirC/`
- Added explicit diagnosis outputs:
  - `03_mono_motionbert/results_mono/depth_diagnosis_dirC.json`
  - `03_mono_motionbert/results_mono/depth_diagnosis_dirC.md`
- Summary:
  - mean overlay joint distance: `218.15 cm`
  - mean overlay pelvis distance: `207.11 cm`
  - mean 8-joint angle MAE shown in video: `24.77°`
  - mean depth component: `109.79 cm`
  - mean non-depth component: `182.41 cm`
  - mean depth share: `34.53%`
  - fraction of frames where depth > non-depth: `1.44%`

### Interpretation

- The new Direction C diagnosis does **not** support the claim that the large overlay mismatch is mainly a pure depth-axis problem.
- On the current MTL output, planar / structural drift is larger than the decomposed depth component on average.
- This means the updated visualization is now useful not only for showing that the overlay is poor, but also for showing that the failure mode is broader than “monocular depth only”.

### Direction C result

- Overall MAE vs native GT: `31.20°`
- Overall MAE vs fair GT: `30.35°`
- This confirms the previous `35.38°` figure was partly inflated by evaluation semantics, but MTL still remains far behind SKT.

### Direction B SGBM audit

- Re-ran `02_dense_stereo_sgbm/src/08_disparity_analysis.py` with block-size sweep:
  - `blockSize=9`  → fill `0.782`, MAE vs DLT `36.10 px`, corr `0.240`
  - `blockSize=11` → fill `0.781`, MAE vs DLT `37.18 px`, corr `0.232`
  - `blockSize=15` → fill `0.716`, MAE vs DLT `32.75 px`, corr `0.240`

### Direction B interpretation

- No tested block-size variant produced a meaningful qualitative recovery.
- `blockSize=11` offered no gain over the current default.
- `blockSize=15` reduced MAE slightly but damaged fill rate too much to help full-body reconstruction.
- Working conclusion: DDM is currently limited more by SGBM + low-texture industrial scenes than by an obvious single-parameter mistake.

### Visualization outputs

- Created Direction A comparison video:
  - `01_stereo_triangulation/results/skeleton_comparison_dirA.mp4`
- Created Direction C comparison video:
  - `03_mono_motionbert/results_mono/skeleton_comparison_dirC.mp4`
- Created Direction B interactive point-cloud viewer:
  - `02_dense_stereo_sgbm/results/interactive_pointcloud.html`
