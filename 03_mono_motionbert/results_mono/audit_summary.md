# Monocular Direction Audit Summary

## Structure / engineering issues
- `evaluate_vs_gt.py` still assumed the pre-refactor repo layout and referenced `src/` plus a root-level `results/alignment_summary.json`; this needed to be updated to `shared/` and `01_stereo_triangulation/results/alignment_summary.json`.
- `run_pipeline.sh` was passing outdated `--output-dir` flags and using the raw video again for stage 2 instead of the generated `video_outputs.json`.
- The monocular branch had no lightweight diagnostic visualization of the final 3D output, which made temporal drift and body-shape failures hard to inspect.

## Tunable parameters
- The primary stable artifact for downstream visualization is `results_mono/markers_results_mono.trc`, not the raw `video_outputs.json`.
- Render-side choices that matter are camera view, missing-marker handling, and frame-rate preservation; they should not modify the underlying 3D sequence.

## Method limitations observed so far
- The monocular branch still depends on learned depth priors and is expected to drift more in the sagittal/depth direction than stereo.
- Upside-down video handling is currently correct for `03`: `RTMDet-MotionBert-OpenSim/run_inference.py` rotates the raw video by 180° before pose inference. The resulting TRC is already upright and must not be rotated again during rendering.

## New diagnostic outputs from this round
- A lightweight 3D skeleton renderer now produces:
  - `markers_results_mono_3d.mp4`
  - `markers_results_mono_3d.json`
- The short smoke render passed the upright check, but the full-sequence metadata failed the same geometric check, which is a useful diagnostic sign: the current monocular sequence appears to drift structurally over time rather than simply suffering from a double-rotation bug.
