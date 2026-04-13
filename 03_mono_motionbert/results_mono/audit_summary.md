# Monocular Direction Audit Summary

## Structure / engineering issues
- `evaluate_vs_gt.py` still assumed the pre-refactor repo layout and referenced `src/` plus a root-level `results/alignment_summary.json`; this needed to be updated to `shared/` and `01_stereo_triangulation/results/alignment_summary.json`.
- `run_pipeline.sh` was passing outdated `--output-dir` flags and using the raw video again for stage 2 instead of the generated `video_outputs.json`.
- The monocular branch had no lightweight diagnostic visualization of the final 3D output, which made temporal drift and body-shape failures hard to inspect.

## Tunable parameters
- The primary stable artifact for downstream visualization is `results_mono/markers_results_mono.trc`, not the raw `video_outputs.json`.
- Render-side choices that matter are camera view, missing-marker handling, and frame-rate preservation; they should not modify the underlying 3D sequence.
- For evaluation, the immediate high-impact fix was semantic correctness, not model retraining: elbow flexion needed `180 - interior_angle`, and hip angles needed to be added so Direction C could be compared on the same 8-joint basis as Direction A.

## Method limitations observed so far
- The monocular branch still depends on learned depth priors and is expected to drift more in the sagittal/depth direction than stereo.
- The current Aitor inference path does **not** explicitly rotate the upside-down raw video inside `run_inference.py`. This is an important caveat: at the moment, Direction C is consistent with Aitor's released pipeline, but not yet guaranteed to be consistent with Direction A's upright-input assumption.
- After fixing the elbow semantics bug and adding hip evaluation, the current monocular branch evaluates at roughly `31.20°` overall MAE instead of the previously inflated `35.38°`. This is a meaningful correction, but it still leaves MTL clearly behind SKT.

## New diagnostic outputs from this round
- A new comparison video now produces:
  - `results_mono/skeleton_comparison_dirC.mp4`
  - `results_mono/skeleton_comparison_dirC.json`
- This video overlays the monocular TRC skeleton and Xsens GT in one aligned 3D coordinate frame, which makes long-horizon drift immediately visible.
- The current metadata shows very large mean overlay errors (`~218 cm` anchor distance, `~207 cm` pelvis distance), confirming that the main remaining weakness is not just evaluation syntax but the monocular 3D sequence itself.
