# Direction 03: FastSAM3D / EasyErgo Hybrid Pose

This directory keeps the retained FastSAM3D / EasyErgo pose branch:

- Input: EasyErgo final `*.mvnx`
- Output: fair-angle evaluation against Xsens
- Kept result target: overall fair MAE `16.59°`

## Retained Scope

The active implementation is the retained FastSAM3D / EasyErgo branch plus the
core hybrid skeleton used by the motion-level elbow evaluation.

Execution order:

- `src/01_evaluate_final_mvnx.py`
  - Main evaluation entry
  - Defaults to the retained affine timing from `results/02_final_mvnx_timing/affine_fit.json`
- `src/02_diagnose_final_mvnx_timing.py`
  - Timing diagnosis script that produced the retained affine mapping
- `src/03_render_final_mvnx_video.py`
  - Optional visualization entry for the retained final MVNX output

Earlier v2 variants, trunk-only hybrids, OpenSim-side comparisons, and point-cloud
diagnostics were moved to `archive_intermediate_20260520/` to keep this folder
focused.

## Input Files

Place the downloaded EasyErgo final export here:

- `data/easyergo_uploaded/*.mvnx`

The current kept dataset may still include original TRC / MOT / OSIM downloads,
but they are no longer part of the active FastSAM3D evaluation path.
Large uploaded AVI files were moved to `archive_intermediate_20260520/` because
they are only needed for optional video rendering.

## Retained Timing Mapping

The final kept evaluation uses:

- `gt_t = 1.0102 * est_t - 16.83`

This affine mapping is stored in:

- `results/02_final_mvnx_timing/affine_fit.json`

## Main Outputs

The retained outputs are:

- `results/01_final_mvnx_eval/`
- `results/02_final_mvnx_timing/`
- `results/03_final_mvnx_video.mp4`
- `results/03_final_mvnx_video.json`
- `results/03_final_mvnx_snapshots/`

## Quick Commands

Evaluation:

```bash
/opt/anaconda3/envs/pose/bin/python 03_FastSAM3D/src/01_evaluate_final_mvnx.py
```

Timing diagnosis:

```bash
/opt/anaconda3/envs/pose/bin/python 03_FastSAM3D/src/02_diagnose_final_mvnx_timing.py
```

Video:

```bash
/opt/anaconda3/envs/pose/bin/python 03_FastSAM3D/src/03_render_final_mvnx_video.py
```
