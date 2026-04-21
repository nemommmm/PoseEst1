# Direction 04: EasyErgo Final MVNX

This directory is now trimmed to the retained Direction 04 path only:

- Input: EasyErgo final `*.mvnx`
- Output: fair-angle evaluation against Xsens
- Kept result target: overall fair MAE `16.59°`

## Retained Scope

Only the final MVNX branch is kept as the active implementation.

Execution order:

- `src/01_evaluate_final_mvnx.py`
  - Main evaluation entry
  - Defaults to the retained affine timing from `results/02_final_mvnx_timing/affine_fit.json`
- `src/02_diagnose_final_mvnx_timing.py`
  - Timing diagnosis script that produced the retained affine mapping
- `src/03_render_final_mvnx_video.py`
  - Optional visualization entry for the retained final MVNX output

Earlier AFH1 variants, trunk-only hybrids, OpenSim-side comparisons, and elbow-only exploratory branches were removed from this workspace to keep Direction 04 focused.

## Input Files

Place the downloaded EasyErgo final export here:

- `data/easyergo_uploaded/*.mvnx`

The current kept dataset also includes the original TRC / MOT / OSIM downloads, but they are no longer part of the active Direction 04 evaluation path.

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
/opt/anaconda3/envs/pose/bin/python 04_hybrid_afh1/src/01_evaluate_final_mvnx.py
```

Timing diagnosis:

```bash
/opt/anaconda3/envs/pose/bin/python 04_hybrid_afh1/src/02_diagnose_final_mvnx_timing.py
```

Video:

```bash
/opt/anaconda3/envs/pose/bin/python 04_hybrid_afh1/src/03_render_final_mvnx_video.py
```
