# AFH1 Hybrid Pipeline

This directory contains the implementation workspace for the AFH1 hybrid
pipeline:

- EasyErgo / FastSAM3D provides the relative 3D skeleton
- Stereo triangulation provides the global pelvis anchor
- The hybrid result is evaluated against Xsens using the existing ergonomic
  evaluation pipeline

## Current Scope

Version `AFH1 v1` intentionally stays minimal:

- Use stereo pelvis as the only global translation anchor
- Use one constant rotation to align EasyErgo coordinates into the stereo frame
- Do not fit scale beyond unit normalization
- Do not block the first evaluation on TRC / OpenSim export

## Unit and Time Conventions

- Internal coordinates: `cm`
- Output skeleton frame: stereo coordinate system
- Hybrid interpolation grid: stereo relative timestamps
- Final evaluation timestamps: stereo absolute timestamps copied into the output
  NPZ

## Manual Input

Place the downloaded EasyErgo TRC file here:

- `04_hybrid_afh1/data/easyergo_uploaded/markers_easyergo.trc`

## Results

Intermediate outputs are written into:

- `04_hybrid_afh1/results/`

Important result files:

- `easyergo_trc_inspection.json`
- `easyergo_trc_inspection.md`
- `stereo_pelvis_anchor.npz`
- `experiment_log.md`
