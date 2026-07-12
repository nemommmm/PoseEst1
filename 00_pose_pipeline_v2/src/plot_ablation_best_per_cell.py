#!/opt/anaconda3/envs/pose/bin/python
"""Plot best-of-best angle time series per (pipeline, model) cell.

For each of the 8 (dataset, pipeline, model) cells, apply the offline
post-processing variant that gave the lowest MAE (established by running
offline_bone_constraint_experiment.py with all 8 cells), then plot the
resulting angle series against FastSAM3D reference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.angles import compute_angle_sequence
from common.config import load_config, section
from common.dataset import apply_depth_consistency_filter, apply_skt_quality_filter, load_skt_keypoints
from eval_angles import prepare_angles
from eval_filter_ablation import smooth_keypoints_savgol
from estimate_offset import load_selected_offset
from offline_bone_constraint_experiment import (
    DEFAULT_DATASETS,
    apply_soft_bone_constraints,
    estimate_limb_priors,
    process_angles,
    resolve_run_dir,
)


@dataclass
class CellBest:
    label: str
    dataset_key: str      # DEFAULT_DATASETS key
    variant: str           # one of: current_eval_chain, savgol_only_no_depth,
                           # savgol_only_with_depth, bone_savgol_no_depth,
                           # bone_savgol_with_depth
    lam: float | None = None


# Controlled ablation: fix post-processing at last-weekly-report's pipeline
# (bone constraint + depth filter + savgol, λ=3.0, savgol window=7), then vary
# only pipeline (V1/V2) and model (YOLOv8m/YOLO11l).
FIXED_VARIANT = "bone_savgol_with_depth"
FIXED_LAMBDA = 3.0

BEST_PER_CELL = {
    "fanbo7": [
        CellBest("V1+YOLOv8m", "fanbo7_v1_yolov8m", FIXED_VARIANT, lam=FIXED_LAMBDA),
        CellBest("V1+YOLO11l", "fanbo7_v1_yolo11l", FIXED_VARIANT, lam=FIXED_LAMBDA),
        CellBest("V2+YOLOv8m", "fanbo7_v2_yolov8m", FIXED_VARIANT, lam=FIXED_LAMBDA),
        CellBest("V2+YOLO11l", "fanbo7_v2_yolo11l", FIXED_VARIANT, lam=FIXED_LAMBDA),
    ],
    "fanbo4": [
        CellBest("V1+YOLOv8m", "fanbo4_v1_yolov8m", FIXED_VARIANT, lam=FIXED_LAMBDA),
        CellBest("V1+YOLO11l", "fanbo4_v1_yolo11l", FIXED_VARIANT, lam=FIXED_LAMBDA),
        CellBest("V2+YOLOv8m", "fanbo4_v2_yolov8m", FIXED_VARIANT, lam=FIXED_LAMBDA),
        CellBest("V2+YOLO11l", "fanbo4_v2_yolo11l", FIXED_VARIANT, lam=FIXED_LAMBDA),
    ],
}

COLORS = {
    "V1+YOLOv8m": "tab:orange",
    "V1+YOLO11l": "tab:green",
    "V2+YOLOv8m": "tab:red",
    "V2+YOLO11l": "tab:purple",
    "FastSAM3D":  "tab:blue",
}

ANGLE_NAME = "RightElbow"


def compute_variant_angles(cell: CellBest) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_s, skt_angles, fastsam_angles, valid_mask) for the chosen variant."""
    spec = DEFAULT_DATASETS[cell.dataset_key]
    config = load_config(PROJECT_ROOT / spec.config_path if not spec.config_path.is_absolute() else spec.config_path)
    run_dir = resolve_run_dir(config, spec)
    offset_s = load_selected_offset(run_dir)
    time_s, all_angles, _ = prepare_angles(config, run_dir, offset_s)
    fastsam = all_angles["FastSAM3D"][ANGLE_NAME]

    if cell.variant == "current_eval_chain":
        skt = all_angles["SKT"][ANGLE_NAME]
        return time_s, skt, fastsam, np.isfinite(skt) & np.isfinite(fastsam)

    _, raw_kp, payload = load_skt_keypoints(config, run_dir)
    raw_kp = raw_kp[: len(time_s)]
    quality = None
    if "stereo_quality" in payload.files:
        quality = np.asarray(payload["stereo_quality"], dtype=np.float64)[: len(time_s)]

    max_gap = int(section(config, "evaluation").get("max_gap_frames", 5))

    if cell.variant == "savgol_only_no_depth":
        kp, _ = apply_skt_quality_filter(raw_kp, payload, config)
        kp = smooth_keypoints_savgol(kp, time_s, max_gap=max_gap, window=7, polyorder=2)
    elif cell.variant == "savgol_only_with_depth":
        kp, _ = apply_skt_quality_filter(raw_kp, payload, config)
        kp, _ = apply_depth_consistency_filter(kp, config)
        kp = smooth_keypoints_savgol(kp, time_s, max_gap=max_gap, window=7, polyorder=2)
    elif cell.variant in ("bone_savgol_no_depth", "bone_savgol_with_depth"):
        priors = estimate_limb_priors(raw_kp, 25.0)
        kp = apply_soft_bone_constraints(raw_kp, priors, float(cell.lam), quality)
        kp, _ = apply_skt_quality_filter(kp, payload, config)
        if cell.variant == "bone_savgol_with_depth":
            kp, _ = apply_depth_consistency_filter(kp, config)
        kp = smooth_keypoints_savgol(kp, time_s, max_gap=max_gap, window=7, polyorder=2)
    else:
        raise ValueError(f"Unknown variant: {cell.variant}")

    processed = process_angles(kp, time_s, config, [ANGLE_NAME])
    skt = processed[ANGLE_NAME]
    return time_s, skt, fastsam, np.isfinite(skt) & np.isfinite(fastsam)


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for ax, dataset in zip(axes, ["fanbo7", "fanbo4"]):
        cells = BEST_PER_CELL[dataset]
        # Plot FastSAM3D once (any cell has same reference)
        t0, skt0, fast0, _ = compute_variant_angles(cells[0])
        fv = np.isfinite(fast0)
        ax.plot(t0, fast0, color=COLORS["FastSAM3D"], lw=2.2, alpha=0.85,
                label="FastSAM3D (reference)", zorder=3)
        xlim = (t0[fv].min() - 0.3, t0[fv].max() + 0.3) if fv.any() else (t0.min(), t0.max())

        for cell in cells:
            t, skt, fast, valid = compute_variant_angles(cell)
            n = int(valid.sum())
            mae = float(np.nanmean(np.abs(skt - fast)))
            bias = float(np.nanmean(skt - fast))
            variant_short = cell.variant if cell.lam is None else f"{cell.variant} λ={cell.lam}"
            ax.plot(t, skt, color=COLORS[cell.label], lw=1.2, alpha=0.9,
                    label=f"{cell.label}: MAE={mae:.2f}°  bias={bias:+.2f}°  n={n}  ({variant_short})")

        depth_tag = "near ~180 cm" if dataset == "fanbo7" else "far ~410 cm"
        ax.set_title(f"{dataset} ({depth_tag}) — Right Elbow, fixed post-processing "
                     f"(bone+savgol+depth, λ=3.0, window=7)",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylabel("Angle (°)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Controlled ablation: Pipeline (V1/V2) × Model (YOLOv8m/YOLO11l) "
                 "under identical post-processing",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = PROJECT_ROOT / "00_pose_pipeline_v2/runs/ablation_pipeline_model/controlled_ablation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
