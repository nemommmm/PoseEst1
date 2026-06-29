#!/opt/anaconda3/envs/pose/bin/python
"""Offline bone-length constraint ablation for existing SKT NPZ outputs.

This script is intentionally not part of the main inference pipeline. It reads
existing SKT keypoints, applies a soft per-frame limb-length constraint, and
evaluates the resulting angle series against FastSAM3D. The goal is to test
whether bone-length priors are a useful stabilizer before integrating them into
the online SKT pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.angles import compute_angle_sequence, fill_short_gaps, moving_average, odd_window_from_ms
from common.config import load_config, resolve_path, section
from common.dataset import (
    apply_depth_consistency_filter,
    apply_skt_quality_filter,
    load_skt_keypoints,
)
from common.metrics import jsonable, mae, median_abs_error, rmse
from eval_angles import prepare_angles
from eval_filter_ablation import smooth_keypoints_savgol
from eval_vs_fastsam import angular_acc_rms, count_jumps
from estimate_offset import load_selected_offset


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration needed to evaluate one existing pipeline run."""

    name: str
    config_path: Path
    run_dir: Path | None = None


DEFAULT_DATASETS = {
    "fanbo4": DatasetSpec(
        name="fanbo4",
        config_path=Path("00_pose_pipeline_v2/configs/assar2026_fanbo4_a257.yaml"),
    ),
    "fanbo7": DatasetSpec(
        name="fanbo7",
        config_path=Path("00_pose_pipeline_v2/configs/assar2026_fanbo7_a257.yaml"),
        run_dir=Path("00_pose_pipeline_v2/runs/assar2026_fanbo7_a257_stage1_geometry"),
    ),
    "2025_ergonomics": DatasetSpec(
        name="2025_ergonomics",
        config_path=Path("00_pose_pipeline_v2/configs/current_2025_ergonomics.yaml"),
    ),
}

DEFAULT_PRIORS_CM = {
    "left_upper_arm": 28.0,
    "right_upper_arm": 28.0,
    "left_lower_arm": 26.0,
    "right_lower_arm": 26.0,
    "left_thigh": 39.5,
    "right_thigh": 39.5,
    "left_shank": 40.5,
    "right_shank": 40.5,
}

LIMB_CHAINS = [
    ("left_arm", 5, 7, 9, "left_upper_arm", "left_lower_arm"),
    ("right_arm", 6, 8, 10, "right_upper_arm", "right_lower_arm"),
    ("left_leg", 11, 13, 15, "left_thigh", "left_shank"),
    ("right_leg", 12, 14, 16, "right_thigh", "right_shank"),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fanbo4", "fanbo7", "2025_ergonomics"],
        choices=sorted(DEFAULT_DATASETS),
        help="Dataset aliases to evaluate.",
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("00_pose_pipeline_v2/runs/stage1.5_bone_offline"),
        help="Output directory for the retained experiment summary.",
    )
    parser.add_argument(
        "--trim-percentile",
        type=float,
        default=25.0,
        help="Trim percentile used when estimating bone priors from each session.",
    )
    parser.add_argument(
        "--jump-threshold-deg",
        type=float,
        default=10.0,
        help="Consecutive-frame angle jump threshold.",
    )
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=7,
        help="Odd Savitzky-Golay keypoint smoothing window for bone->Savgol variants.",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order for bone->Savgol variants.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve a project-root-relative path."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_run_dir(config: dict, spec: DatasetSpec) -> Path:
    """Resolve the run directory used by one dataset spec."""
    if spec.run_dir is not None:
        return resolve_project_path(spec.run_dir)
    outputs = section(config, "outputs")
    runs_dir = resolve_path(outputs.get("runs_dir", "00_pose_pipeline_v2/runs"), must_exist=True)
    tag = outputs.get("run_tag") or section(config, "dataset").get("name", spec.name)
    return runs_dir / str(tag)


def robust_median_distance(keypoints: np.ndarray, idx_a: int, idx_b: int, trim_percentile: float) -> float:
    """Return an IQR-trimmed median distance between two joints."""
    dists = np.linalg.norm(keypoints[:, idx_a, :] - keypoints[:, idx_b, :], axis=1)
    finite = dists[np.isfinite(dists)]
    if len(finite) == 0:
        return math.nan
    lo = np.percentile(finite, trim_percentile)
    hi = np.percentile(finite, 100.0 - trim_percentile)
    trimmed = finite[(finite >= lo) & (finite <= hi)]
    if len(trimmed) == 0:
        return math.nan
    return float(np.median(trimmed))


def estimate_limb_priors(keypoints: np.ndarray, trim_percentile: float) -> dict[str, float]:
    """Estimate session-specific limb-length priors from robust medians."""
    priors = dict(DEFAULT_PRIORS_CM)
    for _, prox, mid, dist, upper_name, lower_name in LIMB_CHAINS:
        upper = robust_median_distance(keypoints, prox, mid, trim_percentile)
        lower = robust_median_distance(keypoints, mid, dist, trim_percentile)
        if np.isfinite(upper):
            priors[upper_name] = upper
        if np.isfinite(lower):
            priors[lower_name] = lower
    return priors


def solve_chain_soft_constraint(
    pose: np.ndarray,
    chain: tuple[str, int, int, int, str, str],
    priors: dict[str, float],
    lam: float,
    quality: np.ndarray | None,
) -> np.ndarray:
    """Constrain one three-joint limb chain with soft bone-length residuals."""
    _, prox, mid, dist, upper_name, lower_name = chain
    joint_ids = [prox, mid, dist]
    initial = np.asarray(pose[joint_ids, :], dtype=np.float64)
    if not np.isfinite(initial).all():
        return pose

    upper_prior = float(priors.get(upper_name, math.nan))
    lower_prior = float(priors.get(lower_name, math.nan))
    if not (np.isfinite(upper_prior) and np.isfinite(lower_prior)):
        return pose

    q = np.ones(3, dtype=np.float64)
    if quality is not None:
        q = np.asarray(quality[joint_ids], dtype=np.float64)
        q = np.where(np.isfinite(q), np.clip(q, 0.05, 1.0), 0.25)
    # The proximal joint is usually more stable and anchors the limb direction.
    q[0] *= 3.0

    x0 = initial.reshape(-1)

    def residual(x: np.ndarray) -> np.ndarray:
        pts = x.reshape(3, 3)
        obs = ((pts - initial) * np.sqrt(q)[:, None]).reshape(-1)
        upper = np.linalg.norm(pts[0] - pts[1]) - upper_prior
        lower = np.linalg.norm(pts[1] - pts[2]) - lower_prior
        return np.concatenate([obs, [lam * upper, lam * lower]])

    result = least_squares(residual, x0, method="trf", max_nfev=50)
    if not result.success or not np.isfinite(result.x).all():
        return pose
    corrected = pose.copy()
    corrected[joint_ids, :] = result.x.reshape(3, 3)
    return corrected


def apply_soft_bone_constraints(
    keypoints: np.ndarray,
    priors: dict[str, float],
    lam: float,
    quality: np.ndarray | None,
) -> np.ndarray:
    """Apply soft limb-length constraints frame by frame."""
    corrected = np.asarray(keypoints, dtype=np.float64).copy()
    for frame_idx in range(len(corrected)):
        pose = corrected[frame_idx]
        q_frame = None if quality is None else quality[frame_idx]
        for chain in LIMB_CHAINS:
            pose = solve_chain_soft_constraint(pose, chain, priors, lam, q_frame)
        corrected[frame_idx] = pose
    return corrected


def process_angles(
    keypoints: np.ndarray,
    time_s: np.ndarray,
    config: dict,
    angle_names: list[str],
) -> dict[str, np.ndarray]:
    """Compute angle series with the same fill + moving-average convention."""
    eval_cfg = section(config, "evaluation")
    raw = compute_angle_sequence(keypoints, angle_names)
    _, radius, _ = odd_window_from_ms(time_s, float(eval_cfg.get("camera_smooth_window_ms", 200.0)))
    max_gap = int(eval_cfg.get("max_gap_frames", 5))
    processed = {}
    for name, values in raw.items():
        filled, _ = fill_short_gaps(values, time_s, max_gap)
        smoothed = moving_average(filled, radius)
        smoothed[~np.isfinite(filled)] = np.nan
        processed[name] = smoothed
    return processed


def bone_stats(keypoints: np.ndarray, priors: dict[str, float]) -> dict[str, float | int]:
    """Return limb-length stability diagnostics for constrained keypoints."""
    out: dict[str, float | int] = {}
    catastrophic = np.zeros(len(keypoints), dtype=bool)
    for _, prox, mid, dist, upper_name, lower_name in LIMB_CHAINS:
        for bone_name, idx_a, idx_b in [(upper_name, prox, mid), (lower_name, mid, dist)]:
            dists = np.linalg.norm(keypoints[:, idx_a, :] - keypoints[:, idx_b, :], axis=1)
            finite = np.isfinite(dists)
            prior = float(priors.get(bone_name, math.nan))
            if np.isfinite(prior):
                bad = finite & ((dists < 0.5 * prior) | (dists > 1.5 * prior))
                catastrophic |= bad
            out[f"{bone_name}_median_cm"] = float(np.nanmedian(dists)) if np.any(finite) else math.nan
            out[f"{bone_name}_std_cm"] = float(np.nanstd(dists)) if np.any(finite) else math.nan
    out["catastrophic_bone_frame_count"] = int(np.sum(catastrophic))
    return out


def metric_rows_for_variant(
    dataset_name: str,
    variant: str,
    lam: float | None,
    time_s: np.ndarray,
    target_angles: dict[str, np.ndarray],
    reference_angles: dict[str, np.ndarray],
    angle_names: list[str],
    jump_threshold: float,
    bones: dict[str, float | int],
) -> list[dict[str, object]]:
    """Build metric rows for one constrained-keypoint variant."""
    rows = []
    for angle_name in angle_names:
        target = target_angles.get(angle_name)
        reference = reference_angles.get(angle_name)
        if target is None or reference is None:
            continue
        valid = np.isfinite(target) & np.isfinite(reference)
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            start = int(valid_idx[0])
            end = int(valid_idx[-1]) + 1
            target_window = target[start:end]
            reference_window = reference[start:end]
            time_window = time_s[start:end]
        else:
            target_window = target[:0]
            reference_window = reference[:0]
            time_window = time_s[:0]
        row = {
            "dataset": dataset_name,
            "variant": variant,
            "lambda": "" if lam is None else lam,
            "angle": angle_name,
            "valid_pair_count": int(np.sum(valid)),
            "valid_ratio": float(np.mean(valid)) if len(valid) else 0.0,
            "mae_deg": mae(target, reference),
            "median_abs_error_deg": median_abs_error(target, reference),
            "rmse_deg": rmse(target, reference),
            "bias_deg": float(np.nanmean(target[valid] - reference[valid])) if np.any(valid) else None,
            "target_angular_acc_rms_deg_s2": angular_acc_rms(target_window, time_window),
            "reference_angular_acc_rms_deg_s2": angular_acc_rms(reference_window, time_window),
            "target_jump_count": count_jumps(target_window, jump_threshold),
            "reference_jump_count": count_jumps(reference_window, jump_threshold),
            "jump_threshold_deg": jump_threshold,
        }
        row.update(bones)
        rows.append(row)
    return rows


def summarize_dataset(
    spec: DatasetSpec,
    lambdas: Iterable[float],
    trim_percentile: float,
    jump_threshold: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the offline bone ablation for one dataset."""
    config = load_config(resolve_project_path(spec.config_path))
    run_dir = resolve_run_dir(config, spec)
    if not (run_dir / "alignment_summary.json").exists():
        raise FileNotFoundError(f"Missing alignment_summary.json in {run_dir}")
    offset_s = load_selected_offset(run_dir)
    time_s, all_angles, info = prepare_angles(config, run_dir, offset_s)
    if "FastSAM3D" not in all_angles:
        raise RuntimeError(f"{spec.name}: FastSAM3D reference is not available.")

    _, raw_keypoints, payload = load_skt_keypoints(config, run_dir)
    raw_keypoints = raw_keypoints[: len(time_s)]
    quality = None
    if "stereo_quality" in payload.files:
        quality = np.asarray(payload["stereo_quality"], dtype=np.float64)[: len(time_s)]
    priors = estimate_limb_priors(raw_keypoints, trim_percentile)

    eval_angle_names = [
        name
        for name in section(config, "evaluation").get("angle_names", list(all_angles["FastSAM3D"]))
        if name in all_angles["FastSAM3D"]
    ]
    rows: list[dict[str, object]] = []
    baseline_bones = bone_stats(raw_keypoints, priors)
    rows.extend(
        metric_rows_for_variant(
            dataset_name=spec.name,
            variant="current_eval_chain",
            lam=None,
            time_s=time_s,
            target_angles=all_angles["SKT"],
            reference_angles=all_angles["FastSAM3D"],
            angle_names=eval_angle_names,
            jump_threshold=jump_threshold,
            bones=baseline_bones,
        )
    )

    for lam in lambdas:
        constrained = apply_soft_bone_constraints(raw_keypoints, priors, float(lam), quality)
        # Keep the same confidence/epipolar gate, but do not reapply the depth-consistency
        # filter here: this experiment tests whether bone priors can replace that brittle
        # NaN -> fill -> smooth failure mode.
        constrained, _ = apply_skt_quality_filter(constrained, payload, config)
        constrained_angles = process_angles(constrained, time_s, config, eval_angle_names)
        rows.extend(
            metric_rows_for_variant(
                dataset_name=spec.name,
                variant="soft_bone_no_depth_filter",
                lam=float(lam),
                time_s=time_s,
                target_angles=constrained_angles,
                reference_angles=all_angles["FastSAM3D"],
                angle_names=eval_angle_names,
                jump_threshold=jump_threshold,
                bones=bone_stats(constrained, priors),
            )
        )

        constrained_savgol = smooth_keypoints_savgol(
            constrained,
            time_s,
            max_gap=int(section(config, "evaluation").get("max_gap_frames", 5)),
            window=int(savgol_window),
            polyorder=int(savgol_polyorder),
        )
        constrained_savgol_angles = process_angles(constrained_savgol, time_s, config, eval_angle_names)
        rows.extend(
            metric_rows_for_variant(
                dataset_name=spec.name,
                variant="soft_bone_no_depth_filter_savgol",
                lam=float(lam),
                time_s=time_s,
                target_angles=constrained_savgol_angles,
                reference_angles=all_angles["FastSAM3D"],
                angle_names=eval_angle_names,
                jump_threshold=jump_threshold,
                bones=bone_stats(constrained_savgol, priors),
            )
        )

        constrained_depth, _ = apply_depth_consistency_filter(constrained, config)
        constrained_depth_angles = process_angles(constrained_depth, time_s, config, eval_angle_names)
        rows.extend(
            metric_rows_for_variant(
                dataset_name=spec.name,
                variant="soft_bone_with_current_depth_filter",
                lam=float(lam),
                time_s=time_s,
                target_angles=constrained_depth_angles,
                reference_angles=all_angles["FastSAM3D"],
                angle_names=eval_angle_names,
                jump_threshold=jump_threshold,
                bones=bone_stats(constrained_depth, priors),
            )
        )

        constrained_depth_savgol = smooth_keypoints_savgol(
            constrained_depth,
            time_s,
            max_gap=int(section(config, "evaluation").get("max_gap_frames", 5)),
            window=int(savgol_window),
            polyorder=int(savgol_polyorder),
        )
        constrained_depth_savgol_angles = process_angles(constrained_depth_savgol, time_s, config, eval_angle_names)
        rows.extend(
            metric_rows_for_variant(
                dataset_name=spec.name,
                variant="soft_bone_with_current_depth_filter_savgol",
                lam=float(lam),
                time_s=time_s,
                target_angles=constrained_depth_savgol_angles,
                reference_angles=all_angles["FastSAM3D"],
                angle_names=eval_angle_names,
                jump_threshold=jump_threshold,
                bones=bone_stats(constrained_depth_savgol, priors),
            )
        )

    details = {
        "dataset": spec.name,
        "config_path": str(resolve_project_path(spec.config_path)),
        "run_dir": str(run_dir),
        "selected_offset_seconds": offset_s,
        "angle_names": eval_angle_names,
        "bone_priors_cm": priors,
        "trc_summaries": info.get("trc_summaries", {}),
    }
    return rows, details


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the offline experiment."""
    args = parse_args()
    out_dir = resolve_project_path(args.out)
    all_rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        spec = DEFAULT_DATASETS[dataset_name]
        print(f"[bone_offline] {dataset_name}")
        rows, info = summarize_dataset(
            spec=spec,
            lambdas=args.lambdas,
            trim_percentile=float(args.trim_percentile),
            jump_threshold=float(args.jump_threshold_deg),
            savgol_window=int(args.savgol_window),
            savgol_polyorder=int(args.savgol_polyorder),
        )
        all_rows.extend(rows)
        details.append(info)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "summary.csv", all_rows)
    summary = {
        "datasets": args.datasets,
        "lambdas": args.lambdas,
        "trim_percentile": args.trim_percentile,
        "jump_threshold_deg": args.jump_threshold_deg,
        "savgol_window": args.savgol_window,
        "savgol_polyorder": args.savgol_polyorder,
        "details": details,
        "rows": all_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[bone_offline] saved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
