#!/opt/anaconda3/envs/pose/bin/python
"""Ablate position-level postprocessing on existing SKT outputs.

The variants in this script keep suspicious observations in the trajectory and
change only how much each stage trusts them. Xsens is not used as absolute
ground truth; FastSAM3D is used here as the available comparison trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.angles import compute_angle_sequence, fill_short_gaps, moving_average, odd_window_from_ms
from common.config import load_config, resolve_path, section
from common.dataset import load_skt_keypoints
from common.metrics import jsonable, mae, median_abs_error, rmse
from common.position_postprocess import (
    flag_positions,
    kf_rts_smooth_positions,
    soft_bone_constrain_positions,
)
from estimate_offset import load_selected_offset
from eval_angles import prepare_angles
from eval_vs_fastsam import angular_acc_rms, count_jumps


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration needed to evaluate one existing pipeline run."""

    name: str
    config_path: Path
    run_dir: Path | None = None


DEFAULT_DATASETS = {
    "fanbo3": DatasetSpec(
        name="fanbo3",
        config_path=Path("00_pose_pipeline_v2/configs/assar2026_fanbo3_a255.yaml"),
    ),
    "fanbo4": DatasetSpec(
        name="fanbo4",
        config_path=Path("00_pose_pipeline_v2/configs/assar2026_fanbo4_a257.yaml"),
    ),
    "fanbo7": DatasetSpec(
        name="fanbo7",
        config_path=Path("00_pose_pipeline_v2/configs/assar2026_fanbo7_a257.yaml"),
        run_dir=Path("00_pose_pipeline_v2/runs/assar2026_fanbo7_a257_stage1_geometry"),
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["fanbo7", "fanbo4", "fanbo3"], choices=sorted(DEFAULT_DATASETS))
    parser.add_argument("--out", type=Path, default=Path("00_pose_pipeline_v2/runs/stage2_position_postprocess_ablation"))
    parser.add_argument("--bone-lambda", type=float, default=1.0)
    parser.add_argument("--process-accel-std", type=float, default=250.0)
    parser.add_argument("--measurement-std", type=float, default=2.0)
    parser.add_argument("--jump-threshold-deg", type=float, default=10.0)
    parser.add_argument("--trim-percentile", type=float, default=25.0)
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


def process_angles(
    keypoints: np.ndarray,
    time_s: np.ndarray,
    config: dict,
    angle_names: list[str],
) -> dict[str, np.ndarray]:
    """Compute angle series using the shared 200 ms evaluation convention."""
    eval_cfg = section(config, "evaluation")
    raw = compute_angle_sequence(keypoints, angle_names)
    _, radius, _ = odd_window_from_ms(time_s, float(eval_cfg.get("camera_smooth_window_ms", 200.0)))
    max_gap = int(eval_cfg.get("max_gap_frames", 5))
    processed: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        filled, _ = fill_short_gaps(values, time_s, max_gap)
        smoothed = moving_average(filled, radius)
        smoothed[~np.isfinite(filled)] = np.nan
        processed[name] = smoothed
    return processed


def all_finite_weight(keypoints: np.ndarray) -> np.ndarray:
    """Return unit measurement weight for finite joint positions."""
    finite = np.isfinite(keypoints).all(axis=2)
    return finite.astype(np.float64)


def metric_rows_for_variant(
    dataset_name: str,
    variant: str,
    time_s: np.ndarray,
    target_angles: dict[str, np.ndarray],
    reference_angles: dict[str, np.ndarray],
    angle_names: list[str],
    jump_threshold: float,
    variant_meta: dict[str, object],
) -> list[dict[str, object]]:
    """Build metric rows for one position-postprocess variant."""
    rows: list[dict[str, object]] = []
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
        row.update(variant_meta)
        rows.append(row)
    return rows


def build_variants(
    raw_keypoints: np.ndarray,
    time_s: np.ndarray,
    payload,
    config: dict,
    *,
    bone_lambda: float,
    process_accel_std: float,
    measurement_std: float,
    trim_percentile: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Create position-level ablation variants from raw SKT keypoints."""
    flags = flag_positions(raw_keypoints, time_s, payload, trim_percentile=trim_percentile)
    priors = flags.bone_priors_cm
    unit_weight = all_finite_weight(raw_keypoints)
    variants: dict[str, np.ndarray] = {
        "raw_positions": raw_keypoints,
    }
    meta: dict[str, dict[str, object]] = {
        "raw_positions": {
            "velocity_flag": False,
            "bone_correct": False,
            "kf_rts_smooth": False,
        }
    }

    bone_only = soft_bone_constrain_positions(raw_keypoints, priors, lam=bone_lambda)
    variants["bone_only"] = bone_only
    meta["bone_only"] = {"velocity_flag": False, "bone_correct": True, "kf_rts_smooth": False}

    smooth_only = kf_rts_smooth_positions(
        raw_keypoints,
        time_s,
        unit_weight,
        process_accel_std_cm_s2=process_accel_std,
        measurement_std_cm=measurement_std,
    )
    variants["smooth_only"] = smooth_only
    meta["smooth_only"] = {"velocity_flag": False, "bone_correct": False, "kf_rts_smooth": True}

    bone_smooth = kf_rts_smooth_positions(
        bone_only,
        time_s,
        unit_weight,
        process_accel_std_cm_s2=process_accel_std,
        measurement_std_cm=measurement_std,
    )
    variants["bone_smooth"] = bone_smooth
    meta["bone_smooth"] = {"velocity_flag": False, "bone_correct": True, "kf_rts_smooth": True}

    flag_bone = soft_bone_constrain_positions(
        raw_keypoints,
        priors,
        lam=bone_lambda,
        measurement_weight=flags.measurement_weight,
        flagged_prior_boost=1.0,
    )
    flag_bone_smooth = kf_rts_smooth_positions(
        flag_bone,
        time_s,
        flags.measurement_weight,
        process_accel_std_cm_s2=process_accel_std,
        measurement_std_cm=measurement_std,
    )
    variants["flag_bone_smooth"] = flag_bone_smooth
    meta["flag_bone_smooth"] = {
        "velocity_flag": True,
        "bone_correct": True,
        "kf_rts_smooth": True,
        **{f"flag_{key}": value for key, value in flags.stats.items()},
    }
    for variant_meta in meta.values():
        variant_meta.update({
            "bone_lambda": bone_lambda,
            "process_accel_std_cm_s2": process_accel_std,
            "measurement_std_cm": measurement_std,
        })
    return variants, meta


def summarize_dataset(
    spec: DatasetSpec,
    *,
    bone_lambda: float,
    process_accel_std: float,
    measurement_std: float,
    jump_threshold: float,
    trim_percentile: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the position-level ablation for one dataset."""
    config = load_config(resolve_project_path(spec.config_path))
    run_dir = resolve_run_dir(config, spec)
    if not (run_dir / "alignment_summary.json").exists():
        raise FileNotFoundError(f"Missing alignment_summary.json in {run_dir}")
    offset_s = load_selected_offset(run_dir)
    time_s, all_angles, info = prepare_angles(config, run_dir, offset_s)
    if "FastSAM3D" not in all_angles:
        raise RuntimeError(f"{spec.name}: FastSAM3D comparison trajectory is not available.")

    _, raw_keypoints, payload = load_skt_keypoints(config, run_dir)
    raw_keypoints = raw_keypoints[: len(time_s)]
    eval_angle_names = [
        name
        for name in section(config, "evaluation").get("angle_names", list(all_angles["FastSAM3D"]))
        if name in all_angles["FastSAM3D"]
    ]

    rows: list[dict[str, object]] = []
    rows.extend(
        metric_rows_for_variant(
            dataset_name=spec.name,
            variant="current_eval_chain",
            time_s=time_s,
            target_angles=all_angles["SKT"],
            reference_angles=all_angles["FastSAM3D"],
            angle_names=eval_angle_names,
            jump_threshold=jump_threshold,
            variant_meta={
                "velocity_flag": False,
                "bone_correct": False,
                "kf_rts_smooth": False,
                "uses_existing_hard_filters": True,
            },
        )
    )

    variants, meta = build_variants(
        raw_keypoints,
        time_s,
        payload,
        config,
        bone_lambda=bone_lambda,
        process_accel_std=process_accel_std,
        measurement_std=measurement_std,
        trim_percentile=trim_percentile,
    )
    for variant_name, positions in variants.items():
        angles = process_angles(positions, time_s, config, eval_angle_names)
        rows.extend(
            metric_rows_for_variant(
                dataset_name=spec.name,
                variant=variant_name,
                time_s=time_s,
                target_angles=angles,
                reference_angles=all_angles["FastSAM3D"],
                angle_names=eval_angle_names,
                jump_threshold=jump_threshold,
                variant_meta={**meta[variant_name], "uses_existing_hard_filters": False},
            )
        )

    details = {
        "dataset": spec.name,
        "config_path": str(resolve_project_path(spec.config_path)),
        "run_dir": str(run_dir),
        "selected_offset_seconds": offset_s,
        "angle_names": eval_angle_names,
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


def write_compact_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dataset/variant compact averages to CSV."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["variant"])), []).append(row)
    compact = []
    for (dataset, variant), group_rows in sorted(groups.items()):
        compact.append({
            "dataset": dataset,
            "variant": variant,
            "mean_mae_deg": float(np.nanmean([float(row["mae_deg"]) for row in group_rows])),
            "mean_rmse_deg": float(np.nanmean([float(row["rmse_deg"]) for row in group_rows])),
            "mean_acc_rms_deg_s2": float(np.nanmean([float(row["target_angular_acc_rms_deg_s2"]) for row in group_rows])),
            "total_jumps": int(np.nansum([int(row["target_jump_count"]) for row in group_rows])),
            "mean_valid_ratio": float(np.nanmean([float(row["valid_ratio"]) for row in group_rows])),
        })
    write_csv(path, compact)


def main() -> None:
    """Run the position-level postprocess ablation."""
    args = parse_args()
    out_dir = resolve_project_path(args.out)
    all_rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        spec = DEFAULT_DATASETS[dataset_name]
        print(f"[position_ablation] {dataset_name}")
        rows, info = summarize_dataset(
            spec,
            bone_lambda=float(args.bone_lambda),
            process_accel_std=float(args.process_accel_std),
            measurement_std=float(args.measurement_std),
            jump_threshold=float(args.jump_threshold_deg),
            trim_percentile=float(args.trim_percentile),
        )
        all_rows.extend(rows)
        details.append(info)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "summary.csv", all_rows)
    write_compact_summary(out_dir / "compact_summary.csv", all_rows)
    summary = {
        "datasets": args.datasets,
        "config": {
            "bone_lambda": args.bone_lambda,
            "process_accel_std_cm_s2": args.process_accel_std,
            "measurement_std_cm": args.measurement_std,
            "jump_threshold_deg": args.jump_threshold_deg,
            "trim_percentile": args.trim_percentile,
            "reference": "FastSAM3D comparison trajectory; not absolute ground truth.",
        },
        "details": details,
        "rows": all_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[position_ablation] saved {out_dir / 'summary.csv'}")
    print(f"[position_ablation] saved {out_dir / 'compact_summary.csv'}")


if __name__ == "__main__":
    main()
