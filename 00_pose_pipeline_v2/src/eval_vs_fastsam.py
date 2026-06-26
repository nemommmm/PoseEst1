"""Evaluate selected angle series against FastSAM3D as the comparison reference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common.config import section
from common.metrics import jsonable, mae, median_abs_error, rmse, rula_bin
from eval_angles import prepare_angles
from estimate_offset import load_selected_offset


def angular_acc_rms(angle: np.ndarray, time_s: np.ndarray) -> float | None:
    """Return RMS angular acceleration in deg/s^2 for finite second differences."""
    values = np.asarray(angle, dtype=np.float64)
    if len(values) < 3:
        return None
    diffs = np.diff(np.asarray(time_s, dtype=np.float64))
    finite_dt = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(finite_dt) == 0:
        return None
    dt = float(np.median(finite_dt))
    second_diff = np.diff(values, n=2) / (dt * dt)
    finite = second_diff[np.isfinite(second_diff)]
    if len(finite) == 0:
        return None
    return float(np.sqrt(np.mean(finite * finite)))


def count_jumps(angle: np.ndarray, threshold_deg: float) -> int:
    """Count consecutive-frame angle jumps above a threshold."""
    diffs = np.diff(np.asarray(angle, dtype=np.float64))
    finite = diffs[np.isfinite(diffs)]
    return int(np.sum(np.abs(finite) > float(threshold_deg)))


def _valid_interval(time_s: np.ndarray, valid: np.ndarray) -> tuple[float | None, float | None]:
    """Return first/last timestamp covered by a valid finite-pair mask."""
    if not np.any(valid):
        return None, None
    covered = np.asarray(time_s, dtype=np.float64)[valid]
    return float(covered[0]), float(covered[-1])


def build_rows(
    time_s: np.ndarray,
    all_angles: dict[str, dict[str, np.ndarray]],
    angle_names: list[str],
    targets: list[str],
    rula_bins: dict[str, list[float]],
    jump_threshold_deg: float,
) -> list[dict[str, object]]:
    """Build FastSAM3D-referenced rows for selected target systems."""
    if "FastSAM3D" not in all_angles:
        raise RuntimeError("FastSAM3D is not available. Check references.fastsam_trc in the config.")

    rows: list[dict[str, object]] = []
    reference = all_angles["FastSAM3D"]
    for target_name in targets:
        if target_name not in all_angles:
            continue
        target_angles = all_angles[target_name]
        for angle_name in angle_names:
            target = target_angles.get(angle_name)
            ref = reference.get(angle_name)
            if target is None or ref is None:
                continue
            valid = np.isfinite(target) & np.isfinite(ref)
            t0, t1 = _valid_interval(time_s, valid)
            if np.any(valid):
                valid_idx = np.where(valid)[0]
                start_idx = int(valid_idx[0])
                end_idx = int(valid_idx[-1]) + 1
                target_window = target[start_idx:end_idx]
                ref_window = ref[start_idx:end_idx]
                time_window = time_s[start_idx:end_idx]
            else:
                target_window = target[:0]
                ref_window = ref[:0]
                time_window = time_s[:0]
            bins = rula_bins.get(angle_name)
            agreement = None
            if bins and np.any(valid):
                agreement = float(np.mean(rula_bin(target[valid], bins) == rula_bin(ref[valid], bins)))
            rows.append({
                "target": target_name,
                "reference": "FastSAM3D",
                "angle": angle_name,
                "valid_pair_count": int(valid.sum()),
                "valid_ratio": float(valid.mean()) if len(valid) else 0.0,
                "overlap_start_s": t0,
                "overlap_end_s": t1,
                "mae_deg": mae(target, ref),
                "median_abs_error_deg": median_abs_error(target, ref),
                "rmse_deg": rmse(target, ref),
                "bias_deg": float(np.nanmean(target[valid] - ref[valid])) if np.any(valid) else None,
                "rula_like_agreement": agreement,
                "target_angular_acc_rms_deg_s2": angular_acc_rms(target_window, time_window),
                "reference_angular_acc_rms_deg_s2": angular_acc_rms(ref_window, time_window),
                "target_jump_count": count_jumps(target_window, jump_threshold_deg),
                "reference_jump_count": count_jumps(ref_window, jump_threshold_deg),
                "target_jump_count_full_timeline": count_jumps(target, jump_threshold_deg),
                "reference_jump_count_full_timeline": count_jumps(ref, jump_threshold_deg),
                "jump_threshold_deg": float(jump_threshold_deg),
            })
    return rows


def evaluate_vs_fastsam(config: dict, run_dir: Path) -> Path:
    """Run FastSAM3D-referenced angle and smoothness evaluation."""
    offset_s = load_selected_offset(run_dir)
    time_s, all_angles, info = prepare_angles(config, run_dir, offset_s)
    eval_cfg = section(config, "evaluation")
    fast_cfg = section(config, "fasteval")
    angle_names = list(fast_cfg.get("angle_names", info["angle_names"]))
    targets = list(fast_cfg.get("targets", ["SKT"]))
    jump_threshold = float(fast_cfg.get("jump_threshold_deg", 10.0))
    rows = build_rows(
        time_s=time_s,
        all_angles=all_angles,
        angle_names=angle_names,
        targets=targets,
        rula_bins=eval_cfg.get("rula_bins", {}),
        jump_threshold_deg=jump_threshold,
    )

    out_dir = run_dir / "eval_vs_fastsam"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "target", "reference", "angle", "valid_pair_count", "valid_ratio",
        "overlap_start_s", "overlap_end_s", "mae_deg", "median_abs_error_deg",
        "rmse_deg", "bias_deg", "rula_like_agreement",
        "target_angular_acc_rms_deg_s2", "reference_angular_acc_rms_deg_s2",
        "target_jump_count", "reference_jump_count",
        "target_jump_count_full_timeline", "reference_jump_count_full_timeline",
        "jump_threshold_deg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": {
            "selected_offset_seconds": offset_s,
            "reference": "FastSAM3D comparison trajectory",
            "targets": targets,
            "angle_names": angle_names,
            "jump_threshold_deg": jump_threshold,
            "trc_summaries": info.get("trc_summaries", {}),
        },
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[fasteval] saved {csv_path}")
    return csv_path
