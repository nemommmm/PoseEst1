"""K-frame motion-delta evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common.angles import (
    SEMANTIC_ANGLE_NAMES,
    build_fair_angle_interpolators,
    build_native_angle_interpolators,
    compute_angle_sequence,
    fill_short_gaps,
    moving_average,
    odd_window_from_ms,
    sample_interpolators,
)
from common.config import resolve_path, section
from common.dataset import load_method_keypoints
from common.metrics import jsonable, mae, pearson, regression_slope, rmse, spearman
from estimate_offset import load_selected_offset


def k_delta(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute K-frame delta and validity mask."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full_like(values, np.nan)
    valid = np.zeros(len(values), dtype=bool)
    if len(values) > k:
        valid[k:] = np.isfinite(values[k:]) & np.isfinite(values[:-k])
        out[k:] = values[k:] - values[:-k]
    return out, valid


def threshold_for_k(base: float, k: int, mode: str) -> float:
    """Scale thresholds roughly like the previous Phase-4 evaluator."""
    factors = {
        1: {"anomaly": 1.0, "active": 1.0, "noise": 1.0},
        6: {"anomaly": 2.0, "active": 5.0, "noise": 4.0},
        12: {"anomaly": 3.0, "active": 10.0, "noise": 10.0},
        25: {"anomaly": 4.0, "active": 20.0, "noise": 20.0},
    }
    if k in factors:
        return float(base) * factors[k][mode]
    return float(base) * max(1.0, k / 6.0)


def build_motion_angles(config: dict, run_dir: Path, offset_s: float):
    """Load and smooth all systems before delta calculation."""
    time_s, meta, methods = load_method_keypoints(config, run_dir)
    eval_cfg = section(config, "evaluation")
    refs = section(config, "references")
    angle_names = [name for name in eval_cfg.get("angle_names", ["LeftElbow", "RightElbow"]) if name in SEMANTIC_ANGLE_NAMES]
    _, radius, actual_ms = odd_window_from_ms(time_s, float(eval_cfg.get("camera_smooth_window_ms", 200.0)))
    max_gap = int(eval_cfg.get("max_gap_frames", 5))
    all_angles = {}
    interpolated = {}
    for system, keypoints in methods.items():
        raw = compute_angle_sequence(keypoints, angle_names)
        all_angles[system] = {}
        interpolated[system] = {}
        for name, values in raw.items():
            filled, flags = fill_short_gaps(values, time_s, max_gap)
            all_angles[system][name] = moving_average(filled, radius)
            interpolated[system][name] = flags

    fair_path = resolve_path(refs.get("xsens_fair_angles"), must_exist=False)
    fair_interps = build_fair_angle_interpolators(fair_path)
    if not fair_interps:
        fair_interps = build_native_angle_interpolators(resolve_path(refs.get("xsens_mvnx"), must_exist=True))
    query_t = time_s - offset_s
    all_angles["XsensFair"] = sample_interpolators(fair_interps, query_t, angle_names)
    interpolated["XsensFair"] = {name: np.zeros(len(time_s), dtype=bool) for name in angle_names}
    if refs.get("include_xsens_native", True):
        native_interps = build_native_angle_interpolators(resolve_path(refs.get("xsens_mvnx"), must_exist=True))
        all_angles["XsensNative"] = sample_interpolators(native_interps, query_t, angle_names)
        interpolated["XsensNative"] = {name: np.zeros(len(time_s), dtype=bool) for name in angle_names}
    return time_s, all_angles, interpolated, {
        "angle_names": angle_names,
        "camera_smooth_radius_frames": radius,
        "camera_smooth_window_actual_ms": actual_ms,
        "trc_summaries": meta.get("trc_summaries", {}),
    }


def pair_metrics(target_delta: np.ndarray, ref_delta: np.ndarray, active_threshold: float, noise_threshold: float) -> dict:
    """Summarize one target/reference delta pair."""
    mask = np.isfinite(target_delta) & np.isfinite(ref_delta)
    td = target_delta[mask]
    rd = ref_delta[mask]
    active_mask = mask & (np.abs(ref_delta) > active_threshold)
    quiet_mask = mask & (np.abs(ref_delta) < noise_threshold)
    target_path = float(np.sum(np.abs(td))) if len(td) else None
    ref_path = float(np.sum(np.abs(rd))) if len(rd) else None
    return {
        "valid_pair_count": int(mask.sum()),
        "pearson_delta": pearson(ref_delta, target_delta),
        "spearman_delta": spearman(ref_delta, target_delta),
        "slope_target_vs_reference": regression_slope(ref_delta, target_delta),
        "delta_mae_deg": mae(target_delta, ref_delta),
        "delta_rmse_deg": rmse(target_delta, ref_delta),
        "active_pair_count": int(active_mask.sum()),
        "active_delta_mae_deg": mae(target_delta[active_mask], ref_delta[active_mask]),
        "active_delta_rmse_deg": rmse(target_delta[active_mask], ref_delta[active_mask]),
        "target_quiet_delta_std_deg": float(np.nanstd(target_delta[quiet_mask])) if np.any(quiet_mask) else None,
        "target_path_deg": target_path,
        "reference_path_deg": ref_path,
        "path_ratio_target_reference": None if not ref_path else target_path / ref_path,
    }


def evaluate_motion_delta(config: dict, run_dir: Path) -> Path:
    """Run K-frame motion-delta evaluation."""
    offset_s = load_selected_offset(run_dir)
    time_s, all_angles, interpolated, info = build_motion_angles(config, run_dir, offset_s)
    eval_cfg = section(config, "evaluation")
    k_list = [int(k) for k in eval_cfg.get("k_frame_list", [1, 6, 12, 25])]
    angle_names = info["angle_names"]
    systems = list(all_angles.keys())
    deltas = {system: {name: {} for name in angle_names} for system in systems}
    valid_masks = {system: {name: {} for name in angle_names} for system in systems}
    for system in systems:
        for name in angle_names:
            for k in k_list:
                deltas[system][name][k], valid_masks[system][name][k] = k_delta(all_angles[system][name], k)

    out_dir = run_dir / "motion_delta"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "motion_delta_combined.csv"
    fieldnames = ["Frame", "Time_s", "FrameDt_s"]
    for system in systems:
        for name in angle_names:
            fieldnames.append(f"{system}_{name}_deg")
            for k in k_list:
                fieldnames.extend([f"{system}_{name}_delta_k{k}_deg", f"{system}_{name}_delta_valid_k{k}"])
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t in enumerate(time_s):
            row = {"Frame": idx, "Time_s": f"{float(t):.6f}", "FrameDt_s": "" if idx == 0 else f"{float(time_s[idx] - time_s[idx - 1]):.6f}"}
            for system in systems:
                for name in angle_names:
                    value = all_angles[system][name][idx]
                    row[f"{system}_{name}_deg"] = "" if not np.isfinite(value) else f"{float(value):.6f}"
                    for k in k_list:
                        d = deltas[system][name][k][idx]
                        row[f"{system}_{name}_delta_k{k}_deg"] = "" if not np.isfinite(d) else f"{float(d):.6f}"
                        row[f"{system}_{name}_delta_valid_k{k}"] = bool(valid_masks[system][name][k][idx])
            writer.writerow(row)

    summary = {
        "config": {
            "selected_offset_seconds": offset_s,
            "reference": "XsensFair (Xsens-derived comparison/reference system)",
            "k_frame_list": k_list,
            "anomaly_delta_deg": float(eval_cfg.get("anomaly_delta_deg", 30.0)),
            "active_delta_threshold_deg": float(eval_cfg.get("active_delta_threshold_deg", 1.0)),
            "noise_floor_threshold_deg": float(eval_cfg.get("noise_floor_threshold_deg", 0.5)),
            **info,
        },
        "timeline": {
            "frame_count": int(len(time_s)),
            "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) else 0.0,
            "median_dt_s": float(np.nanmedian(np.diff(time_s))) if len(time_s) > 1 else None,
        },
        "motion_agreement": {},
    }
    reference = "XsensFair"
    for name in angle_names:
        summary["motion_agreement"][name] = {}
        for system in systems:
            if system == reference:
                continue
            pair_key = f"{system}_vs_{reference}"
            summary["motion_agreement"][name][pair_key] = {}
            for k in k_list:
                active = threshold_for_k(float(eval_cfg.get("active_delta_threshold_deg", 1.0)), k, "active")
                noise = threshold_for_k(float(eval_cfg.get("noise_floor_threshold_deg", 0.5)), k, "noise")
                metrics = pair_metrics(deltas[system][name][k], deltas[reference][name][k], active, noise)
                anomaly = np.isfinite(deltas[system][name][k]) & (np.abs(deltas[system][name][k]) > threshold_for_k(float(eval_cfg.get("anomaly_delta_deg", 30.0)), k, "anomaly"))
                metrics["target_high_delta_count"] = int(np.sum(anomaly))
                metrics["active_delta_threshold_deg"] = active
                metrics["noise_floor_threshold_deg"] = noise
                summary["motion_agreement"][name][pair_key][f"k{k}"] = metrics
    (out_dir / "motion_delta_summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[motion] saved {combined_path}")
    return combined_path
