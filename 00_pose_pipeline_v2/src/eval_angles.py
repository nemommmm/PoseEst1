"""Traditional joint-angle agreement evaluation."""

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
from common.metrics import jsonable, mae, median_abs_error, rula_bin
from estimate_offset import load_selected_offset


def prepare_angles(config: dict, run_dir: Path, offset_s: float) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict]:
    """Load all method/reference angle series on the shared timeline."""
    time_s, meta, methods = load_method_keypoints(config, run_dir)
    eval_cfg = section(config, "evaluation")
    refs = section(config, "references")
    angle_names = [name for name in eval_cfg.get("angle_names", list(SEMANTIC_ANGLE_NAMES)) if name in SEMANTIC_ANGLE_NAMES]
    _, radius, actual_ms = odd_window_from_ms(time_s, float(eval_cfg.get("camera_smooth_window_ms", 200.0)))
    max_gap = int(eval_cfg.get("max_gap_frames", 5))

    all_angles: dict[str, dict[str, np.ndarray]] = {}
    for system, keypoints in methods.items():
        raw = compute_angle_sequence(keypoints, angle_names)
        processed = {}
        for name, values in raw.items():
            filled, _ = fill_short_gaps(values, time_s, max_gap)
            smoothed = moving_average(filled, radius)
            # Preserve NaN positions from after fill_short_gaps so that moving_average
            # cannot bleed valid values into quality-filtered (or unfillable) NaN frames.
            smoothed[~np.isfinite(filled)] = np.nan
            processed[name] = smoothed
        all_angles[system] = processed

    fair_path = resolve_path(refs.get("xsens_fair_angles"), must_exist=False)
    fair_interps = build_fair_angle_interpolators(fair_path)
    if not fair_interps:
        fair_interps = build_native_angle_interpolators(resolve_path(refs.get("xsens_mvnx"), must_exist=True))
    query_t = time_s - offset_s
    all_angles["XsensFair"] = sample_interpolators(fair_interps, query_t, angle_names)
    if refs.get("include_xsens_native", True):
        native_interps = build_native_angle_interpolators(resolve_path(refs.get("xsens_mvnx"), must_exist=True))
        all_angles["XsensNative"] = sample_interpolators(native_interps, query_t, angle_names)

    info = {
        "angle_names": angle_names,
        "camera_smooth_radius_frames": radius,
        "camera_smooth_window_actual_ms": actual_ms,
        "trc_summaries": meta.get("trc_summaries", {}),
    }
    return time_s, all_angles, info


def angle_summary_rows(
    all_angles: dict[str, dict[str, np.ndarray]],
    angle_names: list[str],
    rula_bins: dict[str, list[float]],
) -> list[dict[str, object]]:
    """Build per-system/per-angle summary rows against XsensFair."""
    rows = []
    reference = all_angles["XsensFair"]
    for system, angles in all_angles.items():
        if system == "XsensFair":
            continue
        for name in angle_names:
            target = angles.get(name)
            ref = reference.get(name)
            if target is None or ref is None:
                continue
            valid = np.isfinite(target) & np.isfinite(ref)
            bins = rula_bins.get(name)
            agreement = None
            if bins and np.any(valid):
                agreement = float(np.mean(rula_bin(target[valid], bins) == rula_bin(ref[valid], bins)))
            rows.append({
                "system": system,
                "angle": name,
                "valid_pair_count": int(valid.sum()),
                "valid_ratio": float(valid.mean()) if len(valid) else 0.0,
                "mae_deg": mae(target, ref),
                "median_abs_error_deg": median_abs_error(target, ref),
                "bias_deg": float(np.nanmean(target[valid] - ref[valid])) if np.any(valid) else None,
                "rula_like_agreement": agreement,
            })
    return rows


def write_angle_timeseries(path: Path, time_s: np.ndarray, all_angles: dict[str, dict[str, np.ndarray]], angle_names: list[str]) -> None:
    """Write full angle time series CSV."""
    systems = list(all_angles.keys())
    fieldnames = ["Frame", "Time_s"]
    for system in systems:
        for name in angle_names:
            fieldnames.append(f"{system}_{name}_deg")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t in enumerate(time_s):
            row = {"Frame": idx, "Time_s": f"{float(t):.6f}"}
            for system in systems:
                for name in angle_names:
                    value = all_angles[system].get(name, np.full(len(time_s), np.nan))[idx]
                    row[f"{system}_{name}_deg"] = "" if not np.isfinite(value) else f"{float(value):.6f}"
            writer.writerow(row)


def evaluate_angles(config: dict, run_dir: Path) -> Path:
    """Run traditional angle evaluation."""
    offset_s = load_selected_offset(run_dir)
    time_s, all_angles, info = prepare_angles(config, run_dir, offset_s)
    eval_cfg = section(config, "evaluation")
    angle_names = info["angle_names"]
    rows = angle_summary_rows(all_angles, angle_names, eval_cfg.get("rula_bins", {}))

    out_dir = run_dir / "angle_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_angle_timeseries(out_dir / "angle_timeseries.csv", time_s, all_angles, angle_names)
    csv_path = out_dir / "angle_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["system", "angle", "valid_pair_count", "valid_ratio", "mae_deg", "median_abs_error_deg", "bias_deg", "rula_like_agreement"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config": {
            "selected_offset_seconds": offset_s,
            "reference": "XsensFair (Xsens-derived comparison/reference system)",
            **info,
        },
        "rows": rows,
    }
    (out_dir / "angle_summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[angle] saved {csv_path}")
    return csv_path
