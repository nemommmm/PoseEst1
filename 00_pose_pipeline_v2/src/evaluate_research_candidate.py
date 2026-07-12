"""Evaluate a canonical research candidate against the project comparison system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.angles import (
    build_fair_angle_interpolators,
    build_native_angle_interpolators,
    compute_angle_sequence,
    fill_short_gaps,
    moving_average,
    odd_window_from_ms,
    sample_interpolators,
)
from common.config import load_config, resolve_path, section
from common.metrics import jsonable, mae, median_abs_error, rmse, rula_bin


def process_angles(values: np.ndarray, timestamps: np.ndarray, config: dict) -> np.ndarray:
    """Apply the same gap fill and temporal smoothing used by the main evaluator."""
    eval_cfg = section(config, "evaluation")
    _, radius, _ = odd_window_from_ms(timestamps, float(eval_cfg.get("camera_smooth_window_ms", 200.0)))
    processed = np.full_like(values, np.nan, dtype=np.float64)
    for angle_idx in range(values.shape[1]):
        filled, _ = fill_short_gaps(values[:, angle_idx], timestamps, int(eval_cfg.get("max_gap_frames", 5)))
        processed[:, angle_idx] = moving_average(filled, radius)
        processed[~np.isfinite(filled), angle_idx] = np.nan
    return processed


def temporal_metrics(values: np.ndarray, timestamps: np.ndarray) -> dict[str, float | int | None]:
    """Summarize continuity without using the external comparison system."""
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return {"jump_count_10deg": 0, "angular_acceleration_rms_deg_s2": None}
    series = np.interp(np.arange(len(values)), np.flatnonzero(finite), values[finite])
    jumps = int(np.count_nonzero(np.abs(np.diff(series)) > 10.0))
    dt = float(np.median(np.diff(timestamps)))
    acceleration = np.diff(series, n=2) / max(dt**2, 1e-12)
    return {
        "jump_count_10deg": jumps,
        "angular_acceleration_rms_deg_s2": float(np.sqrt(np.mean(acceleration**2))) if acceleration.size else None,
    }


def evaluate(candidate_path: Path, config_path: Path, offset_s: float) -> dict:
    """Return baseline and candidate agreement metrics on identical frames."""
    config = load_config(config_path)
    eval_cfg = section(config, "evaluation")
    refs = section(config, "references")
    with np.load(candidate_path, allow_pickle=False) as payload:
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        candidate_names = [str(name) for name in payload["angle_names"]]
        candidate_raw = np.asarray(payload["angles"], dtype=np.float64)
        keypoints = np.asarray(payload["keypoints_3d"], dtype=np.float64)
        metadata = json.loads(str(payload["metadata_json"]))
        candidate_name = str(payload["candidate_name"])
        time_keys = [key for key in payload.files if key.startswith("time_")]
        timing = {key: np.asarray(payload[key]).tolist() for key in time_keys}
    angle_names = [name for name in eval_cfg.get("angle_names", candidate_names) if name in candidate_names]
    indices = [candidate_names.index(name) for name in angle_names]
    candidate = process_angles(candidate_raw[:, indices], timestamps, config)
    baseline_dict = compute_angle_sequence(keypoints, angle_names)
    baseline_raw = np.column_stack([baseline_dict[name] for name in angle_names])
    baseline = process_angles(baseline_raw, timestamps, config)

    fair_path = resolve_path(refs.get("xsens_fair_angles"), must_exist=False)
    reference_interps = build_fair_angle_interpolators(fair_path)
    reference_kind = "Xsens-derived geometric reference"
    if not reference_interps:
        reference_interps = build_native_angle_interpolators(resolve_path(refs.get("xsens_mvnx"), must_exist=True))
        reference_kind = "Xsens native comparison signal"
    reference_dict = sample_interpolators(reference_interps, timestamps - offset_s, angle_names)
    reference = np.column_stack([reference_dict[name] for name in angle_names])

    rows = []
    for angle_idx, angle_name in enumerate(angle_names):
        ref = reference[:, angle_idx]
        bins = eval_cfg.get("rula_bins", {}).get(angle_name)
        for system_name, values in (("SKT-geometric", baseline[:, angle_idx]), (candidate_name, candidate[:, angle_idx])):
            valid = np.isfinite(values) & np.isfinite(ref)
            agreement = None
            if bins and valid.any():
                agreement = float(np.mean(rula_bin(values[valid], bins) == rula_bin(ref[valid], bins)))
            rows.append(
                {
                    "system": system_name,
                    "angle": angle_name,
                    "valid_pair_count": int(valid.sum()),
                    "valid_ratio": float(valid.mean()),
                    "mae_deg": mae(values, ref),
                    "median_abs_error_deg": median_abs_error(values, ref),
                    "rmse_deg": rmse(values, ref),
                    "bias_deg": float(np.mean(values[valid] - ref[valid])) if valid.any() else None,
                    "rula_like_agreement": agreement,
                    **temporal_metrics(values, timestamps),
                }
            )
    return {
        "candidate": candidate_name,
        "candidate_path": str(candidate_path),
        "config_path": str(config_path),
        "selected_offset_seconds": offset_s,
        "reference": f"{reference_kind}; external comparison only, not absolute ground truth",
        "frames": len(timestamps),
        "timing": timing,
        "metadata": metadata,
        "rows": rows,
    }


def main() -> None:
    """Evaluate and optionally persist one compact JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offset-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = evaluate(args.candidate, args.config, args.offset_seconds)
    text = json.dumps(jsonable(summary), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
