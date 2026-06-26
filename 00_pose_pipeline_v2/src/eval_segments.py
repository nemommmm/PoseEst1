"""Activity-segment ROM, DTW, and RULA-like evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common.config import section
from common.metrics import jsonable, mae, rula_bin


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Load a motion combined CSV into typed arrays."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    data = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key.endswith("_valid") or "_valid_k" in key:
            data[key] = np.asarray([value == "True" for value in values], dtype=bool)
        elif key == "Frame":
            data[key] = np.asarray([int(value) for value in values], dtype=np.int64)
        else:
            data[key] = np.asarray([float(value) if value else np.nan for value in values], dtype=np.float64)
    return data


def infer_systems(data: dict[str, np.ndarray], angle_name: str) -> list[str]:
    """Infer systems from angle columns."""
    suffix = f"_{angle_name}_deg"
    systems = [key[: -len(suffix)] for key in data if key.endswith(suffix)]
    preferred = ["SKT", "FastSAM3D", "Merge", "XsensFair", "XsensNative"]
    order = {name: idx for idx, name in enumerate(preferred)}
    return sorted(systems, key=lambda item: (order.get(item, len(order)), item))


def detect_segments(time_s: np.ndarray, activity: np.ndarray, min_duration_s: float, merge_gap_s: float) -> list[tuple[int, int]]:
    """Detect contiguous activity segments and merge short gaps."""
    segments = []
    idx = 0
    while idx < len(activity):
        if not activity[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(activity) and activity[idx]:
            idx += 1
        end = idx - 1
        if time_s[end] - time_s[start] >= min_duration_s:
            segments.append((start, end))
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if time_s[start] - time_s[prev_end] <= merge_gap_s:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def prepare_dtw(values: np.ndarray, preprocess: str) -> np.ndarray | None:
    """Prepare one segment for DTW."""
    seq = np.asarray(values, dtype=np.float64)
    seq = seq[np.isfinite(seq)]
    if len(seq) < 3:
        return None
    if preprocess in {"mean", "mean_l2"}:
        seq = seq - float(np.mean(seq))
    if preprocess == "mean_l2":
        norm = float(np.linalg.norm(seq))
        if norm < 1e-9:
            return None
        seq = seq / norm
    return seq


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """Compute simple absolute-cost DTW distance."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return None
    cost = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            step = abs(float(a[i - 1] - b[j - 1]))
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[-1, -1] / max(len(a) + len(b), 1))


def evaluate_segments(config: dict, run_dir: Path) -> Path:
    """Run segment-level ROM/DTW/RULA-like evaluation."""
    motion_path = run_dir / "motion_delta" / "motion_delta_combined.csv"
    if not motion_path.exists():
        raise FileNotFoundError(f"Run motion stage first; missing {motion_path}")
    data = load_csv(motion_path)
    seg_cfg = section(config, "segment")
    eval_cfg = section(config, "evaluation")
    time_s = data["Time_s"]
    angle_names = [name for name in eval_cfg.get("angle_names", ["LeftElbow", "RightElbow"]) if f"XsensFair_{name}_deg" in data]
    k = int(seg_cfg.get("activity_lag_frames", 12))
    threshold = float(seg_cfg.get("activity_threshold_deg", 10.0))
    out_dir = run_dir / "segment_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seg_id = 0
    for angle_name in angle_names:
        ref_delta_key = f"XsensFair_{angle_name}_delta_k{k}_deg"
        if ref_delta_key not in data:
            continue
        activity = np.isfinite(data[ref_delta_key]) & (np.abs(data[ref_delta_key]) >= threshold)
        segments = detect_segments(time_s, activity, float(seg_cfg.get("min_duration_s", 1.5)), float(seg_cfg.get("merge_gap_s", 2.0)))
        systems = infer_systems(data, angle_name)
        for start, end in segments:
            ref_values = data[f"XsensFair_{angle_name}_deg"][start : end + 1]
            ref_finite = ref_values[np.isfinite(ref_values)]
            if len(ref_finite) < 3:
                continue
            ref_rom = float(np.nanmax(ref_finite) - np.nanmin(ref_finite))
            if ref_rom < float(seg_cfg.get("min_reference_rom_deg", 15.0)):
                continue
            ref_dtw = prepare_dtw(ref_values, seg_cfg.get("dtw_preprocess", "mean_l2"))
            bins = eval_cfg.get("rula_bins", {}).get(angle_name)
            for system in systems:
                if system == "XsensFair":
                    continue
                key = f"{system}_{angle_name}_deg"
                if key not in data:
                    continue
                target = data[key][start : end + 1]
                finite = target[np.isfinite(target)]
                valid_ratio = float(len(finite) / max(len(target), 1))
                if valid_ratio < float(seg_cfg.get("min_valid_ratio", 0.5)) or len(finite) < 3:
                    continue
                target_rom = float(np.nanmax(finite) - np.nanmin(finite))
                agreement = None
                if bins:
                    mask = np.isfinite(target) & np.isfinite(ref_values)
                    agreement = float(np.mean(rula_bin(target[mask], bins) == rula_bin(ref_values[mask], bins))) if np.any(mask) else None
                rows.append({
                    "segment_id": seg_id,
                    "angle": angle_name,
                    "system": system,
                    "start_frame": start,
                    "end_frame": end,
                    "start_time_s": float(time_s[start]),
                    "end_time_s": float(time_s[end]),
                    "duration_s": float(time_s[end] - time_s[start]),
                    "valid_ratio": valid_ratio,
                    "target_rom_deg": target_rom,
                    "reference_rom_deg": ref_rom,
                    "rom_abs_error_deg": abs(target_rom - ref_rom),
                    "angle_mae_deg": mae(target, ref_values),
                    "dtw_distance": dtw_distance(prepare_dtw(target, seg_cfg.get("dtw_preprocess", "mean_l2")), ref_dtw),
                    "rula_like_agreement": agreement,
                })
            seg_id += 1

    csv_path = out_dir / "segment_summary.csv"
    fieldnames = [
        "segment_id", "angle", "system", "start_frame", "end_frame", "start_time_s", "end_time_s",
        "duration_s", "valid_ratio", "target_rom_deg", "reference_rom_deg", "rom_abs_error_deg",
        "angle_mae_deg", "dtw_distance", "rula_like_agreement",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "segment_summary.json").write_text(json.dumps(jsonable({"rows": rows}), indent=2), encoding="utf-8")
    print(f"[segment] saved {csv_path}")
    return csv_path
