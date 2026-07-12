#!/opt/anaconda3/envs/pose/bin/python
"""Refine FastSAM3D TRC time offset against an existing SKT run.

The SKT-to-Xsens offset is handled by estimate_offset.py. This helper only
searches the TRC source_time_offset_s used to place FastSAM3D on the video
timeline, using SKT and FastSAM3D angle trajectories on the same frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.angles import (  # noqa: E402
    SEMANTIC_ANGLE_NAMES,
    fill_short_gaps,
    moving_average,
    odd_window_from_ms,
)
from common.config import get_run_dir, load_config, resolve_path, section  # noqa: E402
from common.dataset import (  # noqa: E402
    apply_depth_consistency_filter,
    apply_skt_quality_filter,
    build_pose_timeline,
    load_skt_keypoints,
)
from common.metrics import jsonable  # noqa: E402
from common.trc import interpolate_keypoints, load_trc, trc_to_coco17, unit_to_cm  # noqa: E402

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline YAML config.")
    parser.add_argument("--run-dir", type=Path, help="Existing run directory. Defaults to config outputs.")
    parser.add_argument(
        "--angles",
        default="RightElbow,RightKnee",
        help="Comma-separated angle names used for scoring. Use 'config' for all configured angles.",
    )
    parser.add_argument("--min-offset", type=float, default=-6.0, help="Minimum FastSAM3D source offset in seconds.")
    parser.add_argument("--max-offset", type=float, default=2.0, help="Maximum FastSAM3D source offset in seconds.")
    parser.add_argument("--step", type=float, default=0.01, help="Search step in seconds.")
    parser.add_argument("--min-pairs", type=int, default=40, help="Minimum finite pairs per angle.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for CSV/PNG/JSON. Defaults to <run-dir>/fastsam_offset_refine.",
    )
    return parser.parse_args()


def parse_angle_names(raw: str, config: dict) -> list[str]:
    """Resolve selected angle names."""
    if raw.strip().lower() == "config":
        candidates = section(config, "evaluation").get("angle_names", list(SEMANTIC_ANGLE_NAMES))
    else:
        candidates = [item.strip() for item in raw.split(",") if item.strip()]
    return [name for name in candidates if name in SEMANTIC_ANGLE_NAMES]


def process_angles(raw_angles: dict[str, np.ndarray], time_s: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    """Apply the same gap-fill and moving-average chain used in angle evaluation."""
    eval_cfg = section(config, "evaluation")
    _, radius, _ = odd_window_from_ms(time_s, float(eval_cfg.get("camera_smooth_window_ms", 200.0)))
    max_gap = int(eval_cfg.get("max_gap_frames", 5))
    out: dict[str, np.ndarray] = {}
    for name, values in raw_angles.items():
        filled, _ = fill_short_gaps(values, time_s, max_gap)
        smoothed = moving_average(filled, radius)
        smoothed[~np.isfinite(filled)] = np.nan
        out[name] = smoothed
    return out


def _angle_between_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """Vectorized angle between two frame-wise 3D vector series."""
    out = np.full(len(vec_a), np.nan, dtype=np.float64)
    finite = np.isfinite(vec_a).all(axis=1) & np.isfinite(vec_b).all(axis=1)
    norm_a = np.linalg.norm(vec_a, axis=1)
    norm_b = np.linalg.norm(vec_b, axis=1)
    finite &= (norm_a > 1e-8) & (norm_b > 1e-8)
    if not np.any(finite):
        return out
    dot = np.sum(vec_a[finite] * vec_b[finite], axis=1) / (norm_a[finite] * norm_b[finite])
    out[finite] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return out


def _interior_flexion(keypoints: np.ndarray, p1: int, p2: int, p3: int) -> np.ndarray:
    """Vectorized flexion-style angle, matching common.angles 180 - interior."""
    interior = _angle_between_vectors(keypoints[:, p1] - keypoints[:, p2], keypoints[:, p3] - keypoints[:, p2])
    out = np.full_like(interior, np.nan)
    finite = np.isfinite(interior)
    out[finite] = 180.0 - interior[finite]
    return out


def compute_selected_angles(keypoints: np.ndarray, angle_names: list[str]) -> dict[str, np.ndarray]:
    """Compute selected ergonomic angle series with vectorized NumPy operations."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    out = {name: np.full(len(keypoints), np.nan, dtype=np.float64) for name in angle_names}
    hip_mid = 0.5 * (keypoints[:, LEFT_HIP] + keypoints[:, RIGHT_HIP])
    shoulder_mid = 0.5 * (keypoints[:, LEFT_SHOULDER] + keypoints[:, RIGHT_SHOULDER])
    torso_down = hip_mid - shoulder_mid
    if "LeftShoulder" in out:
        out["LeftShoulder"] = _angle_between_vectors(keypoints[:, LEFT_ELBOW] - keypoints[:, LEFT_SHOULDER], torso_down)
    if "RightShoulder" in out:
        out["RightShoulder"] = _angle_between_vectors(keypoints[:, RIGHT_ELBOW] - keypoints[:, RIGHT_SHOULDER], torso_down)
    if "LeftElbow" in out:
        out["LeftElbow"] = _interior_flexion(keypoints, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    if "RightElbow" in out:
        out["RightElbow"] = _interior_flexion(keypoints, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    if "LeftHip" in out:
        out["LeftHip"] = _interior_flexion(keypoints, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    if "RightHip" in out:
        out["RightHip"] = _interior_flexion(keypoints, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
    if "LeftKnee" in out:
        out["LeftKnee"] = _interior_flexion(keypoints, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    if "RightKnee" in out:
        out["RightKnee"] = _interior_flexion(keypoints, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    return out


def load_processed_skt_angles(config: dict, run_dir: Path, angle_names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load SKT keypoints and compute processed angle trajectories."""
    _, keypoints, payload = load_skt_keypoints(config, run_dir)
    time_s, _, _, _ = build_pose_timeline(config, len(keypoints))
    keypoints = keypoints[: len(time_s)]
    keypoints, _ = apply_skt_quality_filter(keypoints, payload, config)
    keypoints, _ = apply_depth_consistency_filter(keypoints, config)
    raw = compute_selected_angles(keypoints, angle_names)
    return time_s, process_angles(raw, time_s, config)


def load_trc_keypoints(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load configured FastSAM3D TRC keypoints in COCO-17 order and centimeters."""
    refs = section(config, "references")
    trc_path = resolve_path(refs.get("fastsam_trc"), must_exist=True)
    timestamps, marker_names, positions, _, units = load_trc(trc_path)
    keypoints, missing = trc_to_coco17(marker_names, positions * unit_to_cm(units))
    if missing:
        raise RuntimeError(f"FastSAM3D TRC is missing COCO-17 joints: {missing}")
    return timestamps, keypoints


def finite_metrics(target: np.ndarray, reference: np.ndarray, min_pairs: int) -> dict[str, float | int | None]:
    """Compute finite-pair agreement metrics."""
    valid = np.isfinite(target) & np.isfinite(reference)
    count = int(valid.sum())
    if count < min_pairs:
        return {"pairs": count, "mae": None, "rmse": None, "pearson": None}
    diff = target[valid] - reference[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(np.mean(diff * diff)))
    if np.std(target[valid]) < 1e-9 or np.std(reference[valid]) < 1e-9:
        pearson = None
    else:
        pearson = float(np.corrcoef(target[valid], reference[valid])[0, 1])
    return {"pairs": count, "mae": mae, "rmse": rmse, "pearson": pearson}


def score_offset(
    offset_s: float,
    time_s: np.ndarray,
    skt_angles: dict[str, np.ndarray],
    trc_time: np.ndarray,
    trc_keypoints: np.ndarray,
    angle_names: list[str],
    config: dict,
    min_pairs: int,
) -> dict[str, object]:
    """Score one FastSAM3D source offset."""
    fast_kp = interpolate_keypoints(trc_time, trc_keypoints, time_s, source_time_offset_s=offset_s)
    fast_angles = process_angles(compute_selected_angles(fast_kp, angle_names), time_s, config)
    rows = []
    for angle_name in angle_names:
        metrics = finite_metrics(skt_angles[angle_name], fast_angles[angle_name], min_pairs)
        rows.append({"angle": angle_name, **metrics})
    valid_mae = [float(row["mae"]) for row in rows if row["mae"] is not None]
    valid_rmse = [float(row["rmse"]) for row in rows if row["rmse"] is not None]
    valid_corr = [float(row["pearson"]) for row in rows if row["pearson"] is not None]
    return {
        "offset_s": float(offset_s),
        "median_mae": float(np.median(valid_mae)) if valid_mae else None,
        "median_rmse": float(np.median(valid_rmse)) if valid_rmse else None,
        "median_pearson": float(np.median(valid_corr)) if valid_corr else None,
        "angle_count": len(valid_mae),
        "total_pairs": int(sum(int(row["pairs"]) for row in rows)),
        "per_angle": rows,
    }


def write_outputs(out_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    """Write score curve CSV, summary JSON, and diagnostic plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fastsam_offset_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["offset_s", "median_mae", "median_rmse", "median_pearson", "angle_count", "total_pairs"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    (out_dir / "fastsam_offset_summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")

    offsets = np.asarray([float(row["offset_s"]) for row in rows], dtype=np.float64)
    mae = np.asarray([np.nan if row["median_mae"] is None else float(row["median_mae"]) for row in rows])
    corr = np.asarray([np.nan if row["median_pearson"] is None else float(row["median_pearson"]) for row in rows])
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(offsets, mae, color="#1f77b4", label="Median MAE")
    ax1.set_xlabel("FastSAM3D source offset (s)")
    ax1.set_ylabel("Median MAE (deg)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(offsets, corr, color="#ff7f0e", alpha=0.75, label="Median Pearson")
    ax2.set_ylabel("Median Pearson", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    best = summary.get("best_by_mae", {})
    if best and best.get("offset_s") is not None:
        ax1.axvline(float(best["offset_s"]), color="black", linestyle="--", linewidth=1)
    fig.suptitle("FastSAM3D offset refinement against SKT")
    fig.tight_layout()
    fig.savefig(out_dir / "fastsam_offset_scores.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run FastSAM3D offset refinement."""
    args = parse_args()
    config = load_config(args.config)
    run_dir = args.run_dir or get_run_dir(config)
    out_dir = args.out_dir or (run_dir / "fastsam_offset_refine")
    angle_names = parse_angle_names(args.angles, config)
    if not angle_names:
        raise ValueError("No valid angle names selected.")
    if args.step <= 0:
        raise ValueError("--step must be positive.")

    time_s, skt_angles = load_processed_skt_angles(config, run_dir, angle_names)
    trc_time, trc_keypoints = load_trc_keypoints(config)
    offsets = np.round(np.arange(args.min_offset, args.max_offset + 0.5 * args.step, args.step), 6)
    rows = [
        score_offset(
            offset_s=float(offset),
            time_s=time_s,
            skt_angles=skt_angles,
            trc_time=trc_time,
            trc_keypoints=trc_keypoints,
            angle_names=angle_names,
            config=config,
            min_pairs=int(args.min_pairs),
        )
        for offset in offsets
    ]
    valid_by_mae = [row for row in rows if row["median_mae"] is not None]
    best_by_mae = min(valid_by_mae, key=lambda row: float(row["median_mae"])) if valid_by_mae else None
    valid_by_corr = [row for row in rows if row["median_pearson"] is not None]
    best_by_corr = max(valid_by_corr, key=lambda row: float(row["median_pearson"])) if valid_by_corr else None
    current = section(config, "references").get("trc_time_offsets_seconds", {}).get("FastSAM3D")
    summary = {
        "config": str(resolve_path(args.config, must_exist=False) or args.config),
        "run_dir": str(run_dir),
        "angles": angle_names,
        "search_range_seconds": [float(args.min_offset), float(args.max_offset)],
        "search_step_seconds": float(args.step),
        "current_config_offset_seconds": current,
        "best_by_mae": best_by_mae,
        "best_by_pearson": best_by_corr,
        "note": "This refines FastSAM3D TRC-to-video timing only; SKT-to-Xsens alignment remains handled separately.",
    }
    write_outputs(out_dir, rows, summary)
    print(f"[fastsam-offset] current={current}, best_mae={None if best_by_mae is None else best_by_mae['offset_s']}, best_corr={None if best_by_corr is None else best_by_corr['offset_s']}")
    print(f"[fastsam-offset] saved {out_dir}")


if __name__ == "__main__":
    main()
