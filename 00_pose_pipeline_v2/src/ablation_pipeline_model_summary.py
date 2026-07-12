"""Aggregate 2x2 ablation matrix (pipeline x model) into a summary table.

Reads MAE / RMS / valid_ratio from filter_ablation/summary.csv (v2 runs) or
angle_eval/angle_summary.csv (v1 fallback), plus per-frame YOLO timing from
skt_pose_optimized.npz when available.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Cell definitions: (pipeline_version, model, dataset) -> (eval_dir, npz_dir)
# eval_dir holds filter_ablation/summary.csv; npz_dir holds skt_pose_optimized.npz with timing.
ABL_ROOT = REPO_ROOT / "00_pose_pipeline_v2" / "runs" / "ablation_pipeline_model"
CELLS = {
    ("V1", "YOLOv8m", "fanbo7"): (
        ABL_ROOT / "fanbo7_v1_yolov8m_eval",
        REPO_ROOT / "00_pose_pipeline" / "runs" / "assar2026_fanbo7_a257_elbow_test",
    ),
    ("V1", "YOLOv8m", "fanbo4"): (
        ABL_ROOT / "fanbo4_v1_yolov8m_eval",
        REPO_ROOT / "00_pose_pipeline" / "runs" / "assar2026_fanbo4_a257_elbow_test",
    ),
    ("V1", "YOLO11l", "fanbo7"): (ABL_ROOT / "fanbo7_v1_yolo11l", ABL_ROOT / "fanbo7_v1_yolo11l"),
    ("V1", "YOLO11l", "fanbo4"): (ABL_ROOT / "fanbo4_v1_yolo11l", ABL_ROOT / "fanbo4_v1_yolo11l"),
    ("V2", "YOLOv8m", "fanbo7"): (ABL_ROOT / "fanbo7_v2_yolov8m", ABL_ROOT / "fanbo7_v2_yolov8m"),
    ("V2", "YOLOv8m", "fanbo4"): (ABL_ROOT / "fanbo4_v2_yolov8m", ABL_ROOT / "fanbo4_v2_yolov8m"),
    ("V2", "YOLO11l", "fanbo7"): (
        REPO_ROOT / "00_pose_pipeline_v2" / "runs" / "assar2026_fanbo7_a257_stage1_geometry",
        REPO_ROOT / "00_pose_pipeline_v2" / "runs" / "assar2026_fanbo7_a257_stage1_geometry",
    ),
    ("V2", "YOLO11l", "fanbo4"): (
        ABL_ROOT / "fanbo4_v2_yolo11l_eval",
        REPO_ROOT / "00_pose_pipeline_v2" / "runs" / "assar2026_fanbo4_a257_stage1_geometry",
    ),
}

# Prefer these variant labels when reading filter_ablation/summary.csv (raw SKT angle, no smoothing).
RAW_VARIANT_PREFERENCE = ["no_filter_raw_angle", "hard_filter_raw_angle"]
POST_VARIANT_PREFERENCE = ["quality_aware_repair_keypoint_savgol", "hard_filter_keypoint_savgol"]


def _read_filter_ablation(run_dir: Path, variant_prefs: list[str]) -> Optional[dict]:
    csv_path = run_dir / "filter_ablation" / "summary.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    by_target = {row["target"]: row for row in rows if row.get("reference") == "FastSAM3D"}
    for pref in variant_prefs:
        if pref in by_target:
            return by_target[pref]
    return None


def _read_angle_summary(run_dir: Path) -> Optional[dict]:
    csv_path = run_dir / "angle_eval" / "angle_summary.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    for row in rows:
        if row.get("system") == "SKT":
            return row
    return None


def _read_timing_from_npz(run_dir: Path) -> tuple[Optional[float], Optional[float], Optional[int]]:
    npz_path = run_dir / "skt_pose_optimized.npz"
    if not npz_path.exists():
        return None, None, None
    try:
        data = np.load(npz_path)
        n_frames = int(data["timestamps"].shape[0]) if "timestamps" in data.files else None
        yolo_ms: Optional[float] = None
        frame_ms: Optional[float] = None
        if "yolo_time_ms" in data.files:
            arr = data["yolo_time_ms"]
            if arr.size > 1:
                yolo_ms = float(np.mean(arr[1:]))  # exclude first frame (model load)
        if "frame_time_ms" in data.files:
            arr = data["frame_time_ms"]
            if arr.size > 1:
                frame_ms = float(np.mean(arr[1:]))
        return yolo_ms, frame_ms, n_frames
    except Exception:
        return None, None, None


def _read_wallclock(run_dir: Path) -> Optional[float]:
    log_path = run_dir.parent / f"{run_dir.name}_stdout.log"
    if not log_path.exists():
        return None
    try:
        tail = log_path.read_text().strip().splitlines()
        for line in reversed(tail[-10:]):
            line = line.strip()
            if line and line.replace(".", "").isdigit():
                return float(line)
    except Exception:
        return None
    return None


def _fmt_float(x, digits=2, default="--"):
    if x is None:
        return default
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return default


def summarize() -> None:
    print(f"{'Pipeline':<8} {'Model':<10} {'Dataset':<8} {'Variant':<32} {'MAE':>6} {'RMS':>7} {'Valid':>7} {'YOLOms':>8} {'Frame':<8} {'Wall(s)':>8}")
    print("-" * 120)
    out_rows = []
    for (pipeline, model, dataset), (eval_dir, npz_dir) in CELLS.items():
        run_dir = eval_dir  # for filter_ablation / angle_summary reads
        raw = _read_filter_ablation(run_dir, RAW_VARIANT_PREFERENCE)
        post = _read_filter_ablation(run_dir, POST_VARIANT_PREFERENCE)
        angle_row = _read_angle_summary(run_dir) if raw is None else None

        # Choose the "raw SKT" row for the primary comparison
        source = "filter_ablation.no_filter_raw_angle"
        if raw is not None:
            mae = float(raw["mae_deg"])
            rms = float(raw["target_angular_acc_rms_deg_s2"])
            valid = float(raw["valid_ratio"])
        elif angle_row is not None:
            source = "angle_summary.SKT"
            mae = float(angle_row["mae_deg"])
            rms = None
            valid = float(angle_row["valid_ratio"])
        else:
            source = "MISSING"
            mae = rms = valid = None

        yolo_ms, frame_ms, n_frames = _read_timing_from_npz(npz_dir)
        wall_s = _read_wallclock(npz_dir)

        print(
            f"{pipeline:<8} {model:<10} {dataset:<8} {source:<32} "
            f"{_fmt_float(mae):>6} {_fmt_float(rms, 0):>7} {_fmt_float(valid, 3):>7} "
            f"{_fmt_float(yolo_ms, 1):>8} {_fmt_float(frame_ms, 1):<8} {_fmt_float(wall_s, 1):>8}"
        )

        # Also record post-processed if available
        if post is not None:
            print(
                f"{'':<8} {'':<10} {'':<8} {'  +postproc(bone→savgol)':<32} "
                f"{_fmt_float(float(post['mae_deg'])):>6} {_fmt_float(float(post['target_angular_acc_rms_deg_s2']), 0):>7} "
                f"{_fmt_float(float(post['valid_ratio']), 3):>7} {'':>8} {'':<8} {'':>8}"
            )

        out_rows.append({
            "pipeline": pipeline, "model": model, "dataset": dataset,
            "eval_dir": str(eval_dir), "npz_dir": str(npz_dir), "metric_source": source,
            "mae_deg": mae, "angular_acc_rms": rms, "valid_ratio": valid,
            "yolo_ms_per_frame": yolo_ms, "frame_ms_per_frame": frame_ms,
            "wallclock_s": wall_s, "n_frames": n_frames,
            "postproc_mae_deg": float(post["mae_deg"]) if post else None,
            "postproc_rms": float(post["target_angular_acc_rms_deg_s2"]) if post else None,
        })

    out_dir = REPO_ROOT / "00_pose_pipeline_v2" / "runs" / "ablation_pipeline_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_2x2_summary.json").write_text(json.dumps(out_rows, indent=2))
    print(f"\n[saved] {out_dir / 'ablation_2x2_summary.json'}")


if __name__ == "__main__":
    summarize()
