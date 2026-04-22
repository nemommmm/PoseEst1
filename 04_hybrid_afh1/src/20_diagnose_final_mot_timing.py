#!/opt/anaconda3/envs/pose/bin/python
"""Diagnose timing drift for the EasyErgo final OpenSim MOT output."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AFH1_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from opensim_mot_utils import (  # noqa: E402
    build_semantic_angles_from_mot,
    load_opensim_mot,
    resolve_easyergo_final_outputs,
)
from pose_angle_utils import build_gt_angle_interpolators  # noqa: E402
from utils_mvnx import MvnxParser  # noqa: E402


INPUT_DIR = Path(
    os.environ.get(
        "POSE_EASYERGO_INPUT_DIR",
        str(AFH1_DIR / "data" / "easyergo_uploaded"),
    )
).resolve()
RESULTS_DIR = Path(
    os.environ.get(
        "POSE_RESULTS_DIR",
        str(AFH1_DIR / "results" / "20_final_mot_timing"),
    )
).resolve()
GT_MVNX_PATH = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"
WINDOW_SCAN_CSV = RESULTS_DIR / "window_offset_scan.csv"
SUMMARY_MD = RESULTS_DIR / "timing_diagnosis_summary.md"
AFFINE_JSON = RESULTS_DIR / "affine_fit.json"

WINDOWS_SEC = [
    (20.0, 50.0),
    (50.0, 80.0),
    (80.0, 110.0),
    (110.0, 140.0),
    (140.0, 170.0),
    (170.0, 200.0),
]
ANGLE_NAMES = [
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftHip",
    "RightHip",
    "LeftKnee",
    "RightKnee",
]
OFFSET_GRID = np.arange(12.0, 21.01, 0.02)
TIME_SCALE_COARSE = np.arange(1.000, 1.0211, 0.001)
OFFSET_GRID_COARSE = np.arange(12.0, 21.01, 0.10)
TIME_SCALE_FINE_RADIUS = 0.0015
TIME_SCALE_FINE_STEP = 0.0001
OFFSET_FINE_RADIUS = 0.35
OFFSET_FINE_STEP = 0.01
FIXED_OFFSET_OLD = 17.25
FIXED_OFFSET_ALT = 15.90


def evaluate_mapping(
    est_ts: np.ndarray,
    est_angles: dict[str, np.ndarray],
    gt_interp: dict[str, object],
    *,
    offset_s: float,
    time_scale: float,
    frame_mask: np.ndarray | None = None,
) -> tuple[float, int]:
    """Return mean absolute error and sample count for one time mapping."""
    if frame_mask is None:
        frame_mask = np.ones_like(est_ts, dtype=bool)

    target_ts = time_scale * est_ts - offset_s
    error_blocks: list[np.ndarray] = []
    for angle_name in ANGLE_NAMES:
        interp = gt_interp.get(angle_name)
        if interp is None:
            continue
        est_vals = np.asarray(est_angles[angle_name], dtype=np.float64)
        gt_vals = np.asarray(interp(target_ts), dtype=np.float64)
        valid = frame_mask & np.isfinite(est_vals) & np.isfinite(gt_vals)
        if not np.any(valid):
            continue
        error_blocks.append(np.abs(est_vals[valid] - gt_vals[valid]))

    if not error_blocks:
        return float("nan"), 0
    errors = np.concatenate(error_blocks)
    return float(np.mean(errors)), int(errors.size)


def scan_window_offsets(
    est_ts: np.ndarray,
    est_angles: dict[str, np.ndarray],
    gt_interp: dict[str, object],
) -> pd.DataFrame:
    """Scan the best constant offset in several local windows."""
    records: list[dict[str, float]] = []
    for start_s, end_s in WINDOWS_SEC:
        frame_mask = (est_ts >= start_s) & (est_ts <= end_s)
        best_offset = np.nan
        best_mae = float("inf")
        best_samples = 0
        for offset_s in OFFSET_GRID:
            mae, samples = evaluate_mapping(
                est_ts,
                est_angles,
                gt_interp,
                offset_s=float(offset_s),
                time_scale=1.0,
                frame_mask=frame_mask,
            )
            if samples <= 0 or not np.isfinite(mae):
                continue
            if mae < best_mae:
                best_offset = float(offset_s)
                best_mae = float(mae)
                best_samples = int(samples)
        records.append(
            {
                "window_start_s": float(start_s),
                "window_end_s": float(end_s),
                "window_mid_s": float(0.5 * (start_s + end_s)),
                "best_offset_s": float(best_offset),
                "semantic_mae_deg": float(best_mae),
                "samples": int(best_samples),
            }
        )
    return pd.DataFrame.from_records(records)


def fit_affine_mapping(
    est_ts: np.ndarray,
    est_angles: dict[str, np.ndarray],
    gt_interp: dict[str, object],
) -> dict[str, float]:
    """Fit gt_t = time_scale * est_t - offset by a small grid search."""
    coarse_best = {
        "time_scale": np.nan,
        "offset_s": np.nan,
        "semantic_mae_deg": float("inf"),
        "samples": 0,
    }
    for time_scale in TIME_SCALE_COARSE:
        for offset_s in OFFSET_GRID_COARSE:
            mae, samples = evaluate_mapping(
                est_ts,
                est_angles,
                gt_interp,
                offset_s=float(offset_s),
                time_scale=float(time_scale),
            )
            if samples <= 0 or not np.isfinite(mae):
                continue
            if mae < coarse_best["semantic_mae_deg"]:
                coarse_best = {
                    "time_scale": float(time_scale),
                    "offset_s": float(offset_s),
                    "semantic_mae_deg": float(mae),
                    "samples": int(samples),
                }

    fine_best = coarse_best.copy()
    fine_time_scales = np.arange(
        coarse_best["time_scale"] - TIME_SCALE_FINE_RADIUS,
        coarse_best["time_scale"] + TIME_SCALE_FINE_RADIUS + 0.5 * TIME_SCALE_FINE_STEP,
        TIME_SCALE_FINE_STEP,
    )
    fine_offsets = np.arange(
        coarse_best["offset_s"] - OFFSET_FINE_RADIUS,
        coarse_best["offset_s"] + OFFSET_FINE_RADIUS + 0.5 * OFFSET_FINE_STEP,
        OFFSET_FINE_STEP,
    )
    for time_scale in fine_time_scales:
        for offset_s in fine_offsets:
            mae, samples = evaluate_mapping(
                est_ts,
                est_angles,
                gt_interp,
                offset_s=float(offset_s),
                time_scale=float(time_scale),
            )
            if samples <= 0 or not np.isfinite(mae):
                continue
            if mae < fine_best["semantic_mae_deg"]:
                fine_best = {
                    "time_scale": float(time_scale),
                    "offset_s": float(offset_s),
                    "semantic_mae_deg": float(mae),
                    "samples": int(samples),
                }

    fixed_old_mae, fixed_old_samples = evaluate_mapping(
        est_ts,
        est_angles,
        gt_interp,
        offset_s=FIXED_OFFSET_OLD,
        time_scale=1.0,
    )
    fixed_alt_mae, fixed_alt_samples = evaluate_mapping(
        est_ts,
        est_angles,
        gt_interp,
        offset_s=FIXED_OFFSET_ALT,
        time_scale=1.0,
    )

    return {
        **fine_best,
        "fixed_17p25_semantic_mae_deg": float(fixed_old_mae),
        "fixed_17p25_samples": int(fixed_old_samples),
        "fixed_15p9_semantic_mae_deg": float(fixed_alt_mae),
        "fixed_15p9_samples": int(fixed_alt_samples),
    }


def build_summary_markdown(
    mot_path: Path,
    est_duration_s: float,
    gt_duration_s: float,
    window_df: pd.DataFrame,
    affine_payload: dict[str, float],
) -> str:
    """Render a short human-readable timing summary."""
    lines = [
        "# Final MOT Timing Diagnosis",
        "",
        f"- EasyErgo MOT: `{mot_path}`",
        f"- EasyErgo duration: `{est_duration_s:.2f} s`",
        f"- GT duration: `{gt_duration_s:.2f} s`",
        "",
        "## Window Offset Scan",
        "",
    ]
    for row in window_df.itertuples(index=False):
        lines.append(
            f"- `{row.window_start_s:.0f}-{row.window_end_s:.0f} s`: "
            f"best offset `{row.best_offset_s:.2f} s`, "
            f"semantic MAE `{row.semantic_mae_deg:.2f} deg`"
        )

    lines.extend(
        [
            "",
            "## Affine Fit",
            "",
            f"- Use `gt_t = {affine_payload['time_scale']:.6f} * est_t - {affine_payload['offset_s']:.2f}`",
            f"- Affine semantic MAE: `{affine_payload['semantic_mae_deg']:.2f} deg`",
            f"- Fixed `17.25 s` semantic MAE: `{affine_payload['fixed_17p25_semantic_mae_deg']:.2f} deg`",
            f"- Fixed `15.90 s` semantic MAE: `{affine_payload['fixed_15p9_semantic_mae_deg']:.2f} deg`",
            "",
            "## Interpretation",
            "",
            "- The final OpenSim MOT export shows the same type of cross-platform timing mismatch as the final MVNX export.",
            "- A small affine time-scale correction should be preferred over a constant offset whenever the window-wise best offset drifts across time.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run timing diagnosis for the final MOT branch."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = resolve_easyergo_final_outputs(INPUT_DIR)
    mot_path = outputs["mot_path"]
    if mot_path is None:
        raise FileNotFoundError(f"No MOT file found in {INPUT_DIR}")

    mot_data = load_opensim_mot(mot_path)
    est_ts = np.asarray(mot_data["time"], dtype=np.float64)
    est_ts = est_ts - est_ts[0]
    est_angles, _ = build_semantic_angles_from_mot(mot_data["coordinates"])

    gt_mvnx = MvnxParser(str(GT_MVNX_PATH))
    gt_mvnx.parse()
    gt_ts = gt_mvnx.timestamps.copy()
    gt_ts, unique_idx = np.unique(gt_ts, return_index=True)
    gt_ts = gt_ts - gt_ts[0]
    gt_interp = build_gt_angle_interpolators(gt_mvnx, gt_ts, unique_idx)

    window_df = scan_window_offsets(est_ts, est_angles, gt_interp)
    window_df.to_csv(WINDOW_SCAN_CSV, index=False)

    affine_payload = fit_affine_mapping(est_ts, est_angles, gt_interp)
    with AFFINE_JSON.open("w", encoding="utf-8") as handle:
        json.dump(affine_payload, handle, indent=2)

    summary = build_summary_markdown(
        mot_path=mot_path,
        est_duration_s=float(est_ts[-1]) if len(est_ts) else 0.0,
        gt_duration_s=float(gt_ts[-1]) if len(gt_ts) else 0.0,
        window_df=window_df,
        affine_payload=affine_payload,
    )
    SUMMARY_MD.write_text(summary, encoding="utf-8")

    print(f"[saved] {WINDOW_SCAN_CSV}")
    print(f"[saved] {AFFINE_JSON}")
    print(f"[saved] {SUMMARY_MD}")
    print(
        "[result] time_scale="
        f"{affine_payload['time_scale']:.6f}, "
        f"offset={affine_payload['offset_s']:.2f}, "
        f"semantic_MAE={affine_payload['semantic_mae_deg']:.2f} deg"
    )


if __name__ == "__main__":
    main()
