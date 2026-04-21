#!/opt/anaconda3/envs/pose/bin/python
"""Diagnose timing drift for EasyErgo final MVNX outputs."""

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

from pose_angle_utils import compute_semantic_joint_angles  # noqa: E402
from utils_mvnx import MvnxParser  # noqa: E402


INPUT_DIR = AFH1_DIR / "data" / "easyergo_uploaded"
RESULTS_DIR = AFH1_DIR / "results" / "02_final_mvnx_timing"
SUMMARY_MD = RESULTS_DIR / "timing_diagnosis_summary.md"
WINDOW_CSV = RESULTS_DIR / "window_offset_scan.csv"
AFFINE_JSON = RESULTS_DIR / "affine_fit.json"
FAIR_GT_NPZ = PROJECT_ROOT / "shared" / "fair_gt_angles.npz"

ANGLE_NAMES = (
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftHip",
    "RightHip",
    "LeftKnee",
    "RightKnee",
)

XSENS_TO_COCO: dict[str, int] = {
    "LeftUpperArm": 5,
    "RightUpperArm": 6,
    "LeftForeArm": 7,
    "RightForeArm": 8,
    "LeftHand": 9,
    "RightHand": 10,
    "LeftUpperLeg": 11,
    "RightUpperLeg": 12,
    "LeftLowerLeg": 13,
    "RightLowerLeg": 14,
    "LeftFoot": 15,
    "RightFoot": 16,
}


def resolve_single_mvnx(input_dir: Path) -> Path:
    """Resolve the EasyErgo final MVNX file from the upload folder."""
    override = os.environ.get("POSE_EASYERGO_MVNX", "").strip()
    if override:
        path = Path(override).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"POSE_EASYERGO_MVNX does not exist: {path}")
        return path

    matches = sorted(input_dir.glob("*.mvnx"))
    if not matches:
        raise FileNotFoundError(f"No EasyErgo MVNX file found in {input_dir}.")
    if len(matches) > 1:
        formatted = ", ".join(str(path) for path in matches)
        raise FileNotFoundError(f"Multiple EasyErgo MVNX files found in {input_dir}: {formatted}")
    return matches[0]


def build_xsens_coco_poses(mvnx: MvnxParser) -> np.ndarray:
    """Map MVNX segment origins into a pseudo-COCO skeleton."""
    n_frames = mvnx.data.shape[0]
    poses = np.full((n_frames, 17, 3), np.nan, dtype=np.float64)
    for seg_name, coco_idx in XSENS_TO_COCO.items():
        seg_data = mvnx.get_segment_data(seg_name)
        if seg_data is not None:
            poses[:, coco_idx, :] = seg_data
    return poses


def compute_angle_matrix(poses: np.ndarray) -> np.ndarray:
    """Compute the eight semantic joint angles for every frame."""
    angle_matrix = np.full((len(poses), len(ANGLE_NAMES)), np.nan, dtype=np.float64)
    for frame_idx, pose in enumerate(poses):
        frame_angles = compute_semantic_joint_angles(pose)
        for angle_idx, angle_name in enumerate(ANGLE_NAMES):
            angle_matrix[frame_idx, angle_idx] = frame_angles[angle_name]
    return angle_matrix


def mean_angle_error(
    est_ts: np.ndarray,
    est_angles: np.ndarray,
    gt_ts: np.ndarray,
    gt_angles: np.ndarray,
    time_scale: float,
    offset_s: float,
    sample_mask: np.ndarray | None = None,
) -> tuple[float, int]:
    """Compute mean absolute error for one affine time mapping."""
    if sample_mask is None:
        sample_mask = np.ones(len(est_ts), dtype=bool)

    target_ts = time_scale * est_ts[sample_mask] - offset_s
    angle_samples = est_angles[sample_mask]
    errors: list[np.ndarray] = []

    for angle_idx in range(len(ANGLE_NAMES)):
        gt_series = gt_angles[:, angle_idx]
        finite_gt = np.isfinite(gt_series)
        interp = np.interp(
            target_ts,
            gt_ts[finite_gt],
            gt_series[finite_gt],
            left=np.nan,
            right=np.nan,
        )
        est_series = angle_samples[:, angle_idx]
        finite = np.isfinite(est_series) & np.isfinite(interp)
        if np.any(finite):
            errors.append(np.abs(est_series[finite] - interp[finite]))

    if not errors:
        return float("inf"), 0
    merged = np.concatenate(errors)
    return float(np.mean(merged)), int(len(merged))


def scan_window_offsets(
    est_ts: np.ndarray,
    est_angles: np.ndarray,
    gt_ts: np.ndarray,
    gt_angles: np.ndarray,
) -> pd.DataFrame:
    """Estimate the best constant offset in multiple windows."""
    windows = [(20, 50), (50, 80), (80, 110), (110, 140), (140, 170), (170, 200)]
    rows: list[dict[str, float]] = []

    for start_t, end_t in windows:
        mask = (est_ts >= start_t) & (est_ts < end_t)
        best_offset = float("nan")
        best_mae = float("inf")
        best_count = 0
        for offset_s in np.arange(14.0, 18.51, 0.02):
            mae, count = mean_angle_error(
                est_ts=est_ts,
                est_angles=est_angles,
                gt_ts=gt_ts,
                gt_angles=gt_angles,
                time_scale=1.0,
                offset_s=float(offset_s),
                sample_mask=mask,
            )
            if mae < best_mae:
                best_offset = float(offset_s)
                best_mae = mae
                best_count = count
        rows.append(
            {
                "window_start_s": float(start_t),
                "window_end_s": float(end_t),
                "window_mid_s": float(0.5 * (start_t + end_t)),
                "best_offset_s": best_offset,
                "fair_mae_deg": best_mae,
                "samples": int(best_count),
            }
        )
    return pd.DataFrame(rows)


def fit_affine_mapping(
    est_ts: np.ndarray,
    est_angles: np.ndarray,
    gt_ts: np.ndarray,
    gt_angles: np.ndarray,
) -> dict[str, float]:
    """Fit gt_t = time_scale * est_t - offset using a small grid search."""
    central_mask = (est_ts >= 17.0) & (est_ts <= 205.0)
    coarse_best = {"time_scale": np.nan, "offset_s": np.nan, "fair_mae_deg": float("inf"), "samples": 0}

    for time_scale in np.arange(1.005, 1.0261, 0.001):
        for offset_s in np.arange(16.0, 18.21, 0.05):
            mae, count = mean_angle_error(
                est_ts=est_ts,
                est_angles=est_angles,
                gt_ts=gt_ts,
                gt_angles=gt_angles,
                time_scale=float(time_scale),
                offset_s=float(offset_s),
                sample_mask=central_mask,
            )
            if mae < coarse_best["fair_mae_deg"]:
                coarse_best = {
                    "time_scale": float(time_scale),
                    "offset_s": float(offset_s),
                    "fair_mae_deg": float(mae),
                    "samples": int(count),
                }

    fine_best = {"time_scale": np.nan, "offset_s": np.nan, "fair_mae_deg": float("inf"), "samples": 0}
    for time_scale in np.arange(coarse_best["time_scale"] - 0.002, coarse_best["time_scale"] + 0.0021, 0.0002):
        for offset_s in np.arange(coarse_best["offset_s"] - 0.15, coarse_best["offset_s"] + 0.1501, 0.01):
            mae, count = mean_angle_error(
                est_ts=est_ts,
                est_angles=est_angles,
                gt_ts=gt_ts,
                gt_angles=gt_angles,
                time_scale=float(time_scale),
                offset_s=float(offset_s),
                sample_mask=central_mask,
            )
            if mae < fine_best["fair_mae_deg"]:
                fine_best = {
                    "time_scale": float(time_scale),
                    "offset_s": float(offset_s),
                    "fair_mae_deg": float(mae),
                    "samples": int(count),
                }

    fixed_15p9_mae, fixed_15p9_count = mean_angle_error(
        est_ts=est_ts,
        est_angles=est_angles,
        gt_ts=gt_ts,
        gt_angles=gt_angles,
        time_scale=1.0,
        offset_s=15.9,
        sample_mask=central_mask,
    )
    fixed_17p25_mae, fixed_17p25_count = mean_angle_error(
        est_ts=est_ts,
        est_angles=est_angles,
        gt_ts=gt_ts,
        gt_angles=gt_angles,
        time_scale=1.0,
        offset_s=17.25,
        sample_mask=central_mask,
    )

    return {
        "time_scale": fine_best["time_scale"],
        "offset_s": fine_best["offset_s"],
        "fair_mae_deg": fine_best["fair_mae_deg"],
        "samples": fine_best["samples"],
        "fixed_15p9_fair_mae_deg": fixed_15p9_mae,
        "fixed_15p9_samples": fixed_15p9_count,
        "fixed_17p25_fair_mae_deg": fixed_17p25_mae,
        "fixed_17p25_samples": fixed_17p25_count,
    }


def write_summary(
    mvnx_path: Path,
    est_duration_s: float,
    gt_duration_s: float,
    window_df: pd.DataFrame,
    affine_payload: dict[str, float],
) -> None:
    """Write a concise markdown diagnosis summary."""
    lines = [
        "# Final MVNX Timing Diagnosis",
        "",
        "## Inputs",
        f"- EasyErgo MVNX: `{mvnx_path}`",
        f"- EasyErgo duration: `{est_duration_s:.2f} s`",
        f"- GT duration: `{gt_duration_s:.2f} s`",
        "",
        "## Windowed Constant-Offset Scan",
    ]
    for _, row in window_df.iterrows():
        lines.append(
            f"- `{row['window_start_s']:.0f}-{row['window_end_s']:.0f} s`: "
            f"best offset `{row['best_offset_s']:.2f} s`, "
            f"fair MAE `{row['fair_mae_deg']:.2f} deg`"
        )

    lines += [
        "",
        "## Recommended Affine Mapping",
        (
            f"- Use `gt_t = {affine_payload['time_scale']:.6f} * est_t - {affine_payload['offset_s']:.2f}`"
        ),
        f"- Affine fair MAE (central overlap): `{affine_payload['fair_mae_deg']:.2f} deg`",
        f"- Fixed `offset=15.9`: `{affine_payload['fixed_15p9_fair_mae_deg']:.2f} deg`",
        f"- Fixed `offset=17.25`: `{affine_payload['fixed_17p25_fair_mae_deg']:.2f} deg`",
        "",
        "## Interpretation",
        "- The best offset is not constant across the sequence, which indicates timing drift rather than a single global shift.",
        "- A linear time-scale correction is a better model than a constant offset for the final MVNX branch.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run timing diagnosis for the final MVNX branch."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mvnx_path = resolve_single_mvnx(INPUT_DIR)

    est_mvnx = MvnxParser(str(mvnx_path))
    est_mvnx.parse()
    est_ts = est_mvnx.timestamps.copy()
    est_ts, unique_idx = np.unique(est_ts, return_index=True)
    est_ts -= est_ts[0]

    # Downsample a bit for faster diagnosis while keeping the trend visible.
    sample_idx = np.arange(0, len(est_ts), 3)
    est_ts = est_ts[sample_idx]
    est_poses = build_xsens_coco_poses(est_mvnx)[unique_idx][sample_idx]
    est_angles = compute_angle_matrix(est_poses)

    fair_gt = np.load(FAIR_GT_NPZ)
    gt_ts = np.asarray(fair_gt["timestamps"], dtype=np.float64)
    gt_angles = np.stack([np.asarray(fair_gt[name], dtype=np.float64) for name in ANGLE_NAMES], axis=1)

    window_df = scan_window_offsets(est_ts, est_angles, gt_ts, gt_angles)
    affine_payload = fit_affine_mapping(est_ts, est_angles, gt_ts, gt_angles)

    window_df.to_csv(WINDOW_CSV, index=False)
    AFFINE_JSON.write_text(json.dumps(affine_payload, indent=2), encoding="utf-8")
    write_summary(
        mvnx_path=mvnx_path,
        est_duration_s=float(est_ts[-1]) if len(est_ts) else 0.0,
        gt_duration_s=float(gt_ts[-1]) if len(gt_ts) else 0.0,
        window_df=window_df,
        affine_payload=affine_payload,
    )

    print(f"[saved] {WINDOW_CSV}")
    print(f"[saved] {AFFINE_JSON}")
    print(f"[saved] {SUMMARY_MD}")
    print(
        "[result] time_scale="
        f"{affine_payload['time_scale']:.6f}, "
        f"offset={affine_payload['offset_s']:.2f}, "
        f"affine_fair_mae={affine_payload['fair_mae_deg']:.2f}"
    )


if __name__ == "__main__":
    main()
