"""Diagnostic analysis for AFH1 v1/v2 hybrid experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags
from scipy.spatial.transform import Rotation

from easyergo_marker_mapping import COCO17_JOINT_NAMES


matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AFH1_DIR.parent
RESULTS_DIR = AFH1_DIR / "results"
DIAG_DIR = RESULTS_DIR / "diagnose_v1"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.md"

STEREO_POSE_NPZ = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
EASYERGO_NPZ = RESULTS_DIR / "easyergo_normalized.npz"
STEREO_ANCHOR_NPZ = RESULTS_DIR / "stereo_pelvis_anchor.npz"
ALIGNMENT_JSON = RESULTS_DIR / "coordinate_alignment.json"
INSPECTION_JSON = RESULTS_DIR / "easyergo_trc_inspection.json"

TIME_SYNC_JSON = DIAG_DIR / "time_sync.json"
TIME_SYNC_PNG = DIAG_DIR / "time_sync.png"
PER_FRAME_KABSCH_JSON = DIAG_DIR / "per_frame_kabsch.json"
PER_FRAME_KABSCH_PNG = DIAG_DIR / "per_frame_kabsch.png"
PER_JOINT_RESIDUAL_CSV = DIAG_DIR / "per_joint_residual.csv"
PER_JOINT_RESIDUAL_PNG = DIAG_DIR / "per_joint_residual.png"
BONE_LENGTH_CSV = DIAG_DIR / "bone_length_comparison.csv"
BONE_LENGTH_PNG = DIAG_DIR / "bone_length_comparison.png"
SUMMARY_MD = DIAG_DIR / "diagnosis_summary.md"

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
NOSE = 0

CALIB_WINDOW_SEC = 5.0
CALIB_JOINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
MAX_LAG_SEC = 3.0
SYNC_BAD_THRESHOLD_SEC = 0.3
SYNC_RELIABLE_CORR_THRESHOLD = 0.2
MOTION_DRIVEN_THRESHOLD_DEG = 20.0


def append_experiment_log(message: str) -> None:
    """Append one line to the AFH1 experiment log."""
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- {message}\n")


def midpoint(points: np.ndarray, left_idx: int, right_idx: int) -> np.ndarray:
    """Return the midpoint between two joints for each frame."""
    out = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
    left = points[:, left_idx, :]
    right = points[:, right_idx, :]
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    out[valid] = 0.5 * (left[valid] + right[valid])
    return out


def pelvis_center(points: np.ndarray) -> np.ndarray:
    """Return pelvis midpoint."""
    return midpoint(points, LEFT_HIP, RIGHT_HIP)


def shoulder_mid(points: np.ndarray) -> np.ndarray:
    """Return shoulder midpoint."""
    return midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER)


def kabsch_rotation(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Return the optimal rotation that aligns src onto tgt."""
    src_center = np.mean(src, axis=0)
    tgt_center = np.mean(tgt, axis=0)
    h_mat = (src - src_center).T @ (tgt - tgt_center)
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[2, :] *= -1
        rotation = vt_mat.T @ u_mat.T
    return rotation


def finite_interp(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a 1D signal onto a common grid with NaN support."""
    mask = np.isfinite(times) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    interp = interp1d(
        times[mask],
        values[mask],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(grid)


def zscore(values: np.ndarray) -> np.ndarray:
    """Return a z-scored copy with NaN support."""
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return np.full_like(values, np.nan, dtype=np.float64)
    centered = values.copy()
    mean = np.nanmean(centered)
    std = np.nanstd(centered)
    if std < 1e-8:
        return np.full_like(values, np.nan, dtype=np.float64)
    centered[finite] = (centered[finite] - mean) / std
    return centered


def cross_correlation_lag(
    ref_times: np.ndarray,
    ref_signal: np.ndarray,
    query_times: np.ndarray,
    query_signal: np.ndarray,
    step_s: float,
    max_lag_s: float,
) -> Dict[str, object]:
    """Compute the best correlation lag between two signals."""
    t_start = max(float(np.nanmin(ref_times)), float(np.nanmin(query_times)))
    t_end = min(float(np.nanmax(ref_times)), float(np.nanmax(query_times)))
    if t_end - t_start < step_s * 10:
        raise RuntimeError("Not enough overlap for time-sync analysis.")

    grid = np.arange(t_start, t_end + 0.5 * step_s, step_s, dtype=np.float64)
    ref_grid = zscore(finite_interp(ref_times, ref_signal, grid))
    query_grid = zscore(finite_interp(query_times, query_signal, grid))

    mask = np.isfinite(ref_grid) & np.isfinite(query_grid)
    ref_valid = ref_grid[mask]
    query_valid = query_grid[mask]
    if len(ref_valid) < 10:
        raise RuntimeError("Not enough valid samples for cross-correlation.")

    corr = correlate(ref_valid, query_valid, mode="full")
    lags = correlation_lags(len(ref_valid), len(query_valid), mode="full")
    corr = corr / (len(ref_valid) * np.std(ref_valid) * np.std(query_valid))
    lag_seconds = lags * step_s
    lag_mask = np.abs(lag_seconds) <= max_lag_s
    corr = corr[lag_mask]
    lag_seconds = lag_seconds[lag_mask]

    best_idx = int(np.nanargmax(corr))
    return {
        "grid_time_s": grid[mask].tolist(),
        "ref_zscore": ref_valid.tolist(),
        "query_zscore": query_valid.tolist(),
        "lag_seconds": lag_seconds.tolist(),
        "correlation": corr.tolist(),
        "best_lag_seconds": float(lag_seconds[best_idx]),
        "peak_correlation": float(corr[best_idx]),
        "num_samples": int(len(ref_valid)),
    }


def axis_signal(
    relative_points: np.ndarray,
    joint_idx: int,
    axis_idx: int,
    sign: float,
) -> np.ndarray:
    """Extract one signed coordinate component from a relative pose array."""
    signal = np.full(relative_points.shape[0], np.nan, dtype=np.float64)
    joint = relative_points[:, joint_idx, :]
    valid = np.isfinite(joint[:, axis_idx])
    signal[valid] = sign * joint[valid, axis_idx]
    return signal


def compute_rotation_deviation_deg(reference_rotation: np.ndarray, frame_rotation: np.ndarray) -> float:
    """Return the angular deviation between two rotations in degrees."""
    delta = reference_rotation.T @ frame_rotation
    return float(np.degrees(Rotation.from_matrix(delta).magnitude()))


def compute_bone_median(points: np.ndarray, spec: str) -> float:
    """Compute the median length of one skeletal segment in cm."""
    if spec == "shoulder_width":
        values = np.linalg.norm(points[:, LEFT_SHOULDER, :] - points[:, RIGHT_SHOULDER, :], axis=1)
    elif spec == "hip_width":
        values = np.linalg.norm(points[:, LEFT_HIP, :] - points[:, RIGHT_HIP, :], axis=1)
    elif spec == "torso":
        values = np.linalg.norm(shoulder_mid(points) - pelvis_center(points), axis=1)
    elif spec == "upper_arm":
        left = np.linalg.norm(points[:, LEFT_SHOULDER, :] - points[:, LEFT_ELBOW, :], axis=1)
        right = np.linalg.norm(points[:, RIGHT_SHOULDER, :] - points[:, RIGHT_ELBOW, :], axis=1)
        values = 0.5 * (left + right)
    elif spec == "forearm":
        left = np.linalg.norm(points[:, LEFT_ELBOW, :] - points[:, LEFT_WRIST, :], axis=1)
        right = np.linalg.norm(points[:, RIGHT_ELBOW, :] - points[:, RIGHT_WRIST, :], axis=1)
        values = 0.5 * (left + right)
    elif spec == "thigh":
        left = np.linalg.norm(points[:, LEFT_HIP, :] - points[:, LEFT_KNEE, :], axis=1)
        right = np.linalg.norm(points[:, RIGHT_HIP, :] - points[:, RIGHT_KNEE, :], axis=1)
        values = 0.5 * (left + right)
    elif spec == "shank":
        left = np.linalg.norm(points[:, LEFT_KNEE, :] - points[:, LEFT_ANKLE, :], axis=1)
        right = np.linalg.norm(points[:, RIGHT_KNEE, :] - points[:, RIGHT_ANKLE, :], axis=1)
        values = 0.5 * (left + right)
    else:
        raise ValueError(f"Unsupported bone spec: {spec}")

    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else math.nan


def write_summary_markdown(
    time_sync: Dict[str, object],
    per_frame_payload: Dict[str, object],
    per_joint_df: pd.DataFrame,
    bone_df: pd.DataFrame,
    decision: str,
    verdict_sync: str,
    verdict_model: str,
) -> None:
    """Write the one-page diagnosis summary markdown."""
    top3 = per_joint_df.sort_values("mean_residual_cm", ascending=False).head(3)
    lines = [
        "# AFH1 v1 Diagnosis Summary",
        "",
        "## 1. Time sync",
        f"- Primary best lag: {time_sync['primary']['best_lag_seconds']:.3f} s",
        f"- Primary peak correlation: {time_sync['primary']['peak_correlation']:.3f}",
        f"- Secondary best lag: {time_sync['secondary']['best_lag_seconds']:.3f} s",
        f"- Secondary peak correlation: {time_sync['secondary']['peak_correlation']:.3f}",
            f"- Verdict: {verdict_sync}",
        "",
        "## 2. Per-frame Kabsch spectrum",
        f"- Mean rotation deviation from pooled: {per_frame_payload['rotation_dev_deg_mean']:.2f} deg",
        f"- P95 rotation deviation from pooled: {per_frame_payload['rotation_dev_deg_p95']:.2f} deg",
        f"- Per-frame residual mean: {per_frame_payload['residual_cm_mean']:.2f} cm",
        f"- Per-frame residual p95: {per_frame_payload['residual_cm_p95']:.2f} cm",
        f"- Verdict: {verdict_model}",
        "",
        "## 3. Per-joint residual (pooled R, first 5 s)",
        "",
        "| Joint | Mean Residual (cm) | P95 (cm) |",
        "|------|---------------------|----------|",
    ]
    for _, row in per_joint_df.iterrows():
        lines.append(
            f"| {row['joint_name']} | {row['mean_residual_cm']:.2f} | {row['p95_residual_cm']:.2f} |"
        )

    lines.extend(
        [
            "",
            f"- Top-3 worst joints: {', '.join(top3['joint_name'].tolist())}",
            "",
            "## 4. Bone-length comparison",
            "",
            "| Segment | Stereo (cm) | EasyErgo (cm) | Ratio |",
            "|---------|-------------|---------------|-------|",
        ]
    )
    for _, row in bone_df.iterrows():
        lines.append(
            f"| {row['segment']} | {row['stereo_cm']:.2f} | {row['easyergo_cm']:.2f} | {row['ratio_easy_over_stereo']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            f"- Recommended next step: {decision}",
            "",
            "Decision rules applied:",
            f"- If correlation is reliable and |lag| > {SYNC_BAD_THRESHOLD_SEC:.1f}s -> fix time sync first",
            f"- If mean rotation deviation > {MOTION_DRIVEN_THRESHOLD_DEG:.1f} deg -> motion-driven",
            "- If sync is OK and hips dominate residuals -> D2 selective trunk mixing",
            "- Otherwise -> D3 negative-result write-up",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the AFH1 v1 diagnosis package."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    stereo = np.load(STEREO_POSE_NPZ, allow_pickle=True)
    easy = np.load(EASYERGO_NPZ, allow_pickle=True)
    stereo_anchor = np.load(STEREO_ANCHOR_NPZ, allow_pickle=True)
    with ALIGNMENT_JSON.open("r", encoding="utf-8") as handle:
        alignment = json.load(handle)
    with INSPECTION_JSON.open("r", encoding="utf-8") as handle:
        inspection = json.load(handle)

    pooled_rotation = np.asarray(alignment["rotation_3x3"], dtype=np.float64)
    stereo_ts_abs = stereo["timestamps"].astype(np.float64)
    stereo_ts_rel = stereo_ts_abs - stereo_ts_abs[0]
    stereo_kpts = stereo["keypoints"].astype(np.float64)
    easy_ts = easy["timestamps"].astype(np.float64)
    easy_kpts = easy["keypoints_3d"].astype(np.float64)
    easy_native_pelvis = pelvis_center(easy_kpts)
    easy_native_rel = easy_kpts - easy_native_pelvis[:, None, :]

    easy_interp = interp1d(
        easy_ts,
        easy_kpts,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )(stereo_ts_rel)

    stereo_pelvis = pelvis_center(stereo_kpts)
    easy_pelvis = pelvis_center(easy_interp)
    stereo_rel = stereo_kpts - stereo_pelvis[:, None, :]
    easy_rel = easy_interp - easy_pelvis[:, None, :]

    # 1. Time sync on two signals.
    stereo_head_pelvis = axis_signal(stereo_rel, NOSE, axis_idx=1, sign=-1.0)
    up_hint = inspection["geometry_hints"].get("up_axis_from_head_pelvis", {"axis": "X", "sign": -1.0})
    easy_axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(str(up_hint.get("axis", "X")).upper(), 0)
    easy_sign = float(up_hint.get("sign", -1.0))
    easy_head_pelvis = axis_signal(easy_native_rel, NOSE, axis_idx=easy_axis_idx, sign=easy_sign)
    stereo_torso_length = np.linalg.norm(shoulder_mid(stereo_kpts) - stereo_pelvis, axis=1)
    easy_torso_length = np.linalg.norm(shoulder_mid(easy_kpts) - easy_native_pelvis, axis=1)

    step_s = float(np.median(np.diff(stereo_anchor["timestamps_rel"].astype(np.float64))))
    primary_sync = cross_correlation_lag(
        stereo_ts_rel,
        stereo_head_pelvis,
        easy_ts,
        easy_head_pelvis,
        step_s=step_s,
        max_lag_s=MAX_LAG_SEC,
    )
    secondary_sync = cross_correlation_lag(
        stereo_ts_rel,
        stereo_torso_length,
        easy_ts,
        easy_torso_length,
        step_s=step_s,
        max_lag_s=MAX_LAG_SEC,
    )

    time_sync_payload = {
        "primary": {
            "signal_name": "head_minus_pelvis_height_proxy",
            **{k: v for k, v in primary_sync.items() if k not in {"grid_time_s", "ref_zscore", "query_zscore", "lag_seconds", "correlation"}},
        },
        "secondary": {
            "signal_name": "torso_length",
            **{k: v for k, v in secondary_sync.items() if k not in {"grid_time_s", "ref_zscore", "query_zscore", "lag_seconds", "correlation"}},
        },
        "signal_ranges": {
            "stereo_head_pelvis": [float(np.nanmin(stereo_head_pelvis)), float(np.nanmax(stereo_head_pelvis))],
            "easy_head_pelvis": [float(np.nanmin(easy_head_pelvis)), float(np.nanmax(easy_head_pelvis))],
            "stereo_torso_length": [float(np.nanmin(stereo_torso_length)), float(np.nanmax(stereo_torso_length))],
            "easy_torso_length": [float(np.nanmin(easy_torso_length)), float(np.nanmax(easy_torso_length))],
        },
    }
    TIME_SYNC_JSON.write_text(json.dumps(time_sync_payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes[0].plot(primary_sync["grid_time_s"], primary_sync["ref_zscore"], label="Stereo", lw=1.5)
    axes[0].plot(primary_sync["grid_time_s"], primary_sync["query_zscore"], label="EasyErgo", lw=1.5)
    axes[0].set_title(
        "Primary sync signal: head-pelvis proxy "
        f"(best lag {primary_sync['best_lag_seconds']:.3f}s, corr {primary_sync['peak_correlation']:.3f})"
    )
    axes[0].set_ylabel("z-score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(secondary_sync["grid_time_s"], secondary_sync["ref_zscore"], label="Stereo", lw=1.5)
    axes[1].plot(secondary_sync["grid_time_s"], secondary_sync["query_zscore"], label="EasyErgo", lw=1.5)
    axes[1].set_title(
        "Secondary sync signal: torso length "
        f"(best lag {secondary_sync['best_lag_seconds']:.3f}s, corr {secondary_sync['peak_correlation']:.3f})"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("z-score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(TIME_SYNC_PNG, dpi=160)
    plt.close(fig)

    # 2. Per-frame Kabsch spectrum on first 5 seconds.
    calib_mask = (
        np.isfinite(stereo_rel[:, CALIB_JOINTS, :]).all(axis=(1, 2))
        & np.isfinite(easy_rel[:, CALIB_JOINTS, :]).all(axis=(1, 2))
        & (stereo_ts_rel >= 0.0)
        & (stereo_ts_rel <= CALIB_WINDOW_SEC)
    )
    calib_indices = np.flatnonzero(calib_mask)

    per_frame_records = []
    pooled_records = []
    for frame_idx in calib_indices:
        src = easy_rel[frame_idx, CALIB_JOINTS, :]
        tgt = stereo_rel[frame_idx, CALIB_JOINTS, :]
        frame_rotation = kabsch_rotation(src, tgt)
        frame_aligned = src @ frame_rotation.T
        pooled_aligned = src @ pooled_rotation.T
        frame_residual = np.linalg.norm(frame_aligned - tgt, axis=1)
        pooled_residual = np.linalg.norm(pooled_aligned - tgt, axis=1)
        rotvec = Rotation.from_matrix(frame_rotation).as_rotvec()
        per_frame_records.append(
            {
                "frame_idx": int(frame_idx),
                "time_s": float(stereo_ts_rel[frame_idx]),
                "rotvec_x": float(rotvec[0]),
                "rotvec_y": float(rotvec[1]),
                "rotvec_z": float(rotvec[2]),
                "rotation_angle_deg": float(np.degrees(np.linalg.norm(rotvec))),
                "rotation_dev_from_pooled_deg": compute_rotation_deviation_deg(pooled_rotation, frame_rotation),
                "residual_cm_mean": float(np.mean(frame_residual)),
                "residual_cm_p95": float(np.percentile(frame_residual, 95)),
                "pooled_residual_cm_mean": float(np.mean(pooled_residual)),
            }
        )
        pooled_records.extend(pooled_residual.tolist())

    per_frame_df = pd.DataFrame(per_frame_records)
    per_frame_payload = {
        "num_frames": int(len(calib_indices)),
        "rotation_dev_deg_mean": float(per_frame_df["rotation_dev_from_pooled_deg"].mean()),
        "rotation_dev_deg_p95": float(per_frame_df["rotation_dev_from_pooled_deg"].quantile(0.95)),
        "residual_cm_mean": float(per_frame_df["residual_cm_mean"].mean()),
        "residual_cm_p95": float(per_frame_df["residual_cm_mean"].quantile(0.95)),
        "frames": per_frame_records,
    }
    PER_FRAME_KABSCH_JSON.write_text(json.dumps(per_frame_payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(per_frame_df["time_s"], per_frame_df["rotation_dev_from_pooled_deg"], marker="o", ms=3)
    axes[0].axhline(MOTION_DRIVEN_THRESHOLD_DEG, color="tab:red", linestyle="--", lw=1.0)
    axes[0].set_ylabel("Rotation dev (deg)")
    axes[0].set_title("Per-frame Kabsch deviation from pooled rotation")
    axes[0].grid(alpha=0.3)

    axes[1].plot(per_frame_df["time_s"], per_frame_df["residual_cm_mean"], marker="o", ms=3, label="Per-frame Kabsch")
    axes[1].plot(per_frame_df["time_s"], per_frame_df["pooled_residual_cm_mean"], marker="o", ms=3, label="Pooled R")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual mean (cm)")
    axes[1].set_title("Per-frame residual on shoulders+hips")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PER_FRAME_KABSCH_PNG, dpi=160)
    plt.close(fig)

    # 3. Per-joint residual with pooled rotation over the calibration window.
    easy_pooled = np.einsum("ij,tkj->tki", pooled_rotation, easy_rel[calib_indices])
    stereo_window = stereo_rel[calib_indices]
    joint_rows = []
    for joint_idx, joint_name in enumerate(COCO17_JOINT_NAMES):
        valid_joint = np.isfinite(easy_pooled[:, joint_idx, :]).all(axis=1) & np.isfinite(stereo_window[:, joint_idx, :]).all(axis=1)
        residual = np.linalg.norm(easy_pooled[valid_joint, joint_idx, :] - stereo_window[valid_joint, joint_idx, :], axis=1)
        joint_rows.append(
            {
                "joint_idx": joint_idx,
                "joint_name": joint_name,
                "samples": int(len(residual)),
                "mean_residual_cm": float(np.mean(residual)) if len(residual) else math.nan,
                "median_residual_cm": float(np.median(residual)) if len(residual) else math.nan,
                "p95_residual_cm": float(np.percentile(residual, 95)) if len(residual) else math.nan,
            }
        )
    per_joint_df = pd.DataFrame(joint_rows)
    per_joint_df.to_csv(PER_JOINT_RESIDUAL_CSV, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(per_joint_df["joint_name"], per_joint_df["mean_residual_cm"], color="tab:blue", alpha=0.85)
    ax.set_ylabel("Mean residual (cm)")
    ax.set_title("Per-joint residual after pooled rotation (first 5 s)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PER_JOINT_RESIDUAL_PNG, dpi=160)
    plt.close(fig)

    # 4. Bone-length comparison on full sequences.
    bone_specs = ["shoulder_width", "hip_width", "torso", "upper_arm", "forearm", "thigh", "shank"]
    bone_rows = []
    for spec in bone_specs:
        stereo_val = compute_bone_median(stereo_kpts, spec)
        easy_val = compute_bone_median(easy_kpts, spec)
        bone_rows.append(
            {
                "segment": spec,
                "stereo_cm": stereo_val,
                "easyergo_cm": easy_val,
                "ratio_easy_over_stereo": easy_val / stereo_val if np.isfinite(stereo_val) and abs(stereo_val) > 1e-8 else math.nan,
            }
        )
    bone_df = pd.DataFrame(bone_rows)
    bone_df.to_csv(BONE_LENGTH_CSV, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bone_df))
    width = 0.38
    ax.bar(x - 0.5 * width, bone_df["stereo_cm"], width=width, label="Stereo", color="tab:green")
    ax.bar(x + 0.5 * width, bone_df["easyergo_cm"], width=width, label="EasyErgo", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(bone_df["segment"], rotation=35)
    ax.set_ylabel("Median length (cm)")
    ax.set_title("Bone-length comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(BONE_LENGTH_PNG, dpi=160)
    plt.close(fig)

    # 5. Decision logic.
    primary_corr = abs(float(time_sync_payload["primary"]["peak_correlation"]))
    secondary_corr = abs(float(time_sync_payload["secondary"]["peak_correlation"]))
    sync_reliable = max(primary_corr, secondary_corr) >= SYNC_RELIABLE_CORR_THRESHOLD
    if not sync_reliable:
        verdict_sync = "sync_inconclusive"
    elif abs(time_sync_payload["primary"]["best_lag_seconds"]) > SYNC_BAD_THRESHOLD_SEC:
        verdict_sync = "sync_bad"
    else:
        verdict_sync = "sync_ok"

    mean_rot_dev = per_frame_payload["rotation_dev_deg_mean"]
    verdict_model = "motion_driven" if mean_rot_dev > MOTION_DRIVEN_THRESHOLD_DEG else "model_driven"
    top3 = per_joint_df.sort_values("mean_residual_cm", ascending=False).head(3)["joint_name"].tolist()
    hips_dominate = ("LHip" in top3) or ("RHip" in top3)
    high_joint_count = int(np.sum(per_joint_df["mean_residual_cm"] > 15.0))
    broad_segment_mismatch = int(
        np.sum(
            np.abs(bone_df["ratio_easy_over_stereo"].to_numpy(dtype=np.float64) - 1.0) > 0.2
        )
    )

    if verdict_sync == "sync_bad":
        decision = "Fix time alignment first, then rerun AFH1 v1/v2."
    elif verdict_sync == "sync_inconclusive" and verdict_model == "model_driven":
        decision = (
            "Do not spend effort on lag correction yet; the sync signals are inconclusive "
            "and the dominant issue is broad skeleton-model mismatch. Proceed to D3."
        )
    elif verdict_model == "motion_driven":
        decision = "Replace the pooled 5 s calibration window with a cleaner static segment and recalibrate."
    elif hips_dominate:
        decision = "Proceed to D2 selective trunk-only mixing; hip semantics appear to be the main mismatch."
    elif high_joint_count >= 6 or broad_segment_mismatch >= 4:
        decision = "Proceed to D3 and document AFH1 as a negative result; mismatch is broad, not hip-local."
    else:
        decision = "Proceed to D3 unless a very targeted trunk-only hypothesis remains."

    write_summary_markdown(
        time_sync=time_sync_payload,
        per_frame_payload=per_frame_payload,
        per_joint_df=per_joint_df,
        bone_df=bone_df,
        decision=decision,
        verdict_sync=verdict_sync,
        verdict_model=verdict_model,
    )

    append_experiment_log(
        "D1 diagnosis completed: "
        f"lag={time_sync_payload['primary']['best_lag_seconds']:.3f}s, "
        f"rotation_dev_mean={per_frame_payload['rotation_dev_deg_mean']:.2f}deg, "
        f"decision={decision}"
    )

    print(f"[saved] {TIME_SYNC_JSON}")
    print(f"[saved] {TIME_SYNC_PNG}")
    print(f"[saved] {PER_FRAME_KABSCH_JSON}")
    print(f"[saved] {PER_FRAME_KABSCH_PNG}")
    print(f"[saved] {PER_JOINT_RESIDUAL_CSV}")
    print(f"[saved] {PER_JOINT_RESIDUAL_PNG}")
    print(f"[saved] {BONE_LENGTH_CSV}")
    print(f"[saved] {BONE_LENGTH_PNG}")
    print(f"[saved] {SUMMARY_MD}")
    print(
        "[info] decision: "
        f"{decision}"
    )


if __name__ == "__main__":
    main()
