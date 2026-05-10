"""Estimate a constant EasyErgo-to-stereo rotation for AFH1 v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AFH1_DIR.parent
RESULTS_DIR = AFH1_DIR / "results"

STEREO_POSE_NPZ = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
EASYERGO_NPZ = RESULTS_DIR / "easyergo_normalized.npz"
ALIGNMENT_OUT = RESULTS_DIR / "coordinate_alignment.json"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.md"

CALIB_JOINTS = [5, 6, 11, 12]
CALIB_JOINT_NAMES = ["LShoulder", "RShoulder", "LHip", "RHip"]
CALIB_WINDOW_SEC = 5.0
DEFAULT_EASYERGO_TO_XSENS_SCALE = 1.0102
DEFAULT_EASYERGO_TO_XSENS_OFFSET_S = 16.83
DEFAULT_STEREO_TO_XSENS_OFFSET_S = 17.25


def append_experiment_log(message: str) -> None:
    """Append one line to the AFH1 experiment log."""
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- {message}\n")


def pelvis_center(points: np.ndarray) -> np.ndarray:
    """Compute pelvis center as the midpoint of left and right hips."""
    pelvis = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
    left = points[:, 11, :]
    right = points[:, 12, :]
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    pelvis[valid] = 0.5 * (left[valid] + right[valid])
    return pelvis


def stereo_time_to_easyergo_time(
    stereo_time_s: np.ndarray,
    easyergo_to_xsens_scale: float,
    easyergo_to_xsens_offset_s: float,
    stereo_to_xsens_offset_s: float,
) -> np.ndarray:
    """Map stereo-video relative time to EasyErgo time via the Xsens clock."""
    xsens_time_s = stereo_time_s - stereo_to_xsens_offset_s
    return (xsens_time_s + easyergo_to_xsens_offset_s) / easyergo_to_xsens_scale


def kabsch_rotation(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Return the optimal rotation that aligns src onto tgt."""
    c_src = np.mean(src, axis=0)
    c_tgt = np.mean(tgt, axis=0)
    h_mat = (src - c_src).T @ (tgt - c_tgt)
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[2, :] *= -1
        rotation = vt_mat.T @ u_mat.T
    return rotation


def main() -> None:
    """Calibrate one constant rotation from EasyErgo space into stereo space."""
    stereo = np.load(STEREO_POSE_NPZ)
    easy = np.load(EASYERGO_NPZ, allow_pickle=True)

    stereo_ts_abs = stereo["timestamps"].astype(np.float64)
    stereo_ts_rel = stereo_ts_abs - stereo_ts_abs[0]
    stereo_kpts = stereo["keypoints"].astype(np.float64)

    easy_ts = easy["timestamps"].astype(np.float64)
    easy_kpts = easy["keypoints_3d"].astype(np.float64)
    easy_query_ts = stereo_time_to_easyergo_time(
        stereo_ts_rel,
        easyergo_to_xsens_scale=DEFAULT_EASYERGO_TO_XSENS_SCALE,
        easyergo_to_xsens_offset_s=DEFAULT_EASYERGO_TO_XSENS_OFFSET_S,
        stereo_to_xsens_offset_s=DEFAULT_STEREO_TO_XSENS_OFFSET_S,
    )

    easy_interp = interp1d(
        easy_ts,
        easy_kpts,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )(easy_query_ts)

    stereo_pelvis = pelvis_center(stereo_kpts)
    easy_pelvis = pelvis_center(easy_interp)
    stereo_rel = stereo_kpts - stereo_pelvis[:, None, :]
    easy_rel = easy_interp - easy_pelvis[:, None, :]

    valid = (
        np.isfinite(stereo_rel[:, CALIB_JOINTS, :]).all(axis=(1, 2))
        & np.isfinite(easy_rel[:, CALIB_JOINTS, :]).all(axis=(1, 2))
        & (stereo_ts_rel >= 0.0)
        & (stereo_ts_rel <= CALIB_WINDOW_SEC)
    )
    frame_indices = np.flatnonzero(valid)
    if len(frame_indices) < 10:
        raise RuntimeError(
            "Not enough valid calibration frames within the initial window. "
            f"Found {len(frame_indices)} valid frames."
        )

    src = easy_rel[frame_indices][:, CALIB_JOINTS, :].reshape(-1, 3)
    tgt = stereo_rel[frame_indices][:, CALIB_JOINTS, :].reshape(-1, 3)
    rotation = kabsch_rotation(src, tgt)

    src_aligned = src @ rotation.T
    residual = np.linalg.norm(src_aligned - tgt, axis=1)

    # Secondary diagnostic on all 17 mapped joints for the same frames.
    all_src = easy_rel[frame_indices].reshape(-1, 3)
    all_tgt = stereo_rel[frame_indices].reshape(-1, 3)
    valid_all = np.isfinite(all_src).all(axis=1) & np.isfinite(all_tgt).all(axis=1)
    all_residual = np.linalg.norm(all_src[valid_all] @ rotation.T - all_tgt[valid_all], axis=1)

    payload = {
        "rotation_3x3": rotation.tolist(),
        "det_rotation": float(np.linalg.det(rotation)),
        "calibration_window_seconds": CALIB_WINDOW_SEC,
        "time_mapping": {
            "formula": (
                "xsens_t = scale * easyergo_t - easyergo_offset; "
                "xsens_t = stereo_t - stereo_offset"
            ),
            "easyergo_to_xsens_scale": DEFAULT_EASYERGO_TO_XSENS_SCALE,
            "easyergo_to_xsens_offset_s": DEFAULT_EASYERGO_TO_XSENS_OFFSET_S,
            "stereo_to_xsens_offset_s": DEFAULT_STEREO_TO_XSENS_OFFSET_S,
            "easyergo_query_start_s": float(np.nanmin(easy_query_ts)),
            "easyergo_query_end_s": float(np.nanmax(easy_query_ts)),
        },
        "calibration_frame_indices": frame_indices.tolist(),
        "calibration_joint_indices": CALIB_JOINTS,
        "calibration_joint_names": CALIB_JOINT_NAMES,
        "residual_cm_mean": float(np.mean(residual)),
        "residual_cm_median": float(np.median(residual)),
        "residual_cm_p95": float(np.percentile(residual, 95)),
        "residual_cm_max": float(np.max(residual)),
        "all_joint_residual_cm_mean": float(np.mean(all_residual)),
        "all_joint_residual_cm_p95": float(np.percentile(all_residual, 95)),
        "all_joint_residual_cm_max": float(np.max(all_residual)),
        "num_calibration_frames": int(len(frame_indices)),
    }

    with ALIGNMENT_OUT.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    append_experiment_log(
        "Calibrated AFH1 constant rotation using shoulders+hips over the first "
        f"{CALIB_WINDOW_SEC:.1f}s ({len(frame_indices)} frames, mean residual "
        f"{payload['residual_cm_mean']:.2f} cm)."
    )

    print(f"[saved] {ALIGNMENT_OUT}")
    print(
        "[info] calibration residual: "
        f"mean={payload['residual_cm_mean']:.2f} cm "
        f"p95={payload['residual_cm_p95']:.2f} cm "
        f"max={payload['residual_cm_max']:.2f} cm"
    )
    print(
        "[info] all-joint residual on calibration window: "
        f"mean={payload['all_joint_residual_cm_mean']:.2f} cm "
        f"p95={payload['all_joint_residual_cm_p95']:.2f} cm"
    )


if __name__ == "__main__":
    main()
