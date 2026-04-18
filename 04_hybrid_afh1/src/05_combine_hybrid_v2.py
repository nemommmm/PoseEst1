"""Combine EasyErgo and stereo using pelvis + torso anchors for AFH1 v2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AFH1_DIR.parent
RESULTS_DIR = AFH1_DIR / "results"

EASYERGO_NPZ = RESULTS_DIR / "easyergo_normalized.npz"
ROTATION_JSON = RESULTS_DIR / "coordinate_alignment.json"
STEREO_POSE_NPZ = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
HYBRID_V2_NPZ = RESULTS_DIR / "hybrid_skeleton_afh1_v2.npz"
HYBRID_V2_SUMMARY_JSON = RESULTS_DIR / "hybrid_skeleton_afh1_v2_summary.json"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.md"

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
SCALE_RADIUS = 2
SCALE_MIN = 0.55
SCALE_MAX = 1.25


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


def vector_alignment_rotation(src_vec: np.ndarray, tgt_vec: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that aligns one 3D vector with another."""
    if not (np.isfinite(src_vec).all() and np.isfinite(tgt_vec).all()):
        return np.eye(3)

    src_norm = np.linalg.norm(src_vec)
    tgt_norm = np.linalg.norm(tgt_vec)
    if src_norm < 1e-8 or tgt_norm < 1e-8:
        return np.eye(3)

    src_unit = src_vec / src_norm
    tgt_unit = tgt_vec / tgt_norm
    cross = np.cross(src_unit, tgt_unit)
    dot = float(np.clip(np.dot(src_unit, tgt_unit), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3)
        # 180-degree flip: choose any stable orthogonal axis.
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src_unit[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cross = np.cross(src_unit, axis)
        cross /= np.linalg.norm(cross)
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + 2.0 * (skew @ skew)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - dot) / (cross_norm ** 2))


def median_filter_series(values: np.ndarray, radius: int) -> np.ndarray:
    """Apply a simple temporal median filter to a 1D series with NaN support."""
    if radius <= 0:
        return values.copy()
    out = values.copy()
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        window = values[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size > 0:
            out[idx] = float(np.median(finite))
    return out


def main() -> None:
    """Build AFH1 v2 using pelvis translation plus torso-vector alignment."""
    easy = np.load(EASYERGO_NPZ, allow_pickle=True)
    stereo = np.load(STEREO_POSE_NPZ, allow_pickle=True)
    with ROTATION_JSON.open("r", encoding="utf-8") as handle:
        alignment = json.load(handle)

    base_rotation = np.asarray(alignment["rotation_3x3"], dtype=np.float64)

    stereo_ts_abs = stereo["timestamps"].astype(np.float64)
    stereo_ts_rel = stereo_ts_abs - stereo_ts_abs[0]
    stereo_kpts = stereo["keypoints"].astype(np.float64)

    easy_ts = easy["timestamps"].astype(np.float64)
    easy_kpts = easy["keypoints_3d"].astype(np.float64)
    easy_interp = interp1d(
        easy_ts,
        easy_kpts,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )(stereo_ts_rel)

    stereo_pelvis = midpoint(stereo_kpts, LEFT_HIP, RIGHT_HIP)
    stereo_shoulder = midpoint(stereo_kpts, LEFT_SHOULDER, RIGHT_SHOULDER)
    easy_pelvis = midpoint(easy_interp, LEFT_HIP, RIGHT_HIP)
    easy_shoulder = midpoint(easy_interp, LEFT_SHOULDER, RIGHT_SHOULDER)

    easy_rel = easy_interp - easy_pelvis[:, None, :]
    easy_rel_base = np.einsum("ij,tkj->tki", base_rotation, easy_rel)
    easy_torso_base = np.einsum("ij,tj->ti", base_rotation, easy_shoulder - easy_pelvis)
    stereo_torso = stereo_shoulder - stereo_pelvis

    raw_scale = np.full(len(stereo_ts_abs), np.nan, dtype=np.float64)
    torso_valid = np.isfinite(easy_torso_base).all(axis=1) & np.isfinite(stereo_torso).all(axis=1)
    for idx in np.flatnonzero(torso_valid):
        easy_len = np.linalg.norm(easy_torso_base[idx])
        stereo_len = np.linalg.norm(stereo_torso[idx])
        if easy_len > 1e-8 and stereo_len > 1e-8:
            raw_scale[idx] = np.clip(stereo_len / easy_len, SCALE_MIN, SCALE_MAX)

    smooth_scale = median_filter_series(raw_scale, SCALE_RADIUS)
    global_scale = float(np.nanmedian(smooth_scale[np.isfinite(smooth_scale)]))
    if not np.isfinite(global_scale):
        global_scale = 1.0

    hybrid = np.full_like(easy_rel_base, np.nan)
    dynamic_rotation_used = np.zeros(len(stereo_ts_abs), dtype=bool)
    scale_used = np.full(len(stereo_ts_abs), np.nan, dtype=np.float64)

    rel_valid = np.isfinite(easy_rel_base).all(axis=2)
    pelvis_valid = np.isfinite(stereo_pelvis).all(axis=1)
    for idx in range(len(hybrid)):
        if not pelvis_valid[idx]:
            continue

        scale = smooth_scale[idx] if np.isfinite(smooth_scale[idx]) else global_scale
        dyn_rotation = np.eye(3)
        if torso_valid[idx]:
            dyn_rotation = vector_alignment_rotation(easy_torso_base[idx], stereo_torso[idx])
            dynamic_rotation_used[idx] = True

        corrected_rel = scale * np.einsum("ij,kj->ki", dyn_rotation, easy_rel_base[idx])
        valid_joints = rel_valid[idx]
        hybrid[idx, valid_joints, :] = stereo_pelvis[idx][None, :] + corrected_rel[valid_joints]
        scale_used[idx] = scale

    np.savez(
        HYBRID_V2_NPZ,
        timestamps=stereo_ts_abs,
        keypoints=hybrid,
        source_method="AFH1_v2_pelvis_torso",
        units="cm",
        base_rotation_3x3=base_rotation,
        per_frame_scale=scale_used,
        dynamic_rotation_used=dynamic_rotation_used,
        notes="Pelvis translation + per-frame torso vector alignment + torso-length scale",
    )

    valid_joint_mask = np.isfinite(hybrid).all(axis=2)
    summary = {
        "output_npz_path": str(HYBRID_V2_NPZ),
        "num_frames": int(len(stereo_ts_abs)),
        "frame_valid_any_joint_ratio": float(np.mean(np.any(valid_joint_mask, axis=1))),
        "dynamic_rotation_frame_ratio": float(np.mean(dynamic_rotation_used)),
        "median_scale": float(np.nanmedian(scale_used)),
        "scale_p05": float(np.nanpercentile(scale_used, 5)),
        "scale_p95": float(np.nanpercentile(scale_used, 95)),
        "median_bone_lengths_cm": {
            "shoulder_width": float(
                np.nanmedian(np.linalg.norm(hybrid[:, 5, :] - hybrid[:, 6, :], axis=1))
            ),
            "hip_width": float(
                np.nanmedian(np.linalg.norm(hybrid[:, 11, :] - hybrid[:, 12, :], axis=1))
            ),
            "left_thigh": float(
                np.nanmedian(np.linalg.norm(hybrid[:, 11, :] - hybrid[:, 13, :], axis=1))
            ),
            "right_thigh": float(
                np.nanmedian(np.linalg.norm(hybrid[:, 12, :] - hybrid[:, 14, :], axis=1))
            ),
        },
    }

    with HYBRID_V2_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_experiment_log(
        "Built AFH1 v2 with pelvis + torso anchors. "
        f"Median scale {summary['median_scale']:.3f}, dynamic rotation used on "
        f"{summary['dynamic_rotation_frame_ratio']:.3f} of frames."
    )

    print(f"[saved] {HYBRID_V2_NPZ}")
    print(f"[saved] {HYBRID_V2_SUMMARY_JSON}")
    print(
        "[info] dynamic rotation usage: "
        f"{summary['dynamic_rotation_frame_ratio']:.3f}, "
        f"median scale={summary['median_scale']:.3f}"
    )


if __name__ == "__main__":
    main()
