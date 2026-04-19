"""Build a trunk-only hybrid pose using EasyErgo torso direction on SKT."""

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
TRUNK_ONLY_NPZ = RESULTS_DIR / "hybrid_trunk_only_afh1_v1.npz"
TRUNK_ONLY_SUMMARY_JSON = RESULTS_DIR / "hybrid_trunk_only_afh1_v1_summary.json"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.md"

UPPER_BODY_JOINTS = list(range(0, 11))
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


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


def main() -> None:
    """Rotate only the upper body to match EasyErgo torso direction."""
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

    easy_pelvis = midpoint(easy_interp, LEFT_HIP, RIGHT_HIP)
    easy_shoulder = midpoint(easy_interp, LEFT_SHOULDER, RIGHT_SHOULDER)
    easy_torso = easy_shoulder - easy_pelvis
    easy_torso_aligned = np.einsum("ij,tj->ti", base_rotation, easy_torso)

    stereo_pelvis = midpoint(stereo_kpts, LEFT_HIP, RIGHT_HIP)
    stereo_shoulder = midpoint(stereo_kpts, LEFT_SHOULDER, RIGHT_SHOULDER)
    stereo_torso = stereo_shoulder - stereo_pelvis

    hybrid = np.array(stereo_kpts, dtype=np.float64, copy=True)
    valid_frame_mask = np.zeros(len(stereo_ts_abs), dtype=bool)
    torso_angle_delta_deg = np.full(len(stereo_ts_abs), np.nan, dtype=np.float64)

    for idx in range(len(hybrid)):
        pelvis = stereo_pelvis[idx]
        src_vec = stereo_torso[idx]
        tgt_vec = easy_torso_aligned[idx]
        if not (
            np.isfinite(pelvis).all()
            and np.isfinite(src_vec).all()
            and np.isfinite(tgt_vec).all()
        ):
            continue

        rotation = vector_alignment_rotation(src_vec, tgt_vec)
        frame_updated = False
        for joint_idx in UPPER_BODY_JOINTS:
            joint = stereo_kpts[idx, joint_idx, :]
            if not np.isfinite(joint).all():
                continue
            rel = joint - pelvis
            hybrid[idx, joint_idx, :] = pelvis + rel @ rotation.T
            frame_updated = True
        if not frame_updated:
            continue

        valid_frame_mask[idx] = True

        src_unit = src_vec / np.linalg.norm(src_vec)
        tgt_unit = tgt_vec / np.linalg.norm(tgt_vec)
        cos_val = float(np.clip(np.dot(src_unit, tgt_unit), -1.0, 1.0))
        torso_angle_delta_deg[idx] = float(np.degrees(np.arccos(cos_val)))

    np.savez(
        TRUNK_ONLY_NPZ,
        timestamps=stereo_ts_abs,
        keypoints=hybrid,
        source_method="AFH1_trunk_only_v1",
        units="cm",
        base_rotation_3x3=base_rotation,
        upper_body_joints=np.asarray(UPPER_BODY_JOINTS, dtype=np.int32),
        valid_frame_mask=valid_frame_mask,
        torso_angle_delta_deg=torso_angle_delta_deg,
        notes="Rotate SKT upper body around pelvis to match EasyErgo torso direction",
    )

    summary = {
        "output_npz_path": str(TRUNK_ONLY_NPZ),
        "num_frames": int(len(stereo_ts_abs)),
        "valid_frame_ratio": float(np.mean(valid_frame_mask)),
        "median_torso_angle_delta_deg": float(np.nanmedian(torso_angle_delta_deg)),
        "p95_torso_angle_delta_deg": float(np.nanpercentile(torso_angle_delta_deg, 95)),
        "mean_torso_angle_delta_deg": float(np.nanmean(torso_angle_delta_deg)),
    }
    with TRUNK_ONLY_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_experiment_log(
        "Built AFH1 trunk-only hybrid pose by rotating the SKT upper body around "
        f"the pelvis to match EasyErgo torso direction (valid frames "
        f"{summary['valid_frame_ratio']:.3f}, median torso delta "
        f"{summary['median_torso_angle_delta_deg']:.2f} deg)."
    )

    print(f"[saved] {TRUNK_ONLY_NPZ}")
    print(f"[saved] {TRUNK_ONLY_SUMMARY_JSON}")
    print(
        "[info] valid frame ratio="
        f"{summary['valid_frame_ratio']:.3f}, median torso delta="
        f"{summary['median_torso_angle_delta_deg']:.2f} deg"
    )


if __name__ == "__main__":
    main()
