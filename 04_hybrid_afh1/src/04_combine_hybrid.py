"""Combine EasyErgo relative skeletons with the stereo pelvis anchor."""

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
STEREO_ANCHOR_NPZ = RESULTS_DIR / "stereo_pelvis_anchor.npz"
ROTATION_JSON = RESULTS_DIR / "coordinate_alignment.json"
HYBRID_NPZ = RESULTS_DIR / "hybrid_skeleton_afh1_v1.npz"
HYBRID_SUMMARY_JSON = RESULTS_DIR / "hybrid_skeleton_afh1_v1_summary.json"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.md"
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


def main() -> None:
    """Create AFH1 v1 skeletons in stereo coordinates."""
    easy = np.load(EASYERGO_NPZ, allow_pickle=True)
    stereo_anchor = np.load(STEREO_ANCHOR_NPZ, allow_pickle=True)
    with ROTATION_JSON.open("r", encoding="utf-8") as handle:
        alignment = json.load(handle)

    rotation = np.asarray(alignment["rotation_3x3"], dtype=np.float64)
    stereo_ts_abs = stereo_anchor["timestamps_abs"].astype(np.float64)
    stereo_ts_rel = stereo_anchor["timestamps_rel"].astype(np.float64)
    pelvis_xyz = stereo_anchor["pelvis_xyz_cm"].astype(np.float64)
    pelvis_valid = stereo_anchor["valid_mask"].astype(bool)

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

    easy_pelvis = pelvis_center(easy_interp)
    easy_rel = easy_interp - easy_pelvis[:, None, :]
    easy_rel_aligned = np.einsum("ij,tkj->tki", rotation, easy_rel)

    hybrid = np.full_like(easy_rel_aligned, np.nan)
    rel_valid = np.isfinite(easy_rel_aligned).all(axis=2)
    for frame_idx in range(len(hybrid)):
        if not pelvis_valid[frame_idx] or not np.isfinite(pelvis_xyz[frame_idx]).all():
            continue
        valid_joints = rel_valid[frame_idx]
        hybrid[frame_idx, valid_joints, :] = (
            pelvis_xyz[frame_idx][None, :] + easy_rel_aligned[frame_idx, valid_joints, :]
        )

    np.savez(
        HYBRID_NPZ,
        timestamps=stereo_ts_abs,
        keypoints=hybrid,
        source_method="AFH1_v1_time_aligned",
        units="cm",
        rotation_3x3=rotation,
        easyergo_query_timestamps=easy_query_ts,
        easyergo_to_xsens_scale=DEFAULT_EASYERGO_TO_XSENS_SCALE,
        easyergo_to_xsens_offset_s=DEFAULT_EASYERGO_TO_XSENS_OFFSET_S,
        stereo_to_xsens_offset_s=DEFAULT_STEREO_TO_XSENS_OFFSET_S,
        stereo_anchor_path=str(STEREO_ANCHOR_NPZ),
        easyergo_source_path=str(EASYERGO_NPZ),
    )

    valid_joint_mask = np.isfinite(hybrid).all(axis=2)
    joint_coverage = {
        int(idx): float(np.mean(valid_joint_mask[:, idx])) for idx in range(hybrid.shape[1])
    }
    summary = {
        "output_npz_path": str(HYBRID_NPZ),
        "num_frames": int(len(stereo_ts_abs)),
        "num_valid_pelvis_frames": int(np.count_nonzero(pelvis_valid)),
        "frame_valid_any_joint_ratio": float(np.mean(np.any(valid_joint_mask, axis=1))),
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
            "query_out_of_bounds_frame_count": int(
                np.sum(
                    (easy_query_ts < float(np.nanmin(easy_ts)))
                    | (easy_query_ts > float(np.nanmax(easy_ts)))
                )
            ),
        },
        "joint_coverage": joint_coverage,
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

    with HYBRID_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_experiment_log(
        "Combined stereo pelvis anchor with time-aligned EasyErgo relative skeleton and saved "
        f"AFH1 v1 hybrid NPZ ({summary['num_frames']} frames)."
    )

    print(f"[saved] {HYBRID_NPZ}")
    print(f"[saved] {HYBRID_SUMMARY_JSON}")
    print(
        "[info] frame ratio with at least one valid joint: "
        f"{summary['frame_valid_any_joint_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
