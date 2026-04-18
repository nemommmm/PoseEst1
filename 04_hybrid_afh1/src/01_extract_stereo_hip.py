"""Extract the stereo pelvis anchor from the historical best stereo NPZ."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AFH1_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from pose_postprocess import LEFT_HIP, RIGHT_HIP


SOURCE_POSE_NPZ = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
RESULTS_DIR = AFH1_DIR / "results"
ANCHOR_OUT = RESULTS_DIR / "stereo_pelvis_anchor.npz"
SUMMARY_OUT = RESULTS_DIR / "stereo_pelvis_anchor_summary.json"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.md"


def append_experiment_log(message: str) -> None:
    """Append one line to the AFH1 experiment log."""
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- {message}\n")


def main() -> None:
    """Load stereo 3D keypoints and save the pelvis anchor sequence."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_POSE_NPZ.is_file():
        raise FileNotFoundError(f"Stereo pose NPZ not found: {SOURCE_POSE_NPZ}")

    data = np.load(SOURCE_POSE_NPZ)
    timestamps_abs = np.asarray(data["timestamps"], dtype=np.float64)
    keypoints = np.asarray(data["keypoints"], dtype=np.float64)

    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(
            f"Unexpected keypoint shape: {keypoints.shape}. Expected (N, 17, 3)."
        )

    timestamps_rel = timestamps_abs - timestamps_abs[0] if len(timestamps_abs) else timestamps_abs
    left_hip = keypoints[:, LEFT_HIP, :]
    right_hip = keypoints[:, RIGHT_HIP, :]
    valid_mask = np.isfinite(left_hip).all(axis=1) & np.isfinite(right_hip).all(axis=1)

    pelvis = np.full_like(left_hip, np.nan)
    pelvis[valid_mask] = 0.5 * (left_hip[valid_mask] + right_hip[valid_mask])
    pelvis_valid = pelvis[valid_mask]

    np.savez(
        ANCHOR_OUT,
        timestamps_abs=timestamps_abs,
        timestamps_rel=timestamps_rel,
        pelvis_xyz_cm=pelvis,
        left_hip_xyz_cm=left_hip,
        right_hip_xyz_cm=right_hip,
        valid_mask=valid_mask,
        source_pose_path=str(SOURCE_POSE_NPZ),
        units="cm",
    )

    summary = {
        "source_pose_path": str(SOURCE_POSE_NPZ),
        "output_npz_path": str(ANCHOR_OUT),
        "num_frames": int(len(timestamps_abs)),
        "num_valid_pelvis_frames": int(np.count_nonzero(valid_mask)),
        "valid_ratio": float(np.mean(valid_mask)) if len(valid_mask) else 0.0,
        "first_timestamp_abs": float(timestamps_abs[0]) if len(timestamps_abs) else None,
        "last_timestamp_abs": float(timestamps_abs[-1]) if len(timestamps_abs) else None,
        "median_dt_s": float(np.median(np.diff(timestamps_abs))) if len(timestamps_abs) > 1 else None,
        "pelvis_xyz_cm_min": np.nanmin(pelvis_valid, axis=0).tolist(),
        "pelvis_xyz_cm_max": np.nanmax(pelvis_valid, axis=0).tolist(),
        "pelvis_xyz_cm_mean": np.nanmean(pelvis_valid, axis=0).tolist(),
        "pelvis_xyz_cm_median": np.nanmedian(pelvis_valid, axis=0).tolist(),
        "pelvis_xyz_cm_p05": np.nanpercentile(pelvis_valid, 5, axis=0).tolist(),
        "pelvis_xyz_cm_p95": np.nanpercentile(pelvis_valid, 95, axis=0).tolist(),
    }

    with SUMMARY_OUT.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_experiment_log(
        "Extracted stereo pelvis anchor from historical best optimized_pose.npz "
        f"({summary['num_valid_pelvis_frames']}/{summary['num_frames']} valid frames)."
    )

    print(f"[saved] {ANCHOR_OUT}")
    print(f"[saved] {SUMMARY_OUT}")
    print(
        "[info] valid pelvis frames: "
        f"{summary['num_valid_pelvis_frames']}/{summary['num_frames']} "
        f"({summary['valid_ratio']:.3f})"
    )


if __name__ == "__main__":
    main()
