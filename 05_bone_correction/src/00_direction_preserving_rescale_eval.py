"""Evaluate whether direction-preserving bone-length rescaling changes angle MAE.

This is a sanity-check for the bone-length hypothesis. It keeps the original
stereo bone directions, replaces segment lengths with target lengths, then
recomputes semantic joint angles and compares them against Xsens.

If angle MAE barely changes, then a pure rescale is not enough and P1 should be
redefined as a constrained kinematic correction problem instead.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))

from pose_angle_utils import (  # noqa: E402
    SEMANTIC_ANGLE_NAMES,
    build_fair_gt_interpolators,
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
    median_filter_angle_sequence,
)
from utils_mvnx import MvnxParser  # noqa: E402


STEREO_NPZ = os.environ.get(
    "POSE_INPUT_FILENAME",
    os.path.join(
        PROJECT_ROOT,
        "01_stereo_triangulation",
        "results",
        "historical_best_20260324",
        "recovered_baseline",
        "optimized_pose.npz",
    ),
)
EASYERGO_NPZ = os.environ.get(
    "POSE_EASYERGO_NPZ",
    os.path.join(PROJECT_ROOT, "04_hybrid_afh1", "results", "easyergo_normalized.npz"),
)
ALIGNMENT_JSON = os.environ.get(
    "POSE_ALIGNMENT_SUMMARY_NAME",
    os.path.join(
        PROJECT_ROOT,
        "01_stereo_triangulation",
        "results",
        "historical_best_20260324",
        "recovered_baseline",
        "alignment.json",
    ),
)
GT_MVNX = os.environ.get(
    "POSE_GT_MVNX",
    os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx"),
)
FAIR_GT_NPZ = os.environ.get(
    "POSE_FAIR_GT_NPZ",
    os.path.join(PROJECT_ROOT, "shared", "fair_gt_angles.npz"),
)
RESULTS_DIR = os.environ.get(
    "POSE_RESULTS_DIR",
    os.path.join(PROJECT_ROOT, "05_bone_correction", "results"),
)
ANGLE_SMOOTH_RADIUS = int(os.environ.get("POSE_ANGLE_SMOOTH_RADIUS", "8"))


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


def safe_unit(vec: np.ndarray) -> np.ndarray | None:
    """Return a unit vector, or None for invalid/degenerate inputs."""
    if vec is None or not np.isfinite(vec).all():
        return None
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


def segment_length(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame Euclidean distance between two joint trajectories."""
    return np.linalg.norm(a - b, axis=1)


def compute_target_lengths_from_easyergo(npz_path: str) -> Dict[str, float]:
    """Estimate target segment lengths from the EasyErgo skeleton."""
    data = np.load(npz_path)
    kpts = data["keypoints_3d"].astype(np.float64)
    hip_mid = 0.5 * (kpts[:, LEFT_HIP] + kpts[:, RIGHT_HIP])
    shoulder_mid = 0.5 * (kpts[:, LEFT_SHOULDER] + kpts[:, RIGHT_SHOULDER])

    targets = {
        "hip_width": float(np.nanmedian(segment_length(kpts[:, LEFT_HIP], kpts[:, RIGHT_HIP]))),
        "shoulder_width": float(np.nanmedian(segment_length(kpts[:, LEFT_SHOULDER], kpts[:, RIGHT_SHOULDER]))),
        "torso": float(np.nanmedian(segment_length(hip_mid, shoulder_mid))),
        "upper_arm": float(
            np.nanmedian(
                np.concatenate(
                    [
                        segment_length(kpts[:, LEFT_SHOULDER], kpts[:, LEFT_ELBOW]),
                        segment_length(kpts[:, RIGHT_SHOULDER], kpts[:, RIGHT_ELBOW]),
                    ]
                )
            )
        ),
        "forearm": float(
            np.nanmedian(
                np.concatenate(
                    [
                        segment_length(kpts[:, LEFT_ELBOW], kpts[:, LEFT_WRIST]),
                        segment_length(kpts[:, RIGHT_ELBOW], kpts[:, RIGHT_WRIST]),
                    ]
                )
            )
        ),
        "thigh": float(
            np.nanmedian(
                np.concatenate(
                    [
                        segment_length(kpts[:, LEFT_HIP], kpts[:, LEFT_KNEE]),
                        segment_length(kpts[:, RIGHT_HIP], kpts[:, RIGHT_KNEE]),
                    ]
                )
            )
        ),
        "shank": float(
            np.nanmedian(
                np.concatenate(
                    [
                        segment_length(kpts[:, LEFT_KNEE], kpts[:, LEFT_ANKLE]),
                        segment_length(kpts[:, RIGHT_KNEE], kpts[:, RIGHT_ANKLE]),
                    ]
                )
            )
        ),
    }
    return targets


def direction_preserving_rescale(kpts: np.ndarray, targets: Dict[str, float]) -> np.ndarray:
    """Rebuild a skeleton with target lengths while preserving local directions."""
    out = np.array(kpts, dtype=np.float64, copy=True)

    for frame_idx, pose in enumerate(kpts):
        hip_mid = 0.5 * (pose[LEFT_HIP] + pose[RIGHT_HIP])
        shoulder_mid = 0.5 * (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER])

        hip_lat = safe_unit(pose[RIGHT_HIP] - pose[LEFT_HIP])
        shoulder_lat = safe_unit(pose[RIGHT_SHOULDER] - pose[LEFT_SHOULDER])
        torso_dir = safe_unit(shoulder_mid - hip_mid)
        l_upper = safe_unit(pose[LEFT_ELBOW] - pose[LEFT_SHOULDER])
        r_upper = safe_unit(pose[RIGHT_ELBOW] - pose[RIGHT_SHOULDER])
        l_fore = safe_unit(pose[LEFT_WRIST] - pose[LEFT_ELBOW])
        r_fore = safe_unit(pose[RIGHT_WRIST] - pose[RIGHT_ELBOW])
        l_thigh = safe_unit(pose[LEFT_KNEE] - pose[LEFT_HIP])
        r_thigh = safe_unit(pose[RIGHT_KNEE] - pose[RIGHT_HIP])
        l_shank = safe_unit(pose[LEFT_ANKLE] - pose[LEFT_KNEE])
        r_shank = safe_unit(pose[RIGHT_ANKLE] - pose[RIGHT_KNEE])

        required = [
            hip_lat,
            shoulder_lat,
            torso_dir,
            l_upper,
            r_upper,
            l_fore,
            r_fore,
            l_thigh,
            r_thigh,
            l_shank,
            r_shank,
        ]
        if any(vec is None for vec in required):
            continue

        new_hip_mid = hip_mid
        new_l_hip = new_hip_mid - 0.5 * targets["hip_width"] * hip_lat
        new_r_hip = new_hip_mid + 0.5 * targets["hip_width"] * hip_lat
        new_shoulder_mid = new_hip_mid + targets["torso"] * torso_dir
        new_l_shoulder = new_shoulder_mid - 0.5 * targets["shoulder_width"] * shoulder_lat
        new_r_shoulder = new_shoulder_mid + 0.5 * targets["shoulder_width"] * shoulder_lat

        new_l_elbow = new_l_shoulder + targets["upper_arm"] * l_upper
        new_r_elbow = new_r_shoulder + targets["upper_arm"] * r_upper
        new_l_wrist = new_l_elbow + targets["forearm"] * l_fore
        new_r_wrist = new_r_elbow + targets["forearm"] * r_fore
        new_l_knee = new_l_hip + targets["thigh"] * l_thigh
        new_r_knee = new_r_hip + targets["thigh"] * r_thigh
        new_l_ankle = new_l_knee + targets["shank"] * l_shank
        new_r_ankle = new_r_knee + targets["shank"] * r_shank

        out[frame_idx, LEFT_HIP] = new_l_hip
        out[frame_idx, RIGHT_HIP] = new_r_hip
        out[frame_idx, LEFT_SHOULDER] = new_l_shoulder
        out[frame_idx, RIGHT_SHOULDER] = new_r_shoulder
        out[frame_idx, LEFT_ELBOW] = new_l_elbow
        out[frame_idx, RIGHT_ELBOW] = new_r_elbow
        out[frame_idx, LEFT_WRIST] = new_l_wrist
        out[frame_idx, RIGHT_WRIST] = new_r_wrist
        out[frame_idx, LEFT_KNEE] = new_l_knee
        out[frame_idx, RIGHT_KNEE] = new_r_knee
        out[frame_idx, LEFT_ANKLE] = new_l_ankle
        out[frame_idx, RIGHT_ANKLE] = new_r_ankle

    return out


def load_best_offset(path: str) -> float:
    """Load temporal alignment offset in seconds."""
    with open(path, "r", encoding="utf-8") as handle:
        return float(json.load(handle)["best_offset_seconds"])


def evaluate_angle_mae(keypoints: np.ndarray, timestamps: np.ndarray, offset_s: float) -> dict:
    """Compute end-to-end and fair-GT angle MAE for a keypoint sequence."""
    mvnx = MvnxParser(GT_MVNX)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, unique_idx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, unique_idx)
    fair_interps = build_fair_gt_interpolators(FAIR_GT_NPZ)

    _, angle_values = compute_semantic_angle_sequence(keypoints)
    angle_values = median_filter_angle_sequence(angle_values, radius=ANGLE_SMOOTH_RADIUS)

    e2e_errors = []
    fair_errors = []
    per_angle_delta = {name: [] for name in SEMANTIC_ANGLE_NAMES}

    for frame_idx, curr_t in enumerate(timestamps):
        target_t = float(curr_t - offset_s)
        for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
            est_val = float(angle_values[frame_idx, angle_idx])
            if not np.isfinite(est_val) or angle_name not in gt_interps:
                continue
            gt_val = float(gt_interps[angle_name](target_t))
            if np.isfinite(gt_val):
                e2e_errors.append(abs(est_val - gt_val))
            if angle_name in fair_interps:
                fair_val = float(fair_interps[angle_name](target_t))
                if np.isfinite(fair_val):
                    fair_errors.append(abs(est_val - fair_val))

    return {
        "joint_angle_mae": float(np.mean(e2e_errors)) if e2e_errors else np.nan,
        "fair_angle_mae": float(np.mean(fair_errors)) if fair_errors else np.nan,
    }


def summarize_angle_change(original_kpts: np.ndarray, rescaled_kpts: np.ndarray) -> dict:
    """Measure how much semantic angles change before GT comparison."""
    _, angles_orig = compute_semantic_angle_sequence(original_kpts)
    _, angles_new = compute_semantic_angle_sequence(rescaled_kpts)
    angles_orig = median_filter_angle_sequence(angles_orig, radius=ANGLE_SMOOTH_RADIUS)
    angles_new = median_filter_angle_sequence(angles_new, radius=ANGLE_SMOOTH_RADIUS)
    deltas = np.abs(angles_new - angles_orig)

    summary = {}
    for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
        col = deltas[:, angle_idx]
        finite = col[np.isfinite(col)]
        summary[angle_name] = {
            "mean_abs_delta_deg": float(np.mean(finite)) if finite.size else np.nan,
            "p95_abs_delta_deg": float(np.percentile(finite, 95)) if finite.size else np.nan,
        }
    return summary


def main() -> None:
    """Run the sanity-check and write a compact summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stereo = np.load(STEREO_NPZ)
    kpts = stereo["keypoints"].astype(np.float64)
    timestamps = stereo["timestamps"].astype(float)
    timestamps = timestamps - timestamps[0]
    offset_s = load_best_offset(ALIGNMENT_JSON)
    targets = compute_target_lengths_from_easyergo(EASYERGO_NPZ)

    rescaled = direction_preserving_rescale(kpts, targets)

    orig_metrics = evaluate_angle_mae(kpts, timestamps, offset_s)
    rescaled_metrics = evaluate_angle_mae(rescaled, timestamps, offset_s)
    angle_change = summarize_angle_change(kpts, rescaled)

    summary = {
        "targets_cm": targets,
        "original_metrics": orig_metrics,
        "rescaled_metrics": rescaled_metrics,
        "delta_joint_angle_mae_deg": rescaled_metrics["joint_angle_mae"] - orig_metrics["joint_angle_mae"],
        "delta_fair_angle_mae_deg": rescaled_metrics["fair_angle_mae"] - orig_metrics["fair_angle_mae"],
        "per_angle_change": angle_change,
    }

    json_path = os.path.join(RESULTS_DIR, "direction_preserving_rescale_summary.json")
    md_path = os.path.join(RESULTS_DIR, "direction_preserving_rescale_summary.md")
    np.savez(
        os.path.join(RESULTS_DIR, "direction_preserving_rescaled_pose.npz"),
        timestamps=stereo["timestamps"],
        keypoints=rescaled,
    )
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Direction-Preserving Bone Rescale Sanity Check\n\n")
        handle.write("## Target lengths (cm)\n\n")
        for name, value in targets.items():
            handle.write(f"- {name}: {value:.2f}\n")
        handle.write("\n## MAE comparison\n\n")
        handle.write(f"- Original end-to-end MAE: {orig_metrics['joint_angle_mae']:.2f} deg\n")
        handle.write(f"- Rescaled end-to-end MAE: {rescaled_metrics['joint_angle_mae']:.2f} deg\n")
        handle.write(f"- Delta end-to-end MAE: {summary['delta_joint_angle_mae_deg']:+.2f} deg\n")
        handle.write(f"- Original fair MAE: {orig_metrics['fair_angle_mae']:.2f} deg\n")
        handle.write(f"- Rescaled fair MAE: {rescaled_metrics['fair_angle_mae']:.2f} deg\n")
        handle.write(f"- Delta fair MAE: {summary['delta_fair_angle_mae_deg']:+.2f} deg\n")
        handle.write("\n## Per-angle absolute change caused by rescale\n\n")
        handle.write("| Angle | Mean abs delta (deg) | P95 abs delta (deg) |\n")
        handle.write("|-------|----------------------|---------------------|\n")
        for angle_name, stats in angle_change.items():
            handle.write(
                f"| {angle_name} | {stats['mean_abs_delta_deg']:.2f} | {stats['p95_abs_delta_deg']:.2f} |\n"
            )

    print(f"[info] saved: {json_path}")
    print(f"[info] saved: {md_path}")


if __name__ == "__main__":
    main()
