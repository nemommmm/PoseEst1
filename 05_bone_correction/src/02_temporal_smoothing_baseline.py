"""Benchmark simple joint-level temporal smoothing baselines for stereo SKT.

This script evaluates a few cheap coordinate-domain smoothing variants before
any more complex kinematic correction is attempted.
"""

from __future__ import annotations

import csv
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

BONE_SPECS = {
    "shoulder_width": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "hip_width": (LEFT_HIP, RIGHT_HIP),
    "upper_arm_left": (LEFT_SHOULDER, LEFT_ELBOW),
    "upper_arm_right": (RIGHT_SHOULDER, RIGHT_ELBOW),
    "forearm_left": (LEFT_ELBOW, LEFT_WRIST),
    "forearm_right": (RIGHT_ELBOW, RIGHT_WRIST),
    "thigh_left": (LEFT_HIP, LEFT_KNEE),
    "thigh_right": (RIGHT_HIP, RIGHT_KNEE),
    "shank_left": (LEFT_KNEE, LEFT_ANKLE),
    "shank_right": (RIGHT_KNEE, RIGHT_ANKLE),
}


def load_best_offset(path: str) -> float:
    """Load pose/Xsens alignment offset in seconds."""
    with open(path, "r", encoding="utf-8") as handle:
        return float(json.load(handle)["best_offset_seconds"])


def median_filter_1d_nan(values: np.ndarray, radius: int) -> np.ndarray:
    """Median filter a 1D array while ignoring NaN values."""
    if radius <= 0:
        return values.copy()
    out = np.array(values, dtype=np.float64, copy=True)
    n = len(out)
    for idx in range(n):
        lo = max(0, idx - radius)
        hi = min(n, idx + radius + 1)
        finite = out[lo:hi][np.isfinite(out[lo:hi])]
        if finite.size:
            out[idx] = float(np.median(finite))
    return out


def smooth_keypoints(kpts: np.ndarray, radius: int) -> np.ndarray:
    """Apply per-joint, per-axis temporal median smoothing."""
    out = np.array(kpts, dtype=np.float64, copy=True)
    for joint_idx in range(out.shape[1]):
        for axis in range(out.shape[2]):
            out[:, joint_idx, axis] = median_filter_1d_nan(out[:, joint_idx, axis], radius=radius)
    return out


def reject_spikes(kpts: np.ndarray, sigma_scale: float = 3.0) -> tuple[np.ndarray, int]:
    """Replace extreme frame-to-frame jumps with NaN before smoothing."""
    out = np.array(kpts, dtype=np.float64, copy=True)
    replaced = 0

    for joint_idx in range(out.shape[1]):
        seq = out[:, joint_idx]
        diffs = np.linalg.norm(np.diff(seq, axis=0), axis=1)
        finite_diffs = diffs[np.isfinite(diffs)]
        if finite_diffs.size < 10:
            continue
        med = float(np.median(finite_diffs))
        mad = float(np.median(np.abs(finite_diffs - med)))
        threshold = med + sigma_scale * max(mad * 1.4826, 1.0)
        spike_idx = np.where(diffs > threshold)[0] + 1
        for idx in spike_idx:
            if np.isfinite(out[idx, joint_idx]).all():
                out[idx, joint_idx] = np.nan
                replaced += 1

    return out, replaced


def bone_length_std_summary(kpts: np.ndarray) -> tuple[Dict[str, float], float]:
    """Compute per-bone std and the mean std across tracked bones."""
    stats = {}
    for name, (a, b) in BONE_SPECS.items():
        dist = np.linalg.norm(kpts[:, a] - kpts[:, b], axis=1)
        dist = dist[np.isfinite(dist)]
        stats[name] = float(np.std(dist)) if dist.size else np.nan
    finite = [v for v in stats.values() if np.isfinite(v)]
    mean_std = float(np.mean(finite)) if finite else np.nan
    return stats, mean_std


def evaluate_angles(keypoints: np.ndarray, timestamps: np.ndarray, offset_s: float) -> Dict[str, float]:
    """Compute raw/fair angle MAE and elbow RULA accuracy for one keypoint sequence."""
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
    elbow_matches = []

    for frame_idx, curr_t in enumerate(timestamps):
        target_t = float(curr_t - offset_s)
        for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
            est_val = float(angle_values[frame_idx, angle_idx])
            if not np.isfinite(est_val) or angle_name not in gt_interps:
                continue
            gt_val = float(gt_interps[angle_name](target_t))
            if np.isfinite(gt_val):
                e2e_errors.append(abs(est_val - gt_val))
                if "Elbow" in angle_name:
                    est_abs = abs(est_val)
                    gt_abs = abs(gt_val)
                    est_class = 0 if est_abs < 60 else (1 if est_abs < 100 else 2)
                    gt_class = 0 if gt_abs < 60 else (1 if gt_abs < 100 else 2)
                    elbow_matches.append(int(est_class == gt_class))
            if angle_name in fair_interps:
                fair_val = float(fair_interps[angle_name](target_t))
                if np.isfinite(fair_val):
                    fair_errors.append(abs(est_val - fair_val))

    return {
        "joint_angle_mae": float(np.mean(e2e_errors)) if e2e_errors else np.nan,
        "fair_angle_mae": float(np.mean(fair_errors)) if fair_errors else np.nan,
        "elbow_rula_accuracy": float(np.mean(elbow_matches)) if elbow_matches else np.nan,
    }


def run_variant(name: str, kpts: np.ndarray, timestamps: np.ndarray, offset_s: float) -> Dict[str, object]:
    """Evaluate one smoothing variant."""
    metrics = evaluate_angles(kpts, timestamps, offset_s)
    bone_std_by_name, mean_bone_std = bone_length_std_summary(kpts)
    return {
        "variant": name,
        "joint_angle_mae": metrics["joint_angle_mae"],
        "fair_angle_mae": metrics["fair_angle_mae"],
        "elbow_rula_accuracy": metrics["elbow_rula_accuracy"],
        "mean_bone_std_cm": mean_bone_std,
        "bone_std_by_name": bone_std_by_name,
    }


def save_npz_from_template(path: str, template_path: str, keypoints: np.ndarray) -> None:
    """Save a new NPZ while preserving other metadata fields from the template."""
    template = np.load(template_path)
    payload = {key: template[key] for key in template.files}
    payload["keypoints"] = keypoints
    np.savez(path, **payload)


def main() -> None:
    """Run the smoothing benchmark and save the best candidate."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = np.load(STEREO_NPZ)
    kpts = data["keypoints"].astype(np.float64)
    timestamps = data["timestamps"].astype(float)
    timestamps = timestamps - timestamps[0]
    offset_s = load_best_offset(ALIGNMENT_JSON)

    variants = []
    variants.append((run_variant("raw", kpts, timestamps, offset_s), kpts))

    med3 = smooth_keypoints(kpts, radius=3)
    variants.append((run_variant("median_r3", med3, timestamps, offset_s), med3))

    med5 = smooth_keypoints(kpts, radius=5)
    variants.append((run_variant("median_r5", med5, timestamps, offset_s), med5))

    spike_clean, replaced3 = reject_spikes(kpts, sigma_scale=3.0)
    spike_med3 = smooth_keypoints(spike_clean, radius=3)
    variant_spike3 = run_variant("spike3_then_median_r3", spike_med3, timestamps, offset_s)
    variant_spike3["replaced_joint_frames"] = replaced3
    variants.append((variant_spike3, spike_med3))

    spike_clean5, replaced5 = reject_spikes(kpts, sigma_scale=3.0)
    spike_med5 = smooth_keypoints(spike_clean5, radius=5)
    variant_spike5 = run_variant("spike3_then_median_r5", spike_med5, timestamps, offset_s)
    variant_spike5["replaced_joint_frames"] = replaced5
    variants.append((variant_spike5, spike_med5))

    rows = [item[0] for item in variants]
    rows.sort(key=lambda row: float(row["joint_angle_mae"]))
    best_name = rows[0]["variant"]
    best_kpts = next(k for r, k in variants if r["variant"] == best_name)

    csv_path = os.path.join(RESULTS_DIR, "temporal_smoothing_benchmark.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "joint_angle_mae",
                "fair_angle_mae",
                "elbow_rula_accuracy",
                "mean_bone_std_cm",
                "replaced_joint_frames",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["variant"],
                    row["joint_angle_mae"],
                    row["fair_angle_mae"],
                    row["elbow_rula_accuracy"],
                    row["mean_bone_std_cm"],
                    row.get("replaced_joint_frames", 0),
                ]
            )

    save_npz_from_template(
        os.path.join(RESULTS_DIR, f"{best_name}.npz"),
        STEREO_NPZ,
        best_kpts,
    )

    summary_path = os.path.join(RESULTS_DIR, "temporal_smoothing_benchmark.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# Temporal Smoothing Baseline Benchmark\n\n")
        handle.write("| Variant | Angle MAE | Fair MAE | Elbow RULA | Mean bone std (cm) | Replaced joint-frames |\n")
        handle.write("|---------|-----------|----------|------------|--------------------|-----------------------|\n")
        for row in rows:
            handle.write(
                f"| {row['variant']} | {row['joint_angle_mae']:.2f} | {row['fair_angle_mae']:.2f} | "
                f"{row['elbow_rula_accuracy']:.4f} | {row['mean_bone_std_cm']:.2f} | "
                f"{int(row.get('replaced_joint_frames', 0))} |\n"
            )
        handle.write(f"\n- Best variant by raw angle MAE: `{best_name}`\n")

    log_path = os.path.join(RESULTS_DIR, "experiment_log.md")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("\n## 2026-04-19 - Temporal Smoothing Benchmark\n\n")
        handle.write(f"- Best variant: `{best_name}`\n")
        for row in rows:
            handle.write(
                f"- `{row['variant']}`: MAE={row['joint_angle_mae']:.2f}°, "
                f"fair={row['fair_angle_mae']:.2f}°, elbow RULA={row['elbow_rula_accuracy']:.4f}, "
                f"mean bone std={row['mean_bone_std_cm']:.2f} cm\n"
            )

    print(f"[info] saved: {csv_path}")
    print(f"[info] saved: {summary_path}")
    print(f"[info] saved: {os.path.join(RESULTS_DIR, f'{best_name}.npz')}")


if __name__ == "__main__":
    main()
