"""Search selective quality-aware calibration subsets.

This script tests which angles should use quality-aware calibration and which
should stay on global calibration. The goal is to keep the MAE gain from the
quality-aware prototype while protecting elbow RULA accuracy.

Usage:
    /opt/anaconda3/envs/pose/bin/python 14_selective_quality_calibration_search.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import dataclass

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))

from pose_angle_utils import (  # noqa: E402
    SEMANTIC_ANGLE_NAMES,
    apply_piecewise_calibration,
    build_fair_gt_interpolators,
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
    fit_piecewise_calibration,
    median_filter_angle_sequence,
)
from utils_mvnx import MvnxParser  # noqa: E402


POSE_NPZ = os.environ.get(
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
    os.path.join(
        PROJECT_ROOT,
        "01_stereo_triangulation",
        "results",
        "selective_quality_search",
    ),
)
ANGLE_SMOOTH_RADIUS = int(os.environ.get("POSE_ANGLE_SMOOTH_RADIUS", "8"))
CAL_BINS = int(os.environ.get("POSE_CALIBRATION_BINS", "10"))
QUALITY_SPLIT_PERCENTILE = float(os.environ.get("POSE_QA_SPLIT_PERCENTILE", "35"))
MIN_FIT_SAMPLES = int(os.environ.get("POSE_QA_MIN_FIT_SAMPLES", "80"))
MAX_RULA_DROP = float(os.environ.get("POSE_QA_MAX_ELBOW_RULA_DROP", "0.01"))


ACTIVITY_SEGMENTS = {
    "Walking (Normal)": [17, 32],
    "Walking (Late)": [220, 240],
    "Sitting (Lower Occluded)": [32, 62],
    "Walking (Upper Occluded)": [87, 97],
    "Walking (Lower Occluded 1)": [130, 140],
    "Walking (Lower Occluded 2)": [164, 170],
    "Chair Interaction (Complex)": [140, 160],
    "Lifting Box (Near Chair)": [214, 218],
    "Squatting": [66, 69],
    "Squatting (Check)": [156, 160],
}
SCENARIO_MAPPING = {
    "Walking (Normal)": "Baseline",
    "Walking (Late)": "Baseline",
    "Sitting (Lower Occluded)": "Occlusion",
    "Walking (Upper Occluded)": "Occlusion",
    "Walking (Lower Occluded 1)": "Occlusion",
    "Walking (Lower Occluded 2)": "Occlusion",
    "Chair Interaction (Complex)": "Environmental Interference",
    "Lifting Box (Near Chair)": "Environmental Interference",
    "Squatting": "Dynamic Action",
    "Squatting (Check)": "Dynamic Action",
}

RULA_THRESHOLDS_LOWER_ARM = [60, 100]

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


ANGLE_TO_JOINTS = {
    "LeftShoulder": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, LEFT_HIP, RIGHT_HIP],
    "RightShoulder": [LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW, LEFT_HIP, RIGHT_HIP],
    "LeftElbow": [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST],
    "RightElbow": [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
    "LeftHip": [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE],
    "RightHip": [RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE],
    "LeftKnee": [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE],
    "RightKnee": [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE],
}


@dataclass(frozen=True)
class QualityChoice:
    """Signal choice for one semantic angle."""

    signal_name: str
    bad_tail: str


BEST_SIGNAL_BY_ANGLE = {
    "LeftShoulder": QualityChoice("epipolar_error_max", "high"),
    "RightShoulder": QualityChoice("pair_conf_min", "low"),
    "LeftElbow": QualityChoice("detect_conf_min", "low"),
    "RightElbow": QualityChoice("detect_conf_min", "low"),
    "LeftHip": QualityChoice("detect_conf_min", "low"),
    "RightHip": QualityChoice("stereo_quality_min", "low"),
    "LeftKnee": QualityChoice("detect_conf_min", "low"),
    "RightKnee": QualityChoice("stereo_quality_min", "low"),
}


def get_scenario(t_pose: float) -> str:
    """Map pose-relative time to evaluation scenario."""
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t_pose < end:
            return SCENARIO_MAPPING.get(label)
    return "Unclassified"


def classify_angle_by_thresholds(angle: float, thresholds: list[float]) -> int:
    """Classify a RULA-style angle into bins."""
    for idx, threshold in enumerate(thresholds):
        if angle < threshold:
            return idx
    return len(thresholds)


def load_alignment_offset(path: str) -> float:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["best_offset_seconds"])


def nanmin_over_joints(array: np.ndarray, joint_indices: list[int]) -> np.ndarray:
    subset = np.asarray(array[:, joint_indices], dtype=np.float64)
    out = np.full(subset.shape[0], np.nan, dtype=np.float64)
    for frame_idx, row in enumerate(subset):
        finite = row[np.isfinite(row)]
        if finite.size:
            out[frame_idx] = float(np.min(finite))
    return out


def nanmax_over_joints(array: np.ndarray, joint_indices: list[int]) -> np.ndarray:
    subset = np.asarray(array[:, joint_indices], dtype=np.float64)
    out = np.full(subset.shape[0], np.nan, dtype=np.float64)
    for frame_idx, row in enumerate(subset):
        finite = row[np.isfinite(row)]
        if finite.size:
            out[frame_idx] = float(np.max(finite))
    return out


def build_quality_signals(pose: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Precompute angle-specific quality signals."""
    signals: dict[str, np.ndarray] = {}
    for angle_name, joints in ANGLE_TO_JOINTS.items():
        signals[f"{angle_name}__pair_conf_min"] = nanmin_over_joints(pose["pair_confidence"], joints)
        signals[f"{angle_name}__stereo_quality_min"] = nanmin_over_joints(pose["stereo_quality"], joints)
        signals[f"{angle_name}__epipolar_error_max"] = nanmax_over_joints(pose["epipolar_error"], joints)
        signals[f"{angle_name}__detect_conf_min"] = np.minimum(
            nanmin_over_joints(pose["conf_left"], joints),
            nanmin_over_joints(pose["conf_right"], joints),
        )
    return signals


def load_records() -> list[dict[str, object]]:
    """Load frame-level records with GT and quality metadata."""
    pose_npz = np.load(POSE_NPZ)
    pose = {key: pose_npz[key] for key in pose_npz.files}
    keypoints = pose["keypoints"].astype(np.float64)
    timestamps = pose["timestamps"].astype(float)

    est_center = 0.5 * (keypoints[:, LEFT_HIP] + keypoints[:, RIGHT_HIP])
    valid_mask = (
        (est_center[:, 2] > 10.0)
        & (est_center[:, 2] < 1000.0)
        & np.isfinite(est_center).all(axis=1)
    )
    keypoints = keypoints[valid_mask]
    timestamps = timestamps[valid_mask]
    timestamps, unique_idx = np.unique(timestamps, return_index=True)
    keypoints = keypoints[unique_idx]
    timestamps = timestamps - timestamps[0]

    filtered_pose = {
        key: value[valid_mask][unique_idx]
        if isinstance(value, np.ndarray) and value.shape[:1] == pose["timestamps"].shape[:1]
        else value
        for key, value in pose.items()
    }

    _, angle_values = compute_semantic_angle_sequence(keypoints)
    angle_values = median_filter_angle_sequence(angle_values, radius=ANGLE_SMOOTH_RADIUS)
    offset_s = load_alignment_offset(ALIGNMENT_JSON)
    quality_signals = build_quality_signals(filtered_pose)

    mvnx = MvnxParser(GT_MVNX)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, unique_idx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, unique_idx)
    fair_interps = build_fair_gt_interpolators(FAIR_GT_NPZ)

    records = []
    for frame_idx, t_pose in enumerate(timestamps):
        scenario = get_scenario(float(t_pose))
        t_xsens = float(t_pose - offset_s)
        est_row = angle_values[frame_idx]
        gt_row = np.array(
            [gt_interps[name](t_xsens) if name in gt_interps else np.nan for name in SEMANTIC_ANGLE_NAMES],
            dtype=np.float64,
        )
        fair_row = np.array(
            [fair_interps[name](t_xsens) if name in fair_interps else np.nan for name in SEMANTIC_ANGLE_NAMES],
            dtype=np.float64,
        )
        quality_row = {
            angle_name: {
                signal_name: quality_signals[f"{angle_name}__{signal_name}"][frame_idx]
                for signal_name in {"pair_conf_min", "stereo_quality_min", "epipolar_error_max", "detect_conf_min"}
            }
            for angle_name in SEMANTIC_ANGLE_NAMES
        }
        records.append(
            {
                "scenario": scenario,
                "est": est_row,
                "gt": gt_row,
                "fair_gt": fair_row,
                "quality": quality_row,
            }
        )
    return records


def fit_models(records: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Fit global and quality-aware models from all records."""
    global_models: dict[str, object] = {}
    quality_models: dict[str, dict[str, object]] = {}
    for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
        est = np.array([rec["est"][angle_idx] for rec in records], dtype=np.float64)
        gt = np.array([rec["gt"][angle_idx] for rec in records], dtype=np.float64)
        finite = np.isfinite(est) & np.isfinite(gt)
        global_cal = (
            fit_piecewise_calibration(est[finite], gt[finite], n_bins=CAL_BINS)
            if int(finite.sum()) >= MIN_FIT_SAMPLES
            else None
        )
        global_models[angle_name] = global_cal

        choice = BEST_SIGNAL_BY_ANGLE[angle_name]
        signal = np.array(
            [rec["quality"][angle_name][choice.signal_name] for rec in records],
            dtype=np.float64,
        )
        finite_signal = finite & np.isfinite(signal)
        if int(finite_signal.sum()) < MIN_FIT_SAMPLES:
            quality_models[angle_name] = {
                "choice": choice,
                "threshold": np.nan,
                "good": None,
                "bad": None,
            }
            continue

        threshold = float(np.nanpercentile(signal[finite_signal], QUALITY_SPLIT_PERCENTILE))
        if choice.bad_tail == "high":
            bad_mask = finite_signal & (signal >= threshold)
            good_mask = finite_signal & (signal < threshold)
        else:
            bad_mask = finite_signal & (signal <= threshold)
            good_mask = finite_signal & (signal > threshold)

        quality_models[angle_name] = {
            "choice": choice,
            "threshold": threshold,
            "bad": fit_piecewise_calibration(est[bad_mask], gt[bad_mask], n_bins=CAL_BINS)
            if int(bad_mask.sum()) >= MIN_FIT_SAMPLES
            else None,
            "good": fit_piecewise_calibration(est[good_mask], gt[good_mask], n_bins=CAL_BINS)
            if int(good_mask.sum()) >= MIN_FIT_SAMPLES
            else None,
        }
    return global_models, quality_models


def apply_one_value(raw_value: float, calibration: object) -> float:
    if not np.isfinite(raw_value) or calibration is None:
        return raw_value
    return float(apply_piecewise_calibration(np.array([raw_value]), calibration)[0])


def apply_quality_value(raw_value: float, signal_value: float, model: dict[str, object], fallback: object) -> float:
    if not np.isfinite(raw_value):
        return np.nan
    if not np.isfinite(signal_value):
        return apply_one_value(raw_value, fallback)
    is_bad = signal_value >= model["threshold"] if model["choice"].bad_tail == "high" else signal_value <= model["threshold"]
    selected = model["bad"] if is_bad else model["good"]
    if selected is None:
        selected = fallback
    return apply_one_value(raw_value, selected)


def evaluate_subset(
    records: list[dict[str, object]],
    global_models: dict[str, object],
    quality_models: dict[str, dict[str, object]],
    qa_angles: set[str],
) -> dict[str, float]:
    """Evaluate one subset choice on all records."""
    e2e_errors = []
    fair_errors = []
    elbow_matches = []
    elbow_total = 0

    for rec in records:
        est = rec["est"]
        gt = rec["gt"]
        fair = rec["fair_gt"]
        quality = rec["quality"]
        for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
            raw_value = float(est[angle_idx])
            gt_value = float(gt[angle_idx])
            fair_value = float(fair[angle_idx])
            if not (np.isfinite(raw_value) and np.isfinite(gt_value)):
                continue

            if angle_name in qa_angles:
                choice = quality_models[angle_name]["choice"]
                signal_value = float(quality[angle_name][choice.signal_name])
                est_value = apply_quality_value(
                    raw_value,
                    signal_value,
                    quality_models[angle_name],
                    global_models.get(angle_name),
                )
            else:
                est_value = apply_one_value(raw_value, global_models.get(angle_name))

            e2e_errors.append(abs(est_value - gt_value))
            if np.isfinite(fair_value):
                fair_errors.append(abs(est_value - fair_value))

            if "Elbow" in angle_name:
                elbow_total += 1
                est_class = classify_angle_by_thresholds(abs(est_value), RULA_THRESHOLDS_LOWER_ARM)
                gt_class = classify_angle_by_thresholds(abs(gt_value), RULA_THRESHOLDS_LOWER_ARM)
                elbow_matches.append(int(est_class == gt_class))

    return {
        "joint_angle_mae": float(np.mean(e2e_errors)) if e2e_errors else np.nan,
        "fair_angle_mae": float(np.mean(fair_errors)) if fair_errors else np.nan,
        "elbow_rula_accuracy": float(np.mean(elbow_matches)) if elbow_total else np.nan,
    }


def save_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(item) for item in row) + "\n")


def main() -> None:
    """Search every subset and report the best trade-offs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = load_records()
    global_models, quality_models = fit_models(records)

    baseline = evaluate_subset(records, global_models, quality_models, set())
    full_quality = evaluate_subset(records, global_models, quality_models, set(SEMANTIC_ANGLE_NAMES))

    rows = []
    feasible = []
    for mask in range(1 << len(SEMANTIC_ANGLE_NAMES)):
        qa_angles = {
            angle_name
            for idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES)
            if mask & (1 << idx)
        }
        metrics = evaluate_subset(records, global_models, quality_models, qa_angles)
        row = [
            mask,
            ";".join(sorted(qa_angles)),
            len(qa_angles),
            metrics["joint_angle_mae"],
            metrics["fair_angle_mae"],
            metrics["elbow_rula_accuracy"],
            metrics["joint_angle_mae"] - baseline["joint_angle_mae"],
            metrics["elbow_rula_accuracy"] - baseline["elbow_rula_accuracy"],
        ]
        rows.append(row)
        if (
            np.isfinite(metrics["joint_angle_mae"])
            and np.isfinite(metrics["elbow_rula_accuracy"])
            and metrics["elbow_rula_accuracy"] >= baseline["elbow_rula_accuracy"] - MAX_RULA_DROP
        ):
            feasible.append((qa_angles, metrics))

    rows.sort(key=lambda row: float(row[3]))
    feasible.sort(key=lambda item: float(item[1]["joint_angle_mae"]))

    csv_path = os.path.join(RESULTS_DIR, "selective_quality_subset_search.csv")
    save_csv(
        csv_path,
        [
            "Mask",
            "QualityAwareAngles",
            "NumQualityAwareAngles",
            "JointAngleMAE",
            "FairAngleMAE",
            "ElbowRULAAccuracy",
            "DeltaVsGlobal_MAE",
            "DeltaVsGlobal_ElbowRULA",
        ],
        rows,
    )

    summary_lines = [
        "# Selective Quality-Aware Calibration Search",
        "",
        f"- Pose file: `{POSE_NPZ}`",
        f"- Angle smoothing radius: `{ANGLE_SMOOTH_RADIUS}`",
        f"- Quality split percentile: `{QUALITY_SPLIT_PERCENTILE}`",
        f"- Max allowed elbow RULA drop vs global: `{MAX_RULA_DROP:.3f}`",
        "",
        "## Baselines",
        "",
        f"- Global only: MAE `{baseline['joint_angle_mae']:.2f}°`, fair `{baseline['fair_angle_mae']:.2f}°`, elbow RULA `{baseline['elbow_rula_accuracy']:.4f}`",
        f"- Full quality-aware: MAE `{full_quality['joint_angle_mae']:.2f}°`, fair `{full_quality['fair_angle_mae']:.2f}°`, elbow RULA `{full_quality['elbow_rula_accuracy']:.4f}`",
        "",
        "## Best feasible subsets",
        "",
        "| Rank | Quality-aware angles | MAE | Fair MAE | Elbow RULA | Delta MAE vs global | Delta RULA vs global |",
        "|------|----------------------|-----|----------|------------|----------------------|----------------------|",
    ]

    for rank, (qa_angles, metrics) in enumerate(feasible[:10], start=1):
        summary_lines.append(
            f"| {rank} | {'; '.join(sorted(qa_angles)) or '(none)'} | "
            f"{metrics['joint_angle_mae']:.2f} | {metrics['fair_angle_mae']:.2f} | "
            f"{metrics['elbow_rula_accuracy']:.4f} | "
            f"{metrics['joint_angle_mae'] - baseline['joint_angle_mae']:+.2f} | "
            f"{metrics['elbow_rula_accuracy'] - baseline['elbow_rula_accuracy']:+.4f} |"
        )

    summary_path = os.path.join(RESULTS_DIR, "selective_quality_subset_search.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    if feasible:
        best_angles, best_metrics = feasible[0]
        payload = {
            "quality_aware_angles": sorted(best_angles),
            "joint_angle_mae": best_metrics["joint_angle_mae"],
            "fair_angle_mae": best_metrics["fair_angle_mae"],
            "elbow_rula_accuracy": best_metrics["elbow_rula_accuracy"],
            "baseline_joint_angle_mae": baseline["joint_angle_mae"],
            "baseline_elbow_rula_accuracy": baseline["elbow_rula_accuracy"],
        }
        best_json = os.path.join(RESULTS_DIR, "best_selective_quality_subset.json")
        with open(best_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[info] saved: {best_json}")

    print(f"[info] saved: {csv_path}")
    print(f"[info] saved: {summary_path}")


if __name__ == "__main__":
    main()
