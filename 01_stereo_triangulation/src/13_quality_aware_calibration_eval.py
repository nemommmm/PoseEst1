"""Evaluate whether quality-aware angle calibration improves over global calibration.

The idea is simple: current piecewise calibration learns one correction curve per
angle. This script tests whether splitting calibration into good-quality and
bad-quality regimes yields better end-to-end angle MAE, especially out of
scenario.

Usage:
    /opt/anaconda3/envs/pose/bin/python 13_quality_aware_calibration_eval.py
"""

from __future__ import annotations

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
        "quality_aware_calibration_eval",
    ),
)
ANGLE_SMOOTH_RADIUS = int(os.environ.get("POSE_ANGLE_SMOOTH_RADIUS", "8"))
CAL_BINS = int(os.environ.get("POSE_CALIBRATION_BINS", "10"))
MIN_FIT_SAMPLES = int(os.environ.get("POSE_QA_MIN_FIT_SAMPLES", "80"))
QUALITY_SPLIT_PERCENTILE = float(os.environ.get("POSE_QA_SPLIT_PERCENTILE", "40"))


ACTIVITY_SEGMENTS = {
    "Walking (Normal)": [17, 32],
    "Walking (Late)": [220, 240],
    "Sitting (Lower Occluded)": [32, 62],
    "Walking (Upper Occluded)": [87, 97],
    "Walking (Lower Occluded 1)": [130, 140],
    "Walking (Lower Occluded 2)": [164, 170],
    "Chair Interaction (Complex)": [140, 160],
    "Lifting Box (Near Chair)": [214, 218],
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
}
CV_SCENARIOS = ["Baseline", "Occlusion", "Environmental Interference"]


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
    """Define the best currently known quality signal for one angle."""

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


def get_scenario(t_pose: float) -> str | None:
    """Map pose-relative time to evaluation scenario."""
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t_pose < end:
            return SCENARIO_MAPPING.get(label)
    return None


def load_alignment_offset(path: str) -> float:
    """Read temporal alignment summary."""
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
    """Precompute angle-specific quality vectors for all candidate signals."""
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


def load_pose_records() -> list[dict[str, object]]:
    """Build frame-level records with angles, GT, fair GT, and quality signals."""
    pose_npz = np.load(POSE_NPZ)
    pose = {key: pose_npz[key] for key in pose_npz.files}
    timestamps = pose["timestamps"].astype(float)
    timestamps = timestamps - timestamps[0]
    keypoints = pose["keypoints"].astype(np.float64)

    angle_names, angle_values = compute_semantic_angle_sequence(keypoints)
    angle_values = median_filter_angle_sequence(angle_values, radius=ANGLE_SMOOTH_RADIUS)
    offset_s = load_alignment_offset(ALIGNMENT_JSON)

    quality_signals = build_quality_signals(pose)

    mvnx = MvnxParser(GT_MVNX)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, unique_idx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, unique_idx)
    fair_interps = build_fair_gt_interpolators(FAIR_GT_NPZ)

    records: list[dict[str, object]] = []
    for frame_idx, t_pose in enumerate(timestamps):
        scenario = get_scenario(float(t_pose))
        if scenario is None:
            continue
        t_xsens = float(t_pose - offset_s)
        est_row = angle_values[frame_idx]
        gt_row = np.array(
            [gt_interps[name](t_xsens) if name in gt_interps else np.nan for name in angle_names],
            dtype=np.float64,
        )
        fair_row = np.array(
            [fair_interps[name](t_xsens) if name in fair_interps else np.nan for name in angle_names],
            dtype=np.float64,
        )
        quality_row = {
            angle_name: {
                signal_name: quality_signals[f"{angle_name}__{signal_name}"][frame_idx]
                for signal_name in {"pair_conf_min", "stereo_quality_min", "epipolar_error_max", "detect_conf_min"}
            }
            for angle_name in angle_names
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


def fit_global_calibration(train_recs: list[dict[str, object]]) -> dict[str, object]:
    """Fit one piecewise calibration curve per angle."""
    calibrations: dict[str, object] = {}
    for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
        est = np.array([rec["est"][angle_idx] for rec in train_recs], dtype=np.float64)
        gt = np.array([rec["gt"][angle_idx] for rec in train_recs], dtype=np.float64)
        finite = np.isfinite(est) & np.isfinite(gt)
        if int(finite.sum()) >= MIN_FIT_SAMPLES:
            calibrations[angle_name] = fit_piecewise_calibration(est[finite], gt[finite], n_bins=CAL_BINS)
        else:
            calibrations[angle_name] = None
    return calibrations


def fit_quality_aware_calibration(train_recs: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Fit good/bad calibration curves using the strongest quality signal per angle."""
    quality_models: dict[str, dict[str, object]] = {}
    for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
        choice = BEST_SIGNAL_BY_ANGLE[angle_name]
        est = np.array([rec["est"][angle_idx] for rec in train_recs], dtype=np.float64)
        gt = np.array([rec["gt"][angle_idx] for rec in train_recs], dtype=np.float64)
        signal = np.array(
            [rec["quality"][angle_name][choice.signal_name] for rec in train_recs],
            dtype=np.float64,
        )
        finite = np.isfinite(est) & np.isfinite(gt) & np.isfinite(signal)
        if int(finite.sum()) < MIN_FIT_SAMPLES:
            quality_models[angle_name] = {"choice": choice, "threshold": np.nan, "good": None, "bad": None}
            continue

        threshold = float(np.nanpercentile(signal[finite], QUALITY_SPLIT_PERCENTILE))
        if choice.bad_tail == "high":
            bad_mask = finite & (signal >= threshold)
            good_mask = finite & (signal < threshold)
        else:
            bad_mask = finite & (signal <= threshold)
            good_mask = finite & (signal > threshold)

        bad_cal = None
        if int(bad_mask.sum()) >= MIN_FIT_SAMPLES:
            bad_cal = fit_piecewise_calibration(est[bad_mask], gt[bad_mask], n_bins=CAL_BINS)
        good_cal = None
        if int(good_mask.sum()) >= MIN_FIT_SAMPLES:
            good_cal = fit_piecewise_calibration(est[good_mask], gt[good_mask], n_bins=CAL_BINS)

        quality_models[angle_name] = {
            "choice": choice,
            "threshold": threshold,
            "good": good_cal,
            "bad": bad_cal,
        }
    return quality_models


def apply_quality_model(raw_value: float, quality_value: float, model: dict[str, object], global_cal: object) -> float:
    """Apply quality-aware calibration with graceful fallback."""
    if not np.isfinite(raw_value):
        return np.nan
    if not np.isfinite(quality_value):
        return apply_piecewise_calibration(np.array([raw_value]), global_cal)[0] if global_cal is not None else raw_value

    choice: QualityChoice = model["choice"]
    threshold = float(model["threshold"])
    bad_region = quality_value >= threshold if choice.bad_tail == "high" else quality_value <= threshold
    selected = model["bad"] if bad_region else model["good"]
    if selected is None:
        selected = global_cal
    return apply_piecewise_calibration(np.array([raw_value]), selected)[0] if selected is not None else raw_value


def evaluate_records(
    records: list[dict[str, object]],
    global_cal: dict[str, object],
    quality_cal: dict[str, dict[str, object]],
) -> dict[str, float]:
    """Evaluate raw, global, and quality-aware calibration on a record split."""
    errors = {
        "raw_e2e": [],
        "global_e2e": [],
        "quality_e2e": [],
        "raw_fair": [],
        "global_fair": [],
        "quality_fair": [],
    }
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

            global_value = apply_piecewise_calibration(
                np.array([raw_value]),
                global_cal.get(angle_name),
            )[0] if global_cal.get(angle_name) is not None else raw_value

            choice = BEST_SIGNAL_BY_ANGLE[angle_name]
            quality_value = float(quality[angle_name][choice.signal_name])
            quality_value_cal = apply_quality_model(
                raw_value,
                quality_value,
                quality_cal[angle_name],
                global_cal.get(angle_name),
            )

            errors["raw_e2e"].append(abs(raw_value - gt_value))
            errors["global_e2e"].append(abs(global_value - gt_value))
            errors["quality_e2e"].append(abs(quality_value_cal - gt_value))

            if np.isfinite(fair_value):
                errors["raw_fair"].append(abs(raw_value - fair_value))
                errors["global_fair"].append(abs(global_value - fair_value))
                errors["quality_fair"].append(abs(quality_value_cal - fair_value))

    return {
        key: float(np.mean(values)) if values else np.nan
        for key, values in errors.items()
    }


def save_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    """Write a small CSV without pandas."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(item) for item in row) + "\n")


def main() -> None:
    """Run in-sample and leave-one-scenario-out evaluation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = load_pose_records()

    rows: list[list[object]] = []
    summary_lines = [
        "# Quality-Aware Calibration Evaluation",
        "",
        f"- Pose file: `{POSE_NPZ}`",
        f"- Angle smoothing radius: `{ANGLE_SMOOTH_RADIUS}`",
        f"- Piecewise bins: `{CAL_BINS}`",
        f"- Quality split percentile: `{QUALITY_SPLIT_PERCENTILE}`",
        "",
    ]

    for held_out in CV_SCENARIOS + ["InSample"]:
        if held_out == "InSample":
            train_recs = records
            test_recs = records
        else:
            train_recs = [rec for rec in records if rec["scenario"] != held_out]
            test_recs = [rec for rec in records if rec["scenario"] == held_out]

        global_cal = fit_global_calibration(train_recs)
        quality_cal = fit_quality_aware_calibration(train_recs)
        metrics = evaluate_records(test_recs, global_cal, quality_cal)

        rows.append(
            [
                held_out,
                len(train_recs),
                len(test_recs),
                metrics["raw_e2e"],
                metrics["global_e2e"],
                metrics["quality_e2e"],
                metrics["raw_fair"],
                metrics["global_fair"],
                metrics["quality_fair"],
                metrics["quality_e2e"] - metrics["global_e2e"]
                if np.isfinite(metrics["quality_e2e"]) and np.isfinite(metrics["global_e2e"])
                else np.nan,
            ]
        )

    csv_path = os.path.join(RESULTS_DIR, "quality_aware_calibration_eval.csv")
    save_csv(
        csv_path,
        [
            "Split",
            "TrainFrames",
            "TestFrames",
            "Raw_E2E_MAE",
            "GlobalCal_E2E_MAE",
            "QualityAware_E2E_MAE",
            "Raw_Fair_MAE",
            "GlobalCal_Fair_MAE",
            "QualityAware_Fair_MAE",
            "Delta_QualityAware_minus_Global",
        ],
        rows,
    )

    summary_lines.append("## Results")
    summary_lines.append("")
    summary_lines.append(
        "| Split | Raw E2E | Global Cal E2E | Quality-Aware E2E | Raw Fair | Global Cal Fair | Quality-Aware Fair | Delta vs Global |"
    )
    summary_lines.append(
        "|-------|---------|----------------|-------------------|----------|-----------------|--------------------|-----------------|"
    )
    for row in rows:
        summary_lines.append(
            f"| {row[0]} | {float(row[3]):.2f} | {float(row[4]):.2f} | {float(row[5]):.2f} | "
            f"{float(row[6]):.2f} | {float(row[7]):.2f} | {float(row[8]):.2f} | {float(row[9]):+.2f} |"
        )

    summary_md = os.path.join(RESULTS_DIR, "quality_aware_calibration_eval.md")
    with open(summary_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print(f"[info] saved: {csv_path}")
    print(f"[info] saved: {summary_md}")


if __name__ == "__main__":
    main()
