"""Diagnose how stereo quality signals relate to semantic joint-angle errors.

This script analyzes the historical best SKT result and checks whether the
existing per-joint quality signals can predict large downstream angle errors.
The goal is to identify actionable improvement space without changing the main
pipeline first.

Usage:
    /opt/anaconda3/envs/pose/bin/python 12_angle_error_quality_diagnostics.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))

from pose_angle_utils import (  # noqa: E402
    SEMANTIC_ANGLE_NAMES,
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
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
RESULTS_DIR = os.environ.get(
    "POSE_RESULTS_DIR",
    os.path.join(
        PROJECT_ROOT,
        "01_stereo_triangulation",
        "results",
        "quality_error_diagnostics",
    ),
)
ANGLE_SMOOTH_RADIUS = int(os.environ.get("POSE_ANGLE_SMOOTH_RADIUS", "8"))


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
class SignalSpec:
    """Describe a diagnostic signal and how to interpret high/low values."""

    name: str
    extractor: Callable[[dict[str, np.ndarray], list[int]], np.ndarray]
    bad_tail: str
    description: str


def nanmean_over_joints(array: np.ndarray, joint_indices: list[int]) -> np.ndarray:
    subset = np.asarray(array[:, joint_indices], dtype=np.float64)
    finite = np.isfinite(subset)
    summed = np.nansum(subset, axis=1)
    counts = np.sum(finite, axis=1)
    out = np.full(subset.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = summed[valid] / counts[valid]
    return out


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


def get_scenario_label(t_pose: float) -> str:
    """Map pose-relative time to a coarse scenario label."""
    for activity_name, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t_pose < end:
            return SCENARIO_MAPPING[activity_name]
    return "Unclassified"


def load_alignment_offset(path: str) -> float:
    """Read the best pose-vs-Xsens offset in seconds."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["best_offset_seconds"])


def load_pose_data(npz_path: str) -> dict[str, np.ndarray]:
    """Load the optimized stereo pose result."""
    pose = np.load(npz_path)
    return {key: pose[key] for key in pose.files}


def build_angle_error_table(pose_data: dict[str, np.ndarray], offset_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute filtered semantic angle estimates and absolute GT errors."""
    timestamps = pose_data["timestamps"].astype(float)
    timestamps = timestamps - timestamps[0]
    keypoints = pose_data["keypoints"].astype(np.float64)

    angle_names, angle_values = compute_semantic_angle_sequence(keypoints)
    angle_values = median_filter_angle_sequence(angle_values, radius=ANGLE_SMOOTH_RADIUS)

    mvnx = MvnxParser(GT_MVNX)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, unique_idx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, unique_idx)

    angle_errors = np.full_like(angle_values, np.nan, dtype=np.float64)
    for angle_idx, angle_name in enumerate(angle_names):
        if angle_name not in gt_interps:
            continue
        gt_values = gt_interps[angle_name](timestamps - offset_s)
        finite = np.isfinite(angle_values[:, angle_idx]) & np.isfinite(gt_values)
        angle_errors[finite, angle_idx] = np.abs(angle_values[finite, angle_idx] - gt_values[finite])

    return timestamps, angle_errors


def build_signal_specs() -> list[SignalSpec]:
    """Return the quality signals to evaluate."""
    return [
        SignalSpec(
            name="pair_conf_min",
            extractor=lambda pose, joints: nanmin_over_joints(pose["pair_confidence"], joints),
            bad_tail="low",
            description="Lowest stereo pair confidence among angle-defining joints",
        ),
        SignalSpec(
            name="stereo_quality_min",
            extractor=lambda pose, joints: nanmin_over_joints(pose["stereo_quality"], joints),
            bad_tail="low",
            description="Lowest fused stereo quality among angle-defining joints",
        ),
        SignalSpec(
            name="reprojection_error_max",
            extractor=lambda pose, joints: nanmax_over_joints(pose["reprojection_error"], joints),
            bad_tail="high",
            description="Worst reprojection error among angle-defining joints",
        ),
        SignalSpec(
            name="epipolar_error_max",
            extractor=lambda pose, joints: nanmax_over_joints(pose["epipolar_error"], joints),
            bad_tail="high",
            description="Worst post-rectification epipolar mismatch among angle-defining joints",
        ),
        SignalSpec(
            name="detect_conf_min",
            extractor=lambda pose, joints: np.minimum(
                nanmin_over_joints(pose["conf_left"], joints),
                nanmin_over_joints(pose["conf_right"], joints),
            ),
            bad_tail="low",
            description="Lowest 2D detector confidence across both cameras",
        ),
        SignalSpec(
            name="temporal_rescue_fraction",
            extractor=lambda pose, joints: np.mean(
                np.logical_or(
                    pose["temporal_rescue_left"][:, joints],
                    pose["temporal_rescue_right"][:, joints],
                ),
                axis=1,
            ),
            bad_tail="high",
            description="Fraction of angle-defining joints rescued temporally",
        ),
    ]


def summarize_signal_for_angle(
    errors: np.ndarray,
    signal_values: np.ndarray,
    bad_tail: str,
) -> dict[str, float]:
    """Compute robust correlation and bad-vs-good band MAE gap."""
    valid = np.isfinite(errors) & np.isfinite(signal_values)
    if valid.sum() < 50:
        return {
            "n": int(valid.sum()),
            "rho": np.nan,
            "p_value": np.nan,
            "mae_bad_quartile": np.nan,
            "mae_good_quartile": np.nan,
            "mae_gap_bad_minus_good": np.nan,
        }

    e = errors[valid]
    s = signal_values[valid]
    rho, p_value = spearmanr(s, e, nan_policy="omit")

    q25, q75 = np.nanpercentile(s, [25.0, 75.0])
    if bad_tail == "high":
        bad_mask = s >= q75
        good_mask = s <= q25
    else:
        bad_mask = s <= q25
        good_mask = s >= q75

    mae_bad = float(np.nanmean(e[bad_mask])) if np.any(bad_mask) else np.nan
    mae_good = float(np.nanmean(e[good_mask])) if np.any(good_mask) else np.nan

    return {
        "n": int(valid.sum()),
        "rho": float(rho) if np.isfinite(rho) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "mae_bad_quartile": mae_bad,
        "mae_good_quartile": mae_good,
        "mae_gap_bad_minus_good": float(mae_bad - mae_good)
        if np.isfinite(mae_bad) and np.isfinite(mae_good)
        else np.nan,
    }


def save_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    """Save rows to a small CSV file without pandas dependency."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(item) for item in row) + "\n")


def main() -> None:
    """Run diagnostics and write summary artifacts."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    offset_s = load_alignment_offset(ALIGNMENT_JSON)
    pose_data = load_pose_data(POSE_NPZ)
    timestamps, angle_errors = build_angle_error_table(pose_data, offset_s)
    signal_specs = build_signal_specs()

    correlation_rows: list[list[object]] = []
    best_signal_rows: list[list[object]] = []

    scenario_rows: list[list[object]] = []
    scenario_labels = np.array([get_scenario_label(t) for t in timestamps], dtype=object)
    unique_scenarios = ["Baseline", "Occlusion", "Environmental Interference", "Unclassified"]

    for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
        joints = ANGLE_TO_JOINTS[angle_name]
        best_gap = -np.inf
        best_signal_name = ""
        best_signal_desc = ""
        best_rho = np.nan
        best_mae_bad = np.nan
        best_mae_good = np.nan

        for spec in signal_specs:
            signal_values = spec.extractor(pose_data, joints)
            stats = summarize_signal_for_angle(angle_errors[:, angle_idx], signal_values, spec.bad_tail)
            correlation_rows.append(
                [
                    angle_name,
                    spec.name,
                    spec.description,
                    stats["n"],
                    stats["rho"],
                    stats["p_value"],
                    stats["mae_bad_quartile"],
                    stats["mae_good_quartile"],
                    stats["mae_gap_bad_minus_good"],
                ]
            )

            gap = stats["mae_gap_bad_minus_good"]
            if np.isfinite(gap) and gap > best_gap:
                best_gap = gap
                best_signal_name = spec.name
                best_signal_desc = spec.description
                best_rho = stats["rho"]
                best_mae_bad = stats["mae_bad_quartile"]
                best_mae_good = stats["mae_good_quartile"]

            for scenario_name in unique_scenarios:
                mask = (
                    np.isfinite(angle_errors[:, angle_idx])
                    & np.isfinite(signal_values)
                    & (scenario_labels == scenario_name)
                )
                if int(mask.sum()) < 30:
                    continue
                scenario_rows.append(
                    [
                        angle_name,
                        scenario_name,
                        spec.name,
                        int(mask.sum()),
                        float(np.nanmean(angle_errors[mask, angle_idx])),
                        float(np.nanmean(signal_values[mask])),
                    ]
                )

        best_signal_rows.append(
            [
                angle_name,
                best_signal_name,
                best_signal_desc,
                best_rho,
                best_mae_bad,
                best_mae_good,
                best_gap,
            ]
        )

    correlation_csv = os.path.join(RESULTS_DIR, "per_angle_signal_correlations.csv")
    best_signal_csv = os.path.join(RESULTS_DIR, "per_angle_best_signal.csv")
    scenario_csv = os.path.join(RESULTS_DIR, "scenario_signal_summary.csv")

    save_csv(
        correlation_csv,
        [
            "AngleName",
            "Signal",
            "Description",
            "N",
            "SpearmanRho",
            "PValue",
            "MAE_BadQuartile",
            "MAE_GoodQuartile",
            "MAE_Gap_BadMinusGood",
        ],
        correlation_rows,
    )
    save_csv(
        best_signal_csv,
        [
            "AngleName",
            "BestSignal",
            "Description",
            "SpearmanRho",
            "MAE_BadQuartile",
            "MAE_GoodQuartile",
            "MAE_Gap_BadMinusGood",
        ],
        best_signal_rows,
    )
    save_csv(
        scenario_csv,
        [
            "AngleName",
            "Scenario",
            "Signal",
            "N",
            "MeanAngleError",
            "MeanSignalValue",
        ],
        scenario_rows,
    )

    strongest_rows = sorted(
        [row for row in best_signal_rows if np.isfinite(row[-1])],
        key=lambda row: float(row[-1]),
        reverse=True,
    )
    overall_angle_mae = [
        (
            angle_name,
            float(np.nanmean(angle_errors[:, angle_idx])),
            int(np.isfinite(angle_errors[:, angle_idx]).sum()),
        )
        for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES)
    ]
    overall_angle_mae.sort(key=lambda item: item[1], reverse=True)

    summary_path = os.path.join(RESULTS_DIR, "quality_error_summary.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# Quality vs Angle Error Diagnostic Summary\n\n")
        handle.write(f"- Pose file: `{POSE_NPZ}`\n")
        handle.write(f"- Alignment offset: `{offset_s:.2f} s`\n")
        handle.write(f"- Angle smoothing radius: `{ANGLE_SMOOTH_RADIUS}`\n\n")

        handle.write("## Highest-MAE angles\n\n")
        handle.write("| Angle | MAE (deg) | N |\n")
        handle.write("|-------|-----------|---|\n")
        for angle_name, mae, count in overall_angle_mae:
            handle.write(f"| {angle_name} | {mae:.2f} | {count} |\n")

        handle.write("\n## Best predictive quality signal per angle\n\n")
        handle.write("| Angle | Signal | Spearman rho | Bad quartile MAE | Good quartile MAE | Gap |\n")
        handle.write("|-------|--------|--------------|------------------|-------------------|-----|\n")
        for row in strongest_rows:
            handle.write(
                f"| {row[0]} | {row[1]} | {float(row[3]):+.3f} | "
                f"{float(row[4]):.2f} | {float(row[5]):.2f} | {float(row[6]):+.2f} |\n"
            )

        handle.write("\n## Interpretation\n\n")
        if strongest_rows:
            top_three = strongest_rows[:3]
            top_names = ", ".join(f"`{row[0]}` via `{row[1]}`" for row in top_three)
            handle.write(
                "- The strongest error-predictive combinations are "
                f"{top_names}. These are the best candidates for selective correction.\n"
            )
        handle.write(
            "- If bad-quality quartiles show clearly higher MAE than good-quality quartiles, "
            "we have room for quality-aware gating instead of replacing the full pipeline.\n"
        )
        handle.write(
            "- Signals tied to `reprojection_error` / `epipolar_error` point to stereo geometry issues; "
            "signals tied to `detect_conf` / `temporal_rescue` point to 2D detector instability or rescue-side effects.\n"
        )

    print(f"[info] saved: {correlation_csv}")
    print(f"[info] saved: {best_signal_csv}")
    print(f"[info] saved: {scenario_csv}")
    print(f"[info] saved: {summary_path}")


if __name__ == "__main__":
    main()
