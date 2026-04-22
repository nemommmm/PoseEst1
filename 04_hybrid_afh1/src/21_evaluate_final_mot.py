#!/opt/anaconda3/envs/pose/bin/python
"""Evaluate the EasyErgo final OpenSim MOT output against Xsens."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AFH1_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from opensim_mot_utils import (  # noqa: E402
    build_semantic_angles_from_mot,
    load_opensim_mot,
    resolve_easyergo_final_outputs,
)
from pose_angle_utils import build_gt_angle_interpolators  # noqa: E402
from utils_mvnx import MvnxParser  # noqa: E402


INPUT_DIR = Path(
    os.environ.get(
        "POSE_EASYERGO_INPUT_DIR",
        str(AFH1_DIR / "data" / "easyergo_uploaded"),
    )
).resolve()
RESULTS_DIR = Path(
    os.environ.get("POSE_RESULTS_DIR", str(AFH1_DIR / "results" / "21_final_mot_eval"))
).resolve()
TIMING_JSON = Path(
    os.environ.get(
        "POSE_TIMING_JSON",
        str(AFH1_DIR / "results" / "20_final_mot_timing" / "affine_fit.json"),
    )
).resolve()
SUMMARY_MD = RESULTS_DIR / "easyergo_final_mot_summary.md"
INSPECTION_JSON = RESULTS_DIR / "easyergo_final_mot_inspection.json"
GT_MVNX_PATH = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"
BEST_OFFSET_FALLBACK = 17.25
TIME_SCALE_FALLBACK = 1.0
OFFSET_OVERRIDE = os.environ.get("POSE_OFFSET_SECONDS", "").strip()
TIME_SCALE_OVERRIDE = os.environ.get("POSE_TIME_SCALE", "").strip()

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

RULA_THRESHOLDS_LOWER_ARM = [60.0, 100.0]
ANGLE_NAMES = [
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftHip",
    "RightHip",
    "LeftKnee",
    "RightKnee",
]


def resolve_time_mapping() -> tuple[float, float]:
    """Load the retained affine timing parameters for the final MOT branch."""
    if OFFSET_OVERRIDE or TIME_SCALE_OVERRIDE:
        offset_s = float(OFFSET_OVERRIDE) if OFFSET_OVERRIDE else BEST_OFFSET_FALLBACK
        time_scale = float(TIME_SCALE_OVERRIDE) if TIME_SCALE_OVERRIDE else TIME_SCALE_FALLBACK
        return offset_s, time_scale
    if TIMING_JSON.is_file():
        with TIMING_JSON.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            float(payload.get("offset_s", BEST_OFFSET_FALLBACK)),
            float(payload.get("time_scale", TIME_SCALE_FALLBACK)),
        )
    return BEST_OFFSET_FALLBACK, TIME_SCALE_FALLBACK


def classify_angle_by_thresholds(angle_deg: float, thresholds: list[float]) -> int:
    """Classify one angle into a coarse RULA-like bucket."""
    for idx, threshold in enumerate(thresholds):
        if angle_deg < threshold:
            return idx
    return len(thresholds)


def get_activity_and_scenario(curr_t: float) -> tuple[str, str]:
    """Assign one timestamp to the manual scenario annotation."""
    for label, (start_s, end_s) in ACTIVITY_SEGMENTS.items():
        if start_s <= curr_t <= end_s:
            return label, SCENARIO_MAPPING.get(label, "Other")
    return "Unclassified", "Unclassified"


def build_trunk_interp(mvnx: MvnxParser, xsens_ts: np.ndarray, unique_idx: np.ndarray):
    """Build GT trunk flexion interpolator from ergonomic angles."""
    trunk_ergo = mvnx.get_ergo_angle_data("Pelvis_T8")
    if trunk_ergo is None:
        return None
    return interp1d(
        xsens_ts,
        np.abs(trunk_ergo[unique_idx, 0]),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def build_summary_markdown(
    mot_path: Path,
    core_metrics: dict[str, float],
    joint_summary: pd.DataFrame,
    time_scale: float,
    offset_s: float,
) -> str:
    """Render a short human-readable evaluation summary."""
    lines = [
        "# EasyErgo Final MOT Evaluation",
        "",
        "## Inputs",
        f"- EasyErgo MOT: `{mot_path}`",
        f"- Offset: `{offset_s:.2f} s`",
        f"- Time scale: `{time_scale:.6f}`",
        "",
        "## Core Metrics",
        f"- Joint Angle MAE: `{core_metrics['Joint_Angle_MAE_deg']:.2f} deg`",
        f"- Joint Angle Median: `{core_metrics['Joint_Angle_Median_deg']:.2f} deg`",
        f"- Elbow RULA Accuracy: `{core_metrics['Elbow_RULA_Accuracy'] * 100.0:.2f}%`",
        (
            f"- Trunk Flexion Proxy MAE: `{core_metrics['Trunk_Flexion_MAE_deg']:.2f} deg`"
            if np.isfinite(core_metrics["Trunk_Flexion_MAE_deg"])
            else "- Trunk Flexion Proxy MAE: not available"
        ),
        "",
        "## Joint Breakdown",
    ]
    for _, row in joint_summary.iterrows():
        lines.append(
            f"- {row['AngleName']}: MAE `{row['MAE_deg']:.2f} deg`, "
            f"median `{row['Median_deg']:.2f} deg`, count `{int(row['Count'])}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This branch evaluates the scale-fixed OpenSim/MOT output directly in the local semantic-angle space.",
            "- It is comparable to Xsens on shoulders, elbows, hips, and knees, but it is not a fair-geometry metric because MOT provides angles rather than segment positions.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the final MOT evaluation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = resolve_easyergo_final_outputs(INPUT_DIR)
    mot_path = outputs["mot_path"]
    if mot_path is None:
        raise FileNotFoundError(f"No MOT file found in {INPUT_DIR}")

    offset_s, time_scale = resolve_time_mapping()

    mot_data = load_opensim_mot(mot_path)
    est_ts = np.asarray(mot_data["time"], dtype=np.float64)
    est_ts = est_ts - est_ts[0]
    est_angles, angle_sources = build_semantic_angles_from_mot(mot_data["coordinates"])

    gt_mvnx = MvnxParser(str(GT_MVNX_PATH))
    gt_mvnx.parse()
    gt_ts = gt_mvnx.timestamps.copy()
    gt_ts, unique_idx = np.unique(gt_ts, return_index=True)
    gt_ts = gt_ts - gt_ts[0]
    gt_interp = build_gt_angle_interpolators(gt_mvnx, gt_ts, unique_idx)
    gt_trunk_interp = build_trunk_interp(gt_mvnx, gt_ts, unique_idx)

    angle_records: list[dict[str, object]] = []
    elbow_matches: list[int] = []
    trunk_records: list[dict[str, object]] = []
    for frame_idx, curr_t in enumerate(est_ts):
        target_t = float(time_scale * curr_t - offset_s)
        activity, scenario = get_activity_and_scenario(float(curr_t))

        for angle_name in ANGLE_NAMES:
            interp = gt_interp.get(angle_name)
            if interp is None:
                continue
            est_val = float(est_angles[angle_name][frame_idx])
            if not np.isfinite(est_val):
                continue
            gt_val = float(interp(target_t))
            if not np.isfinite(gt_val):
                continue
            error = abs(est_val - gt_val)
            angle_records.append(
                {
                    "Time": float(curr_t),
                    "TargetTime": float(target_t),
                    "Activity": activity,
                    "Scenario": scenario,
                    "AngleName": angle_name,
                    "Estimated_deg": est_val,
                    "GroundTruth_deg": gt_val,
                    "Error": error,
                }
            )
            if "Elbow" in angle_name:
                est_class = classify_angle_by_thresholds(abs(est_val), RULA_THRESHOLDS_LOWER_ARM)
                gt_class = classify_angle_by_thresholds(abs(gt_val), RULA_THRESHOLDS_LOWER_ARM)
                elbow_matches.append(int(est_class == gt_class))

        trunk_est = float(est_angles["TrunkFlexionProxy"][frame_idx])
        if np.isfinite(trunk_est) and gt_trunk_interp is not None:
            trunk_gt = float(gt_trunk_interp(target_t))
            if np.isfinite(trunk_gt):
                trunk_records.append(
                    {
                        "Time": float(curr_t),
                        "TargetTime": float(target_t),
                        "Activity": activity,
                        "Scenario": scenario,
                        "Estimated_deg": trunk_est,
                        "GroundTruth_deg": trunk_gt,
                        "Error": abs(trunk_est - trunk_gt),
                    }
                )

    df_angles = pd.DataFrame(angle_records)
    if df_angles.empty:
        raise RuntimeError("Final MOT evaluation produced no valid samples.")

    df_joint = (
        df_angles.groupby("AngleName")["Error"]
        .agg(MAE_deg="mean", Median_deg="median", Count="count")
        .reset_index()
        .sort_values("MAE_deg")
    )
    df_scenario = (
        df_angles.groupby("Scenario")["Error"]
        .agg(MAE_deg="mean", Median_deg="median", Count="count")
        .reset_index()
        .sort_values("MAE_deg")
    )
    df_trunk = pd.DataFrame(trunk_records)

    core_metrics = {
        "offset_seconds": float(offset_s),
        "time_scale": float(time_scale),
        "Joint_Angle_MAE_deg": float(df_angles["Error"].mean()),
        "Joint_Angle_Median_deg": float(df_angles["Error"].median()),
        "Elbow_RULA_Accuracy": float(np.mean(elbow_matches)) if elbow_matches else np.nan,
        "Trunk_Flexion_MAE_deg": float(df_trunk["Error"].mean()) if not df_trunk.empty else np.nan,
        "Valid_Angle_Samples": float(len(df_angles)),
        "Valid_Trunk_Samples": float(len(df_trunk)),
    }

    pd.DataFrame.from_dict(core_metrics, orient="index", columns=["Value"]).to_csv(
        RESULTS_DIR / "eval_core_metrics_easyergo_final_mot.csv"
    )
    df_angles.to_csv(RESULTS_DIR / "eval_semantic_angles_easyergo_final_mot.csv", index=False)
    df_joint.to_csv(RESULTS_DIR / "eval_angle_by_joint_easyergo_final_mot.csv", index=False)
    df_scenario.to_csv(RESULTS_DIR / "eval_angle_by_scenario_easyergo_final_mot.csv", index=False)
    if not df_trunk.empty:
        df_trunk.to_csv(RESULTS_DIR / "eval_trunk_flexion_easyergo_final_mot.csv", index=False)

    SUMMARY_MD.write_text(
        build_summary_markdown(
            mot_path=mot_path,
            core_metrics=core_metrics,
            joint_summary=df_joint,
            time_scale=time_scale,
            offset_s=offset_s,
        ),
        encoding="utf-8",
    )
    INSPECTION_JSON.write_text(
        json.dumps(
            {
                "mot_path": str(mot_path),
                "core_metrics": core_metrics,
                "angle_sources": angle_sources,
                "joint_order": ANGLE_NAMES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[saved] {RESULTS_DIR / 'eval_core_metrics_easyergo_final_mot.csv'}")
    print(f"[saved] {RESULTS_DIR / 'eval_angle_by_joint_easyergo_final_mot.csv'}")
    print(f"[saved] {SUMMARY_MD}")
    print(
        "[result] MAE="
        f"{core_metrics['Joint_Angle_MAE_deg']:.2f} deg, "
        "Elbow_RULA="
        f"{core_metrics['Elbow_RULA_Accuracy'] * 100.0:.2f}%"
    )


if __name__ == "__main__":
    main()
