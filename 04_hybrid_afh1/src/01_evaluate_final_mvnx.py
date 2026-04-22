"""Evaluate EasyErgo final MVNX output against Xsens ground truth."""

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
sys.path.insert(0, str(SHARED_DIR))

from pose_angle_utils import (  # noqa: E402
    SEMANTIC_ANGLE_NAMES,
    build_fair_gt_interpolators,
    build_gt_angle_interpolators,
    compute_aligned_trunk_flexion,
    compute_semantic_joint_angles,
)
from utils_mvnx import MvnxParser  # noqa: E402


INPUT_DIR = Path(
    os.environ.get(
        "POSE_EASYERGO_INPUT_DIR",
        str(AFH1_DIR / "data" / "easyergo_uploaded"),
    )
).resolve()
RESULTS_DIR = Path(
    os.environ.get(
        "POSE_RESULTS_DIR",
        str(AFH1_DIR / "results" / "01_final_mvnx_eval"),
    )
).resolve()
SUMMARY_MD = RESULTS_DIR / "easyergo_final_mvnx_summary.md"
INSPECTION_JSON = RESULTS_DIR / "easyergo_final_mvnx_inspection.json"
EXPERIMENT_LOG = AFH1_DIR / "results" / "experiment_log.md"
APPEND_EXPERIMENT_LOG = os.environ.get("POSE_APPEND_EXPERIMENT_LOG", "1").lower() not in {
    "0",
    "false",
    "no",
}
ALIGNMENT_JSON = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "alignment.json"
)
TIMING_JSON = Path(
    os.environ.get(
        "POSE_TIMING_JSON",
        str(AFH1_DIR / "results" / "02_final_mvnx_timing" / "affine_fit.json"),
    )
).resolve()
GT_MVNX_PATH = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"
FAIR_GT_NPZ = PROJECT_ROOT / "shared" / "fair_gt_angles.npz"
BEST_OFFSET_FALLBACK = 16.83
TIME_SCALE_FALLBACK = 1.0102
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

XSENS_TO_COCO: dict[str, int] = {
    "LeftUpperArm": 5,
    "RightUpperArm": 6,
    "LeftForeArm": 7,
    "RightForeArm": 8,
    "LeftHand": 9,
    "RightHand": 10,
    "LeftUpperLeg": 11,
    "RightUpperLeg": 12,
    "LeftLowerLeg": 13,
    "RightLowerLeg": 14,
    "LeftFoot": 15,
    "RightFoot": 16,
}

BASELINE_ACTIVITY_NAMES = {
    "Walking (Normal)",
    "Walking (Late)",
}


def append_experiment_log(message: str) -> None:
    """Append one line to the AFH1 experiment log."""
    if not APPEND_EXPERIMENT_LOG:
        return
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- {message}\n")


def resolve_best_offset() -> float:
    """Load the retained affine timing offset for the final MVNX branch."""
    if OFFSET_OVERRIDE:
        return float(OFFSET_OVERRIDE)
    if TIMING_JSON.is_file():
        with TIMING_JSON.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return float(payload.get("offset_s", BEST_OFFSET_FALLBACK))
    if ALIGNMENT_JSON.is_file():
        with ALIGNMENT_JSON.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return float(payload.get("best_offset_seconds", BEST_OFFSET_FALLBACK))
    return BEST_OFFSET_FALLBACK


def resolve_time_scale() -> float:
    """Load the retained affine time scale for the final MVNX branch."""
    if TIME_SCALE_OVERRIDE:
        return float(TIME_SCALE_OVERRIDE)
    if TIMING_JSON.is_file():
        with TIMING_JSON.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return float(payload.get("time_scale", TIME_SCALE_FALLBACK))
    return TIME_SCALE_FALLBACK


def map_est_time_to_gt(curr_t: float, best_offset: float, time_scale: float) -> float:
    """Map one EasyErgo timestamp onto the GT timeline."""
    return float(time_scale * curr_t - best_offset)


def get_activity_and_scenario(curr_t: float) -> tuple[str, str]:
    """Map one evaluation time to the annotated activity / scenario labels."""
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= curr_t <= end:
            return label, SCENARIO_MAPPING.get(label, "Other")
    return "Unclassified", "Unclassified"


def resolve_single_mvnx(input_dir: Path) -> Path:
    """Resolve the EasyErgo final MVNX file from the upload folder."""
    override = os.environ.get("POSE_EASYERGO_MVNX", "").strip()
    if override:
        path = Path(override).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"POSE_EASYERGO_MVNX does not exist: {path}")
        return path

    preferred = input_dir / "markers_easyergo.mvnx"
    if preferred.is_file():
        return preferred

    matches = sorted(input_dir.glob("*.mvnx"))
    if not matches:
        raise FileNotFoundError(
            f"No EasyErgo MVNX file found in {input_dir}. "
            "Please place the downloaded Xsens (MVNX) file there."
        )
    if len(matches) > 1:
        formatted = ", ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Multiple EasyErgo MVNX files found in {input_dir}: {formatted}"
        )
    return matches[0]


def build_trunk_interp(mvnx: MvnxParser, ts: np.ndarray, unique_idx: np.ndarray):
    """Build trunk-flexion interpolator from Pelvis_T8 ergonomic angles."""
    trunk_ergo = mvnx.get_ergo_angle_data("Pelvis_T8")
    if trunk_ergo is None:
        return None
    return interp1d(
        ts,
        trunk_ergo[unique_idx, 0],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def build_xsens_coco_poses(mvnx: MvnxParser) -> np.ndarray:
    """Build pseudo-COCO poses from MVNX segment origins."""
    n_frames = mvnx.data.shape[0]
    poses = np.full((n_frames, 17, 3), np.nan, dtype=np.float64)
    for seg_name, coco_idx in XSENS_TO_COCO.items():
        seg_data = mvnx.get_segment_data(seg_name)
        if seg_data is not None:
            poses[:, coco_idx, :] = seg_data
    return poses


def compute_angles_from_poses(poses: np.ndarray) -> dict[str, np.ndarray]:
    """Run the local semantic-angle calculation on every pseudo-COCO frame."""
    output = {name: np.full(len(poses), np.nan, dtype=np.float64) for name in SEMANTIC_ANGLE_NAMES}
    for frame_idx, pose in enumerate(poses):
        frame_angles = compute_semantic_joint_angles(pose)
        for name, value in frame_angles.items():
            output[name][frame_idx] = value
    return output


def infer_vertical_axis_from_poses(poses: np.ndarray, est_ts: np.ndarray) -> np.ndarray:
    """Infer the dominant upright axis from baseline trunk vectors."""
    unit_vectors: list[np.ndarray] = []

    def collect_vectors(use_baseline_only: bool) -> list[np.ndarray]:
        samples: list[np.ndarray] = []
        for frame_idx, pose in enumerate(poses):
            if use_baseline_only:
                activity, _ = get_activity_and_scenario(float(est_ts[frame_idx]))
                if activity not in BASELINE_ACTIVITY_NAMES:
                    continue
            if not np.isfinite(pose[[5, 6, 11, 12], :]).all():
                continue
            hip_mid = 0.5 * (pose[11] + pose[12])
            shoulder_mid = 0.5 * (pose[5] + pose[6])
            trunk_vec = shoulder_mid - hip_mid
            norm = np.linalg.norm(trunk_vec)
            if norm > 1e-8:
                samples.append(trunk_vec / norm)
        return samples

    unit_vectors = collect_vectors(use_baseline_only=True)
    if not unit_vectors:
        unit_vectors = collect_vectors(use_baseline_only=False)
    if not unit_vectors:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    vertical_axis = np.mean(unit_vectors, axis=0)
    norm = np.linalg.norm(vertical_axis)
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vertical_axis / norm


def compute_fair_trunk_from_poses(poses: np.ndarray, vertical_axis: np.ndarray) -> np.ndarray:
    """Compute trunk flexion from pseudo-COCO poses using an inferred upright axis."""
    values = np.full(len(poses), np.nan, dtype=np.float64)
    for frame_idx, pose in enumerate(poses):
        if np.isfinite(pose[[5, 6, 11, 12], :]).all():
            values[frame_idx] = compute_aligned_trunk_flexion(pose, vertical_axis=vertical_axis)
    return values


def evaluate_native_angles(
    est_ts: np.ndarray,
    est_interps: dict[str, interp1d],
    gt_interps: dict[str, interp1d],
    best_offset: float,
    time_scale: float,
) -> tuple[pd.DataFrame, list[int]]:
    """Evaluate native Xsens-style joint angles from EasyErgo MVNX."""
    records: list[dict[str, object]] = []
    elbow_matches: list[int] = []

    for curr_t in est_ts:
        target_t = map_est_time_to_gt(float(curr_t), best_offset, time_scale)
        activity, scenario = get_activity_and_scenario(float(curr_t))
        for angle_name in SEMANTIC_ANGLE_NAMES:
            if angle_name not in est_interps or angle_name not in gt_interps:
                continue
            est_val = float(est_interps[angle_name](curr_t))
            gt_val = float(gt_interps[angle_name](target_t))
            if not (np.isfinite(est_val) and np.isfinite(gt_val)):
                continue

            error = abs(est_val - gt_val)
            records.append(
                {
                    "Time": float(curr_t),
                    "Activity": activity,
                    "Scenario": scenario,
                    "AngleName": angle_name,
                    "Estimated_deg": est_val,
                    "GroundTruth_deg": gt_val,
                    "Error": error,
                }
            )

            if "Elbow" in angle_name:
                est_abs = abs(est_val)
                gt_abs = abs(gt_val)
                est_class = 0 if est_abs < 60.0 else (1 if est_abs < 100.0 else 2)
                gt_class = 0 if gt_abs < 60.0 else (1 if gt_abs < 100.0 else 2)
                elbow_matches.append(int(est_class == gt_class))

    return pd.DataFrame(records), elbow_matches


def evaluate_trunk(
    est_ts: np.ndarray,
    est_trunk_interp,
    gt_trunk_interp,
    best_offset: float,
    time_scale: float,
) -> pd.DataFrame:
    """Evaluate native Pelvis_T8 trunk angles."""
    if est_trunk_interp is None or gt_trunk_interp is None:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for curr_t in est_ts:
        target_t = map_est_time_to_gt(float(curr_t), best_offset, time_scale)
        est_val = float(est_trunk_interp(curr_t))
        gt_val = float(gt_trunk_interp(target_t))
        if not (np.isfinite(est_val) and np.isfinite(gt_val)):
            continue
        activity, scenario = get_activity_and_scenario(float(curr_t))
        records.append(
            {
                "Time": float(curr_t),
                "Activity": activity,
                "Scenario": scenario,
                "Estimated_deg": abs(est_val),
                "GroundTruth_deg": abs(gt_val),
                "Error": abs(abs(est_val) - abs(gt_val)),
            }
        )
    return pd.DataFrame(records)


def evaluate_fair_geometry(
    est_ts: np.ndarray,
    fair_angles: dict[str, np.ndarray],
    fair_gt_interp: dict[str, interp1d],
    best_offset: float,
    time_scale: float,
) -> pd.DataFrame:
    """Evaluate semantic geometry from EasyErgo MVNX segment positions."""
    if not fair_gt_interp:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for frame_idx, curr_t in enumerate(est_ts):
        target_t = map_est_time_to_gt(float(curr_t), best_offset, time_scale)
        activity, scenario = get_activity_and_scenario(float(curr_t))
        for angle_name in SEMANTIC_ANGLE_NAMES:
            if angle_name not in fair_gt_interp:
                continue
            est_val = float(fair_angles[angle_name][frame_idx])
            gt_val = float(fair_gt_interp[angle_name](target_t))
            if not (np.isfinite(est_val) and np.isfinite(gt_val)):
                continue
            records.append(
                {
                    "Time": float(curr_t),
                    "Activity": activity,
                    "Scenario": scenario,
                    "AngleName": angle_name,
                    "Estimated_deg": est_val,
                    "GroundTruth_deg": gt_val,
                    "Error": abs(est_val - gt_val),
                }
            )
    return pd.DataFrame(records)


def evaluate_fair_trunk(
    est_ts: np.ndarray,
    fair_trunk: np.ndarray,
    gt_trunk_interp,
    best_offset: float,
    time_scale: float,
) -> pd.DataFrame:
    """Evaluate local-geometry trunk flexion against GT Pelvis_T8."""
    if gt_trunk_interp is None:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for frame_idx, curr_t in enumerate(est_ts):
        target_t = map_est_time_to_gt(float(curr_t), best_offset, time_scale)
        est_val = float(fair_trunk[frame_idx])
        gt_val = float(gt_trunk_interp(target_t))
        if not (np.isfinite(est_val) and np.isfinite(gt_val)):
            continue
        activity, scenario = get_activity_and_scenario(float(curr_t))
        records.append(
            {
                "Time": float(curr_t),
                "Activity": activity,
                "Scenario": scenario,
                "Estimated_deg": est_val,
                "GroundTruth_deg": abs(gt_val),
                "Error": abs(est_val - abs(gt_val)),
            }
        )
    return pd.DataFrame(records)


def write_summary(
    mvnx_path: Path,
    core_metrics: dict[str, float],
    native_joint_summary: pd.DataFrame,
    fair_metrics: dict[str, float],
    ergo_finite_counts: dict[str, int],
    inferred_vertical_axis: np.ndarray,
) -> None:
    """Write a concise markdown summary for the final MVNX evaluation."""
    native_trunk_note = (
        "- Trunk Flexion MAE: not available "
        "(EasyErgo MVNX ergonomic-angle channels are all NaN)"
        if not np.isfinite(core_metrics["Trunk_Flexion_MAE_deg"])
        and all(count == 0 for count in ergo_finite_counts.values())
        else "- Trunk Flexion MAE: not available"
    )
    lines = [
        "# EasyErgo Final MVNX Evaluation",
        "",
        "## Inputs",
        f"- EasyErgo MVNX: `{mvnx_path}`",
        f"- Best offset: `{core_metrics['best_offset_seconds']:.2f} s`",
        f"- Time scale: `{core_metrics['time_scale']:.6f}`",
        "",
        "## Native Xsens-style Metrics",
        f"- Joint Angle MAE: `{core_metrics['Joint_Angle_MAE_deg']:.2f} deg`",
        f"- Joint Angle Median: `{core_metrics['Joint_Angle_Median_deg']:.2f} deg`",
        f"- Elbow RULA Accuracy: `{core_metrics['Elbow_RULA_Accuracy'] * 100.0:.2f}%`",
        (
            f"- Trunk Flexion MAE: `{core_metrics['Trunk_Flexion_MAE_deg']:.2f} deg`"
            if np.isfinite(core_metrics["Trunk_Flexion_MAE_deg"])
            else native_trunk_note
        ),
        "",
        "## Native Joint Breakdown",
    ]
    for _, row in native_joint_summary.iterrows():
        lines.append(
            f"- {row['AngleName']}: MAE `{row['MAE_deg']:.2f} deg`, "
            f"median `{row['Median_deg']:.2f} deg`, count `{int(row['Count'])}`"
        )

    lines += [
        "",
        "## Fair Geometry Metrics",
        (
            f"- Fair Joint Angle MAE: `{fair_metrics['Fair_Joint_Angle_MAE_deg']:.2f} deg`"
            if np.isfinite(fair_metrics["Fair_Joint_Angle_MAE_deg"])
            else "- Fair Joint Angle MAE: not available"
        ),
        (
            f"- Fair Joint Angle Median: `{fair_metrics['Fair_Joint_Angle_Median_deg']:.2f} deg`"
            if np.isfinite(fair_metrics["Fair_Joint_Angle_Median_deg"])
            else "- Fair Joint Angle Median: not available"
        ),
        (
            f"- Fair Trunk Flexion MAE: `{fair_metrics['Fair_Trunk_Flexion_MAE_deg']:.2f} deg`"
            if np.isfinite(fair_metrics["Fair_Trunk_Flexion_MAE_deg"])
            else "- Fair Trunk Flexion MAE: not available"
        ),
        (
            "- Inferred upright axis for fair trunk: "
            f"`[{inferred_vertical_axis[0]:.4f}, {inferred_vertical_axis[1]:.4f}, {inferred_vertical_axis[2]:.4f}]`"
        ),
        "",
        "## Interpretation",
        "- `Native Xsens-style metrics` compare EasyErgo final export and ground truth in the same MVNX angle definition.",
        "- `Fair geometry metrics` isolate the 3D skeleton quality after mapping segment origins into the local COCO-style geometry.",
        "- The fair trunk metric uses segment positions plus an inferred upright axis, so it remains usable even when EasyErgo leaves ergonomic-angle channels empty or rotates the global frame.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the final MVNX evaluation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mvnx_path = resolve_single_mvnx(INPUT_DIR)
    best_offset = resolve_best_offset()
    time_scale = resolve_time_scale()

    est_mvnx = MvnxParser(str(mvnx_path))
    est_mvnx.parse()
    gt_mvnx = MvnxParser(str(GT_MVNX_PATH))
    gt_mvnx.parse()

    est_ts = est_mvnx.timestamps.copy()
    est_ts, eidx = np.unique(est_ts, return_index=True)
    est_ts -= est_ts[0]

    gt_ts = gt_mvnx.timestamps.copy()
    gt_ts, gidx = np.unique(gt_ts, return_index=True)
    gt_ts -= gt_ts[0]

    est_native_interps = build_gt_angle_interpolators(est_mvnx, est_ts, eidx)
    gt_native_interps = build_gt_angle_interpolators(gt_mvnx, gt_ts, gidx)
    est_trunk_interp = build_trunk_interp(est_mvnx, est_ts, eidx)
    gt_trunk_interp = build_trunk_interp(gt_mvnx, gt_ts, gidx)

    df_native, elbow_matches = evaluate_native_angles(
        est_ts=est_ts,
        est_interps=est_native_interps,
        gt_interps=gt_native_interps,
        best_offset=best_offset,
        time_scale=time_scale,
    )
    if df_native.empty:
        raise RuntimeError("Native MVNX evaluation produced no valid samples.")

    df_native_joint = (
        df_native.groupby("AngleName")["Error"]
        .agg(MAE_deg="mean", Median_deg="median", Count="count")
        .reset_index()
    )
    df_native_scenario = (
        df_native.groupby("Scenario")["Error"]
        .agg(MAE_deg="mean", Median_deg="median", Count="count")
        .reset_index()
    )
    df_trunk = evaluate_trunk(
        est_ts=est_ts,
        est_trunk_interp=est_trunk_interp,
        gt_trunk_interp=gt_trunk_interp,
        best_offset=best_offset,
        time_scale=time_scale,
    )

    est_poses = build_xsens_coco_poses(est_mvnx)[eidx]
    fair_angles = compute_angles_from_poses(est_poses)
    inferred_vertical_axis = infer_vertical_axis_from_poses(est_poses, est_ts)
    fair_trunk = compute_fair_trunk_from_poses(est_poses, inferred_vertical_axis)
    fair_gt_interp = build_fair_gt_interpolators(str(FAIR_GT_NPZ))
    df_fair = evaluate_fair_geometry(
        est_ts=est_ts,
        fair_angles=fair_angles,
        fair_gt_interp=fair_gt_interp,
        best_offset=best_offset,
        time_scale=time_scale,
    )
    df_fair_trunk = evaluate_fair_trunk(
        est_ts=est_ts,
        fair_trunk=fair_trunk,
        gt_trunk_interp=gt_trunk_interp,
        best_offset=best_offset,
        time_scale=time_scale,
    )
    df_fair_joint = (
        df_fair.groupby("AngleName")["Error"]
        .agg(MAE_deg="mean", Median_deg="median", Count="count")
        .reset_index()
        if not df_fair.empty
        else pd.DataFrame(columns=["AngleName", "MAE_deg", "Median_deg", "Count"])
    )

    core_metrics = {
        "best_offset_seconds": best_offset,
        "time_scale": time_scale,
        "Joint_Angle_MAE_deg": float(df_native["Error"].mean()),
        "Joint_Angle_Median_deg": float(df_native["Error"].median()),
        "Elbow_RULA_Accuracy": float(np.mean(elbow_matches)) if elbow_matches else np.nan,
        "Trunk_Flexion_MAE_deg": float(df_trunk["Error"].mean()) if not df_trunk.empty else np.nan,
        "Valid_Angle_Samples": float(len(df_native)),
        "Valid_Trunk_Samples": float(len(df_trunk)),
    }
    fair_metrics = {
        "Fair_Joint_Angle_MAE_deg": float(df_fair["Error"].mean()) if not df_fair.empty else np.nan,
        "Fair_Joint_Angle_Median_deg": float(df_fair["Error"].median()) if not df_fair.empty else np.nan,
        "Fair_Trunk_Flexion_MAE_deg": (
            float(df_fair_trunk["Error"].mean()) if not df_fair_trunk.empty else np.nan
        ),
        "Valid_Fair_Samples": float(len(df_fair)),
    }
    ergo_finite_counts = {
        label: int(np.isfinite(est_mvnx.get_ergo_angle_data(label)[:, 0]).sum())
        for label in est_mvnx.ergo_labels
        if est_mvnx.get_ergo_angle_data(label) is not None
    }

    pd.DataFrame.from_dict(core_metrics, orient="index", columns=["Value"]).to_csv(
        RESULTS_DIR / "eval_core_metrics_easyergo_final_mvnx.csv"
    )
    pd.DataFrame.from_dict(fair_metrics, orient="index", columns=["Value"]).to_csv(
        RESULTS_DIR / "eval_core_metrics_fair_easyergo_final_mvnx.csv"
    )
    df_native.to_csv(RESULTS_DIR / "eval_native_angles_easyergo_final_mvnx.csv", index=False)
    df_native_joint.to_csv(RESULTS_DIR / "eval_angle_by_joint_easyergo_final_mvnx.csv", index=False)
    df_native_scenario.to_csv(
        RESULTS_DIR / "eval_angle_by_scenario_easyergo_final_mvnx.csv",
        index=False,
    )
    if not df_trunk.empty:
        df_trunk.to_csv(RESULTS_DIR / "eval_trunk_flexion_easyergo_final_mvnx.csv", index=False)
    if not df_fair.empty:
        df_fair.to_csv(RESULTS_DIR / "eval_fair_angles_easyergo_final_mvnx.csv", index=False)
        df_fair_joint.to_csv(
            RESULTS_DIR / "eval_angle_by_joint_fair_easyergo_final_mvnx.csv",
            index=False,
        )
    if not df_fair_trunk.empty:
        df_fair_trunk.to_csv(
            RESULTS_DIR / "eval_trunk_flexion_fair_easyergo_final_mvnx.csv",
            index=False,
        )

    inspection_payload = {
        "easyergo_mvnx_path": str(mvnx_path),
        "best_offset_seconds": best_offset,
        "time_scale": time_scale,
        "joint_labels": est_mvnx.joint_labels,
        "ergo_labels": est_mvnx.ergo_labels,
        "native_core_metrics": core_metrics,
        "fair_core_metrics": fair_metrics,
        "native_angle_sources": list(est_native_interps.keys()),
        "ergo_finite_counts": ergo_finite_counts,
        "inferred_vertical_axis": inferred_vertical_axis.tolist(),
        "segment_coverage": {
            seg_name: float(np.mean(np.isfinite(est_mvnx.get_segment_data(seg_name)).all(axis=1)))
            for seg_name in XSENS_TO_COCO
            if est_mvnx.get_segment_data(seg_name) is not None
        },
    }
    with INSPECTION_JSON.open("w", encoding="utf-8") as handle:
        json.dump(inspection_payload, handle, indent=2)

    write_summary(
        mvnx_path,
        core_metrics,
        df_native_joint,
        fair_metrics,
        ergo_finite_counts,
        inferred_vertical_axis,
    )
    append_experiment_log(
        "Evaluated EasyErgo final MVNX output against Xsens "
        f"(time-scale {time_scale:.6f}, native MAE {core_metrics['Joint_Angle_MAE_deg']:.2f} deg, "
        f"fair MAE {fair_metrics['Fair_Joint_Angle_MAE_deg']:.2f} deg)."
    )

    print(f"[saved] {RESULTS_DIR / 'eval_core_metrics_easyergo_final_mvnx.csv'}")
    print(f"[saved] {RESULTS_DIR / 'eval_core_metrics_fair_easyergo_final_mvnx.csv'}")
    print(f"[saved] {RESULTS_DIR / 'eval_angle_by_joint_easyergo_final_mvnx.csv'}")
    print(f"[saved] {SUMMARY_MD}")
    print(
        "[result] native_MAE="
        f"{core_metrics['Joint_Angle_MAE_deg']:.2f} deg, "
        "fair_MAE="
        f"{fair_metrics['Fair_Joint_Angle_MAE_deg']:.2f} deg"
    )


if __name__ == "__main__":
    main()
