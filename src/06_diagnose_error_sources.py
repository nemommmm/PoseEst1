import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser


BEST_OFFSET = 17.20
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]
TOP_K = 150

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

JOINT_NAMES = {
    0: "Nose",
    1: "LeftEye",
    2: "RightEye",
    3: "LeftEar",
    4: "RightEar",
    5: "LeftShoulder",
    6: "RightShoulder",
    7: "LeftElbow",
    8: "RightElbow",
    9: "LeftWrist",
    10: "RightWrist",
    11: "LeftHip",
    12: "RightHip",
    13: "LeftKnee",
    14: "RightKnee",
    15: "LeftAnkle",
    16: "RightAnkle",
}

WINDOW_SECONDS = 15.0
WINDOW_STEP_SECONDS = 5.0
OFFSET_SEARCH_RADIUS = 1.0
OFFSET_STEP_SECONDS = 0.05
MIN_WINDOW_ROOT_SAMPLES = 60

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")
ALIGNMENT_SUMMARY_PATH = os.path.join(RESULTS_DIR, "alignment_summary.json")


def resolve_yolo_data_path():
    override = os.environ.get("POSE_RESULT_PATH")
    if override:
        return override if os.path.isabs(override) else os.path.join(PROJECT_ROOT, override)
    candidates = [
        os.path.join(PROJECT_ROOT, "results", "yolo_3d_optimized.npz"),
        os.path.join(PROJECT_ROOT, "results", "yolo_3d_raw.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def resolve_best_offset():
    if os.path.exists(ALIGNMENT_SUMMARY_PATH):
        with open(ALIGNMENT_SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("best_offset_seconds", BEST_OFFSET))
    return BEST_OFFSET


def calculate_limb_error(kpts, gt_lengths):
    lengths = np.vstack([
        np.linalg.norm(kpts[:, 11] - kpts[:, 13], axis=1),
        np.linalg.norm(kpts[:, 12] - kpts[:, 14], axis=1),
        np.linalg.norm(kpts[:, 13] - kpts[:, 15], axis=1),
        np.linalg.norm(kpts[:, 14] - kpts[:, 16], axis=1),
    ]).T
    return np.sum(np.abs(lengths - gt_lengths), axis=1)


def kabsch_transform(P, Q):
    mask = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P = P[mask]
    Q = Q[mask]
    if len(P) < 10:
        return np.eye(3), np.zeros(3)

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    AA = P - centroid_P
    BB = Q - centroid_Q
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    rot = Vt.T @ U.T
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
    t = centroid_Q - rot @ centroid_P
    return rot, t


def classify_time(t_sec):
    activity = "Unclassified"
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t_sec < end:
            activity = label
            break
    scenario = SCENARIO_MAPPING.get(activity, "Unclassified")
    return activity, scenario


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def evaluate_shift_window(shift, y_ts, y_center, f_x, elite_indices, p_elite):
    t_elite_shifted = y_ts[elite_indices] - shift
    q_elite = f_x(t_elite_shifted)
    R_mat, t_vec = kabsch_transform(p_elite, q_elite)
    y_final = (R_mat @ y_center.T).T + t_vec
    x_final_gt = f_x(y_ts - shift)
    dist = np.linalg.norm(y_final - x_final_gt, axis=1)
    valid_dist = dist[np.isfinite(dist)]
    if len(valid_dist) < MIN_WINDOW_ROOT_SAMPLES:
        return np.inf
    return float(np.mean(valid_dist))


def diagnose_offset_drift(y_ts, y_kpts, x_ts, x_pelvis, global_offset):
    y_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    valid_mask = (y_center[:, 2] > 10) & (y_center[:, 2] < 1000) & np.isfinite(y_center).all(axis=1)
    y_ts = y_ts[valid_mask]
    y_kpts = y_kpts[valid_mask]
    y_center = y_center[valid_mask]
    y_ts, uidx = np.unique(y_ts, return_index=True)
    y_kpts = y_kpts[uidx]
    y_center = y_center[uidx]

    f_x = interp1d(x_ts, x_pelvis, axis=0, kind="linear", bounds_error=False, fill_value=np.nan)
    window_rows = []
    search_offsets = np.arange(
        global_offset - OFFSET_SEARCH_RADIUS,
        global_offset + OFFSET_SEARCH_RADIUS + 0.5 * OFFSET_STEP_SECONDS,
        OFFSET_STEP_SECONDS,
    )

    errors = calculate_limb_error(y_kpts, GT_LIMB_LENGTHS)
    t_start = float(np.min(y_ts))
    t_end = float(np.max(y_ts))
    current = t_start
    while current + WINDOW_SECONDS <= t_end:
        window_mask = (y_ts >= current) & (y_ts < current + WINDOW_SECONDS)
        if np.sum(window_mask) < MIN_WINDOW_ROOT_SAMPLES:
            current += WINDOW_STEP_SECONDS
            continue
        y_ts_w = y_ts[window_mask]
        y_center_w = y_center[window_mask]
        y_kpts_w = y_kpts[window_mask]
        errors_w = errors[window_mask]

        valid_err_idx = np.where(np.isfinite(errors_w))[0]
        if len(valid_err_idx) < 20:
            current += WINDOW_STEP_SECONDS
            continue
        local_top_k = min(TOP_K, len(valid_err_idx))
        sorted_idx = np.argsort(errors_w[valid_err_idx])
        elite_indices = valid_err_idx[sorted_idx[:local_top_k]]
        p_elite = y_center_w[elite_indices]

        best_shift = np.nan
        best_error = np.inf
        for shift in search_offsets:
            err = evaluate_shift_window(shift, y_ts_w, y_center_w, f_x, elite_indices, p_elite)
            if err < best_error:
                best_error = err
                best_shift = shift

        activity, scenario = classify_time(0.5 * (current + current + WINDOW_SECONDS))
        window_rows.append({
            "window_start_s": current,
            "window_end_s": current + WINDOW_SECONDS,
            "window_center_s": current + 0.5 * WINDOW_SECONDS,
            "activity": activity,
            "scenario": scenario,
            "best_offset_s": best_shift,
            "offset_delta_vs_global_s": best_shift - global_offset,
            "root_mean_error_cm": best_error,
            "valid_root_samples": int(np.sum(window_mask)),
        })
        current += WINDOW_STEP_SECONDS

    return pd.DataFrame(window_rows)


def build_joint_diagnostics(rel_ts, keypoints, reproj_error, conf_left=None, conf_right=None):
    frame_rows = []
    joint_rows = []

    conf_pair = None
    if conf_left is not None and conf_right is not None:
        conf_pair = 0.5 * (conf_left + conf_right)

    for idx, t in enumerate(rel_ts):
        activity, scenario = classify_time(float(t))
        valid_mask = np.isfinite(keypoints[idx]).all(axis=1)
        reproj_frame = reproj_error[idx] if reproj_error is not None else np.full(keypoints.shape[1], np.nan)
        frame_rows.append({
            "time_s": float(t),
            "activity": activity,
            "scenario": scenario,
            "valid_joint_ratio": float(np.mean(valid_mask)),
            "valid_joint_count": int(np.sum(valid_mask)),
            "mean_reprojection_px": float(np.nanmean(reproj_frame)) if np.isfinite(reproj_frame).any() else np.nan,
        })
        for joint_idx in range(keypoints.shape[1]):
            row = {
                "time_s": float(t),
                "activity": activity,
                "scenario": scenario,
                "joint_idx": joint_idx,
                "joint_name": JOINT_NAMES.get(joint_idx, f"Joint{joint_idx}"),
                "valid": bool(valid_mask[joint_idx]),
                "reprojection_px": float(reproj_frame[joint_idx]) if np.isfinite(reproj_frame[joint_idx]) else np.nan,
            }
            if conf_pair is not None:
                row["pair_confidence"] = float(conf_pair[idx, joint_idx]) if np.isfinite(conf_pair[idx, joint_idx]) else np.nan
                row["left_confidence"] = float(conf_left[idx, joint_idx]) if np.isfinite(conf_left[idx, joint_idx]) else np.nan
                row["right_confidence"] = float(conf_right[idx, joint_idx]) if np.isfinite(conf_right[idx, joint_idx]) else np.nan
            joint_rows.append(row)

    return pd.DataFrame(frame_rows), pd.DataFrame(joint_rows)


def grouped_joint_summary(df_joints):
    grouped = df_joints.groupby(["scenario", "joint_name"], observed=True)
    summary = grouped.agg(
        valid_ratio=("valid", "mean"),
        reprojection_mean_px=("reprojection_px", "mean"),
        reprojection_median_px=("reprojection_px", "median"),
        reprojection_p95_px=("reprojection_px", lambda x: np.nanpercentile(x, 95) if np.isfinite(x).any() else np.nan),
        samples=("valid", "count"),
    ).reset_index()

    if "pair_confidence" in df_joints.columns:
        conf_stats = grouped["pair_confidence"].agg(pair_confidence_mean="mean", pair_confidence_median="median").reset_index()
        summary = summary.merge(conf_stats, on=["scenario", "joint_name"], how="left")
    return summary


def grouped_frame_summary(df_frames):
    return (
        df_frames.groupby("scenario", observed=True)
        .agg(
            valid_joint_ratio_mean=("valid_joint_ratio", "mean"),
            valid_joint_ratio_median=("valid_joint_ratio", "median"),
            reprojection_mean_px=("mean_reprojection_px", "mean"),
            reprojection_median_px=("mean_reprojection_px", "median"),
            frames=("valid_joint_count", "count"),
        )
        .reset_index()
    )


def save_plot(df_offsets, path):
    if df_offsets.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(df_offsets["window_center_s"], df_offsets["best_offset_s"], marker="o", linewidth=1.5)
    plt.axhline(df_offsets["best_offset_s"].median(), color="r", linestyle="--", linewidth=1.0, label="window median")
    plt.title("Window-wise Best Temporal Offset")
    plt.xlabel("Window center (s)")
    plt.ylabel("Best offset (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data_path = resolve_yolo_data_path()
    data = np.load(data_path)
    rel_ts = data["timestamps"].astype(np.float64)
    rel_ts = rel_ts - rel_ts[0]
    keypoints = data["keypoints"].astype(np.float64)
    reproj_error = data["reprojection_error"].astype(np.float64) if "reprojection_error" in data else None
    conf_left = data["conf_left"].astype(np.float64) if "conf_left" in data else None
    conf_right = data["conf_right"].astype(np.float64) if "conf_right" in data else None

    global_offset = resolve_best_offset()

    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    x_pelvis = mvnx.get_segment_data("Pelvis")
    x_ts = mvnx.timestamps.astype(np.float64)
    x_ts, xidx = np.unique(x_ts, return_index=True)
    x_pelvis = x_pelvis[xidx]
    x_ts = x_ts - x_ts[0]

    print(f"[Info] Running diagnostics on: {os.path.basename(data_path)}")
    print(f"[Info] Using global temporal offset: {global_offset:.2f} s")

    df_offsets = diagnose_offset_drift(rel_ts, keypoints, x_ts, x_pelvis, global_offset)
    df_frames, df_joints = build_joint_diagnostics(rel_ts, keypoints, reproj_error, conf_left, conf_right)
    df_joint_summary = grouped_joint_summary(df_joints)
    df_frame_summary = grouped_frame_summary(df_frames)

    offset_csv = os.path.join(RESULTS_DIR, "diagnostic_offset_windows.csv")
    joint_csv = os.path.join(RESULTS_DIR, "diagnostic_joint_quality_by_scenario.csv")
    frame_csv = os.path.join(RESULTS_DIR, "diagnostic_frame_quality_by_scenario.csv")
    summary_json = os.path.join(RESULTS_DIR, "diagnostic_error_sources_summary.json")
    offset_plot = os.path.join(SRC_DIR, "diagnostic_offset_drift.png")

    df_offsets.to_csv(offset_csv, index=False)
    df_joint_summary.to_csv(joint_csv, index=False)
    df_frame_summary.to_csv(frame_csv, index=False)
    save_plot(df_offsets, offset_plot)

    finite_offsets = df_offsets[np.isfinite(df_offsets["root_mean_error_cm"])] if not df_offsets.empty else df_offsets
    offset_summary = {
        "window_count": int(len(df_offsets)),
        "finite_window_count": int(len(finite_offsets)) if not df_offsets.empty else 0,
        "invalid_window_count": int(len(df_offsets) - len(finite_offsets)) if not df_offsets.empty else 0,
        "offset_std_s": float(finite_offsets["best_offset_s"].std()) if not finite_offsets.empty else np.nan,
        "offset_range_s": float(finite_offsets["best_offset_s"].max() - finite_offsets["best_offset_s"].min()) if not finite_offsets.empty else np.nan,
        "max_abs_delta_vs_global_s": float(np.max(np.abs(finite_offsets["offset_delta_vs_global_s"]))) if not finite_offsets.empty else np.nan,
        "root_mean_error_window_mean_cm": float(finite_offsets["root_mean_error_cm"].mean()) if not finite_offsets.empty else np.nan,
    }

    focus_joint_summary = df_joint_summary[df_joint_summary["scenario"].isin(["Baseline", "Environmental Interference"])]
    focus_frame_summary = df_frame_summary[df_frame_summary["scenario"].isin(["Baseline", "Environmental Interference"])]

    report = {
        "data_path": data_path,
        "global_best_offset_s": global_offset,
        "offset_drift_summary": offset_summary,
        "frame_quality_by_scenario": json.loads(focus_frame_summary.to_json(orient="records")),
        "joint_quality_by_scenario_focus": json.loads(focus_joint_summary.to_json(orient="records")),
        "confidence_available": bool(conf_left is not None and conf_right is not None),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[Result] Offset Drift Summary")
    print("-" * 60)
    if df_offsets.empty:
        print("No valid windows for drift analysis.")
    else:
        print(f"Windows analyzed: {offset_summary['window_count']}")
        print(f"Finite windows:    {offset_summary['finite_window_count']}")
        print(f"Invalid windows:   {offset_summary['invalid_window_count']}")
        print(f"Offset std:       {offset_summary['offset_std_s']:.3f} s")
        print(f"Offset range:     {offset_summary['offset_range_s']:.3f} s")
        print(f"Max |delta|:      {offset_summary['max_abs_delta_vs_global_s']:.3f} s")
        print(f"Mean root error:  {offset_summary['root_mean_error_window_mean_cm']:.2f} cm")

    print("\n[Result] Scenario Frame Quality")
    print("-" * 60)
    if df_frame_summary.empty:
        print("No frame-level diagnostics available.")
    else:
        print(df_frame_summary.to_string(index=False))

    print("\n[Result] Scenario Joint Quality (Baseline vs Environmental Interference)")
    print("-" * 60)
    if focus_joint_summary.empty:
        print("No joint-level diagnostics available.")
    else:
        print(focus_joint_summary.sort_values(["scenario", "valid_ratio", "reprojection_mean_px"]).to_string(index=False))

    print(f"\n[Info] Offset windows saved to: {offset_csv}")
    print(f"[Info] Joint quality saved to: {joint_csv}")
    print(f"[Info] Frame quality saved to: {frame_csv}")
    print(f"[Info] Drift plot saved to: {offset_plot}")
    print(f"[Info] Summary saved to: {summary_json}")
    if conf_left is None or conf_right is None:
        print("[Info] Confidence arrays are not present in the current pose result file.")
        print("[Info] The updated 02_batch_inference.py will save them on the next rerun.")


if __name__ == "__main__":
    main()
