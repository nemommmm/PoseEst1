import importlib.util
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from utils_mvnx import MvnxParser


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
EVAL_SCRIPT_PATH = os.path.join(SRC_DIR, "05_detailed_evaluation.py")

FRAME_TIMELINE_CSV = os.path.join(RESULTS_DIR, "eval_error_timeline.csv")
WINDOW_SUMMARY_CSV = os.path.join(RESULTS_DIR, "eval_error_window_summary.csv")
OVERALL_PLOT_PATH = os.path.join(SRC_DIR, "eval_error_timeline_overall.png")
ACTIVITY_PLOT_PATH = os.path.join(SRC_DIR, "eval_error_timeline_by_activity.png")


def load_eval_module():
    spec = importlib.util.spec_from_file_location("pose_eval", EVAL_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_frame(curr_t, segments, scenario_mapping):
    activity_label = "Unclassified"
    scenario_label = "Unclassified"
    for label, (start, end) in segments.items():
        if start <= curr_t <= end:
            activity_label = label
            scenario_label = scenario_mapping.get(label, "Other")
            break
    return activity_label, scenario_label


def rolling_frames_from_timestamps(timestamps, seconds):
    if len(timestamps) < 3:
        return 3
    dt = np.median(np.diff(timestamps))
    if not np.isfinite(dt) or dt <= 0:
        return 3
    return max(int(round(seconds / dt)), 3)


def compute_window_summary(df_frame, window_frames):
    rows = []
    for activity, group in df_frame.groupby("Activity", sort=False):
        group = group.sort_values("Time").copy()
        series = group["Mean_Joint_Error_cm"]
        rolling = series.rolling(window_frames, center=True, min_periods=max(3, window_frames // 2)).mean()

        best_idx = rolling.idxmin()
        worst_idx = rolling.idxmax()
        best_time = group.loc[best_idx, "Time"] if pd.notna(best_idx) else np.nan
        worst_time = group.loc[worst_idx, "Time"] if pd.notna(worst_idx) else np.nan
        best_value = rolling.loc[best_idx] if pd.notna(best_idx) else np.nan
        worst_value = rolling.loc[worst_idx] if pd.notna(worst_idx) else np.nan

        rows.append(
            {
                "Activity": activity,
                "Scenario": group["Scenario"].iloc[0],
                "Frames": int(len(group)),
                "Mean_Joint_Error_cm": float(series.mean()),
                "Median_Joint_Error_cm": float(series.median()),
                "P10_Joint_Error_cm": float(series.quantile(0.10)),
                "P90_Joint_Error_cm": float(series.quantile(0.90)),
                "Std_Joint_Error_cm": float(series.std(ddof=0)),
                "Mean_Angle_Error_deg": float(group["Mean_Angle_Error_deg"].mean()),
                "Mean_Root_Error_cm": float(group["Root_Error_cm"].mean()),
                "Best_Window_Error_cm": float(best_value),
                "Best_Window_Center_s": float(best_time),
                "Worst_Window_Error_cm": float(worst_value),
                "Worst_Window_Center_s": float(worst_time),
                "Worst_to_Best_Ratio": float(worst_value / best_value)
                if np.isfinite(best_value) and best_value > 1e-6 and np.isfinite(worst_value)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["Scenario", "Mean_Joint_Error_cm", "Activity"])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    eval_mod = load_eval_module()

    yolo_data_path = eval_mod.resolve_yolo_data_path()
    best_offset = eval_mod.resolve_best_offset()
    yolo_data = np.load(yolo_data_path)
    y_kpts = yolo_data["keypoints"]
    y_ts = yolo_data["timestamps"]
    reprojection_error = yolo_data["reprojection_error"] if "reprojection_error" in yolo_data else None

    y_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    valid_mask = (y_center[:, 2] > 10) & (y_center[:, 2] < 1000) & np.isfinite(y_center).all(axis=1)
    y_kpts = y_kpts[valid_mask]
    y_ts = y_ts[valid_mask]
    if reprojection_error is not None:
        reprojection_error = reprojection_error[valid_mask]

    y_ts, uidx = np.unique(y_ts, return_index=True)
    y_kpts = y_kpts[uidx]
    if reprojection_error is not None:
        reprojection_error = reprojection_error[uidx]
    y_ts = y_ts - y_ts[0]

    mvnx = MvnxParser(eval_mod.MVNX_PATH)
    mvnx.parse()
    xsens_ts = mvnx.timestamps
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts = xsens_ts - xsens_ts[0]

    xsens_interp = {}
    for seg in set(eval_mod.JOINT_MAPPING.values()):
        data = mvnx.get_segment_data(seg)[xidx]
        xsens_interp[seg] = interp1d(xsens_ts, data, axis=0, kind="linear", bounds_error=False, fill_value=np.nan)

    y_pelvis = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    limb_errors = eval_mod.calculate_limb_error(y_kpts, eval_mod.GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(limb_errors))[0]
    elite_indices = valid_err_idx[np.argsort(limb_errors[valid_err_idx])[: eval_mod.TOP_K]]
    p_elite = y_pelvis[elite_indices]
    q_elite = xsens_interp["Pelvis"](y_ts[elite_indices] - best_offset)
    rot, trans = eval_mod.kabsch_transform(p_elite, q_elite)

    y_aligned = ((rot @ y_kpts.reshape(-1, 3).T).T + trans).reshape(y_kpts.shape)

    frame_rows = []
    for i, curr_t in enumerate(y_ts):
        activity_label, scenario_label = label_frame(curr_t, eval_mod.ACTIVITY_SEGMENTS, eval_mod.SCENARIO_MAPPING)
        target_t = curr_t - best_offset

        joint_errors = []
        angle_errors = []
        for y_idx, x_name in eval_mod.JOINT_MAPPING.items():
            gt_pos = xsens_interp[x_name](target_t)
            est_pos = y_aligned[i, y_idx]
            if np.isnan(gt_pos).any() or np.isnan(est_pos).any():
                continue
            joint_errors.append(np.linalg.norm(est_pos - gt_pos))

        for angle_name, angle_def in eval_mod.ANGLE_DEFINITIONS.items():
            est_idx_a, est_idx_b, est_idx_c = angle_def["est_triplet"]
            gt_name_a, gt_name_b, gt_name_c = angle_def["gt_triplet"]
            est_angle = eval_mod.compute_angle_deg(
                y_aligned[i, est_idx_a],
                y_aligned[i, est_idx_b],
                y_aligned[i, est_idx_c],
            )
            gt_angle = eval_mod.compute_angle_deg(
                xsens_interp[gt_name_a](target_t),
                xsens_interp[gt_name_b](target_t),
                xsens_interp[gt_name_c](target_t),
            )
            if np.isnan(est_angle) or np.isnan(gt_angle):
                continue
            angle_errors.append(abs(est_angle - gt_angle))

        gt_pelvis = xsens_interp["Pelvis"](target_t)
        est_pelvis = (y_aligned[i, 11] + y_aligned[i, 12]) / 2.0
        root_error = np.linalg.norm(est_pelvis - gt_pelvis) if not np.isnan(gt_pelvis).any() else np.nan

        reproj_frame = np.nan
        valid_joint_ratio = np.nan
        if reprojection_error is not None:
            reproj_frame = float(np.nanmean(reprojection_error[i]))
            valid_joint_ratio = float(np.isfinite(y_aligned[i]).all(axis=1).mean())

        frame_rows.append(
            {
                "Time": float(curr_t),
                "Activity": activity_label,
                "Scenario": scenario_label,
                "Mean_Joint_Error_cm": float(np.mean(joint_errors)) if joint_errors else np.nan,
                "Median_Joint_Error_cm": float(np.median(joint_errors)) if joint_errors else np.nan,
                "P90_Joint_Error_cm": float(np.percentile(joint_errors, 90)) if joint_errors else np.nan,
                "Valid_Joints": int(len(joint_errors)),
                "Root_Error_cm": float(root_error) if np.isfinite(root_error) else np.nan,
                "Mean_Angle_Error_deg": float(np.mean(angle_errors)) if angle_errors else np.nan,
                "Valid_Angles": int(len(angle_errors)),
                "Mean_Reprojection_px": reproj_frame,
                "Valid_Joint_Ratio": valid_joint_ratio,
            }
        )

    df_frame = pd.DataFrame(frame_rows)
    smooth_frames = rolling_frames_from_timestamps(df_frame["Time"].to_numpy(), seconds=1.0)
    df_frame["Joint_Error_Rolling_cm"] = (
        df_frame["Mean_Joint_Error_cm"].rolling(smooth_frames, center=True, min_periods=max(3, smooth_frames // 2)).median()
    )
    df_frame["Angle_Error_Rolling_deg"] = (
        df_frame["Mean_Angle_Error_deg"].rolling(smooth_frames, center=True, min_periods=max(3, smooth_frames // 2)).median()
    )
    df_frame["Root_Error_Rolling_cm"] = (
        df_frame["Root_Error_cm"].rolling(smooth_frames, center=True, min_periods=max(3, smooth_frames // 2)).median()
    )
    df_frame.to_csv(FRAME_TIMELINE_CSV, index=False)

    window_frames = rolling_frames_from_timestamps(df_frame["Time"].to_numpy(), seconds=2.0)
    window_summary = compute_window_summary(df_frame[df_frame["Activity"] != "Unclassified"], window_frames)
    window_summary.to_csv(WINDOW_SUMMARY_CSV, index=False)

    scenario_palette = {
        "Baseline": "#7AA95C",
        "Occlusion": "#D8A03D",
        "Dynamic Action": "#3B82A0",
        "Environmental Interference": "#C65D4A",
    }

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
    for label, (start, end) in eval_mod.ACTIVITY_SEGMENTS.items():
        scenario = eval_mod.SCENARIO_MAPPING.get(label, "Other")
        color = scenario_palette.get(scenario, "#CFCFCF")
        for ax in axes:
            ax.axvspan(start, end, color=color, alpha=0.12)
            ax.text((start + end) / 2.0, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)

    axes[0].plot(df_frame["Time"], df_frame["Mean_Joint_Error_cm"], color="#7F8C8D", alpha=0.35, linewidth=1.0, label="Frame MPJPE proxy")
    axes[0].plot(df_frame["Time"], df_frame["Joint_Error_Rolling_cm"], color="#1F77B4", linewidth=2.2, label="1 s rolling median")
    axes[0].plot(df_frame["Time"], df_frame["Root_Error_Rolling_cm"], color="#D62728", linewidth=1.7, linestyle="--", label="Root error rolling median")
    axes[0].set_ylabel("Position Error (cm)")
    axes[0].set_title("Frame-wise Error Timeline with Scenario Segments")
    axes[0].legend(loc="upper right")

    axes[1].plot(df_frame["Time"], df_frame["Mean_Angle_Error_deg"], color="#7F8C8D", alpha=0.35, linewidth=1.0, label="Frame mean angle error")
    axes[1].plot(df_frame["Time"], df_frame["Angle_Error_Rolling_deg"], color="#9467BD", linewidth=2.2, label="1 s rolling median")
    axes[1].set_ylabel("Angle Error (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    fig.savefig(OVERALL_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    activities = list(eval_mod.ACTIVITY_SEGMENTS.keys())
    fig, axes = plt.subplots(len(activities), 1, figsize=(14, 2.5 * len(activities)), sharex=False, constrained_layout=True)
    if len(activities) == 1:
        axes = [axes]
    for ax, activity in zip(axes, activities):
        start, end = eval_mod.ACTIVITY_SEGMENTS[activity]
        subset = df_frame[df_frame["Activity"] == activity].copy()
        if subset.empty:
            ax.set_title(f"{activity} | no aligned samples")
            ax.axis("off")
            continue
        ax.plot(subset["Time"], subset["Mean_Joint_Error_cm"], color="#7F8C8D", alpha=0.35, linewidth=1.0)
        ax.plot(subset["Time"], subset["Joint_Error_Rolling_cm"], color="#1F77B4", linewidth=2.0)
        ax.plot(subset["Time"], subset["Root_Error_Rolling_cm"], color="#D62728", linewidth=1.5, linestyle="--")
        ax.set_xlim(start, end)
        ax.set_ylabel("cm")
        ax.set_title(
            f"{activity} | {eval_mod.SCENARIO_MAPPING.get(activity, 'Other')} | "
            f"mean={subset['Mean_Joint_Error_cm'].mean():.1f} cm | "
            f"p90={subset['Mean_Joint_Error_cm'].quantile(0.9):.1f} cm"
        )
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(ACTIVITY_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Info] Frame timeline table saved to: {FRAME_TIMELINE_CSV}")
    print(f"[Info] Window summary saved to: {WINDOW_SUMMARY_CSV}")
    print(f"[Info] Plot generated: {OVERALL_PLOT_PATH}")
    print(f"[Info] Plot generated: {ACTIVITY_PLOT_PATH}")
    print("\n[Result] Activity window summary")
    print(window_summary.to_string(index=False))


if __name__ == "__main__":
    main()
