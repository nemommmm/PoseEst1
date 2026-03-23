"""
05_detailed_evaluation.py  (Restructured)

Core evaluation script: compares estimated joint angles from the vision
pipeline against Xsens ground-truth joint angles.

Primary metric: Joint Angle MAE (°) — directly relevant for ergonomic assessment.
Supporting metric: MPJPE (cm) — spatial accuracy diagnostic only.

Input:
  - results/yolo_3d_ik_refined.npz   (from 02b_ik_refinement.py)
  - Xsens_ground_truth/Aitor-001.mvnx
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import json
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser

# ================= Configuration =================
BEST_OFFSET = 17.20  # Temporal offset (seconds); overridden by alignment_summary.json
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]
TOP_K = 150

# Activity segments (seconds) — manual annotation
ACTIVITY_SEGMENTS = {
    "Walking (Normal)":          [17, 32],
    "Walking (Late)":            [220, 240],
    "Sitting (Lower Occluded)":  [32, 62],
    "Walking (Upper Occluded)":  [87, 97],
    "Walking (Lower Occluded 1)":[130, 140],
    "Walking (Lower Occluded 2)":[164, 170],
    "Chair Interaction (Complex)":[140, 160],
    "Lifting Box (Near Chair)":  [214, 218],
    "Squatting":                 [66, 69],
    "Squatting (Check)":         [156, 160],
}

SCENARIO_MAPPING = {
    "Walking (Normal)":            "Baseline",
    "Walking (Late)":              "Baseline",
    "Sitting (Lower Occluded)":    "Occlusion",
    "Walking (Upper Occluded)":    "Occlusion",
    "Walking (Lower Occluded 1)":  "Occlusion",
    "Walking (Lower Occluded 2)":  "Occlusion",
    "Chair Interaction (Complex)": "Environmental Interference",
    "Lifting Box (Near Chair)":    "Environmental Interference",
    "Squatting":                   "Dynamic Action",
    "Squatting (Check)":           "Dynamic Action",
}

# Mapping: estimated joint angle name → Xsens jointAngle label + axis index
# Xsens jointAngle format is (3 values per joint): flexion/extension, 
# abduction/adduction, rotation. Index 1 (Y-axis) = flexion for most joints.
ANGLE_GT_MAPPING = {
    "LeftElbow":     ("jLeftElbow",     1),  # Y = flexion
    "RightElbow":    ("jRightElbow",    1),
    "LeftKnee":      ("jLeftKnee",      1),
    "RightKnee":     ("jRightKnee",     1),
    "LeftShoulder":  ("jLeftShoulder",  1),
    "RightShoulder": ("jRightShoulder", 1),
    "LeftHip":       ("jLeftHip",       1),
    "RightHip":      ("jRightHip",      1),
}

# RULA-relevant angle thresholds for classification accuracy
RULA_THRESHOLDS = {
    "UpperArm": [20, 45, 90],   # Shoulder elevation thresholds
    "LowerArm": [60, 100],      # Elbow flexion thresholds
    "Trunk":    [10, 20, 60],   # Trunk flexion thresholds
}

# ================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ALIGNMENT_SUMMARY_PATH = os.path.join(RESULTS_DIR, "alignment_summary.json")
os.makedirs(RESULTS_DIR, exist_ok=True)


def resolve_best_offset():
    if os.path.exists(ALIGNMENT_SUMMARY_PATH):
        with open(ALIGNMENT_SUMMARY_PATH, "r") as f:
            data = json.load(f)
        return float(data.get("best_offset_seconds", BEST_OFFSET))
    return BEST_OFFSET


def kabsch_transform(P, Q):
    """Optimal rotation+translation to align P onto Q."""
    mask = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P, Q = P[mask], Q[mask]
    if len(P) < 10:
        return np.eye(3), np.zeros(3)
    cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    rot = Vt.T @ U.T
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
    t = cQ - rot @ cP
    return rot, t


def calculate_limb_error(kpts, gt_lengths):
    lengths = np.vstack([
        np.linalg.norm(kpts[:, 11] - kpts[:, 13], axis=1),
        np.linalg.norm(kpts[:, 12] - kpts[:, 14], axis=1),
        np.linalg.norm(kpts[:, 13] - kpts[:, 15], axis=1),
        np.linalg.norm(kpts[:, 14] - kpts[:, 16], axis=1),
    ]).T
    return np.sum(np.abs(lengths - gt_lengths), axis=1)


def classify_angle_by_thresholds(angle, thresholds):
    """Classify an angle into a RULA-like category based on thresholds."""
    for i, thresh in enumerate(thresholds):
        if angle < thresh:
            return i
    return len(thresholds)


def get_activity_and_scenario(t):
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t <= end:
            return label, SCENARIO_MAPPING.get(label, "Other")
    return "Unclassified", "Unclassified"


def main():
    print("=" * 60)
    print("[Evaluation] Joint Angle-based Ergonomic Evaluation")
    print("=" * 60)

    best_offset = resolve_best_offset()
    print(f"[Info] Temporal offset: {best_offset:.2f} s")

    # --- 1. Load IK-refined pose data ---
    ik_path = os.path.join(RESULTS_DIR, "yolo_3d_ik_refined.npz")
    if not os.path.exists(ik_path):
        print("[Error] IK-refined data not found. Run 02b_ik_refinement.py first.")
        return
    ik_data = np.load(ik_path)
    est_kpts = ik_data['keypoints']          # (N, 17, 3)
    est_ts = ik_data['timestamps']           # (N,)
    est_angle_names = list(ik_data['angle_names'])
    est_angle_values = ik_data['angle_values']  # (N, 8)
    est_trunk_flexion = ik_data['trunk_flexion']  # (N,)
    print(f"[Info] Loaded IK data: {len(est_ts)} frames, {len(est_angle_names)} angles")

    # --- 2. Load Xsens GT ---
    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]

    # Build interpolators for GT joint angles
    gt_angle_interp = {}
    for est_name, (xsens_label, axis_idx) in ANGLE_GT_MAPPING.items():
        raw_data = mvnx.get_joint_angle_data(xsens_label)
        if raw_data is not None:
            flexion = raw_data[xidx, axis_idx]
            gt_angle_interp[est_name] = interp1d(
                xsens_ts, flexion, kind='linear', bounds_error=False, fill_value=np.nan
            )
        else:
            print(f"[Warning] Xsens joint '{xsens_label}' not found for {est_name}")

    # GT trunk flexion from ergonomic angles (Pelvis_T8 or Vertical_T8)
    trunk_ergo = mvnx.get_ergo_angle_data("Pelvis_T8")
    gt_trunk_interp = None
    if trunk_ergo is not None:
        trunk_flexion_gt = trunk_ergo[xidx, 0]  # X-axis = flexion/extension
        gt_trunk_interp = interp1d(
            xsens_ts, trunk_flexion_gt, kind='linear', bounds_error=False, fill_value=np.nan
        )

    # Also set up position interpolator for MPJPE (supporting metric)
    xsens_pos_interp = {}
    JOINT_POSITION_MAPPING = {
        0: 'Head', 5: 'LeftShoulder', 6: 'RightShoulder',
        7: 'LeftUpperArm', 8: 'RightUpperArm',
        9: 'LeftForeArm', 10: 'RightForeArm',
        11: 'Pelvis', 12: 'Pelvis',
        13: 'LeftUpperLeg', 14: 'RightUpperLeg',
        15: 'LeftLowerLeg', 16: 'RightLowerLeg',
    }
    for seg_name in set(JOINT_POSITION_MAPPING.values()):
        seg_data = mvnx.get_segment_data(seg_name)
        if seg_data is not None:
            xsens_pos_interp[seg_name] = interp1d(
                xsens_ts, seg_data[xidx], axis=0, kind='linear',
                bounds_error=False, fill_value=np.nan
            )

    # --- 3. Prepare alignment for MPJPE ---
    est_center = (est_kpts[:, 11] + est_kpts[:, 12]) / 2.0
    valid_mask = (est_center[:, 2] > 10) & (est_center[:, 2] < 1000) & np.isfinite(est_center).all(axis=1)
    est_kpts_v = est_kpts[valid_mask]
    est_ts_v = est_ts[valid_mask]
    est_angle_vals_v = est_angle_values[valid_mask]
    est_trunk_v = est_trunk_flexion[valid_mask]

    # Deduplicate timestamps
    est_ts_v, uidx = np.unique(est_ts_v, return_index=True)
    est_kpts_v = est_kpts_v[uidx]
    est_angle_vals_v = est_angle_vals_v[uidx]
    est_trunk_v = est_trunk_v[uidx]
    est_ts_v -= est_ts_v[0]

    # Kabsch alignment for MPJPE
    y_pelvis = (est_kpts_v[:, 11] + est_kpts_v[:, 12]) / 2.0
    errors = calculate_limb_error(est_kpts_v, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    elite_indices = valid_err_idx[np.argsort(errors[valid_err_idx])[:TOP_K]]
    p_elite = y_pelvis[elite_indices]
    q_elite = xsens_pos_interp['Pelvis'](est_ts_v[elite_indices] - best_offset)
    R, t_vec = kabsch_transform(p_elite, q_elite)

    N_frames = len(est_ts_v)
    est_kpts_aligned = (R @ est_kpts_v.reshape(-1, 3).T).T.reshape(N_frames, -1, 3) + t_vec

    # --- 4. Core evaluation ---
    print("\n[Info] Computing evaluation metrics...")
    angle_records = []
    mpjpe_records = []
    trunk_records = []

    for i, curr_t in enumerate(est_ts_v):
        target_t = curr_t - best_offset
        activity, scenario = get_activity_and_scenario(curr_t)

        # 4a. Joint angle errors (PRIMARY metric)
        for angle_idx, angle_name in enumerate(est_angle_names):
            est_val = est_angle_vals_v[i, angle_idx]
            if not np.isfinite(est_val):
                continue
            if angle_name not in gt_angle_interp:
                continue
            gt_val = gt_angle_interp[angle_name](target_t)
            if not np.isfinite(gt_val):
                continue

            # Our IK reports interior angles (0°=fully folded, 180°=straight).
            # Xsens reports flexion (0°=neutral). We compare using abs values
            # and compute the error as the min angular distance.
            est_flexion = 180.0 - est_val  # Convert interior → flexion

            angle_error = abs(est_flexion - gt_val)
            angle_records.append({
                "Time": curr_t, "Activity": activity, "Scenario": scenario,
                "AngleName": angle_name,
                "Estimated_deg": est_flexion, "GroundTruth_deg": gt_val,
                "Error": angle_error,
            })

        # 4b. Trunk flexion (computed from aligned keypoints)
        # Derive trunk angle from aligned pose: angle between torso vector
        # and the vertical direction (estimated from GT upright pose).
        hip_mid = 0.5 * (est_kpts_aligned[i, 11] + est_kpts_aligned[i, 12])
        shoulder_mid = 0.5 * (est_kpts_aligned[i, 5] + est_kpts_aligned[i, 6])
        if np.isfinite(hip_mid).all() and np.isfinite(shoulder_mid).all():
            torso_vec = shoulder_mid - hip_mid
            torso_len = np.linalg.norm(torso_vec)
            if torso_len > 1e-3:
                # Use Z-axis of the Kabsch-aligned frame as vertical
                # (after alignment, Xsens Z ≈ true vertical)
                cos_angle = np.clip(torso_vec[2] / torso_len, -1.0, 1.0)
                est_trunk_aligned = np.degrees(np.arccos(cos_angle))

                if gt_trunk_interp is not None:
                    gt_trunk_val = gt_trunk_interp(target_t)
                    if np.isfinite(gt_trunk_val):
                        trunk_records.append({
                            "Time": curr_t, "Activity": activity, "Scenario": scenario,
                            "Estimated_deg": est_trunk_aligned,
                            "GroundTruth_deg": abs(gt_trunk_val),
                            "Error": abs(est_trunk_aligned - abs(gt_trunk_val)),
                        })

        # 4c. MPJPE (SUPPORTING metric)
        for y_idx, x_name in JOINT_POSITION_MAPPING.items():
            if x_name not in xsens_pos_interp:
                continue
            x_pos = xsens_pos_interp[x_name](target_t)
            y_pos = est_kpts_aligned[i, y_idx]
            if np.isnan(x_pos).any() or np.isnan(y_pos).any():
                continue
            dist = np.linalg.norm(y_pos - x_pos)
            mpjpe_records.append({
                "Time": curr_t, "Activity": activity, "Scenario": scenario,
                "JointName": x_name, "Error": dist,
            })

    df_angles = pd.DataFrame(angle_records)
    df_trunk = pd.DataFrame(trunk_records)
    df_mpjpe = pd.DataFrame(mpjpe_records)

    # --- 5. Print results ---
    print("\n" + "=" * 60)
    print("📐 PRIMARY: Joint Angle Error (degrees)")
    print("=" * 60)

    if not df_angles.empty:
        # Per-joint angle error
        angle_by_joint = df_angles.groupby("AngleName")["Error"].agg(
            MAE="mean", Median="median", Std="std", Samples="count"
        ).sort_values("MAE")
        print("\n[Result] Per-Joint Angle MAE (°)")
        print("-" * 50)
        print(angle_by_joint.to_string())

        # Per-scenario angle error
        angle_by_scenario = df_angles.groupby("Scenario")["Error"].agg(
            MAE="mean", Median="median", Std="std", Samples="count"
        ).sort_values("MAE")
        print("\n[Result] Angle MAE by Scenario (°)")
        print("-" * 50)
        print(angle_by_scenario.to_string())

        # Overall
        overall_angle_mae = df_angles["Error"].mean()
        overall_angle_med = df_angles["Error"].median()
        print(f"\n[Result] Overall Joint Angle MAE:    {overall_angle_mae:.2f}°")
        print(f"[Result] Overall Joint Angle Median: {overall_angle_med:.2f}°")
        print(f"[Result] Total valid angle samples:  {len(df_angles)}")

        # RULA threshold classification accuracy for elbow (flexion)
        elbow_df = df_angles[df_angles["AngleName"].str.contains("Elbow")]
        if not elbow_df.empty:
            thresholds = RULA_THRESHOLDS["LowerArm"]
            est_classes = elbow_df["Estimated_deg"].apply(
                lambda x: classify_angle_by_thresholds(abs(x), thresholds))
            gt_classes = elbow_df["GroundTruth_deg"].apply(
                lambda x: classify_angle_by_thresholds(abs(x), thresholds))
            accuracy = (est_classes == gt_classes).mean()
            print(f"\n[Result] Elbow RULA-category accuracy: {accuracy:.2%}")
    else:
        print("[Warning] No valid angle samples computed.")

    if not df_trunk.empty:
        print(f"\n[Result] Trunk Flexion MAE:    {df_trunk['Error'].mean():.2f}°")
        print(f"[Result] Trunk Flexion Median: {df_trunk['Error'].median():.2f}°")
        print(f"[Result] Trunk valid samples:  {len(df_trunk)}")

    print("\n" + "=" * 60)
    print("📏 SUPPORTING: MPJPE (cm) — Spatial Reference Only")
    print("=" * 60)

    if not df_mpjpe.empty:
        overall_mpjpe = df_mpjpe["Error"].mean()
        mpjpe_median = df_mpjpe["Error"].median()
        print(f"\n[Result] Overall MPJPE:    {overall_mpjpe:.2f} cm")
        print(f"[Result] Overall Median:   {mpjpe_median:.2f} cm")

        mpjpe_by_scenario = df_mpjpe.groupby("Scenario")["Error"].agg(
            MPJPE="mean", Median="median", Samples="count"
        ).sort_values("MPJPE")
        print("\n[Result] MPJPE by Scenario (cm)")
        print("-" * 50)
        print(mpjpe_by_scenario.to_string())

    # --- 6. Save CSV results ---
    core_metrics = {
        "Joint_Angle_MAE_deg": df_angles["Error"].mean() if not df_angles.empty else np.nan,
        "Joint_Angle_Median_deg": df_angles["Error"].median() if not df_angles.empty else np.nan,
        "Trunk_Flexion_MAE_deg": df_trunk["Error"].mean() if not df_trunk.empty else np.nan,
        "MPJPE_cm": df_mpjpe["Error"].mean() if not df_mpjpe.empty else np.nan,
        "Valid_Angle_Samples": len(df_angles),
        "Valid_Trunk_Samples": len(df_trunk),
        "Valid_MPJPE_Samples": len(df_mpjpe),
    }
    pd.Series(core_metrics).to_frame("Value").to_csv(
        os.path.join(RESULTS_DIR, "eval_core_metrics.csv"))

    if not df_angles.empty:
        angle_by_joint.to_csv(os.path.join(RESULTS_DIR, "eval_angle_by_joint.csv"))
        angle_by_scenario.to_csv(os.path.join(RESULTS_DIR, "eval_angle_by_scenario.csv"))
    if not df_trunk.empty:
        df_trunk.to_csv(os.path.join(RESULTS_DIR, "eval_trunk_flexion.csv"), index=False)

    # --- 7. Visualization ---
    print("\n[Info] Generating plots...")
    sns.set_theme(style="whitegrid")

    # Figure 1: Per-joint angle error bar chart
    if not df_angles.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=angle_by_joint["MAE"].values, y=angle_by_joint.index,
            hue=angle_by_joint.index, palette="crest", legend=False,
        )
        plt.title("Per-Joint Angle MAE (°)", fontsize=14)
        plt.xlabel("Mean Absolute Error (degrees)", fontsize=12)
        plt.ylabel("Joint", fontsize=12)
        plt.tight_layout()
        path = os.path.join(SRC_DIR, "eval_angle_by_joint.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[Info] Saved: {path}")

    # Figure 2: Angle error by scenario
    if not df_angles.empty:
        plt.figure(figsize=(10, 6))
        order = ['Baseline', 'Dynamic Action', 'Occlusion', 'Environmental Interference']
        plot_df = df_angles[df_angles['Scenario'].isin(order)]
        if not plot_df.empty:
            sns.boxplot(
                x='Scenario', y='Error', hue='Scenario', data=plot_df,
                order=order, palette="Set2", showfliers=False, legend=False,
            )
            plt.title("Joint Angle Error Distribution by Scenario", fontsize=14)
            plt.ylabel("Absolute Angle Error (°)", fontsize=12)
            plt.xlabel("Scenario", fontsize=12)
            plt.tight_layout()
            path = os.path.join(SRC_DIR, "eval_angle_by_scenario.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"[Info] Saved: {path}")

    # Figure 3: Trunk flexion comparison (time-series)
    if not df_trunk.empty:
        plt.figure(figsize=(14, 5))
        plt.plot(df_trunk["Time"], df_trunk["GroundTruth_deg"],
                 'k-', alpha=0.6, linewidth=1, label="Xsens GT (Pelvis_T8)")
        plt.plot(df_trunk["Time"], df_trunk["Estimated_deg"],
                 'r-', alpha=0.7, linewidth=1, label="Estimated (Stereo Vision)")
        plt.title("Trunk Flexion: Ground Truth vs. Estimated", fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Trunk Flexion Angle (°)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        path = os.path.join(SRC_DIR, "eval_trunk_flexion_compare.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[Info] Saved: {path}")

    print("\n[Info] Evaluation complete.")


if __name__ == "__main__":
    main()
