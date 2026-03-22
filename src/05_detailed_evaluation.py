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
# Best temporal offset derived from the automated grid search optimizer
BEST_OFFSET = 17.20
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5] # Reference limb lengths (cm)
TOP_K = 150 # Number of elite frames

# 🕒 Detailed Time Segmentation (Manual Annotation)
# Format: "Label": [start_time, end_time] in seconds
ACTIVITY_SEGMENTS = {
    # --- Baseline Group ---
    "Walking (Normal)": [17, 32],
    "Walking (Late)": [220, 240], # Stable walking phase at the end

    # --- Occlusion Group ---
    "Sitting (Lower Occluded)": [32, 62],     # Lower body heavily occluded by desk
    "Walking (Upper Occluded)": [87, 97],     # Upper body partially occluded
    "Walking (Lower Occluded 1)": [130, 140], # Lower body occluded by obstacles
    "Walking (Lower Occluded 2)": [164, 170], # Lower body occluded by obstacles

    # --- Environmental Interference (Black Chair) ---
    "Chair Interaction (Complex)": [140, 160], # Complex interaction near a black chair
    "Lifting Box (Near Chair)": [214, 218],    # Lifting action near the chair
    
    # --- Dynamic/Special Actions ---
    "Squatting": [66, 69],
    "Squatting (Check)": [156, 160]
}

# 🏷️ High-Level Scenario Mapping (For aggregated statistical tables)
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
    "Squatting (Check)": "Dynamic Action"
}

# 💀 Kinematic Body Part Grouping
BODY_PARTS = {
    "Head": [0, 1, 2, 3, 4],
    "Torso": [5, 6, 11, 12],
    "Arms": [7, 8, 9, 10],
    "Legs": [13, 14, 15, 16]
}

JOINT_MAPPING = {
    0: 'Head', 11: 'Pelvis', 12: 'Pelvis',
    5: 'LeftShoulder', 6: 'RightShoulder',
    7: 'LeftUpperArm', 8: 'RightUpperArm',
    9: 'LeftForeArm', 10: 'RightForeArm',
    13: 'LeftUpperLeg', 14: 'RightUpperLeg',
    15: 'LeftLowerLeg', 16: 'RightLowerLeg'
}

ANGLE_DEFINITIONS = {
    "LeftElbow": {
        "est_triplet": (5, 7, 9),
        "gt_triplet": ("LeftShoulder", "LeftUpperArm", "LeftForeArm"),
    },
    "RightElbow": {
        "est_triplet": (6, 8, 10),
        "gt_triplet": ("RightShoulder", "RightUpperArm", "RightForeArm"),
    },
    "LeftKnee": {
        "est_triplet": (11, 13, 15),
        "gt_triplet": ("Pelvis", "LeftUpperLeg", "LeftLowerLeg"),
    },
    "RightKnee": {
        "est_triplet": (12, 14, 16),
        "gt_triplet": ("Pelvis", "RightUpperLeg", "RightLowerLeg"),
    },
    "LeftShoulder": {
        "est_triplet": (11, 5, 7),
        "gt_triplet": ("Pelvis", "LeftShoulder", "LeftUpperArm"),
    },
    "RightShoulder": {
        "est_triplet": (12, 6, 8),
        "gt_triplet": ("Pelvis", "RightShoulder", "RightUpperArm"),
    },
    "LeftHip": {
        "est_triplet": (5, 11, 13),
        "gt_triplet": ("LeftShoulder", "Pelvis", "LeftUpperLeg"),
    },
    "RightHip": {
        "est_triplet": (6, 12, 14),
        "gt_triplet": ("RightShoulder", "Pelvis", "RightUpperLeg"),
    },
}
# ===============================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")
SUMMARY_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
ALIGNMENT_SUMMARY_PATH = os.path.join(SUMMARY_OUTPUT_DIR, "alignment_summary.json")
os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

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
        np.linalg.norm(kpts[:, 14] - kpts[:, 16], axis=1) 
    ]).T
    return np.sum(np.abs(lengths - gt_lengths), axis=1)

def kabsch_transform(P, Q):
    mask = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P = P[mask]
    Q = Q[mask]
    if len(P) < 10: return np.eye(3), np.zeros(3)
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    AA = P - centroid_P
    BB = Q - centroid_Q
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    rot = Vt.T @ U.T
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
    t = centroid_Q - rot @ centroid_P
    return rot, t

def summarize_metric(df, group_col, mean_col, median_col, std_col):
    return (
        df.groupby(group_col)["Error"]
        .agg(**{mean_col: "mean", median_col: "median", std_col: "std", "Samples": "count"})
        .sort_values(mean_col)
    )

def compute_angle_deg(p1, p2, p3):
    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
        return np.nan
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def compute_velocity(traj, timestamps):
    vel = np.full_like(traj, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(traj).all(axis=1)
    if valid_mask.sum() < 5:
        return vel

    valid_traj = traj[valid_mask]
    valid_ts = timestamps[valid_mask]
    vel_valid = np.gradient(valid_traj, valid_ts, axis=0, edge_order=1)
    vel[valid_mask] = vel_valid
    return vel

def compute_acceleration(traj, timestamps):
    acc = np.full_like(traj, np.nan, dtype=np.float64)
    vel = compute_velocity(traj, timestamps)
    valid_mask = np.isfinite(vel).all(axis=1)
    if valid_mask.sum() < 5:
        return acc

    valid_traj = vel[valid_mask]
    valid_ts = timestamps[valid_mask]
    acc_valid = np.gradient(valid_traj, valid_ts, axis=0, edge_order=1)
    acc[valid_mask] = acc_valid
    return acc

def scalarize_npz_value(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
    elif arr.size == 1:
        item = arr.reshape(()).item()
    else:
        return arr.tolist()
    return item.decode("utf-8") if isinstance(item, bytes) else item

def build_metric_overview(core_metrics):
    metric_specs = [
        ("Visual", "MPJPE_cm", "cm", "Mean joint position error after one global rigid alignment."),
        ("Visual", "Root_Mean_Error_cm", "cm", "Mean pelvis/root trajectory error."),
        ("Visual", "Median_Joint_Error_cm", "cm", "Median joint position error across all valid samples."),
        ("Visual", "P90_Joint_Error_cm", "cm", "90th percentile joint position error."),
        ("Visual", "P95_Joint_Error_cm", "cm", "95th percentile joint position error."),
        ("Kinematic", "Mean_Joint_Angle_Error_deg", "deg", "Mean absolute joint-angle error from pose triplets."),
        ("Dynamic", "Pelvis_Velocity_Error_cm_s", "cm/s", "Mean pelvis velocity vector error."),
        ("Dynamic", "Mean_Joint_Velocity_Error_cm_s", "cm/s", "Mean joint velocity vector error."),
        ("Dynamic", "Pelvis_Acceleration_Error_cm_s2", "cm/s^2", "Mean pelvis acceleration error."),
        ("Dynamic", "Mean_Joint_Acceleration_Error_cm_s2", "cm/s^2", "Mean joint acceleration error."),
        ("Coverage", "Valid_Joint_Samples", "samples", "Valid aligned joint samples used for MPJPE."),
        ("Coverage", "Valid_Root_Samples", "samples", "Valid pelvis samples used for root error."),
        ("Coverage", "Valid_Angle_Samples", "samples", "Valid joint-angle samples."),
        ("Coverage", "Valid_Velocity_Samples", "samples", "Valid joint-velocity samples."),
        ("Coverage", "Valid_Acceleration_Samples", "samples", "Valid joint-acceleration samples."),
    ]
    rows = []
    for category, metric, unit, definition in metric_specs:
        rows.append(
            {
                "Category": category,
                "Metric": metric,
                "Value": core_metrics.get(metric, np.nan),
                "Unit": unit,
                "Definition": definition,
            }
        )
    return pd.DataFrame(rows)

def build_ergonomic_overview(core_metrics):
    metric_specs = [
        ("Primary", "Posture", "Mean_Joint_Angle_Error_deg", "deg", "Most relevant ergonomic posture signal."),
        ("Secondary", "Dynamics", "Mean_Joint_Velocity_Error_cm_s", "cm/s", "Temporal motion consistency for moving joints."),
        ("Secondary", "Dynamics", "Mean_Joint_Acceleration_Error_cm_s2", "cm/s^2", "Jitter and motion smoothness proxy."),
        ("Secondary", "Dynamics", "Pelvis_Velocity_Error_cm_s", "cm/s", "Whole-body/root motion consistency."),
        ("Secondary", "Dynamics", "Pelvis_Acceleration_Error_cm_s2", "cm/s^2", "Whole-body/root jitter proxy."),
        ("Supporting", "Visual", "MPJPE_cm", "cm", "Useful diagnostic, but less reliable than angle-based evaluation for Xsens."),
        ("Supporting", "Visual", "Root_Mean_Error_cm", "cm", "Useful diagnostic for global alignment only."),
        ("Coverage", "Coverage", "Valid_Angle_Samples", "samples", "Valid posture samples supporting the angle metrics."),
        ("Coverage", "Coverage", "Valid_Velocity_Samples", "samples", "Valid dynamic samples supporting the velocity metrics."),
        ("Coverage", "Coverage", "Valid_Acceleration_Samples", "samples", "Valid dynamic samples supporting the acceleration metrics."),
    ]
    rows = []
    for priority, dimension, metric, unit, interpretation in metric_specs:
        rows.append(
            {
                "Priority": priority,
                "Dimension": dimension,
                "Metric": metric,
                "Value": core_metrics.get(metric, np.nan),
                "Unit": unit,
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)

def main():
    print("[Info] Initializing detailed evaluation and scenario analysis...")
    yolo_data_path = resolve_yolo_data_path()
    best_offset = resolve_best_offset()
    print(f"[Info] Loading pose data from: {os.path.basename(yolo_data_path)}")
    print(f"[Info] Using temporal offset: {best_offset:.2f} s")
    
    # --- 1. Data Loading ---
    yolo_data = np.load(yolo_data_path)
    y_kpts = yolo_data['keypoints']
    y_ts = yolo_data['timestamps']
    pose_model_name = scalarize_npz_value(yolo_data["model_name"]) if "model_name" in yolo_data else "unknown"
    postprocess_variant = scalarize_npz_value(yolo_data["postprocess_variant"]) if "postprocess_variant" in yolo_data else "unknown"
    reprojection_threshold = scalarize_npz_value(yolo_data["reprojection_threshold_px"]) if "reprojection_threshold_px" in yolo_data else np.nan
    print(f"[Info] Pose model: {pose_model_name}")
    print(f"[Info] Postprocess variant: {postprocess_variant}")
    if np.isfinite(reprojection_threshold):
        print(f"[Info] Reprojection threshold: {float(reprojection_threshold):.1f} px")
    y_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    valid_mask = (y_center[:, 2] > 10) & (y_center[:, 2] < 1000) & np.isfinite(y_center).all(axis=1)
    y_kpts = y_kpts[valid_mask]
    y_ts = y_ts[valid_mask]
    y_ts, uidx = np.unique(y_ts, return_index=True)
    y_kpts = y_kpts[uidx]
    y_ts -= y_ts[0]

    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    xsens_ts = mvnx.timestamps
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    
    xsens_interp = {}
    all_segments = set(JOINT_MAPPING.values())
    for seg in all_segments:
        data = mvnx.get_segment_data(seg)[xidx]
        xsens_interp[seg] = interp1d(xsens_ts, data, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)

    # --- 2. Global Alignment ---
    print("[Info] Executing global spatial alignment...")
    y_pelvis = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    errors = calculate_limb_error(y_kpts, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    elite_indices = valid_err_idx[np.argsort(errors[valid_err_idx])[:TOP_K]]
    
    p_elite = y_pelvis[elite_indices]
    q_elite = xsens_interp['Pelvis'](y_ts[elite_indices] - best_offset)
    R, t = kabsch_transform(p_elite, q_elite)
    
    N, J, _ = y_kpts.shape
    y_kpts_flat = y_kpts.reshape(-1, 3)
    y_aligned_flat = (R @ y_kpts_flat.T).T + t
    y_kpts_aligned = y_aligned_flat.reshape(N, J, 3)

    # --- 3. Compute Error Matrix ---
    print("[Info] Calculating segment-wise Euclidean distance errors...")
    records = []
    angle_records = []
    velocity_records = []
    accel_records = []
    gt_tracks = {
        joint_name: np.full((len(y_ts), 3), np.nan, dtype=np.float64)
        for joint_name in set(JOINT_MAPPING.values())
    }
    activity_labels = []
    scenario_labels = []
    
    for i, curr_t in enumerate(y_ts):
        target_t = curr_t - best_offset
        
        # Determine Activity and Scenario Labels
        activity_label = "Unclassified"
        scenario_label = "Unclassified"
        for label, (start, end) in ACTIVITY_SEGMENTS.items():
            if start <= curr_t <= end:
                activity_label = label
                scenario_label = SCENARIO_MAPPING.get(label, "Other")
                break
        activity_labels.append(activity_label)
        scenario_labels.append(scenario_label)
        
        # 1. Root Error (Pelvis)
        x_pelvis_pos = xsens_interp['Pelvis'](target_t)
        gt_tracks['Pelvis'][i] = x_pelvis_pos
        if not np.isnan(x_pelvis_pos).any():
            y_root = (y_kpts_aligned[i, 11] + y_kpts_aligned[i, 12]) / 2.0
            root_err = np.linalg.norm(y_root - x_pelvis_pos)
            records.append({
                'Time': curr_t, 'Activity': activity_label, 'Scenario': scenario_label,
                'Type': 'Root', 'Error': root_err
            })
        
        # 2. Joint Errors
        for y_idx, x_name in JOINT_MAPPING.items():
            x_pos = xsens_interp[x_name](target_t)
            gt_tracks[x_name][i] = x_pos
            y_pos = y_kpts_aligned[i, y_idx]
            if np.isnan(x_pos).any() or np.isnan(y_pos).any(): continue
            
            dist = np.linalg.norm(y_pos - x_pos)
            part_group = "Other"
            for p_name, p_idxs in BODY_PARTS.items():
                if y_idx in p_idxs: part_group = p_name; break
            
            records.append({
                'Time': curr_t, 'Activity': activity_label, 'Scenario': scenario_label,
                'Type': 'Joint', 'Part': part_group, 'JointName': x_name, 'Error': dist
            })

        # 3. Joint Angle Errors
        for angle_name, angle_def in ANGLE_DEFINITIONS.items():
            est_idx_a, est_idx_b, est_idx_c = angle_def["est_triplet"]
            gt_name_a, gt_name_b, gt_name_c = angle_def["gt_triplet"]
            est_angle = compute_angle_deg(
                y_kpts_aligned[i, est_idx_a],
                y_kpts_aligned[i, est_idx_b],
                y_kpts_aligned[i, est_idx_c],
            )
            gt_angle = compute_angle_deg(
                gt_tracks[gt_name_a][i],
                gt_tracks[gt_name_b][i],
                gt_tracks[gt_name_c][i],
            )
            if np.isnan(est_angle) or np.isnan(gt_angle):
                continue
            angle_records.append(
                {
                    "Time": curr_t,
                    "Activity": activity_label,
                    "Scenario": scenario_label,
                    "AngleName": angle_name,
                    "Estimated_deg": est_angle,
                    "GroundTruth_deg": gt_angle,
                    "Error": abs(est_angle - gt_angle),
                }
            )

    df = pd.DataFrame(records)
    df_joints = df[df['Type'] == 'Joint']
    df_root = df[df['Type'] == 'Root']
    df_angles = pd.DataFrame(angle_records)

    activity_labels = np.array(activity_labels)
    scenario_labels = np.array(scenario_labels)

    # --- 3b. Velocity / Acceleration Proxies ---
    for y_idx, x_name in JOINT_MAPPING.items():
        est_vel = compute_velocity(y_kpts_aligned[:, y_idx], y_ts)
        gt_vel = compute_velocity(gt_tracks[x_name], y_ts)
        vel_err = np.linalg.norm(est_vel - gt_vel, axis=1)
        for i, curr_t in enumerate(y_ts):
            if not np.isfinite(vel_err[i]):
                continue
            velocity_records.append(
                {
                    "Time": curr_t,
                    "Activity": activity_labels[i],
                    "Scenario": scenario_labels[i],
                    "JointName": x_name,
                    "Error": vel_err[i],
                }
            )
    df_velocity = pd.DataFrame(velocity_records)

    for y_idx, x_name in JOINT_MAPPING.items():
        est_acc = compute_acceleration(y_kpts_aligned[:, y_idx], y_ts)
        gt_acc = compute_acceleration(gt_tracks[x_name], y_ts)
        acc_err = np.linalg.norm(est_acc - gt_acc, axis=1)
        for i, curr_t in enumerate(y_ts):
            if not np.isfinite(acc_err[i]):
                continue
            accel_records.append(
                {
                    "Time": curr_t,
                    "Activity": activity_labels[i],
                    "Scenario": scenario_labels[i],
                    "JointName": x_name,
                    "Error": acc_err[i],
                }
            )
    df_accel = pd.DataFrame(accel_records)

    # --- 4. Core Metrics and Statistical Tables ---
    core_metrics = pd.Series(
        {
            "MPJPE_cm": df_joints["Error"].mean(),
            "Root_Mean_Error_cm": df_root["Error"].mean(),
            "Median_Joint_Error_cm": df_joints["Error"].median(),
            "P90_Joint_Error_cm": df_joints["Error"].quantile(0.90),
            "P95_Joint_Error_cm": df_joints["Error"].quantile(0.95),
            "Mean_Joint_Angle_Error_deg": df_angles["Error"].mean() if not df_angles.empty else np.nan,
            "Pelvis_Velocity_Error_cm_s": df_velocity[df_velocity["JointName"] == "Pelvis"]["Error"].mean() if not df_velocity.empty else np.nan,
            "Mean_Joint_Velocity_Error_cm_s": df_velocity["Error"].mean() if not df_velocity.empty else np.nan,
            "Pelvis_Acceleration_Error_cm_s2": df_accel[df_accel["JointName"] == "Pelvis"]["Error"].mean() if not df_accel.empty else np.nan,
            "Mean_Joint_Acceleration_Error_cm_s2": df_accel["Error"].mean() if not df_accel.empty else np.nan,
            "Valid_Joint_Samples": int(len(df_joints)),
            "Valid_Root_Samples": int(len(df_root)),
            "Valid_Angle_Samples": int(len(df_angles)),
            "Valid_Velocity_Samples": int(len(df_velocity)),
            "Valid_Acceleration_Samples": int(len(df_accel)),
        }
    )
    activity_stats = summarize_metric(df_joints, "Activity", "MPJPE_cm", "Median_cm", "Std_cm")
    scenario_stats = summarize_metric(df_joints, "Scenario", "MPJPE_cm", "Median_cm", "Std_cm")
    joint_stats = summarize_metric(df_joints, "JointName", "MPJPE_cm", "Median_cm", "Std_cm")
    part_stats = summarize_metric(df_joints, "Part", "MPJPE_cm", "Median_cm", "Std_cm")
    angle_stats = summarize_metric(df_angles, "AngleName", "AngleError_deg", "Median_deg", "Std_deg") if not df_angles.empty else pd.DataFrame()
    velocity_stats = summarize_metric(df_velocity, "JointName", "VelocityError_cm_s", "Median_cm_s", "Std_cm_s") if not df_velocity.empty else pd.DataFrame()
    accel_stats = summarize_metric(df_accel, "JointName", "AccelError_cm_s2", "Median_cm_s2", "Std_cm_s2") if not df_accel.empty else pd.DataFrame()
    angle_scenario_stats = summarize_metric(df_angles, "Scenario", "AngleError_deg", "Median_deg", "Std_deg") if not df_angles.empty else pd.DataFrame()
    velocity_scenario_stats = summarize_metric(df_velocity, "Scenario", "VelocityError_cm_s", "Median_cm_s", "Std_cm_s") if not df_velocity.empty else pd.DataFrame()
    accel_scenario_stats = summarize_metric(df_accel, "Scenario", "AccelError_cm_s2", "Median_cm_s2", "Std_cm_s2") if not df_accel.empty else pd.DataFrame()
    metric_overview = build_metric_overview(core_metrics)
    ergonomic_overview = build_ergonomic_overview(core_metrics)

    print("\n[Result] Ergonomic Overview")
    print("-" * 72)
    print(ergonomic_overview[["Priority", "Dimension", "Metric", "Value", "Unit"]].to_string(index=False))

    print("\n[Result] Metric Overview")
    print("-" * 72)
    for category in ["Visual", "Kinematic", "Dynamic", "Coverage"]:
        section = metric_overview[metric_overview["Category"] == category][["Metric", "Value", "Unit"]]
        print(f"\n[{category}]")
        print(section.to_string(index=False))

    print("\n[Result] MPJPE by Activity (cm)")
    print("-" * 60)
    print(activity_stats.to_string())

    print("\n[Result] MPJPE by Scenario (cm)")
    print("-" * 60)
    print(scenario_stats)
    
    print("\n[Result] MPJPE by Joint (cm)")
    print("-" * 60)
    print(joint_stats.to_string())

    print("\n[Result] MPJPE by Body Part (cm)")
    print("-" * 60)
    print(part_stats.to_string())

    print("\n[Result] Body Part MPJPE in 'Environmental Interference' (cm)")
    print("-" * 60)
    chair_df = df_joints[df_joints['Scenario'] == 'Environmental Interference']
    if not chair_df.empty:
        print(summarize_metric(chair_df, 'Part', 'MPJPE_cm', 'Median_cm', 'Std_cm').to_string())
    else:
        print("No data captured in Environmental Interference region.")

    print("\n[Result] Joint Angle Error (deg)")
    print("-" * 60)
    if not angle_stats.empty:
        print(angle_stats.to_string())
    else:
        print("No valid angle samples.")

    print("\n[Result] Joint Velocity Error (cm/s)")
    print("-" * 60)
    if not velocity_stats.empty:
        print(velocity_stats.to_string())
    else:
        print("No valid velocity samples.")

    print("\n[Result] Joint Acceleration Error (cm/s^2)")
    print("-" * 60)
    if not accel_stats.empty:
        print(accel_stats.to_string())
    else:
        print("No valid acceleration samples.")

    print("\n[Result] Angle Error by Scenario (deg)")
    print("-" * 60)
    if not angle_scenario_stats.empty:
        print(angle_scenario_stats.to_string())
    else:
        print("No valid angle samples.")

    print("\n[Result] Velocity Error by Scenario (cm/s)")
    print("-" * 60)
    if not velocity_scenario_stats.empty:
        print(velocity_scenario_stats.to_string())
    else:
        print("No valid velocity samples.")

    print("\n[Result] Acceleration Error by Scenario (cm/s^2)")
    print("-" * 60)
    if not accel_scenario_stats.empty:
        print(accel_scenario_stats.to_string())
    else:
        print("No valid acceleration samples.")

    core_metrics_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_core_metrics.csv")
    ergonomic_overview_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_ergonomic_overview.csv")
    metric_overview_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_metric_overview.csv")
    activity_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_mpjpe_by_activity.csv")
    scenario_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_mpjpe_by_scenario.csv")
    joint_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_mpjpe_by_joint.csv")
    part_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_mpjpe_by_part.csv")
    angle_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_angle_error_by_joint.csv")
    velocity_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_velocity_error_by_joint.csv")
    accel_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_acceleration_error_by_joint.csv")
    angle_scenario_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_angle_error_by_scenario.csv")
    velocity_scenario_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_velocity_error_by_scenario.csv")
    accel_scenario_stats_path = os.path.join(SUMMARY_OUTPUT_DIR, "eval_acceleration_error_by_scenario.csv")

    core_metrics.to_frame(name="Value").to_csv(core_metrics_path)
    ergonomic_overview.to_csv(ergonomic_overview_path, index=False)
    metric_overview.to_csv(metric_overview_path, index=False)
    activity_stats.to_csv(activity_stats_path)
    scenario_stats.to_csv(scenario_stats_path)
    joint_stats.to_csv(joint_stats_path)
    part_stats.to_csv(part_stats_path)
    if not angle_stats.empty:
        angle_stats.to_csv(angle_stats_path)
    if not velocity_stats.empty:
        velocity_stats.to_csv(velocity_stats_path)
    if not accel_stats.empty:
        accel_stats.to_csv(accel_stats_path)
    if not angle_scenario_stats.empty:
        angle_scenario_stats.to_csv(angle_scenario_stats_path)
    if not velocity_scenario_stats.empty:
        velocity_scenario_stats.to_csv(velocity_scenario_stats_path)
    if not accel_scenario_stats.empty:
        accel_scenario_stats.to_csv(accel_scenario_stats_path)

    # --- 5. Visualization generation ---
    print("\n[Info] Generating output plots...")
    sns.set_theme(style="whitegrid")
    
    # Figure 1: Boxplot of Scenario Impact
    plt.figure(figsize=(10, 6))
    order = ['Baseline', 'Dynamic Action', 'Occlusion', 'Environmental Interference']
    plot_df = df_joints[df_joints['Scenario'].isin(order)]
    sns.boxplot(x='Scenario', y='Error', hue='Scenario', data=plot_df, order=order, palette="Set2", showfliers=False, legend=False)
    plt.title('Scenario-wise Joint Error Distribution (MPJPE Samples)', fontsize=14)
    plt.ylabel('Joint Position Error (cm)', fontsize=12)
    plt.xlabel('Evaluation Scenario', fontsize=12)
    scenario_plot_path = os.path.join(SRC_DIR, "eval_scenario_impact.png")
    plt.savefig(scenario_plot_path, dpi=300, bbox_inches='tight')
    
    # Figure 2: Barplot of Mean Error by Detailed Activity
    plt.figure(figsize=(12, 8))
    order = activity_stats.index
    sns.barplot(x='Error', y='Activity', hue='Activity', data=df_joints, order=order, palette="viridis", errorbar=None, legend=False)
    plt.title('MPJPE by Activity Segment', fontsize=14)
    plt.xlabel('Mean Joint Position Error (cm)', fontsize=12)
    plt.ylabel('Activity Segment', fontsize=12)
    plt.tight_layout()
    activity_plot_path = os.path.join(SRC_DIR, "eval_activity_bar.png")
    plt.savefig(activity_plot_path, dpi=300, bbox_inches='tight')

    # Figure 3: Per-joint MPJPE ranking
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=joint_stats["MPJPE_cm"].values,
        y=joint_stats.index,
        hue=joint_stats.index,
        palette="mako",
        legend=False,
    )
    plt.title('Per-joint MPJPE Ranking', fontsize=14)
    plt.xlabel('MPJPE (cm)', fontsize=12)
    plt.ylabel('Joint / Segment Proxy', fontsize=12)
    plt.tight_layout()
    joint_plot_path = os.path.join(SRC_DIR, "eval_joint_mpjpe_bar.png")
    plt.savefig(joint_plot_path, dpi=300, bbox_inches='tight')

    angle_plot_path = None
    if not angle_stats.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=angle_stats["AngleError_deg"].values,
            y=angle_stats.index,
            hue=angle_stats.index,
            palette="crest",
            legend=False,
        )
        plt.title('Per-joint Angle Error Ranking', fontsize=14)
        plt.xlabel('Mean Absolute Angle Error (deg)', fontsize=12)
        plt.ylabel('Joint Angle', fontsize=12)
        plt.tight_layout()
        angle_plot_path = os.path.join(SRC_DIR, "eval_joint_angle_error_bar.png")
        plt.savefig(angle_plot_path, dpi=300, bbox_inches='tight')

    velocity_plot_path = None
    if not velocity_stats.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=velocity_stats["VelocityError_cm_s"].values,
            y=velocity_stats.index,
            hue=velocity_stats.index,
            palette="flare",
            legend=False,
        )
        plt.title('Per-joint Velocity Error Ranking', fontsize=14)
        plt.xlabel('Mean Velocity Error (cm/s)', fontsize=12)
        plt.ylabel('Joint / Segment Proxy', fontsize=12)
        plt.tight_layout()
        velocity_plot_path = os.path.join(SRC_DIR, "eval_joint_velocity_error_bar.png")
        plt.savefig(velocity_plot_path, dpi=300, bbox_inches='tight')

    accel_plot_path = None
    if not accel_stats.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=accel_stats["AccelError_cm_s2"].values,
            y=accel_stats.index,
            hue=accel_stats.index,
            palette="rocket",
            legend=False,
        )
        plt.title('Per-joint Acceleration Error Ranking', fontsize=14)
        plt.xlabel('Mean Acceleration Error (cm/s^2)', fontsize=12)
        plt.ylabel('Joint / Segment Proxy', fontsize=12)
        plt.tight_layout()
        accel_plot_path = os.path.join(SRC_DIR, "eval_joint_accel_error_bar.png")
        plt.savefig(accel_plot_path, dpi=300, bbox_inches='tight')

    print("[Info] Evaluation complete.")
    print(f"[Info] Core metrics saved to: {core_metrics_path}")
    print(f"[Info] Ergonomic overview saved to: {ergonomic_overview_path}")
    print(f"[Info] Metric overview saved to: {metric_overview_path}")
    print(f"[Info] Activity MPJPE table saved to: {activity_stats_path}")
    print(f"[Info] Scenario MPJPE table saved to: {scenario_stats_path}")
    print(f"[Info] Joint MPJPE table saved to: {joint_stats_path}")
    print(f"[Info] Body-part MPJPE table saved to: {part_stats_path}")
    if not angle_stats.empty:
        print(f"[Info] Joint angle error table saved to: {angle_stats_path}")
    if not velocity_stats.empty:
        print(f"[Info] Velocity error table saved to: {velocity_stats_path}")
    if not accel_stats.empty:
        print(f"[Info] Acceleration error table saved to: {accel_stats_path}")
    if not angle_scenario_stats.empty:
        print(f"[Info] Scenario angle error table saved to: {angle_scenario_stats_path}")
    if not velocity_scenario_stats.empty:
        print(f"[Info] Scenario velocity error table saved to: {velocity_scenario_stats_path}")
    if not accel_scenario_stats.empty:
        print(f"[Info] Scenario acceleration error table saved to: {accel_scenario_stats_path}")
    print(f"[Info] Plot generated: {scenario_plot_path}")
    print(f"[Info] Plot generated: {activity_plot_path}")
    print(f"[Info] Plot generated: {joint_plot_path}")
    if angle_plot_path is not None:
        print(f"[Info] Plot generated: {angle_plot_path}")
    if velocity_plot_path is not None:
        print(f"[Info] Plot generated: {velocity_plot_path}")
    if accel_plot_path is not None:
        print(f"[Info] Plot generated: {accel_plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
