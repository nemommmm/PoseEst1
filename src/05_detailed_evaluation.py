import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
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
# ===============================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DATA_PATH = os.path.join(PROJECT_ROOT, "results", "yolo_3d_raw.npz")
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")

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

def main():
    print("[Info] Initializing detailed evaluation and scenario analysis...")
    
    # --- 1. Data Loading ---
    yolo_data = np.load(YOLO_DATA_PATH)
    y_kpts = yolo_data['keypoints']
    y_ts = yolo_data['timestamps']
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
    q_elite = xsens_interp['Pelvis'](y_ts[elite_indices] - BEST_OFFSET)
    R, t = kabsch_transform(p_elite, q_elite)
    
    N, J, _ = y_kpts.shape
    y_kpts_flat = y_kpts.reshape(-1, 3)
    y_aligned_flat = (R @ y_kpts_flat.T).T + t
    y_kpts_aligned = y_aligned_flat.reshape(N, J, 3)

    # --- 3. Compute Error Matrix ---
    print("[Info] Calculating segment-wise Euclidean distance errors...")
    records = []
    
    for i, curr_t in enumerate(y_ts):
        target_t = curr_t - BEST_OFFSET
        
        # Determine Activity and Scenario Labels
        activity_label = "Unclassified"
        scenario_label = "Unclassified"
        for label, (start, end) in ACTIVITY_SEGMENTS.items():
            if start <= curr_t <= end:
                activity_label = label
                scenario_label = SCENARIO_MAPPING.get(label, "Other")
                break
        
        # 1. Root Error (Pelvis)
        x_pelvis_pos = xsens_interp['Pelvis'](target_t)
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
            y_pos = y_kpts_aligned[i, y_idx]
            if np.isnan(x_pos).any() or np.isnan(y_pos).any(): continue
            
            dist = np.linalg.norm(y_pos - x_pos)
            part_group = "Other"
            for p_name, p_idxs in BODY_PARTS.items():
                if y_idx in p_idxs: part_group = p_name; break
            
            records.append({
                'Time': curr_t, 'Activity': activity_label, 'Scenario': scenario_label,
                'Type': 'Joint', 'Part': part_group, 'Error': dist
            })

    df = pd.DataFrame(records)
    df_joints = df[df['Type'] == 'Joint']

    # --- 4. Generate Advanced Statistical Tables ---
    
    # [Table 1] Detailed Activity Performance
    print("\n[Result] Table 1: Detailed Activity Performance (cm)")
    print("-" * 60)
    print(df_joints.groupby('Activity')['Error'].mean().sort_values().to_string())

    # [Table 2] Performance by Challenge Scenario
    print("\n[Result] Table 2: Performance by Challenge Scenario (cm)")
    print("-" * 60)
    scenario_stats = df_joints.groupby('Scenario')['Error'].agg(['mean', 'std', 'count']).sort_values('mean')
    print(scenario_stats)
    
    # [Table 3] Body Part Error in Complex Interference Environments
    print("\n[Result] Table 3: Body Part Error in 'Environmental Interference' Region")
    print("-" * 60)
    chair_df = df_joints[df_joints['Scenario'] == 'Environmental Interference']
    if not chair_df.empty:
        print(chair_df.groupby('Part')['Error'].mean().sort_values())
    else:
        print("No data captured in Environmental Interference region.")

    # --- 5. Visualization generation ---
    print("\n[Info] Generating output plots...")
    sns.set_theme(style="whitegrid")
    
    # Figure 1: Boxplot of Scenario Impact
    plt.figure(figsize=(10, 6))
    order = ['Baseline', 'Dynamic Action', 'Occlusion', 'Environmental Interference']
    plot_df = df_joints[df_joints['Scenario'].isin(order)]
    sns.boxplot(x='Scenario', y='Error', hue='Scenario', data=plot_df, order=order, palette="Set2", showfliers=False, legend=False)
    plt.title('Impact of Environmental Challenges on Pose Estimation Error', fontsize=14)
    plt.ylabel('Absolute Position Error (cm)', fontsize=12)
    plt.xlabel('Evaluation Scenario', fontsize=12)
    plt.savefig("eval_scenario_impact.png", dpi=300, bbox_inches='tight')
    
    # Figure 2: Barplot of Mean Error by Detailed Activity
    plt.figure(figsize=(12, 8))
    order = df_joints.groupby('Activity')['Error'].mean().sort_values().index
    sns.barplot(x='Error', y='Activity', hue='Activity', data=df_joints, order=order, palette="viridis", errorbar=None, legend=False)
    plt.title('Mean Error by Activity Segment', fontsize=14)
    plt.xlabel('Mean Position Error (cm)', fontsize=12)
    plt.ylabel('Activity Segment', fontsize=12)
    plt.tight_layout()
    plt.savefig("eval_activity_bar.png", dpi=300, bbox_inches='tight')

    print("[Info] Evaluation complete. Output files 'eval_scenario_impact.png' and 'eval_activity_bar.png' generated.")
    plt.show()

if __name__ == "__main__":
    main()