import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser

# ================= 🔧 配置区域 =================
BEST_OFFSET = 17.20
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]
TOP_K = 150

# 🕒 详细的时间分段 (基于你的人工标注)
# 格式: "Label": [start_time, end_time]
ACTIVITY_SEGMENTS = {
    # --- 基准组 (Baseline) ---
    "Walking (Normal)": [17, 32],
    "Walking (Late)": [220, 240], # 后面比较正常的走路

    # --- 遮挡组 (Occlusion) ---
    "Sitting (Lower Occluded)": [32, 62],     # 坐着，下半身遮挡
    "Walking (Upper Occluded)": [87, 97],     # 上半身遮挡
    "Walking (Lower Occluded 1)": [130, 140], # 下半身遮挡
    "Walking (Lower Occluded 2)": [164, 170], # 下半身遮挡

    # --- 黑色椅子/复杂交互组 (Black Chair / Complex Env) ---
    "Chair Interaction (Complex)": [140, 160], # 在黑色椅子附近做动作
    "Lifting Box (Near Chair)": [214, 218],    # 抬箱子，也在椅子附近
    
    # --- 特殊动作 ---
    "Squatting": [66, 69],
    "Squatting (Check)": [156, 160]
}

# 🏷️ 场景分类映射 (用于生成高级统计表)
# 将上面的详细动作归类为三大挑战场景
SCENARIO_MAPPING = {
    "Walking (Normal)": "Baseline (Normal)",
    "Walking (Late)": "Baseline (Normal)",
    
    "Sitting (Lower Occluded)": "Occlusion",
    "Walking (Upper Occluded)": "Occlusion",
    "Walking (Lower Occluded 1)": "Occlusion",
    "Walking (Lower Occluded 2)": "Occlusion",
    
    "Chair Interaction (Complex)": "Black Chair Area",
    "Lifting Box (Near Chair)": "Black Chair Area",
    
    "Squatting": "Dynamic Action",
    "Squatting (Check)": "Dynamic Action"
}

# 💀 部位定义
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
    print("📊 启动详细评估 (含场景与遮挡分析)...")
    
    # --- 1. 数据加载 ---
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

    # --- 2. 全局对齐 ---
    print("📐 执行全局对齐...")
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

    # --- 3. 计算误差矩阵 ---
    print("🧮 正在计算逐个场景的误差...")
    records = []
    
    for i, curr_t in enumerate(y_ts):
        target_t = curr_t - BEST_OFFSET
        
        # 确定 Activity 和 Scenario
        activity_label = "Unclassified"
        scenario_label = "Unclassified"
        for label, (start, end) in ACTIVITY_SEGMENTS.items():
            if start <= curr_t <= end:
                activity_label = label
                scenario_label = SCENARIO_MAPPING.get(label, "Other")
                break
        
        # 1. 根节点误差
        x_pelvis_pos = xsens_interp['Pelvis'](target_t)
        if not np.isnan(x_pelvis_pos).any():
            y_root = (y_kpts_aligned[i, 11] + y_kpts_aligned[i, 12]) / 2.0
            root_err = np.linalg.norm(y_root - x_pelvis_pos)
            records.append({
                'Time': curr_t, 'Activity': activity_label, 'Scenario': scenario_label,
                'Type': 'Root', 'Error': root_err
            })
        
        # 2. 关节误差
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

    # --- 4. 生成高级统计表 ---
    
    # [Table 1] 按具体活动 (Activity)
    print("\n🏆 [Table 1] Detailed Activity Performance (cm)")
    print("-" * 60)
    print(df_joints.groupby('Activity')['Error'].mean().sort_values().to_string())

    # [Table 2] 按场景分类 (Scenario) - 这是重点！
    print("\n🏆 [Table 2] Performance by Challenge Scenario (cm)")
    print("-" * 60)
    scenario_stats = df_joints.groupby('Scenario')['Error'].agg(['mean', 'std', 'count']).sort_values('mean')
    print(scenario_stats)
    
    # [Table 3] 黑色椅子区域的具体部位分析
    # 我们想看看在椅子附近，是不是腿部误差特别大
    print("\n🏆 [Table 3] Body Part Error in 'Black Chair Area'")
    print("-" * 60)
    chair_df = df_joints[df_joints['Scenario'] == 'Black Chair Area']
    if not chair_df.empty:
        print(chair_df.groupby('Part')['Error'].mean().sort_values())
    else:
        print("No data captured in Black Chair Area.")

    # --- 5. 绘图 ---
    sns.set_theme(style="whitegrid")
    
    # Figure 1: 场景对比箱线图
    plt.figure(figsize=(10, 6))
    order = ['Baseline (Normal)', 'Dynamic Action', 'Occlusion', 'Black Chair Area']
    # 过滤掉 Unclassified 绘图
    plot_df = df_joints[df_joints['Scenario'].isin(order)]
    # sns.boxplot(x='Scenario', y='Error', data=plot_df, order=order, palette="Set2", showfliers=False)
    sns.boxplot(x='Scenario', y='Error', hue='Scenario', data=plot_df, order=order, palette="Set2", showfliers=False, legend=False)
    plt.title('Impact of Environmental Challenges on Error')
    plt.ylabel('Position Error (cm)')
    plt.savefig("eval_scenario_impact.png")
    
    # Figure 2: 详细活动误差条形图
    plt.figure(figsize=(12, 8))
    # 按误差大小排序
    order = df_joints.groupby('Activity')['Error'].mean().sort_values().index
    sns.barplot(x='Error', y='Activity', data=df_joints, order=order, palette="viridis", ci=None)
    plt.title('Mean Error by Activity Segment')
    plt.xlabel('Mean Error (cm)')
    plt.tight_layout()
    plt.savefig("eval_activity_bar.png")

    print("\n✅ 评估完成！重点关注 Table 2 和 eval_scenario_impact.png")
    plt.show()

if __name__ == "__main__":
    main()