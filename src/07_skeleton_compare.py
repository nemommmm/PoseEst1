import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser

# ================= 🏆 最终参数 (来自 06_auto_optimizer) =================
BEST_OFFSET = 17.20   # 你的最佳时间偏移
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]
TOP_K = 150

# 📸 抓拍配置
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
MAX_SNAPSHOTS = 5      # 限制保存几张最好的，防止存太多
MIN_TIME_GAP = 2.0     # 抓拍间隔至少 2 秒，防止存重复动作

# ================= 💀 骨架定义 =================
# YOLO (COCO) 连接
YOLO_EDGES = [
    (0,1), (0,2), (1,3), (2,4),          # Face
    (5,6), (5,7), (7,9), (6,8), (8,10),  # Arms
    (5,11), (6,12),                      # Torso
    (11,12), (11,13), (13,15), (12,14), (14,16) # Legs
]

# Xsens 需要读取的部位名称
XSENS_SEGMENTS_TO_LOAD = [
    'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
    'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
    'RightUpperLeg', 'RightLowerLeg', 'RightFoot',
    'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot'
]

# Xsens 画线逻辑 (Parent -> Child)
# 注意：这里是用上面列表的 index 连接
XSENS_LINKS = [
    ('Pelvis', 'L5'), ('L5', 'L3'), ('L3', 'T12'), ('T12', 'T8'), ('T8', 'Neck'), ('Neck', 'Head'), # 脊柱
    ('T8', 'RightShoulder'), ('RightShoulder', 'RightUpperArm'), ('RightUpperArm', 'RightForeArm'), ('RightForeArm', 'RightHand'), # 右臂
    ('T8', 'LeftShoulder'), ('LeftShoulder', 'LeftUpperArm'), ('LeftUpperArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'), # 左臂
    ('Pelvis', 'RightUpperLeg'), ('RightUpperLeg', 'RightLowerLeg'), ('RightLowerLeg', 'RightFoot'), # 右腿
    ('Pelvis', 'LeftUpperLeg'), ('LeftUpperLeg', 'LeftLowerLeg'), ('LeftLowerLeg', 'LeftFoot') # 左腿
]
# ==================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DATA_PATH = os.path.join(PROJECT_ROOT, "results", "yolo_3d_raw.npz")
MVNX_PATH = os.path.join(PROJECT_ROOT,"..", "Xsens_ground_truth", "Aitor-001.mvnx")

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
    print(f"🎬 启动最终可视化 (Full Skeleton + Snapshot)...")

    # 1. 加载 YOLO
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

    # 2. 加载 Xsens (全骨架)
    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    
    # 建立 Xsens 数据字典: {'Pelvis': (N,3), 'Head': (N,3), ...}
    xsens_data = {}
    xsens_ts = mvnx.timestamps
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]

    print("📥 读取 Xsens 全身骨骼数据...")
    for seg_name in XSENS_SEGMENTS_TO_LOAD:
        raw_data = mvnx.get_segment_data(seg_name)
        if raw_data is not None:
            xsens_data[seg_name] = raw_data[xidx]
        else:
            print(f"⚠️ 警告: MVNX 中找不到部位 {seg_name}")

    # 插值器字典
    xsens_interp = {}
    for name, data in xsens_data.items():
        xsens_interp[name] = interp1d(xsens_ts, data, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)

    # 3. 计算旋转矩阵 (使用 Pelvis 中心)
    # 这里的 f_x_pelvis 专门用于计算对齐
    f_x_pelvis = xsens_interp['Pelvis']
    
    y_pelvis_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    errors = calculate_limb_error(y_kpts, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    sorted_idx = np.argsort(errors[valid_err_idx])
    elite_indices = valid_err_idx[sorted_idx[:TOP_K]]
    
    p_elite = y_pelvis_center[elite_indices]
    t_elite_shifted = y_ts[elite_indices] - BEST_OFFSET
    q_elite = f_x_pelvis(t_elite_shifted)
    
    print("📐 计算最佳旋转矩阵...")
    R_mat, t_vec = kabsch_transform(p_elite, q_elite)

    # 4. 变换 YOLO (应用到全身)
    N, J, _ = y_kpts.shape
    y_kpts_flat = y_kpts.reshape(-1, 3)
    y_kpts_transformed = (R_mat @ y_kpts_flat.T).T + t_vec
    y_kpts_final = y_kpts_transformed.reshape(N, J, 3)

    # 5. 动画展示 + 自动抓拍
    print("🎥 开始播放 (将自动保存最佳帧)...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    start_time = max(0, BEST_OFFSET)
    end_time = min(y_ts[-1], xsens_ts[-1] + BEST_OFFSET)
    step = 5
    
    snapshots_taken = 0
    last_snapshot_time = -999

    for i in range(0, len(y_ts), step):
        curr_time = y_ts[i]
        if curr_time < start_time or curr_time > end_time: continue
        
        target_x_time = curr_time - BEST_OFFSET
        y_pose = y_kpts_final[i]
        
        # 获取当前时刻所有 Xsens 骨骼点
        x_pose = {}
        for name, func in xsens_interp.items():
            pos = func(target_x_time)
            if not np.isnan(pos).any():
                x_pose[name] = pos
        
        if 'Pelvis' not in x_pose: continue

        # 计算当前帧误差 (Pelvis Distance)
        curr_error = np.linalg.norm(y_pose[11:13].mean(0) - x_pose['Pelvis'])

        ax.cla()
        
        # --- 画 Xsens (黑色/灰色) ---
        # 画点
        for name, pos in x_pose.items():
            ax.scatter(pos[0], pos[1], pos[2], c='k', s=10, alpha=0.5)
        # 画线
        for p1_name, p2_name in XSENS_LINKS:
            if p1_name in x_pose and p2_name in x_pose:
                p1 = x_pose[p1_name]
                p2 = x_pose[p2_name]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='k', linewidth=1.5, alpha=0.6)
        
        # 手动添加标签 (只加一次防止图例混乱)
        ax.plot([], [], [], c='k', label='Ground Truth (Xsens)')

        # --- 画 YOLO (红色) ---
        ax.scatter(y_pose[:, 0], y_pose[:, 1], y_pose[:, 2], c='r', s=25, label='YOLO Stereo')
        for edge in YOLO_EDGES:
            pt1, pt2 = y_pose[edge[0]], y_pose[edge[1]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='r', linewidth=2)

        # 视角跟随
        center = x_pose['Pelvis']
        ax.set_xlim(center[0]-100, center[0]+100)
        ax.set_ylim(center[1]-100, center[1]+100)
        ax.set_zlim(center[2]-100, center[2]+100)
        
        title_str = f"Time: {curr_time:.2f}s | Error: {curr_error:.1f} cm"
        ax.set_title(title_str)
        ax.legend(loc='upper right')

        # --- 自动抓拍逻辑 ---
        # 条件: 误差 < 20cm (非常准) + 间隔 > 2s + 没拍够数量
        if curr_error < 20.0 and (curr_time - last_snapshot_time) > MIN_TIME_GAP and snapshots_taken < MAX_SNAPSHOTS:
            filename = os.path.join(SNAPSHOT_DIR, f"snapshot_t{curr_time:.1f}_err{curr_error:.1f}.png")
            plt.savefig(filename, dpi=150)
            print(f"📸 已抓拍优质帧: {filename}")
            last_snapshot_time = curr_time
            snapshots_taken += 1

        plt.pause(0.001)
        if not plt.fignum_exists(fig.number): break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()