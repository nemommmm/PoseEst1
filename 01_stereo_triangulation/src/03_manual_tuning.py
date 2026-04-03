import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt
from scipy.interpolate import interp1d

# 导入工具
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # 01_stereo_triangulation/src/
_METHOD_DIR = os.path.dirname(_SRC_DIR)                         # 01_stereo_triangulation/
PROJECT_ROOT = os.path.dirname(_METHOD_DIR)                     # PoseEst1/
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))
from utils_mvnx import MvnxParser

# ================= 🔧 手动调参区域 =================
# 核心参数：时间平移量 (单位：秒)
# 观察你的图，YOLO (红) 似乎比 Xsens (黑) 晚了大概 20~30 秒
# 正数表示：把 YOLO 的时间轴往前推 N 秒 (即 YOLO - N)
TIME_SHIFT_SECONDS = 17.5  

# 策略参数
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]
TOP_K = 150
# =================================================

YOLO_DATA_PATH = os.path.join(_METHOD_DIR, "results", "yolo_3d_raw.npz")
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")

def calculate_limb_error(kpts, gt_lengths):
    lengths = np.vstack([
        np.linalg.norm(kpts[:, 11] - kpts[:, 13], axis=1), 
        np.linalg.norm(kpts[:, 12] - kpts[:, 14], axis=1),
        np.linalg.norm(kpts[:, 13] - kpts[:, 15], axis=1), 
        np.linalg.norm(kpts[:, 14] - kpts[:, 16], axis=1) 
    ]).T
    error = np.sum(np.abs(lengths - gt_lengths), axis=1)
    return error

def kabsch_transform(P, Q):
    # 移除 NaN
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
    print(f"🔧 启动手动微调模式 (Time Shift: {TIME_SHIFT_SECONDS} s)...")

    # 1. 加载数据
    yolo_data = np.load(YOLO_DATA_PATH)
    y_kpts = yolo_data['keypoints']
    y_ts = yolo_data['timestamps']
    y_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    
    # 清洗
    valid_mask = (y_center[:, 2] > 10) & (y_center[:, 2] < 1000) & np.isfinite(y_center).all(axis=1)
    y_kpts = y_kpts[valid_mask]
    y_ts = y_ts[valid_mask]
    y_center = y_center[valid_mask]
    y_ts, uidx = np.unique(y_ts, return_index=True)
    y_kpts = y_kpts[uidx]
    y_center = y_center[uidx]
    y_ts -= y_ts[0] 

    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    x_pelvis = mvnx.get_segment_data('Pelvis')
    x_ts = mvnx.timestamps
    x_ts, xidx = np.unique(x_ts, return_index=True)
    x_pelvis = x_pelvis[xidx]
    x_ts -= x_ts[0]

    # 2. 筛选黄金帧 (计算旋转矩阵用)
    errors = calculate_limb_error(y_kpts, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    sorted_idx = np.argsort(errors[valid_err_idx])
    elite_indices = valid_err_idx[sorted_idx[:TOP_K]]
    print(f"   -> 使用 {len(elite_indices)} 个黄金帧计算旋转")

    # 3. 应用手动时间偏移
    # 目标：找到 YOLO 时间 t 对应的 Xsens 时间 (t - shift)
    f_x = interp1d(x_ts, x_pelvis, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
    
    # 获取黄金帧的配对点
    p_elite = y_center[elite_indices]
    t_elite_shifted = y_ts[elite_indices] - TIME_SHIFT_SECONDS
    q_elite = f_x(t_elite_shifted) # 插值找真值

    # 4. 计算旋转矩阵 (Kabsch)
    R_mat, t_vec = kabsch_transform(p_elite, q_elite)

    # 5. 全局应用与评估
    y_final = (R_mat @ y_center.T).T + t_vec
    
    # 评估所有数据
    t_full_shifted = y_ts - TIME_SHIFT_SECONDS
    x_final_gt = f_x(t_full_shifted)
    
    # 计算误差 (忽略 NaN)
    diff = y_final - x_final_gt
    dist = np.linalg.norm(diff, axis=1)
    valid_dist = dist[np.isfinite(dist)]
    mean_error = np.mean(valid_dist)
    
    print(f"🏆 当前 Offset ({TIME_SHIFT_SECONDS}s) 下的平均误差: {mean_error:.2f} cm")

    # 6. 画图 (重点看对齐)
    plt.figure(figsize=(15, 10))
    
    # X 轴对齐 (通常这里的特征最明显)
    plt.subplot(3, 1, 1)
    plt.plot(y_ts, x_final_gt[:, 0], 'k-', label='GT X', linewidth=2)
    plt.plot(y_ts, medfilt(y_final[:, 0], 5), 'r-', label='YOLO X', alpha=0.8)
    plt.title(f"X-Axis Alignment (Shift: {TIME_SHIFT_SECONDS}s) - Mean Err: {mean_error:.1f} cm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Z 轴对齐 (高度)
    plt.subplot(3, 1, 2)
    plt.plot(y_ts, x_final_gt[:, 2], 'k-', label='GT Z (Height)')
    plt.plot(y_ts, medfilt(y_final[:, 2], 5), 'r-', label='YOLO Z', alpha=0.8)
    plt.title("Z-Axis (Height) Alignment")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 俯视轨迹
    plt.subplot(3, 1, 3)
    plt.plot(x_final_gt[:, 0], x_final_gt[:, 1], 'k-', label='GT Path')
    plt.plot(medfilt(y_final[:, 0], 15), medfilt(y_final[:, 1], 15), 'r-', label='YOLO Path')
    plt.axis('equal')
    plt.title("Trajectory Top-Down")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()