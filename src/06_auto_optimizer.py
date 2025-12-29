import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser

# ================= 🤖 自动搜索配置 =================
# 搜索范围：在你手动发现的 17.5s 附近 ±5秒 搜索
SEARCH_START = 12.5
SEARCH_END = 22.5
STEP_SIZE = 0.05  # 步长 (秒)，越小越慢但越准

# 策略参数
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]
TOP_K = 150
# =================================================

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

def evaluate_shift(shift, y_ts, y_center, f_x, elite_indices, p_elite):
    """
    核心评估函数：给定一个 shift，计算全局误差
    """
    # 1. 使用 shift 对齐时间，找到黄金帧对应的真值
    t_elite_shifted = y_ts[elite_indices] - shift
    q_elite = f_x(t_elite_shifted) # 插值
    
    # 2. 计算该 shift 下的最佳旋转矩阵
    R_mat, t_vec = kabsch_transform(p_elite, q_elite)
    
    # 3. 应用变换到所有帧
    y_final = (R_mat @ y_center.T).T + t_vec
    
    # 4. 计算全局误差 (Global Error)
    t_full_shifted = y_ts - shift
    x_final_gt = f_x(t_full_shifted)
    
    diff = y_final - x_final_gt
    dist = np.linalg.norm(diff, axis=1)
    
    # 只计算有效的点
    valid_dist = dist[np.isfinite(dist)]
    if len(valid_dist) < 100: return np.inf # 惩罚无效对齐
    
    return np.mean(valid_dist), R_mat, t_vec

def main():
    print(f"🤖 启动自动寻优 (范围: {SEARCH_START}s ~ {SEARCH_END}s)...")

    # --- 1. 数据准备 (只做一次) ---
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

    # 插值器
    f_x = interp1d(x_ts, x_pelvis, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)

    # 黄金帧索引
    errors = calculate_limb_error(y_kpts, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    sorted_idx = np.argsort(errors[valid_err_idx])
    elite_indices = valid_err_idx[sorted_idx[:TOP_K]]
    p_elite = y_center[elite_indices]

    # --- 2. 网格搜索 (Grid Search) ---
    shifts = np.arange(SEARCH_START, SEARCH_END, STEP_SIZE)
    results = []
    
    print(f"   -> 正在评估 {len(shifts)} 个可能的时间点...")
    
    for shift in shifts:
        err, _, _ = evaluate_shift(shift, y_ts, y_center, f_x, elite_indices, p_elite)
        results.append(err)
        # 简单的进度打印
        # print(f"Shift: {shift:.2f}s | Error: {err:.2f} cm")

    # --- 3. 结果分析 ---
    results = np.array(results)
    best_idx = np.argmin(results)
    best_shift = shifts[best_idx]
    min_error = results[best_idx]

    print("\n" + "="*40)
    print(f"🏆 最佳时间偏移 (Best Offset): {best_shift:.2f} 秒")
    print(f"📉 最低平均误差 (Min Error):  {min_error:.2f} cm")
    print("="*40)

    # --- 4. 使用最佳参数绘图 ---
    _, R_best, t_best = evaluate_shift(best_shift, y_ts, y_center, f_x, elite_indices, p_elite)
    
    # 生成最终对齐数据
    y_final = (R_best @ y_center.T).T + t_best
    t_full_shifted = y_ts - best_shift
    x_final_gt = f_x(t_full_shifted)

    # 绘制优化曲线
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(shifts, results, 'b-', linewidth=2)
    plt.plot(best_shift, min_error, 'r*', markersize=15, label=f'Optimum ({best_shift:.2f}s)')
    plt.title("Optimization Landscape (Error vs Time Shift)")
    plt.xlabel("Time Shift (seconds)")
    plt.ylabel("Mean Position Error (cm)")
    plt.legend()
    plt.grid(True)

    # 绘制最佳对齐结果 (X轴)
    plt.subplot(2, 2, 2)
    plt.plot(y_ts, x_final_gt[:, 0], 'k-', label='GT X')
    plt.plot(y_ts, medfilt(y_final[:, 0], 5), 'r-', label='YOLO X')
    plt.title(f"Best Alignment (X-Axis) @ {best_shift:.2f}s")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制最佳对齐结果 (Z轴)
    plt.subplot(2, 1, 2)
    plt.plot(y_ts, x_final_gt[:, 2], 'k-', label='GT Height (Z)')
    plt.plot(y_ts, medfilt(y_final[:, 2], 5), 'r-', label='YOLO Height (Z)')
    plt.title("Best Alignment (Height)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("auto_optimizer_result.png")
    print("✅ 优化完成，结果已保存至 auto_optimizer_result.png")
    plt.show()

if __name__ == "__main__":
    main()