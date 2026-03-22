import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt
from scipy.interpolate import interp1d

# Import utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser

# ================= Manual Tuning Configuration =================
# Core Parameter: Temporal Shift (in seconds)
# A positive value shifts the estimated timeline forward relative to the Ground Truth.
# This serves as the initial hypothesis before automated grid search.
TIME_SHIFT_SECONDS = 17.5  

# Strategy Parameters
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5] # Reference anatomical limb lengths (cm)
TOP_K = 150 # Number of elite frames used to compute the optimal rigid body transformation
# ===============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")

def resolve_yolo_data_path():
    candidates = [
        os.path.join(PROJECT_ROOT, "results", "yolo_3d_optimized.npz"),
        os.path.join(PROJECT_ROOT, "results", "yolo_3d_raw.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]

def calculate_limb_error(kpts, gt_lengths):
    """
    Compute the anatomical deformation error based on Euclidean distance of limbs.
    """
    lengths = np.vstack([
        np.linalg.norm(kpts[:, 11] - kpts[:, 13], axis=1), 
        np.linalg.norm(kpts[:, 12] - kpts[:, 14], axis=1),
        np.linalg.norm(kpts[:, 13] - kpts[:, 15], axis=1), 
        np.linalg.norm(kpts[:, 14] - kpts[:, 16], axis=1) 
    ]).T
    error = np.sum(np.abs(lengths - gt_lengths), axis=1)
    return error

def kabsch_transform(P, Q):
    """
    Compute optimal rotation and translation matrices to align two point clouds
    using the Kabsch algorithm (Procrustes analysis via SVD).
    """
    # Filter out NaNs
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
    
    # Ensure a proper rotation (det(R) == 1) rather than a reflection
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
        
    t = centroid_Q - rot @ centroid_P
    return rot, t

def main():
    print(f"[Info] Starting manual temporal tuning mode (Time Shift: {TIME_SHIFT_SECONDS} s)...")
    yolo_data_path = resolve_yolo_data_path()
    print(f"[Info] Loading pose data from: {os.path.basename(yolo_data_path)}")

    # 1. Load Data
    yolo_data = np.load(yolo_data_path)
    y_kpts = yolo_data['keypoints']
    y_ts = yolo_data['timestamps']
    y_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    
    # Data Cleaning: Filter out frames with invalid depth or NaN values
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

    # 2. Select Elite Frames (for computing the optimal rotation matrix)
    errors = calculate_limb_error(y_kpts, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    sorted_idx = np.argsort(errors[valid_err_idx])
    elite_indices = valid_err_idx[sorted_idx[:TOP_K]]
    print(f"[Info] Extracted {len(elite_indices)} elite frames for rigid body transformation.")

    # 3. Apply Manual Temporal Shift
    # Objective: Interpolate Ground Truth to match the estimated trajectory's timestamps
    f_x = interp1d(x_ts, x_pelvis, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
    
    # Extract paired points for elite frames
    p_elite = y_center[elite_indices]
    t_elite_shifted = y_ts[elite_indices] - TIME_SHIFT_SECONDS
    q_elite = f_x(t_elite_shifted)

    # 4. Compute Rotation Matrix (Kabsch Algorithm)
    R_mat, t_vec = kabsch_transform(p_elite, q_elite)

    # 5. Global Application and Evaluation
    y_final = (R_mat @ y_center.T).T + t_vec
    
    # Evaluate over the entire trajectory
    t_full_shifted = y_ts - TIME_SHIFT_SECONDS
    x_final_gt = f_x(t_full_shifted)
    
    # Compute error (ignoring NaNs)
    diff = y_final - x_final_gt
    dist = np.linalg.norm(diff, axis=1)
    valid_dist = dist[np.isfinite(dist)]
    mean_error = np.mean(valid_dist)
    
    print(f"[Result] Mean error at configured offset ({TIME_SHIFT_SECONDS}s): {mean_error:.2f} cm")

    # 6. Visualization
    plt.figure(figsize=(15, 10))
    
    # X-Axis Alignment
    plt.subplot(3, 1, 1)
    plt.plot(y_ts, x_final_gt[:, 0], 'k-', label='Ground Truth X', linewidth=2)
    plt.plot(y_ts, medfilt(y_final[:, 0], 5), 'r-', label='Estimated X', alpha=0.8)
    plt.title(f"X-Axis Alignment (Shift: {TIME_SHIFT_SECONDS}s) - Mean Error: {mean_error:.1f} cm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Z-Axis Alignment (Height)
    plt.subplot(3, 1, 2)
    plt.plot(y_ts, x_final_gt[:, 2], 'k-', label='Ground Truth Height (Z)')
    plt.plot(y_ts, medfilt(y_final[:, 2], 5), 'r-', label='Estimated Height (Z)', alpha=0.8)
    plt.title("Z-Axis (Height) Alignment")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Top-Down Trajectory
    plt.subplot(3, 1, 3)
    plt.plot(x_final_gt[:, 0], x_final_gt[:, 1], 'k-', label='Ground Truth Path')
    plt.plot(medfilt(y_final[:, 0], 15), medfilt(y_final[:, 1], 15), 'r-', label='Estimated Path')
    plt.axis('equal')
    plt.title("Top-Down Trajectory (X-Y Plane)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
