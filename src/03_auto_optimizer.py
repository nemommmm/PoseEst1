import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.signal import medfilt
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser
from pose_angle_utils import (
    DEFAULT_ANGLE_SMOOTH_RADIUS,
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
    median_filter_angle_sequence,
)

# ================= Optimization Configuration =================
# Grid Search space around the estimated time shift
SEARCH_START = 12.5
SEARCH_END = 22.5
STEP_SIZE = 0.05  # Step size in seconds. Smaller = slower but more precise.

# Strategy Parameters
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5] # Reference limb lengths (cm)
TOP_K = 150 # Number of elite frames to compute the rigid body transformation
ANGLE_REFINE_RADIUS = 0.5
ANGLE_MIN_VALID_SAMPLES = 120
SHOW_PLOTS = os.environ.get("POSE_SHOW_PLOTS", "0") == "1"
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def calculate_limb_error(kpts, gt_lengths):
    """
    Compute the anatomical error based on Euclidean distance of limbs.
    """
    lengths = np.vstack([
        np.linalg.norm(kpts[:, 11] - kpts[:, 13], axis=1), 
        np.linalg.norm(kpts[:, 12] - kpts[:, 14], axis=1),
        np.linalg.norm(kpts[:, 13] - kpts[:, 15], axis=1), 
        np.linalg.norm(kpts[:, 14] - kpts[:, 16], axis=1) 
    ]).T
    return np.sum(np.abs(lengths - gt_lengths), axis=1)

def kabsch_transform(P, Q):
    """
    Compute optimal rotation and translation matrices to align two point clouds
    using the Kabsch algorithm (Procrustes analysis via SVD).
    """
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
    
    # Correct reflection if necessary
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
        
    t = centroid_Q - rot @ centroid_P
    return rot, t

def evaluate_shift(shift, y_ts, y_center, f_x, elite_indices, p_elite):
    """
    Evaluate the global alignment error given a specific time shift.
    """
    # 1. Align time using the shift and find corresponding ground truth for elite frames
    t_elite_shifted = y_ts[elite_indices] - shift
    q_elite = f_x(t_elite_shifted) 
    
    # 2. Compute optimal rotation matrix for this specific shift
    R_mat, t_vec = kabsch_transform(p_elite, q_elite)
    
    # 3. Apply transformation to the entire trajectory
    y_final = (R_mat @ y_center.T).T + t_vec
    
    # 4. Compute Global Error
    t_full_shifted = y_ts - shift
    x_final_gt = f_x(t_full_shifted)
    
    diff = y_final - x_final_gt
    dist = np.linalg.norm(diff, axis=1)
    
    # Only calculate mean error on valid finite points
    valid_dist = dist[np.isfinite(dist)]
    if len(valid_dist) < 100: return np.inf # Penalize invalid alignments
    
    return np.mean(valid_dist), R_mat, t_vec


def evaluate_angle_shift(shift, y_ts, y_angle_values, angle_names, gt_angle_interp):
    """
    Evaluate a local temporal shift by the mean correlation of semantic joint angles.
    """
    correlations = []
    for angle_idx, angle_name in enumerate(angle_names):
        if angle_name not in gt_angle_interp:
            continue
        est_vals = y_angle_values[:, angle_idx]
        gt_vals = gt_angle_interp[angle_name](y_ts - shift)
        valid = np.isfinite(est_vals) & np.isfinite(gt_vals)
        if np.sum(valid) < ANGLE_MIN_VALID_SAMPLES:
            continue
        est_valid = est_vals[valid]
        gt_valid = gt_vals[valid]
        if np.std(est_valid) < 1e-6 or np.std(gt_valid) < 1e-6:
            continue
        corr = np.corrcoef(est_valid, gt_valid)[0, 1]
        if np.isfinite(corr):
            correlations.append(float(corr))
    if not correlations:
        return -np.inf
    return float(np.mean(correlations))

def main():
    print(f"[Info] Starting automated temporal-spatial optimization (Search Range: {SEARCH_START}s ~ {SEARCH_END}s)...")
    yolo_data_path = resolve_yolo_data_path()
    print(f"[Info] Loading pose data from: {os.path.basename(yolo_data_path)}")

    # --- 1. Data Preparation ---
    yolo_data = np.load(yolo_data_path)
    y_kpts = yolo_data['keypoints']
    y_ts = yolo_data['timestamps']
    y_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    
    # Data Cleaning (Filtering invalid depth)
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
    gt_angle_interp = build_gt_angle_interpolators(mvnx, x_ts, xidx)

    # Interpolation function for Ground Truth (Xsens)
    f_x = interp1d(x_ts, x_pelvis, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
    angle_names, angle_values = compute_semantic_angle_sequence(y_kpts)
    angle_values = median_filter_angle_sequence(angle_values, radius=DEFAULT_ANGLE_SMOOTH_RADIUS)

    # Extract elite frames (frames with minimal anatomical deformation)
    errors = calculate_limb_error(y_kpts, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    sorted_idx = np.argsort(errors[valid_err_idx])
    elite_indices = valid_err_idx[sorted_idx[:TOP_K]]
    p_elite = y_center[elite_indices]

    # --- 2. Grid Search ---
    shifts = np.arange(SEARCH_START, SEARCH_END, STEP_SIZE)
    results = []
    
    print(f"[Info] Evaluating {len(shifts)} discrete time shifts...")
    
    for shift in shifts:
        err, _, _ = evaluate_shift(shift, y_ts, y_center, f_x, elite_indices, p_elite)
        results.append(err)

    # --- 3. Result Analysis ---
    results = np.array(results)
    best_idx = np.argmin(results)
    best_shift = shifts[best_idx]
    min_error = results[best_idx]

    refine_start = max(SEARCH_START, best_shift - ANGLE_REFINE_RADIUS)
    refine_end = min(SEARCH_END, best_shift + ANGLE_REFINE_RADIUS + 0.5 * STEP_SIZE)
    refine_shifts = np.arange(refine_start, refine_end, STEP_SIZE)
    angle_scores = np.array([
        evaluate_angle_shift(shift, y_ts, angle_values, angle_names, gt_angle_interp)
        for shift in refine_shifts
    ], dtype=np.float64)
    valid_angle_idx = np.where(np.isfinite(angle_scores))[0]
    if len(valid_angle_idx) > 0:
        refined_idx = valid_angle_idx[np.argmax(angle_scores[valid_angle_idx])]
        refined_shift = float(refine_shifts[refined_idx])
        best_angle_corr = float(angle_scores[refined_idx])
    else:
        refined_shift = float(best_shift)
        best_angle_corr = np.nan

    final_shift = refined_shift if np.isfinite(refined_shift) else float(best_shift)

    print("\n" + "="*50)
    print(f"[Result] Position Best Shift: {best_shift:.2f} seconds")
    print(f"[Result] Minimum Mean Error:  {min_error:.2f} cm")
    print(f"[Result] Angle-Refined Shift: {final_shift:.2f} seconds")
    if np.isfinite(best_angle_corr):
        print(f"[Result] Mean Angle Corr.:   {best_angle_corr:.3f}")
    print("="*50)

    alignment_summary_path = os.path.join(RESULTS_DIR, "alignment_summary.json")
    with open(alignment_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_offset_seconds": float(final_shift),
                "minimum_mean_error_cm": float(min_error),
                "position_best_offset_seconds": float(best_shift),
                "position_minimum_mean_error_cm": float(min_error),
                "angle_refined_offset_seconds": float(final_shift),
                "angle_refined_mean_correlation": None if not np.isfinite(best_angle_corr) else float(best_angle_corr),
            },
            f,
            indent=2,
        )
    print(f"[Info] Alignment summary saved to {alignment_summary_path}")

    # --- 4. Visualization with Optimal Parameters ---
    _, R_best, t_best = evaluate_shift(final_shift, y_ts, y_center, f_x, elite_indices, p_elite)
    
    # Generate final aligned data
    y_final = (R_best @ y_center.T).T + t_best
    t_full_shifted = y_ts - final_shift
    x_final_gt = f_x(t_full_shifted)

    # Plotting
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(shifts, results, 'b-', linewidth=2)
    plt.plot(best_shift, min_error, 'r*', markersize=15, label=f'Position optimum ({best_shift:.2f}s)')
    if abs(final_shift - best_shift) > 0.5 * STEP_SIZE:
        plt.axvline(final_shift, color='orange', linestyle='--', linewidth=1.5, label=f'Angle-refined ({final_shift:.2f}s)')
    plt.title("Optimization Landscape (Error vs Time Shift)")
    plt.xlabel("Time Shift (seconds)")
    plt.ylabel("Mean Position Error (cm)")
    plt.legend()
    plt.grid(True)

    # Best Alignment (X-Axis)
    plt.subplot(2, 2, 2)
    plt.plot(y_ts, x_final_gt[:, 0], 'k-', label='Ground Truth X')
    plt.plot(y_ts, medfilt(y_final[:, 0], 5), 'r-', label='Estimated X')
    plt.title(f"Best Alignment (X-Axis) @ {final_shift:.2f}s")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Best Alignment (Z-Axis / Height)
    plt.subplot(2, 1, 2)
    plt.plot(y_ts, x_final_gt[:, 2], 'k-', label='Ground Truth Height (Z)')
    plt.plot(y_ts, medfilt(y_final[:, 2], 5), 'r-', label='Estimated Height (Z)')
    plt.title("Best Alignment (Z-Axis / Height)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_optimizer_result.png")
    plt.savefig(plot_path)
    print(f"\n[Info] Optimization complete. Results saved to {plot_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()
