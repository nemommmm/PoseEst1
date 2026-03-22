import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import json
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser

# ================= Optimized Parameters =================
# Best temporal offset obtained from the automated optimizer
BEST_OFFSET = 17.25   
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5] # Reference limb lengths (cm)
TOP_K = 150 # Number of elite frames for rigid body transformation

# ================= Snapshot Configuration =================
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
MAX_SNAPSHOTS = 5      # Maximum number of snapshots to save
MIN_TIME_GAP = 2.0     # Minimum time gap (seconds) between snapshots to ensure diversity

# ================= Skeleton Definitions =================
# Estimated Pose (COCO format) edges
YOLO_EDGES = [
    (0,1), (0,2), (1,3), (2,4),          # Face
    (5,6), (5,7), (7,9), (6,8), (8,10),  # Arms
    (5,11), (6,12),                      # Torso
    (11,12), (11,13), (13,15), (12,14), (14,16) # Legs
]

# Ground Truth (Xsens) segments to extract
XSENS_SEGMENTS_TO_LOAD = [
    'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
    'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
    'RightUpperLeg', 'RightLowerLeg', 'RightFoot',
    'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot'
]

# Ground Truth Kinematic Tree (Parent -> Child connections)
XSENS_LINKS = [
    ('Pelvis', 'L5'), ('L5', 'L3'), ('L3', 'T12'), ('T12', 'T8'), ('T8', 'Neck'), ('Neck', 'Head'), # Spine
    ('T8', 'RightShoulder'), ('RightShoulder', 'RightUpperArm'), ('RightUpperArm', 'RightForeArm'), ('RightForeArm', 'RightHand'), # Right Arm
    ('T8', 'LeftShoulder'), ('LeftShoulder', 'LeftUpperArm'), ('LeftUpperArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'), # Left Arm
    ('Pelvis', 'RightUpperLeg'), ('RightUpperLeg', 'RightLowerLeg'), ('RightLowerLeg', 'RightFoot'), # Right Leg
    ('Pelvis', 'LeftUpperLeg'), ('LeftUpperLeg', 'LeftLowerLeg'), ('LeftLowerLeg', 'LeftFoot') # Left Leg
]
# ========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MVNX_PATH = os.path.join(PROJECT_ROOT,"..", "Xsens_ground_truth", "Aitor-001.mvnx")
ALIGNMENT_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "alignment_summary.json")

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
    """
    Compute the anatomical deformation error based on Euclidean distance of limbs.
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
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
    t = centroid_Q - rot @ centroid_P
    return rot, t

def main():
    print("[Info] Starting full skeleton comparative visualization and snapshot generation...")
    yolo_data_path = resolve_yolo_data_path()
    best_offset = resolve_best_offset()
    print(f"[Info] Loading pose data from: {os.path.basename(yolo_data_path)}")
    print(f"[Info] Using temporal offset: {best_offset:.2f} s")

    # 1. Load Estimated Pose Data
    yolo_data = np.load(yolo_data_path)
    y_kpts = yolo_data['keypoints']
    y_ts = yolo_data['timestamps']
    
    # Filter valid frames
    y_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    valid_mask = (y_center[:, 2] > 10) & (y_center[:, 2] < 1000) & np.isfinite(y_center).all(axis=1)
    y_kpts = y_kpts[valid_mask]
    y_ts = y_ts[valid_mask]
    y_ts, uidx = np.unique(y_ts, return_index=True)
    y_kpts = y_kpts[uidx]
    y_ts -= y_ts[0]

    # 2. Load Ground Truth (Xsens) Full Body Data
    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    
    xsens_data = {}
    xsens_ts = mvnx.timestamps
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]

    print("[Info] Extracting relevant kinematic segments from Xsens data...")
    for seg_name in XSENS_SEGMENTS_TO_LOAD:
        raw_data = mvnx.get_segment_data(seg_name)
        if raw_data is not None:
            xsens_data[seg_name] = raw_data[xidx]
        else:
            print(f"[Warning] Segment {seg_name} not found in MVNX file.")

    # Interpolation dictionary for Ground Truth
    xsens_interp = {}
    for name, data in xsens_data.items():
        xsens_interp[name] = interp1d(xsens_ts, data, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)

    # 3. Compute Rigid Body Transformation Matrix
    f_x_pelvis = xsens_interp['Pelvis']
    
    y_pelvis_center = (y_kpts[:, 11] + y_kpts[:, 12]) / 2.0
    errors = calculate_limb_error(y_kpts, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    sorted_idx = np.argsort(errors[valid_err_idx])
    elite_indices = valid_err_idx[sorted_idx[:TOP_K]]
    
    p_elite = y_pelvis_center[elite_indices]
    t_elite_shifted = y_ts[elite_indices] - best_offset
    q_elite = f_x_pelvis(t_elite_shifted)
    
    print("[Info] Computing optimal rigid body transformation matrix via Kabsch algorithm...")
    R_mat, t_vec = kabsch_transform(p_elite, q_elite)

    # 4. Apply Transformation to the Full Estimated Skeleton
    N, J, _ = y_kpts.shape
    y_kpts_flat = y_kpts.reshape(-1, 3)
    y_kpts_transformed = (R_mat @ y_kpts_flat.T).T + t_vec
    y_kpts_final = y_kpts_transformed.reshape(N, J, 3)

    # 5. Animation and Automated Snapshot Logic
    print("[Info] Initiating 3D playback (Snapshots will be saved automatically)...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    start_time = max(0, best_offset)
    end_time = min(y_ts[-1], xsens_ts[-1] + best_offset)
    step = 5
    
    snapshots_taken = 0
    last_snapshot_time = -999

    for i in range(0, len(y_ts), step):
        curr_time = y_ts[i]
        if curr_time < start_time or curr_time > end_time: continue
        
        target_x_time = curr_time - best_offset
        y_pose = y_kpts_final[i]
        
        # Retrieve all Xsens joint positions for the current interpolated timestamp
        x_pose = {}
        for name, func in xsens_interp.items():
            pos = func(target_x_time)
            if not np.isnan(pos).any():
                x_pose[name] = pos
        
        if 'Pelvis' not in x_pose: continue

        # Compute instantaneous alignment error (Pelvis Distance)
        curr_error = np.linalg.norm(y_pose[11:13].mean(0) - x_pose['Pelvis'])

        ax.cla()
        
        # --- Plot Ground Truth (Black/Grey) ---
        for name, pos in x_pose.items():
            ax.scatter(pos[0], pos[1], pos[2], c='k', s=10, alpha=0.5)
        for p1_name, p2_name in XSENS_LINKS:
            if p1_name in x_pose and p2_name in x_pose:
                p1 = x_pose[p1_name]
                p2 = x_pose[p2_name]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='k', linewidth=1.5, alpha=0.6)
        
        # Add legend label once to avoid clutter
        ax.plot([], [], [], c='k', label='Ground Truth (Xsens)')

        # --- Plot Estimated Pose (Red) ---
        ax.scatter(y_pose[:, 0], y_pose[:, 1], y_pose[:, 2], c='r', s=25, label='Estimated Pose (Stereo)')
        for edge in YOLO_EDGES:
            pt1, pt2 = y_pose[edge[0]], y_pose[edge[1]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='r', linewidth=2)

        # Viewport following logic (Camera tracks the pelvis)
        center = x_pose['Pelvis']
        ax.set_xlim(center[0]-100, center[0]+100)
        ax.set_ylim(center[1]-100, center[1]+100)
        ax.set_zlim(center[2]-100, center[2]+100)
        
        title_str = f"Time: {curr_time:.2f}s | Absolute Error: {curr_error:.1f} cm"
        ax.set_title(title_str)
        ax.legend(loc='upper right')

        # --- Automated Snapshot Logic ---
        # Trigger condition: Error < 20cm (High accuracy), Time Gap > 2s (Avoid redundancy), Cap not reached
        if curr_error < 20.0 and (curr_time - last_snapshot_time) > MIN_TIME_GAP and snapshots_taken < MAX_SNAPSHOTS:
            filename = os.path.join(SNAPSHOT_DIR, f"snapshot_t{curr_time:.1f}_err{curr_error:.1f}.png")
            plt.savefig(filename, dpi=150)
            print(f"[Result] High-quality alignment captured and saved: {os.path.basename(filename)}")
            last_snapshot_time = curr_time
            snapshots_taken += 1

        plt.pause(0.001)
        if not plt.fignum_exists(fig.number): break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
