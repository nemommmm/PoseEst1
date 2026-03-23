import cv2
import numpy as np
import os
import sys
import time
from ultralytics import YOLO
from tqdm import tqdm

# Import utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader
from pose_postprocess import estimate_bone_priors, postprocess_sequence

# ================= Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")
MODEL_NAME = os.environ.get("POSE_MODEL_NAME", "yolov8m-pose.pt")
MODEL_PATH = os.path.join(SRC_DIR, MODEL_NAME)
MODEL_SLUG = os.path.splitext(MODEL_NAME)[0].replace("-", "_")

MIN_KEYPOINT_CONF = float(os.environ.get("POSE_MIN_KEYPOINT_CONF", "0.35"))
# Empirically, this dataset's rectified reprojection error distribution is much
# broader than the original 12 px guess. Keep the default loose enough to
# preserve coverage, but make it configurable for stricter reruns.
REPROJECTION_ERROR_THRESHOLD = float(
    os.environ.get("POSE_REPROJECTION_ERROR_THRESHOLD", "80.0")
)  # pixels, averaged across left/right rectified views
ENABLE_BONE_CONSTRAINT = os.environ.get("POSE_ENABLE_BONE_CONSTRAINT", "1") != "0"
# Quality-aware blending remains opt-in. On the current benchmark it increased
# semantic joint-angle MAE relative to the simpler rigid-chain + One Euro path.
ENABLE_QUALITY_AWARE_BLEND = os.environ.get("POSE_ENABLE_QUALITY_AWARE_BLEND", "0") == "1"
# Pure rigid constraints suppress bone-length explosions, but work best when
# paired with light temporal smoothing rather than used alone.
ENABLE_ONE_EURO = os.environ.get("POSE_ENABLE_ONE_EURO", "1") == "1"
# Ground clamp is intentionally disabled by default because the raw stereo frame
# coordinate system is camera-centric rather than world-aligned.
FLOOR_AXIS = None
FLOOR_VALUE = None
# ===============================================


def resolve_postprocess_variant():
    if ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO and not ENABLE_QUALITY_AWARE_BLEND:
        return "rigid_chain_plus_one_euro"
    if ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO and ENABLE_QUALITY_AWARE_BLEND:
        return "rigid_chain_plus_one_euro_quality_aware"
    if ENABLE_BONE_CONSTRAINT and not ENABLE_ONE_EURO:
        return "rigid_chain_only"
    if not ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO:
        return "one_euro_only"
    return "raw_only"

def compute_rectified_reprojection_error(P, pts_3d, pts_2d_rect):
    reproj_error = np.full(len(pts_3d), np.nan, dtype=np.float64)
    valid_idx = np.where(np.isfinite(pts_3d).all(axis=1))[0]
    if len(valid_idx) == 0:
        return reproj_error

    pts_h = np.hstack([pts_3d[valid_idx], np.ones((len(valid_idx), 1), dtype=np.float64)])
    proj = (P @ pts_h.T).T
    valid_depth = np.abs(proj[:, 2]) > 1e-8
    if not np.any(valid_depth):
        return reproj_error

    proj = proj[valid_depth]
    valid_idx = valid_idx[valid_depth]
    proj_xy = proj[:, :2] / proj[:, 2:3]
    gt_xy = pts_2d_rect[valid_idx, 0, :]
    reproj_error[valid_idx] = np.linalg.norm(proj_xy - gt_xy, axis=1)
    return reproj_error

def estimate_synchronized_pair_count(left_data, right_data):
    ptr_l = 0
    ptr_r = 0
    matches = 0
    while ptr_l < len(left_data) and ptr_r < len(right_data):
        id_l = left_data[ptr_l]["id"]
        id_r = right_data[ptr_r]["id"]
        if id_l == id_r:
            matches += 1
            ptr_l += 1
            ptr_r += 1
        elif id_l < id_r:
            ptr_l += 1
        else:
            ptr_r += 1
    return matches

def main():
    print("[Info] Starting batch 3D inference and saving raw data...")
    
    # 1. Load Calibration Parameters
    if not os.path.exists(PARAM_PATH):
        print(f"[Error] Calibration parameters not found: {PARAM_PATH}")
        return
        
    data = np.load(PARAM_PATH)
    mtx_l, dist_l = data['mtx_l'], data['dist_l']
    mtx_r, dist_r = data['mtx_r'], data['dist_r']
    R, T = data['R'], data['T']

    # 2. Pre-compute Rectification Matrices
    print("[Info] Computing rectification matrices...")
    # Attempt to read one frame to get image dimensions
    loader_test = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"), os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"), os.path.join(DATA_DIR, "1_video_right.txt")
    )
    estimated_total_pairs = estimate_synchronized_pair_count(loader_test.left_data, loader_test.right_data)
    print(f"[Info] Estimated synchronized frame pairs: {estimated_total_pairs}")
    frame_l, _, _, _ = loader_test.get_next_pair()
    if frame_l is None:
        print("[Error] Failed to read video. Please check the paths.")
        return
    h, w = frame_l.shape[:2]
    loader_test.release()
    
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0)

    # 3. Load YOLO Model
    print("[Info] Loading YOLO model...")
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] YOLO model not found: {MODEL_PATH}")
        return
    print(f"[Info] Using pose model: {os.path.basename(MODEL_PATH)}")
    model = YOLO(MODEL_PATH)
    
    # Re-initialize the data loader for the full process
    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"), os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"), os.path.join(DATA_DIR, "1_video_right.txt")
    )

    all_timestamps = []
    all_keypoints_3d = []
    all_reprojection_errors = []
    all_keypoints_left_2d = []
    all_keypoints_right_2d = []
    all_conf_left = []
    all_conf_right = []
    
    print("[Info] Processing video stream...")
    
    # Initialize progress bar with ETA and throughput.
    pbar = tqdm(
        total=estimated_total_pairs if estimated_total_pairs > 0 else None,
        desc="Processing Frames",
        unit="frame",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    start_time = time.perf_counter()
    
    while True:
        frame_l, frame_r, fid, ts = loader.get_next_pair()
        if frame_l is None: break
        
        # YOLO Inference
        res_l = model(frame_l, verbose=False, conf=0.5)[0]
        res_r = model(frame_r, verbose=False, conf=0.5)[0]
        
        kpts_3d = np.full((17, 3), np.nan) # Fill with NaN by default
        reproj_error = np.full(17, np.nan, dtype=np.float64)
        pts_l_xy = np.full((17, 2), np.nan, dtype=np.float64)
        pts_r_xy = np.full((17, 2), np.nan, dtype=np.float64)
        conf_l_out = np.full(17, np.nan, dtype=np.float64)
        conf_r_out = np.full(17, np.nan, dtype=np.float64)

        if len(res_l.keypoints) > 0 and len(res_r.keypoints) > 0:
            # Reshape to (-1, 1, 2) to meet cv2.undistortPoints requirements
            pts_l = res_l.keypoints.xy[0].cpu().numpy().reshape(-1, 1, 2)
            pts_r = res_r.keypoints.xy[0].cpu().numpy().reshape(-1, 1, 2)
            conf_l = res_l.keypoints.conf[0].cpu().numpy()
            conf_r = res_r.keypoints.conf[0].cpu().numpy()
            pts_l_xy = pts_l[:, 0, :].astype(np.float64)
            pts_r_xy = pts_r[:, 0, :].astype(np.float64)
            conf_l_out = conf_l.astype(np.float64)
            conf_r_out = conf_r.astype(np.float64)
            valid_mask = (conf_l >= MIN_KEYPOINT_CONF) & (conf_r >= MIN_KEYPOINT_CONF)
            
            # Coordinate Rectification (Undistortion)
            pts_l_rect = cv2.undistortPoints(pts_l, mtx_l, dist_l, R=R1, P=P1)
            pts_r_rect = cv2.undistortPoints(pts_r, mtx_r, dist_r, R=R2, P=P2)
            
            # Triangulation
            pts4d = cv2.triangulatePoints(P1, P2, pts_l_rect, pts_r_rect)
            
            # Normalization from homogeneous coordinates (X/W, Y/W, Z/W)
            w_vec = pts4d[3, :]
            valid_w = w_vec != 0 # Prevent division by zero
            pts3d_raw = np.full((3, 17), np.nan)
            pts3d_raw[:, valid_w] = pts4d[:3, valid_w] / w_vec[valid_w]
            
            kpts_3d = pts3d_raw.T # Transpose back to (17, 3)
            kpts_3d[~valid_mask] = np.nan
            reproj_l = compute_rectified_reprojection_error(P1, kpts_3d, pts_l_rect)
            reproj_r = compute_rectified_reprojection_error(P2, kpts_3d, pts_r_rect)
            reproj_error = np.nanmean(np.vstack([reproj_l, reproj_r]), axis=0)
            valid_reproj = reproj_error <= REPROJECTION_ERROR_THRESHOLD
            valid_reproj &= np.isfinite(reproj_error)
            kpts_3d[~valid_reproj] = np.nan

        all_timestamps.append(ts)
        all_keypoints_3d.append(kpts_3d)
        all_reprojection_errors.append(reproj_error)
        all_keypoints_left_2d.append(pts_l_xy)
        all_keypoints_right_2d.append(pts_r_xy)
        all_conf_left.append(conf_l_out)
        all_conf_right.append(conf_r_out)
        pbar.update(1)
        processed_frames = len(all_timestamps)
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        fps = processed_frames / elapsed
        valid_joints = int(np.isfinite(kpts_3d).all(axis=1).sum())
        eta_seconds = None
        if estimated_total_pairs > 0 and fps > 1e-6:
            eta_seconds = max(estimated_total_pairs - processed_frames, 0) / fps
        eta_text = "--:--"
        if eta_seconds is not None:
            eta_text = time.strftime("%M:%S", time.gmtime(int(eta_seconds)))
        pbar.set_postfix(valid_joints=valid_joints, fps=f"{fps:.2f}", eta=eta_text)

    loader.release()
    pbar.close()

    all_timestamps = np.array(all_timestamps, dtype=np.float64)
    all_keypoints_3d = np.array(all_keypoints_3d, dtype=np.float64)
    all_reprojection_errors = np.array(all_reprojection_errors, dtype=np.float64)
    all_keypoints_left_2d = np.array(all_keypoints_left_2d, dtype=np.float64)
    all_keypoints_right_2d = np.array(all_keypoints_right_2d, dtype=np.float64)
    all_conf_left = np.array(all_conf_left, dtype=np.float64)
    all_conf_right = np.array(all_conf_right, dtype=np.float64)
    pair_confidence = np.nanmean(np.stack([all_conf_left, all_conf_right], axis=0), axis=0)

    finite_reproj = all_reprojection_errors[np.isfinite(all_reprojection_errors)]
    if finite_reproj.size > 0:
        keep_ratio = np.mean(finite_reproj <= REPROJECTION_ERROR_THRESHOLD)
        print("[Info] Reprojection error summary (px):")
        print(f"       p50={np.percentile(finite_reproj, 50):.2f}, p90={np.percentile(finite_reproj, 90):.2f}, p95={np.percentile(finite_reproj, 95):.2f}")
        print(f"       Keep ratio @ {REPROJECTION_ERROR_THRESHOLD:.1f}px: {keep_ratio:.3f}")

    # 4. Sequence-level postprocessing
    optimized_keypoints = all_keypoints_3d.copy()
    priors = estimate_bone_priors(all_keypoints_3d, all_timestamps)
    print("[Info] Estimated bone priors (cm):")
    for bone_name, value in priors.items():
        print(f"       {bone_name}: {value:.2f}")

    if ENABLE_BONE_CONSTRAINT or ENABLE_ONE_EURO:
        optimized_keypoints = postprocess_sequence(
            all_keypoints_3d,
            all_timestamps,
            priors=priors,
            reprojection_errors=all_reprojection_errors,
            pair_confidence=pair_confidence,
            floor_axis=FLOOR_AXIS,
            floor_value=FLOOR_VALUE,
            enable_bone_constraint=ENABLE_BONE_CONSTRAINT,
            enable_quality_blend=ENABLE_QUALITY_AWARE_BLEND,
            enable_one_euro=ENABLE_ONE_EURO,
        )

    # Save results
    postprocess_variant = resolve_postprocess_variant()
    save_file = os.path.join(OUTPUT_DIR, "yolo_3d_raw.npz")
    model_save_file = os.path.join(OUTPUT_DIR, f"yolo_3d_raw_{MODEL_SLUG}.npz")
    np.savez(save_file, 
             timestamps=all_timestamps,
             keypoints=all_keypoints_3d,
             reprojection_error=all_reprojection_errors,
             keypoints_left_2d=all_keypoints_left_2d,
             keypoints_right_2d=all_keypoints_right_2d,
             conf_left=all_conf_left,
             conf_right=all_conf_right,
             model_name=np.array(MODEL_NAME),
             postprocess_variant=np.array("raw_only"),
             reprojection_threshold_px=np.array(REPROJECTION_ERROR_THRESHOLD, dtype=np.float64))
    np.savez(model_save_file, 
             timestamps=all_timestamps,
             keypoints=all_keypoints_3d,
             reprojection_error=all_reprojection_errors,
             keypoints_left_2d=all_keypoints_left_2d,
             keypoints_right_2d=all_keypoints_right_2d,
             conf_left=all_conf_left,
             conf_right=all_conf_right,
             model_name=np.array(MODEL_NAME),
             postprocess_variant=np.array("raw_only"),
             reprojection_threshold_px=np.array(REPROJECTION_ERROR_THRESHOLD, dtype=np.float64))

    optimized_save_file = os.path.join(OUTPUT_DIR, "yolo_3d_optimized.npz")
    optimized_model_save_file = os.path.join(OUTPUT_DIR, f"yolo_3d_optimized_{MODEL_SLUG}.npz")
    np.savez(
        optimized_save_file,
        timestamps=all_timestamps,
        keypoints=optimized_keypoints,
        reprojection_error=all_reprojection_errors,
        keypoints_left_2d=all_keypoints_left_2d,
        keypoints_right_2d=all_keypoints_right_2d,
        conf_left=all_conf_left,
        conf_right=all_conf_right,
        prior_names=np.array(list(priors.keys())),
        prior_values=np.array(list(priors.values()), dtype=np.float64),
        model_name=np.array(MODEL_NAME),
        postprocess_variant=np.array(postprocess_variant),
        reprojection_threshold_px=np.array(REPROJECTION_ERROR_THRESHOLD, dtype=np.float64),
    )
    np.savez(
        optimized_model_save_file,
        timestamps=all_timestamps,
        keypoints=optimized_keypoints,
        reprojection_error=all_reprojection_errors,
        keypoints_left_2d=all_keypoints_left_2d,
        keypoints_right_2d=all_keypoints_right_2d,
        conf_left=all_conf_left,
        conf_right=all_conf_right,
        prior_names=np.array(list(priors.keys())),
        prior_values=np.array(list(priors.values()), dtype=np.float64),
        model_name=np.array(MODEL_NAME),
        postprocess_variant=np.array(postprocess_variant),
        reprojection_threshold_px=np.array(REPROJECTION_ERROR_THRESHOLD, dtype=np.float64),
    )
             
    print(f"\n[Info] Data successfully saved to {save_file}")
    print(f"[Info] Model-specific raw data saved to {model_save_file}")
    print(f"[Info] Optimized data successfully saved to {optimized_save_file}")
    print(f"[Info] Model-specific optimized data saved to {optimized_model_save_file}")
    print(f"[Info] Total frames processed: {len(all_timestamps)}")

if __name__ == "__main__":
    main()
