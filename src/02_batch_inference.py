import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
from tqdm import tqdm

# Import utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader

# ================= Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_params.npz")
# ===============================================

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
    frame_l, _, _, _ = loader_test.get_next_pair()
    if frame_l is None:
        print("[Error] Failed to read video. Please check the paths.")
        return
    h, w = frame_l.shape[:2]
    loader_test.release()
    
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0)

    # 3. Load YOLO Model
    print("[Info] Loading YOLO model...")
    model = YOLO('yolov8l-pose.pt')
    
    # Re-initialize the data loader for the full process
    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"), os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"), os.path.join(DATA_DIR, "1_video_right.txt")
    )

    all_timestamps = []
    all_keypoints_3d = [] 
    
    print("[Info] Processing video stream...")
    
    # Initialize progress bar (dynamic length to accommodate skipped frames automatically)
    pbar = tqdm(total=None, desc="Processing Frames", unit="frame")
    
    while True:
        frame_l, frame_r, fid, ts = loader.get_next_pair()
        if frame_l is None: break
        
        # YOLO Inference
        res_l = model(frame_l, verbose=False, conf=0.5)[0]
        res_r = model(frame_r, verbose=False, conf=0.5)[0]
        
        kpts_3d = np.full((17, 3), np.nan) # Fill with NaN by default

        if len(res_l.keypoints) > 0 and len(res_r.keypoints) > 0:
            # Reshape to (-1, 1, 2) to meet cv2.undistortPoints requirements
            pts_l = res_l.keypoints.xy[0].cpu().numpy().reshape(-1, 1, 2)
            pts_r = res_r.keypoints.xy[0].cpu().numpy().reshape(-1, 1, 2)
            
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

        all_timestamps.append(ts)
        all_keypoints_3d.append(kpts_3d)
        pbar.update(1)

    loader.release()
    pbar.close()

    # Save results
    save_file = os.path.join(OUTPUT_DIR, "yolo_3d_raw.npz")
    np.savez(save_file, 
             timestamps=np.array(all_timestamps),
             keypoints=np.array(all_keypoints_3d))
             
    print(f"\n[Info] Data successfully saved to {save_file}")
    print(f"[Info] Total frames processed: {len(all_timestamps)}")

if __name__ == "__main__":
    main()