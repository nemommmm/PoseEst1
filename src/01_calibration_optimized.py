import cv2
import numpy as np
import os
import sys

# Ensure utils.py can be imported from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader

# ================= Configuration Area =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Calibration_video")

# Strategy: Include all available video pairs; rely on the algorithm to automatically clean bad data.
VIDEO_PAIRS = [
    # ("cap_0_left.avi", "cap_0_right.avi", "cap_0_left.txt", "cap_0_right.txt"),
    ("cap_1_left.avi", "cap_1_right.avi", "cap_1_left.txt", "cap_1_right.txt")
]

# Core Parameters
PATTERN_SIZE = (5, 9)       # Asymmetric Circle Grid
SQUARE_SIZE = 15.0          # Center distance 15cm
MAX_ERROR_THRESHOLD = 1.0   # [Core Optimization] Only keep frames with single-camera error < 1.0 px
# ======================================================

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """
    Calculate single-camera reprojection error to filter out bad frames.
    """
    total_error = 0
    errors_per_frame = []
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors_per_frame.append(error)
        
    return errors_per_frame

def calculate_epipolar_error(imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, F):
    """
    Calculate Epipolar Error.
    This metric measures whether a point in the left image precisely falls onto the 
    corresponding epipolar line in the right image.
    """
    total_error = 0
    total_points = 0

    # Iterate through each frame
    for i in range(len(imgpoints_l)):
        n_pts = len(imgpoints_l[i])
        
        # 1. Points must be undistorted first
        p_l = cv2.undistortPoints(imgpoints_l[i], mtx_l, dist_l, P=mtx_l)
        p_r = cv2.undistortPoints(imgpoints_r[i], mtx_r, dist_r, P=mtx_r)

        # 2. Compute epilines
        lines1 = cv2.computeCorrespondEpilines(p_l, 1, F)
        lines1 = lines1.reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(p_r, 2, F)
        lines2 = lines2.reshape(-1, 3)

        # 3. Calculate distance error from point to line
        # Distance from points in right image to epilines from left image
        for j in range(n_pts):
            x, y = p_r[j][0]
            a, b, c = lines1[j]
            dist = abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
            total_error += dist

        # Distance from points in left image to epilines from right image
        for j in range(n_pts):
            x, y = p_l[j][0]
            a, b, c = lines2[j]
            dist = abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
            total_error += dist
            
        total_points += (n_pts * 2) 

    return total_error / total_points

def main():
    # 1. Prepare Object Points (Asymmetric Grid)
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    for i in range(PATTERN_SIZE[1]): 
        for j in range(PATTERN_SIZE[0]): 
            x = (j + ((i % 2) * 0.5)) * SQUARE_SIZE
            y = i * (SQUARE_SIZE / 2)
            index = i * PATTERN_SIZE[0] + j
            objp[index] = [x, y, 0]

    all_data = [] # Temporary storage for all raw data
    
    # --- Phase 1: Initial Scan (Collect all detectable frames) ---
    print("🚀 Phase 1: Scanning all videos for initial collection...")
    
    for l_vid, r_vid, l_txt, r_txt in VIDEO_PAIRS:
        l_path, r_path = os.path.join(DATA_DIR, l_vid), os.path.join(DATA_DIR, r_vid)
        l_txt_path, r_txt_path = os.path.join(DATA_DIR, l_txt), os.path.join(DATA_DIR, r_txt)
        
        if not os.path.exists(l_path): continue
        loader = StereoDataLoader(l_path, r_path, l_txt_path, r_txt_path)
        
        img_shape = None
        count = 0
        
        while True:
            frame_l, frame_r, fid, _ = loader.get_next_pair()
            if frame_l is None: break
            if img_shape is None: img_shape = frame_l.shape[:2][::-1]

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            found_l, corners_l = cv2.findCirclesGrid(gray_l, PATTERN_SIZE, flags=flags)
            found_r, corners_r = cv2.findCirclesGrid(gray_r, PATTERN_SIZE, flags=flags)
            
            # If found in both, collect it first
            if found_l and found_r:
                all_data.append({
                    'obj': objp,
                    'img_l': corners_l,
                    'img_r': corners_r,
                    'video': l_vid,
                    'fid': fid
                })
                count += 1
                print(f"\r  -> {l_vid}: Collected {count} frames", end="")
        
        print("") # New line
        loader.release()
    
    print(f"\n✅ Initial collection complete. Total raw frames: {len(all_data)}.")
    if len(all_data) < 10: return

    # --- Phase 2: Cleaning (Outlier Rejection) ---
    print("\n🧹 Phase 2: Data Cleaning (Removing high-error frames)...")
    
    temp_obj = [d['obj'] for d in all_data]
    temp_img_l = [d['img_l'] for d in all_data]
    temp_img_r = [d['img_r'] for d in all_data]
    
    # Pre-calibration to get errors for each frame
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(temp_obj, temp_img_l, img_shape, None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(temp_obj, temp_img_r, img_shape, None, None)
    
    err_l = calculate_reprojection_error(temp_obj, temp_img_l, rvecs_l, tvecs_l, mtx_l, dist_l)
    err_r = calculate_reprojection_error(temp_obj, temp_img_r, rvecs_r, tvecs_r, mtx_r, dist_r)
    
    # Filter elite frames
    clean_obj = []
    clean_img_l = []
    clean_img_r = []
    
    kept_count = 0
    for i in range(len(all_data)):
        e_l = err_l[i]
        e_r = err_r[i]
        
        # Only keep frames with error below threshold
        if e_l < MAX_ERROR_THRESHOLD and e_r < MAX_ERROR_THRESHOLD:
            clean_obj.append(all_data[i]['obj'])
            clean_img_l.append(all_data[i]['img_l'])
            clean_img_r.append(all_data[i]['img_r'])
            kept_count += 1
        else:
            # Print rejected frame info for debugging (optional)
            pass 
            # print(f"❌ Rejecting bad frame {all_data[i]['fid']}: L={e_l:.2f}, R={e_r:.2f}")

    print(f"✅ Cleaning complete! Kept {kept_count}/{len(all_data)} high-quality frames (Threshold: {MAX_ERROR_THRESHOLD} px)")

    # --- Phase 3: Final High-Precision Calibration ---
    print("\n🏆 Phase 3: Final Stereo Calibration...")
    
    # Strict termination criteria: 100 iterations or epsilon 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    
    # Re-calculate single camera intrinsics using clean data
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(clean_obj, clean_img_l, img_shape, None, None, criteria=criteria)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(clean_obj, clean_img_r, img_shape, None, None, criteria=criteria)

    # Fix intrinsics to ensure stability
    flags = cv2.CALIB_FIX_INTRINSIC
    
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        clean_obj, clean_img_l, clean_img_r,
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_shape,
        criteria=criteria,
        flags=flags
    )
    
    # --- Phase 4: Calculate Epipolar Error ---
    epi_error = calculate_epipolar_error(clean_img_l, clean_img_r, M1, d1, M2, d2, F)

    print("="*50)
    print(f"📊 Final Report (Based on {kept_count} optimized frames):")
    print(f"✅ Reprojection Error (RMS): {ret:.4f} pixels")
    print(f"✅ Epipolar Error:           {epi_error:.4f} pixels")
    print(f"📏 Calculated Baseline:      {np.linalg.norm(T):.2f} cm")
    print("="*50)

    # Save optimized parameters
    save_path = os.path.join(SRC_DIR, "camera_params.npz")
    np.savez(save_path, mtx_l=M1, dist_l=d1, mtx_r=M2, dist_r=d2, R=R, T=T)
    print(f"💾 Optimized parameters saved to: {save_path}")

if __name__ == "__main__":
    main()