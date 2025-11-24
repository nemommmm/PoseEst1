import cv2
import numpy as np
import os
import sys

# Ensure utils.py can be imported from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader

# ================= 1. Paths & Configuration =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Calibration_video")

# Define all video pairs to process
VIDEO_PAIRS = [
    ("cap_0_left.avi", "cap_0_right.avi", "cap_0_left.txt", "cap_0_right.txt"),
    ("cap_1_left.avi", "cap_1_right.avi", "cap_1_left.txt", "cap_1_right.txt")]

# Key Parameter Configuration: Asymmetric Circle Grid
# Based on board: 5 columns, 9 rows
PATTERN_SIZE = (5, 9)   
SQUARE_SIZE = 15.0      # 15 cm (Physical distance between centers)
# ===================================================

def main():
    # ================= 2. Generate Real-World Coordinates =================
    # Coordinate generation for Asymmetric Circle Grid
    # This creates a flattened array storing (x, y, 0)
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    
    # 填充坐标
    for i in range(PATTERN_SIZE[1]): # Iterate rows (0 to 8)
        for j in range(PATTERN_SIZE[0]): # Iterate columns (0 to 4)
            # X-axis logic: Column index * Spacing + (Offset by 0.5 spacing if odd row)
            x = (j + ((i % 2) * 0.5)) * SQUARE_SIZE
            
            # Y-axis logic: Row index * (Half spacing)
            y = i * (SQUARE_SIZE / 2)
            
            # Calculate index in the flattened array
            index = i * PATTERN_SIZE[0] + j
            objp[index] = [x, y, 0]            
    # ================================================================

    # Global containers: store points found in all video frames
    all_objpoints = []   
    all_imgpoints_l = [] 
    all_imgpoints_r = [] 
    
    total_valid_frames = 0
    img_shape = None

    print(f"🚀 Starting processing for {len(VIDEO_PAIRS)} video pairs...")

    # --- Phase 1: Iterate through video pairs to collect data ---
    for v_idx, (l_vid, r_vid, l_txt, r_txt) in enumerate(VIDEO_PAIRS):
        l_path = os.path.join(DATA_DIR, l_vid)
        r_path = os.path.join(DATA_DIR, r_vid)
        l_txt_path = os.path.join(DATA_DIR, l_txt)
        r_txt_path = os.path.join(DATA_DIR, r_txt)

        if not os.path.exists(l_path):
            print(f"⚠️ Skipping: File not found {l_vid}")
            continue

        print(f"\n📂 Reading pair {v_idx + 1}: {l_vid} & {r_vid}")
        loader = StereoDataLoader(l_path, r_path, l_txt_path, r_txt_path)

        while True:
            frame_l, frame_r, frame_id, _ = loader.get_next_pair()
            if frame_l is None:
                break
            
            # Record image shape (only once)
            if img_shape is None:
                img_shape = frame_l.shape[:2][::-1]

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            
            # ⚠️ Must use the ASYMMETRIC flag
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            found_l, corners_l = cv2.findCirclesGrid(gray_l, PATTERN_SIZE, flags=flags)
            found_r, corners_r = cv2.findCirclesGrid(gray_r, PATTERN_SIZE, flags=flags)
            
            if found_l and found_r:
                total_valid_frames += 1
                all_objpoints.append(objp)
                all_imgpoints_l.append(corners_l)
                all_imgpoints_r.append(corners_r)
                
                # Visualization
                cv2.drawChessboardCorners(frame_l, PATTERN_SIZE, corners_l, found_l)
                status = f"Valid: {total_valid_frames} | Pair: {v_idx+1}"
                cv2.putText(frame_l, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Resize for display (show left camera only)
                display = cv2.resize(frame_l, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('Collecting...', display) 
                cv2.waitKey(1) # Fast forward
            
        loader.release()
    
    cv2.destroyAllWindows()
    
    if total_valid_frames < 10:
        print(f"\n❌ Too few valid frames ({total_valid_frames}). Calibration impossible!")
        return

    print(f"\n🛑 Data collection complete! Total valid frames: {total_valid_frames}.")
    print("------------------------------------------------")
    
    # --- Phase 2: Single Camera Calibration (Intrinsics) ---
    print("Running Single Camera Calibration (1/3): Left Camera...")
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(all_objpoints, all_imgpoints_l, img_shape, None, None)
    print(f"   -> Left Camera Initial Error: {ret_l:.4f}")

    print("Running Single Camera Calibration (2/3): Right Camera...")
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(all_objpoints, all_imgpoints_r, img_shape, None, None)
    print(f"   -> Right Camera Initial Error: {ret_r:.4f}")

    # --- Phase 3: Stereo Calibration (Extrinsics) ---
    print("Running Stereo Calibration (3/3): Joint Optimization...")
    # Use single camera results as initial values and fix intrinsics for stability
    flags = cv2.CALIB_FIX_INTRINSIC
    
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        all_objpoints, all_imgpoints_l, all_imgpoints_r,
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_shape,
        flags=flags
    )
    
    print("------------------------------------------------")
    print(f"✅ Final Stereo Calibration RMS: {ret:.4f}")
    
    if ret < 0.5:
        print("🏆 Perfect! High calibration accuracy.")
    elif ret < 1.0:
        print("🎉 Acceptable! Ready for the next step.")
    else:
        print("⚠️ Warning: RMS is high. Please check if circle detection had misalignments.")

    # Save results
    save_path = os.path.join(SRC_DIR, "camera_params.npz")
    np.savez(save_path, mtx_l=M1, dist_l=d1, mtx_r=M2, dist_r=d2, R=R, T=T)
    print(f"💾 Parameters saved to: {save_path}")

if __name__ == "__main__":
    main()