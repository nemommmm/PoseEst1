import numpy as np
import os

# ================= Path Configuration =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")

def main():
    if not os.path.exists(PARAM_PATH):
        print("[Error] Parameter file not found. Please run the calibration script first.")
        return

    print(f"[Info] Loading parameter file: {os.path.basename(PARAM_PATH)}\n")
    data = np.load(PARAM_PATH)

    # Extract data
    mtx_l = data['mtx_l']    # Left camera intrinsics
    dist_l = data['dist_l']  # Left camera distortion
    mtx_r = data['mtx_r']    # Right camera intrinsics
    dist_r = data['dist_r']  # Right camera distortion
    R = data['R']            # Rotation matrix
    T = data['T']            # Translation vector (Unit: cm)

    print("="*50)
    print("       Stereo Camera Calibration Report       ")
    print("="*50)

    # 1. Intrinsics
    # Intrinsic matrix format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    print(f"\n[1. Left Camera Intrinsics]")
    print(f"  - Focal Length (fx, fy): ({mtx_l[0,0]:.2f}, {mtx_l[1,1]:.2f})")
    print(f"  - Principal Point (cx, cy): ({mtx_l[0,2]:.2f}, {mtx_l[1,2]:.2f})")
    
    print(f"\n[2. Right Camera Intrinsics]")
    print(f"  - Focal Length (fx, fy): ({mtx_r[0,0]:.2f}, {mtx_r[1,1]:.2f})")
    print(f"  - Principal Point (cx, cy): ({mtx_r[0,2]:.2f}, {mtx_r[1,2]:.2f})")

    # 2. Extrinsics
    # T is the translation vector, representing the right camera's position relative to the left camera
    print(f"\n[3. Stereo Extrinsics (Relative Pose)]")
    print(f"  - Translation Vector T (cm): \n{T.T}") # Transposed for readability
    
    # Calculate baseline distance
    baseline = np.linalg.norm(T)
    print(f"  - Calculated Baseline: {baseline:.2f} cm")
    
    # 3. Distortion
    print(f"\n[4. Distortion Coefficients (k1, k2, p1, p2, k3)]")
    print(f"  - Left Camera:  {dist_l.ravel()[:5]}")
    print(f"  - Right Camera: {dist_r.ravel()[:5]}")

if __name__ == "__main__":
    main()