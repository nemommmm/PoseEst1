import numpy as np
import os
import cv2

# ================= Path Configuration =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")


def distortion_model_name(dist):
    num_coeffs = int(dist.size)
    if num_coeffs >= 14:
        return "OpenCV rational model (14 coeffs; k1-k6 + thin-prism/tilt slots)"
    if num_coeffs >= 8:
        return "OpenCV rational model (8 coeffs; k1-k6)"
    if num_coeffs >= 5:
        return "OpenCV standard model (5 coeffs)"
    return f"OpenCV custom distortion vector ({num_coeffs} coeffs)"


def radial_scale(dist, r):
    coeffs = np.zeros(8, dtype=np.float64)
    coeffs[: min(8, dist.size)] = dist.ravel()[: min(8, dist.size)]
    k1, k2, p1, p2, k3, k4, k5, k6 = coeffs
    num = 1.0 + k1 * r**2 + k2 * r**4 + k3 * r**6
    den = 1.0 + k4 * r**2 + k5 * r**4 + k6 * r**6
    return num / den


def summarize_effective_distortion(mtx, dist):
    h, w = 1648, 2048
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1.0, (w, h))
    sample_scales = {r: radial_scale(dist, r) for r in (0.4, 0.8, 1.0, 1.2)}
    pts = np.array(
        [
            [[w / 2.0, h / 2.0]],
            [[0.0, 0.0]],
            [[w, 0.0]],
            [[0.0, h]],
            [[w, h]],
            [[w / 2.0, 0.0]],
            [[w / 2.0, h]],
            [[0.0, h / 2.0]],
            [[w, h / 2.0]],
        ],
        dtype=np.float32,
    )
    und = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    shifts = np.linalg.norm(und.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
    return {
        "new_mtx": new_mtx,
        "roi": roi,
        "sample_scales": sample_scales,
        "corner_shift_max_px": float(np.max(shifts[1:])),
        "edge_shift_mean_px": float(np.mean(shifts[1:])),
    }

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
    print(f"\n[4. Distortion Model]")
    print(f"  - Left Camera:  {distortion_model_name(dist_l)}")
    print(f"  - Right Camera: {distortion_model_name(dist_r)}")

    print(f"\n[5. Raw Distortion Coefficients]")
    print(f"  - Left Camera:  {dist_l.ravel()}")
    print(f"  - Right Camera: {dist_r.ravel()}")
    if dist_l.size > 5 or dist_r.size > 5:
        print("  - Note: with the rational model, raw k1 values are not directly comparable across cameras.")
        print("          Compare the effective radial scaling curve or undistortion shift instead.")

    left_summary = summarize_effective_distortion(mtx_l, dist_l)
    right_summary = summarize_effective_distortion(mtx_r, dist_r)
    print(f"\n[6. Effective Distortion Summary]")
    print(f"  - Left Camera radial scale  r=[0.4, 0.8, 1.0, 1.2]: {left_summary['sample_scales']}")
    print(f"  - Right Camera radial scale r=[0.4, 0.8, 1.0, 1.2]: {right_summary['sample_scales']}")
    print(
        f"  - Left Camera undistortion shift:  max corner {left_summary['corner_shift_max_px']:.2f}px, "
        f"mean edge {left_summary['edge_shift_mean_px']:.2f}px"
    )
    print(
        f"  - Right Camera undistortion shift: max corner {right_summary['corner_shift_max_px']:.2f}px, "
        f"mean edge {right_summary['edge_shift_mean_px']:.2f}px"
    )
    print(f"  - Left Camera ROI after undistortion:  {left_summary['roi']}")
    print(f"  - Right Camera ROI after undistortion: {right_summary['roi']}")

if __name__ == "__main__":
    main()
