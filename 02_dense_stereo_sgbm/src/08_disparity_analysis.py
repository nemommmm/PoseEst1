"""Direction B — Step 0: SGBM Disparity Quality Analysis

Validates whether StereoSGBM can reliably provide disparity at keypoint locations
before committing to the full dense-stereo pipeline run.

What it does:
  1. Load camera params and compute stereoRectify (R1, R2, P1, P2, Q, maps).
  2. Load existing yolo_3d_raw.npz for reference keypoint locations and DLT disparities.
  3. Sample every SAMPLE_STEP-th frame from the stereo video pair.
  4. For each sampled frame: compute SGBM disparity; query the disparity map at each
     keypoint position using a 5×5 median window.
  5. Report:
     - Fill rate per joint (fraction of frames where SGBM returns a valid value).
     - Consistency vs DLT (correlation and MAE between SGBM and stored DLT disparity).
     - Per-distance-bucket (near/far) analysis.
  6. Save diagnostic plots to results/disparity_analysis_*.png.

Decision rule: if mean keypoint fill rate < 0.80, dense stereo is unlikely to work
reliably and Direction B should be reconsidered.

Usage:
    /opt/anaconda3/envs/pose/bin/python src/08_disparity_analysis.py
"""

import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # 02_dense_stereo_sgbm/src/
_METHOD_DIR = os.path.dirname(_SRC_DIR)                         # 02_dense_stereo_sgbm/
PROJECT_ROOT = os.path.dirname(_METHOD_DIR)                     # PoseEst1/
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))
from utils import StereoDataLoader

# ── Configuration ──────────────────────────────────────────────────────────────
SRC_DIR = _SRC_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
RESULTS_DIR = os.path.join(_METHOD_DIR, "results")
PARAM_PATH = os.path.join(PROJECT_ROOT, "shared", "camera_params.npz")
# Reference triangulation results from Direction A (cross-method dependency)
NPZ_PATH = os.path.join(PROJECT_ROOT, "01_stereo_triangulation", "results", "yolo_3d_raw.npz")

SAMPLE_STEP = 30        # analyse every Nth frame
LOOKUP_WINDOW = 5       # window size for median disparity lookup

SGBM_MIN_DISPARITY = 100
SGBM_NUM_DISPARITIES = 256   # must be multiple of 16; covers 100-356 px
SGBM_BLOCK_SIZE = 9

JOINT_NAMES = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle",
]

NEAR_Z_CM = 400.0   # Z < this → "near"
FAR_Z_CM = 500.0    # Z > this → "far"


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_sgbm(min_d: int, num_d: int, block: int) -> cv2.StereoSGBM:
    """Create a StereoSGBM matcher."""
    return cv2.StereoSGBM_create(
        minDisparity=min_d,
        numDisparities=num_d,
        blockSize=block,
        P1=8 * 3 * block**2,
        P2=32 * 3 * block**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def query_disparity_at_keypoints(
    disp_map: np.ndarray,
    kpts: np.ndarray,
    hw: int,
) -> np.ndarray:
    """Return median disparity in a (2hw+1)×(2hw+1) window around each keypoint.

    Args:
        disp_map: (H, W) float32; NaN = invalid.
        kpts: (17, 2) rectified keypoint coordinates.
        hw: half-window size.

    Returns:
        (17,) float32 — NaN where no valid disparity was found.
    """
    H, W = disp_map.shape
    result = np.full(len(kpts), np.nan, dtype=np.float32)
    for j, (x, y) in enumerate(kpts):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - hw), min(W, xi + hw + 1)
        y0, y1 = max(0, yi - hw), min(H, yi + hw + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = disp_map[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch >= SGBM_MIN_DISPARITY)]
        if valid.size > 0:
            result[j] = float(np.median(valid))
    return result


# ── Main analysis ──────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load calibration ──────────────────────────────────────────────────────
    print("[Step 1] Loading calibration parameters...")
    cal = np.load(PARAM_PATH)
    mtx_l, dist_l = cal["mtx_l"], cal["dist_l"]
    mtx_r, dist_r = cal["mtx_r"], cal["dist_r"]
    R_stereo, T_stereo = cal["R"], cal["T"]

    # Get frame size from first video frame
    cap_test = cv2.VideoCapture(os.path.join(DATA_DIR, "0_video_left.avi"))
    ret, frame_test = cap_test.read()
    cap_test.release()
    if not ret:
        print("[Error] Cannot read left video.")
        return
    h, w = frame_test.shape[:2]
    print(f"         Frame size: {w}×{h}")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, (w, h), R_stereo, T_stereo, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w, h), cv2.CV_32FC1)

    baseline_cm = float(np.linalg.norm(T_stereo))
    focal_rect = float(P1[0, 0])
    print(f"         Baseline: {baseline_cm:.2f} cm  |  Focal (rect): {focal_rect:.1f} px")

    # ── Load reference NPZ ────────────────────────────────────────────────────
    print("[Step 2] Loading reference NPZ...")
    npz = np.load(NPZ_PATH, allow_pickle=True)
    kpts_rect_all = npz["keypoints_left_rect"]     # (F, 17, 2)
    dlt_disp_all = npz["disparity_px"]             # (F, 17)
    kpts_3d_all = npz["keypoints"]                 # (F, 17, 3)
    n_total = kpts_rect_all.shape[0]
    print(f"         Total frames in NPZ: {n_total}")

    sample_indices = list(range(0, n_total, SAMPLE_STEP))
    print(f"         Sampling every {SAMPLE_STEP} frames → {len(sample_indices)} analysis frames")

    # ── Initialise SGBM ───────────────────────────────────────────────────────
    print("[Step 3] Building SGBM matcher...")
    sgbm = build_sgbm(SGBM_MIN_DISPARITY, SGBM_NUM_DISPARITIES, SGBM_BLOCK_SIZE)
    hw = LOOKUP_WINDOW // 2

    # ── Open video ────────────────────────────────────────────────────────────
    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"),
        os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"),
        os.path.join(DATA_DIR, "1_video_right.txt"),
    )

    # ── Per-frame analysis ────────────────────────────────────────────────────
    print("[Step 4] Computing SGBM disparity for sampled frames...")
    sgbm_disp_samples = []   # list of (17,) arrays
    dlt_disp_samples = []    # list of (17,) arrays
    z_samples = []           # list of (17,) depth arrays from DLT 3D

    frame_idx = 0
    sample_set = set(sample_indices)

    while True:
        frame_l, frame_r, _, _ = loader.get_next_pair()
        if frame_l is None:
            break
        if frame_idx in sample_set and frame_idx < n_total:
            fl_rect = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
            fr_rect = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

            gray_l = cv2.cvtColor(fl_rect, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(fr_rect, cv2.COLOR_BGR2GRAY)
            raw = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
            dmap = raw.copy()
            dmap[raw < 0] = np.nan

            sgbm_q = query_disparity_at_keypoints(dmap, kpts_rect_all[frame_idx], hw)
            sgbm_disp_samples.append(sgbm_q)
            dlt_disp_samples.append(dlt_disp_all[frame_idx])
            z_samples.append(kpts_3d_all[frame_idx, :, 2])

        frame_idx += 1
        if frame_idx >= n_total:
            break

    loader.release()

    sgbm_disp = np.array(sgbm_disp_samples, dtype=np.float64)   # (S, 17)
    dlt_disp = np.array(dlt_disp_samples, dtype=np.float64)     # (S, 17)
    z_depth = np.array(z_samples, dtype=np.float64)             # (S, 17)

    S = sgbm_disp.shape[0]
    print(f"         Processed {S} sampled frames.")

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n[Step 5] Results")
    print("=" * 70)

    # Per-joint fill rate
    fill_rate = np.array([
        np.mean(np.isfinite(sgbm_disp[:, j])) for j in range(17)
    ])
    mean_fill = float(np.mean(fill_rate))
    print(f"\nMean SGBM fill rate across all joints: {mean_fill:.3f}")
    if mean_fill >= 0.80:
        print("  → PASS (≥ 0.80): dense stereo is feasible")
    else:
        print("  → FAIL (< 0.80): low fill rate — dense stereo may be unreliable")

    print("\nPer-joint fill rate:")
    for j, name in enumerate(JOINT_NAMES):
        bar = "█" * int(fill_rate[j] * 20)
        print(f"  {name:12s}  {fill_rate[j]:.3f}  {bar}")

    # Consistency vs DLT (only where both are valid)
    both_valid = np.isfinite(sgbm_disp) & np.isfinite(dlt_disp) & (dlt_disp > 0)
    sgbm_v = sgbm_disp[both_valid]
    dlt_v = dlt_disp[both_valid]
    if sgbm_v.size > 10:
        mae = float(np.mean(np.abs(sgbm_v - dlt_v)))
        corr = float(np.corrcoef(sgbm_v, dlt_v)[0, 1])
        bias = float(np.mean(sgbm_v - dlt_v))
        print(f"\nSGBM vs DLT disparity consistency ({sgbm_v.size} joint-frames):")
        print(f"  MAE   = {mae:.2f} px")
        print(f"  Bias  = {bias:.2f} px  (SGBM − DLT)")
        print(f"  Corr  = {corr:.4f}")
    else:
        mae, corr, bias = np.nan, np.nan, np.nan
        print("\n  Not enough co-valid samples for consistency analysis.")

    # Distance-bucket analysis
    near_mask = (z_depth < NEAR_Z_CM) & np.isfinite(z_depth)
    far_mask = (z_depth >= FAR_Z_CM) & np.isfinite(z_depth)
    for label, mask in [("Near (<400 cm)", near_mask), ("Far (≥500 cm)", far_mask)]:
        sub_sgbm = sgbm_disp[mask]
        sub_dlt = dlt_disp[mask]
        both = np.isfinite(sub_sgbm) & np.isfinite(sub_dlt) & (sub_dlt > 0)
        fr = float(np.mean(np.isfinite(sub_sgbm))) if mask.any() else np.nan
        print(f"\n{label}  (n={int(mask.sum())} joint-frames):")
        print(f"  Fill rate = {fr:.3f}")
        if both.sum() > 5:
            m = float(np.mean(np.abs(sub_sgbm[both] - sub_dlt[both])))
            print(f"  MAE vs DLT = {m:.2f} px")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Step 6] Saving diagnostic plots...")

    # Plot 1: per-joint fill rate bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(JOINT_NAMES, fill_rate, color=["#2196F3" if v >= 0.80 else "#F44336" for v in fill_rate])
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1.2, label="0.80 threshold")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fill Rate")
    ax.set_title("SGBM Disparity Fill Rate per Joint")
    ax.legend()
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "disparity_analysis_fill_rate.png")
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out1}")

    # Plot 2: SGBM vs DLT scatter
    if sgbm_v.size > 10:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(dlt_v, sgbm_v, alpha=0.15, s=4, color="#1565C0")
        lo = min(dlt_v.min(), sgbm_v.min())
        hi = max(dlt_v.max(), sgbm_v.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y = x")
        ax.set_xlabel("DLT Disparity (px)")
        ax.set_ylabel("SGBM Disparity (px)")
        ax.set_title(f"SGBM vs DLT Disparity\nMAE={mae:.2f}px  Bias={bias:+.2f}px  r={corr:.3f}")
        ax.legend()
        plt.tight_layout()
        out2 = os.path.join(RESULTS_DIR, "disparity_analysis_scatter.png")
        fig.savefig(out2, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out2}")

    # Plot 3: error distribution histogram
    if sgbm_v.size > 10:
        errors = sgbm_v - dlt_v
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(errors, bins=60, range=(-50, 50), color="#1976D2", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.axvline(bias, color="orange", linestyle="-", linewidth=1.5, label=f"Bias={bias:+.2f}px")
        ax.set_xlabel("SGBM − DLT Disparity Error (px)")
        ax.set_ylabel("Count")
        ax.set_title("Disparity Error Distribution")
        ax.legend()
        plt.tight_layout()
        out3 = os.path.join(RESULTS_DIR, "disparity_analysis_error_hist.png")
        fig.savefig(out3, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out3}")

    print("\n[Done] Analysis complete.")
    print(f"  Mean fill rate : {mean_fill:.3f}")
    if np.isfinite(mae):
        print(f"  MAE vs DLT    : {mae:.2f} px")
        print(f"  Bias           : {bias:+.2f} px")
        print(f"  Correlation    : {corr:.4f}")
    decision = "PROCEED with Direction B" if mean_fill >= 0.80 else "RECONSIDER Direction B (low fill rate)"
    print(f"\n→ Decision: {decision}")


if __name__ == "__main__":
    main()
