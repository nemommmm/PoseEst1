#!/opt/anaconda3/envs/pose/bin/python
"""Generate an interactive dense-stereo point cloud HTML for Direction B (DDM).

Features
--------
- Timestamp-based matching between video frames and SKT NPZ frames
  (video has 3015 frames, NPZ has 2801 frames — they are NOT 1:1 by index)
- Per-frame metrics: pixel fill rate, keypoint fill rate, SKT valid joints
- Frame image embedded as base64 JPEG for standalone HTML
- Plotly 3D scatter viewer with skeleton overlay and metadata panel
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from utils_mvnx import MvnxParser
from scipy.interpolate import interp1d
from pose_angle_utils import compute_semantic_joint_angles, build_fair_gt_interpolators, SEMANTIC_ANGLE_NAMES

DATA_DIR = PROJECT_ROOT / "2025_Ergonomics_Data"
RESULTS_DIR = METHOD_DIR / "results"
PARAM_PATH = SHARED_DIR / "camera_params.npz"
DEFAULT_REF_NPZ = PROJECT_ROOT / "01_stereo_triangulation" / "results" / "yolo_3d_raw.npz"
DEFAULT_SUMMARY = RESULTS_DIR / "point_cloud_summary_audit.json"
GT_COMPARISON_JSON = PROJECT_ROOT / "01_stereo_triangulation" / "results" / "skeleton_comparison_dirA.json"
MVNX_PATH = Path.home() / "Desktop" / "MVE386 Project Course" / "Xsens_ground_truth" / "Aitor-001.mvnx"
FAIR_GT_NPZ = SHARED_DIR / "fair_gt_angles.npz"

SGBM_MIN_DISPARITY = int(os.environ.get("POSE_SGBM_MIN_DISPARITY", "100"))
SGBM_NUM_DISPARITIES = int(os.environ.get("POSE_SGBM_NUM_DISPARITIES", "256"))
SGBM_BLOCK_SIZE = int(os.environ.get("POSE_SGBM_BLOCK_SIZE", "9"))
LOOKUP_WINDOW = int(os.environ.get("POSE_DISPARITY_WINDOW", "5"))

# Default frames: good_1, good_2, good_3 chosen from previous audit; bad & low_coverage
# re-selected with correct timestamp-based kp_fill after skeleton alignment fix.
DEFAULT_FRAMES = {
    "good_1": 2685,       # kp_fill 93%, pixel_fill 54% — high coverage, person walking
    "good_2": 1625,       # kp_fill 88%, pixel_fill 55% — good mid-sequence frame
    "good_3": 1905,       # kp_fill 87%, pixel_fill 54% — alternative good example
    "bad": 2040,          # kp_fill 46%, pixel_fill 57% — dark uniform texture failure
    "low_coverage": 2110, # kp_fill 53%, pixel_fill 56% — partial body coverage failure
}

FRAME_LABELS = {
    "good_1": "Good frame #1 — high body disparity coverage",
    "good_2": "Good frame #2 — typical mid-sequence coverage",
    "good_3": "Good frame #3 — alternative good example",
    "bad": "Bad frame — dark clothing causes keypoint disparity failure",
    "low_coverage": "Low coverage — partial body texture failure",
}

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Xsens segment connections for GT skeleton overlay
XSENS_SEGMENTS = [
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
]
XSENS_LINKS = [
    ("Pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"), ("T8", "Neck"), ("Neck", "Head"),
    ("T8", "RightShoulder"), ("RightShoulder", "RightUpperArm"), ("RightUpperArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("T8", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"), ("LeftUpperArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Pelvis", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"), ("RightLowerLeg", "RightFoot"),
    ("Pelvis", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"), ("LeftLowerLeg", "LeftFoot"),
]

# COCO index → Xsens segment name (for per-frame GT distance computation)
COCO_TO_XSENS = {
    5: "LeftShoulder", 6: "RightShoulder",
    7: "LeftUpperArm", 8: "RightUpperArm",
    11: "LeftUpperLeg", 12: "RightUpperLeg",
    13: "LeftLowerLeg", 14: "RightLowerLeg",
}


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

def load_video_timestamps(txt_path: Path) -> np.ndarray:
    """Load unix timestamps from StereoDataLoader txt file."""
    ts: List[float] = []
    with open(txt_path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                ts.append(int(parts[1]) + int(parts[2]) / 1e6)
    return np.array(ts, dtype=np.float64)


def vid_to_npz(vid_frame_idx: int, vid_ts: np.ndarray, npz_ts: np.ndarray) -> int:
    """Find the NPZ frame whose timestamp is closest to the given video frame."""
    return int(np.argmin(np.abs(npz_ts - vid_ts[vid_frame_idx])))


# ---------------------------------------------------------------------------
# Ground truth (Xsens) loading and alignment
# ---------------------------------------------------------------------------

def load_gt_data(mvnx_path: Path) -> Optional[Dict]:
    """Load Xsens MVNX and build per-segment time interpolators.

    Returns a dict with keys 'interps' (segment→interp) and 'ts_range' (min, max).
    Positions are in cm. Timestamps are zeroed to recording start.
    """
    if not mvnx_path.exists():
        return None
    parser = MvnxParser(str(mvnx_path))
    parser.parse()
    xsens_ts = parser.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]  # zero-base relative to Xsens start
    interps: Dict[str, Any] = {}
    for seg_name in XSENS_SEGMENTS:
        seg_data = parser.get_segment_data(seg_name)
        if seg_data is None:
            continue
        interps[seg_name] = interp1d(
            xsens_ts, seg_data[xidx], axis=0, kind="linear",
            bounds_error=False, fill_value=np.nan,
        )
    return {"interps": interps, "ts_range": (float(xsens_ts[0]), float(xsens_ts[-1]))}


def load_gt_alignment(json_path: Path) -> Optional[Dict]:
    """Load Kabsch alignment from skeleton_comparison_dirA.json.

    The stored R, t maps SKT camera coords → Xsens world coords:
        p_gt = R @ p_skt + t
    Inverse to map GT → camera:
        p_cam = R.T @ (p_gt - t)
    """
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    R = np.asarray(data["rotation_matrix"], dtype=np.float64)
    t = np.asarray(data["translation_cm"], dtype=np.float64)
    offset_s = float(data.get("offset_seconds", 17.25))
    return {"R": R, "t": t, "offset_s": offset_s,
            "R_inv": R.T, "t_inv": -R.T @ t}


def gt_skeleton_for_frame(
    vid_idx: int,
    vid_ts: np.ndarray,
    gt_data: Dict,
    alignment: Dict,
) -> Optional[Dict]:
    """Compute GT skeleton in camera display coordinates for one video frame.

    Parameters
    ----------
    vid_idx : video frame index
    vid_ts  : absolute unix timestamps for each video frame
    gt_data : output of load_gt_data()
    alignment : output of load_gt_alignment()

    Returns
    -------
    Dict with 'points' (segment_name → 3-vec in camera cm) and 'links'.
    Returns None if GT data is unavailable at this timestamp.
    """
    # Video relative time → Xsens relative time
    subject_t = vid_ts[vid_idx] - vid_ts[0]
    gt_t = subject_t - alignment["offset_s"]
    if gt_t < 0:
        return None

    R_inv = alignment["R_inv"]
    t_inv = alignment["t_inv"]

    points_cam: Dict[str, np.ndarray] = {}
    for seg_name, interp_fn in gt_data["interps"].items():
        p_gt = interp_fn(gt_t)
        if not np.isfinite(p_gt).all():
            continue
        p_cam = R_inv @ p_gt + t_inv
        points_cam[seg_name] = p_cam

    if not points_cam:
        return None

    # Build display coords dict (for rendering) and keep camera coords (for distance)
    seg_display: Dict[str, List] = {}
    for seg_name, p_cam in points_cam.items():
        seg_display[seg_name] = to_display_coords(p_cam[np.newaxis])[0].tolist()

    return {
        "segment_points": seg_display,   # display coords for Plotly
        "_cam_points": points_cam,        # camera coords for distance computation
        "links": XSENS_LINKS,
    }


# ---------------------------------------------------------------------------
# Camera / SGBM setup
# ---------------------------------------------------------------------------

def load_rectification(param_path: Path, img_hw: Tuple[int, int]) -> Tuple:
    """Build stereo rectification maps and Q matrix from calibration npz."""
    calibration = np.load(param_path)
    mtx_l, dist_l = calibration["mtx_l"], calibration["dist_l"]
    mtx_r, dist_r = calibration["mtx_r"], calibration["dist_r"]
    rot, trans = calibration["R"], calibration["T"]
    h, w = img_hw
    r1, r2, p1, p2, q_mat, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, (w, h), rot, trans, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, r1, p1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, r2, p2, (w, h), cv2.CV_32FC1)
    return map1_l, map2_l, map1_r, map2_r, q_mat, mtx_l


def build_sgbm() -> cv2.StereoSGBM:
    """Create a StereoSGBM matcher with current environment settings."""
    bs = SGBM_BLOCK_SIZE
    return cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISPARITY,
        numDisparities=SGBM_NUM_DISPARITIES,
        blockSize=bs,
        P1=8 * 3 * bs**2,
        P2=32 * 3 * bs**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def fetch_frame_pair(
    vid_idx: int,
    cap_l: cv2.VideoCapture,
    cap_r: cv2.VideoCapture,
) -> Tuple[np.ndarray, np.ndarray]:
    """Seek to vid_idx and read one stereo pair (180° rotation applied)."""
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
    ok_l, fl = cap_l.read()
    ok_r, fr = cap_r.read()
    if not ok_l or not ok_r:
        raise IndexError(f"Cannot read video frame {vid_idx}")
    return cv2.rotate(fl, cv2.ROTATE_180), cv2.rotate(fr, cv2.ROTATE_180)


# ---------------------------------------------------------------------------
# Disparity + point cloud
# ---------------------------------------------------------------------------

def compute_disparity(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    map1_l, map2_l, map1_r, map2_r,
    sgbm: cv2.StereoSGBM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rectify a stereo pair and compute SGBM disparity. Returns (left_rect, disparity)."""
    left_rect = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)
    gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    disparity = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disparity[disparity < 0] = np.nan
    return left_rect, disparity


def build_point_cloud(
    disparity: np.ndarray,
    color_bgr: np.ndarray,
    q_mat: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Reproject disparity to 3D and downsample for browser rendering.

    Returns (points_cm, colors_rgb, total_count_before_subsample).
    """
    disparity_filled = np.nan_to_num(disparity, nan=0.0).astype(np.float32)
    points_3d = cv2.reprojectImageTo3D(disparity_filled, q_mat)
    mask = np.isfinite(disparity) & (disparity >= SGBM_MIN_DISPARITY)
    mask &= np.isfinite(points_3d).all(axis=2)
    z_vals = points_3d[:, :, 2]
    mask &= (z_vals > 30.0) & (z_vals < 1200.0)
    all_points = points_3d[mask].astype(np.float32)
    all_colors = color_bgr[mask][:, ::-1].astype(np.uint8)
    total_count = int(len(all_points))
    if total_count > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(total_count, size=max_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]
    return all_points, all_colors, total_count


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gt_anchor_distance_cm(
    kp_3d: np.ndarray,
    gt_skeleton: Optional[Dict],
    alignment: Optional[Dict],
) -> Optional[float]:
    """Mean distance (cm) between SKT joints and GT segments in camera space.

    Uses COCO_TO_XSENS mapping (8 anchor pairs). Returns None if not available.
    Note: kp_3d is in camera coords; GT is also mapped to camera coords inside
    gt_skeleton_for_frame() using the inverse Kabsch transform.
    """
    if gt_skeleton is None or alignment is None:
        return None
    R_inv = alignment["R_inv"]
    t_inv = alignment["t_inv"]
    dists = []
    seg_pts = gt_skeleton["segment_points"]
    # segment_points are already in display coords; we need camera coords
    # Re-compute from GT world via inverse Kabsch for fair comparison with kp_3d
    for coco_idx, seg_name in COCO_TO_XSENS.items():
        if seg_name not in gt_skeleton.get("_cam_points", {}):
            continue
        p_gt_cam = gt_skeleton["_cam_points"][seg_name]
        p_skt = kp_3d[coco_idx]
        if np.isfinite(p_skt).all() and np.isfinite(p_gt_cam).all():
            dists.append(float(np.linalg.norm(p_skt - p_gt_cam)))
    return float(np.mean(dists)) if dists else None


def compute_angle_metrics(
    kp_3d: np.ndarray,
    vid_idx: int,
    vid_ts: np.ndarray,
    fair_gt_interps: Dict,
    offset_s: float,
) -> Dict:
    """Compute per-joint angle error (estimated vs Fair GT) for one frame.

    Uses Fair GT = angles computed from Xsens 3D segment positions with the same
    geometric formula as the SKT pipeline — removes angle-definition bias.

    Returns dict with per-joint estimated/GT/error values and overall MAE.
    """
    subject_t = vid_ts[vid_idx] - vid_ts[0]
    gt_t = subject_t - offset_s

    est_angles = compute_semantic_joint_angles(kp_3d)
    results: Dict[str, Any] = {}
    errors = []
    for name in SEMANTIC_ANGLE_NAMES:
        est = est_angles.get(name, np.nan)
        interp_fn = fair_gt_interps.get(name)
        gt_val = float(interp_fn(gt_t)) if interp_fn is not None else np.nan
        err = abs(est - gt_val) if np.isfinite(est) and np.isfinite(gt_val) else np.nan
        results[name] = {
            "est": round(float(est), 1) if np.isfinite(est) else None,
            "gt": round(float(gt_val), 1) if np.isfinite(gt_val) else None,
            "err": round(float(err), 1) if np.isfinite(err) else None,
        }
        if np.isfinite(err):
            errors.append(err)

    mae = round(float(np.mean(errors)), 1) if errors else None
    worst = max(results.items(), key=lambda x: x[1]["err"] or 0) if errors else (None, {})
    return {
        "angle_mae_deg": mae,
        "worst_joint": worst[0],
        "worst_joint_err_deg": worst[1].get("err"),
        "per_joint": results,
    }


def compute_metrics(
    disparity: np.ndarray,
    kp_3d: np.ndarray,
    mtx_l: np.ndarray,
) -> Dict[str, float]:
    """Compute fill-rate metrics for a frame.

    Parameters
    ----------
    disparity:
        SGBM disparity map (NaN = invalid).
    kp_3d:
        17×3 array of 3D keypoints from SKT NPZ (may contain NaN).
    mtx_l:
        Left camera intrinsic matrix.

    Returns
    -------
    dict with keys: pixel_fill_pct, kp_fill_pct, skt_valid_joints
    """
    h, w = disparity.shape
    pixel_fill = float((disparity >= SGBM_MIN_DISPARITY).mean()) * 100.0

    valid_mask = np.isfinite(kp_3d).all(axis=1)
    valid_joints = int(valid_mask.sum())

    kp_fill = 0.0
    if valid_joints > 0:
        kp_valid = kp_3d[valid_mask]
        fx, fy = mtx_l[0, 0], mtx_l[1, 1]
        cx_cam, cy_cam = mtx_l[0, 2], mtx_l[1, 2]
        z = kp_valid[:, 2]
        x_pix = (kp_valid[:, 0] * fx / z + cx_cam).astype(int)
        y_pix = (kp_valid[:, 1] * fy / z + cy_cam).astype(int)
        x_pix = np.clip(x_pix, 0, w - 1)
        y_pix = np.clip(y_pix, 0, h - 1)
        lw = LOOKUP_WINDOW
        kp_good = sum(
            1
            for xp, yp in zip(x_pix, y_pix)
            if np.any(
                disparity[
                    max(0, yp - lw):min(h, yp + lw + 1),
                    max(0, xp - lw):min(w, xp + lw + 1),
                ] >= SGBM_MIN_DISPARITY
            )
        )
        kp_fill = kp_good / valid_joints * 100.0

    return {
        "pixel_fill_pct": round(pixel_fill, 1),
        "kp_fill_pct": round(kp_fill, 1),
        "skt_valid_joints": valid_joints,
    }


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def frame_to_b64_jpeg(frame_bgr: np.ndarray, width: int = 800) -> str:
    """Encode a BGR frame as a resized base64 JPEG data URI."""
    h, w = frame_bgr.shape[:2]
    new_h = int(h * width / w)
    resized = cv2.resize(frame_bgr, (width, new_h))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def disparity_to_b64_jpeg(
    disparity: np.ndarray,
    frame_bgr: np.ndarray,
    width: int = 800,
) -> str:
    """Render a colormap of the disparity map side-by-side with the frame."""
    h, w_full = disparity.shape
    vis = np.zeros_like(disparity)
    mask = disparity >= SGBM_MIN_DISPARITY
    if mask.any():
        d = disparity[mask]
        vis[mask] = (d - d.min()) / (d.max() - d.min() + 1e-6) * 255
    disp_color = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_TURBO)
    disp_color[~mask] = 30
    combined = np.hstack([frame_bgr, disp_color])
    new_h = int(combined.shape[0] * width / combined.shape[1])
    resized = cv2.resize(combined, (width, new_h))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Viewer coordinate transform
# ---------------------------------------------------------------------------

def to_display_coords(points: np.ndarray) -> np.ndarray:
    """Map OpenCV camera coords to viewer coords (upright person).

    OpenCV: X-right, Y-down, Z-forward.
    Display: X-right, Y-depth, Z-up (i.e. Z = -Y_opencv).
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 3:
        return pts
    return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])


def skeleton_payload(points_3d: np.ndarray) -> Dict:
    """Convert 17-joint skeleton to viewer-coordinate JSON payload."""
    disp = to_display_coords(np.asarray(points_3d, dtype=np.float32))
    return {"points": disp.tolist(), "edges": COCO_EDGES}


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CACHE = Path.home() / ".cache" / "plotly-2.35.2.min.js"


def get_plotly_tag() -> str:
    """Return a self-contained <script> tag with Plotly JS.

    Downloads from CDN on first use and caches locally. Falls back to CDN
    link if download fails.
    """
    if PLOTLY_CACHE.exists():
        js = PLOTLY_CACHE.read_text(encoding="utf-8")
        return f"<script>{js}</script>"
    try:
        import urllib.request
        print(f"Downloading Plotly from {PLOTLY_CDN} ...")
        PLOTLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PLOTLY_CDN, PLOTLY_CACHE)
        js = PLOTLY_CACHE.read_text(encoding="utf-8")
        print(f"  Cached to {PLOTLY_CACHE} ({len(js)//1024} KB)")
        return f"<script>{js}</script>"
    except Exception as exc:
        print(f"  WARNING: could not download Plotly ({exc}). Using CDN link.")
        return f'<script src="{PLOTLY_CDN}"></script>'


def make_html(frames: List[Dict], plotly_tag: str = "") -> str:
    """Generate the self-contained Plotly HTML viewer."""
    if not plotly_tag:
        plotly_tag = f'<script src="{PLOTLY_CDN}"></script>'
    frames_json = json.dumps(frames)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Direction B (DDM): Interactive Dense Stereo Point Cloud</title>
  {plotly_tag}
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; background: #f0f2f5; color: #1a1a2e; }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 1.4rem; margin: 0 0 6px; }}
    .subtitle {{ color: #555; font-size: 0.88rem; margin-bottom: 16px; }}
    .row {{ display: grid; grid-template-columns: 300px 1fr; gap: 16px; }}
    .card {{ background: white; border-radius: 12px;
             box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 16px; }}
    #plot {{ width: 100%; height: 720px; }}
    select {{ width: 100%; padding: 8px 10px; font-size: 14px; border-radius: 6px;
             border: 1px solid #ccc; margin-bottom: 14px; }}
    .meta-section {{ margin-bottom: 12px; }}
    .meta-section h4 {{ margin: 0 0 6px; font-size: 0.82rem; text-transform: uppercase;
                        letter-spacing: 0.05em; color: #888; }}
    .metric-row {{ display: flex; justify-content: space-between;
                   padding: 5px 0; border-bottom: 1px solid #f0f0f0; font-size: 0.85rem; }}
    .metric-row:last-child {{ border-bottom: none; }}
    .metric-key {{ color: #555; }}
    .metric-val {{ font-weight: 600; }}
    .val-good {{ color: #2a8a4a; }}
    .val-warn {{ color: #c0622a; }}
    .val-bad  {{ color: #b71c1c; }}
    .tag-badge {{ display: inline-block; padding: 2px 8px; border-radius: 20px;
                  font-size: 0.78rem; font-weight: 600; margin-bottom: 10px; }}
    .tag-good_1,.tag-good_2,.tag-good_3 {{ background:#d4edda; color:#155724; }}
    .tag-bad {{ background:#f8d7da; color:#721c24; }}
    .tag-low_coverage {{ background:#fff3cd; color:#856404; }}
    #frameImg {{ width: 100%; border-radius: 8px; margin-top: 12px; }}
    #frameLabel {{ font-size: 0.82rem; color: #555; margin-top: 6px; line-height: 1.4; }}
    .legend-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; font-size:0.82rem; }}
    .legend-dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
    .swatch-line {{ width:24px; height:4px; border-radius:2px; flex-shrink:0; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Direction B (DDM): Interactive Dense Stereo Point Cloud</h1>
  <p class="subtitle">SGBM dense disparity viewer. Select a representative frame to explore the 3D point cloud, skeleton overlay, and fill-rate diagnostics. All frames are upright (180° raw-video rotation applied before rectification).</p>
  <div class="row">
    <div class="card">
      <label for="frameSelect"><strong>Representative frame</strong></label>
      <select id="frameSelect"></select>

      <div id="tagBadge" class="tag-badge"></div>
      <div id="frameLabel"></div>
      <img id="frameImg" alt="Left camera frame" />

      <div class="meta-section" style="margin-top:14px">
        <h4>Disparity fill-rate &amp; accuracy</h4>
        <div id="metricPanel"></div>
      </div>

      <div class="meta-section">
        <h4>Joint angles vs Fair GT</h4>
        <div id="anglePanel"></div>
      </div>

      <div class="meta-section">
        <h4>Definitions</h4>
        <div style="font-size:0.79rem; color:#555; line-height:1.55;">
          <b>Pixel fill rate:</b> % of image pixels with valid SGBM disparity; scene-level texture coverage.<br/>
          <b>Keypoint fill rate:</b> % of SKT joints whose pixel has valid SGBM disparity; body-specific coverage.<br/>
          <b>GT anchor dist:</b> mean 3D distance (cm) between 8 SKT–Xsens joint pairs; position accuracy.<br/>
          <b>Angle MAE:</b> mean absolute error across 8 joints (Shoulder/Elbow/Hip/Knee) vs Fair GT (Xsens geometric angles). Lower = better.<br/>
          <b>Fair GT:</b> angles computed from Xsens 3D positions using the same formula as SKT — removes angle-definition bias.
        </div>
      </div>

      <div class="meta-section">
        <h4>Legend</h4>
        <div class="legend-row"><span class="legend-dot" style="background:#888"></span>Dense point cloud</div>
        <div class="legend-row"><span class="swatch-line" style="background:#ff7a18"></span>SKT skeleton (estimated)</div>
        <div class="legend-row"><span class="legend-dot" style="background:#00bcd4"></span>SKT skeleton joints</div>
        <div class="legend-row"><span class="swatch-line" style="background:#43a047"></span>Xsens GT skeleton</div>
        <div class="legend-row"><span class="legend-dot" style="background:#a5d6a7"></span>Xsens GT joints</div>
      </div>
    </div>

    <div class="card">
      <div id="plot"></div>
    </div>
  </div>
</div>
<script>
  const payload = {frames_json};
  const select = document.getElementById('frameSelect');
  payload.forEach((frame, idx) => {{
    const opt = document.createElement('option');
    opt.value = idx;
    const kf = frame.meta.kp_fill_pct;
    const indicator = kf >= 80 ? '✅' : kf >= 60 ? '⚠️' : '❌';
    opt.textContent = `${{indicator}} ${{frame.label}} (vid=${{frame.frame_idx}}, kp_fill=${{kf}}%)`;
    select.appendChild(opt);
  }});

  function metricClass(key, val) {{
    if (key === 'kp_fill_pct') return val >= 80 ? 'val-good' : val >= 60 ? 'val-warn' : 'val-bad';
    if (key === 'pixel_fill_pct') return val >= 50 ? 'val-good' : val >= 35 ? 'val-warn' : 'val-bad';
    if (key === 'gt_anchor_dist_cm') return val == null ? '' : val <= 20 ? 'val-good' : val <= 50 ? 'val-warn' : 'val-bad';
    return '';
  }}

  function renderFrame(idx) {{
    const frame = payload[idx];
    // Tag badge
    const badge = document.getElementById('tagBadge');
    badge.textContent = frame.tag.replace('_', ' ');
    badge.className = 'tag-badge tag-' + frame.tag;

    // Label
    document.getElementById('frameLabel').textContent = frame.label;

    // Frame image
    const img = document.getElementById('frameImg');
    if (frame.frame_image) {{
      img.src = frame.frame_image;
      img.style.display = 'block';
    }} else {{
      img.style.display = 'none';
    }}

    // Metric panel
    const panel = document.getElementById('metricPanel');
    const metricDefs = [
      ['pixel_fill_pct', 'Pixel fill rate', '%'],
      ['kp_fill_pct', 'Keypoint fill rate', '%'],
      ['skt_valid_joints', 'SKT joints visible', '/17'],
      ['gt_anchor_dist_cm', 'GT anchor dist', ' cm'],
      ['point_count_rendered', 'Points rendered', ''],
    ];
    panel.innerHTML = metricDefs.map(([key, label, unit]) => {{
      const val = frame.meta[key];
      const cls = metricClass(key, val);
      const display = (val == null || val === undefined) ? 'N/A' : val + unit;
      return `<div class="metric-row"><span class="metric-key">${{label}}</span>
              <span class="metric-val ${{cls}}">${{display}}</span></div>`;
    }}).join('');

    // Angle panel
    const anglePanel = document.getElementById('anglePanel');
    const am = frame.angle_metrics;
    if (!am || !am.per_joint) {{
      anglePanel.innerHTML = '<div style="color:#aaa;font-size:0.82rem">Not available</div>';
    }} else {{
      const mae = am.angle_mae_deg;
      const maeCls = mae == null ? '' : mae <= 15 ? 'val-good' : mae <= 30 ? 'val-warn' : 'val-bad';
      let html = `<div class="metric-row" style="margin-bottom:6px">
        <span class="metric-key"><b>MAE (8 joints)</b></span>
        <span class="metric-val ${{maeCls}}">${{mae != null ? mae + '°' : 'N/A'}}</span>
      </div>`;
      Object.entries(am.per_joint).forEach(([joint, v]) => {{
        const errCls = v.err == null ? '' : v.err <= 15 ? 'val-good' : v.err <= 30 ? 'val-warn' : 'val-bad';
        const isWorst = joint === am.worst_joint;
        const errStr = v.err != null ? v.err + '°' : '—';
        html += `<div class="metric-row" style="${{isWorst ? 'background:#fff8e1;' : ''}}">
          <span class="metric-key" style="font-size:0.79rem">${{isWorst ? '⚠ ' : ''}}${{joint}}</span>
          <span class="metric-val ${{errCls}}" style="font-size:0.79rem">${{errStr}}</span>
        </div>`;
      }});
      anglePanel.innerHTML = html;
    }}

    // 3D plot
    const skel = frame.skeleton;
    const traces = [{{
      type: 'scatter3d', mode: 'markers',
      x: frame.points.map(p => p[0]),
      y: frame.points.map(p => p[1]),
      z: frame.points.map(p => p[2]),
      marker: {{ size: 1.8, color: frame.colors.map(c => `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`), opacity: 0.80 }},
      name: 'Dense point cloud', hoverinfo: 'skip'
    }}];
    skel.edges.forEach((edge, i) => {{
      const p0 = skel.points[edge[0]], p1 = skel.points[edge[1]];
      if (!Number.isFinite(p0[0]) || !Number.isFinite(p1[0])) return;
      traces.push({{
        type: 'scatter3d', mode: 'lines',
        x: [p0[0], p1[0]], y: [p0[1], p1[1]], z: [p0[2], p1[2]],
        line: {{ color: '#ff7a18', width: 6 }},
        name: i === 0 ? 'SKT skeleton' : '', showlegend: i === 0, hoverinfo: 'skip'
      }});
    }});
    traces.push({{
      type: 'scatter3d', mode: 'markers',
      x: skel.points.map(p => p[0]),
      y: skel.points.map(p => p[1]),
      z: skel.points.map(p => p[2]),
      marker: {{ size: 4, color: '#00bcd4' }},
      name: 'SKT joints', hoverinfo: 'skip'
    }});

    // GT skeleton overlay (Xsens ground truth)
    const gt = frame.gt_skeleton;
    if (gt && gt.segment_points) {{
      const segPts = gt.segment_points;
      let gtFirstEdge = true;
      gt.links.forEach(([segA, segB]) => {{
        const p0 = segPts[segA], p1 = segPts[segB];
        if (!p0 || !p1) return;
        if (!Number.isFinite(p0[0]) || !Number.isFinite(p1[0])) return;
        traces.push({{
          type: 'scatter3d', mode: 'lines',
          x: [p0[0], p1[0]], y: [p0[1], p1[1]], z: [p0[2], p1[2]],
          line: {{ color: '#43a047', width: 5 }},
          name: gtFirstEdge ? 'Xsens GT' : '', showlegend: gtFirstEdge, hoverinfo: 'skip'
        }});
        gtFirstEdge = false;
      }});
      // GT joint markers
      const gtJoints = Object.values(segPts).filter(p => p && Number.isFinite(p[0]));
      if (gtJoints.length > 0) {{
        traces.push({{
          type: 'scatter3d', mode: 'markers',
          x: gtJoints.map(p => p[0]), y: gtJoints.map(p => p[1]), z: gtJoints.map(p => p[2]),
          marker: {{ size: 3.5, color: '#a5d6a7' }},
          name: 'Xsens GT joints', hoverinfo: 'skip'
        }});
      }}
    }}

    Plotly.newPlot('plot', traces, {{
      scene: {{
        xaxis: {{ title: 'X (cm, right)' }},
        yaxis: {{ title: 'Y (cm, depth)' }},
        zaxis: {{ title: 'Z (cm, up)' }},
        aspectmode: 'data',
        camera: {{ eye: {{ x: 1.55, y: -1.65, z: 0.85 }} }}
      }},
      margin: {{ l: 0, r: 0, b: 0, t: 40 }},
      title: `Frame ${{frame.frame_idx}} — ${{frame.tag}} (kp_fill ${{frame.meta.kp_fill_pct}}%)`
    }}, {{responsive: true}});
  }}

  select.addEventListener('change', e => renderFrame(Number(e.target.value)));
  renderFrame(0);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Create an interactive dense stereo point-cloud HTML.")
    p.add_argument("--reference-npz", default=str(DEFAULT_REF_NPZ))
    p.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    p.add_argument("--output-html", default=str(RESULTS_DIR / "interactive_pointcloud.html"))
    p.add_argument("--output-json", default=str(RESULTS_DIR / "interactive_pointcloud.json"))
    p.add_argument("--max-points", type=int, default=30000)
    p.add_argument("--frames", default="", help="Comma-separated vid_idx:tag pairs, e.g. '2685:good_1,2040:bad'.")
    return p.parse_args()


def main() -> None:
    """Generate the HTML viewer with all enhanced features."""
    args = parse_args()

    # Parse frame selection
    if args.frames.strip():
        selected: Dict[str, int] = {}
        for token in args.frames.split(","):
            token = token.strip()
            if ":" in token:
                idx_str, tag = token.split(":", 1)
                selected[tag.strip()] = int(idx_str.strip())
            else:
                selected[f"frame_{token}"] = int(token)
    else:
        selected = dict(DEFAULT_FRAMES)

    # Load NPZ and timestamps
    print("Loading NPZ...")
    skt = np.load(args.reference_npz, allow_pickle=True)
    npz_kp: np.ndarray = skt["keypoints"]    # shape N×17×3
    npz_ts: np.ndarray = skt["timestamps"]

    print("Loading video timestamps...")
    vid_ts = load_video_timestamps(DATA_DIR / "0_video_left.txt")
    print(f"  Video frames: {len(vid_ts)}, NPZ frames: {len(npz_ts)}")

    # Load Xsens GT and alignment
    print("Loading Xsens GT...")
    gt_data = load_gt_data(MVNX_PATH)
    alignment = load_gt_alignment(GT_COMPARISON_JSON)
    if gt_data is None:
        print("  WARNING: MVNX file not found — GT skeleton overlay disabled.")
    if alignment is None:
        print("  WARNING: skeleton_comparison_dirA.json not found — GT overlay disabled.")
    gt_available = gt_data is not None and alignment is not None
    if gt_available:
        print(f"  GT loaded. Xsens range: {gt_data['ts_range'][0]:.1f}–{gt_data['ts_range'][1]:.1f}s, "
              f"offset={alignment['offset_s']}s")

    print("Loading Fair GT angle interpolators...")
    fair_gt_interps = build_fair_gt_interpolators(str(FAIR_GT_NPZ))
    if fair_gt_interps:
        print(f"  Fair GT angles loaded: {list(fair_gt_interps.keys())}")
    else:
        print("  WARNING: fair_gt_angles.npz not found — angle metrics disabled.")

    # Camera setup
    cap_l = cv2.VideoCapture(str(DATA_DIR / "0_video_left.avi"))
    cap_r = cv2.VideoCapture(str(DATA_DIR / "1_video_right.avi"))
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, tmp = cap_l.read()
    if not ok:
        raise RuntimeError("Cannot read left video.")
    img_hw = tmp.shape[:2]
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)

    map1_l, map2_l, map1_r, map2_r, q_mat, mtx_l = load_rectification(PARAM_PATH, img_hw)
    sgbm = build_sgbm()

    # Process frames
    payload_frames: List[Dict] = []
    for tag, vid_idx in selected.items():
        print(f"Processing {tag} (video frame {vid_idx})...")
        frame_l, frame_r = fetch_frame_pair(vid_idx, cap_l, cap_r)
        left_rect, disparity = compute_disparity(frame_l, frame_r, map1_l, map2_l, map1_r, map2_r, sgbm)

        # Timestamp-based NPZ frame lookup
        npz_idx = vid_to_npz(vid_idx, vid_ts, npz_ts)
        delta_ms = abs(npz_ts[npz_idx] - vid_ts[vid_idx]) * 1000
        print(f"  -> NPZ frame {npz_idx} (Δt={delta_ms:.0f}ms)")

        kp_3d = npz_kp[npz_idx]
        metrics = compute_metrics(disparity, kp_3d, mtx_l)
        print(f"  pixel_fill={metrics['pixel_fill_pct']}%  kp_fill={metrics['kp_fill_pct']}%  vis={metrics['skt_valid_joints']}")

        # Angle metrics vs Fair GT
        angle_metrics = None
        if fair_gt_interps:
            angle_metrics = compute_angle_metrics(
                kp_3d, vid_idx, vid_ts, fair_gt_interps,
                offset_s=alignment["offset_s"] if alignment else 17.25,
            )
            mae = angle_metrics["angle_mae_deg"]
            worst = angle_metrics["worst_joint"]
            print(f"  Angle MAE: {mae}°  worst: {worst} {angle_metrics['worst_joint_err_deg']}°")

        # GT skeleton at this frame's timestamp
        gt_skeleton = None
        gt_dist_cm = None
        if gt_available:
            gt_skeleton = gt_skeleton_for_frame(vid_idx, vid_ts, gt_data, alignment)
            gt_dist_cm = gt_anchor_distance_cm(kp_3d, gt_skeleton, alignment)
            if gt_dist_cm is not None:
                print(f"  GT anchor distance: {gt_dist_cm:.1f} cm")

        # Strip internal camera-coord key before JSON serialization
        gt_payload = None
        if gt_skeleton is not None:
            gt_payload = {
                "segment_points": gt_skeleton["segment_points"],
                "links": [[a, b] for a, b in gt_skeleton["links"]],
            }

        # Point cloud
        points, colors, total_count = build_point_cloud(disparity, left_rect, q_mat, args.max_points)
        display_points = to_display_coords(points)
        metrics["point_count_rendered"] = len(points)
        metrics["point_count_total"] = total_count
        if gt_dist_cm is not None:
            metrics["gt_anchor_dist_cm"] = round(gt_dist_cm, 1)
        if angle_metrics:
            metrics["angle_mae_deg"] = angle_metrics["angle_mae_deg"]
            metrics["worst_joint"] = angle_metrics["worst_joint"]
            metrics["worst_joint_err_deg"] = angle_metrics["worst_joint_err_deg"]

        # Encode images
        frame_img_b64 = frame_to_b64_jpeg(left_rect, width=800)

        payload_frames.append({
            "tag": tag,
            "label": FRAME_LABELS.get(tag, tag),
            "frame_idx": vid_idx,
            "npz_idx": npz_idx,
            "points": display_points.tolist(),
            "colors": colors.tolist(),
            "skeleton": skeleton_payload(kp_3d),
            "gt_skeleton": gt_payload,
            "angle_metrics": angle_metrics,
            "frame_image": frame_img_b64,
            "meta": {
                **metrics,
                "frame_idx": vid_idx,
                "npz_idx": npz_idx,
                "timestamp_delta_ms": round(delta_ms, 1),
                "sgbm": {
                    "min_disparity": SGBM_MIN_DISPARITY,
                    "num_disparities": SGBM_NUM_DISPARITIES,
                    "block_size": SGBM_BLOCK_SIZE,
                    "lookup_window": LOOKUP_WINDOW,
                },
            },
        })

    cap_l.release()
    cap_r.release()

    # Write outputs
    output_html = Path(args.output_html)
    output_json = Path(args.output_json)
    plotly_tag = get_plotly_tag()
    output_html.write_text(make_html(payload_frames, plotly_tag=plotly_tag), encoding="utf-8")
    # JSON without embedded images (too large)
    json_payload = [{k: v for k, v in f.items() if k not in ("points", "colors", "frame_image")}
                    for f in payload_frames]
    output_json.write_text(json.dumps({"frames": json_payload}, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_html": str(output_html),
        "output_json": str(output_json),
        "frame_count": len(payload_frames),
        "html_size_kb": round(output_html.stat().st_size / 1024),
    }, indent=2))


if __name__ == "__main__":
    main()
