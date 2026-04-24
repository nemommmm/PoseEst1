#!/opt/anaconda3/envs/pose/bin/python
"""Interactive 3-skeleton point-cloud viewer for manual agreement diagnosis.

Overlays SKT (stereo triangulation), AFH (hybrid), and Xsens GT on the same
SGBM point cloud.  Frame selection is driven by lowest-SKT fair-MAE sliding
windows so the viewer opens on the frames where the camera pipeline performs best.

Point cloud is coloured with real camera RGB (per supervisor request); an
optional height-based palette is available via --color-mode height.

Outputs
-------
- <output_dir>/viewer_<tag>.html  — self-contained Plotly HTML
- <output_dir>/metrics_<tag>.json — per-frame quantitative summary
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
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from utils_mvnx import MvnxParser  # noqa: E402
from pose_angle_utils import compute_semantic_joint_angles, build_fair_gt_interpolators, SEMANTIC_ANGLE_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "2025_Ergonomics_Data"
PARAM_PATH = SHARED_DIR / "camera_params.npz"
MVNX_PATH = Path.home() / "Desktop" / "MVE386 Project Course" / "Xsens_ground_truth" / "Aitor-001.mvnx"
GT_JSON_PATH = (
    PROJECT_ROOT / "01_stereo_triangulation" / "results" / "skeleton_comparison_dirA.json"
)
FAIR_GT_NPZ = SHARED_DIR / "fair_gt_angles.npz"
DEFAULT_SKT_NPZ = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
DEFAULT_AFH_NPZ = PROJECT_ROOT / "04_hybrid_afh1" / "results" / "hybrid_skeleton_afh1_v1.npz"
DEFAULT_OUTPUT_DIR = METHOD_DIR / "results" / "manual_agreement"

# ---------------------------------------------------------------------------
# SGBM defaults (env-overridable)
# ---------------------------------------------------------------------------

SGBM_MIN_DISP = int(os.environ.get("POSE_SGBM_MIN_DISPARITY", "100"))
SGBM_NUM_DISP = int(os.environ.get("POSE_SGBM_NUM_DISPARITIES", "256"))
SGBM_BLOCK = int(os.environ.get("POSE_SGBM_BLOCK_SIZE", "9"))

# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

XSENS_SEGMENTS = [
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
]
XSENS_LINKS = [
    ("Pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"), ("T8", "Neck"), ("Neck", "Head"),
    ("T8", "RightShoulder"), ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("T8", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Pelvis", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"), ("RightLowerLeg", "RightFoot"),
    ("Pelvis", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"), ("LeftLowerLeg", "LeftFoot"),
]

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def to_display(pts: np.ndarray) -> np.ndarray:
    """OpenCV camera coords → viewer frame: X-right, Y-depth, Z-up."""
    p = np.asarray(pts, dtype=np.float32)
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def finite_joints(pose: np.ndarray) -> np.ndarray:
    return pose[np.isfinite(pose).all(axis=1)]

# ---------------------------------------------------------------------------
# Video / stereo synchronization metadata
# ---------------------------------------------------------------------------

def parse_stereo_meta(txt_path: Path) -> List[Dict[str, float | int]]:
    """Parse one stereo metadata txt file using the same convention as StereoDataLoader."""
    rows: List[Dict[str, float | int]] = []
    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                frame_id = int(parts[0])
                timestamp = float(f"{parts[1]}.{parts[2]}")
            except ValueError:
                continue
            rows.append({"id": frame_id, "ts": timestamp})
    return rows


def build_synced_pairs(left_txt: Path, right_txt: Path) -> List[Dict[str, float | int]]:
    """Replicate StereoDataLoader hardware-ID sync without decoding the videos."""
    left_rows = parse_stereo_meta(left_txt)
    right_rows = parse_stereo_meta(right_txt)
    synced: List[Dict[str, float | int]] = []
    ptr_l = 0
    ptr_r = 0
    while ptr_l < len(left_rows) and ptr_r < len(right_rows):
        meta_l = left_rows[ptr_l]
        meta_r = right_rows[ptr_r]
        id_l = int(meta_l["id"])
        id_r = int(meta_r["id"])
        if id_l == id_r:
            synced.append(
                {
                    "left_idx": ptr_l,
                    "right_idx": ptr_r,
                    "frame_id": id_l,
                    "ts": float(meta_l["ts"]),
                }
            )
            ptr_l += 1
            ptr_r += 1
        elif id_l < id_r:
            ptr_l += 1
        else:
            ptr_r += 1
    return synced


def npz_to_sync(npz_idx: int, sync_ts: np.ndarray, npz_ts: np.ndarray) -> int:
    """Find the synchronized stereo pair whose timestamp is closest to one NPZ frame."""
    return int(np.argmin(np.abs(sync_ts - npz_ts[npz_idx])))

# ---------------------------------------------------------------------------
# Xsens GT loading & alignment
# ---------------------------------------------------------------------------

def load_gt(mvnx_path: Path) -> Optional[Dict]:
    if not mvnx_path.exists():
        return None
    parser = MvnxParser(str(mvnx_path))
    parser.parse()
    ts = parser.timestamps.copy()
    ts, idx = np.unique(ts, return_index=True)
    ts -= ts[0]
    interps: Dict = {}
    for seg in XSENS_SEGMENTS:
        data = parser.get_segment_data(seg)
        if data is None:
            continue
        interps[seg] = interp1d(ts, data[idx], axis=0, kind="linear",
                                bounds_error=False, fill_value=np.nan)
    return {"interps": interps, "ts_range": (float(ts[0]), float(ts[-1]))}


def load_alignment(json_path: Path) -> Optional[Dict]:
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as fh:
        d = json.load(fh)
    R = np.asarray(d["rotation_matrix"], dtype=np.float64)
    t = np.asarray(d["translation_cm"], dtype=np.float64)
    offset_s = float(d.get("offset_seconds", 17.25))
    return {"R": R, "t": t, "offset_s": offset_s, "R_inv": R.T, "t_inv": -R.T @ t}


def gt_pose_in_cam(subject_t: float, gt: Dict, align: Dict) -> Optional[Dict]:
    """Return Xsens segments in camera coords and display coords for one timestamp."""
    gt_t = subject_t - align["offset_s"]
    if gt_t < 0:
        return None
    R_inv, t_inv = align["R_inv"], align["t_inv"]
    cam: Dict[str, np.ndarray] = {}
    for seg, fn in gt["interps"].items():
        p = fn(gt_t)
        if np.isfinite(p).all():
            cam[seg] = R_inv @ p + t_inv
    if not cam:
        return None
    disp = {seg: to_display(p[np.newaxis])[0].tolist() for seg, p in cam.items()}
    return {"_cam": cam, "segment_points": disp, "links": XSENS_LINKS}

# ---------------------------------------------------------------------------
# Camera / SGBM / point cloud
# ---------------------------------------------------------------------------

def build_sgbm() -> cv2.StereoSGBM:
    bs = SGBM_BLOCK
    return cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP, numDisparities=SGBM_NUM_DISP, blockSize=bs,
        P1=8 * 3 * bs**2, P2=32 * 3 * bs**2,
        disp12MaxDiff=1, uniquenessRatio=5,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def setup_rectification(param_path: Path, img_hw: Tuple[int, int]):
    cal = np.load(param_path)
    h, w = img_hw
    r1, r2, p1, p2, q_mat, _, _ = cv2.stereoRectify(
        cal["mtx_l"], cal["dist_l"], cal["mtx_r"], cal["dist_r"],
        (w, h), cal["R"], cal["T"], alpha=0,
    )
    ml1 = cv2.initUndistortRectifyMap(cal["mtx_l"], cal["dist_l"], r1, p1, (w, h), cv2.CV_32FC1)
    mr1 = cv2.initUndistortRectifyMap(cal["mtx_r"], cal["dist_r"], r2, p2, (w, h), cv2.CV_32FC1)
    return *ml1, *mr1, q_mat, p1


def read_stereo_pair(
    left_idx: int,
    right_idx: int,
    cap_l: cv2.VideoCapture,
    cap_r: cv2.VideoCapture,
):
    """Read one hardware-synchronized stereo pair by original per-stream frame indices."""
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, left_idx)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, right_idx)
    ok_l, fl = cap_l.read()
    ok_r, fr = cap_r.read()
    if not ok_l or not ok_r:
        raise IndexError(f"Cannot read synchronized stereo pair L={left_idx}, R={right_idx}")
    return cv2.rotate(fl, cv2.ROTATE_180), cv2.rotate(fr, cv2.ROTATE_180)


def compute_disparity(fl, fr, m1l, m2l, m1r, m2r, sgbm):
    lr = cv2.remap(fl, m1l, m2l, cv2.INTER_LINEAR)
    rr = cv2.remap(fr, m1r, m2r, cv2.INTER_LINEAR)
    d = sgbm.compute(cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY),
                     cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    d[d < 0] = np.nan
    return lr, d


def build_cloud(disparity: np.ndarray, color_bgr: np.ndarray, q_mat: np.ndarray,
                color_mode: str = "rgb"):
    """Reproject to 3D; return (points, colors_uint8, pixel_coords_xy, total_count).

    color_mode='rgb'    use real camera texture (recommended, per supervisor request)
    color_mode='height' colour by Y-axis height for depth debugging
    """
    disp_fill = np.nan_to_num(disparity, nan=0.0).astype(np.float32)
    pts3 = cv2.reprojectImageTo3D(disp_fill, q_mat)
    mask = (np.isfinite(disparity) & (disparity >= SGBM_MIN_DISP)
            & np.isfinite(pts3).all(axis=2))
    z = pts3[:, :, 2]
    mask &= (z > 30.0) & (z < 1200.0)
    gy, gx = np.indices(disparity.shape)
    pts = pts3[mask].astype(np.float32)
    pix = np.column_stack([gx[mask], gy[mask]]).astype(np.int32)
    total = int(len(pts))
    if total == 0:
        return pts, np.zeros((0, 3), dtype=np.uint8), pix, total

    if color_mode == "rgb":
        cols = color_bgr[mask][:, ::-1].astype(np.uint8)  # BGR → RGB
    else:
        disp_pts = to_display(pts)
        heights = disp_pts[:, 2]
        vmin, vmax = np.nanpercentile(heights, 5), np.nanpercentile(heights, 95)
        norm = np.clip((heights - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        import matplotlib.pyplot as plt  # local import; only used in height mode
        rgba = plt.get_cmap("viridis")(norm)
        cols = (rgba[:, :3] * 255).astype(np.uint8)

    return pts, cols, pix, total


def subsample(pts: np.ndarray, cols: np.ndarray, max_pts: int, seed: int = 0):
    if len(pts) <= max_pts:
        return pts, cols
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=max_pts, replace=False)
    return pts[idx], cols[idx]


def crop_context_cloud(
    pts: np.ndarray,
    cols: np.ndarray,
    poses: List[np.ndarray],
    margin_xyz: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a loose near-person environment cloud around camera-method skeletons."""
    all_finite = [finite_joints(pose) for pose in poses if len(finite_joints(pose)) > 0]
    if not all_finite:
        return pts, cols
    joint_union = np.concatenate(all_finite, axis=0)
    mx, my, mz = margin_xyz
    lo = np.nanmin(joint_union, axis=0) - [mx, my, mz]
    hi = np.nanmax(joint_union, axis=0) + [mx, my, mz]
    sel = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[sel], cols[sel]


def display_bounds_payload(
    point_sets_cam: List[np.ndarray],
    margin_cm: float = 30.0,
) -> Optional[Dict[str, List[float]]]:
    """Build display-coordinate bounds for front-end focus framing."""
    finite_sets = []
    for pts in point_sets_cam:
        arr = np.asarray(pts, dtype=np.float32)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 3)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr) > 0:
            finite_sets.append(arr)
    if not finite_sets:
        return None
    disp = to_display(np.concatenate(finite_sets, axis=0))
    lo = np.nanmin(disp, axis=0) - margin_cm
    hi = np.nanmax(disp, axis=0) + margin_cm
    return {
        "min": [round(float(v), 3) for v in lo],
        "max": [round(float(v), 3) for v in hi],
    }

# ---------------------------------------------------------------------------
# Person cloud crop
# ---------------------------------------------------------------------------

def project_to_rect(pose: np.ndarray, p1_mat: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    out = np.full((len(pose), 2), np.nan, dtype=np.float32)
    valid = np.isfinite(pose).all(axis=1)
    if not np.any(valid):
        return out
    pts_h = np.hstack([pose[valid].astype(np.float32), np.ones((valid.sum(), 1), np.float32)])
    uvw = pts_h @ p1_mat.T
    ok = np.abs(uvw[:, 2]) > 1e-6
    uv = np.full((len(uvw), 2), np.nan, np.float32)
    uv[ok, 0] = uvw[ok, 0] / uvw[ok, 2]
    uv[ok, 1] = uvw[ok, 1] / uvw[ok, 2]
    in_bounds = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    tmp = np.full((len(uv), 2), np.nan, np.float32)
    tmp[in_bounds] = uv[in_bounds]
    out[valid] = tmp
    return out


def build_person_mask(hw: Tuple[int, int], proj_list: List[np.ndarray]) -> np.ndarray:
    """Build a tighter 2D person mask from camera-skeleton projections only."""
    h, w = hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for proj in proj_list:
        valid = np.isfinite(proj).all(axis=1)
        if not np.any(valid):
            continue
        pts = np.round(proj[valid]).astype(np.int32)
        y_span = int(np.max(pts[:, 1]) - np.min(pts[:, 1])) if len(pts) > 1 else 160
        lw = int(np.clip(round(y_span * 0.055), 9, 22))
        jr = int(np.clip(round(y_span * 0.032), 6, 14))
        torso_idx = [i for i in (5, 6, 11, 12) if valid[i]]
        if len(torso_idx) >= 3:
            tp = np.round(proj[torso_idx]).astype(np.int32)
            cv2.fillConvexPoly(mask, tp, 255)
        for a, b in COCO_EDGES:
            if valid[a] and valid[b]:
                cv2.line(mask, tuple(np.round(proj[a]).astype(int)),
                         tuple(np.round(proj[b]).astype(int)), 255, lw)
        for pt in pts:
            cv2.circle(mask, tuple(pt), jr, 255, -1)
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kern_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_close)
    mask = cv2.dilate(mask, kern_dil, iterations=1)
    return mask


def crop_person_cloud(pts: np.ndarray, cols: np.ndarray, pixels: np.ndarray,
                      poses: List[np.ndarray], mask_2d: Optional[np.ndarray],
                      margin_xyz: Tuple[float, float, float],
                      disable_mask: bool = False,
                      depth_margin_cm: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Crop cloud to person region using 3D bbox union + 2D mask.

    A coarse global depth slice is disabled by default because it tends to keep
    same-depth background strips while discarding valid limbs that move forward/backward.
    """
    all_finite = [finite_joints(p) for p in poses if len(finite_joints(p)) > 0]
    if not all_finite:
        return pts[:0], cols[:0]
    joint_union = np.concatenate(all_finite, axis=0)
    mx, my, mz = margin_xyz
    lo = np.nanmin(joint_union, axis=0) - [mx, my, mz]
    hi = np.nanmax(joint_union, axis=0) + [mx, my, mz]
    sel = np.all((pts >= lo) & (pts <= hi), axis=1)
    if not disable_mask and mask_2d is not None and len(pixels) > 0:
        sel &= mask_2d[pixels[:, 1], pixels[:, 0]] > 0
    # Depth consistency: keep only points near person's median depth
    if depth_margin_cm > 0:
        median_depth = float(np.nanmedian(joint_union[:, 2]))  # Z = forward depth in camera coords
        sel &= np.abs(pts[:, 2] - median_depth) <= depth_margin_cm
    return pts[sel], cols[sel]

# ---------------------------------------------------------------------------
# Nearest-neighbour skeleton-to-cloud distance
# ---------------------------------------------------------------------------

def nn_distance(pose_cam: np.ndarray, cloud_cam: np.ndarray) -> Optional[float]:
    joints = finite_joints(pose_cam)
    if len(joints) == 0 or len(cloud_cam) == 0:
        return None
    d, _ = cKDTree(cloud_cam).query(joints, k=1)
    return float(np.mean(d))

# ---------------------------------------------------------------------------
# Per-frame fair MAE computation
# ---------------------------------------------------------------------------

def compute_frame_mae(pose: np.ndarray, gt_t: float, fair_gt_interps: Dict) -> Optional[float]:
    if gt_t < 0 or not fair_gt_interps:
        return None
    est = compute_semantic_joint_angles(pose)
    errors = []
    for name in SEMANTIC_ANGLE_NAMES:
        ev = est.get(name, np.nan)
        fn = fair_gt_interps.get(name)
        if fn is None:
            continue
        gv = float(fn(gt_t))
        if np.isfinite(ev) and np.isfinite(gv):
            errors.append(abs(ev - gv))
    return float(np.mean(errors)) if errors else None


def precompute_skt_maes(npz_kp: np.ndarray, npz_ts: np.ndarray,
                        subject_start: float, offset_s: float,
                        fair_gt_interps: Dict) -> np.ndarray:
    maes = np.full(len(npz_kp), np.nan)
    for i, (pose, ts) in enumerate(zip(npz_kp, npz_ts)):
        subject_t = ts - subject_start
        mae = compute_frame_mae(pose, subject_t - offset_s, fair_gt_interps)
        if mae is not None:
            maes[i] = mae
    return maes

# ---------------------------------------------------------------------------
# Window-based frame selection
# ---------------------------------------------------------------------------

def find_low_mae_windows(maes: np.ndarray, npz_ts: np.ndarray,
                         window_sec: float, top_k: int) -> List[Tuple[float, int, int]]:
    """Return top-K non-overlapping [start, end) NPZ-frame windows with lowest mean MAE."""
    duration = npz_ts[-1] - npz_ts[0]
    fps_npz = len(npz_ts) / max(duration, 1.0)
    w = max(5, int(round(window_sec * fps_npz)))
    step = max(1, w // 3)
    candidates: List[Tuple[float, int, int]] = []
    for s in range(0, len(maes) - w + 1, step):
        e = s + w
        chunk = maes[s:e]
        valid = np.isfinite(chunk)
        if valid.sum() < w * 0.7:
            continue
        m = float(np.nanmean(chunk))
        candidates.append((m, s, e))
    candidates.sort(key=lambda x: x[0])
    selected: List[Tuple[float, int, int]] = []
    used: set[int] = set()
    for entry in candidates:
        mae_w, s, e = entry
        if any(f in used for f in range(s, e)):
            continue
        selected.append(entry)
        used.update(range(s, e))
        if len(selected) >= top_k:
            break
    return selected


def window_frames(window: Tuple[float, int, int], n_frames: int) -> List[int]:
    """Pick n_frames evenly-spaced NPZ indices within window [start, end)."""
    _, s, e = window
    length = e - s
    if n_frames >= length:
        return list(range(s, e))
    return [s + int(round(i * (length - 1) / (n_frames - 1))) for i in range(n_frames)]

# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def frame_to_b64(frame_bgr: np.ndarray, width: int = 720) -> str:
    h, w = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

# ---------------------------------------------------------------------------
# Plotly CDN helper (mirrors script 10)
# ---------------------------------------------------------------------------

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CACHE = Path.home() / ".cache" / "plotly-2.35.2.min.js"


def get_plotly_tag() -> str:
    if PLOTLY_CACHE.exists():
        return f"<script>{PLOTLY_CACHE.read_text(encoding='utf-8')}</script>"
    try:
        import urllib.request
        print(f"Downloading Plotly from {PLOTLY_CDN} ...")
        PLOTLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PLOTLY_CDN, PLOTLY_CACHE)
        return f"<script>{PLOTLY_CACHE.read_text(encoding='utf-8')}</script>"
    except Exception as exc:
        print(f"  WARNING: Plotly download failed ({exc}). Using CDN link.")
        return f'<script src="{PLOTLY_CDN}"></script>'

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def make_html(windows_payload: List[Dict], plotly_tag: str) -> str:
    data_json = json.dumps(windows_payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Manual Agreement Viewer — SKT / AFH / Xsens GT</title>
  {plotly_tag}
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; background: #f0f2f5; color: #1a1a2e; }}
    .wrap {{ max-width: 1520px; margin: 0 auto; padding: 18px; }}
    h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
    .sub {{ color: #555; font-size: 0.86rem; margin-bottom: 12px; }}
    .row {{ display: grid; grid-template-columns: 310px 1fr; gap: 14px; }}
    .card {{ background: #fff; border-radius: 12px;
             box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 14px; }}
    #plot {{ width: 100%; height: 760px; background:#0d1117; border-radius:8px; }}
    #errBox {{ display:none; padding:16px; color:#b71c1c; font-size:0.88rem;
               background:#fff0f0; border-radius:8px; border:1px solid #f5c6cb; margin-bottom:10px; }}
    select {{ width:100%; padding:6px 8px; border:1px solid #ccc; border-radius:6px;
              font-size:13px; margin-bottom:8px; cursor:pointer; }}
    .btn-row {{ display:flex; gap:5px; margin-bottom:10px; }}
    .btn-row button {{ flex:1; padding:6px 4px; border:1px solid #bbb; border-radius:6px;
                       font-size:12px; background:#f5f5f5; cursor:pointer; }}
    .btn-row button.active {{ background:#005BAA; color:#fff; border-color:#005BAA; }}
    .skel-toggles {{ display:flex; flex-direction:column; gap:4px; margin-bottom:8px; }}
    .skel-toggle {{ display:flex; align-items:center; gap:8px; font-size:13px;
                    padding:6px 8px; border-radius:6px; cursor:pointer;
                    border:1px solid #ddd; user-select:none;
                    transition: background 0.15s; }}
    .skel-toggle:hover {{ background:#f0f4ff; }}
    .skel-toggle input[type=checkbox] {{ width:16px; height:16px; cursor:pointer;
                                         flex-shrink:0; accent-color:#005BAA; }}
    .swatch-line {{ width:24px; height:5px; border-radius:3px; flex-shrink:0; }}
    .metric-row {{ display:flex; justify-content:space-between; padding:4px 0;
                   border-bottom:1px solid #f0f0f0; font-size:0.83rem; }}
    .metric-row:last-child {{ border-bottom:none; }}
    .mkey {{ color:#555; }}
    .mval {{ font-weight:600; }}
    .good {{ color:#2a8a4a; }} .warn {{ color:#c0622a; }} .bad {{ color:#b71c1c; }}
    .sec-title {{ font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;
                  color:#888; margin:10px 0 5px; font-weight:600; }}
    #frameImg {{ width:100%; border-radius:6px; margin-top:8px; }}
    .cloud-legend {{ font-size:0.77rem; color:#555; margin-top:4px; line-height:1.6; }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%;
            vertical-align:middle; margin-right:4px; }}
    #loadMsg {{ padding:16px; text-align:center; color:#666; font-size:0.88rem; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Manual Agreement Viewer: SKT / AFH / Xsens GT on SGBM Point Cloud</h1>
  <p class="sub">
    Frames from lowest-SKT-MAE windows · Left-drag to orbit · Right-drag to pan · Scroll to zoom
  </p>
  <div id="errBox"></div>
  <div class="row">
    <!-- ── Left sidebar ── -->
    <div class="card">
      <div class="sec-title">Window (by SKT MAE)</div>
      <select id="winSelect"></select>
      <div class="sec-title">Frame</div>
      <select id="frmSelect"></select>
      <div class="btn-row frame-controls">
        <button onclick="stepFrame(-1)">Prev</button>
        <button id="btnPlay" onclick="togglePlay()">Play</button>
        <button onclick="stepFrame(1)">Next</button>
      </div>

      <div class="sec-title">Point cloud mode</div>
      <div class="btn-row">
        <button id="btnFocus" class="active" onclick="setMode('focus')">Focus</button>
        <button id="btnCtx" onclick="setMode('ctx')">Full&nbsp;room</button>
        <button id="btnRGB"  onclick="setMode('rgb')">Person&nbsp;RGB</button>
      </div>
      <div class="cloud-legend" id="cloudLegend"></div>

      <div class="sec-title">Skeletons</div>
      <div class="skel-toggles">
        <label class="skel-toggle">
          <input type="checkbox" id="chkSKT" checked onchange="toggleSkel('SKT',this.checked)">
          <span class="swatch-line" style="background:#ff7a18"></span>
          <span style="color:#ff7a18;font-weight:600">SKT</span> stereo triangulation
        </label>
        <label class="skel-toggle">
          <input type="checkbox" id="chkAFH" checked onchange="toggleSkel('AFH',this.checked)">
          <span class="swatch-line" style="background:#2196F3"></span>
          <span style="color:#2196F3;font-weight:600">AFH</span> hybrid
        </label>
        <label class="skel-toggle">
          <input type="checkbox" id="chkGT" checked onchange="toggleSkel('GT',this.checked)">
          <span class="swatch-line" style="background:#43a047"></span>
          <span style="color:#43a047;font-weight:600">Xsens GT</span> ground truth
        </label>
      </div>

      <div class="sec-title">Camera image</div>
      <img id="frameImg" alt="Camera frame" style="display:none"/>

      <div class="sec-title">Metrics</div>
      <div id="metricPanel"></div>

      <div class="sec-title" style="margin-top:10px">Note</div>
      <div style="font-size:0.76rem;color:#666;line-height:1.55">
        NN distance is a rough guide (depends on SGBM quality).<br>
        Use your eyes: which skeleton fits the <b>cyan person region</b>?
      </div>
    </div>

    <!-- ── 3D plot ── -->
    <div class="card" style="padding:0;overflow:hidden">
      <div id="loadMsg">Rendering 3D scene…</div>
      <div id="plot"></div>
    </div>
  </div>
</div>
<script>
// ── WebGL check ──────────────────────────────────────────────────────────────
(function() {{
  try {{
    var c = document.createElement('canvas');
    var gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    if (!gl) throw new Error('no WebGL context');
  }} catch(e) {{
    document.getElementById('errBox').style.display = 'block';
    document.getElementById('errBox').innerHTML =
      '<b>WebGL not available.</b> Open in Chrome/Safari with hardware acceleration enabled.<br>'
      + 'Chrome: Settings → System → "Use hardware acceleration when available".';
    document.getElementById('loadMsg').style.display = 'none';
  }}
}})();

// ── State ────────────────────────────────────────────────────────────────────
const wins = {data_json};
let curWin = 0, curFrm = 0, curMode = 'focus';
let showSKT = true, showAFH = true, showGT = true;
let plotInited = false;
let playTimer = null;
const PLAY_MS = 450;

// ── Selects ──────────────────────────────────────────────────────────────────
const winSel = document.getElementById('winSelect');
const frmSel = document.getElementById('frmSelect');

wins.forEach((w, wi) => {{
  const o = document.createElement('option'); o.value = wi;
  const s = (w.window_mae_deg != null && isFinite(w.window_mae_deg))
    ? w.window_mae_deg.toFixed(1) + '°' : 'n/a';
  o.textContent = `Window ${{wi+1}}  —  mean SKT MAE ${{s}}`;
  winSel.appendChild(o);
}});

function populateFrames(wi) {{
  frmSel.innerHTML = '';
  wins[wi].frames.forEach((f, fi) => {{
    const o = document.createElement('option'); o.value = fi;
    const t = f.subject_time_s != null ? f.subject_time_s.toFixed(1) + 's' : '?';
    const m = f.skt_mae_deg != null ? f.skt_mae_deg.toFixed(1) + '°' : 'n/a';
    o.textContent = `t = ${{t}}    SKT-MAE = ${{m}}`;
    frmSel.appendChild(o);
  }});
}}
populateFrames(0);

winSel.addEventListener('change', e => {{
  stopPlay();
  curWin = +e.target.value; curFrm = 0;
  frmSel.value = 0; populateFrames(curWin); render();
}});
frmSel.addEventListener('change', e => {{ curFrm = +e.target.value; render(); }});

function stepFrame(delta) {{
  const n = wins[curWin].frames.length;
  if (!n) return;
  curFrm = (curFrm + delta + n) % n;
  frmSel.value = curFrm;
  render();
}}

function stopPlay() {{
  if (playTimer !== null) {{
    clearInterval(playTimer);
    playTimer = null;
  }}
  const btn = document.getElementById('btnPlay');
  if (btn) btn.textContent = 'Play';
}}

function togglePlay() {{
  if (playTimer !== null) {{
    stopPlay();
    return;
  }}
  document.getElementById('btnPlay').textContent = 'Pause';
  playTimer = setInterval(() => stepFrame(1), PLAY_MS);
}}

// ── Cloud mode ────────────────────────────────────────────────────────────────
function setMode(m) {{
  curMode = m;
  document.getElementById('btnFocus').className = m === 'focus' ? 'active' : '';
  document.getElementById('btnCtx').className = m === 'ctx' ? 'active' : '';
  document.getElementById('btnRGB').className = m === 'rgb' ? 'active' : '';
  render();
}}

// ── Skeleton toggles ──────────────────────────────────────────────────────────
function toggleSkel(name, checked) {{
  if (name === 'SKT') showSKT = checked;
  if (name === 'AFH') showAFH = checked;
  if (name === 'GT')  showGT  = checked;
  render();
}}

// ── Helpers ───────────────────────────────────────────────────────────────────
function mClass(k, v) {{
  if (v == null || !isFinite(v)) return '';
  if (k === 'skt_mae_deg') return v <= 15 ? 'good' : v <= 25 ? 'warn' : 'bad';
  if (k.endsWith('_nn_cm')) return v <= 15 ? 'good' : v <= 30 ? 'warn' : 'bad';
  return '';
}}

function ok(p) {{
  return p != null && p[0] != null && isFinite(p[0]) && isFinite(p[1]) && isFinite(p[2]);
}}

function rgb(c) {{
  return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
}}

function addCloudTrace(traces, cloud, opts) {{
  if (!cloud || !cloud.pts || !cloud.pts.length) return;
  const marker = {{
    size: opts.size,
    opacity: opts.opacity,
  }};
  marker.color = opts.rgb ? cloud.cols.map(rgb) : opts.color;
  traces.push({{ type:'scatter3d', mode:'markers',
    x:cloud.pts.map(p=>p[0]), y:cloud.pts.map(p=>p[1]), z:cloud.pts.map(p=>p[2]),
    marker, name:opts.name, hoverinfo:'skip', showlegend:opts.showlegend !== false }});
}}

function addBoundsBox(traces, bounds) {{
  if (!bounds || !bounds.min || !bounds.max) return;
  const mn = bounds.min, mx = bounds.max;
  const c = [
    [mn[0],mn[1],mn[2]], [mx[0],mn[1],mn[2]], [mx[0],mx[1],mn[2]], [mn[0],mx[1],mn[2]],
    [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]], [mx[0],mx[1],mx[2]], [mn[0],mx[1],mx[2]],
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const x = [], y = [], z = [];
  edges.forEach(([a,b]) => {{
    x.push(c[a][0], c[b][0], null);
    y.push(c[a][1], c[b][1], null);
    z.push(c[a][2], c[b][2], null);
  }});
  traces.push({{ type:'scatter3d', mode:'lines',
    x, y, z, line:{{color:'#00e5ff', width:5}},
    name:'Person bounds', hoverinfo:'skip', showlegend:true }});
}}

function applyFocusRanges(scene, bounds, pad) {{
  if (!bounds || !bounds.min || !bounds.max) return;
  scene.xaxis.range = [bounds.min[0] - pad, bounds.max[0] + pad];
  scene.yaxis.range = [bounds.min[1] - pad, bounds.max[1] + pad];
  scene.zaxis.range = [bounds.min[2] - pad, bounds.max[2] + pad];
}}

// ── Render ────────────────────────────────────────────────────────────────────
function render() {{
  try {{
    const frm = wins[curWin].frames[curFrm];

    // Camera image
    const img = document.getElementById('frameImg');
    if (frm.frame_image) {{ img.src = frm.frame_image; img.style.display = 'block'; }}
    else img.style.display = 'none';

    // Metrics
    const mdefs = [
      ['subject_time_s','Subject time',' s'],
      ['skt_mae_deg','SKT fair MAE','°'],
      ['skt_nn_cm','SKT→cloud NN',' cm'],
      ['afh_nn_cm','AFH→cloud NN',' cm'],
      ['gt_nn_cm','GT→cloud NN',' cm'],
      ['person_cloud_pts','Person cloud pts',''],
    ];
    document.getElementById('metricPanel').innerHTML = mdefs.map(([k,label,unit]) => {{
      const v = frm.metrics[k];
      const cls = mClass(k,v);
      const disp = (v==null||(typeof v==='number'&&!isFinite(v))) ? 'N/A'
        : (typeof v==='number' ? v.toFixed(k==='person_cloud_pts'?0:1) : v) + unit;
      return `<div class="metric-row"><span class="mkey">${{label}}</span><span class="mval ${{cls}}">${{disp}}</span></div>`;
    }}).join('');

    // ── Build traces ──────────────────────────────────────────────────────────
    const traces = [];

    if (curMode === 'focus') {{
      const cc = frm.context_cloud || frm.full_cloud;
      addCloudTrace(traces, cc, {{
        size:1.1, color:'#6b7788', opacity:0.18, name:'Near-person context'
      }});
      addCloudTrace(traces, frm.person_cloud, {{
        size:6.0, color:'#00e5ff', opacity:0.22, name:'Person halo', showlegend:false
      }});
      addCloudTrace(traces, frm.person_cloud, {{
        size:3.2, opacity:0.97, rgb:true, name:'Person (camera RGB)'
      }});
      addBoundsBox(traces, frm.focus_bounds);
      document.getElementById('cloudLegend').innerHTML =
        '<span class="dot" style="background:#6b7788;opacity:0.7"></span>Near-person context<br>'
        + '<span class="dot" style="background:#00e5ff"></span><b>Person box + halo</b><br>'
        + '<span class="dot" style="background:#ddd"></span>Person points use camera RGB';
    }} else if (curMode === 'ctx') {{
      addCloudTrace(traces, frm.full_cloud, {{
        size:0.75, color:'#4a5568', opacity:0.055, name:'Full room cloud'
      }});
      addCloudTrace(traces, frm.person_cloud, {{
        size:4.4, color:'#00e5ff', opacity:0.90, name:'Person (cyan highlight)'
      }});
      addBoundsBox(traces, frm.focus_bounds);
      document.getElementById('cloudLegend').innerHTML =
        '<span class="dot" style="background:#4a5568;opacity:0.5"></span>Full room cloud (very dim)<br>'
        + '<span class="dot" style="background:#00e5ff"></span><b>Person region + box</b>';
    }} else {{
      addCloudTrace(traces, frm.person_cloud, {{
        size:3.2, opacity:0.97, rgb:true, name:'Person (camera RGB)'
      }});
      document.getElementById('cloudLegend').innerHTML =
        'Person cloud with real camera RGB colours (camera-mask cropped)';
    }}

    // ── Skeletons ─────────────────────────────────────────────────────────────
    function addSkel(skel, color, name, jColor) {{
      if (!skel || !skel.points) return;
      const pts = skel.points;
      let first = true;
      skel.edges.forEach(([a,b]) => {{
        const p0=pts[a], p1=pts[b];
        if (!ok(p0)||!ok(p1)) return;
        traces.push({{ type:'scatter3d', mode:'lines',
          x:[p0[0],p1[0]], y:[p0[1],p1[1]], z:[p0[2],p1[2]],
          line:{{color, width:8}}, name:first?name:'', showlegend:first, hoverinfo:'skip' }});
        first=false;
      }});
      const vp = pts.filter(ok);
      if (vp.length) traces.push({{ type:'scatter3d', mode:'markers',
        x:vp.map(p=>p[0]), y:vp.map(p=>p[1]), z:vp.map(p=>p[2]),
        marker:{{size:6, color:jColor}}, name:name+' joints', hoverinfo:'skip' }});
    }}

    if (showSKT) addSkel(frm.skt_skel, '#ff7a18', 'SKT', '#ffb74d');
    if (showAFH) addSkel(frm.afh_skel, '#2196F3', 'AFH', '#90caf9');

    if (showGT) {{
      const gt = frm.gt_skel;
      if (gt && gt.segment_points) {{
        const sp = gt.segment_points;
        let first = true;
        gt.links.forEach(([a,b]) => {{
          const p0=sp[a], p1=sp[b];
          if (!ok(p0)||!ok(p1)) return;
          traces.push({{ type:'scatter3d', mode:'lines',
            x:[p0[0],p1[0]], y:[p0[1],p1[1]], z:[p0[2],p1[2]],
            line:{{color:'#43a047', width:7}}, name:first?'Xsens GT':'', showlegend:first, hoverinfo:'skip' }});
          first=false;
        }});
        const gp = Object.values(sp).filter(ok);
        if (gp.length) traces.push({{ type:'scatter3d', mode:'markers',
          x:gp.map(p=>p[0]), y:gp.map(p=>p[1]), z:gp.map(p=>p[2]),
          marker:{{size:4.5, color:'#a5d6a7'}}, name:'GT joints', hoverinfo:'skip' }});
      }}
    }}

    // ── Layout ────────────────────────────────────────────────────────────────
    const maeS = frm.skt_mae_deg!=null ? frm.skt_mae_deg.toFixed(1)+'°' : 'n/a';
    const tS   = frm.subject_time_s!=null ? frm.subject_time_s.toFixed(2)+'s' : '?';
    const modeLabel = curMode==='focus' ? 'Focus' : (curMode==='ctx' ? 'Full room' : 'Person RGB');
    const skelOn = [showSKT?'SKT':'',showAFH?'AFH':'',showGT?'GT':''].filter(Boolean).join('+');
    const title = `Win${{curWin+1}} · t=${{tS}} · SKT-MAE=${{maeS}} · ${{modeLabel}} · ${{skelOn||'(no skeleton)'}}`;

    const scene = {{
      xaxis:{{title:'X (cm, right)', backgroundcolor:'#0d1117', gridcolor:'#2a3a4a', showbackground:true}},
      yaxis:{{title:'Y (cm, depth)', backgroundcolor:'#0d1117', gridcolor:'#2a3a4a', showbackground:true}},
      zaxis:{{title:'Z (cm, up)',    backgroundcolor:'#0d1117', gridcolor:'#2a3a4a', showbackground:true}},
      bgcolor:'#0d1117',
      aspectmode:'manual',
      aspectratio:{{x:1, y:1.6, z:0.9}},
      // Front view (matches camera image): looking from in front of the scene along +Y depth axis
      camera:{{eye:{{x:0.05, y:-2.5, z:0.2}}, up:{{x:0,y:0,z:1}}}}
    }};
    if (curMode === 'focus') applyFocusRanges(scene, frm.focus_bounds, 80);
    if (curMode === 'rgb') applyFocusRanges(scene, frm.focus_bounds, 45);

    const layout = {{
      scene,
      paper_bgcolor:'#0d1117',
      font:{{color:'#bbb'}},
      legend:{{bgcolor:'rgba(20,28,40,0.85)', font:{{color:'#ddd'}}, x:0.01, y:0.99}},
      margin:{{l:0,r:0,b:0,t:36}},
      title:{{text:title, font:{{size:12, color:'#aaa'}}}},
      uirevision: `win${{curWin}}-frm${{curFrm}}-mode${{curMode}}`
    }};

    if (!plotInited) {{
      Plotly.newPlot('plot', traces, layout, {{responsive:true}});
      plotInited = true;
    }} else {{
      Plotly.react('plot', traces, layout, {{responsive:true}});
    }}
    document.getElementById('loadMsg').style.display = 'none';

  }} catch(err) {{
    console.error('render() error:', err);
    document.getElementById('errBox').style.display = 'block';
    document.getElementById('errBox').innerHTML =
      '<b>Render error:</b> ' + err.message
      + '<br><small>Open browser console (F12 → Console) for the full stack trace.</small>';
  }}
}}

render();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive 3-skeleton point-cloud manual agreement viewer.")
    p.add_argument("--skt-path", default=str(DEFAULT_SKT_NPZ))
    p.add_argument("--afh-path", default=str(DEFAULT_AFH_NPZ))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--tag", default="skt_afh_gt",
                   help="Short tag used in output filenames.")
    p.add_argument("--window-sec", type=float, default=1.0,
                   help="Sliding window size in seconds for low-MAE search.")
    p.add_argument("--window-count", type=int, default=3,
                   help="Number of non-overlapping low-MAE windows to render.")
    p.add_argument("--frames-per-window", type=int, default=5,
                   help="Number of evenly-spaced frames to render per window.")
    p.add_argument("--max-full-pts", type=int, default=30000,
                   help="Max points in full cloud for HTML rendering.")
    p.add_argument("--max-context-pts", type=int, default=12000,
                   help="Max near-person context points in focus mode.")
    p.add_argument("--max-person-pts", type=int, default=15000,
                   help="Max points in person-cropped cloud.")
    p.add_argument("--bbox-margin", type=float, default=30.0,
                   help="3D bbox margin around skeleton union (cm).")
    p.add_argument("--context-margin", type=float, default=140.0,
                   help="Loose 3D margin around camera-method skeletons for focus context (cm).")
    p.add_argument("--disable-mask", action="store_true",
                   help="Skip 2D skeleton mask; use only 3D bbox for person crop.")
    p.add_argument("--color-mode", choices=["rgb", "height"], default="rgb",
                   help="Point cloud colour: rgb (real camera texture) or height (debug).")
    p.add_argument("--frames", default="",
                   help="Comma-separated NPZ frame indices to render instead of auto-selection, "
                        "e.g. '500,1000,1500'. Overrides window selection.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load SKT + AFH ----
    print("Loading SKT NPZ...")
    skt_np = np.load(args.skt_path, allow_pickle=True)
    skt_kp: np.ndarray = skt_np["keypoints"]
    skt_ts: np.ndarray = skt_np["timestamps"]

    print("Loading AFH NPZ...")
    afh_np = np.load(args.afh_path, allow_pickle=True)
    afh_kp: np.ndarray = afh_np["keypoints"]

    if skt_kp.shape != afh_kp.shape:
        raise ValueError(f"SKT {skt_kp.shape} vs AFH {afh_kp.shape} shape mismatch.")

    # ---- Synchronized stereo metadata ----
    print("Building synchronized stereo-pair metadata...")
    sync_pairs = build_synced_pairs(DATA_DIR / "0_video_left.txt", DATA_DIR / "1_video_right.txt")
    if not sync_pairs:
        raise RuntimeError("No synchronized stereo pairs found in metadata.")
    sync_ts = np.array([float(row["ts"]) for row in sync_pairs], dtype=np.float64)
    subject_start = float(skt_ts[0])
    print(f"  Synced stereo pairs: {len(sync_pairs)}, NPZ frames: {len(skt_ts)}")

    # ---- Xsens GT + Kabsch ----
    print("Loading Xsens GT...")
    gt = load_gt(MVNX_PATH)
    align = load_alignment(GT_JSON_PATH)
    if gt is None:
        print("  WARNING: MVNX not found; GT overlay disabled.")
    if align is None:
        print("  WARNING: skeleton_comparison_dirA.json not found; GT overlay disabled.")
    offset_s = align["offset_s"] if align else 17.25
    print(f"  offset_s={offset_s}")

    # ---- Fair GT ----
    print("Loading fair GT interpolators...")
    fair_gt = build_fair_gt_interpolators(str(FAIR_GT_NPZ))
    if not fair_gt:
        print("  WARNING: fair_gt_angles.npz not found; frame selection falls back to uniform.")

    # ---- Camera setup ----
    cap_l = cv2.VideoCapture(str(DATA_DIR / "0_video_left.avi"))
    cap_r = cv2.VideoCapture(str(DATA_DIR / "1_video_right.avi"))
    ok, tmp = cap_l.read()
    if not ok:
        raise RuntimeError("Cannot read left video.")
    img_hw: Tuple[int, int] = tmp.shape[:2]
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    m1l, m2l, m1r, m2r, q_mat, p1_mat = setup_rectification(PARAM_PATH, img_hw)
    sgbm = build_sgbm()

    margin = (args.bbox_margin, args.bbox_margin, args.bbox_margin)

    # ---- Frame selection ----
    if args.frames.strip():
        npz_indices_manual = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
        groups: List[Tuple[float, List[int]]] = [(0.0, npz_indices_manual)]
        print(f"Manual frame override: {npz_indices_manual}")
    else:
        print("Computing SKT fair MAE per NPZ frame...")
        if fair_gt:
            maes = precompute_skt_maes(skt_kp, skt_ts, subject_start, offset_s, fair_gt)
            valid_count = int(np.isfinite(maes).sum())
            print(f"  Valid MAE frames: {valid_count}/{len(maes)}, "
                  f"mean={np.nanmean(maes):.1f}°, min={np.nanmin(maes):.1f}°")
        else:
            maes = np.full(len(skt_kp), np.nan)

        windows = find_low_mae_windows(maes, skt_ts, args.window_sec, args.window_count)
        if not windows:
            step = max(1, len(skt_kp) // (args.window_count * args.frames_per_window))
            uniform = list(range(0, len(skt_kp), step))[:args.window_count * args.frames_per_window]
            windows = [(float("nan"), 0, len(skt_kp))]
            groups = [(float("nan"), uniform)]
            print("  WARNING: no valid windows found; using uniform frame spacing.")
        else:
            groups = [(w[0], window_frames(w, args.frames_per_window)) for w in windows]
            for i, (mae_w, _, _) in enumerate(windows):
                print(f"  Window {i+1}: mean SKT MAE={mae_w:.1f}°, frames={groups[i][1]}")

    # ---- Render frames ----
    windows_payload: List[Dict] = []
    metrics_log: List[Dict] = []

    for win_i, (win_mae, npz_indices) in enumerate(groups):
        print(f"\n=== Window {win_i+1}/{len(groups)} (mean MAE={win_mae:.1f}°) ===")
        frame_payloads: List[Dict] = []

        for npz_idx in npz_indices:
            npz_idx = int(np.clip(npz_idx, 0, len(skt_kp) - 1))
            sync_idx = npz_to_sync(npz_idx, sync_ts, skt_ts)
            pair_meta = sync_pairs[sync_idx]
            left_idx = int(pair_meta["left_idx"])
            right_idx = int(pair_meta["right_idx"])
            subject_t = float(skt_ts[npz_idx] - subject_start)
            print(
                f"  NPZ {npz_idx} -> sync {sync_idx} "
                f"(L={left_idx}, R={right_idx}) -> subject_t={subject_t:.2f}s"
            )

            fl, fr = read_stereo_pair(left_idx, right_idx, cap_l, cap_r)
            lr, disp = compute_disparity(fl, fr, m1l, m2l, m1r, m2r, sgbm)

            pts_all, cols_all, pix_all, total = build_cloud(disp, lr, q_mat, args.color_mode)

            skt_pose = skt_kp[npz_idx].astype(np.float32)
            afh_pose = afh_kp[npz_idx].astype(np.float32)
            gt_info = gt_pose_in_cam(subject_t, gt, align) if gt and align else None

            # -- Person cloud crop: use camera-method skeletons only.
            #    GT remains an inspected hypothesis, not part of the crop prior.
            gt_cam_pose_list: List[np.ndarray] = []
            if gt_info:
                gt_cam_arr = np.array([gt_info["_cam"].get(s, np.full(3, np.nan))
                                       for s in XSENS_SEGMENTS], dtype=np.float32)
                gt_cam_pose_list.append(gt_cam_arr)

            poses_for_mask = [skt_pose, afh_pose]
            skt_proj = project_to_rect(skt_pose, p1_mat, img_hw)
            afh_proj = project_to_rect(afh_pose, p1_mat, img_hw)
            all_proj = [skt_proj, afh_proj]

            mask_2d = None if args.disable_mask else build_person_mask(img_hw, all_proj)
            pers_pts, pers_cols = crop_person_cloud(
                pts_all, cols_all, pix_all, poses_for_mask, mask_2d, margin, args.disable_mask
            )
            ctx_pts, ctx_cols = crop_context_cloud(
                pts_all,
                cols_all,
                poses_for_mask,
                (args.context_margin, args.context_margin, args.context_margin),
            )
            focus_bounds = display_bounds_payload(
                [pers_pts, skt_pose, afh_pose] + gt_cam_pose_list,
                margin_cm=25.0,
            )

            # subsample for rendering
            full_pts_r, full_cols_r = subsample(pts_all, cols_all, args.max_full_pts)
            ctx_pts_r, ctx_cols_r = subsample(ctx_pts, ctx_cols, args.max_context_pts, seed=2)
            pers_pts_r, pers_cols_r = subsample(pers_pts, pers_cols, args.max_person_pts, seed=1)

            # NN distances (computed on full person cloud, not subsampled)
            skt_nn = nn_distance(skt_pose, pers_pts)
            afh_nn = nn_distance(afh_pose, pers_pts)
            gt_nn: Optional[float] = None
            if gt_cam_pose_list:
                gt_nn = nn_distance(gt_cam_pose_list[0], pers_pts)

            # fair MAE
            skt_mae_val = compute_frame_mae(skt_pose, subject_t - offset_s, fair_gt)

            def display_skel(pose: np.ndarray) -> Dict:
                disp = to_display(pose)
                return {"points": disp.tolist(), "edges": COCO_EDGES}

            def cloud_payload(p: np.ndarray, c: np.ndarray) -> Dict:
                return {"pts": to_display(p).tolist(), "cols": c.tolist()}

            gt_payload = None
            if gt_info:
                gt_payload = {
                    "segment_points": gt_info["segment_points"],
                    "links": [[a, b] for a, b in gt_info["links"]],
                }

            metrics = {
                "subject_time_s": round(subject_t, 3),
                "skt_mae_deg": round(skt_mae_val, 2) if skt_mae_val is not None else None,
                "skt_nn_cm": round(skt_nn, 2) if skt_nn is not None else None,
                "afh_nn_cm": round(afh_nn, 2) if afh_nn is not None else None,
                "gt_nn_cm": round(gt_nn, 2) if gt_nn is not None else None,
                "person_cloud_pts": int(len(pers_pts)),
                "full_cloud_pts": int(total),
            }
            print(f"    MAE={metrics['skt_mae_deg']}° SKT-NN={metrics['skt_nn_cm']} "
                  f"AFH-NN={metrics['afh_nn_cm']} GT-NN={metrics['gt_nn_cm']} "
                  f"person_pts={metrics['person_cloud_pts']}")

            frame_payloads.append({
                "npz_idx": int(npz_idx),
                "sync_idx": int(sync_idx),
                "left_vid_idx": left_idx,
                "right_vid_idx": right_idx,
                "subject_time_s": round(subject_t, 3),
                "skt_mae_deg": metrics["skt_mae_deg"],
                "full_cloud": cloud_payload(full_pts_r, full_cols_r),
                "context_cloud": cloud_payload(ctx_pts_r, ctx_cols_r),
                "person_cloud": cloud_payload(pers_pts_r, pers_cols_r),
                "focus_bounds": focus_bounds,
                "skt_skel": display_skel(skt_pose),
                "afh_skel": display_skel(afh_pose),
                "gt_skel": gt_payload,
                "frame_image": frame_to_b64(lr),
                "metrics": metrics,
            })
            metrics_log.append({"window": win_i + 1, **metrics})

        windows_payload.append({
            "window_idx": win_i + 1,
            "window_mae_deg": float(win_mae) if np.isfinite(win_mae) else None,
            "frames": frame_payloads,
        })

    cap_l.release()
    cap_r.release()

    # ---- Write outputs ----
    html_path = out_dir / f"viewer_{args.tag}.html"
    json_path = out_dir / f"metrics_{args.tag}.json"
    plotly_tag = get_plotly_tag()
    html_path.write_text(make_html(windows_payload, plotly_tag), encoding="utf-8")
    metrics_only = [
        {
            "window_idx": w["window_idx"],
            "window_mae_deg": w["window_mae_deg"],
            "frames": [
                {k: v for k, v in f.items()
                 if k not in ("full_cloud", "context_cloud", "person_cloud", "skt_skel",
                              "afh_skel", "gt_skel", "frame_image")}
                for f in w["frames"]
            ],
        }
        for w in windows_payload
    ]
    json_path.write_text(json.dumps({"windows": metrics_only, "metrics_flat": metrics_log},
                                    indent=2, default=lambda x: None), encoding="utf-8")
    print(f"\n[saved] {html_path}")
    print(f"[saved] {json_path}")
    print(
        f"\nOpen {html_path.name} in a browser, select a window, "
        "then orbit the 3D cloud to see which skeleton fits best."
    )


if __name__ == "__main__":
    main()
