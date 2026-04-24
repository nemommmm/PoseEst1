#!/opt/anaconda3/envs/pose/bin/python
"""Render manual SKT/AFH agreement diagnostics on top of a 3D point cloud.

This tool builds a dense stereo point cloud for selected synchronized stereo
frames, crops the cloud around the subject using the union of two skeletons, and
renders both skeletons on top of the cropped cloud for manual inspection.

The current defaults compare:
- SKT historical-best optimized stereo pose
- AFH1 v1 hybrid pose

Outputs:
- snapshot PNGs
- machine-readable JSON summary
- lightweight EN/CN HTML reports
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AFH1_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from utils import StereoDataLoader  # noqa: E402


DATA_DIR = PROJECT_ROOT / "2025_Ergonomics_Data"
PARAM_PATH = SHARED_DIR / "camera_params.npz"

DEFAULT_SKT_PATH = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
DEFAULT_AFH_PATH = AFH1_DIR / "results" / "hybrid_skeleton_afh1_v1.npz"
DEFAULT_OUTPUT_DIR = AFH1_DIR / "results" / "22_pointcloud_overlay_afh1_v1"

SGBM_MIN_DISPARITY = int(os.environ.get("POSE_SGBM_MIN_DISPARITY", "100"))
SGBM_NUM_DISPARITIES = int(os.environ.get("POSE_SGBM_NUM_DISPARITIES", "256"))
SGBM_BLOCK_SIZE = int(os.environ.get("POSE_SGBM_BLOCK_SIZE", "9"))

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render SKT/AFH skeleton overlays on a cropped dense stereo point cloud."
    )
    parser.add_argument("--skt-path", default=str(DEFAULT_SKT_PATH), help="SKT NPZ path.")
    parser.add_argument("--afh-path", default=str(DEFAULT_AFH_PATH), help="AFH NPZ path.")
    parser.add_argument("--afh-label", default="AFH1 v1", help="Display label for the AFH skeleton.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for PNG/JSON/HTML artifacts.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Render one explicit synchronized stereo frame index.",
    )
    parser.add_argument(
        "--max-cloud-points",
        type=int,
        default=55000,
        help="Maximum number of full-cloud points rendered per frame.",
    )
    parser.add_argument(
        "--max-person-points",
        type=int,
        default=18000,
        help="Maximum number of cropped person-cloud points rendered per frame.",
    )
    parser.add_argument(
        "--bbox-margin-x",
        type=float,
        default=20.0,
        help="Extra crop margin in X (cm).",
    )
    parser.add_argument(
        "--bbox-margin-y",
        type=float,
        default=22.0,
        help="Extra crop margin in Y (cm).",
    )
    parser.add_argument(
        "--bbox-margin-z",
        type=float,
        default=28.0,
        help="Extra crop margin in Z (cm).",
    )
    return parser.parse_args()


def load_pose_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one pose NPZ into a plain dict."""
    payload = np.load(path, allow_pickle=True)
    return {name: payload[name] for name in payload.files}


def build_sgbm() -> cv2.StereoSGBM:
    """Create a StereoSGBM matcher with the repo defaults."""
    return cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISPARITY,
        numDisparities=SGBM_NUM_DISPARITIES,
        blockSize=SGBM_BLOCK_SIZE,
        P1=8 * 3 * SGBM_BLOCK_SIZE**2,
        P2=32 * 3 * SGBM_BLOCK_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_rectification() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification maps plus Q and the rectified left projection matrix."""
    calibration = np.load(PARAM_PATH)
    mtx_l, dist_l = calibration["mtx_l"], calibration["dist_l"]
    mtx_r, dist_r = calibration["mtx_r"], calibration["dist_r"]
    r_stereo, t_stereo = calibration["R"], calibration["T"]

    cap = cv2.VideoCapture(str(DATA_DIR / "0_video_left.avi"))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Cannot read left video to infer frame size.")
    height, width = frame.shape[:2]

    r1, r2, p1, p2, q_mat, _, _ = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        (width, height),
        r_stereo,
        t_stereo,
        alpha=0,
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, r1, p1, (width, height), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, r2, p2, (width, height), cv2.CV_32FC1)
    return map1_l, map2_l, map1_r, map2_r, q_mat, p1


def fetch_frame_pair(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one synchronized stereo pair by synchronized frame index."""
    loader = StereoDataLoader(
        str(DATA_DIR / "0_video_left.avi"),
        str(DATA_DIR / "1_video_right.avi"),
        str(DATA_DIR / "0_video_left.txt"),
        str(DATA_DIR / "1_video_right.txt"),
    )
    try:
        current = 0
        while True:
            frame_l, frame_r, _, _ = loader.get_next_pair()
            if frame_l is None:
                raise IndexError(f"Frame {frame_idx} exceeds synchronized stereo stream length.")
            if current == frame_idx:
                return frame_l, frame_r
            current += 1
    finally:
        loader.release()


def compute_disparity(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    map1_l: np.ndarray,
    map2_l: np.ndarray,
    map1_r: np.ndarray,
    map2_r: np.ndarray,
    sgbm: cv2.StereoSGBM,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify one stereo pair and compute the SGBM disparity map."""
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
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Reproject disparity to 3D and optionally subsample for rendering."""
    disparity_filled = np.nan_to_num(disparity, nan=0.0).astype(np.float32)
    points_3d = cv2.reprojectImageTo3D(disparity_filled, q_mat)

    mask = np.isfinite(disparity) & (disparity >= SGBM_MIN_DISPARITY)
    mask &= np.isfinite(points_3d).all(axis=2)
    z_vals = points_3d[:, :, 2]
    mask &= (z_vals > 30.0) & (z_vals < 1200.0)

    grid_y, grid_x = np.indices(disparity.shape)
    all_points = points_3d[mask].astype(np.float32)
    all_colors = color_bgr[mask][:, ::-1].astype(np.uint8)
    all_pixels = np.column_stack([grid_x[mask], grid_y[mask]]).astype(np.int32)
    total_count = int(len(all_points))
    if total_count == 0:
        return all_points, all_colors, all_pixels, total_count

    if max_points is not None and total_count > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(total_count, size=max_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]
        all_pixels = all_pixels[idx]
    return all_points, all_colors, all_pixels, total_count


def subsample_cloud(
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample a cloud only for rendering, not for geometric filtering."""
    total_count = int(len(cloud_points))
    if total_count <= max_points:
        return cloud_points, cloud_colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(total_count, size=max_points, replace=False)
    return cloud_points[idx], cloud_colors[idx]


def finite_points(pose: np.ndarray) -> np.ndarray:
    """Return finite joints only."""
    mask = np.isfinite(pose).all(axis=1)
    return pose[mask]


def overlap_joint_distance_cm(skt_pose: np.ndarray, afh_pose: np.ndarray) -> tuple[float, int]:
    """Compute mean 3D joint distance between SKT and AFH on overlapping valid joints."""
    overlap = np.isfinite(skt_pose).all(axis=1) & np.isfinite(afh_pose).all(axis=1)
    overlap_count = int(np.sum(overlap))
    if overlap_count == 0:
        return float("nan"), 0
    dist = np.linalg.norm(skt_pose[overlap] - afh_pose[overlap], axis=1)
    return float(np.mean(dist)), overlap_count


def choose_representative_frames(skt_data: dict[str, np.ndarray], afh_data: dict[str, np.ndarray]) -> dict[str, int]:
    """Pick three representative frames based on SKT/AFH disagreement."""
    skt_points = skt_data["keypoints"]
    afh_points = afh_data["keypoints"]
    if skt_points.shape != afh_points.shape:
        raise ValueError(f"Shape mismatch: SKT {skt_points.shape} vs AFH {afh_points.shape}")

    frame_dist = np.full(len(skt_points), np.nan, dtype=np.float64)
    overlap_counts = np.zeros(len(skt_points), dtype=np.int32)
    for idx in range(len(skt_points)):
        frame_dist[idx], overlap_counts[idx] = overlap_joint_distance_cm(skt_points[idx], afh_points[idx])

    valid_mask = np.isfinite(frame_dist) & (overlap_counts >= 12)
    reasonable_mask = np.zeros(len(skt_points), dtype=bool)
    for idx in np.where(valid_mask)[0]:
        finite_skt = finite_points(skt_points[idx])
        finite_afh = finite_points(afh_points[idx])
        joint_union = np.concatenate([finite_skt, finite_afh], axis=0)
        if len(joint_union) == 0:
            continue
        median_z = float(np.nanmedian(joint_union[:, 2]))
        span_xyz = np.nanmax(joint_union, axis=0) - np.nanmin(joint_union, axis=0)
        reasonable_mask[idx] = (
            50.0 < median_z < 1000.0
            and span_xyz[0] < 250.0
            and span_xyz[1] < 260.0
            and span_xyz[2] < 300.0
        )
    valid_mask &= reasonable_mask
    if not np.any(valid_mask):
        raise RuntimeError("No frame has enough overlapping valid joints in a reasonable camera workspace.")

    valid_idx = np.where(valid_mask)[0]
    valid_dist = frame_dist[valid_mask]
    median_target = float(np.median(valid_dist))
    median_idx = int(valid_idx[np.argmin(np.abs(valid_dist - median_target))])
    return {
        "good_agreement": int(valid_idx[np.argmin(valid_dist)]),
        "median_agreement": median_idx,
        "bad_agreement": int(valid_idx[np.argmax(valid_dist)]),
    }


def format_metric(value: float | None, unit: str = "", decimals: int = 1) -> str:
    """Format a numeric metric with a fallback for missing values."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}{unit}"


def project_pose_to_rectified(
    pose: np.ndarray,
    p1: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Project one 3D pose into rectified-left image coordinates."""
    height, width = image_shape
    proj = np.full((len(pose), 2), np.nan, dtype=np.float32)
    finite = np.isfinite(pose).all(axis=1)
    if not np.any(finite):
        return proj
    pts_h = np.concatenate(
        [pose[finite].astype(np.float32), np.ones((int(np.sum(finite)), 1), dtype=np.float32)],
        axis=1,
    )
    uvw = pts_h @ p1.T
    valid = np.abs(uvw[:, 2]) > 1e-6
    uv = np.full((len(uvw), 2), np.nan, dtype=np.float32)
    uv[valid, 0] = uvw[valid, 0] / uvw[valid, 2]
    uv[valid, 1] = uvw[valid, 1] / uvw[valid, 2]
    in_bounds = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    proj_valid = np.full((len(uv), 2), np.nan, dtype=np.float32)
    proj_valid[in_bounds] = uv[in_bounds]
    proj[finite] = proj_valid
    return proj


def build_person_mask_from_projections(
    image_shape: tuple[int, int],
    projected_poses: Iterable[np.ndarray],
) -> np.ndarray:
    """Create a coarse person mask from projected skeletons in the rectified left image."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    all_valid_points: list[np.ndarray] = []

    for proj in projected_poses:
        valid = np.isfinite(proj).all(axis=1)
        if not np.any(valid):
            continue
        pts = np.round(proj[valid]).astype(np.int32)
        all_valid_points.append(pts.astype(np.float32))
        y_span = int(np.max(pts[:, 1]) - np.min(pts[:, 1])) if len(pts) > 1 else 160
        line_thickness = int(np.clip(round(y_span * 0.085), 16, 34))
        joint_radius = int(np.clip(round(y_span * 0.055), 10, 22))

        torso_idx = [idx for idx in (5, 6, 11, 12) if valid[idx]]
        if len(torso_idx) >= 3:
            torso_pts = np.round(proj[torso_idx]).astype(np.int32)
            cv2.fillConvexPoly(mask, torso_pts, 255)

        hull_pts = cv2.convexHull(pts)
        if len(hull_pts) >= 3:
            cv2.fillConvexPoly(mask, hull_pts, 255)

        for start_idx, end_idx in COCO_EDGES:
            if valid[start_idx] and valid[end_idx]:
                p0 = tuple(np.round(proj[start_idx]).astype(int))
                p1_pt = tuple(np.round(proj[end_idx]).astype(int))
                cv2.line(mask, p0, p1_pt, 255, thickness=line_thickness)
        for point in pts:
            cv2.circle(mask, tuple(point), joint_radius, 255, thickness=-1)

    if not all_valid_points:
        return mask

    merged = np.concatenate(all_valid_points, axis=0)
    x0 = max(int(np.floor(np.min(merged[:, 0])) - 24), 0)
    x1 = min(int(np.ceil(np.max(merged[:, 0])) + 24), width - 1)
    y0 = max(int(np.floor(np.min(merged[:, 1])) - 28), 0)
    y1 = min(int(np.ceil(np.max(merged[:, 1])) + 28), height - 1)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=2)

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    return mask


def crop_person_cloud(
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    cloud_pixels: np.ndarray,
    skeletons: Iterable[np.ndarray],
    person_mask_2d: np.ndarray | None,
    margin_xyz: tuple[float, float, float],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float] | int | float]]:
    """Crop the dense cloud around the subject using 2D person mask + 3D skeleton bounds."""
    all_finite = [finite_points(pose) for pose in skeletons]
    all_finite = [points for points in all_finite if len(points) > 0]
    if not all_finite:
        empty_bbox = {
            "min_xyz_cm": [math.nan] * 3,
            "max_xyz_cm": [math.nan] * 3,
            "full_count": int(len(cloud_points)),
            "mask_fraction": 0.0,
        }
        return cloud_points[:0], cloud_colors[:0], empty_bbox

    joint_union = np.concatenate(all_finite, axis=0)
    margin_x, margin_y, margin_z = margin_xyz
    min_xyz = np.nanmin(joint_union, axis=0) - np.array([margin_x, margin_y, margin_z], dtype=np.float32)
    max_xyz = np.nanmax(joint_union, axis=0) + np.array([margin_x, margin_y, margin_z], dtype=np.float32)
    mask = np.all((cloud_points >= min_xyz) & (cloud_points <= max_xyz), axis=1)
    if person_mask_2d is not None and len(cloud_pixels) > 0:
        pixel_mask = person_mask_2d[cloud_pixels[:, 1], cloud_pixels[:, 0]] > 0
        mask &= pixel_mask

    cropped_points = cloud_points[mask]
    cropped_colors = cloud_colors[mask]
    full_count = int(len(cropped_points))
    if full_count > max_points:
        rng = np.random.default_rng(1)
        idx = rng.choice(full_count, size=max_points, replace=False)
        cropped_points = cropped_points[idx]
        cropped_colors = cropped_colors[idx]

    bbox = {
        "min_xyz_cm": min_xyz.astype(float).tolist(),
        "max_xyz_cm": max_xyz.astype(float).tolist(),
        "full_count": full_count,
        "mask_fraction": float(np.mean(mask)) if len(mask) > 0 else 0.0,
    }
    return cropped_points, cropped_colors, bbox


def nearest_cloud_distance_cm(skeleton: np.ndarray, cloud_points: np.ndarray) -> tuple[float | None, int]:
    """Mean nearest-neighbour distance from skeleton joints to the cropped point cloud."""
    joints = finite_points(skeleton)
    if len(joints) == 0 or len(cloud_points) == 0:
        return None, 0
    tree = cKDTree(cloud_points)
    dist, _ = tree.query(joints, k=1)
    return float(np.mean(dist)), int(len(joints))


def to_display_coords(points: np.ndarray) -> np.ndarray:
    """Map OpenCV camera coords to an upright viewer frame: X-right, Y-depth, Z-up."""
    pts = np.asarray(points, dtype=np.float32)
    return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])


def set_axes_equal(ax, points: np.ndarray) -> None:
    """Set equal scale on a Matplotlib 3D axis."""
    if len(points) == 0:
        return
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_skeleton(ax, pose: np.ndarray, color: str, label: str) -> None:
    """Draw one COCO-17 skeleton in the display frame."""
    finite = np.isfinite(pose).all(axis=1)
    display = to_display_coords(pose[finite]) if np.any(finite) else np.zeros((0, 3), dtype=np.float32)
    display_all = np.full((len(pose), 3), np.nan, dtype=np.float32)
    display_all[finite] = display
    for start_idx, end_idx in COCO_EDGES:
        if finite[start_idx] and finite[end_idx]:
            seg = display_all[[start_idx, end_idx]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=2.6)
    if np.any(finite):
        ax.scatter(
            display[:, 0],
            display[:, 1],
            display[:, 2],
            s=26,
            c=color,
            depthshade=False,
            label=label,
        )


def style_3d_axis(ax) -> None:
    """Apply a higher-contrast dark theme to 3D diagnostic panels."""
    ax.set_facecolor("#0b1220")
    ax.grid(False)
    pane_rgba = (0.07, 0.10, 0.16, 1.0)
    ax.xaxis.set_pane_color(pane_rgba)
    ax.yaxis.set_pane_color(pane_rgba)
    ax.zaxis.set_pane_color(pane_rgba)
    tick_color = "#d7dee9"
    label_color = "#e5eef8"
    ax.tick_params(colors=tick_color, labelsize=8)
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    ax.zaxis.label.set_color(label_color)
    ax.title.set_color(label_color)


def make_person_cloud_colors(person_cloud: np.ndarray) -> np.ndarray:
    """Color the person cloud by height for higher visual contrast than raw RGB."""
    if len(person_cloud) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    disp_cloud = to_display_coords(person_cloud)
    height_vals = disp_cloud[:, 2]
    vmin = float(np.nanpercentile(height_vals, 5))
    vmax = float(np.nanpercentile(height_vals, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(height_vals))
        vmax = float(np.nanmax(height_vals) + 1e-6)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return plt.get_cmap("viridis")(norm(height_vals))


def make_highlight_person_colors(person_cloud: np.ndarray) -> np.ndarray:
    """Create a fixed highlight color so person cloud stands out from the environment."""
    if len(person_cloud) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rgba = np.tile(np.array([[1.0, 0.67, 0.20, 0.95]], dtype=np.float32), (len(person_cloud), 1))
    return rgba


def draw_projected_skeleton_2d(
    image_rgb: np.ndarray,
    projected_pose: np.ndarray,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Draw one projected skeleton on an RGB image."""
    canvas = image_rgb.copy()
    valid = np.isfinite(projected_pose).all(axis=1)
    for start_idx, end_idx in COCO_EDGES:
        if valid[start_idx] and valid[end_idx]:
            p0 = tuple(np.round(projected_pose[start_idx]).astype(int))
            p1_pt = tuple(np.round(projected_pose[end_idx]).astype(int))
            cv2.line(canvas, p0, p1_pt, color_rgb, thickness=3, lineType=cv2.LINE_AA)
    for point in projected_pose[valid]:
        p = tuple(np.round(point).astype(int))
        cv2.circle(canvas, p, 4, color_rgb, thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def make_left_overlay_image(
    left_rect: np.ndarray,
    person_mask_2d: np.ndarray,
    skt_proj_2d: np.ndarray,
    afh_proj_2d: np.ndarray,
) -> np.ndarray:
    """Overlay person mask and both skeleton projections on the left rectified image."""
    image_rgb = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()
    if person_mask_2d is not None and np.any(person_mask_2d > 0):
        tint = np.zeros_like(overlay)
        tint[..., 1] = 220
        tint[..., 2] = 110
        keep = person_mask_2d > 0
        overlay[keep] = np.clip(0.58 * overlay[keep] + 0.42 * tint[keep], 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(person_mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 235, 140), 2, lineType=cv2.LINE_AA)
    overlay = draw_projected_skeleton_2d(overlay, skt_proj_2d, (0, 212, 255))
    overlay = draw_projected_skeleton_2d(overlay, afh_proj_2d, (255, 79, 163))
    return overlay


def draw_person_cloud_panel(
    ax,
    person_cloud: np.ndarray,
    person_colors: np.ndarray,
    skt_pose: np.ndarray,
    afh_pose: np.ndarray,
    skt_label: str,
    afh_label: str,
    title: str,
    elev: float,
    azim: float,
    show_legend: bool,
) -> None:
    """Draw a zoomed-in person-cloud panel from one camera view."""
    style_3d_axis(ax)
    if len(person_cloud) > 0:
        disp_cloud = to_display_coords(person_cloud)
        ax.scatter(
            disp_cloud[:, 0],
            disp_cloud[:, 1],
            disp_cloud[:, 2],
            s=4.6,
            c=person_colors,
            alpha=0.88,
            linewidths=0,
            depthshade=False,
        )
        set_axes_equal(ax, disp_cloud)
    draw_skeleton(ax, skt_pose, color="#00d4ff", label=skt_label)
    draw_skeleton(ax, afh_pose, color="#ff4fa3", label=afh_label)
    ax.set_title(title)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Depth Z (cm)")
    ax.set_zlabel("Height (cm)")
    ax.view_init(elev=elev, azim=azim)
    if show_legend:
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#e5eef8")


def render_snapshot(
    output_path: Path,
    left_rect: np.ndarray,
    left_overlay_rgb: np.ndarray,
    disparity: np.ndarray,
    full_cloud: np.ndarray,
    person_cloud: np.ndarray,
    skt_pose: np.ndarray,
    afh_pose: np.ndarray,
    skt_label: str,
    afh_label: str,
    frame_info: dict[str, object],
) -> None:
    """Render a single diagnostic PNG snapshot."""
    fig = plt.figure(figsize=(20, 11))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.15], height_ratios=[1.0, 1.05])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2], projection="3d")
    ax3 = fig.add_subplot(grid[1, 0], projection="3d")
    ax4 = fig.add_subplot(grid[1, 1], projection="3d")
    ax5 = fig.add_subplot(grid[1, 2], projection="3d")

    ax0.imshow(left_overlay_rgb)
    ax0.set_title(f"Rectified left frame + projected mask\nframe={frame_info['frame_idx']}")
    ax0.axis("off")

    finite = np.isfinite(disparity)
    if np.any(finite):
        disp_min = float(np.nanpercentile(disparity[finite], 5))
        disp_max = float(np.nanpercentile(disparity[finite], 95))
    else:
        disp_min, disp_max = 0.0, 1.0
    ax1.imshow(cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB))
    disp_show = np.where(np.isfinite(disparity), disparity, np.nan)
    ax1.imshow(disp_show, cmap="turbo", vmin=disp_min, vmax=disp_max, alpha=0.66)
    ax1.set_title("Left frame + disparity")
    ax1.axis("off")

    style_3d_axis(ax2)
    if len(full_cloud) > 0:
        disp_cloud = to_display_coords(full_cloud)
        ax2.scatter(
            disp_cloud[:, 0],
            disp_cloud[:, 1],
            disp_cloud[:, 2],
            s=0.28,
            c="#93a4b8",
            alpha=0.035,
            linewidths=0,
            depthshade=False,
        )
    if len(person_cloud) > 0:
        disp_person_context = to_display_coords(person_cloud)
        context_highlight_colors = make_highlight_person_colors(person_cloud)
        ax2.scatter(
            disp_person_context[:, 0],
            disp_person_context[:, 1],
            disp_person_context[:, 2],
            s=1.7,
            c=context_highlight_colors,
            alpha=0.92,
            linewidths=0,
            depthshade=False,
        )
    draw_skeleton(ax2, skt_pose, color="#00d4ff", label=skt_label)
    draw_skeleton(ax2, afh_pose, color="#ff4fa3", label=afh_label)
    ax2.set_title("Full cloud context (env grey, person highlighted) + skeletons")
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Depth Z (cm)")
    ax2.set_zlabel("Height (cm)")
    ax2.view_init(elev=16, azim=-68)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#e5eef8")

    person_render_colors = make_person_cloud_colors(person_cloud)
    cloud_metric_title = (
        f"SKT-cloud {frame_info['skt_cloud_nn_cm']:.1f} cm | "
        f"AFH-cloud {frame_info['afh_cloud_nn_cm']:.1f} cm"
        if frame_info["skt_cloud_nn_cm"] is not None and frame_info["afh_cloud_nn_cm"] is not None
        else "point-cloud distance unavailable"
    )
    draw_person_cloud_panel(
        ax3,
        person_cloud,
        person_render_colors,
        skt_pose,
        afh_pose,
        skt_label,
        afh_label,
        f"Person cloud (oblique)\n{cloud_metric_title}",
        elev=18,
        azim=-70,
        show_legend=True,
    )
    draw_person_cloud_panel(
        ax4,
        person_cloud,
        person_render_colors,
        skt_pose,
        afh_pose,
        skt_label,
        afh_label,
        "Person cloud (camera-view, same left-right as image)",
        elev=0,
        azim=-90,
        show_legend=False,
    )
    draw_person_cloud_panel(
        ax5,
        person_cloud,
        person_render_colors,
        skt_pose,
        afh_pose,
        skt_label,
        afh_label,
        "Person cloud (side view)",
        elev=8,
        azim=2,
        show_legend=False,
    )

    fig.suptitle(
        f"{frame_info['label']} | SKT-AFH overlap MPJPE {frame_info['skt_afh_overlap_cm']:.1f} cm "
        f"({frame_info['overlap_joint_count']} joints)",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_report_html(report_title: str, rows: list[dict[str, object]], zh: bool) -> str:
    """Create a compact HTML report for the rendered snapshots."""
    cards: list[str] = []
    for row in rows:
        png_name = row["png_name"]
        label = row["label"]
        if zh:
            summary = (
                f"帧 {row['frame_idx']} | SKT-AFH 分歧 {format_metric(row['skt_afh_overlap_cm'], ' cm')} | "
                f"SKT-cloud {format_metric(row['skt_cloud_nn_cm'], ' cm')} | "
                f"AFH-cloud {format_metric(row['afh_cloud_nn_cm'], ' cm')} | "
                f"人体点云 {row['person_cloud_points']} 点 | "
                f"mask保留 {row['person_mask_fraction'] * 100:.2f}%"
            )
        else:
            summary = (
                f"Frame {row['frame_idx']} | SKT-AFH {format_metric(row['skt_afh_overlap_cm'], ' cm')} | "
                f"SKT-cloud {format_metric(row['skt_cloud_nn_cm'], ' cm')} | "
                f"AFH-cloud {format_metric(row['afh_cloud_nn_cm'], ' cm')} | "
                f"person cloud {row['person_cloud_points']} pts | "
                f"mask kept {row['person_mask_fraction'] * 100:.2f}%"
            )
        cards.append(
            "\n".join(
                [
                    '<section class="card">',
                    f"<h2>{label}</h2>",
                    f'<img src="{png_name}" alt="{label}"/>',
                    f"<p>{summary}</p>",
                    "</section>",
                ]
            )
        )

    if zh:
        intro = (
            "<p>该初版诊断把 SKT 与 AFH 骨架叠加到同一帧的 SGBM 稠密点云上。"
            "当前版本先在校正后的左图上用两套骨架投影生成粗人体 mask，"
            "再结合两套骨架的联合 3D 包围盒裁切点云，用于尽量去掉背景。"
            "当前版本仍是“局部人体附近点云”，不是经过语义分割得到的纯人体表面；"
            "因此它更适合做几何人工核查，而不是严格的自动定量评分。"
            "</p>"
            "<p>其中 skeleton-to-cloud 最近邻距离越小，通常表示骨架越贴近当前相机观测到的人体表面；"
            "但它仍然受 SGBM 质量影响，所以只能作为人工诊断辅助。</p>"
        )
    else:
        intro = (
            "<p>This first-pass diagnostic overlays SKT and AFH skeletons on the same-frame "
            "SGBM point cloud. The current version first builds a coarse person mask from the "
            "projected skeletons in the rectified left frame, then combines it with the union 3D "
            "bounding box of both skeletons to suppress the background. The result is still a local person-neighbourhood "
            "cloud rather than a clean person-only segmentation, so it should be used for geometric "
            "manual inspection rather than strict automatic scoring.</p>"
            "<p>Lower skeleton-to-cloud nearest-neighbour distance usually "
            "means better geometric agreement with the currently observed person surface, but the "
            "metric still depends on SGBM quality and should be treated as a manual diagnostic aid.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="{'zh-CN' if zh else 'en'}">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{report_title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "PingFang SC", "Helvetica Neue", Arial, sans-serif;
      background: #f4f6f8;
      color: #1d2430;
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 12px;
      color: #154360;
    }}
    p {{
      line-height: 1.75;
    }}
    .card {{
      background: #fff;
      border-radius: 12px;
      padding: 20px;
      margin-top: 20px;
      box-shadow: 0 8px 26px rgba(20, 32, 44, 0.08);
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 8px;
      background: #0f1720;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{report_title}</h1>
    {intro}
    {''.join(cards)}
  </div>
</body>
</html>
"""


def write_reports(output_dir: Path, report_rows: list[dict[str, object]], afh_label: str) -> None:
    """Write EN/CN HTML reports for the rendered diagnostic frames."""
    report_en = make_report_html(
        f"Manual Point-Cloud Agreement Check: SKT vs {afh_label}",
        report_rows,
        zh=False,
    )
    report_cn = make_report_html(
        f"人工点云一致性检查：SKT vs {afh_label}",
        report_rows,
        zh=True,
    )
    (output_dir / "pointcloud_overlay_report.html").write_text(report_en, encoding="utf-8")
    (output_dir / "pointcloud_overlay_report_CN.html").write_text(report_cn, encoding="utf-8")


def main() -> None:
    """Render point-cloud diagnostics for representative SKT/AFH frames."""
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    skt_path = Path(args.skt_path).resolve()
    afh_path = Path(args.afh_path).resolve()
    skt_data = load_pose_npz(skt_path)
    afh_data = load_pose_npz(afh_path)

    if "timestamps" not in skt_data or "timestamps" not in afh_data:
        raise ValueError("Both SKT and AFH NPZ files must contain timestamps.")
    if not np.allclose(skt_data["timestamps"], afh_data["timestamps"], atol=1e-6):
        raise ValueError("SKT and AFH timestamps differ; this diagnostic requires a shared timeline.")

    if args.frame is not None:
        selected = {"manual_frame": int(args.frame)}
    else:
        selected = choose_representative_frames(skt_data, afh_data)

    map1_l, map2_l, map1_r, map2_r, q_mat, p1 = compute_rectification()
    sgbm = build_sgbm()
    margin_xyz = (args.bbox_margin_x, args.bbox_margin_y, args.bbox_margin_z)

    report_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for label, frame_idx in selected.items():
        frame_l, frame_r = fetch_frame_pair(frame_idx)
        left_rect, disparity = compute_disparity(frame_l, frame_r, map1_l, map2_l, map1_r, map2_r, sgbm)
        full_cloud_dense, full_colors_dense, full_pixels_dense, full_count = build_point_cloud(
            disparity,
            left_rect,
            q_mat,
            max_points=None,
        )

        skt_pose = np.asarray(skt_data["keypoints"][frame_idx], dtype=np.float32)
        afh_pose = np.asarray(afh_data["keypoints"][frame_idx], dtype=np.float32)
        skt_proj_2d = project_pose_to_rectified(skt_pose, p1, left_rect.shape[:2])
        afh_proj_2d = project_pose_to_rectified(afh_pose, p1, left_rect.shape[:2])
        person_mask_2d = build_person_mask_from_projections(
            left_rect.shape[:2],
            projected_poses=[skt_proj_2d, afh_proj_2d],
        )
        person_cloud, person_colors, bbox_info = crop_person_cloud(
            full_cloud_dense,
            full_colors_dense,
            full_pixels_dense,
            skeletons=[skt_pose, afh_pose],
            person_mask_2d=person_mask_2d,
            margin_xyz=margin_xyz,
            max_points=args.max_person_points,
        )
        full_cloud_render, _ = subsample_cloud(
            full_cloud_dense,
            full_colors_dense,
            max_points=args.max_cloud_points,
            seed=0,
        )
        left_overlay_rgb = make_left_overlay_image(left_rect, person_mask_2d, skt_proj_2d, afh_proj_2d)

        skt_afh_overlap_cm, overlap_joint_count = overlap_joint_distance_cm(skt_pose, afh_pose)
        skt_cloud_nn_cm, skt_joint_count = nearest_cloud_distance_cm(skt_pose, person_cloud)
        afh_cloud_nn_cm, afh_joint_count = nearest_cloud_distance_cm(afh_pose, person_cloud)

        snapshot_name = f"overlay_{label}.png"
        snapshot_path = output_dir / snapshot_name
        frame_info = {
            "label": label.replace("_", " "),
            "frame_idx": int(frame_idx),
            "timestamp_rel_s": float(skt_data["timestamps"][frame_idx] - skt_data["timestamps"][0]),
            "skt_afh_overlap_cm": float(skt_afh_overlap_cm),
            "overlap_joint_count": int(overlap_joint_count),
            "skt_cloud_nn_cm": skt_cloud_nn_cm,
            "afh_cloud_nn_cm": afh_cloud_nn_cm,
            "skt_valid_joints": int(np.sum(np.isfinite(skt_pose).all(axis=1))),
            "afh_valid_joints": int(np.sum(np.isfinite(afh_pose).all(axis=1))),
            "skt_cloud_nn_joint_count": int(skt_joint_count),
            "afh_cloud_nn_joint_count": int(afh_joint_count),
            "full_cloud_points": int(full_count),
            "person_cloud_points": int(bbox_info["full_count"]),
            "person_bbox_min_xyz_cm": bbox_info["min_xyz_cm"],
            "person_bbox_max_xyz_cm": bbox_info["max_xyz_cm"],
            "person_mask_fraction": bbox_info["mask_fraction"],
        }
        render_snapshot(
            snapshot_path,
            left_rect,
            left_overlay_rgb,
            disparity,
            full_cloud_render,
            person_cloud,
            skt_pose,
            afh_pose,
            skt_label="SKT",
            afh_label=args.afh_label,
            frame_info=frame_info,
        )

        frame_info["png_name"] = snapshot_name
        summary_rows.append(frame_info)
        report_rows.append(frame_info)

    summary = {
        "skt_path": str(skt_path),
        "afh_path": str(afh_path),
        "afh_label": args.afh_label,
        "output_dir": str(output_dir),
        "selection": selected,
        "bbox_margin_cm": {
            "x": args.bbox_margin_x,
            "y": args.bbox_margin_y,
            "z": args.bbox_margin_z,
        },
        "frames": summary_rows,
        "notes": [
            "Raw stereo videos are upside-down; StereoDataLoader rotates them 180 degrees before rectification.",
            "Point cloud uses SGBM disparity, so it is only a manual geometric sanity check rather than a ground-truth surface.",
            "Person cloud is filtered by a projected skeleton mask in the rectified left image and then cropped by the union 3D bounding box of SKT and AFH skeletons plus a configurable margin.",
        ],
    }
    (output_dir / "pointcloud_overlay_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_reports(output_dir, report_rows, args.afh_label)

    print(f"[saved] {output_dir / 'pointcloud_overlay_summary.json'}")
    print(f"[saved] {output_dir / 'pointcloud_overlay_report.html'}")
    print(f"[saved] {output_dir / 'pointcloud_overlay_report_CN.html'}")
    for row in report_rows:
        print(
            "[frame] "
            f"{row['label']} idx={row['frame_idx']} "
            f"SKT-AFH={format_metric(row['skt_afh_overlap_cm'], 'cm', decimals=2)} "
            f"SKT-cloud={format_metric(row['skt_cloud_nn_cm'], 'cm', decimals=2)} "
            f"AFH-cloud={format_metric(row['afh_cloud_nn_cm'], 'cm', decimals=2)}"
        )


if __name__ == "__main__":
    main()
