#!/opt/anaconda3/envs/pose/bin/python
"""
Single-frame dense stereo point-cloud visualizer for Direction B.

The raw stereo videos in this dataset are upside-down. This script relies on
StereoDataLoader, which rotates both frames by 180 degrees before returning
them, so every rectified/disparity/point-cloud result is rendered in an
upright view. The script must not rotate frames a second time.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
sys.path.insert(0, str(SHARED_DIR))

from utils import StereoDataLoader


DATA_DIR = PROJECT_ROOT / "2025_Ergonomics_Data"
RESULTS_DIR = METHOD_DIR / "results"
PARAM_PATH = SHARED_DIR / "camera_params.npz"
DEFAULT_NPZ = RESULTS_DIR / "yolo_3d_raw_yolov8m_sgbm.npz"

SGBM_MIN_DISPARITY = int(os.environ.get("POSE_SGBM_MIN_DISPARITY", "100"))
SGBM_NUM_DISPARITIES = int(os.environ.get("POSE_SGBM_NUM_DISPARITIES", "256"))
SGBM_BLOCK_SIZE = int(os.environ.get("POSE_SGBM_BLOCK_SIZE", "9"))
LOOKUP_WINDOW = int(os.environ.get("POSE_DISPARITY_WINDOW", "5"))

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one dense stereo point cloud frame.")
    parser.add_argument("--frame", type=int, default=None, help="Matched stereo frame index to render.")
    parser.add_argument(
        "--category",
        choices=["good", "bad", "low_coverage"],
        default=None,
        help="Representative frame category when --frame is not given.",
    )
    parser.add_argument("--tag", default="", help="Output tag prefix.")
    parser.add_argument("--output-html", default=None, help="Optional output HTML path.")
    parser.add_argument("--output-png", default=None, help="Optional output PNG path.")
    parser.add_argument("--output-json", default=None, help="Optional output metadata JSON path.")
    parser.add_argument("--npz-path", default=str(DEFAULT_NPZ), help="Dense stereo raw results NPZ.")
    parser.add_argument("--max-points", type=int, default=45000, help="Maximum rendered point count.")
    parser.add_argument("--lookup-window", type=int, default=LOOKUP_WINDOW, help="Median lookup window size.")
    return parser.parse_args()


def build_sgbm() -> cv2.StereoSGBM:
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


def safe_nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    result = np.full(values.shape[:axis] + values.shape[axis + 1 :], np.nan, dtype=np.float64)
    moved = np.moveaxis(values, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    medians = np.full(flat.shape[1], np.nan, dtype=np.float64)
    for idx in range(flat.shape[1]):
        finite = flat[:, idx][np.isfinite(flat[:, idx])]
        if finite.size > 0:
            medians[idx] = float(np.median(finite))
    return medians.reshape(result.shape)


def compute_sgbm_disparity(frame_l_rect: np.ndarray, frame_r_rect: np.ndarray, sgbm) -> np.ndarray:
    gray_l = cv2.cvtColor(frame_l_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_r_rect, cv2.COLOR_BGR2GRAY)
    disparity_raw = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disparity = disparity_raw.copy()
    disparity[disparity_raw < 0] = np.nan
    return disparity


def compute_rectification() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calibration = np.load(PARAM_PATH)
    mtx_l, dist_l = calibration["mtx_l"], calibration["dist_l"]
    mtx_r, dist_r = calibration["mtx_r"], calibration["dist_r"]
    R_stereo, T_stereo = calibration["R"], calibration["T"]

    cap = cv2.VideoCapture(str(DATA_DIR / "0_video_left.avi"))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read the left video for frame size.")
    h, w = frame.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        (w, h),
        R_stereo,
        T_stereo,
        alpha=0,
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w, h), cv2.CV_32FC1)
    return map1_l, map2_l, map1_r, map2_r, Q


def select_representative_frames(npz_data: dict[str, np.ndarray]) -> dict[str, int]:
    keypoints = npz_data["keypoints"]
    epipolar = npz_data["epipolar_error"]
    stereo_quality = npz_data["stereo_quality"]
    pair_conf = npz_data["pair_confidence"]
    disparity = npz_data["disparity_px"]

    valid_mask = np.isfinite(keypoints).all(axis=2)
    valid_ratio = np.mean(valid_mask, axis=1)
    mean_epi = safe_nanmedian(np.where(np.isfinite(epipolar), epipolar, np.nan), axis=1)
    mean_quality = safe_nanmedian(np.where(np.isfinite(stereo_quality), stereo_quality, np.nan), axis=1)
    mean_pair_conf = safe_nanmedian(np.where(np.isfinite(pair_conf), pair_conf, np.nan), axis=1)
    mean_disparity = safe_nanmedian(np.where(np.isfinite(disparity), disparity, np.nan), axis=1)

    def pick_argmin(mask: np.ndarray, score: np.ndarray) -> int:
        if not np.any(mask):
            mask = np.isfinite(score)
        masked = np.where(mask, score, np.inf)
        return int(np.nanargmin(masked))

    def pick_argmax(mask: np.ndarray, score: np.ndarray) -> int:
        if not np.any(mask):
            mask = np.isfinite(score)
        masked = np.where(mask, score, -np.inf)
        return int(np.nanargmax(masked))

    finite = (
        np.isfinite(mean_epi)
        & np.isfinite(mean_quality)
        & np.isfinite(mean_pair_conf)
        & np.isfinite(mean_disparity)
    )
    good_score = 0.85 * mean_epi - 7.0 * mean_quality - 2.5 * mean_pair_conf + (1.0 - valid_ratio) * 8.0
    bad_score = 1.10 * mean_epi + (1.0 - valid_ratio) * 28.0 - 3.0 * mean_quality - 1.5 * mean_pair_conf
    low_cov_score = valid_ratio + 0.10 * mean_quality + 0.05 * mean_pair_conf

    return {
        "good": pick_argmin(finite & (valid_ratio >= 0.70), good_score),
        "bad": pick_argmax(finite & (valid_ratio >= 0.20), bad_score),
        "low_coverage": pick_argmin(finite & (valid_ratio <= 0.45), low_cov_score),
    }


def load_dense_npz(npz_path: Path) -> dict[str, np.ndarray]:
    npz = np.load(npz_path, allow_pickle=True)
    return {name: npz[name] for name in npz.files}


def fetch_frame_pair(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
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


def build_point_cloud(
    disparity_map: np.ndarray,
    color_bgr: np.ndarray,
    Q: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    disparity_filled = np.nan_to_num(disparity_map, nan=0.0).astype(np.float32)
    points_3d = cv2.reprojectImageTo3D(disparity_filled, Q)
    mask = np.isfinite(disparity_map) & (disparity_map >= SGBM_MIN_DISPARITY)
    mask &= np.isfinite(points_3d).all(axis=2)
    z = points_3d[:, :, 2]
    mask &= (z > 30.0) & (z < 1200.0)

    all_points = points_3d[mask].astype(np.float64)
    all_colors = color_bgr[mask][:, ::-1].astype(np.uint8)
    total_points = int(len(all_points))
    if total_points == 0:
        return all_points, all_colors, total_points

    if total_points > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(total_points, size=max_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]
    return all_points, all_colors, total_points


def render_png(
    output_png: Path,
    left_rect: np.ndarray,
    disparity_map: np.ndarray,
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    skeleton_points: np.ndarray,
    frame_idx: int,
    tag: str,
) -> None:
    fig = plt.figure(figsize=(16, 6))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")

    ax0.imshow(cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"Rectified left frame\nframe={frame_idx}")
    ax0.axis("off")

    disp_show = disparity_map.copy()
    finite = np.isfinite(disp_show)
    if np.any(finite):
        disp_min = np.nanpercentile(disp_show[finite], 5)
        disp_max = np.nanpercentile(disp_show[finite], 95)
    else:
        disp_min, disp_max = 0.0, 1.0
    ax1.imshow(disp_show, cmap="turbo", vmin=disp_min, vmax=disp_max)
    ax1.set_title("SGBM disparity")
    ax1.axis("off")

    if len(cloud_points) > 0:
        colors = cloud_colors.astype(np.float32) / 255.0
        ax2.scatter(
            cloud_points[:, 0],
            cloud_points[:, 2],
            cloud_points[:, 1],
            s=0.5,
            c=colors,
            alpha=0.35,
            linewidths=0,
        )
    finite_skeleton = np.isfinite(skeleton_points).all(axis=1)
    for start_idx, end_idx in COCO_EDGES:
        if finite_skeleton[start_idx] and finite_skeleton[end_idx]:
            seg = skeleton_points[[start_idx, end_idx]]
            ax2.plot(seg[:, 0], seg[:, 2], seg[:, 1], color="cyan", linewidth=2.0)
    if np.any(finite_skeleton):
        skel = skeleton_points[finite_skeleton]
        ax2.scatter(skel[:, 0], skel[:, 2], skel[:, 1], s=18, c="orange", depthshade=False)

    ax2.set_title(f"Dense stereo point cloud\n{tag or 'frame'}")
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Z (cm)")
    ax2.set_zlabel("Y (cm)")
    ax2.view_init(elev=18, azim=-64)
    ax2.grid(False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_html(
    output_html: Path,
    png_path: Path,
    metadata: dict,
) -> None:
    png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    metadata_json = json.dumps(metadata, indent=2)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dense Stereo Point Cloud</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #0f1720; color: #e8eef6; }}
    #wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; min-height: 100vh; }}
    #viewer {{ width: 100%; min-height: 100vh; display:flex; align-items:center; justify-content:center; background:#0d141c; }}
    #viewer img {{ max-width: 100%; height: auto; display:block; }}
    #side {{ padding: 20px; overflow: auto; background: #141f2b; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; background: #0b1118; padding: 12px; border-radius: 8px; }}
    h1 {{ font-size: 20px; margin-top: 0; }}
    p {{ line-height: 1.5; color: #c2d2e5; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="viewer"><img alt="Dense stereo point cloud preview" src="data:image/png;base64,{png_b64}" /></div>
    <div id="side">
      <h1>Dense Stereo Point Cloud</h1>
      <p>This self-contained HTML embeds the rendered PNG directly, so it opens reliably without external scripts.</p>
      <p>The raw stereo videos are upside-down. StereoDataLoader rotates both frames by 180° before rectification, and this visualization follows that upright view exactly once.</p>
      <pre>{metadata_json}</pre>
    </div>
  </div>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def default_output_paths(tag: str, suffix: str) -> tuple[Path, Path, Path]:
    stem = f"point_cloud_{tag}_{suffix}" if tag else f"point_cloud_{suffix}"
    return (
        RESULTS_DIR / f"{stem}.html",
        RESULTS_DIR / f"{stem}.png",
        RESULTS_DIR / f"{stem}.json",
    )


def compute_upright_check(rect_points: np.ndarray) -> bool:
    shoulder = rect_points[[5, 6]]
    hip = rect_points[[11, 12]]
    if not (np.isfinite(shoulder).all() and np.isfinite(hip).all()):
        return False
    return float(np.mean(shoulder[:, 1])) < float(np.mean(hip[:, 1]))


def render_one(
    frame_idx: int,
    label: str,
    dense_npz: dict[str, np.ndarray],
    representatives: dict[str, int],
    html_path: Path,
    png_path: Path,
    json_path: Path,
    max_points: int,
) -> dict:
    frame_l, frame_r = fetch_frame_pair(frame_idx)
    map1_l, map2_l, map1_r, map2_r, Q = compute_rectification()
    left_rect = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)
    disparity_map = compute_sgbm_disparity(left_rect, right_rect, build_sgbm())
    point_cloud, point_colors, total_points = build_point_cloud(disparity_map, left_rect, Q, max_points)
    skeleton_points = dense_npz["keypoints"][frame_idx]

    render_png(png_path, left_rect, disparity_map, point_cloud, point_colors, skeleton_points, frame_idx, label)
    metadata = {
        "frame_idx": int(frame_idx),
        "tag": label,
        "representative_frames": representatives,
        "npz_path": str(Path(args.npz_path)),
        "output_html": str(html_path),
        "output_png": str(png_path),
        "output_json": str(json_path),
        "point_count_total": int(total_points),
        "point_count_rendered": int(len(point_cloud)),
        "skeleton_valid_joints": int(np.sum(np.isfinite(skeleton_points).all(axis=1))),
        "sgbm": {
            "min_disparity": int(SGBM_MIN_DISPARITY),
            "num_disparities": int(SGBM_NUM_DISPARITIES),
            "block_size": int(SGBM_BLOCK_SIZE),
            "lookup_window": int(args.lookup_window),
        },
        "upright_handling": (
            "Raw stereo videos are upside-down. StereoDataLoader rotates both frames by "
            "180 degrees before this script rectifies them, so this visualization is upright."
        ),
        "upright_check_pass": bool(compute_upright_check(dense_npz["keypoints_left_rect"][frame_idx])),
        "html_uses_threejs_cdn": True,
    }
    write_html(html_path, png_path, metadata)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    global args
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dense_npz = load_dense_npz(Path(args.npz_path))
    representatives = select_representative_frames(dense_npz)

    if args.frame is not None:
        selections = {args.tag or "frame": int(args.frame)}
    elif args.category is not None:
        selections = {args.category: int(representatives[args.category])}
    else:
        selections = representatives

    summary = {"renders": []}
    for label, frame_idx in selections.items():
        default_html, default_png, default_json = default_output_paths(args.tag or label, label)
        html_path = Path(args.output_html) if args.output_html and len(selections) == 1 else default_html
        png_path = Path(args.output_png) if args.output_png and len(selections) == 1 else default_png
        json_path = Path(args.output_json) if args.output_json and len(selections) == 1 else default_json
        metadata = render_one(
            frame_idx,
            label,
            dense_npz,
            representatives,
            html_path,
            png_path,
            json_path,
            args.max_points,
        )
        summary["renders"].append(metadata)

    summary_path = RESULTS_DIR / (f"point_cloud_summary_{args.tag}.json" if args.tag else "point_cloud_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[dense-point-cloud] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
