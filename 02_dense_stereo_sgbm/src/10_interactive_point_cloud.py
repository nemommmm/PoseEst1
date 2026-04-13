#!/opt/anaconda3/envs/pose/bin/python
"""Generate an interactive dense-stereo point cloud HTML for Direction B."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from utils import StereoDataLoader


DATA_DIR = PROJECT_ROOT / "2025_Ergonomics_Data"
RESULTS_DIR = METHOD_DIR / "results"
PARAM_PATH = SHARED_DIR / "camera_params.npz"
DEFAULT_REF_NPZ = PROJECT_ROOT / "01_stereo_triangulation" / "results" / "yolo_3d_raw.npz"
DEFAULT_SUMMARY = RESULTS_DIR / "point_cloud_summary_audit.json"

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


def to_display_coords(points: np.ndarray) -> np.ndarray:
    """Convert OpenCV camera coordinates to a human-friendly viewer frame.

    OpenCV dense stereo uses X-right, Y-down, Z-forward. For interactive viewing
    we display:
      X_display = X_right
      Y_display = Z_depth
      Z_display = -Y_up
    so that people appear upright.
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 3:
        return pts
    return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Create an interactive dense stereo point-cloud HTML.")
    parser.add_argument("--reference-npz", default=str(DEFAULT_REF_NPZ), help="Reference pose NPZ for representative frames and skeleton overlay.")
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY), help="Optional prior point-cloud summary JSON for frame selection.")
    parser.add_argument("--output-html", default=str(RESULTS_DIR / "interactive_pointcloud.html"), help="Output HTML path.")
    parser.add_argument("--output-json", default=str(RESULTS_DIR / "interactive_pointcloud.json"), help="Output metadata path.")
    parser.add_argument("--max-points", type=int, default=30000, help="Max points per frame for browser rendering.")
    parser.add_argument("--frames", default="", help="Comma-separated explicit frame indices (overrides auto selection).")
    return parser.parse_args()


def build_sgbm() -> cv2.StereoSGBM:
    """Create a StereoSGBM matcher."""
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


def compute_rectification() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stereo rectification maps and Q matrix."""
    calibration = np.load(PARAM_PATH)
    mtx_l, dist_l = calibration["mtx_l"], calibration["dist_l"]
    mtx_r, dist_r = calibration["mtx_r"], calibration["dist_r"]
    rot, trans = calibration["R"], calibration["T"]
    cap = cv2.VideoCapture(str(DATA_DIR / "0_video_left.avi"))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read left video for frame size.")
    h, w = frame.shape[:2]
    r1, r2, p1, p2, q_mat, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, (w, h), rot, trans, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, r1, p1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, r2, p2, (w, h), cv2.CV_32FC1)
    return map1_l, map2_l, map1_r, map2_r, q_mat


def fetch_frame_pair(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one synchronized stereo pair."""
    loader = StereoDataLoader(
        str(DATA_DIR / "0_video_left.avi"),
        str(DATA_DIR / "1_video_right.avi"),
        str(DATA_DIR / "0_video_left.txt"),
        str(DATA_DIR / "1_video_right.txt"),
    )
    current = 0
    try:
        while True:
            frame_l, frame_r, _, _ = loader.get_next_pair()
            if frame_l is None:
                raise IndexError(f"Frame {frame_idx} exceeds stereo stream length.")
            if current == frame_idx:
                return frame_l, frame_r
            current += 1
    finally:
        loader.release()


def compute_disparity(frame_l: np.ndarray, frame_r: np.ndarray, map1_l, map2_l, map1_r, map2_r, sgbm) -> tuple[np.ndarray, np.ndarray]:
    """Rectify frames and compute SGBM disparity."""
    left_rect = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)
    gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    disparity = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disparity[disparity < 0] = np.nan
    return left_rect, disparity


def build_point_cloud(disparity_map: np.ndarray, color_bgr: np.ndarray, q_mat: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Reproject disparity to 3D and downsample for browser rendering."""
    disparity_filled = np.nan_to_num(disparity_map, nan=0.0).astype(np.float32)
    points_3d = cv2.reprojectImageTo3D(disparity_filled, q_mat)
    mask = np.isfinite(disparity_map) & (disparity_map >= SGBM_MIN_DISPARITY)
    mask &= np.isfinite(points_3d).all(axis=2)
    z_vals = points_3d[:, :, 2]
    mask &= (z_vals > 30.0) & (z_vals < 1200.0)
    all_points = points_3d[mask].astype(np.float32)
    all_colors = color_bgr[mask][:, ::-1].astype(np.uint8)
    total_count = int(len(all_points))
    if total_count > max_points:
        rng = np.random.default_rng(0)
        select = rng.choice(total_count, size=max_points, replace=False)
        all_points = all_points[select]
        all_colors = all_colors[select]
    return all_points, all_colors, total_count


def select_frames(args: argparse.Namespace, reference_npz: dict[str, np.ndarray]) -> dict[str, int]:
    """Choose representative frame indices."""
    if args.frames.strip():
        values = [int(token.strip()) for token in args.frames.split(",") if token.strip()]
        return {f"frame_{idx}": idx for idx in values}
    if os.path.isfile(args.summary_json):
        with open(args.summary_json, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        chosen = {}
        for render in summary.get("renders", []):
            chosen[render["tag"]] = int(render["frame_idx"])
        if chosen:
            return chosen
    keypoints = reference_npz["keypoints"]
    valid = np.isfinite(keypoints).all(axis=2)
    valid_ratio = np.mean(valid, axis=1)
    good_idx = int(np.argmax(valid_ratio))
    low_idx = int(np.argmin(valid_ratio))
    middle_idx = int(np.nanargmin(np.abs(valid_ratio - np.nanmedian(valid_ratio))))
    return {"good": good_idx, "bad": middle_idx, "low_coverage": low_idx}


def skeleton_payload(points: np.ndarray) -> dict[str, list]:
    """Convert one 17-joint skeleton into JSON-safe payload."""
    display_points = to_display_coords(np.asarray(points, dtype=np.float32))
    return {
        "points": display_points.tolist(),
        "edges": COCO_EDGES,
    }


def make_plotly_html(frames: list[dict]) -> str:
    """Create a self-contained Plotly HTML viewer."""
    frames_json = json.dumps(frames)
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Direction B Interactive Point Cloud</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f6f7f8; color: #111; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
    .row {{ display: grid; grid-template-columns: 280px 1fr; gap: 20px; }}
    .card {{ background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 16px; }}
    #plot {{ width: 100%; height: 760px; }}
    select {{ width: 100%; padding: 8px; font-size: 14px; }}
    pre {{ white-space: pre-wrap; font-size: 13px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Direction B: Interactive Dense Stereo Point Cloud</h1>
    <p>Raw stereo videos are upside-down; frames are rotated once inside <code>StereoDataLoader</code> before rectification, so the visualization is already upright.</p>
    <div class=\"row\">
      <div class=\"card\">
        <label for=\"frameSelect\"><strong>Representative frame</strong></label>
        <select id=\"frameSelect\"></select>
        <h3>Metadata</h3>
        <pre id=\"metaBox\"></pre>
      </div>
      <div class=\"card\">
        <div id=\"plot\"></div>
      </div>
    </div>
  </div>
  <script>
    const payload = {frames_json};
    const select = document.getElementById('frameSelect');
    const metaBox = document.getElementById('metaBox');
    payload.forEach((frame, idx) => {{
      const option = document.createElement('option');
      option.value = idx;
      option.textContent = `${{frame.tag}} (frame=${{frame.frame_idx}})`;
      select.appendChild(option);
    }});
    function buildTraces(frame) {{
      const points = frame.points;
      const cloud = {{
        type: 'scatter3d',
        mode: 'markers',
        x: points.map(p => p[0]),
        y: points.map(p => p[1]),
        z: points.map(p => p[2]),
        marker: {{
          size: 1.8,
          color: frame.colors.map(c => `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`),
          opacity: 0.85
        }},
        name: 'Dense point cloud',
        hoverinfo: 'skip'
      }};
      const traces = [cloud];
      const skel = frame.skeleton;
      skel.edges.forEach((edge, edgeIdx) => {{
        const p0 = skel.points[edge[0]];
        const p1 = skel.points[edge[1]];
        if (!Number.isFinite(p0[0]) || !Number.isFinite(p1[0])) {{
          return;
        }}
        traces.push({{
          type: 'scatter3d',
          mode: 'lines',
          x: [p0[0], p1[0]],
          y: [p0[1], p1[1]],
          z: [p0[2], p1[2]],
          line: {{ color: '#ff7a18', width: 6 }},
          name: edgeIdx === 0 ? 'Reference skeleton' : '',
          showlegend: edgeIdx === 0,
          hoverinfo: 'skip'
        }});
      }});
      traces.push({{
        type: 'scatter3d',
        mode: 'markers',
        x: skel.points.map(p => p[0]),
        y: skel.points.map(p => p[1]),
        z: skel.points.map(p => p[2]),
        marker: {{ size: 4, color: '#00bcd4' }},
        name: 'Skeleton joints',
        hoverinfo: 'skip'
      }});
      return traces;
    }}
    function renderFrame(idx) {{
      const frame = payload[idx];
      Plotly.newPlot('plot', buildTraces(frame), {{
        scene: {{
          xaxis: {{ title: 'X (cm)' }},
          yaxis: {{ title: 'Y (cm)' }},
          zaxis: {{ title: 'Z (cm)' }},
          aspectmode: 'data',
          camera: {{ eye: {{ x: 1.55, y: -1.65, z: 0.85 }} }}
        }},
        margin: {{ l: 0, r: 0, b: 0, t: 40 }},
        title: `Frame ${{frame.frame_idx}} · ${{frame.tag}}`
      }}, {{responsive: true}});
      metaBox.textContent = JSON.stringify(frame.meta, null, 2);
    }}
    select.addEventListener('change', (event) => renderFrame(Number(event.target.value)));
    renderFrame(0);
  </script>
</body>
</html>"""


def main() -> None:
    """Generate the HTML viewer."""
    args = parse_args()
    reference_npz = np.load(args.reference_npz, allow_pickle=True)
    selected_frames = select_frames(args, reference_npz)

    map1_l, map2_l, map1_r, map2_r, q_mat = compute_rectification()
    sgbm = build_sgbm()
    payload_frames = []

    for tag, frame_idx in selected_frames.items():
        frame_l, frame_r = fetch_frame_pair(frame_idx)
        left_rect, disparity = compute_disparity(frame_l, frame_r, map1_l, map2_l, map1_r, map2_r, sgbm)
        points, colors, total_count = build_point_cloud(disparity, left_rect, q_mat, args.max_points)
        display_points = to_display_coords(points)
        skeleton = skeleton_payload(reference_npz["keypoints"][frame_idx])
        payload_frames.append({
            "tag": tag,
            "frame_idx": frame_idx,
            "points": display_points.tolist(),
            "colors": colors.tolist(),
            "skeleton": skeleton,
            "meta": {
                "frame_idx": frame_idx,
                "tag": tag,
                "point_count_total": total_count,
                "point_count_rendered": int(len(points)),
                "skeleton_valid_joints": int(np.isfinite(reference_npz["keypoints"][frame_idx]).all(axis=1).sum()),
                "sgbm": {
                    "min_disparity": SGBM_MIN_DISPARITY,
                    "num_disparities": SGBM_NUM_DISPARITIES,
                    "block_size": SGBM_BLOCK_SIZE,
                    "lookup_window": LOOKUP_WINDOW,
                },
                "upright_handling": "StereoDataLoader rotates raw upside-down videos by 180 degrees once before rectification.",
                "display_axes": {
                    "x": "Right (X)",
                    "y": "Depth (Z)",
                    "z": "Up (-Y)",
                },
            },
        })

    output_html = Path(args.output_html)
    output_json = Path(args.output_json)
    output_html.write_text(make_plotly_html(payload_frames), encoding="utf-8")
    output_json.write_text(json.dumps({"frames": payload_frames}, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_html": str(output_html),
        "output_json": str(output_json),
        "frame_count": len(payload_frames),
    }, indent=2))


if __name__ == "__main__":
    main()
