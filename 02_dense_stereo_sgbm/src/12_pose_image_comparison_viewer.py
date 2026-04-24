#!/opt/anaconda3/envs/pose/bin/python
"""Generate a 2D camera-image viewer for SKT/Xsens pose comparison.

This viewer is intentionally point-cloud free. It projects both the SKT 3D
reconstruction and the aligned Xsens skeleton back to the rectified left camera
image, making calibration/neutral-pose deviations easier to inspect visually.

Outputs
-------
- <output_dir>/viewer_<tag>.html
- <output_dir>/metrics_<tag>.json
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
MANUAL_VIEWER_PATH = SCRIPT_DIR / "11_manual_agreement_viewer.py"

spec = importlib.util.spec_from_file_location("manual_agreement_viewer", MANUAL_VIEWER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import helpers from {MANUAL_VIEWER_PATH}")
manual = importlib.util.module_from_spec(spec)
sys.modules["manual_agreement_viewer"] = manual
spec.loader.exec_module(manual)

DEFAULT_OUTPUT_DIR = METHOD_DIR / "results" / "manual_agreement"
DEFAULT_TAG = "nt_pose_2d"
DEFAULT_FRAMES = [
    233, 234, 236, 238, 240, 242, 244,
    285, 1108, 1807, 1808, 1809, 1810, 2120, 2121,
]


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>2D Pose Agreement Viewer</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #eef1f5;
      color: #17182b;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .wrap {
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 1.25rem;
      letter-spacing: 0;
    }
    .sub {
      margin: 0 0 12px;
      color: #555b66;
      font-size: 0.9rem;
    }
    .layout {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .panel {
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 14px rgba(20, 26, 40, 0.08);
      padding: 12px;
    }
    .viewer {
      padding: 0;
      overflow: hidden;
      background: #111722;
    }
    .stage-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      min-height: 42px;
      padding: 10px 12px;
      color: #e9edf5;
      background: #111722;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      font-size: 0.86rem;
    }
    .stage {
      padding: 12px;
      background: #121923;
    }
    .single {
      display: block;
      width: 100%;
    }
    .compare-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      align-items: start;
    }
    .tile {
      min-width: 0;
    }
    .tile-title {
      margin-bottom: 6px;
      color: #dce4f2;
      font-size: 0.78rem;
      font-weight: 650;
      letter-spacing: 0;
    }
    canvas {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 6px;
      background: #0c1119;
    }
    .sec-title {
      margin: 9px 0 5px;
      color: #82858d;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    select {
      width: 100%;
      padding: 7px 8px;
      border: 1px solid #c9ccd2;
      border-radius: 6px;
      background: #fff;
      font-size: 13px;
    }
    .btn-row {
      display: flex;
      gap: 6px;
      margin-top: 7px;
    }
    button {
      min-height: 30px;
      border: 1px solid #b9bec8;
      border-radius: 6px;
      background: #f7f8fa;
      color: #151826;
      font-size: 12px;
      cursor: pointer;
    }
    .btn-row button {
      flex: 1;
    }
    button.active {
      border-color: #005baa;
      background: #005baa;
      color: #fff;
    }
    .mode-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .legend {
      display: grid;
      gap: 5px;
      margin-top: 4px;
      font-size: 0.82rem;
      color: #464b55;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .swatch {
      width: 26px;
      height: 5px;
      border-radius: 4px;
      flex: 0 0 auto;
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 4px 0;
      border-bottom: 1px solid #edf0f4;
      font-size: 0.84rem;
    }
    .metric-row:last-child {
      border-bottom: 0;
    }
    .mkey {
      color: #515762;
    }
    .mval {
      font-weight: 700;
    }
    .good { color: #238349; }
    .warn { color: #bf6b21; }
    .bad { color: #bd2637; }
    .note {
      color: #616773;
      font-size: 0.78rem;
      line-height: 1.45;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .compare-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>2D Pose Agreement Viewer: Camera / Xsens / SKT</h1>
    <p class="sub">Selected N/T-pose diagnostic frames</p>
    <div class="layout">
      <div class="panel">
        <div class="sec-title">Frame</div>
        <select id="frameSelect"></select>
        <div class="btn-row">
          <button id="prevBtn">Prev</button>
          <button id="playBtn">Play</button>
          <button id="nextBtn">Next</button>
        </div>

        <div class="sec-title">View</div>
        <div class="mode-grid">
          <button class="mode active" data-mode="compare">Compare</button>
          <button class="mode" data-mode="original">Original</button>
          <button class="mode" data-mode="xsens">Xsens</button>
          <button class="mode" data-mode="skt">SKT</button>
        </div>

        <div class="sec-title">Framing</div>
        <div class="btn-row">
          <button id="cropBtn" class="active">Person crop</button>
          <button id="fullBtn">Full frame</button>
        </div>

        <div class="sec-title">Layers</div>
        <div class="legend">
          <div class="legend-row"><span class="swatch" style="background:#3fb950"></span>Xsens GT</div>
          <div class="legend-row"><span class="swatch" style="background:#ff8a1c"></span>SKT reconstruction</div>
        </div>

        <div class="sec-title">Metrics</div>
        <div id="metrics"></div>

        <div class="sec-title">Note</div>
        <div class="note">
          Projection uses the rectified left camera frame so the skeletons align with the stereo reconstruction coordinate system.
        </div>
      </div>

      <div class="panel viewer">
        <div class="stage-head">
          <div id="titleText"></div>
          <div id="modeText"></div>
        </div>
        <div class="stage">
          <div id="singleWrap" class="single">
            <canvas id="singleCanvas"></canvas>
          </div>
          <div id="compareWrap" class="compare-grid">
            <div class="tile">
              <div class="tile-title">Original</div>
              <canvas id="canvasOriginal"></canvas>
            </div>
            <div class="tile">
              <div class="tile-title">Xsens GT</div>
              <canvas id="canvasXsens"></canvas>
            </div>
            <div class="tile">
              <div class="tile-title">SKT Reconstruction</div>
              <canvas id="canvasSkt"></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const frames = __FRAMES_JSON__;
let current = 0;
let mode = "compare";
let framing = "crop";
let timer = null;
const PLAY_MS = 520;
const imageCache = new Map();

function validPoint(p) {
  return p && Number.isFinite(p[0]) && Number.isFinite(p[1]);
}

function loadImage(src) {
  if (imageCache.has(src)) return imageCache.get(src);
  const promise = new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
  imageCache.set(src, promise);
  return promise;
}

function activeCrop(frame) {
  if (framing !== "crop" || !frame.crop) return [0, 0, frame.image_width, frame.image_height];
  return frame.crop;
}

function setCanvasSize(canvas, frame) {
  const crop = activeCrop(frame);
  canvas.width = crop[2];
  canvas.height = crop[3];
}

function drawBase(ctx, img, frame) {
  const crop = activeCrop(frame);
  ctx.clearRect(0, 0, crop[2], crop[3]);
  ctx.drawImage(img, crop[0], crop[1], crop[2], crop[3], 0, 0, crop[2], crop[3]);
}

function drawBadge(ctx, text, color) {
  ctx.save();
  ctx.font = "600 16px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  const w = ctx.measureText(text).width + 24;
  ctx.fillStyle = "rgba(8, 12, 18, 0.72)";
  ctx.fillRect(12, 12, w, 32);
  ctx.fillStyle = color;
  ctx.fillText(text, 24, 34);
  ctx.restore();
}

function drawPolylineSkeleton(ctx, points, edges, color, jointColor, alpha = 0.94) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const lineWidth = Math.max(5, Math.round(ctx.canvas.height * 0.012));
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  for (const [a, b] of edges) {
    const p0 = points[a];
    const p1 = points[b];
    if (!validPoint(p0) || !validPoint(p1)) continue;
    ctx.beginPath();
    ctx.moveTo(p0[0], p0[1]);
    ctx.lineTo(p1[0], p1[1]);
    ctx.stroke();
  }
  const radius = Math.max(4, Math.round(ctx.canvas.height * 0.009));
  ctx.fillStyle = jointColor;
  for (const p of points) {
    if (!validPoint(p)) continue;
    ctx.beginPath();
    ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawCocoBody(ctx, skt, showBadge = true) {
  if (!skt || !skt.points) return;
  const pts = skt.points;
  ctx.save();
  const torso = [pts[5], pts[6], pts[12], pts[11]];
  if (torso.every(validPoint)) {
    ctx.globalAlpha = 0.26;
    ctx.fillStyle = "#ff8a1c";
    ctx.beginPath();
    ctx.moveTo(torso[0][0], torso[0][1]);
    for (let i = 1; i < torso.length; i += 1) ctx.lineTo(torso[i][0], torso[i][1]);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
  drawPolylineSkeleton(ctx, pts, skt.edges, "#ff8a1c", "#ffd08a", 0.96);
  if (showBadge) drawBadge(ctx, "SKT reconstruction", "#ffb35f");
}

function drawXsensBody(ctx, gt, showBadge = true) {
  if (!gt || !gt.points) return;
  const pts = gt.points;
  ctx.save();
  const torsoNames = ["LeftShoulder", "RightShoulder", "RightUpperLeg", "LeftUpperLeg"];
  const torso = torsoNames.map((name) => pts[name]);
  if (torso.every(validPoint)) {
    ctx.globalAlpha = 0.25;
    ctx.fillStyle = "#3fb950";
    ctx.beginPath();
    ctx.moveTo(torso[0][0], torso[0][1]);
    for (let i = 1; i < torso.length; i += 1) ctx.lineTo(torso[i][0], torso[i][1]);
    ctx.closePath();
    ctx.fill();
  }
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.globalAlpha = 0.96;
  ctx.strokeStyle = "#3fb950";
  ctx.lineWidth = Math.max(5, Math.round(ctx.canvas.height * 0.011));
  for (const [a, b] of gt.links) {
    const p0 = pts[a];
    const p1 = pts[b];
    if (!validPoint(p0) || !validPoint(p1)) continue;
    ctx.beginPath();
    ctx.moveTo(p0[0], p0[1]);
    ctx.lineTo(p1[0], p1[1]);
    ctx.stroke();
  }
  const radius = Math.max(4, Math.round(ctx.canvas.height * 0.008));
  ctx.fillStyle = "#b8f2c0";
  for (const p of Object.values(pts)) {
    if (!validPoint(p)) continue;
    ctx.beginPath();
    ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    ctx.fill();
  }
  const head = pts.Head;
  const neck = pts.Neck;
  if (validPoint(head) && validPoint(neck)) {
    const hr = Math.max(8, Math.hypot(head[0] - neck[0], head[1] - neck[1]) * 0.45);
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = "#3fb950";
    ctx.beginPath();
    ctx.arc(head[0], head[1], hr, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  if (showBadge) drawBadge(ctx, "Xsens GT", "#8cf29a");
}

async function drawCanvas(canvas, frame, drawMode) {
  setCanvasSize(canvas, frame);
  const ctx = canvas.getContext("2d");
  const img = await loadImage(frame.image);
  drawBase(ctx, img, frame);
  const crop = activeCrop(frame);
  ctx.save();
  ctx.translate(-crop[0], -crop[1]);
  if (drawMode === "xsens") drawXsensBody(ctx, frame.xsens, false);
  if (drawMode === "skt") drawCocoBody(ctx, frame.skt, false);
  if (drawMode === "both") {
    drawXsensBody(ctx, frame.xsens, false);
    drawCocoBody(ctx, frame.skt, false);
  }
  ctx.restore();
  if (drawMode === "xsens") drawBadge(ctx, "Xsens GT", "#8cf29a");
  if (drawMode === "skt") drawBadge(ctx, "SKT reconstruction", "#ffb35f");
  if (drawMode === "both") drawBadge(ctx, "Xsens GT + SKT", "#e7edf7");
}

function metricClass(key, value) {
  if (value == null || !Number.isFinite(value)) return "";
  if (key === "skt_mae_deg") return value <= 15 ? "good" : value <= 25 ? "warn" : "bad";
  if (key.endsWith("_err_deg")) return value <= 15 ? "good" : value <= 30 ? "warn" : "bad";
  return "";
}

function fmt(value, suffix, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return value.toFixed(digits) + suffix;
}

function renderMetrics(frame) {
  const rows = [
    ["subject_time_s", "Subject time", " s", 2],
    ["skt_mae_deg", "SKT fair MAE", "°", 1],
    ["elbow_mean_err_deg", "Elbow mean error", "°", 1],
    ["left_elbow_err_deg", "Left elbow error", "°", 1],
    ["right_elbow_err_deg", "Right elbow error", "°", 1],
    ["left_vid_idx", "Left video frame", "", 0],
    ["right_vid_idx", "Right video frame", "", 0],
  ];
  document.getElementById("metrics").innerHTML = rows.map(([key, label, suffix, digits]) => {
    const value = frame.metrics[key];
    return `<div class="metric-row"><span class="mkey">${label}</span><span class="mval ${metricClass(key, value)}">${fmt(value, suffix, digits)}</span></div>`;
  }).join("");
}

function updateButtons() {
  document.querySelectorAll(".mode").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  document.getElementById("cropBtn").classList.toggle("active", framing === "crop");
  document.getElementById("fullBtn").classList.toggle("active", framing === "full");
  document.getElementById("playBtn").textContent = timer ? "Pause" : "Play";
}

async function render() {
  const frame = frames[current];
  document.getElementById("frameSelect").value = String(current);
  const t = frame.subject_time_s.toFixed(2);
  const mae = frame.metrics.skt_mae_deg == null ? "N/A" : frame.metrics.skt_mae_deg.toFixed(1) + "°";
  const elbow = frame.metrics.elbow_mean_err_deg == null ? "N/A" : frame.metrics.elbow_mean_err_deg.toFixed(1) + "°";
  document.getElementById("titleText").textContent = `idx ${frame.npz_idx} · t=${t}s · SKT-MAE=${mae} · elbow=${elbow}`;
  const frameLabel = framing === "crop" ? "Person crop" : "Full frame";
  document.getElementById("modeText").textContent = `${mode === "compare" ? "Compare" : mode.toUpperCase()} · ${frameLabel}`;
  renderMetrics(frame);
  updateButtons();

  const singleWrap = document.getElementById("singleWrap");
  const compareWrap = document.getElementById("compareWrap");
  if (mode === "compare") {
    singleWrap.style.display = "none";
    compareWrap.style.display = "grid";
    await Promise.all([
      drawCanvas(document.getElementById("canvasOriginal"), frame, "original"),
      drawCanvas(document.getElementById("canvasXsens"), frame, "xsens"),
      drawCanvas(document.getElementById("canvasSkt"), frame, "skt"),
    ]);
  } else {
    singleWrap.style.display = "block";
    compareWrap.style.display = "none";
    await drawCanvas(document.getElementById("singleCanvas"), frame, mode);
  }
}

function step(delta) {
  current = (current + delta + frames.length) % frames.length;
  render();
}

function togglePlay() {
  if (timer) {
    clearInterval(timer);
    timer = null;
    updateButtons();
    return;
  }
  timer = setInterval(() => step(1), PLAY_MS);
  updateButtons();
}

function populate() {
  const sel = document.getElementById("frameSelect");
  frames.forEach((frame, i) => {
    const opt = document.createElement("option");
    const mae = frame.metrics.skt_mae_deg == null ? "N/A" : frame.metrics.skt_mae_deg.toFixed(1) + "°";
    const elbow = frame.metrics.elbow_mean_err_deg == null ? "N/A" : frame.metrics.elbow_mean_err_deg.toFixed(1) + "°";
    opt.value = String(i);
    opt.textContent = `idx ${frame.npz_idx} · t=${frame.subject_time_s.toFixed(2)}s · MAE ${mae} · elbow ${elbow}`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", (event) => {
    current = Number(event.target.value);
    render();
  });
  document.getElementById("prevBtn").addEventListener("click", () => step(-1));
  document.getElementById("nextBtn").addEventListener("click", () => step(1));
  document.getElementById("playBtn").addEventListener("click", togglePlay);
  document.getElementById("cropBtn").addEventListener("click", () => {
    framing = "crop";
    render();
  });
  document.getElementById("fullBtn").addEventListener("click", () => {
    framing = "full";
    render();
  });
  document.querySelectorAll(".mode").forEach((btn) => {
    btn.addEventListener("click", () => {
      mode = btn.dataset.mode;
      render();
    });
  });
}

populate();
render();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a point-cloud-free 2D pose comparison viewer.")
    parser.add_argument("--skt-path", default=str(manual.DEFAULT_SKT_NPZ))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--frames",
        default=",".join(str(idx) for idx in DEFAULT_FRAMES),
        help="Comma-separated NPZ frame indices to render.",
    )
    parser.add_argument("--image-width", type=int, default=980, help="Embedded camera image width in pixels.")
    parser.add_argument("--jpeg-quality", type=int, default=84, help="Embedded JPEG quality.")
    return parser.parse_args()


def encode_frame(frame_bgr: np.ndarray, width: int, quality: int) -> Tuple[str, int, int, float]:
    """Encode a BGR frame as resized base64 JPEG."""
    h, w = frame_bgr.shape[:2]
    scale = width / float(w)
    out_h = int(round(h * scale))
    resized = cv2.resize(frame_bgr, (width, out_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG.")
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return data_url, width, out_h, scale


def scaled_points(points: np.ndarray, scale: float) -> List[List[Optional[float]]]:
    """Convert projected points to JSON-friendly scaled coordinates."""
    out: List[List[Optional[float]]] = []
    for point in points:
        if np.isfinite(point).all():
            out.append([round(float(point[0] * scale), 2), round(float(point[1] * scale), 2)])
        else:
            out.append([None, None])
    return out


def scaled_segment_points(
    cam_points: Dict[str, np.ndarray],
    p1_mat: np.ndarray,
    hw: Tuple[int, int],
    scale: float,
) -> Dict[str, List[Optional[float]]]:
    """Project and scale Xsens segment points for HTML rendering."""
    names = list(manual.XSENS_SEGMENTS)
    pts = np.full((len(names), 3), np.nan, dtype=np.float32)
    for i, name in enumerate(names):
        value = cam_points.get(name)
        if value is not None and np.isfinite(value).all():
            pts[i] = value
    projected = manual.project_to_rect(pts, p1_mat, hw)
    projected_scaled = scaled_points(projected, scale)
    return {name: projected_scaled[i] for i, name in enumerate(names)}


def person_crop_box(
    image_width: int,
    image_height: int,
    point_sets: List[np.ndarray],
    min_width: int = 280,
    min_height: int = 360,
) -> List[int]:
    """Build a stable crop around projected person joints in resized-image pixels."""
    valid_sets = []
    for pts in point_sets:
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr):
            valid_sets.append(arr)
    if not valid_sets:
        return [0, 0, int(image_width), int(image_height)]

    pts_all = np.concatenate(valid_sets, axis=0)
    x1, y1 = np.min(pts_all, axis=0)
    x2, y2 = np.max(pts_all, axis=0)
    width = max(float(x2 - x1), float(min_width))
    height = max(float(y2 - y1), float(min_height))
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)

    target_ratio = 0.74
    if width / height > target_ratio:
        height = width / target_ratio
    else:
        width = height * target_ratio

    width *= 1.18
    height *= 1.12
    x1 = max(0, int(round(cx - width / 2.0)))
    y1 = max(0, int(round(cy - height / 2.0)))
    x2 = min(image_width, int(round(cx + width / 2.0)))
    y2 = min(image_height, int(round(cy + height / 2.0)))

    if x2 - x1 < min_width:
        deficit = min_width - (x2 - x1)
        x1 = max(0, x1 - deficit // 2)
        x2 = min(image_width, x2 + deficit - deficit // 2)
    if y2 - y1 < min_height:
        deficit = min_height - (y2 - y1)
        y1 = max(0, y1 - deficit // 2)
        y2 = min(image_height, y2 + deficit - deficit // 2)

    return [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]


def parse_frame_indices(value: str, n_frames: int) -> List[int]:
    """Parse and validate requested NPZ frame indices."""
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("At least one frame index is required.")
    bad = [idx for idx in indices if idx < 0 or idx >= n_frames]
    if bad:
        raise ValueError(f"Frame indices out of range: {bad}")
    return indices


def metric_payload(
    subject_time_s: float,
    skt_mae_deg: Optional[float],
    errors: Dict[str, float],
    left_vid_idx: int,
    right_vid_idx: int,
) -> Dict[str, Optional[float]]:
    """Build per-frame metric payload."""
    left_elbow = errors.get("LeftElbow")
    right_elbow = errors.get("RightElbow")
    elbow_values = [v for v in (left_elbow, right_elbow) if v is not None and np.isfinite(v)]
    elbow_mean = float(np.mean(elbow_values)) if elbow_values else None
    return {
        "subject_time_s": round(float(subject_time_s), 3),
        "skt_mae_deg": round(float(skt_mae_deg), 2) if skt_mae_deg is not None else None,
        "elbow_mean_err_deg": round(float(elbow_mean), 2) if elbow_mean is not None else None,
        "left_elbow_err_deg": round(float(left_elbow), 2) if left_elbow is not None else None,
        "right_elbow_err_deg": round(float(right_elbow), 2) if right_elbow is not None else None,
        "left_vid_idx": float(left_vid_idx),
        "right_vid_idx": float(right_vid_idx),
    }


def build_frame_payloads(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict]]:
    """Build HTML frame payloads and compact metrics records."""
    print("Loading SKT NPZ...")
    skt_np = np.load(args.skt_path, allow_pickle=True)
    skt_kp = skt_np["keypoints"]
    skt_ts = skt_np["timestamps"].astype(float)
    subject_start = float(skt_ts[0])
    frame_indices = parse_frame_indices(args.frames, len(skt_kp))

    print("Building synchronized stereo-pair metadata...")
    sync_pairs = manual.build_synced_pairs(manual.DATA_DIR / "0_video_left.txt", manual.DATA_DIR / "1_video_right.txt")
    if not sync_pairs:
        raise RuntimeError("No synchronized stereo pairs found.")
    sync_ts = np.array([float(row["ts"]) for row in sync_pairs], dtype=np.float64)

    print("Loading Xsens GT and fair-angle interpolators...")
    gt = manual.load_gt(manual.MVNX_PATH)
    align = manual.load_alignment(manual.GT_JSON_PATH)
    fair_gt = manual.build_fair_gt_interpolators(str(manual.FAIR_GT_NPZ))
    offset_s = align["offset_s"] if align else 17.25

    cap_l = cv2.VideoCapture(str(manual.DATA_DIR / "0_video_left.avi"))
    cap_r = cv2.VideoCapture(str(manual.DATA_DIR / "1_video_right.avi"))
    ok, sample = cap_l.read()
    if not ok:
        raise RuntimeError("Cannot read left camera video.")
    img_hw = sample.shape[:2]
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    m1l, m2l, m1r, m2r, _, p1_mat = manual.setup_rectification(manual.PARAM_PATH, img_hw)

    payloads: List[Dict] = []
    metrics_records: List[Dict] = []
    for npz_idx in frame_indices:
        sync_idx = manual.npz_to_sync(npz_idx, sync_ts, skt_ts)
        meta = sync_pairs[sync_idx]
        left_vid_idx = int(meta["left_idx"])
        right_vid_idx = int(meta["right_idx"])
        frame_l, frame_r = manual.read_stereo_pair(left_vid_idx, right_vid_idx, cap_l, cap_r)
        left_rect = cv2.remap(frame_l, m1l, m2l, cv2.INTER_LINEAR)
        _ = cv2.remap(frame_r, m1r, m2r, cv2.INTER_LINEAR)

        image, image_width, image_height, scale = encode_frame(left_rect, args.image_width, args.jpeg_quality)
        subject_time_s = float(skt_ts[npz_idx] - subject_start)
        gt_t = subject_time_s - offset_s
        skt_pose = skt_kp[npz_idx]
        skt_proj = manual.project_to_rect(skt_pose, p1_mat, img_hw)
        skt_points = scaled_points(skt_proj, scale)
        skt_proj_scaled = np.asarray(skt_points, dtype=np.float32)
        errors = manual.compute_frame_angle_errors(skt_pose, gt_t, fair_gt)
        skt_mae = manual.compute_frame_mae(skt_pose, gt_t, fair_gt)
        metrics = metric_payload(subject_time_s, skt_mae, errors, left_vid_idx, right_vid_idx)

        xsens_points: Dict[str, List[Optional[float]]] = {}
        xsens_proj_scaled = np.empty((0, 2), dtype=np.float32)
        if gt is not None and align is not None:
            gt_pose = manual.gt_pose_in_cam(subject_time_s, gt, align)
            if gt_pose is not None:
                xsens_points = scaled_segment_points(gt_pose["_cam"], p1_mat, img_hw, scale)
                xsens_proj_scaled = np.asarray(list(xsens_points.values()), dtype=np.float32)

        crop = person_crop_box(image_width, image_height, [skt_proj_scaled, xsens_proj_scaled])

        frame_payload = {
            "npz_idx": int(npz_idx),
            "sync_idx": int(sync_idx),
            "subject_time_s": round(subject_time_s, 3),
            "image": image,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "crop": crop,
            "skt": {
                "points": skt_points,
                "edges": manual.COCO_EDGES,
            },
            "xsens": {
                "points": xsens_points,
                "links": manual.XSENS_LINKS,
            },
            "metrics": metrics,
        }
        payloads.append(frame_payload)
        metrics_records.append(
            {
                "npz_idx": int(npz_idx),
                "sync_idx": int(sync_idx),
                **metrics,
            }
        )
        print(
            f"  idx {npz_idx:4d} -> t={subject_time_s:7.2f}s "
            f"SKT-MAE={metrics['skt_mae_deg']} elbow={metrics['elbow_mean_err_deg']}"
        )

    cap_l.release()
    cap_r.release()
    return payloads, metrics_records


def write_outputs(payloads: List[Dict], metrics: List[Dict], out_dir: Path, tag: str) -> Tuple[Path, Path]:
    """Write self-contained HTML and metrics JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__FRAMES_JSON__", json.dumps(payloads))
    html_path = out_dir / f"viewer_{tag}.html"
    json_path = out_dir / f"metrics_{tag}.json"
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps({"frames": metrics}, indent=2), encoding="utf-8")
    return html_path, json_path


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    payloads, metrics = build_frame_payloads(args)
    html_path, json_path = write_outputs(payloads, metrics, Path(args.output_dir), args.tag)
    print(f"\n[saved] {html_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
