"""Render a simple raw-video plus 3D skeleton comparison video."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.angles import COCO17_NAMES, compute_angle_sequence
from common.config import resolve_path, section
from common.dataset import load_method_keypoints
from stereo_loader import StereoFrameReader

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = {"SKT": "#ff7a18", "FastSAM3D": "#2196F3", "Merge": "#8e44ad"}


def canonicalize(kp: np.ndarray) -> np.ndarray:
    """Center skeleton at pelvis and map to a stable anatomical view."""
    out = np.full_like(kp, np.nan)
    for idx, pose in enumerate(kp):
        if not (np.isfinite(pose[5]).all() and np.isfinite(pose[6]).all() and np.isfinite(pose[11]).all() and np.isfinite(pose[12]).all()):
            continue
        pelvis = 0.5 * (pose[11] + pose[12])
        neck = 0.5 * (pose[5] + pose[6])
        up = neck - pelvis
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            continue
        up = up / up_norm
        right = pose[12] - pose[11]
        right = right - np.dot(right, up) * up
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            continue
        right = right / right_norm
        forward = np.cross(up, right)
        basis = np.vstack([right, forward, up])
        out[idx] = (basis @ (pose - pelvis).T).T
    return out


def fig_to_bgr(fig) -> np.ndarray:
    """Convert a Matplotlib figure to BGR image."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_skeleton_panel(methods: dict[str, np.ndarray], frame_idx: int, angles: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """Render one white-background skeleton panel."""
    fig = plt.figure(figsize=(6, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.view_init(elev=15, azim=-70)
    lim = 90
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-20, 160)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.25)
    for name, kp in methods.items():
        pose = kp[frame_idx]
        color = COLORS.get(name, "#444444")
        for a, b in EDGES:
            if np.isfinite(pose[a]).all() and np.isfinite(pose[b]).all():
                ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]], [pose[a, 2], pose[b, 2]], color=color, linewidth=2.0)
        finite = np.isfinite(pose).all(axis=1)
        ax.scatter(pose[finite, 0], pose[finite, 1], pose[finite, 2], color=color, s=18, label=name)
    label_lines = []
    for name in methods:
        le = angles[name]["LeftElbow"][frame_idx] if "LeftElbow" in angles[name] else np.nan
        re = angles[name]["RightElbow"][frame_idx] if "RightElbow" in angles[name] else np.nan
        label_lines.append(f"{name}: L {le:.1f} deg | R {re:.1f} deg")
    ax.text2D(0.03, 0.96, "\n".join(label_lines), transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc"))
    ax.legend(loc="lower left")
    fig.tight_layout()
    img = fig_to_bgr(fig)
    plt.close(fig)
    return img


def render_video(config: dict, run_dir: Path) -> Path:
    """Render comparison video for configured systems."""
    video_cfg = section(config, "video")
    dataset = section(config, "dataset")
    time_s, meta, methods = load_method_keypoints(config, run_dir)
    selected = [name for name in video_cfg.get("compare_systems", ["SKT", "FastSAM3D"]) if name in methods]
    if not selected:
        raise RuntimeError("No configured video compare systems are available.")
    canon = {name: canonicalize(methods[name]) for name in selected}
    angles = {name: compute_angle_sequence(methods[name], ["LeftElbow", "RightElbow"]) for name in selected}
    start = int(np.searchsorted(time_s, float(video_cfg.get("start_time_s", 0.0))))
    end = int(np.searchsorted(time_s, float(video_cfg.get("end_time_s", min(time_s[-1], 60.0)))))
    end = max(start + 1, min(end, len(time_s) - 1))
    reader = StereoFrameReader(
        resolve_path(dataset.get("left_video"), must_exist=True),
        resolve_path(dataset.get("right_video"), must_exist=True),
        meta["synced"],
        rotate_180=bool(dataset.get("rotate_180", False)),
    )
    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pose_comparison.mp4"
    writer = None
    try:
        for idx in range(start, end + 1):
            ok, frame_l, _ = reader.read_synced(idx)
            if not ok or frame_l is None:
                continue
            frame_l = cv2.resize(frame_l, (720, 540))
            panel = render_skeleton_panel(canon, idx, angles)
            panel = cv2.resize(panel, (720, 540))
            canvas = np.hstack([frame_l, panel])
            cv2.putText(canvas, f"t={time_s[idx]:.2f}s frame={idx}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, float(video_cfg.get("fps", 12.5)), (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
    finally:
        reader.release()
        if writer is not None:
            writer.release()
    print(f"[video] saved {out_path}")
    return out_path
