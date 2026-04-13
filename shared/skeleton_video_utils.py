"""Shared utilities for skeleton comparison videos.

This module provides:
1. Xsens skeleton loading and interpolation
2. Rigid alignment helpers
3. Matplotlib-based 3D overlay rendering to MP4

All coordinates are assumed to be in centimetres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils_mvnx import MvnxParser


COCO_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

XSENS_SEGMENTS_TO_LOAD: list[str] = [
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
]

XSENS_LINKS: list[tuple[str, str]] = [
    ("Pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"), ("T8", "Neck"), ("Neck", "Head"),
    ("T8", "RightShoulder"), ("RightShoulder", "RightUpperArm"), ("RightUpperArm", "RightForeArm"),
    ("RightForeArm", "RightHand"), ("T8", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"), ("Pelvis", "RightUpperLeg"),
    ("RightUpperLeg", "RightLowerLeg"), ("RightLowerLeg", "RightFoot"), ("Pelvis", "LeftUpperLeg"),
    ("LeftUpperLeg", "LeftLowerLeg"), ("LeftLowerLeg", "LeftFoot"),
]

GT_LIMB_LENGTHS = np.array([38.6, 39.8, 40.3, 39.5], dtype=np.float64)


@dataclass
class XsensSkeleton:
    """Interpolated Xsens skeleton data."""

    timestamps: np.ndarray
    interpolators: dict[str, Callable[[float], np.ndarray]]


def kabsch_transform(points_src: np.ndarray, points_tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rigid transform aligning points_src onto points_tgt."""
    mask = np.isfinite(points_src).all(axis=1) & np.isfinite(points_tgt).all(axis=1)
    src = points_src[mask]
    tgt = points_tgt[mask]
    if len(src) < 10:
        return np.eye(3), np.zeros(3)
    centroid_src = np.mean(src, axis=0)
    centroid_tgt = np.mean(tgt, axis=0)
    centered_src = src - centroid_src
    centered_tgt = tgt - centroid_tgt
    h_mat = centered_src.T @ centered_tgt
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[2, :] *= -1
        rot = vt_mat.T @ u_mat.T
    trans = centroid_tgt - rot @ centroid_src
    return rot, trans


def calculate_leg_limb_error(points: np.ndarray) -> np.ndarray:
    """Calculate lower-limb length error against GT reference lengths."""
    lengths = np.vstack([
        np.linalg.norm(points[:, 11] - points[:, 13], axis=1),
        np.linalg.norm(points[:, 12] - points[:, 14], axis=1),
        np.linalg.norm(points[:, 13] - points[:, 15], axis=1),
        np.linalg.norm(points[:, 14] - points[:, 16], axis=1),
    ]).T
    return np.sum(np.abs(lengths - GT_LIMB_LENGTHS), axis=1)


def load_xsens_skeleton(mvnx_path: str | Path) -> XsensSkeleton:
    """Load Xsens segment trajectories and build per-segment interpolators."""
    parser = MvnxParser(str(mvnx_path))
    parser.parse()
    xsens_ts = parser.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    interps: dict[str, Callable[[float], np.ndarray]] = {}
    for name in XSENS_SEGMENTS_TO_LOAD:
        seg_data = parser.get_segment_data(name)
        if seg_data is None:
            continue
        interps[name] = interp1d(
            xsens_ts,
            seg_data[xidx],
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    return XsensSkeleton(timestamps=xsens_ts, interpolators=interps)


def xsens_pose_at(xsens: XsensSkeleton, target_t: float) -> dict[str, np.ndarray]:
    """Sample all Xsens segments at one timestamp."""
    pose = {}
    for name, interp in xsens.interpolators.items():
        value = interp(target_t)
        if np.isfinite(value).all():
            pose[name] = value
    return pose


def collect_anchor_pairs(
    subject_points: np.ndarray,
    subject_ts: np.ndarray,
    xsens: XsensSkeleton,
    offset_s: float,
    anchor_mapping: dict[int, str],
    elite_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect subject/GT anchor pairs across elite frames for rigid alignment."""
    src_points: list[np.ndarray] = []
    tgt_points: list[np.ndarray] = []
    for idx in elite_indices:
        target_t = float(subject_ts[idx] - offset_s)
        xsens_pose = xsens_pose_at(xsens, target_t)
        if not xsens_pose:
            continue
        frame_points = subject_points[idx]
        for subject_idx, seg_name in anchor_mapping.items():
            if subject_idx >= len(frame_points) or seg_name not in xsens_pose:
                continue
            src = frame_points[subject_idx]
            tgt = xsens_pose[seg_name]
            if np.isfinite(src).all() and np.isfinite(tgt).all():
                src_points.append(src)
                tgt_points.append(tgt)
    if not src_points:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.asarray(src_points, dtype=np.float64), np.asarray(tgt_points, dtype=np.float64)


def align_subject_points(
    subject_points: np.ndarray,
    subject_ts: np.ndarray,
    xsens: XsensSkeleton,
    offset_s: float,
    anchor_mapping: dict[int, str],
    top_k: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly align a whole subject sequence to Xsens using elite frames."""
    errors = calculate_leg_limb_error(subject_points)
    finite_idx = np.where(np.isfinite(errors))[0]
    if finite_idx.size == 0:
        return subject_points.copy(), np.eye(3), np.zeros(3)
    elite = finite_idx[np.argsort(errors[finite_idx])[:top_k]]
    src, tgt = collect_anchor_pairs(subject_points, subject_ts, xsens, offset_s, anchor_mapping, elite)
    rot, trans = kabsch_transform(src, tgt)
    flat = subject_points.reshape(-1, 3)
    aligned = (rot @ flat.T).T + trans
    return aligned.reshape(subject_points.shape), rot, trans


def frame_joint_distance(
    subject_frame: np.ndarray,
    xsens_pose: dict[str, np.ndarray],
    anchor_mapping: dict[int, str],
) -> float:
    """Mean anchor-point distance for one frame."""
    distances = []
    for subject_idx, seg_name in anchor_mapping.items():
        if subject_idx >= len(subject_frame) or seg_name not in xsens_pose:
            continue
        src = subject_frame[subject_idx]
        tgt = xsens_pose[seg_name]
        if np.isfinite(src).all() and np.isfinite(tgt).all():
            distances.append(float(np.linalg.norm(src - tgt)))
    return float(np.mean(distances)) if distances else float("nan")


def draw_pose_edges(ax, points: np.ndarray, edges: list[tuple[int, int]], color: str, linewidth: float) -> None:
    """Draw skeleton edges for one pose."""
    finite = np.isfinite(points).all(axis=1)
    for start_idx, end_idx in edges:
        if start_idx < len(points) and end_idx < len(points) and finite[start_idx] and finite[end_idx]:
            seg = points[[start_idx, end_idx]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=linewidth)


def draw_xsens_pose(ax, pose: dict[str, np.ndarray], color: str = "black", linewidth: float = 1.5) -> None:
    """Draw Xsens skeleton links."""
    for parent, child in XSENS_LINKS:
        if parent in pose and child in pose:
            p0 = pose[parent]
            p1 = pose[child]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=linewidth, alpha=0.75)


def render_comparison_video(
    subject_points: np.ndarray,
    subject_ts: np.ndarray,
    subject_edges: list[tuple[int, int]],
    anchor_mapping: dict[int, str],
    output_mp4: str | Path,
    output_json: str | Path,
    mvnx_path: str | Path,
    offset_s: float,
    subject_label: str,
    subject_color: str,
    fps: float,
    title: str,
    follow_radius_cm: float = 100.0,
    max_frames: int | None = None,
    frame_step: int = 1,
) -> dict:
    """Render an overlay video of subject skeleton and Xsens GT."""
    xsens = load_xsens_skeleton(mvnx_path)
    aligned_points, rot, trans = align_subject_points(
        subject_points,
        subject_ts,
        xsens,
        offset_s,
        anchor_mapping,
    )

    output_mp4 = Path(output_mp4)
    output_json = Path(output_json)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    rendered = 0
    joint_distances: list[float] = []
    pelvis_distances: list[float] = []

    for frame_idx in range(0, len(subject_ts), max(1, frame_step)):
        if max_frames is not None and rendered >= max_frames:
            break
        target_t = float(subject_ts[frame_idx] - offset_s)
        xsens_pose = xsens_pose_at(xsens, target_t)
        if "Pelvis" not in xsens_pose:
            continue
        subject_pose = aligned_points[frame_idx]
        pelvis_candidates = [idx for idx, name in anchor_mapping.items() if name in {"LeftUpperLeg", "RightUpperLeg"}]
        pelvis_center = np.nanmean(subject_pose[pelvis_candidates], axis=0) if pelvis_candidates else np.full(3, np.nan)
        if not np.isfinite(pelvis_center).all():
            pelvis_center = subject_pose[np.isfinite(subject_pose).all(axis=1)].mean(axis=0)

        joint_dist = frame_joint_distance(subject_pose, xsens_pose, anchor_mapping)
        joint_distances.append(joint_dist)
        pelvis_distances.append(float(np.linalg.norm(pelvis_center - xsens_pose["Pelvis"])) if np.isfinite(pelvis_center).all() else np.nan)

        ax.cla()
        draw_xsens_pose(ax, xsens_pose, color="black", linewidth=1.6)
        draw_pose_edges(ax, subject_pose, subject_edges, color=subject_color, linewidth=2.0)

        finite_subject = np.isfinite(subject_pose).all(axis=1)
        if np.any(finite_subject):
            pts = subject_pose[finite_subject]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=subject_color, s=18, alpha=0.9)

        gt_pts = np.array(list(xsens_pose.values()), dtype=np.float64)
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c="black", s=10, alpha=0.5)

        center = xsens_pose["Pelvis"]
        ax.set_xlim(center[0] - follow_radius_cm, center[0] + follow_radius_cm)
        ax.set_ylim(center[1] - follow_radius_cm, center[1] + follow_radius_cm)
        ax.set_zlim(center[2] - follow_radius_cm, center[2] + follow_radius_cm)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Z (cm)")
        ax.view_init(elev=18, azim=-62)
        ax.set_title(
            f"{title}\n"
            f"t={subject_ts[frame_idx]:.2f}s  gt={target_t:.2f}s  "
            f"joint={joint_dist:.1f} cm  pelvis={pelvis_distances[-1]:.1f} cm",
            fontsize=11,
        )
        ax.plot([], [], [], color="black", label="Xsens GT")
        ax.plot([], [], [], color=subject_color, label=subject_label)
        ax.legend(loc="upper right")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        writer.write(bgr)
        rendered += 1

    writer.release()
    plt.close(fig)

    metadata = {
        "output_mp4": str(output_mp4),
        "frames_rendered": rendered,
        "fps": fps,
        "frame_step": frame_step,
        "offset_seconds": offset_s,
        "subject_label": subject_label,
        "mean_joint_distance_cm": float(np.nanmean(joint_distances)) if joint_distances else np.nan,
        "mean_pelvis_distance_cm": float(np.nanmean(pelvis_distances)) if pelvis_distances else np.nan,
        "rotation_matrix": rot.tolist(),
        "translation_cm": trans.tolist(),
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
