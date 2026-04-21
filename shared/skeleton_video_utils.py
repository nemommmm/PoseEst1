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
from typing import Any, Callable

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
    time_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect subject/GT anchor pairs across elite frames for rigid alignment."""
    src_points: list[np.ndarray] = []
    tgt_points: list[np.ndarray] = []
    for idx in elite_indices:
        target_t = float(time_scale * subject_ts[idx] - offset_s)
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
    time_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly align a whole subject sequence to Xsens using elite frames."""
    errors = calculate_leg_limb_error(subject_points)
    finite_idx = np.where(np.isfinite(errors))[0]
    if finite_idx.size == 0:
        return subject_points.copy(), np.eye(3), np.zeros(3)
    elite = finite_idx[np.argsort(errors[finite_idx])[:top_k]]
    src, tgt = collect_anchor_pairs(
        subject_points,
        subject_ts,
        xsens,
        offset_s,
        anchor_mapping,
        elite,
        time_scale=time_scale,
    )
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


def _render_overlay_frame(
    ax,
    title: str,
    subject_pose: np.ndarray,
    xsens_pose: dict[str, np.ndarray],
    subject_edges: list[tuple[int, int]],
    subject_label: str,
    subject_color: str,
    follow_radius_cm: float,
    overlay_lines: list[str] | None = None,
) -> None:
    """Render a single comparison frame onto an existing 3D axis."""
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
    ax.set_title(title, fontsize=12, pad=18)
    ax.plot([], [], [], color="black", label="Xsens GT")
    ax.plot([], [], [], color=subject_color, label=subject_label)
    ax.legend(loc="upper right")

    if overlay_lines:
        ax.text2D(
            0.02,
            0.98,
            "\n".join(overlay_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.88,
                "edgecolor": "#BBBBBB",
            },
        )


def _select_snapshots(
    frame_records: list[dict[str, Any]],
    score_key: str,
    good_count: int,
    bad_count: int,
    min_gap_s: float,
) -> list[dict[str, Any]]:
    """Select low-score and high-score frames as representative snapshots."""

    def _pick_candidates(candidates: list[dict[str, Any]], reverse: bool, count: int, used: set[int]) -> list[dict[str, Any]]:
        chosen: list[dict[str, Any]] = []
        ordered = sorted(
            candidates,
            key=lambda rec: float(rec.get(score_key, np.nan)),
            reverse=reverse,
        )
        for rec in ordered:
            frame_idx = int(rec["frame_idx"])
            score = float(rec.get(score_key, np.nan))
            if frame_idx in used or not np.isfinite(score):
                continue
            if any(abs(float(rec["subject_time_s"]) - float(prev["subject_time_s"])) < min_gap_s for prev in chosen):
                continue
            chosen.append(rec)
            used.add(frame_idx)
            if len(chosen) >= count:
                break
        return chosen

    used_frames: set[int] = set()
    good = _pick_candidates(frame_records, reverse=False, count=good_count, used=used_frames)
    bad = _pick_candidates(frame_records, reverse=True, count=bad_count, used=used_frames)

    snapshots = []
    for label, picked in (("good", good), ("bad", bad)):
        for rank, rec in enumerate(picked, start=1):
            item = dict(rec)
            item["snapshot_type"] = label
            item["snapshot_rank"] = rank
            snapshots.append(item)
    snapshots.sort(key=lambda rec: float(rec["subject_time_s"]))
    return snapshots


def _jsonify(value: Any) -> Any:
    """Convert NumPy-heavy structures into JSON-serialisable Python types."""
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _analysis_summary(frame_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise numeric analysis fields across rendered frames."""
    summary: dict[str, Any] = {}
    if not frame_records:
        return summary

    keys: set[str] = set()
    for rec in frame_records:
        keys.update(rec.get("analysis", {}).keys())

    for key in sorted(keys):
        values = []
        for rec in frame_records:
            value = rec.get("analysis", {}).get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                values.append(float(value))
        if values:
            arr = np.asarray(values, dtype=np.float64)
            summary[key] = {
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "p90": float(np.nanpercentile(arr, 90)),
            }
    return summary


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
    analysis_fn: Callable[
        [int, float, float, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, float, float],
        dict[str, Any],
    ] | None = None,
    overlay_formatter: Callable[[dict[str, Any]], list[str]] | None = None,
    snapshot_dir: str | Path | None = None,
    snapshot_good_count: int = 2,
    snapshot_bad_count: int = 2,
    snapshot_min_gap_s: float = 15.0,
    prealigned: bool = False,
    time_scale: float = 1.0,
) -> dict:
    """Render an overlay video of subject skeleton and Xsens GT."""
    xsens = load_xsens_skeleton(mvnx_path)
    if prealigned:
        aligned_points = np.asarray(subject_points, dtype=np.float64).copy()
        rot = np.eye(3, dtype=np.float64)
        trans = np.zeros(3, dtype=np.float64)
    else:
        aligned_points, rot, trans = align_subject_points(
            subject_points,
            subject_ts,
            xsens,
            offset_s,
            anchor_mapping,
            time_scale=time_scale,
        )

    output_mp4 = Path(output_mp4)
    output_json = Path(output_json)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(snapshot_dir) if snapshot_dir is not None else None
    if snapshot_path is not None:
        snapshot_path.mkdir(parents=True, exist_ok=True)
        for old_png in snapshot_path.glob("*.png"):
            old_png.unlink()

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90)
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
    frame_records: list[dict[str, Any]] = []

    collected = 0
    for frame_idx in range(0, len(subject_ts), max(1, frame_step)):
        if max_frames is not None and collected >= max_frames:
            break
        target_t = float(time_scale * subject_ts[frame_idx] - offset_s)
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
        pelvis_dist = float(np.linalg.norm(pelvis_center - xsens_pose["Pelvis"])) if np.isfinite(pelvis_center).all() else np.nan
        pelvis_distances.append(pelvis_dist)

        analysis = (
            analysis_fn(
                frame_idx,
                float(subject_ts[frame_idx]),
                target_t,
                subject_pose,
                xsens_pose,
                rot,
                trans,
                float(joint_dist),
                float(pelvis_dist),
            )
            if analysis_fn is not None
            else {}
        )
        if analysis is None:
            analysis = {}
        frame_records.append(
            {
                "frame_idx": int(frame_idx),
                "subject_time_s": float(subject_ts[frame_idx]),
                "gt_time_s": float(target_t),
                "subject_pose": subject_pose,
                "xsens_pose": xsens_pose,
                "joint_distance_cm": float(joint_dist),
                "pelvis_distance_cm": float(pelvis_dist),
                "snapshot_score": float(analysis.get("snapshot_score", joint_dist)),
                "analysis": analysis,
            }
        )
        collected += 1

    snapshot_records = _select_snapshots(
        frame_records,
        score_key="snapshot_score",
        good_count=max(0, int(snapshot_good_count)),
        bad_count=max(0, int(snapshot_bad_count)),
        min_gap_s=max(0.0, float(snapshot_min_gap_s)),
    ) if snapshot_path is not None else []
    snapshot_lookup = {int(rec["frame_idx"]): rec for rec in snapshot_records}

    for rec in frame_records:
        overlay_lines = (
            overlay_formatter(rec)
            if overlay_formatter is not None
            else [
                f"t={rec['subject_time_s']:.2f}s | gt={rec['gt_time_s']:.2f}s",
                f"Joint {rec['joint_distance_cm']:.1f} cm | Pelvis {rec['pelvis_distance_cm']:.1f} cm",
            ]
        )
        _render_overlay_frame(
            ax=ax,
            title=title,
            subject_pose=rec["subject_pose"],
            xsens_pose=rec["xsens_pose"],
            subject_edges=subject_edges,
            subject_label=subject_label,
            subject_color=subject_color,
            follow_radius_cm=follow_radius_cm,
            overlay_lines=overlay_lines,
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        writer.write(bgr)
        rendered += 1

        if snapshot_path is not None and int(rec["frame_idx"]) in snapshot_lookup:
            snap = snapshot_lookup[int(rec["frame_idx"])]
            filename = (
                f"{snap['snapshot_type']}_{snap['snapshot_rank']:02d}"
                f"_t{rec['subject_time_s']:.2f}_score{snap['snapshot_score']:.2f}.png"
            )
            png_path = snapshot_path / filename
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            snap["png_path"] = str(png_path)

    writer.release()
    plt.close(fig)

    metadata = {
        "output_mp4": str(output_mp4),
        "frames_rendered": rendered,
        "fps": fps,
        "frame_step": frame_step,
        "offset_seconds": offset_s,
        "time_scale": time_scale,
        "prealigned": bool(prealigned),
        "subject_label": subject_label,
        "mean_joint_distance_cm": float(np.nanmean(joint_distances)) if joint_distances else np.nan,
        "mean_pelvis_distance_cm": float(np.nanmean(pelvis_distances)) if pelvis_distances else np.nan,
        "rotation_matrix": rot.tolist(),
        "translation_cm": trans.tolist(),
        "analysis_summary": _analysis_summary(frame_records),
        "snapshots": [
            {
                key: _jsonify(value)
                for key, value in rec.items()
                if key not in {"subject_pose", "xsens_pose"}
            }
            for rec in snapshot_records
        ],
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
