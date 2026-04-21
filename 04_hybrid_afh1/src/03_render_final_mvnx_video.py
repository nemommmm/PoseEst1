#!/opt/anaconda3/envs/pose/bin/python
"""Render the retained final-MVNX comparison video for Direction 04."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from skeleton_video_utils import (
    COCO_EDGES,
    kabsch_transform,
    load_xsens_skeleton,
    render_comparison_video,
    xsens_pose_at,
)
from pose_angle_utils import (
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
    median_filter_angle_sequence,
)
from utils_mvnx import MvnxParser


DEFAULT_MVNX = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"
DEFAULT_EASYERGO_UPLOAD_DIR = METHOD_DIR / "data" / "easyergo_uploaded"
TIMING_JSON = METHOD_DIR / "results" / "02_final_mvnx_timing" / "affine_fit.json"
OFFSET_FALLBACK = 16.83
TIME_SCALE_FALLBACK = 1.0102

VARIANT_CONFIG = {
    "final_mvnx": {
        "output_mp4": METHOD_DIR / "results" / "03_final_mvnx_video.mp4",
        "output_json": METHOD_DIR / "results" / "03_final_mvnx_video.json",
        "snapshot_dir": METHOD_DIR / "results" / "03_final_mvnx_snapshots",
        "subject_label": "EasyErgo final MVNX",
        "title": "Direction 04: EasyErgo final MVNX vs Xsens",
        "subject_color": "teal",
        "prealigned": True,
    },
}

ANCHOR_MAPPING = {
    5: "LeftShoulder",
    6: "RightShoulder",
    7: "LeftForeArm",
    8: "RightForeArm",
    9: "LeftHand",
    10: "RightHand",
    11: "LeftUpperLeg",
    12: "RightUpperLeg",
    13: "LeftLowerLeg",
    14: "RightLowerLeg",
    15: "LeftFoot",
    16: "RightFoot",
}

FINAL_MVNX_SEGMENT_TO_COCO = {
    "Head": 0,
    "LeftUpperArm": 5,
    "RightUpperArm": 6,
    "LeftForeArm": 7,
    "RightForeArm": 8,
    "LeftHand": 9,
    "RightHand": 10,
    "LeftUpperLeg": 11,
    "RightUpperLeg": 12,
    "LeftLowerLeg": 13,
    "RightLowerLeg": 14,
    "LeftFoot": 15,
    "RightFoot": 16,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Render Direction 04 skeleton comparison video.")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_CONFIG),
        default="final_mvnx",
        help="Retained Direction 04 output variant.",
    )
    parser.add_argument(
        "--easyergo-mvnx",
        default="",
        help="Optional override EasyErgo final MVNX path.",
    )
    parser.add_argument(
        "--offset-seconds",
        type=float,
        default=float("nan"),
        help="Optional direct override of the affine offset (seconds).",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=float("nan"),
        help="Optional affine time scale in gt_t = scale * est_t - offset.",
    )
    parser.add_argument("--output-mp4", default="", help="Optional output MP4 path.")
    parser.add_argument("--output-json", default="", help="Optional output metadata JSON path.")
    parser.add_argument("--snapshot-dir", default="", help="Optional snapshot output directory.")
    parser.add_argument("--subject-label", default="", help="Optional override subject label.")
    parser.add_argument("--title", default="", help="Optional override video title.")
    parser.add_argument("--mvnx", default=str(DEFAULT_MVNX), help="Xsens MVNX file.")
    parser.add_argument("--fps", type=float, default=15.0, help="Output video FPS.")
    parser.add_argument("--frame-step", type=int, default=2, help="Render every N-th frame.")
    parser.add_argument("--follow-radius-cm", type=float, default=110.0, help="Tracking camera radius.")
    parser.add_argument(
        "--angle-smooth-radius",
        type=int,
        default=int(os.environ.get("POSE_ANGLE_SMOOTH_RADIUS", "8")),
        help="Temporal median smoothing radius for video angle overlay.",
    )
    parser.add_argument("--good-snapshots", type=int, default=2, help="Number of low-error snapshots to save.")
    parser.add_argument("--bad-snapshots", type=int, default=2, help="Number of high-error snapshots to save.")
    return parser.parse_args()


def resolve_time_mapping() -> tuple[float, float]:
    """Load the retained affine timing for the final MVNX branch."""
    if TIMING_JSON.is_file():
        with TIMING_JSON.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            float(payload.get("offset_s", OFFSET_FALLBACK)),
            float(payload.get("time_scale", TIME_SCALE_FALLBACK)),
        )
    return OFFSET_FALLBACK, TIME_SCALE_FALLBACK


def build_angle_analysis(
    points: np.ndarray,
    mvnx_path: str,
    smooth_radius: int,
) -> tuple[tuple[str, ...], np.ndarray, dict]:
    """Compute smoothed semantic angles and GT interpolators."""
    angle_names, angle_values = compute_semantic_angle_sequence(points)
    angle_values = median_filter_angle_sequence(angle_values, radius=max(0, smooth_radius))

    mvnx = MvnxParser(str(mvnx_path))
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, xidx)
    return tuple(angle_names), angle_values, gt_interps


def resolve_variant_paths(args: argparse.Namespace) -> dict[str, str]:
    """Resolve defaults from the chosen AFH1 variant."""
    cfg = VARIANT_CONFIG[args.variant]
    return {
        "easyergo_mvnx": args.easyergo_mvnx,
        "output_mp4": args.output_mp4 or str(cfg["output_mp4"]),
        "output_json": args.output_json or str(cfg["output_json"]),
        "snapshot_dir": args.snapshot_dir or str(cfg["snapshot_dir"]),
        "subject_label": args.subject_label or str(cfg["subject_label"]),
        "title": args.title or str(cfg["title"]),
        "subject_color": str(cfg["subject_color"]),
        "prealigned": bool(cfg.get("prealigned", False)),
    }


def resolve_final_mvnx_path(path_override: str) -> Path:
    """Resolve the EasyErgo final MVNX file from the upload folder."""
    if path_override:
        path = Path(path_override).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EasyErgo final MVNX not found: {path}")
        return path

    matches = sorted(DEFAULT_EASYERGO_UPLOAD_DIR.glob("*.mvnx"))
    if not matches:
        raise FileNotFoundError(
            f"No EasyErgo MVNX file found in {DEFAULT_EASYERGO_UPLOAD_DIR}."
        )
    if len(matches) > 1:
        formatted = ", ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Multiple EasyErgo MVNX files found in {DEFAULT_EASYERGO_UPLOAD_DIR}: {formatted}"
        )
    return matches[0]


def build_coco_points_from_mvnx(mvnx: MvnxParser) -> np.ndarray:
    """Map MVNX segment origins into the local 17-joint COCO-style skeleton."""
    n_frames = mvnx.data.shape[0]
    points = np.full((n_frames, 17, 3), np.nan, dtype=np.float64)
    for seg_name, coco_idx in FINAL_MVNX_SEGMENT_TO_COCO.items():
        seg_data = mvnx.get_segment_data(seg_name)
        if seg_data is not None:
            points[:, coco_idx, :] = seg_data
    return points


def estimate_global_rotation(
    points: np.ndarray,
    timestamps: np.ndarray,
    mvnx_path: str,
    offset_s: float,
    time_scale: float,
) -> np.ndarray:
    """Estimate one global rotation from pelvis-centred EasyErgo poses to Xsens."""
    xsens = load_xsens_skeleton(mvnx_path)
    src_vectors: list[np.ndarray] = []
    tgt_vectors: list[np.ndarray] = []

    for frame_idx in range(0, len(timestamps), 3):
        target_t = float(time_scale * timestamps[frame_idx] - offset_s)
        xsens_pose = xsens_pose_at(xsens, target_t)
        if "Pelvis" not in xsens_pose:
            continue
        subject_pose = points[frame_idx]
        subject_center = np.nanmean(subject_pose[[11, 12]], axis=0)
        if not np.isfinite(subject_center).all():
            continue
        gt_center = xsens_pose["Pelvis"]
        for subject_idx, seg_name in ANCHOR_MAPPING.items():
            if seg_name not in xsens_pose:
                continue
            src = subject_pose[subject_idx]
            tgt = xsens_pose[seg_name]
            if np.isfinite(src).all() and np.isfinite(tgt).all():
                src_vectors.append(src - subject_center)
                tgt_vectors.append(tgt - gt_center)

    if len(src_vectors) < 10:
        return np.eye(3, dtype=np.float64)
    rot, _ = kabsch_transform(
        np.asarray(src_vectors, dtype=np.float64),
        np.asarray(tgt_vectors, dtype=np.float64),
    )
    return rot


def prepare_final_mvnx_points(
    easyergo_mvnx_path: Path,
    gt_mvnx_path: str,
    offset_s: float,
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load final MVNX, convert to pseudo-COCO, and place it in GT space for video."""
    mvnx = MvnxParser(str(easyergo_mvnx_path))
    mvnx.parse()
    timestamps = mvnx.timestamps.copy()
    timestamps, unique_idx = np.unique(timestamps, return_index=True)
    timestamps = timestamps - timestamps[0]
    raw_points = build_coco_points_from_mvnx(mvnx)[unique_idx]

    rot = estimate_global_rotation(raw_points, timestamps, gt_mvnx_path, offset_s, time_scale)
    xsens = load_xsens_skeleton(gt_mvnx_path)
    aligned_points = np.full_like(raw_points, np.nan)

    for frame_idx, subject_t in enumerate(timestamps):
        xsens_pose = xsens_pose_at(xsens, float(time_scale * subject_t - offset_s))
        if "Pelvis" not in xsens_pose:
            continue
        subject_pose = raw_points[frame_idx]
        subject_center = np.nanmean(subject_pose[[11, 12]], axis=0)
        if not np.isfinite(subject_center).all():
            continue
        local_pose = subject_pose - subject_center
        aligned_points[frame_idx] = (rot @ local_pose.T).T + xsens_pose["Pelvis"]

    pelvis = 0.5 * (aligned_points[:, 11] + aligned_points[:, 12])
    valid_mask = np.isfinite(aligned_points).all(axis=2).any(axis=1) & np.isfinite(pelvis).all(axis=1)
    return aligned_points[valid_mask], timestamps[valid_mask]


def load_subject_sequence(
    resolved: dict[str, object],
    mvnx_path: str,
    offset_s: float,
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the retained final MVNX subject sequence."""
    return prepare_final_mvnx_points(
        easyergo_mvnx_path=resolve_final_mvnx_path(str(resolved["easyergo_mvnx"])),
        gt_mvnx_path=mvnx_path,
        offset_s=offset_s,
        time_scale=time_scale,
    )


def main() -> None:
    """Render the comparison video."""
    args = parse_args()
    resolved = resolve_variant_paths(args)
    default_offset, default_time_scale = resolve_time_mapping()
    offset = float(args.offset_seconds) if np.isfinite(args.offset_seconds) else default_offset
    time_scale = float(args.time_scale) if np.isfinite(args.time_scale) else default_time_scale

    points, timestamps = load_subject_sequence(
        resolved=resolved,
        mvnx_path=args.mvnx,
        offset_s=offset,
        time_scale=time_scale,
    )
    angle_names, angle_values, gt_interps = build_angle_analysis(
        points=points,
        mvnx_path=args.mvnx,
        smooth_radius=args.angle_smooth_radius,
    )

    def analysis_fn(frame_idx, subject_t, target_t, subject_pose, xsens_pose, rot, trans, joint_dist, pelvis_dist):
        _ = (subject_t, subject_pose, xsens_pose, rot, trans)
        est = angle_values[frame_idx]
        diffs = {}
        for angle_idx, angle_name in enumerate(angle_names):
            interp = gt_interps.get(angle_name)
            if interp is None:
                continue
            gt_val = float(interp(target_t))
            est_val = float(est[angle_idx])
            if np.isfinite(gt_val) and np.isfinite(est_val):
                diffs[angle_name] = abs(est_val - gt_val)
        if diffs:
            worst_joint, worst_err = max(diffs.items(), key=lambda item: item[1])
            angle_mae = float(np.mean(list(diffs.values())))
        else:
            worst_joint, worst_err = "N/A", np.nan
            angle_mae = np.nan
        snapshot_score = (
            angle_mae + 0.12 * float(joint_dist)
            if np.isfinite(angle_mae) and np.isfinite(joint_dist)
            else (angle_mae if np.isfinite(angle_mae) else float(joint_dist))
        )
        return {
            "angle_mae_deg": angle_mae,
            "worst_joint": worst_joint,
            "worst_joint_error_deg": float(worst_err),
            "snapshot_score": float(snapshot_score) if np.isfinite(snapshot_score) else np.nan,
        }

    def overlay_formatter(record):
        analysis = record.get("analysis", {})
        angle_mae = analysis.get("angle_mae_deg", np.nan)
        worst_joint = analysis.get("worst_joint", "N/A")
        worst_err = analysis.get("worst_joint_error_deg", np.nan)
        return [
            f"t={record['subject_time_s']:.2f}s | gt={record['gt_time_s']:.2f}s",
            f"Angle MAE {angle_mae:.1f}° | Worst {worst_joint}: {worst_err:.1f}°",
            f"Anchor dist {record['joint_distance_cm']:.1f} cm | Pelvis dist {record['pelvis_distance_cm']:.1f} cm",
        ]

    metadata = render_comparison_video(
        subject_points=points,
        subject_ts=timestamps,
        subject_edges=COCO_EDGES,
        anchor_mapping=ANCHOR_MAPPING,
        output_mp4=resolved["output_mp4"],
        output_json=resolved["output_json"],
        mvnx_path=args.mvnx,
        offset_s=offset,
        subject_label=resolved["subject_label"],
        subject_color=resolved["subject_color"],
        fps=args.fps,
        title=resolved["title"],
        follow_radius_cm=args.follow_radius_cm,
        frame_step=max(1, args.frame_step),
        analysis_fn=analysis_fn,
        overlay_formatter=overlay_formatter,
        snapshot_dir=resolved["snapshot_dir"],
        snapshot_good_count=max(0, args.good_snapshots),
        snapshot_bad_count=max(0, args.bad_snapshots),
        prealigned=bool(resolved["prealigned"]),
        time_scale=time_scale,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
