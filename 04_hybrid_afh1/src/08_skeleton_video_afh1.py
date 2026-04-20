#!/opt/anaconda3/envs/pose/bin/python
"""Render a skeleton comparison video for Direction 04 (AFH1)."""

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

from skeleton_video_utils import COCO_EDGES, render_comparison_video
from pose_angle_utils import (
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
    median_filter_angle_sequence,
)
from utils_mvnx import MvnxParser


DEFAULT_ALIGNMENT = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "alignment.json"
)
DEFAULT_MVNX = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"

VARIANT_CONFIG = {
    "v1": {
        "pose_npz": METHOD_DIR / "results" / "hybrid_skeleton_afh1_v1.npz",
        "output_mp4": METHOD_DIR / "results" / "skeleton_comparison_afh1_v1.mp4",
        "output_json": METHOD_DIR / "results" / "skeleton_comparison_afh1_v1.json",
        "snapshot_dir": METHOD_DIR / "results" / "skeleton_snapshots_afh1_v1",
        "subject_label": "AFH1 v1",
        "title": "Direction 04: AFH1 v1 vs Xsens",
    },
    "v2": {
        "pose_npz": METHOD_DIR / "results" / "hybrid_skeleton_afh1_v2.npz",
        "output_mp4": METHOD_DIR / "results" / "skeleton_comparison_afh1_v2.mp4",
        "output_json": METHOD_DIR / "results" / "skeleton_comparison_afh1_v2.json",
        "snapshot_dir": METHOD_DIR / "results" / "skeleton_snapshots_afh1_v2",
        "subject_label": "AFH1 v2",
        "title": "Direction 04: AFH1 v2 vs Xsens",
    },
    "trunk_only": {
        "pose_npz": METHOD_DIR / "results" / "hybrid_trunk_only_afh1_v1.npz",
        "output_mp4": METHOD_DIR / "results" / "skeleton_comparison_afh1_trunk_only.mp4",
        "output_json": METHOD_DIR / "results" / "skeleton_comparison_afh1_trunk_only.json",
        "snapshot_dir": METHOD_DIR / "results" / "skeleton_snapshots_afh1_trunk_only",
        "subject_label": "AFH1 trunk-only",
        "title": "Direction 04: AFH1 trunk-only vs Xsens",
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Render Direction 04 skeleton comparison video.")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_CONFIG),
        default="v2",
        help="Which AFH1 pose variant to render.",
    )
    parser.add_argument("--pose-npz", default="", help="Optional override pose NPZ.")
    parser.add_argument("--alignment-json", default=str(DEFAULT_ALIGNMENT), help="Alignment summary JSON.")
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


def load_offset(alignment_json: str) -> float:
    """Load time offset from alignment summary."""
    if not os.path.isfile(alignment_json):
        return 17.25
    with open(alignment_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload.get("best_offset_seconds", 17.25))


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
        "pose_npz": args.pose_npz or str(cfg["pose_npz"]),
        "output_mp4": args.output_mp4 or str(cfg["output_mp4"]),
        "output_json": args.output_json or str(cfg["output_json"]),
        "snapshot_dir": args.snapshot_dir or str(cfg["snapshot_dir"]),
        "subject_label": args.subject_label or str(cfg["subject_label"]),
        "title": args.title or str(cfg["title"]),
    }


def main() -> None:
    """Render the comparison video."""
    args = parse_args()
    resolved = resolve_variant_paths(args)

    pose_npz = np.load(resolved["pose_npz"], allow_pickle=True)
    points = np.asarray(pose_npz["keypoints"], dtype=np.float64)
    timestamps = np.asarray(pose_npz["timestamps"], dtype=np.float64)
    if timestamps.size == 0:
        raise RuntimeError("Pose NPZ contains no timestamps.")
    timestamps = timestamps - timestamps[0]

    pelvis = 0.5 * (points[:, 11] + points[:, 12])
    valid_mask = (
        np.isfinite(points).all(axis=2).any(axis=1)
        & np.isfinite(pelvis).all(axis=1)
        & (pelvis[:, 2] > 10.0)
        & (pelvis[:, 2] < 1000.0)
    )
    points = points[valid_mask]
    timestamps = timestamps[valid_mask]

    offset = load_offset(args.alignment_json)
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
        subject_color="darkorange",
        fps=args.fps,
        title=resolved["title"],
        follow_radius_cm=args.follow_radius_cm,
        frame_step=max(1, args.frame_step),
        analysis_fn=analysis_fn,
        overlay_formatter=overlay_formatter,
        snapshot_dir=resolved["snapshot_dir"],
        snapshot_good_count=max(0, args.good_snapshots),
        snapshot_bad_count=max(0, args.bad_snapshots),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
