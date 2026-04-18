#!/opt/anaconda3/envs/pose/bin/python
"""Render a skeleton comparison video for Direction A (SKT).

This script overlays the recovered historical-best stereo pose sequence and the
Xsens full-body skeleton in the same aligned 3D coordinate frame.
"""

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


DEFAULT_POSE = (
    METHOD_DIR
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
DEFAULT_ALIGNMENT = (
    METHOD_DIR
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "alignment.json"
)
DEFAULT_OUTPUT_MP4 = METHOD_DIR / "results" / "skeleton_comparison_dirA.mp4"
DEFAULT_OUTPUT_JSON = METHOD_DIR / "results" / "skeleton_comparison_dirA.json"
DEFAULT_SNAPSHOT_DIR = METHOD_DIR / "results" / "skeleton_snapshots_dirA"
DEFAULT_MVNX = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"

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
    parser = argparse.ArgumentParser(description="Render Direction A skeleton comparison video.")
    parser.add_argument("--pose-npz", default=str(DEFAULT_POSE), help="Pose NPZ to render.")
    parser.add_argument("--alignment-json", default=str(DEFAULT_ALIGNMENT), help="Alignment summary JSON.")
    parser.add_argument("--output-mp4", default=str(DEFAULT_OUTPUT_MP4), help="Output MP4 path.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output metadata JSON path.")
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR), help="Directory for representative snapshot PNGs.")
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


def main() -> None:
    """Render the comparison video."""
    args = parse_args()
    pose_npz = np.load(args.pose_npz, allow_pickle=True)
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
        _ = (subject_pose, xsens_pose, rot, trans)
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
        output_mp4=args.output_mp4,
        output_json=args.output_json,
        mvnx_path=args.mvnx,
        offset_s=offset,
        subject_label="SKT (Stereo)",
        subject_color="red",
        fps=args.fps,
        title="Direction A: SKT vs Xsens",
        follow_radius_cm=args.follow_radius_cm,
        frame_step=max(1, args.frame_step),
        analysis_fn=analysis_fn,
        overlay_formatter=overlay_formatter,
        snapshot_dir=args.snapshot_dir,
        snapshot_good_count=max(0, args.good_snapshots),
        snapshot_bad_count=max(0, args.bad_snapshots),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
