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
    parser.add_argument("--mvnx", default=str(DEFAULT_MVNX), help="Xsens MVNX file.")
    parser.add_argument("--fps", type=float, default=15.0, help="Output video FPS.")
    parser.add_argument("--frame-step", type=int, default=2, help="Render every N-th frame.")
    parser.add_argument("--follow-radius-cm", type=float, default=110.0, help="Tracking camera radius.")
    return parser.parse_args()


def load_offset(alignment_json: str) -> float:
    """Load time offset from alignment summary."""
    if not os.path.isfile(alignment_json):
        return 17.25
    with open(alignment_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload.get("best_offset_seconds", 17.25))


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
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
