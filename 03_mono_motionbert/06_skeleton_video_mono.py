#!/opt/anaconda3/envs/pose/bin/python
"""Render a skeleton comparison video for Direction C (MTL)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from skeleton_video_utils import render_comparison_video


DEFAULT_TRC = SCRIPT_DIR / "results_mono" / "markers_results_mono.trc"
DEFAULT_OUTPUT_MP4 = SCRIPT_DIR / "results_mono" / "skeleton_comparison_dirC.mp4"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "results_mono" / "skeleton_comparison_dirC.json"
DEFAULT_ALIGNMENT = PROJECT_ROOT / "01_stereo_triangulation" / "results" / "alignment_summary.json"
DEFAULT_MVNX = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"

TRC_EDGES = [
    (1, 0), (0, 14), (0, 15), (14, 15),
    (1, 2), (2, 4), (4, 6),
    (1, 3), (3, 5), (5, 7),
    (2, 8), (3, 9), (8, 9),
    (8, 10), (10, 12), (9, 11), (11, 13),
    (6, 16), (6, 17), (6, 18),
    (7, 19), (7, 20), (7, 21),
]

ANCHOR_MAPPING = {
    1: "Neck",
    2: "LeftShoulder",
    3: "RightShoulder",
    4: "LeftForeArm",
    5: "RightForeArm",
    6: "LeftHand",
    7: "RightHand",
    8: "LeftUpperLeg",
    9: "RightUpperLeg",
    10: "LeftLowerLeg",
    11: "RightLowerLeg",
    12: "LeftFoot",
    13: "RightFoot",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Render Direction C skeleton comparison video.")
    parser.add_argument("--input-trc", default=str(DEFAULT_TRC), help="Input TRC file.")
    parser.add_argument("--alignment-json", default=str(DEFAULT_ALIGNMENT), help="Alignment summary JSON.")
    parser.add_argument("--output-mp4", default=str(DEFAULT_OUTPUT_MP4), help="Output MP4 path.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output metadata JSON path.")
    parser.add_argument("--mvnx", default=str(DEFAULT_MVNX), help="Xsens MVNX path.")
    parser.add_argument("--fps", type=float, default=12.5, help="Output MP4 FPS.")
    parser.add_argument("--frame-step", type=int, default=1, help="Render every N-th frame.")
    parser.add_argument("--follow-radius-cm", type=float, default=110.0, help="Tracking camera radius.")
    parser.add_argument("--offset-seconds", type=float, default=None, help="Optional override of temporal offset.")
    return parser.parse_args()


def load_offset(alignment_json: str, fallback: float = 17.40) -> float:
    """Load time offset from alignment summary."""
    if not os.path.isfile(alignment_json):
        return fallback
    with open(alignment_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload.get("best_offset_seconds", fallback))


def load_trc(trc_path: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load TRC marker positions."""
    with open(trc_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    raw_names = lines[3].strip().split("\t")[2:]
    marker_names = [name for name in raw_names if name.strip()]
    timestamps = []
    frames = []
    for line in lines[6:]:
        if not line.strip():
            continue
        values = line.strip().split("\t")
        timestamps.append(float(values[1]))
        coords = [float(value) if value else np.nan for value in values[2:]]
        frames.append(coords)
    arr = np.asarray(frames, dtype=np.float64)
    positions = arr[:, : len(marker_names) * 3].reshape(-1, len(marker_names), 3)
    positions *= 100.0  # TRC stores metres; convert to centimetres for shared rendering.
    return np.asarray(timestamps, dtype=np.float64), marker_names, positions


def main() -> None:
    """Render the comparison video."""
    args = parse_args()
    timestamps, marker_names, positions = load_trc(args.input_trc)
    if timestamps.size == 0:
        raise RuntimeError("TRC contains no frames.")
    timestamps = timestamps - timestamps[0]

    required = [
        "Nose", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
        "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
        "LEye", "REye", "LThumb", "LIndex", "LPinky", "RThumb", "RIndex", "RPinky",
    ]
    if marker_names != required:
        raise RuntimeError(f"Unexpected TRC marker order: {marker_names}")

    offset = args.offset_seconds if args.offset_seconds is not None else load_offset(args.alignment_json)
    metadata = render_comparison_video(
        subject_points=positions,
        subject_ts=timestamps,
        subject_edges=TRC_EDGES,
        anchor_mapping=ANCHOR_MAPPING,
        output_mp4=args.output_mp4,
        output_json=args.output_json,
        mvnx_path=args.mvnx,
        offset_s=offset,
        subject_label="MTL (Mono)",
        subject_color="royalblue",
        fps=args.fps,
        title="Direction C: MTL vs Xsens",
        follow_radius_cm=args.follow_radius_cm,
        frame_step=max(1, args.frame_step),
    )
    metadata["marker_names"] = marker_names
    Path(args.output_json).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
