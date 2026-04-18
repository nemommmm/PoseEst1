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
from evaluate_vs_gt import compute_geometric_angles, load_trc
from pose_angle_utils import build_gt_angle_interpolators
from utils_mvnx import MvnxParser


DEFAULT_TRC = SCRIPT_DIR / "results_mono" / "markers_results_mono.trc"
DEFAULT_OUTPUT_MP4 = SCRIPT_DIR / "results_mono" / "skeleton_comparison_dirC.mp4"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "results_mono" / "skeleton_comparison_dirC.json"
DEFAULT_SNAPSHOT_DIR = SCRIPT_DIR / "results_mono" / "skeleton_snapshots_dirC"
DEFAULT_DEPTH_JSON = SCRIPT_DIR / "results_mono" / "depth_diagnosis_dirC.json"
DEFAULT_DEPTH_MD = SCRIPT_DIR / "results_mono" / "depth_diagnosis_dirC.md"
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
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR), help="Directory for representative snapshot PNGs.")
    parser.add_argument("--depth-json", default=str(DEFAULT_DEPTH_JSON), help="Depth-diagnosis JSON output.")
    parser.add_argument("--depth-md", default=str(DEFAULT_DEPTH_MD), help="Depth-diagnosis Markdown output.")
    parser.add_argument("--mvnx", default=str(DEFAULT_MVNX), help="Xsens MVNX path.")
    parser.add_argument("--fps", type=float, default=12.5, help="Output MP4 FPS.")
    parser.add_argument("--frame-step", type=int, default=1, help="Render every N-th frame.")
    parser.add_argument("--follow-radius-cm", type=float, default=110.0, help="Tracking camera radius.")
    parser.add_argument("--offset-seconds", type=float, default=None, help="Optional override of temporal offset.")
    parser.add_argument("--good-snapshots", type=int, default=2, help="Number of low-error snapshots to save.")
    parser.add_argument("--bad-snapshots", type=int, default=2, help="Number of high-error snapshots to save.")
    return parser.parse_args()


def load_offset(alignment_json: str, fallback: float = 17.40) -> float:
    """Load time offset from alignment summary."""
    if not os.path.isfile(alignment_json):
        return fallback
    with open(alignment_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload.get("best_offset_seconds", fallback))


def build_angle_analysis(
    positions: np.ndarray,
    marker_names: list[str],
    mvnx_path: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Compute monocular geometric angles and load GT interpolators."""
    mono_angles = compute_geometric_angles(marker_names, positions)
    mvnx = MvnxParser(str(mvnx_path))
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, xidx)
    return mono_angles, gt_interps


def write_depth_diagnosis(metadata: dict, output_json: str, output_md: str) -> None:
    """Write a concise depth-dominance diagnosis next to the comparison video."""
    analysis_summary = metadata.get("analysis_summary", {})
    depth_mean = analysis_summary.get("depth_component_cm", {}).get("mean", np.nan)
    planar_mean = analysis_summary.get("planar_component_cm", {}).get("mean", np.nan)
    depth_share = analysis_summary.get("depth_share_pct", {}).get("mean", np.nan)
    dominant_fraction = analysis_summary.get("depth_dominant_flag", {}).get("mean", np.nan)

    if np.isfinite(depth_share) and depth_share >= 55.0:
        interpretation = "Depth-direction error is the dominant contributor in the average overlay mismatch."
    elif np.isfinite(depth_share) and depth_share >= 45.0:
        interpretation = "Depth-direction error is a major contributor, but planar drift is also substantial."
    else:
        interpretation = "Planar drift is comparable to or larger than depth-direction error."

    payload = {
        "comparison_video": metadata.get("output_mp4"),
        "mean_joint_distance_cm": metadata.get("mean_joint_distance_cm"),
        "mean_pelvis_distance_cm": metadata.get("mean_pelvis_distance_cm"),
        "mean_angle_mae_deg": analysis_summary.get("angle_mae_deg", {}).get("mean", np.nan),
        "mean_depth_component_cm": depth_mean,
        "mean_planar_component_cm": planar_mean,
        "mean_depth_share_pct": depth_share,
        "depth_dominant_fraction": dominant_fraction,
        "interpretation": interpretation,
        "bad_snapshots": [
            snap for snap in metadata.get("snapshots", [])
            if snap.get("snapshot_type") == "bad"
        ],
    }
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Direction C Depth Diagnosis",
        "",
        f"- Mean joint distance: {payload['mean_joint_distance_cm']:.2f} cm",
        f"- Mean pelvis distance: {payload['mean_pelvis_distance_cm']:.2f} cm",
        f"- Mean angle MAE: {payload['mean_angle_mae_deg']:.2f}°",
        f"- Mean depth component: {payload['mean_depth_component_cm']:.2f} cm",
        f"- Mean non-depth component: {payload['mean_planar_component_cm']:.2f} cm",
        f"- Mean depth share: {payload['mean_depth_share_pct']:.1f}%",
        f"- Fraction of frames where depth > non-depth: {100.0 * payload['depth_dominant_fraction']:.1f}%",
        "",
        "Interpretation:",
        payload["interpretation"],
        "",
        "Representative bad frames:",
    ]
    for snap in payload["bad_snapshots"]:
        md.append(
            f"- t={snap['subject_time_s']:.2f}s | "
            f"angle={snap['analysis']['angle_mae_deg']:.1f}° | "
            f"joint={snap['joint_distance_cm']:.1f} cm | "
            f"depth={snap['analysis']['depth_component_cm']:.1f} cm | "
            f"non-depth={snap['analysis']['planar_component_cm']:.1f} cm | "
            f"share={snap['analysis']['depth_share_pct']:.1f}% | "
            f"[png]({Path(snap['png_path']).name})"
        )
    Path(output_md).write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    """Render the comparison video."""
    args = parse_args()
    timestamps, marker_names, positions = load_trc(args.input_trc)
    if timestamps.size == 0:
        raise RuntimeError("TRC contains no frames.")
    timestamps = timestamps - timestamps[0]
    positions_cm = positions * 100.0

    required = [
        "Nose", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
        "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
        "LEye", "REye", "LThumb", "LIndex", "LPinky", "RThumb", "RIndex", "RPinky",
    ]
    if marker_names != required:
        raise RuntimeError(f"Unexpected TRC marker order: {marker_names}")

    offset = args.offset_seconds if args.offset_seconds is not None else load_offset(args.alignment_json)
    mono_angles, gt_interps = build_angle_analysis(positions, marker_names, args.mvnx)

    angle_order = (
        "RightShoulder",
        "LeftShoulder",
        "RightElbow",
        "LeftElbow",
        "RightHip",
        "LeftHip",
        "RightKnee",
        "LeftKnee",
    )

    def analysis_fn(frame_idx, subject_t, target_t, subject_pose, xsens_pose, rot, trans, joint_dist, pelvis_dist):
        _ = (subject_t, trans)
        diffs = {}
        for angle_name in angle_order:
            interp = gt_interps.get(angle_name)
            if interp is None:
                continue
            gt_val = float(interp(target_t))
            est_val = float(mono_angles[angle_name][frame_idx])
            if np.isfinite(gt_val) and np.isfinite(est_val):
                diffs[angle_name] = abs(est_val - gt_val)
        if diffs:
            worst_joint, worst_err = max(diffs.items(), key=lambda item: item[1])
            angle_mae = float(np.mean(list(diffs.values())))
        else:
            worst_joint, worst_err = "N/A", np.nan
            angle_mae = np.nan

        depth_axis_world = np.asarray(rot, dtype=np.float64) @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        depth_axis_world = depth_axis_world / max(np.linalg.norm(depth_axis_world), 1e-8)
        depth_components = []
        planar_components = []
        for subject_idx, seg_name in ANCHOR_MAPPING.items():
            if subject_idx >= len(subject_pose) or seg_name not in xsens_pose:
                continue
            src = subject_pose[subject_idx]
            tgt = xsens_pose[seg_name]
            if not (np.isfinite(src).all() and np.isfinite(tgt).all()):
                continue
            err = src - tgt
            depth_proj = float(np.dot(err, depth_axis_world))
            depth_components.append(abs(depth_proj))
            planar_err = err - depth_proj * depth_axis_world
            planar_components.append(float(np.linalg.norm(planar_err)))

        depth_component = float(np.mean(depth_components)) if depth_components else np.nan
        planar_component = float(np.mean(planar_components)) if planar_components else np.nan
        total_component = depth_component + planar_component
        depth_share = 100.0 * depth_component / total_component if np.isfinite(total_component) and total_component > 1e-8 else np.nan
        snapshot_score = (
            angle_mae + 0.06 * float(joint_dist) + 0.04 * float(depth_component)
            if np.isfinite(angle_mae) and np.isfinite(joint_dist) and np.isfinite(depth_component)
            else (angle_mae if np.isfinite(angle_mae) else float(joint_dist))
        )

        return {
            "angle_mae_deg": angle_mae,
            "worst_joint": worst_joint,
            "worst_joint_error_deg": float(worst_err),
            "depth_component_cm": depth_component,
            "planar_component_cm": planar_component,
            "depth_share_pct": float(depth_share) if np.isfinite(depth_share) else np.nan,
            "depth_dominant_flag": 1.0 if np.isfinite(depth_component) and np.isfinite(planar_component) and depth_component > planar_component else 0.0,
            "snapshot_score": float(snapshot_score) if np.isfinite(snapshot_score) else np.nan,
        }

    def overlay_formatter(record):
        analysis = record.get("analysis", {})
        return [
            f"t={record['subject_time_s']:.2f}s | gt={record['gt_time_s']:.2f}s",
            f"Angle MAE {analysis.get('angle_mae_deg', np.nan):.1f}° | Worst {analysis.get('worst_joint', 'N/A')}: {analysis.get('worst_joint_error_deg', np.nan):.1f}°",
            f"Anchor dist {record['joint_distance_cm']:.1f} cm | Pelvis dist {record['pelvis_distance_cm']:.1f} cm",
            f"Depth {analysis.get('depth_component_cm', np.nan):.1f} cm | Non-depth {analysis.get('planar_component_cm', np.nan):.1f} cm | Share {analysis.get('depth_share_pct', np.nan):.1f}%",
        ]

    metadata = render_comparison_video(
        subject_points=positions_cm,
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
        analysis_fn=analysis_fn,
        overlay_formatter=overlay_formatter,
        snapshot_dir=args.snapshot_dir,
        snapshot_good_count=max(0, args.good_snapshots),
        snapshot_bad_count=max(0, args.bad_snapshots),
    )
    metadata["marker_names"] = marker_names
    Path(args.output_json).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_depth_diagnosis(metadata, args.depth_json, args.depth_md)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
