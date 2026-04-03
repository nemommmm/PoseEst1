#!/opt/anaconda3/envs/pose/bin/python
"""
Render a diagnostic 3D skeleton video from monocular TRC output.

By default the video shows a synchronized comparison:
- left: monocular reconstruction
- right: Xsens GT reference

The raw video is upside-down, but monocular inference already rotates it by
180 degrees inside RTMDet-MotionBert-OpenSim/run_inference.py, so the TRC is
treated as upright. This renderer must not rotate it a second time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from pose_angle_utils import build_gt_angle_interpolators
from utils_mvnx import MvnxParser

from evaluate_vs_gt import BEST_OFFSET_DEFAULT, _rula_grand, compute_geometric_angles


RESULTS_DIR = SCRIPT_DIR / "results_mono"
DEFAULT_TRC = RESULTS_DIR / "markers_results_mono.trc"
DEFAULT_MP4 = RESULTS_DIR / "markers_results_mono_3d_compare.mp4"
DEFAULT_ALIGN_JSON = PROJECT_ROOT / "01_stereo_triangulation" / "results" / "alignment_summary.json"
DEFAULT_MVNX = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"

BODY_MARKER_NAMES = [
    "Nose",
    "Neck",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
]

BODY_EDGES = [
    ("Nose", "Neck"),
    ("Neck", "LShoulder"),
    ("Neck", "RShoulder"),
    ("LShoulder", "RShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("LShoulder", "LHip"),
    ("RShoulder", "RHip"),
    ("LHip", "RHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
]

SEGMENT_DEPENDENCIES = {
    "Pelvis",
    "Neck",
    "Head",
    "LeftShoulder",
    "RightShoulder",
    "LeftUpperArm",
    "RightUpperArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
    "LeftUpperLeg",
    "RightUpperLeg",
    "LeftLowerLeg",
    "RightLowerLeg",
    "LeftFoot",
    "RightFoot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render monocular vs Xsens 3D comparison video.")
    parser.add_argument("--input-trc", default=str(DEFAULT_TRC), help="Path to TRC input.")
    parser.add_argument("--output-mp4", default=str(DEFAULT_MP4), help="Path to output MP4.")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default: TRC DataRate).")
    parser.add_argument("--width", type=int, default=1440, help="Output video width.")
    parser.add_argument("--height", type=int, default=960, help="Output video height.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--azimuth", type=float, default=-58.0, help="Camera azimuth in degrees.")
    parser.add_argument("--elevation", type=float, default=16.0, help="Camera elevation in degrees.")
    parser.add_argument("--mvnx-path", default=str(DEFAULT_MVNX), help="Path to MVNX GT file.")
    parser.add_argument("--alignment-json", default=str(DEFAULT_ALIGN_JSON), help="Path to alignment summary.")
    return parser.parse_args()


def load_trc_with_rate(trc_path: Path) -> tuple[float, np.ndarray, list[str], np.ndarray]:
    with trc_path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    if len(lines) < 7:
        raise ValueError(f"TRC file is too short: {trc_path}")

    header_values = lines[2].split("\t")
    data_rate = float(header_values[0])
    marker_names = [name for name in lines[3].split("\t")[2:] if name.strip()]
    n_markers = len(marker_names)

    timestamps = []
    frames = []
    for line in lines[6:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        timestamps.append(float(parts[1]))
        coords = [float(value) if value else np.nan for value in parts[2 : 2 + 3 * n_markers]]
        frames.append(coords)

    positions = np.asarray(frames, dtype=np.float64).reshape(-1, n_markers, 3)
    return data_rate, np.asarray(timestamps, dtype=np.float64), marker_names, positions


def best_offset(alignment_json: Path) -> float:
    if alignment_json.is_file():
        with alignment_json.open("r", encoding="utf-8") as handle:
            return float(json.load(handle).get("best_offset_seconds", BEST_OFFSET_DEFAULT))
    return BEST_OFFSET_DEFAULT


def build_edges(marker_names: list[str], edge_names: list[tuple[str, str]]) -> list[tuple[int, int]]:
    index = {name: idx for idx, name in enumerate(marker_names)}
    edges = []
    for start_name, end_name in edge_names:
        if start_name in index and end_name in index:
            edges.append((index[start_name], index[end_name]))
    return edges


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def compute_scene_stats(marker_names: list[str], positions: np.ndarray, vertical_axis: int) -> tuple[np.ndarray, float, bool]:
    idx = {name: i for i, name in enumerate(marker_names)}
    if {"LHip", "RHip"} <= idx.keys():
        hip_mid = 0.5 * (positions[:, idx["LHip"]] + positions[:, idx["RHip"]])
        center = np.nanmedian(hip_mid, axis=0)
    else:
        center = np.nanmedian(positions.reshape(-1, 3), axis=0)

    centered = positions - center
    finite = centered[np.isfinite(centered).all(axis=2)]
    if finite.size == 0:
        raise ValueError("No finite 3D markers found.")

    horiz_axes = [0, 2] if vertical_axis == 1 else [0, 1]
    horiz_extent = max(np.nanpercentile(np.abs(finite[:, axis]), 98) for axis in horiz_axes)
    vert_extent = np.nanpercentile(np.abs(finite[:, vertical_axis]), 98)
    scale = 0.42 / max(float(max(horiz_extent, vert_extent)), 1e-6)

    upright_ok = False
    if {"LShoulder", "RShoulder", "LHip", "RHip"} <= idx.keys():
        shoulder_v = 0.5 * (
            positions[:, idx["LShoulder"], vertical_axis] + positions[:, idx["RShoulder"], vertical_axis]
        )
        hip_v = 0.5 * (positions[:, idx["LHip"], vertical_axis] + positions[:, idx["RHip"], vertical_axis])
        delta = shoulder_v - hip_v
        finite_delta = delta[np.isfinite(delta)]
        upright_ok = finite_delta.size > 0 and float(np.nanmedian(finite_delta)) > 0.0

    return center, scale, upright_ok


def project_points(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    scale: float,
    origin_x: int,
    width: int,
    height: int,
    azimuth: float,
    elevation: float,
    vertical_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered = points_xyz - center_xyz
    rot = rotation_matrix_x(elevation) @ rotation_matrix_y(azimuth)
    rotated = centered @ rot.T
    if vertical_axis == 1:
        x_axis = rotated[:, 0]
        y_axis = rotated[:, 1]
        depth = rotated[:, 2]
    else:
        x_axis = rotated[:, 0]
        y_axis = rotated[:, 2]
        depth = rotated[:, 1]
    projected = np.stack(
        [
            origin_x + width * 0.5 + x_axis * min(width, height) * scale,
            height * 0.80 - y_axis * min(width, height) * scale,
        ],
        axis=1,
    )
    return projected, depth


def draw_skeleton(
    image: np.ndarray,
    projected: np.ndarray,
    depth: np.ndarray,
    finite_mask: np.ndarray,
    edges: list[tuple[int, int]],
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
) -> None:
    for start_idx, end_idx in edges:
        if finite_mask[start_idx] and finite_mask[end_idx]:
            p1 = tuple(np.round(projected[start_idx]).astype(int))
            p2 = tuple(np.round(projected[end_idx]).astype(int))
            cv2.line(image, p1, p2, line_color, 2, lineType=cv2.LINE_AA)

    order = np.argsort(depth)
    for idx in order:
        if not finite_mask[idx]:
            continue
        px, py = np.round(projected[idx]).astype(int)
        cv2.circle(image, (px, py), 5, point_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(image, (px, py), 5, (18, 18, 18), thickness=1, lineType=cv2.LINE_AA)


def extract_trc_body(marker_names: list[str], positions: np.ndarray) -> tuple[list[str], np.ndarray]:
    index = {name: idx for idx, name in enumerate(marker_names)}
    body = np.full((positions.shape[0], len(BODY_MARKER_NAMES), 3), np.nan, dtype=np.float64)
    for marker_idx, marker_name in enumerate(BODY_MARKER_NAMES):
        if marker_name in index:
            body[:, marker_idx] = positions[:, index[marker_name]]
    return BODY_MARKER_NAMES, body


def transform_xsens_points(points_xyz: np.ndarray) -> np.ndarray:
    transformed = np.full_like(points_xyz, np.nan, dtype=np.float64)
    transformed[..., 0] = points_xyz[..., 0]
    transformed[..., 1] = points_xyz[..., 2]
    transformed[..., 2] = -points_xyz[..., 1]
    return transformed


def build_xsens_body_positions(mvnx: MvnxParser) -> tuple[np.ndarray, np.ndarray]:
    segment_cache = {
        label: mvnx.get_segment_data(label)
        for label in SEGMENT_DEPENDENCIES
    }
    n_frames = len(mvnx.timestamps)
    body = np.full((n_frames, len(BODY_MARKER_NAMES), 3), np.nan, dtype=np.float64)

    def midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray:
        if a is None or b is None:
            return np.full((n_frames, 3), np.nan, dtype=np.float64)
        return 0.5 * (a + b)

    body[:, 0] = segment_cache["Head"]
    body[:, 1] = segment_cache["Neck"]
    body[:, 2] = segment_cache["LeftShoulder"]
    body[:, 3] = segment_cache["RightShoulder"]
    body[:, 4] = midpoint(segment_cache["LeftUpperArm"], segment_cache["LeftForeArm"])
    body[:, 5] = midpoint(segment_cache["RightUpperArm"], segment_cache["RightForeArm"])
    body[:, 6] = midpoint(segment_cache["LeftForeArm"], segment_cache["LeftHand"])
    body[:, 7] = midpoint(segment_cache["RightForeArm"], segment_cache["RightHand"])
    body[:, 8] = midpoint(segment_cache["Pelvis"], segment_cache["LeftUpperLeg"])
    body[:, 9] = midpoint(segment_cache["Pelvis"], segment_cache["RightUpperLeg"])
    body[:, 10] = midpoint(segment_cache["LeftUpperLeg"], segment_cache["LeftLowerLeg"])
    body[:, 11] = midpoint(segment_cache["RightUpperLeg"], segment_cache["RightLowerLeg"])
    body[:, 12] = midpoint(segment_cache["LeftLowerLeg"], segment_cache["LeftFoot"])
    body[:, 13] = midpoint(segment_cache["RightLowerLeg"], segment_cache["RightFoot"])

    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, unique_idx = np.unique(xsens_ts, return_index=True)
    body = body[unique_idx]
    xsens_ts = xsens_ts - xsens_ts[0]
    body = transform_xsens_points(body)
    return xsens_ts, body


def interpolate_positions(source_ts: np.ndarray, source_positions: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    interpolated = np.full((len(target_ts),) + source_positions.shape[1:], np.nan, dtype=np.float64)
    for marker_idx in range(source_positions.shape[1]):
        for axis in range(3):
            values = source_positions[:, marker_idx, axis]
            finite = np.isfinite(values)
            if np.count_nonzero(finite) < 2:
                continue
            interp = interp1d(
                source_ts[finite],
                values[finite],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            interpolated[:, marker_idx, axis] = interp(target_ts)
    return interpolated


def build_gt_angle_series(mvnx: MvnxParser, aligned_ts: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, xidx)

    gt_angles = {
        name: interp(aligned_ts) if name in gt_interps else np.full(len(aligned_ts), np.nan, dtype=np.float64)
        for name, interp in gt_interps.items()
    }

    trunk_ergo = mvnx.get_ergo_angle_data("Pelvis_T8")
    gt_trunk = np.full(len(aligned_ts), np.nan, dtype=np.float64)
    if trunk_ergo is not None:
        trunk_interp = interp1d(
            xsens_ts,
            trunk_ergo[xidx, 0],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        gt_trunk = trunk_interp(aligned_ts)
    return gt_angles, gt_trunk


def compute_rula_gt(gt_angles: dict[str, np.ndarray], gt_trunk: np.ndarray) -> np.ndarray:
    length = len(gt_trunk)
    scores = np.full(length, np.nan, dtype=np.float64)
    for i in range(length):
        scores[i] = _rula_grand(
            np.nanmax([abs(gt_angles.get("RightShoulder", [np.nan])[i]), abs(gt_angles.get("LeftShoulder", [np.nan])[i])]),
            np.nanmax([gt_angles.get("RightElbow", [np.nan])[i], gt_angles.get("LeftElbow", [np.nan])[i]]),
            abs(gt_trunk[i]),
            np.nanmax([abs(gt_angles.get("RightKnee", [np.nan])[i]), abs(gt_angles.get("LeftKnee", [np.nan])[i])]),
        )
    return scores


def prepare_error_arrays(
    trc_ts: np.ndarray,
    mono_angles: dict[str, np.ndarray],
    rula_est: np.ndarray,
    mvnx: MvnxParser,
    offset_sec: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    aligned_ts = trc_ts + offset_sec
    gt_angles, gt_trunk = build_gt_angle_series(mvnx, aligned_ts)
    rula_gt = compute_rula_gt(gt_angles, gt_trunk)

    per_angle_errors = {}
    for name in ("RightShoulder", "LeftShoulder", "RightElbow", "LeftElbow", "RightKnee", "LeftKnee"):
        est = mono_angles.get(name, np.full(len(trc_ts), np.nan))
        gt = gt_angles.get(name, np.full(len(trc_ts), np.nan))
        per_angle_errors[name] = np.abs(est - gt)
    per_angle_errors["TrunkFlex"] = np.abs(mono_angles["TrunkFlex"] - gt_trunk)

    error_stack = np.stack([per_angle_errors[name] for name in per_angle_errors], axis=1)
    mean_angle_error = np.nanmean(error_stack, axis=1)
    return aligned_ts, per_angle_errors, mean_angle_error, rula_gt


def draw_text_block(
    image: np.ndarray,
    origin_x: int,
    title: str,
    lines: list[str],
    width: int,
) -> None:
    cv2.putText(image, title, (origin_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (245, 245, 245), 2, cv2.LINE_AA)
    y = 74
    for line in lines:
        cv2.putText(image, line, (origin_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (195, 215, 232), 1, cv2.LINE_AA)
        y += 26
    cv2.line(image, (origin_x, 88), (origin_x + width - 24, 88), (56, 82, 106), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    trc_path = Path(args.input_trc)
    output_mp4 = Path(args.output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    output_meta = output_mp4.with_suffix(".json")

    data_rate, trc_ts, marker_names, trc_positions = load_trc_with_rate(trc_path)
    fps = float(args.fps if args.fps is not None else data_rate)
    if args.max_frames is not None:
        trc_ts = trc_ts[: args.max_frames]
        trc_positions = trc_positions[: args.max_frames]

    body_names, mono_body = extract_trc_body(marker_names, trc_positions)
    mono_edges = build_edges(body_names, BODY_EDGES)
    mono_center, mono_scale, mono_upright_ok = compute_scene_stats(body_names, mono_body, vertical_axis=1)

    mono_angles = compute_geometric_angles(marker_names, trc_positions)
    shoulder_deg = np.nanmax(np.stack([mono_angles["RightShoulder"], mono_angles["LeftShoulder"]], axis=1), axis=1)
    elbow_deg = np.nanmax(np.stack([mono_angles["RightElbow"], mono_angles["LeftElbow"]], axis=1), axis=1)
    knee_deg = np.nanmax(np.stack([mono_angles["RightKnee"], mono_angles["LeftKnee"]], axis=1), axis=1)
    trunk_deg = mono_angles["TrunkFlex"]
    rula_est = np.array([
        _rula_grand(shoulder_deg[i], elbow_deg[i], trunk_deg[i], knee_deg[i]) for i in range(len(trc_ts))
    ])

    mvnx = MvnxParser(str(args.mvnx_path))
    mvnx.parse()
    xsens_ts, xsens_body = build_xsens_body_positions(mvnx)
    offset_sec = best_offset(Path(args.alignment_json))
    aligned_ts = trc_ts + offset_sec
    xsens_body_aligned = interpolate_positions(xsens_ts, xsens_body, aligned_ts)
    xsens_edges = build_edges(body_names, BODY_EDGES)
    xsens_center, xsens_scale, xsens_upright_ok = compute_scene_stats(body_names, xsens_body_aligned, vertical_axis=1)

    aligned_ts, angle_errors, mean_angle_error, rula_gt = prepare_error_arrays(trc_ts, mono_angles, rula_est, mvnx, offset_sec)

    canvas = np.full((args.height, args.width, 3), 20, dtype=np.uint8)
    split_x = args.width // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_mp4}")

    try:
        for frame_idx, timestamp in enumerate(trc_ts):
            frame = canvas.copy()

            mono_points = mono_body[frame_idx]
            mono_finite = np.isfinite(mono_points).all(axis=1)
            mono_proj = np.full((len(mono_points), 2), np.nan, dtype=np.float64)
            mono_depth = np.full(len(mono_points), np.nan, dtype=np.float64)
            if np.any(mono_finite):
                mono_proj_valid, mono_depth_valid = project_points(
                    mono_points[mono_finite],
                    mono_center,
                    mono_scale,
                    0,
                    split_x,
                    args.height,
                    args.azimuth,
                    args.elevation,
                    vertical_axis=1,
                )
                mono_proj[mono_finite] = mono_proj_valid
                mono_depth[mono_finite] = mono_depth_valid
            draw_skeleton(frame, mono_proj, mono_depth, mono_finite, mono_edges, (87, 208, 255), (255, 199, 77))

            xsens_points = xsens_body_aligned[frame_idx]
            xsens_finite = np.isfinite(xsens_points).all(axis=1)
            xsens_proj = np.full((len(xsens_points), 2), np.nan, dtype=np.float64)
            xsens_depth = np.full(len(xsens_points), np.nan, dtype=np.float64)
            if np.any(xsens_finite):
                xsens_proj_valid, xsens_depth_valid = project_points(
                    xsens_points[xsens_finite],
                    xsens_center,
                    xsens_scale,
                    split_x,
                    split_x,
                    args.height,
                    args.azimuth,
                    args.elevation,
                    vertical_axis=1,
                )
                xsens_proj[xsens_finite] = xsens_proj_valid
                xsens_depth[xsens_finite] = xsens_depth_valid
            draw_skeleton(frame, xsens_proj, xsens_depth, xsens_finite, xsens_edges, (144, 235, 144), (255, 255, 255))

            cv2.line(frame, (split_x, 0), (split_x, args.height), (44, 66, 88), 2, cv2.LINE_AA)
            draw_text_block(
                frame,
                26,
                "Monocular reconstruction",
                [
                    f"t(video): {float(timestamp):6.2f}s",
                    f"RULA est: {rula_est[frame_idx]:4.1f}" if np.isfinite(rula_est[frame_idx]) else "RULA est: n/a",
                    f"Mean angle error: {mean_angle_error[frame_idx]:5.1f} deg" if np.isfinite(mean_angle_error[frame_idx]) else "Mean angle error: n/a",
                ],
                split_x,
            )
            draw_text_block(
                frame,
                split_x + 26,
                "Xsens GT reference",
                [
                    f"t(gt): {aligned_ts[frame_idx]:6.2f}s",
                    f"RULA gt: {rula_gt[frame_idx]:4.1f}" if np.isfinite(rula_gt[frame_idx]) else "RULA gt: n/a",
                    f"RULA delta: {abs(rula_est[frame_idx] - rula_gt[frame_idx]):4.1f}"
                    if np.isfinite(rula_est[frame_idx]) and np.isfinite(rula_gt[frame_idx])
                    else "RULA delta: n/a",
                ],
                split_x,
            )

            error_lines = [
                f"L/R shoulder err: {angle_errors['LeftShoulder'][frame_idx]:4.1f} / {angle_errors['RightShoulder'][frame_idx]:4.1f}",
                f"L/R elbow err:    {angle_errors['LeftElbow'][frame_idx]:4.1f} / {angle_errors['RightElbow'][frame_idx]:4.1f}",
                f"L/R knee err:     {angle_errors['LeftKnee'][frame_idx]:4.1f} / {angle_errors['RightKnee'][frame_idx]:4.1f}",
                f"Trunk err:        {angle_errors['TrunkFlex'][frame_idx]:4.1f}",
            ]
            base_y = args.height - 116
            cv2.rectangle(frame, (18, base_y - 30), (args.width - 18, args.height - 18), (10, 16, 22), thickness=-1)
            for line_idx, text in enumerate(error_lines):
                cv2.putText(
                    frame,
                    text,
                    (34, base_y + line_idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (220, 228, 236),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame)
    finally:
        writer.release()

    mono_missing = float(1.0 - np.mean(np.isfinite(mono_body).all(axis=2)))
    xsens_missing = float(1.0 - np.mean(np.isfinite(xsens_body_aligned).all(axis=2)))
    metadata = {
        "input_trc": str(trc_path),
        "input_mvnx": str(args.mvnx_path),
        "alignment_json": str(args.alignment_json),
        "output_mp4": str(output_mp4),
        "fps": fps,
        "frame_count": int(len(trc_ts)),
        "marker_count": int(len(body_names)),
        "marker_names": body_names,
        "edge_count": int(len(mono_edges)),
        "monocular_missing_rate": mono_missing,
        "xsens_missing_rate": xsens_missing,
        "upright_handling": {
            "monocular": (
                "TRC is assumed upright because the raw upside-down video is rotated 180 degrees "
                "inside RTMDet-MotionBert-OpenSim/run_inference.py before monocular inference."
            ),
            "xsens": "MVNX segment positions are transformed into a Y-up visualization frame for side-by-side comparison.",
        },
        "upright_check_pass": {
            "monocular": bool(mono_upright_ok),
            "xsens": bool(xsens_upright_ok),
        },
        "best_offset_seconds": float(offset_sec),
        "mean_angle_error_overall": float(np.nanmean(mean_angle_error)),
        "azimuth_deg": float(args.azimuth),
        "elevation_deg": float(args.elevation),
        "canvas_size": [int(args.width), int(args.height)],
    }
    with output_meta.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[mono-render] MP4 saved to: {output_mp4}")
    print(f"[mono-render] Metadata saved to: {output_meta}")
    print(f"[mono-render] Upright checks: mono={mono_upright_ok}, xsens={xsens_upright_ok}")


if __name__ == "__main__":
    main()
