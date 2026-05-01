#!/opt/anaconda3/envs/pose/bin/python
"""Compute frame-to-frame elbow angle deltas on a shared video timeline.

The evaluation intentionally avoids absolute angle MAE.  Each system is first
sampled onto the same corrected stereo-video time axis, then elbow flexion
angles are optionally smoothed and differenced frame-to-frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from pose_angle_utils import (  # noqa: E402
    SEMANTIC_ANGLE_NAMES,
    build_fair_gt_interpolators,
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
    median_filter_angle_sequence,
)
from utils_mvnx import MvnxParser  # noqa: E402

SIDES = ("LeftElbow", "RightElbow")
SYSTEMS = ("SKT", "AFH", "XsensFair", "XsensNative")
ANGLE_COLUMNS = ("LeftElbow_deg", "RightElbow_deg")

DEFAULT_SKT_NPZ = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
DEFAULT_AFH_NPZ = PROJECT_ROOT / "04_hybrid_afh1" / "results" / "hybrid_skeleton_afh1_v1.npz"
DEFAULT_FAIR_GT = SHARED_DIR / "fair_gt_angles.npz"
DEFAULT_MVNX = PROJECT_ROOT.parent / "Xsens_ground_truth" / "Aitor-001.mvnx"
DEFAULT_ALIGNMENT_SUMMARY = PROJECT_ROOT / "01_stereo_triangulation" / "results" / "alignment_summary.json"
DEFAULT_LEFT_TXT = PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.txt"
DEFAULT_RIGHT_TXT = PROJECT_ROOT / "2025_Ergonomics_Data" / "1_video_right.txt"
DEFAULT_OUT_DIR = METHOD_DIR / "results"


@dataclass
class SystemSeries:
    """Container for one system's shared-timeline elbow angles and deltas."""

    name: str
    angles: Dict[str, np.ndarray]
    deltas: Dict[str, np.ndarray]
    valid_angles: Dict[str, np.ndarray]
    interpolated: Dict[str, np.ndarray]
    delta_valid: Dict[str, np.ndarray]
    delta_anomaly: Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skt-npz", default=str(DEFAULT_SKT_NPZ))
    parser.add_argument("--afh-npz", default=str(DEFAULT_AFH_NPZ))
    parser.add_argument("--xsens-fair", default=str(DEFAULT_FAIR_GT))
    parser.add_argument("--xsens-mvnx", default=str(DEFAULT_MVNX))
    parser.add_argument("--alignment-summary", default=str(DEFAULT_ALIGNMENT_SUMMARY))
    parser.add_argument("--left-meta", default=str(DEFAULT_LEFT_TXT))
    parser.add_argument("--right-meta", default=str(DEFAULT_RIGHT_TXT))
    parser.add_argument("--xsens-offset", type=float, default=None,
                        help="Override Xsens temporal offset in seconds.")
    parser.add_argument("--offset-source", choices=("position", "angle", "best"), default="position",
                        help="Which alignment_summary offset to use when --xsens-offset is not set.")
    parser.add_argument("--smooth-radius", type=int, default=2,
                        help="Median smoothing radius on the shared video timeline; 0 disables smoothing.")
    parser.add_argument("--max-gap-frames", type=int, default=5,
                        help="Maximum NaN run length to fill by linear interpolation before smoothing.")
    parser.add_argument("--anomaly-delta-deg", type=float, default=60.0,
                        help="Flag one-frame delta magnitudes above this threshold.")
    parser.add_argument("--start-time", type=float, default=None,
                        help="Optional start time on the corrected video timeline.")
    parser.add_argument("--end-time", type=float, default=None,
                        help="Optional end time on the corrected video timeline.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--skip-plots", action="store_true",
                        help="Only write CSV/JSON; do not invoke 02_plot_delta_curves.py.")
    return parser.parse_args()


def parse_meta_timestamp(parts: List[str]) -> float:
    """Parse metadata seconds + microseconds without losing leading zeros."""
    seconds = int(parts[1])
    micros = int(parts[2])
    return seconds + micros * 1e-6


def parse_stereo_meta(path: Path) -> List[Dict[str, float | int]]:
    """Parse one stereo metadata txt file with corrected microsecond handling."""
    rows: List[Dict[str, float | int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append({"id": int(parts[0]), "ts": parse_meta_timestamp(parts)})
            except ValueError:
                continue
    return rows


def build_synced_video_timeline(left_meta: Path, right_meta: Path) -> Tuple[np.ndarray, List[Dict[str, int | float]]]:
    """Build the corrected shared stereo-video timeline from hardware frame IDs."""
    left_rows = parse_stereo_meta(left_meta)
    right_rows = parse_stereo_meta(right_meta)
    synced: List[Dict[str, int | float]] = []
    ptr_l = 0
    ptr_r = 0
    while ptr_l < len(left_rows) and ptr_r < len(right_rows):
        row_l = left_rows[ptr_l]
        row_r = right_rows[ptr_r]
        id_l = int(row_l["id"])
        id_r = int(row_r["id"])
        if id_l == id_r:
            synced.append({
                "frame_id": id_l,
                "left_idx": ptr_l,
                "right_idx": ptr_r,
                "ts": float(row_l["ts"]),
            })
            ptr_l += 1
            ptr_r += 1
        elif id_l < id_r:
            ptr_l += 1
        else:
            ptr_r += 1
    if not synced:
        raise RuntimeError("No synchronized stereo metadata pairs found.")
    abs_ts = np.array([float(row["ts"]) for row in synced], dtype=np.float64)
    rel_ts = abs_ts - abs_ts[0]
    non_monotonic = int(np.sum(np.diff(rel_ts) <= 0))
    if non_monotonic:
        raise RuntimeError(f"Corrected video timeline is not monotonic ({non_monotonic} non-positive diffs).")
    return rel_ts, synced


def original_timestamp_diagnostics(npz_path: Path) -> Dict[str, float | int]:
    """Summarize the old NPZ timestamp vector for transparency."""
    payload = np.load(npz_path, allow_pickle=True)
    ts = np.asarray(payload["timestamps"], dtype=np.float64)
    rel = ts - ts[0]
    diffs = np.diff(rel)
    return {
        "frame_count": int(len(ts)),
        "duration_s": float(rel[-1] - rel[0]) if len(rel) else 0.0,
        "nonpositive_diff_count": int(np.sum(diffs <= 0)),
        "diff_min_s": float(np.nanmin(diffs)) if len(diffs) else math.nan,
        "diff_median_s": float(np.nanmedian(diffs)) if len(diffs) else math.nan,
        "diff_max_s": float(np.nanmax(diffs)) if len(diffs) else math.nan,
    }


def load_offset(summary_path: Path, source: str, override: Optional[float]) -> float:
    """Load the Xsens temporal offset."""
    if override is not None:
        return float(override)
    fallback = 17.25
    if not summary_path.exists():
        return fallback
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if source == "position":
        return float(payload.get("position_best_offset_seconds", payload.get("best_offset_seconds", fallback)))
    if source == "angle":
        return float(payload.get("angle_refined_offset_seconds", payload.get("best_offset_seconds", fallback)))
    return float(payload.get("best_offset_seconds", fallback))


def truncate_to_pose_length(
    timeline: np.ndarray,
    synced: List[Dict[str, int | float]],
    n_frames: int,
) -> Tuple[np.ndarray, List[Dict[str, int | float]]]:
    """Use the same synchronized-pair order as the NPZ processing pipeline."""
    if len(timeline) < n_frames:
        raise RuntimeError(f"Video timeline has {len(timeline)} frames but pose NPZ has {n_frames}.")
    if len(timeline) > n_frames:
        timeline = timeline[:n_frames]
        synced = synced[:n_frames]
    return timeline, synced


def apply_time_window(
    time_s: np.ndarray,
    synced: List[Dict[str, int | float]],
    arrays: Iterable[np.ndarray],
    start_time: Optional[float],
    end_time: Optional[float],
) -> Tuple[np.ndarray, List[Dict[str, int | float]], List[np.ndarray], np.ndarray]:
    """Filter time-aligned arrays to one optional contiguous time window."""
    mask = np.ones(len(time_s), dtype=bool)
    if start_time is not None:
        mask &= time_s >= float(start_time)
    if end_time is not None:
        mask &= time_s <= float(end_time)
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise RuntimeError("Selected time window contains no frames.")
    filtered_time = time_s[indices]
    filtered_synced = [synced[int(idx)] for idx in indices]
    filtered_arrays = [np.asarray(arr)[indices] for arr in arrays]
    return filtered_time, filtered_synced, filtered_arrays, indices


def compute_pose_elbow_angles(keypoints: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute geometric elbow flexion angles for one 3D keypoint sequence."""
    names, values = compute_semantic_angle_sequence(keypoints, wrist_smooth_radius=0)
    name_to_idx = {name: i for i, name in enumerate(names)}
    return {side: values[:, name_to_idx[side]].astype(np.float64) for side in SIDES}


def build_xsens_fair_angles(
    fair_path: Path,
    video_time_s: np.ndarray,
    xsens_offset_s: float,
) -> Dict[str, np.ndarray]:
    """Sample Xsens-derived geometric fair angles onto the video timeline."""
    interps = build_fair_gt_interpolators(str(fair_path))
    if not interps:
        raise RuntimeError(f"Could not load fair GT angles from {fair_path}")
    query_t = video_time_s - xsens_offset_s
    return {side: np.asarray(interps[side](query_t), dtype=np.float64) for side in SIDES}


def build_xsens_native_angles(
    mvnx_path: Path,
    video_time_s: np.ndarray,
    xsens_offset_s: float,
) -> Dict[str, np.ndarray]:
    """Sample Xsens native elbow jointAngle values onto the video timeline."""
    mvnx = MvnxParser(str(mvnx_path))
    mvnx.parse()
    xsens_ts = np.asarray(mvnx.timestamps, dtype=np.float64)
    xsens_ts, unique_idx = np.unique(xsens_ts, return_index=True)
    xsens_ts = xsens_ts - xsens_ts[0]
    native_interps = build_gt_angle_interpolators(
        mvnx,
        xsens_ts,
        unique_idx,
        specs={side: spec for side, spec in {
            "LeftElbow": {"source": "joint", "label": "jLeftElbow", "axis": 2, "sign": 1.0},
            "RightElbow": {"source": "joint", "label": "jRightElbow", "axis": 2, "sign": 1.0},
        }.items()},
    )
    query_t = video_time_s - xsens_offset_s
    return {side: np.asarray(native_interps[side](query_t), dtype=np.float64) for side in SIDES}


def interpolate_short_gaps(
    values: np.ndarray,
    time_s: np.ndarray,
    max_gap_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill finite-bounded NaN runs up to max_gap_frames."""
    values = np.asarray(values, dtype=np.float64)
    filled = values.copy()
    flags = np.zeros(len(values), dtype=bool)
    if max_gap_frames <= 0 or len(values) == 0:
        return filled, flags
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return filled, flags

    idx = 0
    while idx < len(values):
        if finite[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(values) and not finite[idx]:
            idx += 1
        end = idx
        gap_len = end - start
        left = start - 1
        right = end
        if left >= 0 and right < len(values) and gap_len <= max_gap_frames:
            filled[start:end] = np.interp(time_s[start:end], [time_s[left], time_s[right]], [values[left], values[right]])
            flags[start:end] = True
    return filled, flags


def smooth_angles_on_shared_timeline(angles: Dict[str, np.ndarray], radius: int) -> Dict[str, np.ndarray]:
    """Apply the project's median angle filter after common-timeline resampling."""
    matrix = np.full((len(next(iter(angles.values()))), len(SEMANTIC_ANGLE_NAMES)), np.nan, dtype=np.float64)
    name_to_idx = {name: i for i, name in enumerate(SEMANTIC_ANGLE_NAMES)}
    for side in SIDES:
        matrix[:, name_to_idx[side]] = angles[side]
    smoothed = median_filter_angle_sequence(matrix, radius=max(0, radius))
    return {side: smoothed[:, name_to_idx[side]] for side in SIDES}


def build_system_series(
    name: str,
    raw_angles: Dict[str, np.ndarray],
    time_s: np.ndarray,
    smooth_radius: int,
    max_gap_frames: int,
    anomaly_delta_deg: float,
) -> SystemSeries:
    """Interpolate short gaps, smooth, and difference one system's elbow angles."""
    filled_angles: Dict[str, np.ndarray] = {}
    interpolated: Dict[str, np.ndarray] = {}
    original_valid: Dict[str, np.ndarray] = {}
    for side in SIDES:
        clean = np.asarray(raw_angles[side], dtype=np.float64)
        out_of_range = np.isfinite(clean) & ((clean < -1e-6) | (clean > 180.0 + 1e-6))
        if np.any(out_of_range):
            clean = clean.copy()
            clean[out_of_range] = np.nan
        filled, flags = interpolate_short_gaps(clean, time_s, max_gap_frames=max_gap_frames)
        filled_angles[side] = filled
        interpolated[side] = flags
        original_valid[side] = np.isfinite(clean)

    smoothed_angles = smooth_angles_on_shared_timeline(filled_angles, smooth_radius)
    deltas: Dict[str, np.ndarray] = {}
    delta_valid: Dict[str, np.ndarray] = {}
    delta_anomaly: Dict[str, np.ndarray] = {}
    valid_angles: Dict[str, np.ndarray] = {}
    for side in SIDES:
        angle = smoothed_angles[side].copy()
        angle[~np.isfinite(filled_angles[side])] = np.nan
        valid = np.isfinite(angle)
        delta = np.full_like(angle, np.nan, dtype=np.float64)
        valid_delta = valid.copy()
        valid_delta[0] = False
        valid_delta[1:] = valid[1:] & valid[:-1]
        delta[1:] = angle[1:] - angle[:-1]
        anomaly = np.zeros(len(angle), dtype=bool)
        anomaly[valid_delta] = np.abs(delta[valid_delta]) > anomaly_delta_deg
        deltas[side] = delta
        delta_valid[side] = valid_delta
        delta_anomaly[side] = anomaly
        valid_angles[side] = valid

    return SystemSeries(
        name=name,
        angles=smoothed_angles,
        deltas=deltas,
        valid_angles=valid_angles,
        interpolated=interpolated,
        delta_valid=delta_valid,
        delta_anomaly=delta_anomaly,
    )


def finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite paired samples."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Pearson correlation with guardrails."""
    x_f, y_f = finite_pair(x, y)
    if len(x_f) < 3 or np.nanstd(x_f) < 1e-9 or np.nanstd(y_f) < 1e-9:
        return None
    return float(np.corrcoef(x_f, y_f)[0, 1])


def regression_slope(reference: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Least-squares target = slope * reference + intercept slope."""
    x, y = finite_pair(reference, target)
    if len(x) < 3 or np.nanvar(x) < 1e-12:
        return None
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def mean_abs(values: np.ndarray) -> Optional[float]:
    """Finite mean absolute value."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(np.mean(np.abs(finite)))


def rmse(values: np.ndarray) -> Optional[float]:
    """Finite root mean squared value."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(np.sqrt(np.mean(finite**2)))


def side_system_summary(series: SystemSeries, side: str) -> Dict[str, Optional[float] | int]:
    """Summarize one system for one elbow."""
    angle = series.angles[side]
    delta = series.deltas[side]
    valid_delta = series.delta_valid[side]
    finite_angle = angle[np.isfinite(angle)]
    finite_delta = delta[valid_delta & np.isfinite(delta)]
    return {
        "valid_angle_ratio": float(np.isfinite(angle).mean()) if len(angle) else None,
        "valid_delta_ratio": float((valid_delta & np.isfinite(delta)).mean()) if len(delta) else None,
        "interpolated_frame_count": int(series.interpolated[side].sum()),
        "delta_anomaly_count": int(series.delta_anomaly[side].sum()),
        "rom_deg": float(np.nanmax(finite_angle) - np.nanmin(finite_angle)) if len(finite_angle) else None,
        "total_path_deg": float(np.sum(np.abs(finite_delta))) if len(finite_delta) else None,
        "signed_net_change_deg": float(np.sum(finite_delta)) if len(finite_delta) else None,
    }


def pair_summary(target: SystemSeries, reference: SystemSeries, side: str) -> Dict[str, Optional[float] | int]:
    """Summarize motion agreement between one target and one reference."""
    target_delta = target.deltas[side].copy()
    ref_delta = reference.deltas[side].copy()
    mask = (
        target.delta_valid[side]
        & reference.delta_valid[side]
        & np.isfinite(target_delta)
        & np.isfinite(ref_delta)
    )
    td = target_delta[mask]
    rd = ref_delta[mask]
    diff = td - rd
    target_path = float(np.sum(np.abs(td))) if len(td) else None
    ref_path = float(np.sum(np.abs(rd))) if len(rd) else None
    target_rom = side_system_summary(target, side)["rom_deg"]
    ref_rom = side_system_summary(reference, side)["rom_deg"]
    if target_path is None or ref_path is None or ref_path <= 1e-9:
        path_ratio = None
    else:
        path_ratio = target_path / ref_path
    if target_rom is None or ref_rom is None or max(target_rom, ref_rom) <= 1e-9:
        rom_match = None
    else:
        rom_match = min(float(target_rom), float(ref_rom)) / max(float(target_rom), float(ref_rom))
    return {
        "valid_pair_count": int(mask.sum()),
        "pearson_delta": pearson(rd, td),
        "slope_target_vs_reference": regression_slope(rd, td),
        "delta_mae_deg": mean_abs(diff),
        "delta_rmse_deg": rmse(diff),
        "target_path_deg": target_path,
        "reference_path_deg": ref_path,
        "path_ratio_target_reference": path_ratio,
        "target_rom_deg": target_rom,
        "reference_rom_deg": ref_rom,
        "rom_match_min_over_max": rom_match,
    }


def round_jsonable(value):
    """Round floats recursively for compact JSON."""
    if isinstance(value, dict):
        return {key: round_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [round_jsonable(val) for val in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    return value


def maybe_float(value: float) -> str:
    """CSV float formatter preserving NaN as empty string."""
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.6f}"


def write_single_system_csv(out_path: Path, time_s: np.ndarray, series: SystemSeries) -> None:
    """Write per-system angle/delta CSV."""
    fieldnames = [
        "Frame", "Time_s",
        "LeftElbow_deg", "RightElbow_deg",
        "LeftElbow_delta_deg", "RightElbow_delta_deg",
        "LeftElbow_valid", "RightElbow_valid",
        "LeftElbow_interpolated", "RightElbow_interpolated",
        "LeftElbow_delta_anomaly_flag", "RightElbow_delta_anomaly_flag",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t in enumerate(time_s):
            writer.writerow({
                "Frame": idx,
                "Time_s": maybe_float(t),
                "LeftElbow_deg": maybe_float(series.angles["LeftElbow"][idx]),
                "RightElbow_deg": maybe_float(series.angles["RightElbow"][idx]),
                "LeftElbow_delta_deg": maybe_float(series.deltas["LeftElbow"][idx]),
                "RightElbow_delta_deg": maybe_float(series.deltas["RightElbow"][idx]),
                "LeftElbow_valid": bool(series.valid_angles["LeftElbow"][idx]),
                "RightElbow_valid": bool(series.valid_angles["RightElbow"][idx]),
                "LeftElbow_interpolated": bool(series.interpolated["LeftElbow"][idx]),
                "RightElbow_interpolated": bool(series.interpolated["RightElbow"][idx]),
                "LeftElbow_delta_anomaly_flag": bool(series.delta_anomaly["LeftElbow"][idx]),
                "RightElbow_delta_anomaly_flag": bool(series.delta_anomaly["RightElbow"][idx]),
            })


def write_combined_csv(
    out_path: Path,
    time_s: np.ndarray,
    synced_meta: List[Dict[str, int | float]],
    series_by_name: Dict[str, SystemSeries],
) -> None:
    """Write all systems on the shared video timeline."""
    fieldnames = ["Frame", "Time_s", "FrameDt_s", "StereoFrameId", "LeftVideoFrame", "RightVideoFrame"]
    for system in SYSTEMS:
        for side in SIDES:
            fieldnames.extend([
                f"{system}_{side}_deg",
                f"{system}_{side}_delta_deg",
                f"{system}_{side}_valid",
                f"{system}_{side}_interpolated",
                f"{system}_{side}_delta_anomaly_flag",
            ])
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t in enumerate(time_s):
            row = {
                "Frame": idx,
                "Time_s": maybe_float(t),
                "FrameDt_s": "" if idx == 0 else maybe_float(time_s[idx] - time_s[idx - 1]),
                "StereoFrameId": int(synced_meta[idx]["frame_id"]),
                "LeftVideoFrame": int(synced_meta[idx]["left_idx"]),
                "RightVideoFrame": int(synced_meta[idx]["right_idx"]),
            }
            for system, series in series_by_name.items():
                for side in SIDES:
                    row[f"{system}_{side}_deg"] = maybe_float(series.angles[side][idx])
                    row[f"{system}_{side}_delta_deg"] = maybe_float(series.deltas[side][idx])
                    row[f"{system}_{side}_valid"] = bool(series.valid_angles[side][idx])
                    row[f"{system}_{side}_interpolated"] = bool(series.interpolated[side][idx])
                    row[f"{system}_{side}_delta_anomaly_flag"] = bool(series.delta_anomaly[side][idx])
            writer.writerow(row)


def build_summary(
    args: argparse.Namespace,
    time_s: np.ndarray,
    corrected_meta: List[Dict[str, int | float]],
    selected_indices: np.ndarray,
    xsens_offset_s: float,
    original_ts_diag: Dict[str, float | int],
    series_by_name: Dict[str, SystemSeries],
) -> Dict:
    """Build JSON summary of motion agreement."""
    dt = np.diff(time_s)
    summary = {
        "config": {
            "skt_npz": str(Path(args.skt_npz)),
            "afh_npz": str(Path(args.afh_npz)),
            "xsens_fair": str(Path(args.xsens_fair)),
            "xsens_mvnx": str(Path(args.xsens_mvnx)),
            "xsens_offset_s": float(xsens_offset_s),
            "offset_source": args.offset_source if args.xsens_offset is None else "override",
            "smooth_radius_frames": int(args.smooth_radius),
            "max_gap_frames": int(args.max_gap_frames),
            "anomaly_delta_deg": float(args.anomaly_delta_deg),
            "start_time_s": args.start_time,
            "end_time_s": args.end_time,
        },
        "timeline": {
            "frame_count": int(len(time_s)),
            "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) else 0.0,
            "median_dt_s": float(np.nanmedian(dt)) if len(dt) else None,
            "effective_fps": float(1.0 / np.nanmedian(dt)) if len(dt) and np.nanmedian(dt) > 0 else None,
            "corrected_timestamp_nonpositive_diff_count": int(np.sum(dt <= 0)),
            "first_stereo_frame_id": int(corrected_meta[0]["frame_id"]),
            "last_stereo_frame_id": int(corrected_meta[-1]["frame_id"]),
            "first_original_frame_index": int(selected_indices[0]),
            "last_original_frame_index": int(selected_indices[-1]),
            "old_npz_timestamp_diagnostics": original_ts_diag,
        },
        "systems": {},
        "motion_agreement": {},
    }
    for side in SIDES:
        summary["systems"][side] = {
            system: side_system_summary(series, side)
            for system, series in series_by_name.items()
        }
        summary["motion_agreement"][side] = {
            "SKT_vs_XsensFair": pair_summary(series_by_name["SKT"], series_by_name["XsensFair"], side),
            "AFH_vs_XsensFair": pair_summary(series_by_name["AFH"], series_by_name["XsensFair"], side),
            "SKT_vs_AFH": pair_summary(series_by_name["SKT"], series_by_name["AFH"], side),
            "XsensNative_vs_XsensFair": pair_summary(series_by_name["XsensNative"], series_by_name["XsensFair"], side),
        }
    return round_jsonable(summary)


def run_plot_script(out_dir: Path) -> None:
    """Invoke the plotting script with the project Python interpreter."""
    import subprocess

    plot_script = SCRIPT_DIR / "02_plot_delta_curves.py"
    if not plot_script.exists():
        return
    cmd = [
        sys.executable,
        str(plot_script),
        "--combined-csv",
        str(out_dir / "elbow_delta_combined.csv"),
        "--summary-json",
        str(out_dir / "elbow_delta_summary.json"),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    """Run elbow motion-delta evaluation."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skt_payload = np.load(args.skt_npz, allow_pickle=True)
    afh_payload = np.load(args.afh_npz, allow_pickle=True)
    skt_kp = np.asarray(skt_payload["keypoints"], dtype=np.float64)
    afh_kp = np.asarray(afh_payload["keypoints"], dtype=np.float64)
    if skt_kp.shape != afh_kp.shape:
        raise RuntimeError(f"SKT keypoints {skt_kp.shape} and AFH keypoints {afh_kp.shape} do not match.")

    corrected_time, synced_meta = build_synced_video_timeline(Path(args.left_meta), Path(args.right_meta))
    corrected_time, synced_meta = truncate_to_pose_length(corrected_time, synced_meta, len(skt_kp))
    corrected_time, synced_meta, filtered_arrays, selected_indices = apply_time_window(
        corrected_time,
        synced_meta,
        [skt_kp, afh_kp],
        args.start_time,
        args.end_time,
    )
    skt_kp, afh_kp = filtered_arrays
    xsens_offset_s = load_offset(Path(args.alignment_summary), args.offset_source, args.xsens_offset)

    duration = float(corrected_time[-1] - corrected_time[0]) if len(corrected_time) else 0.0
    print(f"[timeline] frames={len(corrected_time)}, duration={duration:.3f}s, "
          f"median dt={np.median(np.diff(corrected_time)):.4f}s")
    print(f"[xsens] offset={xsens_offset_s:.3f}s ({args.offset_source if args.xsens_offset is None else 'override'})")

    raw_angles = {
        "SKT": compute_pose_elbow_angles(skt_kp),
        "AFH": compute_pose_elbow_angles(afh_kp),
        "XsensFair": build_xsens_fair_angles(Path(args.xsens_fair), corrected_time, xsens_offset_s),
        "XsensNative": build_xsens_native_angles(Path(args.xsens_mvnx), corrected_time, xsens_offset_s),
    }
    series_by_name = {
        name: build_system_series(
            name=name,
            raw_angles=angles,
            time_s=corrected_time,
            smooth_radius=args.smooth_radius,
            max_gap_frames=args.max_gap_frames,
            anomaly_delta_deg=args.anomaly_delta_deg,
        )
        for name, angles in raw_angles.items()
    }

    file_map = {
        "SKT": "elbow_angles_skt.csv",
        "AFH": "elbow_angles_afh.csv",
        "XsensFair": "elbow_angles_xsens_fair.csv",
        "XsensNative": "elbow_angles_xsens_native.csv",
    }
    for name, series in series_by_name.items():
        write_single_system_csv(out_dir / file_map[name], corrected_time, series)
    write_combined_csv(out_dir / "elbow_delta_combined.csv", corrected_time, synced_meta, series_by_name)

    summary = build_summary(
        args=args,
        time_s=corrected_time,
        corrected_meta=synced_meta,
        selected_indices=selected_indices,
        xsens_offset_s=xsens_offset_s,
        original_ts_diag=original_timestamp_diagnostics(Path(args.skt_npz)),
        series_by_name=series_by_name,
    )
    (out_dir / "elbow_delta_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[saved]", out_dir / "elbow_delta_combined.csv")
    print("[saved]", out_dir / "elbow_delta_summary.json")
    for side in SIDES:
        skt_pair = summary["motion_agreement"][side]["SKT_vs_XsensFair"]
        afh_pair = summary["motion_agreement"][side]["AFH_vs_XsensFair"]
        print(
            f"[{side}] SKT Pearson={skt_pair['pearson_delta']} "
            f"slope={skt_pair['slope_target_vs_reference']} path_ratio={skt_pair['path_ratio_target_reference']}; "
            f"AFH Pearson={afh_pair['pearson_delta']} slope={afh_pair['slope_target_vs_reference']}"
        )

    if not args.skip_plots:
        run_plot_script(out_dir)


if __name__ == "__main__":
    main()
