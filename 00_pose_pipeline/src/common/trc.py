"""TRC loading and alignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import interp1d

from common.angles import COCO17_NAMES


def load_trc(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, float, str]:
    """Parse a TRC file into timestamps, marker names, positions, fps, units."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"TRC file is too short: {path}")
    header = lines[2].strip().split("\t")
    if len(header) < 5:
        header = lines[2].strip().split()
    fps = float(header[0])
    n_markers = int(header[3])
    units = header[4]
    marker_names = [name.strip() for name in lines[3].rstrip("\n").split("\t")[2:] if name.strip()]
    if len(marker_names) != n_markers:
        marker_names = [name.strip() for name in lines[3].strip().split()[2:] if name.strip()]
    if len(marker_names) != n_markers:
        raise ValueError(f"Marker count mismatch in {path}")

    timestamps = []
    frames = []
    expected = n_markers * 3
    for line in lines[6:]:
        if not line.strip():
            continue
        values = line.rstrip("\n").split("\t")
        if len(values) < 2:
            values = line.strip().split()
        timestamps.append(float(values[1]))
        coords = values[2:]
        if len(coords) < expected:
            coords += [""] * (expected - len(coords))
        frames.append([float(value) if value else np.nan for value in coords[:expected]])
    positions = np.asarray(frames, dtype=np.float64).reshape(-1, n_markers, 3)
    return np.asarray(timestamps, dtype=np.float64), marker_names, positions, fps, units


def unit_to_cm(units: str) -> float:
    """Return multiplicative scale from units to centimeters."""
    value = units.strip().lower()
    if value == "cm":
        return 1.0
    if value == "mm":
        return 0.1
    if value in {"m", "meter", "meters", "metre", "metres"}:
        return 100.0
    raise ValueError(f"Unsupported TRC units: {units}")


def trc_to_coco17(marker_names: list[str], positions_cm: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map TRC markers to COCO-17 order."""
    name_to_idx = {name: idx for idx, name in enumerate(marker_names)}
    keypoints = np.full((positions_cm.shape[0], len(COCO17_NAMES), 3), np.nan, dtype=np.float64)
    missing = []
    for idx, name in enumerate(COCO17_NAMES):
        if name in name_to_idx:
            keypoints[:, idx, :] = positions_cm[:, name_to_idx[name], :]
        else:
            missing.append(name)
    return keypoints, missing


def interpolate_keypoints(source_time: np.ndarray, keypoints: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """Interpolate keypoints onto a target timeline."""
    source_time = np.asarray(source_time, dtype=np.float64)
    source_time = source_time - source_time[0]
    target_time = np.asarray(target_time, dtype=np.float64)
    out = np.full((len(target_time), keypoints.shape[1], keypoints.shape[2]), np.nan, dtype=np.float64)
    unique_time, unique_idx = np.unique(source_time, return_index=True)
    unique_keypoints = keypoints[unique_idx]
    for joint_idx in range(unique_keypoints.shape[1]):
        for axis_idx in range(unique_keypoints.shape[2]):
            values = unique_keypoints[:, joint_idx, axis_idx]
            finite = np.isfinite(unique_time) & np.isfinite(values)
            if np.count_nonzero(finite) < 2:
                continue
            interp = interp1d(unique_time[finite], values[finite], kind="linear", bounds_error=False, fill_value=np.nan)
            out[:, joint_idx, axis_idx] = interp(target_time)
    return out


def load_trc_on_timeline(
    name: str,
    path: Path,
    corrected_time: np.ndarray,
    synced,
    left_rows: list[dict[str, float | int]],
) -> tuple[np.ndarray, dict[str, object]]:
    """Load and align one TRC source to the corrected stereo timeline."""
    timestamps, marker_names, positions, fps, units = load_trc(path)
    keypoints, missing = trc_to_coco17(marker_names, positions * unit_to_cm(units))
    if len(keypoints) == len(corrected_time):
        aligned = keypoints.copy()
        mode = "synced_frame_index"
    elif len(keypoints) == len(left_rows):
        left_indices = np.asarray([int(row.left_idx) for row in synced], dtype=np.int64)
        aligned = keypoints[left_indices]
        mode = "left_metadata_frame_index"
    else:
        aligned = interpolate_keypoints(timestamps, keypoints, corrected_time)
        mode = "trc_timestamp_interpolation"
    valid = np.isfinite(aligned).all(axis=2)
    summary = {
        "name": name,
        "path": str(path),
        "source_frame_count": int(len(keypoints)),
        "aligned_frame_count": int(len(aligned)),
        "source_fps": float(fps),
        "units": units,
        "alignment_mode": mode,
        "missing_coco17_joints": missing,
        "valid_left_elbow_chain_ratio": float(np.mean(valid[:, [5, 7, 9]].all(axis=1))),
        "valid_right_elbow_chain_ratio": float(np.mean(valid[:, [6, 8, 10]].all(axis=1))),
    }
    return aligned, summary
