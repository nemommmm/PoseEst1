"""Utilities for OpenSim MOT parsing and EasyErgo final-output inspection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np


_WHITESPACE_RE = re.compile(r"\s+")


def _split_fields(line: str) -> list[str]:
    """Split one MOT line on arbitrary whitespace."""
    return [field for field in _WHITESPACE_RE.split(line.strip()) if field]


def _resolve_single_file(
    input_dir: Path,
    patterns: Iterable[str],
    preferred_names: Iterable[str] = (),
) -> Path | None:
    """Resolve a single file from a directory using preferred names and globs."""
    for name in preferred_names:
        candidate = input_dir / name
        if candidate.is_file():
            return candidate

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(input_dir.glob(pattern)))

    unique_matches = sorted({path.resolve() for path in matches})
    if not unique_matches:
        return None
    if len(unique_matches) > 1:
        formatted = ", ".join(str(path) for path in unique_matches)
        raise FileNotFoundError(
            f"Multiple candidate files found in {input_dir}: {formatted}"
        )
    return unique_matches[0]


def resolve_easyergo_final_outputs(
    input_dir: str | Path,
    mot_override: str | None = None,
    osim_override: str | None = None,
) -> dict[str, Path | None]:
    """Resolve EasyErgo final-output files from the drop folder."""
    folder = Path(input_dir).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"EasyErgo input folder not found: {folder}")

    mot_path = Path(mot_override).resolve() if mot_override else _resolve_single_file(
        folder,
        patterns=("*_ik.mot", "*.mot"),
        preferred_names=("markers_easyergo_ik.mot",),
    )
    osim_path = Path(osim_override).resolve() if osim_override else _resolve_single_file(
        folder,
        patterns=("*_model.osim", "*.osim"),
        preferred_names=("markers_easyergo_model.osim",),
    )
    return {
        "input_dir": folder,
        "mot_path": mot_path,
        "osim_path": osim_path,
    }


def load_opensim_mot(mot_path: str | Path) -> dict[str, object]:
    """Load an OpenSim MOT file into time-series coordinate arrays."""
    path = Path(mot_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MOT file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    in_degrees = True
    header_idx = None
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if "inDegrees=yes" in line:
            in_degrees = True
        elif "inDegrees=no" in line:
            in_degrees = False
        if line.startswith("time"):
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError(f"Could not find MOT data header in {path}")

    headers = _split_fields(lines[header_idx])
    if not headers or headers[0] != "time":
        raise ValueError(f"Unexpected MOT header format in {path}: {headers}")

    coord_names = headers[1:]
    data_rows: list[list[float]] = []
    for raw_line in lines[header_idx + 1 :]:
        if not raw_line.strip():
            continue
        fields = _split_fields(raw_line)
        if len(fields) != len(headers):
            continue
        try:
            data_rows.append([float(value) for value in fields])
        except ValueError:
            continue

    if not data_rows:
        raise ValueError(f"No numeric MOT rows parsed from {path}")

    data = np.asarray(data_rows, dtype=np.float64)
    time = data[:, 0]
    coordinates = {
        name: data[:, idx + 1].copy() for idx, name in enumerate(coord_names)
    }

    if not in_degrees:
        for name, values in coordinates.items():
            if name.startswith("pelvis_t"):
                continue
            coordinates[name] = np.degrees(values)

    return {
        "path": path,
        "time": time,
        "coordinates": coordinates,
        "coord_names": coord_names,
        "in_degrees": True,
        "num_frames": int(len(time)),
    }


def _series_from_candidates(
    coordinates: dict[str, np.ndarray],
    candidates: Iterable[str],
    length: int,
) -> tuple[np.ndarray, str | None]:
    """Return the first available coordinate series among several aliases."""
    for name in candidates:
        if name in coordinates:
            return np.asarray(coordinates[name], dtype=np.float64), name
    return np.full(length, np.nan, dtype=np.float64), None


def build_semantic_angles_from_mot(
    coordinates: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
    """Map OpenSim MOT coordinates into the local ergonomic angle set."""
    if not coordinates:
        raise ValueError("MOT coordinate dictionary is empty.")

    first_key = next(iter(coordinates))
    num_frames = len(coordinates[first_key])

    arm_flex_l, arm_flex_l_name = _series_from_candidates(coordinates, ("arm_flex_l",), num_frames)
    arm_add_l, arm_add_l_name = _series_from_candidates(coordinates, ("arm_add_l",), num_frames)
    arm_flex_r, arm_flex_r_name = _series_from_candidates(coordinates, ("arm_flex_r",), num_frames)
    arm_add_r, arm_add_r_name = _series_from_candidates(coordinates, ("arm_add_r",), num_frames)
    elbow_l, elbow_l_name = _series_from_candidates(coordinates, ("elbow_flex_l",), num_frames)
    elbow_r, elbow_r_name = _series_from_candidates(coordinates, ("elbow_flex_r",), num_frames)
    hip_l, hip_l_name = _series_from_candidates(coordinates, ("hip_flexion_l",), num_frames)
    hip_r, hip_r_name = _series_from_candidates(coordinates, ("hip_flexion_r",), num_frames)
    knee_l, knee_l_name = _series_from_candidates(coordinates, ("knee_angle_l",), num_frames)
    knee_r, knee_r_name = _series_from_candidates(coordinates, ("knee_angle_r",), num_frames)
    pelvis_tilt, pelvis_tilt_name = _series_from_candidates(coordinates, ("pelvis_tilt",), num_frames)
    lumbar_ext, lumbar_ext_name = _series_from_candidates(
        coordinates,
        ("lumbar_extension", "L5_S1_Flex_Ext"),
        num_frames,
    )

    trunk_proxy = np.full(num_frames, np.nan, dtype=np.float64)
    pelvis_ok = np.isfinite(pelvis_tilt)
    lumbar_ok = np.isfinite(lumbar_ext)
    both_ok = pelvis_ok & lumbar_ok
    trunk_proxy[both_ok] = np.abs(pelvis_tilt[both_ok] + lumbar_ext[both_ok])
    trunk_proxy[pelvis_ok & ~lumbar_ok] = np.abs(pelvis_tilt[pelvis_ok & ~lumbar_ok])
    trunk_proxy[lumbar_ok & ~pelvis_ok] = np.abs(lumbar_ext[lumbar_ok & ~pelvis_ok])

    semantic = {
        "LeftShoulder": np.hypot(arm_flex_l, arm_add_l),
        "RightShoulder": np.hypot(arm_flex_r, arm_add_r),
        "LeftElbow": np.abs(elbow_l),
        "RightElbow": np.abs(elbow_r),
        "LeftHip": hip_l.copy(),
        "RightHip": hip_r.copy(),
        "LeftKnee": np.abs(knee_l),
        "RightKnee": np.abs(knee_r),
        "TrunkFlexionProxy": trunk_proxy,
    }
    sources = {
        "LeftShoulder": "+".join(
            name for name in (arm_flex_l_name, arm_add_l_name) if name is not None
        ) or None,
        "RightShoulder": "+".join(
            name for name in (arm_flex_r_name, arm_add_r_name) if name is not None
        ) or None,
        "LeftElbow": elbow_l_name,
        "RightElbow": elbow_r_name,
        "LeftHip": hip_l_name,
        "RightHip": hip_r_name,
        "LeftKnee": knee_l_name,
        "RightKnee": knee_r_name,
        "TrunkFlexionProxy": "+".join(
            name for name in (pelvis_tilt_name, lumbar_ext_name) if name is not None
        ) or None,
    }
    return semantic, sources


def finite_ratio(values: np.ndarray) -> float:
    """Return the finite-value coverage ratio of one series."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(array)))


def summarize_coordinate_range(values: np.ndarray) -> dict[str, float] | None:
    """Summarize one coordinate series for inspection."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }
