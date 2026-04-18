"""Inspect an EasyErgo TRC file before hybrid processing."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
INPUT_DIR = AFH1_DIR / "data" / "easyergo_uploaded"
DEFAULT_TRC_PATH = INPUT_DIR / "markers_easyergo.trc"
RESULTS_DIR = AFH1_DIR / "results"
JSON_OUT = RESULTS_DIR / "easyergo_trc_inspection.json"
MD_OUT = RESULTS_DIR / "easyergo_trc_inspection.md"

AXIS_NAMES = ("X", "Y", "Z")

KEYWORD_GROUPS = {
    "pelvis": ("pelvis", "root", "hipcenter", "hip_center", "midhip", "mid_hip"),
    "head": ("head", "nose"),
    "left_shoulder": ("lshoulder", "leftshoulder", "shoulder_l", "left_shoulder"),
    "right_shoulder": ("rshoulder", "rightshoulder", "shoulder_r", "right_shoulder"),
}


@dataclass
class TRCData:
    """Container for parsed TRC content."""

    fps: float
    units: str
    marker_names: List[str]
    timestamps: np.ndarray
    positions: np.ndarray


def resolve_trc_path() -> Path:
    """Resolve the EasyErgo TRC path from the upload folder."""
    if DEFAULT_TRC_PATH.is_file():
        return DEFAULT_TRC_PATH

    candidates = sorted(INPUT_DIR.glob("*.trc"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No TRC file found in {INPUT_DIR}. "
            "Expected markers_easyergo.trc or exactly one *.trc file."
        )
    raise FileNotFoundError(
        f"Multiple TRC files found in {INPUT_DIR}; please keep only one upload result."
    )


def load_trc(trc_path: Path) -> TRCData:
    """Load a TRC file into a structured object."""
    if not trc_path.is_file():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    with trc_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if len(lines) < 7:
        raise ValueError(f"TRC file is too short to parse: {trc_path}")

    header_values = lines[2].strip().split("\t")
    if len(header_values) < 5:
        raise ValueError("Unexpected TRC header format on line 3.")

    fps = float(header_values[0])
    num_frames = int(header_values[2])
    num_markers = int(header_values[3])
    units = header_values[4]

    raw_names = lines[3].strip().split("\t")[2:]
    marker_names = [name for name in raw_names if name.strip()]
    if len(marker_names) != num_markers:
        raise ValueError(
            f"Marker count mismatch: header={num_markers}, names={len(marker_names)}"
        )

    data_lines = [line.strip() for line in lines[6:] if line.strip()]
    timestamps: List[float] = []
    frames: List[List[float]] = []

    for line in data_lines:
        values = line.split("\t")
        timestamps.append(float(values[1]))
        coords = [float(value) if value else math.nan for value in values[2:]]
        frames.append(coords)

    data = np.asarray(frames, dtype=np.float64)
    positions = data[:, : num_markers * 3].reshape(-1, num_markers, 3)

    if len(timestamps) != num_frames:
        print(
            "[warn] Header frame count does not match parsed frames: "
            f"{num_frames} vs {len(timestamps)}"
        )

    return TRCData(
        fps=fps,
        units=units,
        marker_names=marker_names,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        positions=positions,
    )


def summarize_axis(values: np.ndarray) -> Dict[str, float]:
    """Summarize one coordinate axis across all markers and frames."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "range": math.nan,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "range": float(np.max(finite) - np.min(finite)),
    }


def normalize_name(name: str) -> str:
    """Normalize a marker name for keyword matching."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_keyword_matches(marker_names: Iterable[str]) -> Dict[str, List[str]]:
    """Find likely semantic markers using simple keyword rules."""
    matches: Dict[str, List[str]] = {key: [] for key in KEYWORD_GROUPS}
    for name in marker_names:
        normalized = normalize_name(name)
        for label, keywords in KEYWORD_GROUPS.items():
            if any(keyword in normalized for keyword in keywords):
                matches[label].append(name)
    return matches


def first_index(marker_names: List[str], candidates: List[str]) -> int | None:
    """Return the first marker index from a candidate list."""
    for candidate in candidates:
        if candidate in marker_names:
            return marker_names.index(candidate)
    return None


def compute_geometry_hints(data: TRCData) -> Dict[str, object]:
    """Compute simple geometry hints for axis interpretation."""
    marker_names = data.marker_names
    matches = find_keyword_matches(marker_names)

    pelvis_idx = first_index(marker_names, matches["pelvis"])
    head_idx = first_index(marker_names, matches["head"])
    left_shoulder_idx = first_index(marker_names, matches["left_shoulder"])
    right_shoulder_idx = first_index(marker_names, matches["right_shoulder"])

    result: Dict[str, object] = {
        "keyword_matches": matches,
        "head_minus_pelvis_vector_median": None,
        "shoulder_width_median": None,
        "up_axis_from_head_pelvis": None,
    }

    if pelvis_idx is not None and head_idx is not None:
        pelvis = data.positions[:, pelvis_idx, :]
        head = data.positions[:, head_idx, :]
        valid = np.isfinite(pelvis).all(axis=1) & np.isfinite(head).all(axis=1)
        if np.any(valid):
            median_vec = np.nanmedian(head[valid] - pelvis[valid], axis=0)
            dominant_axis = int(np.argmax(np.abs(median_vec)))
            result["head_minus_pelvis_vector_median"] = median_vec.tolist()
            result["up_axis_from_head_pelvis"] = {
                "axis": AXIS_NAMES[dominant_axis],
                "sign": float(np.sign(median_vec[dominant_axis])),
            }

    if left_shoulder_idx is not None and right_shoulder_idx is not None:
        left = data.positions[:, left_shoulder_idx, :]
        right = data.positions[:, right_shoulder_idx, :]
        valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
        if np.any(valid):
            width = np.linalg.norm(left[valid] - right[valid], axis=1)
            result["shoulder_width_median"] = float(np.nanmedian(width))

    return result


def build_summary(data: TRCData) -> Dict[str, object]:
    """Build a machine-readable TRC inspection summary."""
    axis_stats = {
        axis_name: summarize_axis(data.positions[:, :, axis_idx])
        for axis_idx, axis_name in enumerate(AXIS_NAMES)
    }
    axis_by_range = max(AXIS_NAMES, key=lambda axis: axis_stats[axis]["range"])
    axis_by_std = max(AXIS_NAMES, key=lambda axis: axis_stats[axis]["std"])

    timestamps = data.timestamps
    dt = np.diff(timestamps) if len(timestamps) > 1 else np.array([], dtype=np.float64)
    geometry = compute_geometry_hints(data)

    return {
        "input_trc_path": str(DEFAULT_TRC_PATH),
        "fps": data.fps,
        "units": data.units,
        "num_frames_header": int(data.positions.shape[0]),
        "num_markers": len(data.marker_names),
        "marker_names": data.marker_names,
        "time_start_s": float(timestamps[0]) if len(timestamps) else math.nan,
        "time_end_s": float(timestamps[-1]) if len(timestamps) else math.nan,
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "median_dt_s": float(np.median(dt)) if dt.size else math.nan,
        "axis_stats": axis_stats,
        "axis_with_largest_range": axis_by_range,
        "axis_with_largest_std": axis_by_std,
        "geometry_hints": geometry,
        "notes": [
            "Largest range/std are only heuristics and do not prove the world up-axis.",
            "Head-to-pelvis direction is the strongest clue if those markers exist.",
            "Phase 2 should not start until marker semantics look consistent.",
        ],
    }


def build_markdown(summary: Dict[str, object]) -> str:
    """Render a short human-readable markdown report."""
    axis_stats = summary["axis_stats"]
    geometry = summary["geometry_hints"]
    lines = [
        "# EasyErgo TRC Inspection",
        "",
        f"- Input: `{summary['input_trc_path']}`",
        f"- FPS: `{summary['fps']}`",
        f"- Units: `{summary['units']}`",
        f"- Frames: `{summary['num_frames_header']}`",
        f"- Markers: `{summary['num_markers']}`",
        f"- Duration: `{summary['duration_s']:.3f} s`",
        f"- Median dt: `{summary['median_dt_s']:.6f} s`",
        "",
        "## Axis Stats",
        "",
        "| Axis | Min | Max | Mean | Std | Range |",
        "|------|-----|-----|------|-----|-------|",
    ]
    for axis_name in AXIS_NAMES:
        stats = axis_stats[axis_name]
        lines.append(
            f"| {axis_name} | {stats['min']:.3f} | {stats['max']:.3f} | "
            f"{stats['mean']:.3f} | {stats['std']:.3f} | {stats['range']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Heuristics",
            "",
            f"- Largest range axis: `{summary['axis_with_largest_range']}`",
            f"- Largest std axis: `{summary['axis_with_largest_std']}`",
            f"- Keyword matches: `{json.dumps(geometry['keyword_matches'], ensure_ascii=False)}`",
        ]
    )

    if geometry["head_minus_pelvis_vector_median"] is not None:
        lines.append(
            "- Median head-pelvis vector: "
            f"`{geometry['head_minus_pelvis_vector_median']}`"
        )
        lines.append(
            "- Up-axis hint from head-pelvis: "
            f"`{geometry['up_axis_from_head_pelvis']}`"
        )

    if geometry["shoulder_width_median"] is not None:
        lines.append(
            f"- Median shoulder width: `{geometry['shoulder_width_median']:.3f}`"
        )

    lines.extend(
        [
            "",
            "## Marker Names",
            "",
            ", ".join(summary["marker_names"]),
            "",
            "## Notes",
            "",
        ]
    )
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the TRC inspection and write JSON + markdown summaries."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trc_path = resolve_trc_path()
    data = load_trc(trc_path)
    summary = build_summary(data)
    summary["input_trc_path"] = str(trc_path)

    with JSON_OUT.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with MD_OUT.open("w", encoding="utf-8") as handle:
        handle.write(build_markdown(summary))

    print(f"[saved] {JSON_OUT}")
    print(f"[saved] {MD_OUT}")
    print(f"[info] units={summary['units']} fps={summary['fps']} markers={summary['num_markers']}")
    print(
        "[info] axis heuristics: "
        f"range={summary['axis_with_largest_range']} std={summary['axis_with_largest_std']}"
    )


if __name__ == "__main__":
    main()
