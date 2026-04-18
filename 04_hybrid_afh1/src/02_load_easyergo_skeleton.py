"""Load EasyErgo TRC and normalize it into a COCO-17 style NPZ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from easyergo_marker_mapping import COCO17_JOINT_NAMES, MARKER_NAME_TO_COCO17


SCRIPT_DIR = Path(__file__).resolve().parent
AFH1_DIR = SCRIPT_DIR.parent
INPUT_DIR = AFH1_DIR / "data" / "easyergo_uploaded"
RESULTS_DIR = AFH1_DIR / "results"
NPZ_OUT = RESULTS_DIR / "easyergo_normalized.npz"
SUMMARY_OUT = RESULTS_DIR / "easyergo_normalized_summary.json"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.md"


def resolve_trc_path() -> Path:
    """Resolve the uploaded EasyErgo TRC file."""
    preferred = INPUT_DIR / "markers_easyergo.trc"
    if preferred.is_file():
        return preferred
    candidates = sorted(INPUT_DIR.glob("*.trc"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No TRC file found in {INPUT_DIR}")
    raise FileNotFoundError(
        f"Multiple TRC files found in {INPUT_DIR}; please keep only one."
    )


def load_trc(trc_path: Path) -> tuple[np.ndarray, List[str], np.ndarray, float, str]:
    """Parse a TRC file and return timestamps, names, positions, fps, units."""
    with trc_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    header_values = lines[2].strip().split("\t")
    fps = float(header_values[0])
    num_markers = int(header_values[3])
    units = header_values[4]

    raw_names = lines[3].strip().split("\t")[2:]
    marker_names = [name for name in raw_names if name.strip()]
    if len(marker_names) != num_markers:
        raise ValueError(
            f"Marker count mismatch: header={num_markers}, parsed={len(marker_names)}"
        )

    data_lines = [line.strip() for line in lines[6:] if line.strip()]
    timestamps = []
    frames = []
    for line in data_lines:
        values = line.split("\t")
        timestamps.append(float(values[1]))
        coords = [float(value) if value else np.nan for value in values[2:]]
        frames.append(coords)

    positions = np.asarray(frames, dtype=np.float64).reshape(-1, num_markers, 3)
    return np.asarray(timestamps, dtype=np.float64), marker_names, positions, fps, units


def unit_scale_to_cm(units: str) -> float:
    """Return the multiplicative scale that converts a unit into cm."""
    u = units.strip().lower()
    if u == "cm":
        return 1.0
    if u == "mm":
        return 0.1
    if u in {"m", "meter", "meters", "metre", "metres"}:
        return 100.0
    raise ValueError(f"Unsupported TRC unit for EasyErgo normalization: {units}")


def build_coco17_positions(
    marker_names: List[str],
    positions: np.ndarray,
    name_to_coco17: Dict[str, int],
) -> tuple[np.ndarray, List[str], List[str]]:
    """Map original TRC markers into a dense COCO-17 style array."""
    num_frames = positions.shape[0]
    coco17 = np.full((num_frames, len(COCO17_JOINT_NAMES), 3), np.nan, dtype=np.float64)
    mapped_markers: List[str] = []
    missing_joints: List[str] = []

    name_to_index = {name: idx for idx, name in enumerate(marker_names)}
    for joint_idx, joint_name in enumerate(COCO17_JOINT_NAMES):
        source_name = None
        for marker_name, mapped_idx in name_to_coco17.items():
            if mapped_idx == joint_idx:
                source_name = marker_name
                break
        if source_name is None or source_name not in name_to_index:
            missing_joints.append(joint_name)
            continue
        coco17[:, joint_idx, :] = positions[:, name_to_index[source_name], :]
        mapped_markers.append(source_name)

    return coco17, mapped_markers, missing_joints


def append_experiment_log(message: str) -> None:
    """Append one line to the AFH1 experiment log."""
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- {message}\n")


def main() -> None:
    """Load the uploaded EasyErgo TRC and normalize it into COCO-17 in cm."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trc_path = resolve_trc_path()
    timestamps, marker_names, positions, fps, units = load_trc(trc_path)

    scale = unit_scale_to_cm(units)
    positions_cm = positions * scale
    keypoints_3d, mapped_markers, missing_joints = build_coco17_positions(
        marker_names,
        positions_cm,
        MARKER_NAME_TO_COCO17,
    )
    valid_joint_mask = np.isfinite(keypoints_3d).all(axis=2)

    np.savez(
        NPZ_OUT,
        timestamps=timestamps,
        keypoints_3d=keypoints_3d,
        valid_joint_mask=valid_joint_mask,
        marker_names_original=np.asarray(marker_names),
        marker_names_coco17=np.asarray(COCO17_JOINT_NAMES),
        mapped_markers=np.asarray(mapped_markers),
        source_trc_path=str(trc_path),
        fps=float(fps),
        input_units=units,
        output_units="cm",
    )

    coverage_by_joint = {
        joint_name: float(np.mean(valid_joint_mask[:, idx]))
        for idx, joint_name in enumerate(COCO17_JOINT_NAMES)
    }

    summary = {
        "source_trc_path": str(trc_path),
        "output_npz_path": str(NPZ_OUT),
        "fps": float(fps),
        "num_frames": int(len(timestamps)),
        "num_markers_original": int(len(marker_names)),
        "input_units": units,
        "output_units": "cm",
        "scale_to_cm": float(scale),
        "mapped_markers": mapped_markers,
        "missing_joints": missing_joints,
        "coverage_by_joint": coverage_by_joint,
        "median_bone_lengths_cm": {
            "shoulder_width": float(
                np.nanmedian(
                    np.linalg.norm(keypoints_3d[:, 5, :] - keypoints_3d[:, 6, :], axis=1)
                )
            ),
            "hip_width": float(
                np.nanmedian(
                    np.linalg.norm(keypoints_3d[:, 11, :] - keypoints_3d[:, 12, :], axis=1)
                )
            ),
        },
    }

    with SUMMARY_OUT.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_experiment_log(
        "Normalized EasyErgo TRC into COCO-17 skeleton "
        f"({len(timestamps)} frames, input units {units} -> output cm)."
    )

    print(f"[saved] {NPZ_OUT}")
    print(f"[saved] {SUMMARY_OUT}")
    print(f"[info] mapped joints: {len(mapped_markers)}/{len(COCO17_JOINT_NAMES)}")
    if missing_joints:
        print(f"[warn] missing joints: {missing_joints}")


if __name__ == "__main__":
    main()
