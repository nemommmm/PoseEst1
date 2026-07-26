"""Shared evaluation helpers for external 3D pose candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.angles import (
    SEMANTIC_ANGLE_NAMES,
    build_fair_angle_interpolators,
    build_native_angle_interpolators,
    compute_angle_sequence,
    fill_short_gaps,
    moving_average,
    odd_window_from_ms,
    sample_interpolators,
)
from common.config import resolve_path, section
from common.metrics import jsonable, rula_bin
from common.trc import (
    interpolate_keypoints,
    load_trc,
    trc_to_coco17,
    unit_to_cm,
)


def finite_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    """Return robust summary statistics for finite values."""
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def process_camera_angles(
    keypoints_3d: np.ndarray,
    timestamps: np.ndarray,
    config: Mapping[str, Any],
    angle_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Apply the project's fixed gap-fill and smoothing policy."""
    evaluation = section(dict(config), "evaluation")
    raw = compute_angle_sequence(
        np.asarray(keypoints_3d, dtype=np.float64),
        [name for name in angle_names if name in SEMANTIC_ANGLE_NAMES],
    )
    _, radius, _ = odd_window_from_ms(
        np.asarray(timestamps, dtype=np.float64),
        float(evaluation.get("camera_smooth_window_ms", 200.0)),
    )
    maximum_gap = int(evaluation.get("max_gap_frames", 5))
    processed: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        filled, _ = fill_short_gaps(
            values,
            np.asarray(timestamps, dtype=np.float64),
            maximum_gap,
        )
        smoothed = moving_average(filled, radius)
        smoothed[~np.isfinite(filled)] = np.nan
        processed[name] = smoothed
    return processed


def load_reference_angles(
    config: Mapping[str, Any],
    timestamps: np.ndarray,
    angle_names: Sequence[str],
    reference_kind: str,
    offset_seconds: float,
) -> dict[str, np.ndarray]:
    """Load FastSAM3D or Xsens-derived angles on a fixed video timeline."""
    references = section(dict(config), "references")
    target_time = np.asarray(timestamps, dtype=np.float64)
    if reference_kind == "fastsam":
        trc_path = resolve_path(references.get("fastsam_trc"), must_exist=True)
        assert trc_path is not None
        source_time, marker_names, positions, _, units = load_trc(trc_path)
        keypoints, _ = trc_to_coco17(
            marker_names,
            positions * unit_to_cm(units),
        )
        aligned = interpolate_keypoints(
            source_time,
            keypoints,
            target_time,
            source_time_offset_s=float(offset_seconds),
        )
        return process_camera_angles(
            aligned,
            target_time,
            config,
            angle_names,
        )
    if reference_kind != "xsens":
        raise ValueError(f"Unsupported reference kind: {reference_kind}")
    fair_path = resolve_path(
        references.get("xsens_fair_angles"),
        must_exist=False,
    )
    interpolators = build_fair_angle_interpolators(fair_path)
    if not interpolators:
        mvnx_path = resolve_path(
            references.get("xsens_mvnx"),
            must_exist=True,
        )
        assert mvnx_path is not None
        interpolators = build_native_angle_interpolators(mvnx_path)
    return sample_interpolators(
        interpolators,
        target_time - float(offset_seconds),
        list(angle_names),
    )


def angle_agreement(
    candidate: np.ndarray,
    baseline: np.ndarray,
    reference: np.ndarray,
    bins: Sequence[float] | None,
) -> dict[str, Any]:
    """Compare candidate and baseline on one identical finite frame set."""
    candidate_values = np.asarray(candidate, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    common = (
        np.isfinite(candidate_values)
        & np.isfinite(baseline_values)
        & np.isfinite(reference_values)
    )

    def summarize(values: np.ndarray) -> dict[str, Any]:
        difference = values[common] - reference_values[common]
        absolute = np.abs(difference)
        result: dict[str, Any] = {
            "valid_pair_count": int(np.count_nonzero(common)),
            "valid_ratio": float(np.mean(common)) if common.size else 0.0,
            "absolute_error_deg": finite_distribution(absolute),
            "bias_deg": (
                float(np.mean(difference)) if difference.size else None
            ),
            "rmse_deg": (
                float(np.sqrt(np.mean(difference * difference)))
                if difference.size
                else None
            ),
        }
        if bins and difference.size:
            result["rula_like_agreement"] = float(
                np.mean(
                    rula_bin(values[common], list(bins))
                    == rula_bin(reference_values[common], list(bins))
                )
            )
        else:
            result["rula_like_agreement"] = None
        return result

    candidate_summary = summarize(candidate_values)
    baseline_summary = summarize(baseline_values)
    candidate_median = candidate_summary["absolute_error_deg"]["median"]
    baseline_median = baseline_summary["absolute_error_deg"]["median"]
    improvement = None
    if (
        candidate_median is not None
        and baseline_median is not None
        and float(baseline_median) > 0
    ):
        improvement = (
            float(baseline_median) - float(candidate_median)
        ) / float(baseline_median)
    return {
        "common_finite_mask": common,
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "median_improvement_ratio": improvement,
    }


def write_angle_timeseries(
    path: Path,
    timestamps: np.ndarray,
    systems: Mapping[str, Mapping[str, np.ndarray]],
    angle_names: Sequence[str],
) -> None:
    """Write one compact, spreadsheet-ready angle time-series file."""
    fields = ["Frame", "Time_s"] + [
        f"{system}_{angle}_deg"
        for system in systems
        for angle in angle_names
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_index, timestamp in enumerate(timestamps):
            row: dict[str, Any] = {
                "Frame": frame_index,
                "Time_s": f"{float(timestamp):.6f}",
            }
            for system, angles in systems.items():
                for angle in angle_names:
                    value = angles[angle][frame_index]
                    row[f"{system}_{angle}_deg"] = (
                        f"{float(value):.6f}"
                        if np.isfinite(value)
                        else ""
                    )
            writer.writerow(row)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with project-standard NumPy conversion."""
    path.write_text(
        json.dumps(jsonable(dict(payload)), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

