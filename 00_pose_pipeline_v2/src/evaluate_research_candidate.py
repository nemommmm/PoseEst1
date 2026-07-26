#!/usr/bin/env python
"""Evaluate one canonical 3D pose candidate against the fixed reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.candidate_evaluation import (
    angle_agreement,
    load_reference_angles,
    process_camera_angles,
    save_json,
    write_angle_timeseries,
)
from common.config import load_config, section
from common.research_candidate import load_candidate_npz

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: Path) -> Path:
    """Resolve a CLI path against the project root."""
    return (
        path.expanduser().resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _load_baseline(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamps and 3D keypoints from a legacy SKT result."""
    with np.load(path, allow_pickle=True) as payload:
        key = "keypoints_3d" if "keypoints_3d" in payload else "keypoints"
        return (
            np.asarray(payload["timestamps"], dtype=np.float64),
            np.asarray(payload[key], dtype=np.float64),
        )


def _plot_summary(
    output: Path,
    timestamps: np.ndarray,
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    angle_names: Sequence[str],
) -> None:
    """Create intuitive time-series, scatter, and error box plots."""
    count = len(angle_names)
    figure, axes = plt.subplots(
        count,
        3,
        figsize=(14, max(3.2 * count, 4.5)),
        squeeze=False,
    )
    for row, name in enumerate(angle_names):
        axes[row, 0].plot(timestamps, reference[name], label="Reference")
        axes[row, 0].plot(
            timestamps,
            baseline[name],
            label="YOLOv8m+SKT",
            alpha=0.8,
        )
        axes[row, 0].plot(
            timestamps,
            candidate[name],
            label="Candidate",
            alpha=0.8,
        )
        axes[row, 0].set_title(f"{name}: angle over time")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Angle (deg)")

        valid = np.isfinite(candidate[name]) & np.isfinite(reference[name])
        axes[row, 1].scatter(
            reference[name][valid],
            candidate[name][valid],
            s=9,
            alpha=0.55,
        )
        if np.any(valid):
            lower = float(
                min(reference[name][valid].min(), candidate[name][valid].min())
            )
            upper = float(
                max(reference[name][valid].max(), candidate[name][valid].max())
            )
            axes[row, 1].plot([lower, upper], [lower, upper], "k--", lw=1)
        axes[row, 1].set_title(f"{name}: candidate vs reference")
        axes[row, 1].set_xlabel("Reference (deg)")
        axes[row, 1].set_ylabel("Candidate (deg)")

        candidate_error = np.abs(candidate[name] - reference[name])
        baseline_error = np.abs(baseline[name] - reference[name])
        axes[row, 2].boxplot(
            [
                baseline_error[np.isfinite(baseline_error)],
                candidate_error[np.isfinite(candidate_error)],
            ],
            tick_labels=["YOLO+SKT", "Candidate"],
            showfliers=True,
        )
        axes[row, 2].set_title(f"{name}: absolute error")
        axes[row, 2].set_ylabel("Absolute difference (deg)")
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def evaluate(
    candidate_path: Path,
    baseline_path: Path,
    config_path: Path,
    output_dir: Path,
    reference_kind: str,
    reference_label: str,
    reference_offset_seconds: float,
    angle_names: Sequence[str],
) -> dict[str, Any]:
    """Run fixed-reference evaluation and persist reproducible evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    candidate_payload = load_candidate_npz(candidate_path)
    candidate_keypoints = np.asarray(
        candidate_payload["keypoints_3d"],
        dtype=np.float64,
    )
    timestamps = np.asarray(
        candidate_payload["timestamps"],
        dtype=np.float64,
    )
    baseline_timestamps, baseline_keypoints = _load_baseline(baseline_path)
    frame_count = min(
        len(timestamps),
        len(candidate_keypoints),
        len(baseline_timestamps),
        len(baseline_keypoints),
    )
    timestamps = timestamps[:frame_count]
    if not np.allclose(
        timestamps,
        baseline_timestamps[:frame_count],
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("Candidate and baseline timestamps do not match")
    candidate_angles = process_camera_angles(
        candidate_keypoints[:frame_count],
        timestamps,
        config,
        angle_names,
    )
    baseline_angles = process_camera_angles(
        baseline_keypoints[:frame_count],
        timestamps,
        config,
        angle_names,
    )
    reference_angles = load_reference_angles(
        config,
        timestamps,
        angle_names,
        reference_kind,
        reference_offset_seconds,
    )
    bins_by_angle = section(config, "evaluation").get("rula_bins", {})
    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    for name in angle_names:
        result = angle_agreement(
            candidate_angles[name],
            baseline_angles[name],
            reference_angles[name],
            bins_by_angle.get(name),
        )
        masks[name] = result.pop("common_finite_mask")
        rows.append({"angle": name, **result})
    candidate_medians = [
        row["candidate"]["absolute_error_deg"]["median"]
        for row in rows
        if row["candidate"]["absolute_error_deg"]["median"] is not None
    ]
    baseline_medians = [
        row["baseline"]["absolute_error_deg"]["median"]
        for row in rows
        if row["baseline"]["absolute_error_deg"]["median"] is not None
    ]
    aggregate_improvement = None
    if candidate_medians and baseline_medians:
        candidate_median = float(np.mean(candidate_medians))
        baseline_median = float(np.mean(baseline_medians))
        if baseline_median > 0:
            aggregate_improvement = (
                baseline_median - candidate_median
            ) / baseline_median
    metrics = {
        "schema_version": "candidate_fixed_reference_eval_v1",
        "candidate": candidate_payload["candidate_name"],
        "dataset": section(config, "dataset").get("name"),
        "reference": {
            "kind": reference_kind,
            "label": reference_label,
            "offset_seconds": float(reference_offset_seconds),
            "absolute_ground_truth": False,
        },
        "frame_count": frame_count,
        "angle_names": list(angle_names),
        "rows": rows,
        "aggregate_median_improvement_ratio": aggregate_improvement,
        "candidate_metadata": candidate_payload["metadata"],
    }
    save_json(output_dir / "metrics.json", metrics)
    write_angle_timeseries(
        output_dir / "angle_timeseries.csv",
        timestamps,
        {
            "Candidate": candidate_angles,
            "YOLOv8m_SKT": baseline_angles,
            "Reference": reference_angles,
        },
        angle_names,
    )
    _plot_summary(
        output_dir / "angle_summary.png",
        timestamps,
        candidate_angles,
        baseline_angles,
        reference_angles,
        angle_names,
    )
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-kind",
        choices=["fastsam", "xsens"],
        required=True,
    )
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--reference-offset-seconds", type=float, required=True)
    parser.add_argument("--angle-names", nargs="+", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluator."""
    args = parse_args(argv)
    metrics = evaluate(
        _resolve(args.candidate),
        _resolve(args.baseline),
        _resolve(args.config),
        _resolve(args.output_dir),
        args.reference_kind,
        args.reference_label,
        args.reference_offset_seconds,
        args.angle_names,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
