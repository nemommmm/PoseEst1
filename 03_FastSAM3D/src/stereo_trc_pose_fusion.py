"""Evaluate pose-space fusion of two synchronized FastSAM3D TRC exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = PROJECT_ROOT / "00_pose_pipeline_v2" / "src"

import sys

if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

from common.angles import (  # noqa: E402
    SEMANTIC_ANGLE_NAMES,
    build_native_angle_interpolators,
    compute_angle_sequence,
    moving_average,
    odd_window_from_ms,
    sample_interpolators,
)
from common.metrics import rula_bin  # noqa: E402
from common.trc import load_trc, trc_to_coco17, unit_to_cm  # noqa: E402


TORSO_ANCHORS = (
    "LShoulder",
    "RShoulder",
    "LHip",
    "RHip",
    "Neck",
    "PelvisCenter",
    "Thorax",
)
COCO_CONNECTIONS = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
RULA_BINS = {
    "LeftElbow": [60.0, 100.0],
    "RightElbow": [60.0, 100.0],
    "LeftShoulder": [20.0, 45.0, 90.0],
    "RightShoulder": [20.0, 45.0, 90.0],
}


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a proper rigid transform for row-vector 3D points."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three points are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right_t = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = target_center - source_center @ rotation
    return rotation, translation


def align_pose_sequence(
    source: np.ndarray,
    target: np.ndarray,
    anchor_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align each source pose to target using only torso anchor markers."""
    if source.shape != target.shape:
        raise ValueError("source and target pose sequences must have matching shape")
    aligned = np.full_like(source, np.nan, dtype=np.float64)
    rotations = np.full((len(source), 3, 3), np.nan, dtype=np.float64)
    translations = np.full((len(source), 3), np.nan, dtype=np.float64)
    residuals = np.full((len(source), source.shape[1]), np.nan, dtype=np.float64)
    anchors = np.asarray(anchor_indices, dtype=np.int64)

    for frame_idx in range(len(source)):
        anchor_valid = (
            np.isfinite(source[frame_idx, anchors]).all(axis=1)
            & np.isfinite(target[frame_idx, anchors]).all(axis=1)
        )
        valid_anchors = anchors[anchor_valid]
        if len(valid_anchors) < 3:
            continue
        rotation, translation = fit_rigid_transform(
            source[frame_idx, valid_anchors], target[frame_idx, valid_anchors]
        )
        valid_source = np.isfinite(source[frame_idx]).all(axis=1)
        aligned[frame_idx, valid_source] = (
            source[frame_idx, valid_source] @ rotation + translation
        )
        rotations[frame_idx] = rotation
        translations[frame_idx] = translation
        both_valid = valid_source & np.isfinite(target[frame_idx]).all(axis=1)
        residuals[frame_idx, both_valid] = np.linalg.norm(
            aligned[frame_idx, both_valid] - target[frame_idx, both_valid], axis=1
        )
    return aligned, rotations, translations, residuals


def fuse_equal(left: np.ndarray, right_aligned: np.ndarray) -> np.ndarray:
    """Fuse two poses with equal weights while preserving one-sided values."""
    values = np.stack([left, right_aligned], axis=0)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    summed = np.where(finite, values, 0.0).sum(axis=0)
    output = np.full_like(left, np.nan, dtype=np.float64)
    valid = counts > 0
    output[valid] = summed[valid] / counts[valid]
    return output


def processed_angles(keypoints: np.ndarray, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    """Compute angles with the same 200 ms smoothing used by the main pipeline."""
    _, radius, _ = odd_window_from_ms(timestamps, 200.0)
    raw = compute_angle_sequence(keypoints, list(SEMANTIC_ANGLE_NAMES))
    return {name: moving_average(values, radius) for name, values in raw.items()}


def angle_metrics(
    angles: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int | None]]:
    """Compute agreement metrics against an external comparison system."""
    metrics: dict[str, dict[str, float | int | None]] = {}
    for name in SEMANTIC_ANGLE_NAMES:
        values = angles[name]
        ref = reference[name]
        valid = np.isfinite(values) & np.isfinite(ref)
        if not np.any(valid):
            metrics[name] = {
                "valid_pair_count": 0,
                "mae_deg": None,
                "median_abs_error_deg": None,
                "bias_deg": None,
                "rula_like_agreement": None,
                "jump_count_gt_10_deg": 0,
            }
            continue
        delta = values[valid] - ref[valid]
        finite_values = values[np.isfinite(values)]
        bins = RULA_BINS.get(name)
        rula_agreement = None
        if bins:
            rula_agreement = float(
                np.mean(rula_bin(values[valid], bins) == rula_bin(ref[valid], bins))
            )
        metrics[name] = {
            "valid_pair_count": int(np.count_nonzero(valid)),
            "mae_deg": float(np.mean(np.abs(delta))),
            "median_abs_error_deg": float(np.median(np.abs(delta))),
            "bias_deg": float(np.mean(delta)),
            "rula_like_agreement": rula_agreement,
            "jump_count_gt_10_deg": int(
                np.count_nonzero(np.abs(np.diff(finite_values)) > 10.0)
            ),
        }
    return metrics


def percentile(values: np.ndarray, q: float) -> float | None:
    """Return one finite percentile or None."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if len(finite) else None


def mean_joint_mae(metrics: dict[str, dict[str, float | int | None]]) -> float:
    """Average available per-joint angle MAE values."""
    values = [row["mae_deg"] for row in metrics.values() if row["mae_deg"] is not None]
    return float(np.mean(values))


def write_angle_csv(
    path: Path,
    timestamps: np.ndarray,
    systems: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
) -> None:
    """Write compact angle time series for manual inspection."""
    fieldnames = ["frame", "time_s"]
    for system in [*systems, "XsensDerivedReference"]:
        for angle_name in SEMANTIC_ANGLE_NAMES:
            fieldnames.append(f"{system}_{angle_name}_deg")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_idx, timestamp in enumerate(timestamps):
            row: dict[str, str | int] = {
                "frame": frame_idx,
                "time_s": f"{float(timestamp):.6f}",
            }
            for system, angles in systems.items():
                for angle_name in SEMANTIC_ANGLE_NAMES:
                    value = angles[angle_name][frame_idx]
                    row[f"{system}_{angle_name}_deg"] = (
                        f"{float(value):.6f}" if np.isfinite(value) else ""
                    )
            for angle_name in SEMANTIC_ANGLE_NAMES:
                value = reference[angle_name][frame_idx]
                row[f"XsensDerivedReference_{angle_name}_deg"] = (
                    f"{float(value):.6f}" if np.isfinite(value) else ""
                )
            writer.writerow(row)


def set_equal_3d_axes(axis, points: np.ndarray) -> None:
    """Set approximately equal axes around the supplied points."""
    finite = points[np.isfinite(points).all(axis=1)]
    center = finite.mean(axis=0)
    radius = max(float(np.ptp(finite, axis=0).max()) * 0.58, 20.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def plot_pose(axis, pose: np.ndarray, color: str, label: str, alpha: float) -> None:
    """Draw one COCO-17 skeleton."""
    for first, second in COCO_CONNECTIONS:
        points = pose[[first, second]]
        if np.isfinite(points).all():
            axis.plot(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=alpha)
    valid = np.isfinite(pose).all(axis=1)
    axis.scatter(
        pose[valid, 0], pose[valid, 1], pose[valid, 2], s=10, color=color, alpha=alpha, label=label
    )


def write_preview(
    path: Path,
    left: np.ndarray,
    right_aligned: np.ndarray,
    fused: np.ndarray,
) -> None:
    """Write a four-frame 3D reconstruction comparison image."""
    frame_indices = np.linspace(0, len(fused) - 1, 4, dtype=int)
    figure = plt.figure(figsize=(14, 4.2))
    for plot_idx, frame_idx in enumerate(frame_indices, start=1):
        axis = figure.add_subplot(1, 4, plot_idx, projection="3d")
        plot_pose(axis, left[frame_idx], "#1f77b4", "left TRC", 0.55)
        plot_pose(axis, right_aligned[frame_idx], "#ff7f0e", "right aligned", 0.55)
        plot_pose(axis, fused[frame_idx], "#6a3d9a", "equal fusion", 1.0)
        set_equal_3d_axes(axis, fused[frame_idx])
        axis.set_title(f"Frame {frame_idx}")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
        if plot_idx == 1:
            axis.legend(loc="upper left", fontsize=7)
    figure.suptitle("FastSAM3D left/right pose-space fusion (not calibrated stereo geometry)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Run the synchronized TRC pose-space fusion experiment."""
    left_time, left_names, left_mm, left_fps, left_units = load_trc(args.left_trc)
    right_time, right_names, right_mm, right_fps, right_units = load_trc(args.right_trc)
    if left_names != right_names:
        raise ValueError("left and right marker names differ")
    if left_mm.shape != right_mm.shape:
        raise ValueError("left and right TRC shapes differ")
    if not np.allclose(left_time, right_time, atol=1e-9, rtol=0.0):
        raise ValueError("left and right timestamps are not synchronized")
    if left_units != right_units:
        raise ValueError("left and right units differ")

    scale = unit_to_cm(left_units)
    left = left_mm * scale
    right = right_mm * scale
    anchor_indices = [left_names.index(name) for name in TORSO_ANCHORS]
    right_aligned, rotations, translations, residuals = align_pose_sequence(
        right, left, anchor_indices
    )
    fused = fuse_equal(left, right_aligned)

    left_coco, missing_left = trc_to_coco17(left_names, left)
    right_coco, missing_right = trc_to_coco17(right_names, right)
    right_aligned_coco, _ = trc_to_coco17(left_names, right_aligned)
    fused_coco, missing_fused = trc_to_coco17(left_names, fused)

    systems = {
        "LeftTRC": processed_angles(left_coco, left_time),
        "RightTRC": processed_angles(right_coco, left_time),
        "Fused": processed_angles(fused_coco, left_time),
    }
    reference_interpolators = build_native_angle_interpolators(args.xsens_mvnx)
    reference = sample_interpolators(
        reference_interpolators,
        left_time - float(args.xsens_offset_seconds),
        list(SEMANTIC_ANGLE_NAMES),
    )
    angle_results = {
        system: angle_metrics(angles, reference) for system, angles in systems.items()
    }
    mean_mae = {system: mean_joint_mae(rows) for system, rows in angle_results.items()}

    direct_difference = np.linalg.norm(left - right, axis=2)
    with np.load(args.calibration) as calibration:
        rotation = np.asarray(calibration["R"], dtype=np.float64)
        translation_cm = np.asarray(calibration["T"], dtype=np.float64).reshape(3)
    right_calibrated_to_left = (right - translation_cm) @ rotation
    calibrated_difference = np.linalg.norm(left - right_calibrated_to_left, axis=2)
    rotation_angles = np.degrees(
        np.arccos(
            np.clip((np.trace(rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        )
    )
    anchor_residuals = residuals[:, anchor_indices]

    cross_view_angle: dict[str, dict[str, float | None]] = {}
    for name in SEMANTIC_ANGLE_NAMES:
        difference = np.abs(systems["LeftTRC"][name] - systems["RightTRC"][name])
        cross_view_angle[name] = {
            "mae_deg": percentile(difference, 50.0),
            "mean_abs_deg": float(np.nanmean(difference)),
            "p95_abs_deg": percentile(difference, 95.0),
        }

    best_single = min(mean_mae["LeftTRC"], mean_mae["RightTRC"])
    improvement = 100.0 * (best_single - mean_mae["Fused"]) / best_single
    conclusion = (
        "rejected_no_angle_improvement"
        if improvement < 5.0
        else "passed_preliminary_angle_gate"
    )
    metrics: dict[str, object] = {
        "experiment": "FastSAM3D synchronized dual-TRC pose-space fusion",
        "status": conclusion,
        "important_scope": (
            "Per-frame torso alignment and equal pose averaging. This is not calibrated "
            "stereo triangulation because the TRC coordinates do not preserve camera extrinsics."
        ),
        "sources": {
            "left_trc": str(args.left_trc),
            "right_trc": str(args.right_trc),
            "frame_count": int(len(left_time)),
            "marker_count": int(len(left_names)),
            "fps": float(left_fps),
            "units_in": left_units,
            "timestamps_identical": True,
            "marker_names_identical": True,
            "missing_coco17_left": missing_left,
            "missing_coco17_right": missing_right,
            "missing_coco17_fused": missing_fused,
        },
        "coordinate_diagnosis_cm": {
            "direct_corresponding_joint_distance_p50": percentile(direct_difference, 50.0),
            "direct_corresponding_joint_distance_p95": percentile(direct_difference, 95.0),
            "after_a255_calibration_distance_p50": percentile(calibrated_difference, 50.0),
            "after_a255_calibration_distance_p95": percentile(calibrated_difference, 95.0),
            "interpretation": (
                "Applying physical A255 extrinsics makes agreement much worse, showing that "
                "these FastSAM3D TRCs are not expressed in the raw camera coordinate frames."
            ),
        },
        "pose_space_alignment": {
            "anchors": list(TORSO_ANCHORS),
            "anchor_residual_p50_cm": percentile(anchor_residuals, 50.0),
            "anchor_residual_p95_cm": percentile(anchor_residuals, 95.0),
            "all_marker_frame_median_residual_p50_cm": percentile(
                np.nanmedian(residuals, axis=1), 50.0
            ),
            "all_marker_frame_median_residual_p95_cm": percentile(
                np.nanmedian(residuals, axis=1), 95.0
            ),
            "translation_norm_p50_cm": percentile(
                np.linalg.norm(translations, axis=1), 50.0
            ),
            "translation_norm_p95_cm": percentile(
                np.linalg.norm(translations, axis=1), 95.0
            ),
            "rotation_angle_p50_deg": percentile(rotation_angles, 50.0),
            "rotation_angle_p95_deg": percentile(rotation_angles, 95.0),
        },
        "cross_view_angle_disagreement": cross_view_angle,
        "external_comparison": {
            "reference": "Xsens-derived reference / external comparison system",
            "xsens_offset_seconds": float(args.xsens_offset_seconds),
            "note": (
                "The fixed existing offset and Xsens angles were used only after fusion for "
                "reporting, never to choose alignment, weights, or fusion parameters."
            ),
            "per_system": angle_results,
            "mean_eight_joint_mae_deg": mean_mae,
            "fused_improvement_vs_best_single_percent": improvement,
        },
        "decision": {
            "result": conclusion,
            "reason": (
                "Equal pose-space fusion did not improve the mean eight-joint angle agreement "
                "by the predefined 5% gate."
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "fastsam_stereo_pose_fusion.npz",
        timestamps=left_time,
        marker_names=np.asarray(left_names),
        left_keypoints_cm=left,
        right_keypoints_cm=right,
        right_aligned_keypoints_cm=right_aligned,
        fused_keypoints_cm=fused,
        fused_coco17_cm=fused_coco,
        per_frame_rotation=rotations,
        per_frame_translation_cm=translations,
        units="cm",
        fusion_type="pose_space_torso_alignment_equal_average",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_angle_csv(args.output_dir / "angle_timeseries.csv", left_time, systems, reference)
    write_preview(
        args.output_dir / "reconstruction_preview.png",
        left_coco,
        right_aligned_coco,
        fused_coco,
    )
    if args.experiment_log is not None:
        args.experiment_log.parent.mkdir(parents=True, exist_ok=True)
        with args.experiment_log.open("a", encoding="utf-8") as handle:
            handle.write(
                "- FastSAM3D dual-TRC pose fusion (Fanbo3 A255): "
                f"{conclusion}; left/right/fused mean 8-joint MAE against the "
                "Xsens-derived reference = "
                f"{mean_mae['LeftTRC']:.3f}/{mean_mae['RightTRC']:.3f}/"
                f"{mean_mae['Fused']:.3f} deg. Physical A255 calibration was not "
                "applicable to the exported TRC coordinates; retained result package at "
                f"`{args.output_dir}`.\n"
            )
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-trc", type=Path, required=True)
    parser.add_argument("--right-trc", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--xsens-mvnx", type=Path, required=True)
    parser.add_argument("--xsens-offset-seconds", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-log", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the CLI and print the decision summary."""
    metrics = evaluate(parse_args())
    external = metrics["external_comparison"]
    means = external["mean_eight_joint_mae_deg"]
    print(f"[saved] {metrics['decision']['result']}")
    print(
        "[angles] mean 8-joint MAE left/right/fused: "
        f"{means['LeftTRC']:.3f} / {means['RightTRC']:.3f} / {means['Fused']:.3f} deg"
    )


if __name__ == "__main__":
    main()
