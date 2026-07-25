#!/usr/bin/env python
"""Evaluate paired DeepStream BodyPose3DNet outputs with project stereo geometry.

The NVIDIA reference application processes one camera at a time. This adapter
uses the exported 2.5D image coordinates and confidences, synchronizes the two
camera streams from their hardware metadata, maps the NVIDIA 34-joint layout to
COCO-17, and then reuses the existing calibrated stereo reconstruction and
ergonomic-angle code.

Xsens-derived values are read only after all geometric reconstruction choices
have been fixed. They are an external comparison signal, not ground truth and
not a parameter-selection input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from common.angles import (
    COCO17_NAMES,
    SEMANTIC_ANGLE_NAMES,
    compute_angle_sequence,
    fill_short_gaps,
    moving_average,
    odd_window_from_ms,
)
from common.config import load_config, resolve_path, section
from common.dataset import apply_depth_consistency_filter
from common.metrics import jsonable, rula_bin
from common.triangulation import (
    TemporalWindowConfig,
    TriangulationConfig,
    rectify_points,
    retriangulate_sequence,
    temporal_window_rescue_rectified,
)
from stereo_loader import SyncedFrame, build_synced_timeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nvidia_bodypose3d_stereo_eval_v1"

BODYPOSE34_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "torso",
    "left_knee",
    "right_knee",
    "neck",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky_knuckle",
    "right_pinky_knuckle",
    "left_middle_tip",
    "right_middle_tip",
    "left_index_knuckle",
    "right_index_knuckle",
    "left_thumb_tip",
    "right_thumb_tip",
)

COCO17_FROM_BODYPOSE34 = np.asarray(
    [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 1, 2, 4, 5, 7, 8],
    dtype=np.int64,
)


class BodyPoseEvaluationError(RuntimeError):
    """Raised when an input or reconstruction contract is violated."""


@dataclass(frozen=True)
class PoseRecord:
    """One tracked person's BodyPose3DNet output for one source frame."""

    frame_num: int
    object_id: int
    pose25d: np.ndarray
    pose3d: np.ndarray


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: np.ndarray | Iterable[float]) -> dict[str, Any]:
    """Summarize finite numeric values without hiding tail behavior."""

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    finite = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    result: dict[str, Any] = {"count": int(finite.size)}
    if finite.size == 0:
        result.update(
            {
                "mean": None,
                "median": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "max": None,
            }
        )
        return result
    result.update(
        {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "p75": float(np.percentile(finite, 75)),
            "p90": float(np.percentile(finite, 90)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(np.max(finite)),
        }
    )
    return result


def _parse_pose_vector(
    raw: object,
    *,
    name: str,
    frame_num: int,
    object_id: int,
) -> np.ndarray:
    """Validate and reshape one 34-by-4 pose vector."""

    array = np.asarray(raw, dtype=np.float64)
    expected = len(BODYPOSE34_NAMES) * 4
    if array.size != expected:
        raise BodyPoseEvaluationError(
            f"{name} has {array.size} values for frame {frame_num}, "
            f"object {object_id}; expected {expected}"
        )
    return array.reshape(len(BODYPOSE34_NAMES), 4)


def load_deepstream_records(path: Path) -> list[PoseRecord]:
    """Load the JSON emitted by NVIDIA's DeepStream reference application."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BodyPoseEvaluationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise BodyPoseEvaluationError(f"Expected a JSON list in {path}")

    records: list[PoseRecord] = []
    for packet in payload:
        if not isinstance(packet, dict):
            continue
        batches = packet.get("batches", [])
        if not isinstance(batches, list):
            continue
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            frame_num = int(batch.get("frame_num", -1))
            objects = batch.get("objects", [])
            if not isinstance(objects, list):
                continue
            for raw_object in objects:
                if not isinstance(raw_object, dict):
                    continue
                object_id = int(raw_object.get("object_id", -1))
                records.append(
                    PoseRecord(
                        frame_num=frame_num,
                        object_id=object_id,
                        pose25d=_parse_pose_vector(
                            raw_object.get("pose25d"),
                            name="pose25d",
                            frame_num=frame_num,
                            object_id=object_id,
                        ),
                        pose3d=_parse_pose_vector(
                            raw_object.get("pose3d"),
                            name="pose3d",
                            frame_num=frame_num,
                            object_id=object_id,
                        ),
                    )
                )
    if not records:
        raise BodyPoseEvaluationError(f"No pose objects found in {path}")
    return records


def summarize_tracks(records: Sequence[PoseRecord]) -> list[dict[str, Any]]:
    """Summarize track persistence, confidence, and image extent."""

    grouped: dict[int, list[PoseRecord]] = defaultdict(list)
    for record in records:
        grouped[record.object_id].append(record)

    summaries: list[dict[str, Any]] = []
    for object_id, items in grouped.items():
        spans: list[float] = []
        confidences: list[float] = []
        for item in items:
            xy = item.pose25d[:, :2]
            conf = item.pose25d[:, 3]
            valid = np.isfinite(xy).all(axis=1) & np.isfinite(conf)
            if np.any(valid):
                width = float(np.ptp(xy[valid, 0]))
                height = float(np.ptp(xy[valid, 1]))
                spans.append(math.hypot(width, height))
                confidences.extend(conf[valid].tolist())
        summaries.append(
            {
                "object_id": int(object_id),
                "frame_count": int(len({item.frame_num for item in items})),
                "record_count": int(len(items)),
                "median_image_span_px": (
                    float(np.median(spans)) if spans else None
                ),
                "mean_joint_confidence": (
                    float(np.mean(confidences)) if confidences else None
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -int(item["frame_count"]),
            -float(item["median_image_span_px"] or 0.0),
            -float(item["mean_joint_confidence"] or 0.0),
            int(item["object_id"]),
        ),
    )


def select_primary_track(
    records: Sequence[PoseRecord],
    requested_id: int | None = None,
    persistence_ratio: float = 0.75,
) -> tuple[int, list[dict[str, Any]]]:
    """Select the persistent, full-size person or honor an explicit override."""

    summaries = summarize_tracks(records)
    available = {int(item["object_id"]) for item in summaries}
    if requested_id is not None:
        if requested_id not in available:
            raise BodyPoseEvaluationError(
                f"Requested track {requested_id} is absent; available: "
                f"{sorted(available)}"
            )
        return requested_id, summaries
    maximum_frames = max(int(item["frame_count"]) for item in summaries)
    persistent = [
        item
        for item in summaries
        if int(item["frame_count"]) >= persistence_ratio * maximum_frames
    ]
    selected = max(
        persistent,
        key=lambda item: (
            float(item["median_image_span_px"] or 0.0),
            int(item["frame_count"]),
            float(item["mean_joint_confidence"] or 0.0),
            -int(item["object_id"]),
        ),
    )
    return int(selected["object_id"]), summaries


def index_track(
    records: Sequence[PoseRecord],
    object_id: int,
) -> dict[int, PoseRecord]:
    """Index one selected track by source frame number."""

    indexed: dict[int, PoseRecord] = {}
    for record in records:
        if record.object_id != object_id:
            continue
        if record.frame_num in indexed:
            raise BodyPoseEvaluationError(
                f"Duplicate object {object_id} at frame {record.frame_num}"
            )
        indexed[record.frame_num] = record
    return indexed


def index_primary_person(
    records: Sequence[PoseRecord],
    requested_id: int | None = None,
    continuation_span_ratio: float = 0.75,
) -> tuple[dict[int, PoseRecord], dict[str, Any]]:
    """Index the main person while allowing credible tracker-ID changes.

    The DeepStream tracker can assign a new ID after a short detection gap. The
    most persistent track is used as the anchor. When it is absent, only tracks
    whose median body extent is at least ``continuation_span_ratio`` of the
    anchor are eligible. This rejects the small, persistent false detections in
    the Fanbo recordings without borrowing detections from the YOLO baseline.
    """

    anchor_id, summaries = select_primary_track(records, requested_id)
    if requested_id is not None:
        fixed = index_track(records, anchor_id)
        return fixed, {
            "mode": "fixed_requested_track",
            "anchor_object_id": anchor_id,
            "continuation_object_ids": [],
            "selected_record_count_by_object_id": {
                str(anchor_id): len(fixed)
            },
            "continuation_span_ratio": None,
        }

    summaries_by_id = {
        int(item["object_id"]): item for item in summaries
    }
    anchor_span = float(
        summaries_by_id[anchor_id]["median_image_span_px"] or 0.0
    )
    minimum_span = continuation_span_ratio * anchor_span
    eligible_ids = {
        object_id
        for object_id, item in summaries_by_id.items()
        if float(item["median_image_span_px"] or 0.0) >= minimum_span
    }
    grouped_by_frame: dict[int, list[PoseRecord]] = defaultdict(list)
    for record in records:
        grouped_by_frame[record.frame_num].append(record)

    selected: dict[int, PoseRecord] = {}
    selected_counts: dict[int, int] = defaultdict(int)
    for frame_num, items in grouped_by_frame.items():
        anchor = next(
            (item for item in items if item.object_id == anchor_id), None
        )
        if anchor is not None:
            chosen = anchor
        else:
            candidates = [
                item for item in items if item.object_id in eligible_ids
            ]
            if not candidates:
                continue
            chosen = max(
                candidates,
                key=lambda item: float(
                    np.ptp(
                        item.pose25d[
                            np.isfinite(item.pose25d[:, 1]), 1
                        ]
                    )
                ),
            )
        selected[frame_num] = chosen
        selected_counts[chosen.object_id] += 1

    continuation_ids = sorted(
        object_id
        for object_id in selected_counts
        if object_id != anchor_id
    )
    return selected, {
        "mode": "persistent_anchor_with_large_track_continuations",
        "anchor_object_id": anchor_id,
        "anchor_median_image_span_px": anchor_span,
        "minimum_continuation_span_px": minimum_span,
        "continuation_object_ids": continuation_ids,
        "selected_record_count_by_object_id": {
            str(object_id): int(count)
            for object_id, count in sorted(selected_counts.items())
        },
        "continuation_span_ratio": continuation_span_ratio,
    }


def align_tracks_to_synced_timeline(
    left: dict[int, PoseRecord],
    right: dict[int, PoseRecord],
    synced: Sequence[SyncedFrame],
    frame_count: int,
) -> dict[str, np.ndarray]:
    """Map per-camera raw frame numbers onto synchronized stereo indices."""

    left_xy = np.full((frame_count, 17, 2), np.nan, dtype=np.float64)
    right_xy = np.full_like(left_xy, np.nan)
    left_conf = np.full((frame_count, 17), np.nan, dtype=np.float64)
    right_conf = np.full_like(left_conf, np.nan)
    left_pose3d = np.full((frame_count, 17, 4), np.nan, dtype=np.float64)
    right_pose3d = np.full_like(left_pose3d, np.nan)
    source_indices = np.full((frame_count, 2), -1, dtype=np.int64)
    track_present = np.zeros((frame_count, 2), dtype=bool)
    selected_object_ids = np.full((frame_count, 2), -1, dtype=np.int64)

    for synced_idx, item in enumerate(synced[:frame_count]):
        source_indices[synced_idx] = [item.left_idx, item.right_idx]
        left_record = left.get(item.left_idx)
        right_record = right.get(item.right_idx)
        if left_record is not None:
            selected = left_record.pose25d[COCO17_FROM_BODYPOSE34]
            left_xy[synced_idx] = selected[:, :2]
            left_conf[synced_idx] = selected[:, 3]
            left_pose3d[synced_idx] = left_record.pose3d[
                COCO17_FROM_BODYPOSE34
            ]
            track_present[synced_idx, 0] = True
            selected_object_ids[synced_idx, 0] = left_record.object_id
        if right_record is not None:
            selected = right_record.pose25d[COCO17_FROM_BODYPOSE34]
            right_xy[synced_idx] = selected[:, :2]
            right_conf[synced_idx] = selected[:, 3]
            right_pose3d[synced_idx] = right_record.pose3d[
                COCO17_FROM_BODYPOSE34
            ]
            track_present[synced_idx, 1] = True
            selected_object_ids[synced_idx, 1] = right_record.object_id

    return {
        "keypoints_left_2d_raw": left_xy,
        "keypoints_right_2d_raw": right_xy,
        "conf_left": left_conf,
        "conf_right": right_conf,
        "monocular_left_3d_mm": left_pose3d,
        "monocular_right_3d_mm": right_pose3d,
        "source_frame_indices": source_indices,
        "track_present": track_present,
        "selected_object_ids": selected_object_ids,
    }


def _rectify_sequence(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rectification: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Rectify a full keypoint sequence."""

    return np.asarray(
        [
            rectify_points(
                frame,
                camera_matrix,
                distortion,
                rectification,
                projection,
            )
            for frame in points
        ],
        dtype=np.float64,
    )


def apply_candidate_quality_filter(
    keypoints: np.ndarray,
    triangulation: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the same arm quality mask used by the formal SKT evaluation."""

    quality_config = section(config, "evaluation").get(
        "skt_quality_filter", {}
    )
    filtered = np.asarray(keypoints, dtype=np.float64).copy()
    if not quality_config or not quality_config.get("enabled", False):
        return filtered, {}
    minimum_confidence = float(
        quality_config.get("min_triang_conf", 0.2)
    )
    maximum_epipolar = float(
        quality_config.get("max_epipolar_px", 10.0)
    )
    stats: dict[str, int] = {}
    for joint_index in [
        int(value)
        for value in quality_config.get(
            "joint_indices", [5, 6, 7, 8, 9, 10]
        )
    ]:
        confidence = np.minimum(
            triangulation["triang_conf_left"][:, joint_index],
            triangulation["triang_conf_right"][:, joint_index],
        )
        epipolar = triangulation["epipolar_error"][:, joint_index]
        rejected = (
            ~np.isfinite(confidence)
            | (confidence < minimum_confidence)
            | ~np.isfinite(epipolar)
            | (epipolar > maximum_epipolar)
        )
        filtered[rejected, joint_index] = np.nan
        stats[f"joint_{joint_index}_masked_frames"] = int(
            np.count_nonzero(rejected)
        )
    return filtered, stats


def processed_angles(
    keypoints: np.ndarray,
    timestamps: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply the formal gap-fill and smoothing policy to derived angles."""

    evaluation = section(config, "evaluation")
    names = list(SEMANTIC_ANGLE_NAMES)
    raw = compute_angle_sequence(keypoints, names)
    _, radius, actual_ms = odd_window_from_ms(
        timestamps,
        float(evaluation.get("camera_smooth_window_ms", 200.0)),
    )
    maximum_gap = int(evaluation.get("max_gap_frames", 5))
    processed: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        filled, _ = fill_short_gaps(values, timestamps, maximum_gap)
        smoothed = moving_average(filled, radius)
        smoothed[~np.isfinite(filled)] = np.nan
        processed[name] = smoothed
    return processed, {
        "smoothing_radius_frames": int(radius),
        "smoothing_window_actual_ms": float(actual_ms),
        "maximum_filled_gap_frames": maximum_gap,
    }


def load_processed_skt_angles(
    path: Path,
    frame_count: int,
    timestamps: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one SKT result and apply the fixed formal angle policy."""

    with np.load(path, allow_pickle=True) as payload:
        required = (
            "keypoints",
            "triang_conf_left",
            "triang_conf_right",
            "epipolar_error",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise BodyPoseEvaluationError(
                f"SKT control {path} is missing arrays: {missing}"
            )
        keypoints = np.asarray(payload["keypoints"], dtype=np.float64)
        triangulation = {
            "triang_conf_left": np.asarray(
                payload["triang_conf_left"], dtype=np.float64
            ),
            "triang_conf_right": np.asarray(
                payload["triang_conf_right"], dtype=np.float64
            ),
            "epipolar_error": np.asarray(
                payload["epipolar_error"], dtype=np.float64
            ),
        }
        saved_timestamps = (
            np.asarray(payload["timestamps"], dtype=np.float64)
            if "timestamps" in payload
            else None
        )
    if len(keypoints) < frame_count:
        raise BodyPoseEvaluationError(
            f"SKT control {path} has {len(keypoints)} frames; "
            f"{frame_count} are required"
        )
    if (
        saved_timestamps is not None
        and len(saved_timestamps) >= frame_count
        and not np.allclose(
            saved_timestamps[:frame_count],
            timestamps[:frame_count],
            rtol=0.0,
            atol=1e-6,
            equal_nan=True,
        )
    ):
        raise BodyPoseEvaluationError(
            f"SKT control timestamps do not match the synchronized timeline: "
            f"{path}"
        )
    filtered, quality_stats = apply_candidate_quality_filter(
        keypoints[:frame_count],
        {
            name: values[:frame_count]
            for name, values in triangulation.items()
        },
        config,
    )
    filtered, depth_stats = apply_depth_consistency_filter(filtered, config)
    angles, angle_processing = processed_angles(
        filtered, timestamps[:frame_count], config
    )
    return angles, {
        "quality_filter_masked_frames": quality_stats,
        "depth_filter_masked_frames": depth_stats,
        "angle_processing": angle_processing,
    }


def paired_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    bins: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Summarize paired scalar differences and optional RULA-like bins."""

    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    common = np.isfinite(left) & np.isfinite(right)
    signed = left[common] - right[common]
    result: dict[str, Any] = {
        "total_count": int(left.size),
        "common_finite_count": int(np.count_nonzero(common)),
        "valid_ratio": float(np.mean(common)) if common.size else 0.0,
        "absolute_difference": distribution(np.abs(signed)),
        "bias": float(np.mean(signed)) if signed.size else None,
    }
    if bins:
        result["rula_bin_agreement"] = (
            float(
                np.mean(
                    rula_bin(left[common], list(bins))
                    == rula_bin(right[common], list(bins))
                )
            )
            if np.any(common)
            else None
        )
    return result


def load_reference_columns(
    path: Path,
    frame_count: int,
    columns: Sequence[str],
) -> dict[str, np.ndarray]:
    """Load fixed-alignment reference columns by their saved frame index."""

    result = {
        column: np.full(frame_count, np.nan, dtype=np.float64)
        for column in columns
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_text = row.get("Frame")
            if frame_text is None:
                raise BodyPoseEvaluationError(
                    f"Missing Frame column in {path}"
                )
            frame_index = int(frame_text)
            if frame_index < 0 or frame_index >= frame_count:
                continue
            for column in columns:
                value = row.get(column)
                if value not in (None, ""):
                    result[column][frame_index] = float(value)
    return result


def _joint_distributions(values: np.ndarray) -> dict[str, Any]:
    """Return one distribution per COCO joint."""

    return {
        name: distribution(values[:, index])
        for index, name in enumerate(COCO17_NAMES)
    }


def evaluate_candidate(
    *,
    left_json: Path,
    right_json: Path,
    config_path: Path,
    baseline_npz: Path,
    reference_timeseries: Path,
    same_input_baseline_npz: Path | None,
    output_directory: Path,
    candidate_name: str,
    left_track_id: int | None,
    right_track_id: int | None,
    maximum_frames: int | None,
    image_width: int,
    image_height: int,
) -> tuple[Path, Path, Path]:
    """Run the calibrated stereo and fixed-reference evaluation."""

    output_directory.mkdir(parents=True, exist_ok=True)
    npz_path = output_directory / "candidate_result.npz"
    metrics_path = output_directory / "metrics.json"
    timeseries_path = output_directory / "angle_timeseries.csv"
    manifest_path = output_directory / "artifact_manifest.json"
    for path in (npz_path, metrics_path, timeseries_path, manifest_path):
        if path.exists():
            raise BodyPoseEvaluationError(
                f"Refusing to overwrite existing result: {path}"
            )

    config = load_config(config_path)
    dataset = section(config, "dataset")
    calibration_config = section(config, "calibration")
    left_metadata = resolve_path(
        dataset.get("left_metadata"), must_exist=True
    )
    right_metadata = resolve_path(
        dataset.get("right_metadata"), must_exist=True
    )
    calibration_path = resolve_path(
        calibration_config.get("camera_params"), must_exist=True
    )
    assert left_metadata and right_metadata and calibration_path

    timestamps, synced, _, _ = build_synced_timeline(
        left_metadata,
        right_metadata,
        dataset.get(
            "timestamp_format", "seconds_microseconds_columns"
        ),
    )
    with np.load(baseline_npz, allow_pickle=True) as baseline_payload:
        baseline_keypoints = np.asarray(
            baseline_payload["keypoints"], dtype=np.float64
        )
        baseline_left_2d = np.asarray(
            baseline_payload["keypoints_left_2d_raw"], dtype=np.float64
        )
        baseline_right_2d = np.asarray(
            baseline_payload["keypoints_right_2d_raw"], dtype=np.float64
        )
    frame_count = min(
        len(synced),
        len(baseline_keypoints),
        maximum_frames if maximum_frames is not None else len(synced),
    )
    if frame_count <= 0:
        raise BodyPoseEvaluationError("No frames available for evaluation")
    timestamps = timestamps[:frame_count]

    left_records = load_deepstream_records(left_json)
    right_records = load_deepstream_records(right_json)
    selected_left, left_summaries = select_primary_track(
        left_records, left_track_id
    )
    selected_right, right_summaries = select_primary_track(
        right_records, right_track_id
    )
    left_index, left_selection = index_primary_person(
        left_records, left_track_id
    )
    right_index, right_selection = index_primary_person(
        right_records, right_track_id
    )
    aligned = align_tracks_to_synced_timeline(
        left_index,
        right_index,
        synced,
        frame_count,
    )

    with np.load(calibration_path) as camera:
        matrix_left = np.asarray(camera["mtx_l"], dtype=np.float64)
        matrix_right = np.asarray(camera["mtx_r"], dtype=np.float64)
        distortion_left = np.asarray(camera["dist_l"], dtype=np.float64)
        distortion_right = np.asarray(camera["dist_r"], dtype=np.float64)
        rotation = np.asarray(camera["R"], dtype=np.float64)
        translation = np.asarray(camera["T"], dtype=np.float64)
    rect_left, rect_right, projection_left, projection_right, *_ = (
        cv2.stereoRectify(
            matrix_left,
            distortion_left,
            matrix_right,
            distortion_right,
            (image_width, image_height),
            rotation,
            translation,
            alpha=0,
        )
    )
    keypoints_left_rect = _rectify_sequence(
        aligned["keypoints_left_2d_raw"],
        matrix_left,
        distortion_left,
        rect_left,
        projection_left,
    )
    keypoints_right_rect = _rectify_sequence(
        aligned["keypoints_right_2d_raw"],
        matrix_right,
        distortion_right,
        rect_right,
        projection_right,
    )

    skt_config = section(config, "skt")
    triangulation_config = TriangulationConfig.from_skt_config(skt_config)
    temporal_config = TemporalWindowConfig.from_skt_config(skt_config)
    first_pass = retriangulate_sequence(
        projection_left,
        projection_right,
        keypoints_left_rect,
        keypoints_right_rect,
        aligned["conf_left"],
        aligned["conf_right"],
        triangulation_config,
    )
    final = first_pass
    rescue_left = np.zeros((frame_count, 17), dtype=bool)
    rescue_right = np.zeros_like(rescue_left)
    if temporal_config.enabled:
        (
            rescued_left,
            rescued_right,
            rescued_confidence_left,
            rescued_confidence_right,
            rescue_left,
            rescue_right,
        ) = temporal_window_rescue_rectified(
            keypoints_left_rect,
            keypoints_right_rect,
            aligned["conf_left"],
            aligned["conf_right"],
            timestamps,
            first_pass["keypoints"],
            first_pass["stereo_quality"],
            temporal_config,
        )
        final = retriangulate_sequence(
            projection_left,
            projection_right,
            rescued_left,
            rescued_right,
            rescued_confidence_left,
            rescued_confidence_right,
            triangulation_config,
        )

    evaluation_keypoints, quality_stats = apply_candidate_quality_filter(
        final["keypoints"], final, config
    )
    evaluation_keypoints, depth_stats = apply_depth_consistency_filter(
        evaluation_keypoints, config
    )
    candidate_angles, angle_processing = processed_angles(
        evaluation_keypoints, timestamps, config
    )

    angle_matrix = np.column_stack(
        [candidate_angles[name] for name in SEMANTIC_ANGLE_NAMES]
    )
    np.savez_compressed(
        npz_path,
        schema_version=np.asarray(SCHEMA_VERSION),
        candidate_name=np.asarray(candidate_name),
        timestamps=timestamps,
        source_frame_indices=aligned["source_frame_indices"],
        track_present=aligned["track_present"],
        selected_object_ids=aligned["selected_object_ids"],
        selected_track_ids=np.asarray(
            [selected_left, selected_right], dtype=np.int64
        ),
        keypoints_3d_raw=first_pass["keypoints"],
        keypoints_3d=final["keypoints"],
        keypoints_3d_eval=evaluation_keypoints,
        keypoints_left_2d_raw=aligned["keypoints_left_2d_raw"],
        keypoints_right_2d_raw=aligned["keypoints_right_2d_raw"],
        keypoints_left_rect=final["keypoints_left_rect"],
        keypoints_right_rect=final["keypoints_right_rect"],
        conf_left=aligned["conf_left"],
        conf_right=aligned["conf_right"],
        triang_conf_left=final["triang_conf_left"],
        triang_conf_right=final["triang_conf_right"],
        epipolar_error_pre=final["epipolar_error_pre"],
        epipolar_error=final["epipolar_error"],
        reprojection_error=final["reprojection_error"],
        disparity_px=final["disparity_px"],
        stereo_quality=final["stereo_quality"],
        temporal_rescue_left=rescue_left,
        temporal_rescue_right=rescue_right,
        monocular_left_3d_mm=aligned["monocular_left_3d_mm"],
        monocular_right_3d_mm=aligned["monocular_right_3d_mm"],
        angle_names=np.asarray(SEMANTIC_ANGLE_NAMES),
        angles=angle_matrix,
    )

    candidate_to_baseline_3d = np.linalg.norm(
        final["keypoints"] - baseline_keypoints[:frame_count],
        axis=2,
    )
    left_to_baseline_2d = np.linalg.norm(
        aligned["keypoints_left_2d_raw"]
        - baseline_left_2d[:frame_count],
        axis=2,
    )
    right_to_baseline_2d = np.linalg.norm(
        aligned["keypoints_right_2d_raw"]
        - baseline_right_2d[:frame_count],
        axis=2,
    )
    reference_columns = load_reference_columns(
        reference_timeseries,
        frame_count,
        (
            "SKT_RightElbow_deg",
            "XsensFair_RightElbow_deg",
        ),
    )
    right_elbow_bins = (
        section(config, "evaluation")
        .get("rula_bins", {})
        .get("RightElbow", [60.0, 100.0])
    )
    candidate_external = paired_difference(
        candidate_angles["RightElbow"],
        reference_columns["XsensFair_RightElbow_deg"],
        right_elbow_bins,
    )
    baseline_external = paired_difference(
        reference_columns["SKT_RightElbow_deg"],
        reference_columns["XsensFair_RightElbow_deg"],
        right_elbow_bins,
    )
    same_input_control = None
    if same_input_baseline_npz is not None:
        control_angles, control_processing = load_processed_skt_angles(
            same_input_baseline_npz,
            frame_count,
            timestamps,
            config,
        )
        candidate_values = candidate_angles["RightElbow"]
        control_values = control_angles["RightElbow"]
        external_values = reference_columns["XsensFair_RightElbow_deg"]
        matched = (
            np.isfinite(candidate_values)
            & np.isfinite(control_values)
            & np.isfinite(external_values)
        )
        matched_candidate = np.where(matched, candidate_values, np.nan)
        matched_control = np.where(matched, control_values, np.nan)
        matched_external = np.where(matched, external_values, np.nan)
        candidate_matched = paired_difference(
            matched_candidate,
            matched_external,
            right_elbow_bins,
        )
        control_matched = paired_difference(
            matched_control,
            matched_external,
            right_elbow_bins,
        )
        candidate_matched_mae = candidate_matched[
            "absolute_difference"
        ]["mean"]
        control_matched_mae = control_matched[
            "absolute_difference"
        ]["mean"]
        matched_improvement = None
        if (
            candidate_matched_mae is not None
            and control_matched_mae is not None
            and float(control_matched_mae) > 0
        ):
            matched_improvement = 100.0 * (
                float(control_matched_mae) - float(candidate_matched_mae)
            ) / float(control_matched_mae)
        candidate_matched_rula = candidate_matched.get(
            "rula_bin_agreement"
        )
        control_matched_rula = control_matched.get("rula_bin_agreement")
        matched_angle_gate = bool(
            matched_improvement is not None
            and matched_improvement >= 5.0
            and candidate_matched_rula is not None
            and control_matched_rula is not None
            and float(candidate_matched_rula)
            >= float(control_matched_rula)
        )
        same_input_control = {
            "control_right_elbow_own_overlap": paired_difference(
                control_values,
                external_values,
                right_elbow_bins,
            ),
            "matched_common_finite_count": int(np.count_nonzero(matched)),
            "candidate_right_elbow_matched": candidate_matched,
            "control_right_elbow_matched": control_matched,
            "candidate_mae_improvement_percent_matched": (
                matched_improvement
            ),
            "matched_angle_gate_passed": matched_angle_gate,
            "control_processing": control_processing,
            "interpretation": (
                "Candidate and YOLO control used the same upright "
                "near-lossless proxy inputs. Matched metrics use only frames "
                "where candidate, control, and Xsens-derived reference are "
                "all finite; no alignment retuning was performed."
            ),
        }
    candidate_mae = candidate_external["absolute_difference"]["mean"]
    baseline_mae = baseline_external["absolute_difference"]["mean"]
    improvement_percent = None
    if (
        candidate_mae is not None
        and baseline_mae is not None
        and baseline_mae > 0
    ):
        improvement_percent = 100.0 * (
            float(baseline_mae) - float(candidate_mae)
        ) / float(baseline_mae)

    epipolar_pre_summary = distribution(final["epipolar_error_pre"])
    geometry_gate = {
        "thresholds": {
            "maximum_median_epipolar_px": 3.0,
            "maximum_p95_epipolar_px": 10.0,
        },
        "passed": bool(
            epipolar_pre_summary["median"] is not None
            and epipolar_pre_summary["p95"] is not None
            and float(epipolar_pre_summary["median"]) <= 3.0
            and float(epipolar_pre_summary["p95"]) <= 10.0
        ),
    }
    rula_candidate = candidate_external.get("rula_bin_agreement")
    rula_baseline = baseline_external.get("rula_bin_agreement")
    preliminary_angle_gate = bool(
        improvement_percent is not None
        and improvement_percent >= 5.0
        and rula_candidate is not None
        and rula_baseline is not None
        and float(rula_candidate) >= float(rula_baseline)
    )
    if preliminary_angle_gate and geometry_gate["passed"]:
        decision = "advance_to_far_view_gate"
    elif preliminary_angle_gate:
        decision = "promising_angle_signal_geometry_gate_failed"
    else:
        decision = "reject_before_far_view_gate"

    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate": candidate_name,
        "status": "completed",
        "decision": decision,
        "scope": {
            "dataset": dataset.get("name"),
            "frame_count": frame_count,
            "image_size": [image_width, image_height],
            "stereo_coordinate_system": "left camera",
            "stereo_units": "centimetres",
            "input_note": (
                "DeepStream processed per-view upright near-lossless NVDEC "
                "proxies; source raw frame indices were synchronized from "
                "the original hardware metadata."
            ),
            "external_comparison_note": (
                "Xsens-derived reference is an external comparison system, "
                "not absolute ground truth. The existing fixed alignment was "
                "reused without candidate-specific retuning."
            ),
        },
        "tracks": {
            "left_selected": selected_left,
            "right_selected": selected_right,
            "left_selection": left_selection,
            "right_selection": right_selection,
            "left_all": left_summaries,
            "right_all": right_summaries,
            "paired_track_presence_ratio": float(
                np.mean(np.all(aligned["track_present"], axis=1))
            ),
        },
        "geometry": {
            "epipolar_pre_px": epipolar_pre_summary,
            "epipolar_post_px": distribution(final["epipolar_error"]),
            "reprojection_px": distribution(final["reprojection_error"]),
            "finite_3d_joint_ratio": float(
                np.mean(np.isfinite(final["keypoints"]).all(axis=2))
            ),
            "per_joint_epipolar_pre_px": _joint_distributions(
                final["epipolar_error_pre"]
            ),
            "quality_filter_masked_frames": quality_stats,
            "depth_filter_masked_frames": depth_stats,
            "geometry_gate": geometry_gate,
        },
        "comparison_to_deterministic_pytorch": {
            "keypoint_2d_left_px": distribution(left_to_baseline_2d),
            "keypoint_2d_right_px": distribution(right_to_baseline_2d),
            "keypoint_3d_cm": distribution(candidate_to_baseline_3d),
            "right_elbow_angle_deg": paired_difference(
                candidate_angles["RightElbow"],
                compute_angle_sequence(
                    baseline_keypoints[:frame_count], ["RightElbow"]
                )["RightElbow"],
                right_elbow_bins,
            ),
        },
        "fixed_external_comparison": {
            "candidate_right_elbow": candidate_external,
            "baseline_right_elbow_same_overlap": baseline_external,
            "mae_improvement_percent": improvement_percent,
            "preliminary_angle_gate_passed": preliminary_angle_gate,
            "angle_processing": angle_processing,
        },
        "same_input_yolo_control": same_input_control,
        "provenance": {
            "config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
            "calibration": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
            "left_json": {
                "path": str(left_json.resolve()),
                "sha256": sha256_file(left_json),
            },
            "right_json": {
                "path": str(right_json.resolve()),
                "sha256": sha256_file(right_json),
            },
            "baseline_npz": {
                "path": str(baseline_npz.resolve()),
                "sha256": sha256_file(baseline_npz),
            },
            "fixed_reference_timeseries": {
                "path": str(reference_timeseries.resolve()),
                "sha256": sha256_file(reference_timeseries),
            },
            "same_input_baseline_npz": (
                {
                    "path": str(same_input_baseline_npz.resolve()),
                    "sha256": sha256_file(same_input_baseline_npz),
                }
                if same_input_baseline_npz is not None
                else None
            ),
        },
        "licensing": {
            "deepstream_benchmark_disclosure": (
                "Internal feasibility only. Do not publish DeepStream SDK "
                "benchmark or competitive-analysis numbers without checking "
                "the applicable NVIDIA DeepStream EULA and obtaining any "
                "required written permission."
            )
        },
    }
    metrics_path.write_text(
        json.dumps(jsonable(metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with timeseries_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "Frame",
            "Time_s",
            "NVIDIA_RightElbow_deg",
            "SKT_RightElbow_deg",
            "XsensDerived_RightElbow_deg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in range(frame_count):
            row: dict[str, Any] = {
                "Frame": frame_index,
                "Time_s": f"{float(timestamps[frame_index]):.6f}",
            }
            for column, values in (
                ("NVIDIA_RightElbow_deg", candidate_angles["RightElbow"]),
                (
                    "SKT_RightElbow_deg",
                    reference_columns["SKT_RightElbow_deg"],
                ),
                (
                    "XsensDerived_RightElbow_deg",
                    reference_columns["XsensFair_RightElbow_deg"],
                ),
            ):
                value = values[frame_index]
                row[column] = (
                    f"{float(value):.6f}" if np.isfinite(value) else ""
                )
            writer.writerow(row)

    artifacts = []
    for path, media_type in (
        (npz_path, "application/x-numpy-npz"),
        (metrics_path, "application/json"),
        (timeseries_path, "text/csv"),
    ):
        artifacts.append(
            {
                "relative_path": path.name,
                "media_type": media_type,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "research_artifact_manifest_v1",
        "candidate": candidate_name,
        "artifacts": artifacts,
        "excluded_sensitive_assets": [
            "DeepStream model weights and TensorRT engines",
            "original or proxy videos",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return npz_path, metrics_path, manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-json", type=Path, required=True)
    parser.add_argument("--right-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--reference-timeseries", type=Path, required=True)
    parser.add_argument(
        "--same-input-baseline-npz",
        type=Path,
        help=(
            "Optional YOLO/SKT control generated from the exact same proxy "
            "videos as the NVIDIA candidate"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-name", default="nvidia_bodypose3dnet_accuracy"
    )
    parser.add_argument("--left-track-id", type=int)
    parser.add_argument("--right-track-id", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    """Resolve one user path against the project root."""

    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (PROJECT_ROOT / expanded).resolve()
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the adapter and report the saved formal evidence."""

    args = parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise BodyPoseEvaluationError("--max-frames must be positive")
    if args.width <= 0 or args.height <= 0:
        raise BodyPoseEvaluationError("Image dimensions must be positive")
    npz_path, metrics_path, manifest_path = evaluate_candidate(
        left_json=_project_path(args.left_json),
        right_json=_project_path(args.right_json),
        config_path=_project_path(args.config),
        baseline_npz=_project_path(args.baseline_npz),
        reference_timeseries=_project_path(args.reference_timeseries),
        same_input_baseline_npz=(
            _project_path(args.same_input_baseline_npz)
            if args.same_input_baseline_npz is not None
            else None
        ),
        output_directory=_project_path(args.output_dir),
        candidate_name=str(args.candidate_name),
        left_track_id=args.left_track_id,
        right_track_id=args.right_track_id,
        maximum_frames=args.max_frames,
        image_width=args.width,
        image_height=args.height,
    )
    print(f"Wrote candidate NPZ: {npz_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote artifact manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
