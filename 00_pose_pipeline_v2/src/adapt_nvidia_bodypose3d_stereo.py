#!/usr/bin/env python
"""Adapt paired NVIDIA BodyPose3DNet outputs to calibrated stereo 3D."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from common.config import load_config, resolve_path, section
from common.dataset import apply_depth_consistency_filter
from common.research_candidate import CandidateResult
from common.triangulation import (
    TemporalWindowConfig,
    TriangulationConfig,
    rectify_points,
    retriangulate_sequence,
    temporal_window_rescue_rectified,
)
from evaluate_nvidia_bodypose3d_stereo import (
    align_tracks_to_synced_timeline,
    index_primary_person,
    load_deepstream_records,
    select_primary_track,
)
from stereo_loader import build_synced_timeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_cli_path(path: Path) -> Path:
    """Resolve a CLI path against the project root."""
    return (
        path.expanduser().resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def rectify_sequence(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rectification: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Rectify a complete 2D keypoint sequence."""
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


def _load_calibration(
    path: Path,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    """Load fixed calibration and derive stereo-rectification matrices."""
    with np.load(path) as camera:
        matrix_left = np.asarray(camera["mtx_l"], dtype=np.float64)
        matrix_right = np.asarray(camera["mtx_r"], dtype=np.float64)
        distortion_left = np.asarray(camera["dist_l"], dtype=np.float64)
        distortion_right = np.asarray(camera["dist_r"], dtype=np.float64)
        rotation = np.asarray(camera["R"], dtype=np.float64)
        translation = np.asarray(camera["T"], dtype=np.float64)
    (
        rectification_left,
        rectification_right,
        projection_left,
        projection_right,
        _,
        _,
        _,
    ) = cv2.stereoRectify(
        matrix_left,
        distortion_left,
        matrix_right,
        distortion_right,
        image_size,
        rotation,
        translation,
        alpha=0,
    )
    return (
        matrix_left,
        distortion_left,
        matrix_right,
        distortion_right,
        rectification_left,
        rectification_right,
        projection_left,
        projection_right,
    )


def _apply_formal_quality_filters(
    keypoints: np.ndarray,
    triangulation: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the fixed SKT quality and depth filters without retuning."""
    filtered = np.asarray(keypoints, dtype=np.float64).copy()
    quality_config = section(config, "evaluation").get(
        "skt_quality_filter",
        {},
    )
    masked: dict[str, int] = {}
    if quality_config and quality_config.get("enabled", False):
        minimum_confidence = float(
            quality_config.get("min_triang_conf", 0.2)
        )
        maximum_epipolar = float(
            quality_config.get("max_epipolar_px", 10.0)
        )
        for joint_index in [
            int(value)
            for value in quality_config.get(
                "joint_indices",
                [5, 6, 7, 8, 9, 10],
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
            masked[str(joint_index)] = int(np.count_nonzero(rejected))
    filtered, depth_stats = apply_depth_consistency_filter(filtered, config)
    return filtered, {
        "quality_masked_frames_by_joint": masked,
        "depth_filter_masked_frames": depth_stats,
    }


def adapt(
    *,
    left_json: Path,
    right_json: Path,
    config_path: Path,
    output_path: Path,
    candidate_name: str,
    maximum_frames: int | None,
    image_width: int,
    image_height: int,
    left_track_id: int | None,
    right_track_id: int | None,
) -> Path:
    """Create one canonical calibrated-stereo BodyPose3DNet candidate."""
    config = load_config(config_path)
    dataset = section(config, "dataset")
    left_metadata = resolve_path(
        dataset.get("left_metadata"),
        must_exist=True,
    )
    right_metadata = resolve_path(
        dataset.get("right_metadata"),
        must_exist=True,
    )
    calibration_path = resolve_path(
        section(config, "calibration").get("camera_params"),
        must_exist=True,
    )
    assert left_metadata and right_metadata and calibration_path
    timestamps, synced, _, _ = build_synced_timeline(
        left_metadata,
        right_metadata,
        dataset.get(
            "timestamp_format",
            "seconds_microseconds_columns",
        ),
    )
    frame_count = min(
        len(synced),
        maximum_frames if maximum_frames is not None else len(synced),
    )
    timestamps = timestamps[:frame_count]
    synced = synced[:frame_count]

    left_records = load_deepstream_records(left_json)
    right_records = load_deepstream_records(right_json)
    selected_left, left_tracks = select_primary_track(
        left_records,
        left_track_id,
    )
    selected_right, right_tracks = select_primary_track(
        right_records,
        right_track_id,
    )
    left_index, left_selection = index_primary_person(
        left_records,
        left_track_id,
    )
    right_index, right_selection = index_primary_person(
        right_records,
        right_track_id,
    )
    aligned = align_tracks_to_synced_timeline(
        left_index,
        right_index,
        synced,
        frame_count,
    )
    (
        matrix_left,
        distortion_left,
        matrix_right,
        distortion_right,
        rectification_left,
        rectification_right,
        projection_left,
        projection_right,
    ) = _load_calibration(
        calibration_path,
        (image_width, image_height),
    )
    keypoints_left_rect = rectify_sequence(
        aligned["keypoints_left_2d_raw"],
        matrix_left,
        distortion_left,
        rectification_left,
        projection_left,
    )
    keypoints_right_rect = rectify_sequence(
        aligned["keypoints_right_2d_raw"],
        matrix_right,
        distortion_right,
        rectification_right,
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
    filtered, filter_stats = _apply_formal_quality_filters(
        final["keypoints"],
        final,
        config,
    )
    confidence = np.stack(
        [aligned["conf_left"], aligned["conf_right"]],
        axis=2,
    )
    metadata = {
        "source_model": "NVIDIA BodyPose3DNet",
        "source_format": "official DeepStream reference-app pose JSON",
        "route": "calibrated_stereo_from_exported_2d",
        "coordinate_frame": "left_camera",
        "coordinate_unit": "cm",
        "joint_convention": "COCO-17 mapped from NVIDIA BodyPose34",
        "selected_track_ids": [selected_left, selected_right],
        "track_selection": {
            "left": left_selection,
            "right": right_selection,
        },
        "available_tracks": {
            "left": left_tracks,
            "right": right_tracks,
        },
        "paired_track_presence_ratio": float(
            np.mean(np.all(aligned["track_present"], axis=1))
        ),
        "formal_filter_statistics": filter_stats,
        "config_sha256": sha256_file(config_path),
        "calibration_sha256": sha256_file(calibration_path),
        "left_json_sha256": sha256_file(left_json),
        "right_json_sha256": sha256_file(right_json),
        "reference_policy": (
            "Xsens-derived reference is an external comparison only; "
            "no candidate-specific alignment or parameter tuning."
        ),
    }
    CandidateResult(
        candidate_name=candidate_name,
        timestamps=timestamps,
        keypoints_3d=filtered,
        keypoints_3d_raw=first_pass["keypoints"],
        confidence_2d=confidence,
        epipolar_error_px=final["epipolar_error"],
        reprojection_error_px=final["reprojection_error"],
        joint_quality=final["stereo_quality"],
        prior_weight=np.zeros((frame_count, 17), dtype=np.float64),
        metadata=metadata,
        extra_arrays={
            "source_frame_indices": aligned["source_frame_indices"],
            "selected_object_ids": aligned["selected_object_ids"],
            "track_present": aligned["track_present"],
            "keypoints_left_2d_raw": aligned["keypoints_left_2d_raw"],
            "keypoints_right_2d_raw": aligned["keypoints_right_2d_raw"],
            "keypoints_left_rect": final["keypoints_left_rect"],
            "keypoints_right_rect": final["keypoints_right_rect"],
            "conf_left": aligned["conf_left"],
            "conf_right": aligned["conf_right"],
            "triang_conf_left": final["triang_conf_left"],
            "triang_conf_right": final["triang_conf_right"],
            "epipolar_error_pre": final["epipolar_error_pre"],
            "disparity_px": final["disparity_px"],
            "temporal_rescue_left": rescue_left,
            "temporal_rescue_right": rescue_right,
            "monocular_left_3d_mm": aligned["monocular_left_3d_mm"],
            "monocular_right_3d_mm": aligned["monocular_right_3d_mm"],
        },
    ).save(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "nvidia_bodypose3d_stereo_manifest_v1",
                "candidate": candidate_name,
                "frame_count": frame_count,
                "candidate_npz": {
                    "path": output_path.name,
                    "sha256": sha256_file(output_path),
                },
                "excluded_sensitive_assets": [
                    "DeepStream model weights and TensorRT engines",
                    "original or proxy videos",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-json", type=Path, required=True)
    parser.add_argument("--right-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--left-track-id", type=int)
    parser.add_argument("--right-track-id", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stereo adapter."""
    args = parse_args(argv)
    output = adapt(
        left_json=_resolve_cli_path(args.left_json),
        right_json=_resolve_cli_path(args.right_json),
        config_path=_resolve_cli_path(args.config),
        output_path=_resolve_cli_path(args.output),
        candidate_name=args.candidate_name,
        maximum_frames=args.max_frames,
        image_width=args.width,
        image_height=args.height,
        left_track_id=args.left_track_id,
        right_track_id=args.right_track_id,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
