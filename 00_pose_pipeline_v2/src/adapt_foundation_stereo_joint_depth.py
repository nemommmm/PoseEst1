#!/usr/bin/env python
"""Build a 3D skeleton from left-view joints and stereo-model disparities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from common.config import load_config, resolve_path, section
from common.research_candidate import CandidateResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sample_joint_disparity(
    disparity: np.ndarray,
    points_xy: np.ndarray,
    patch_size: int = 7,
    maximum_mad_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample robust local disparities at joint locations."""
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    image = np.asarray(disparity, dtype=np.float64)
    points = np.asarray(points_xy, dtype=np.float64)
    if image.ndim != 2 or points.shape != (17, 2):
        raise ValueError("Expected disparity (H,W) and points (17,2)")
    radius = patch_size // 2
    values = np.full(17, np.nan, dtype=np.float64)
    mad = np.full(17, np.nan, dtype=np.float64)
    for joint_index, point in enumerate(points):
        if not np.isfinite(point).all():
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
        patch = image[y0:y1, x0:x1]
        finite = patch[np.isfinite(patch) & (patch > 0)]
        if finite.size < max(3, patch_size):
            continue
        median = float(np.median(finite))
        local_mad = float(np.median(np.abs(finite - median)))
        mad[joint_index] = local_mad
        if local_mad <= maximum_mad_px:
            values[joint_index] = median
    return values, mad


def rectify_points_sequence(
    points_xy: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rectification: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Rectify a sequence of image points without using the other view."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (17, 2):
        raise ValueError("Expected points with shape (N,17,2)")
    output = np.full_like(points, np.nan, dtype=np.float64)
    for frame_index, frame_points in enumerate(points):
        valid = np.isfinite(frame_points).all(axis=1)
        if not np.any(valid):
            continue
        rectified = cv2.undistortPoints(
            frame_points[valid].reshape(-1, 1, 2),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
            R=np.asarray(rectification, dtype=np.float64),
            P=np.asarray(projection, dtype=np.float64),
        ).reshape(-1, 2)
        output[frame_index, valid] = rectified
    return output


def restore_full_resolution_disparity(
    disparity_scaled: np.ndarray,
    full_size: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """Resize model disparity to full resolution and restore pixel units."""
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0, 1]")
    disparity = np.asarray(disparity_scaled, dtype=np.float32)
    if disparity.ndim != 2:
        raise ValueError("Expected a two-dimensional disparity image")
    width, height = full_size
    restored = cv2.resize(
        disparity,
        (int(width), int(height)),
        interpolation=cv2.INTER_LINEAR,
    )
    return restored / float(scale)


def disparity_to_left_camera_cm(
    points_rectified: np.ndarray,
    disparity_px: np.ndarray,
    projection_left: np.ndarray,
    baseline_cm: float,
) -> np.ndarray:
    """Back-project rectified left pixels and disparity to metric 3D."""
    points = np.asarray(points_rectified, dtype=np.float64)
    disparity = np.asarray(disparity_px, dtype=np.float64)
    projection = np.asarray(projection_left, dtype=np.float64)
    if points.shape[-2:] != (17, 2):
        raise ValueError("points_rectified must end with shape (17,2)")
    if disparity.shape != points.shape[:-1]:
        raise ValueError("disparity shape must match points without XY axis")
    fx = float(projection[0, 0])
    fy = float(projection[1, 1])
    cx = float(projection[0, 2])
    cy = float(projection[1, 2])
    depth = np.where(
        np.isfinite(disparity) & (disparity > 0),
        fx * float(baseline_cm) / disparity,
        np.nan,
    )
    output = np.full(points.shape[:-1] + (3,), np.nan, dtype=np.float64)
    output[..., 2] = depth
    output[..., 0] = (points[..., 0] - cx) * depth / fx
    output[..., 1] = (points[..., 1] - cy) * depth / fy
    output[~np.isfinite(points).all(axis=-1)] = np.nan
    return output


def adapt(
    disparity_path: Path,
    baseline_path: Path,
    config_path: Path,
    output_path: Path,
    candidate_name: str,
    maximum_mad_px: float,
    minimum_confidence: float,
) -> Path:
    """Create one canonical dense-stereo joint-depth candidate."""
    config = load_config(config_path)
    dataset = section(config, "dataset")
    calibration_path = resolve_path(
        section(config, "calibration").get("camera_params"),
        must_exist=True,
    )
    assert calibration_path
    with np.load(disparity_path, allow_pickle=False) as disparity_payload:
        joint_disparity = np.asarray(
            disparity_payload["joint_disparity_px"],
            dtype=np.float64,
        )
        local_mad = np.asarray(
            disparity_payload["local_disparity_mad_px"],
            dtype=np.float64,
        )
        inference_ms = (
            np.asarray(disparity_payload["inference_time_ms"])
            if "inference_time_ms" in disparity_payload
            else np.asarray([], dtype=np.float64)
        )
        inference_metadata = (
            json.loads(
                str(np.asarray(disparity_payload["metadata_json"]).item())
            )
            if "metadata_json" in disparity_payload
            else {}
        )
    metadata_size = inference_metadata.get("image_size")
    if (
        isinstance(metadata_size, list)
        and len(metadata_size) == 2
        and all(int(value) > 0 for value in metadata_size)
    ):
        width, height = (int(value) for value in metadata_size)
    else:
        left_video = resolve_path(dataset.get("left_video"), must_exist=True)
        assert left_video is not None
        capture = cv2.VideoCapture(str(left_video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
    with np.load(calibration_path) as calibration:
        matrix_left = np.asarray(calibration["mtx_l"], dtype=np.float64)
        distortion_left = np.asarray(calibration["dist_l"], dtype=np.float64)
        matrix_right = np.asarray(calibration["mtx_r"], dtype=np.float64)
        distortion_right = np.asarray(
            calibration["dist_r"],
            dtype=np.float64,
        )
        rotation = np.asarray(calibration["R"], dtype=np.float64)
        translation = np.asarray(calibration["T"], dtype=np.float64)
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
        (width, height),
        rotation,
        translation,
        alpha=0,
    )
    baseline_cm = float(np.linalg.norm(translation))
    with np.load(baseline_path, allow_pickle=True) as baseline:
        timestamps = np.asarray(baseline["timestamps"], dtype=np.float64)
        points_left_raw = np.asarray(
            baseline["keypoints_left_2d_raw"],
            dtype=np.float64,
        )
        points_right_raw = np.asarray(
            baseline["keypoints_right_2d_raw"],
            dtype=np.float64,
        )
        confidence = np.asarray(baseline["conf_left"], dtype=np.float64)
    points_left = rectify_points_sequence(
        points_left_raw,
        matrix_left,
        distortion_left,
        rectification_left,
        projection_left,
    )
    points_right = rectify_points_sequence(
        points_right_raw,
        matrix_right,
        distortion_right,
        rectification_right,
        projection_right,
    )
    count = min(
        len(timestamps),
        len(points_left),
        len(joint_disparity),
    )
    timestamps = timestamps[:count]
    points_left = points_left[:count]
    points_right = points_right[:count]
    confidence = confidence[:count]
    joint_disparity = joint_disparity[:count]
    local_mad = local_mad[:count]
    rejected = (
        ~np.isfinite(joint_disparity)
        | (joint_disparity <= 0)
        | ~np.isfinite(local_mad)
        | (local_mad > maximum_mad_px)
        | ~np.isfinite(confidence)
        | (confidence < minimum_confidence)
    )
    accepted_disparity = joint_disparity.copy()
    accepted_disparity[rejected] = np.nan
    keypoints = disparity_to_left_camera_cm(
        points_left,
        accepted_disparity,
        projection_left,
        baseline_cm,
    )
    predicted_right = points_left.copy()
    predicted_right[..., 0] -= accepted_disparity
    right_difference = np.linalg.norm(
        predicted_right - points_right,
        axis=2,
    )
    epipolar = np.abs(points_left[..., 1] - points_right[..., 1])
    quality = np.clip(
        confidence * np.exp(-np.nan_to_num(local_mad, nan=100.0) / 2.0),
        0.0,
        1.0,
    )
    quality[rejected] = 0.0
    result = CandidateResult(
        candidate_name=candidate_name,
        timestamps=timestamps,
        keypoints_3d=keypoints,
        keypoints_3d_raw=keypoints,
        confidence_2d=confidence,
        epipolar_error_px=epipolar,
        reprojection_error_px=right_difference,
        joint_quality=quality,
        prior_weight=np.ones_like(quality),
        stage_time_ms={"dense_stereo_inference": inference_ms},
        metadata={
            "route": "stereo_dense_depth",
            "source_view": "left",
            "coordinate_frame": "left_camera",
            "coordinate_unit": "cm",
            "joint_convention": "COCO-17",
            "semantic_source": "YOLOv8m left-view 2D keypoints",
            "depth_source": candidate_name,
            "right_yolo_usage": "diagnostic_only",
            "left_joint_rectification": (
                "independent_from_raw_left_2d_without_right-view correction"
            ),
            "joint_patch_size": 7,
            "maximum_local_disparity_mad_px": float(maximum_mad_px),
            "minimum_joint_confidence": float(minimum_confidence),
            "calibration_sha256": sha256_file(calibration_path),
            "disparity_sha256": sha256_file(disparity_path),
            "baseline_sha256": sha256_file(baseline_path),
            "inference": inference_metadata,
            "reference_policy": (
                "Xsens-derived reference is external comparison only."
            ),
        },
        extra_arrays={
            "joint_disparity_px": joint_disparity,
            "accepted_joint_disparity_px": accepted_disparity,
            "local_disparity_mad_px": local_mad,
            "keypoints_left_rect": points_left,
            "predicted_keypoints_right_rect": predicted_right,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return result.save(output_path)


def _resolve(path: Path) -> Path:
    """Resolve one CLI path against the project root."""
    return (
        path.expanduser().resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disparity", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--maximum-mad-px", type=float, default=2.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the adapter."""
    args = parse_args(argv)
    print(
        adapt(
            _resolve(args.disparity),
            _resolve(args.baseline),
            _resolve(args.config),
            _resolve(args.output),
            args.candidate_name,
            float(args.maximum_mad_px),
            float(args.minimum_confidence),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
