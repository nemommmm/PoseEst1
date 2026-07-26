#!/usr/bin/env python
"""Adapt official BodyPose3DNet per-view outputs to canonical candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.config import load_config, resolve_path, section
from common.research_candidate import CandidateResult
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


def right_camera_to_left(
    points_right_cm: np.ndarray,
    rotation_left_to_right: np.ndarray,
    translation_left_to_right_cm: np.ndarray,
) -> np.ndarray:
    """Transform right-camera points into the left-camera coordinate frame."""
    points = np.asarray(points_right_cm, dtype=np.float64)
    rotation = np.asarray(rotation_left_to_right, dtype=np.float64)
    translation = np.asarray(
        translation_left_to_right_cm,
        dtype=np.float64,
    ).reshape(3)
    flat = points.reshape(-1, 3)
    output = np.full_like(flat, np.nan)
    valid = np.isfinite(flat).all(axis=1)
    output[valid] = (
        rotation.T @ (flat[valid] - translation).T
    ).T
    return output.reshape(points.shape)


def _resolve_cli_path(path: Path) -> Path:
    """Resolve one CLI path against the project root."""
    return (
        path.expanduser().resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def adapt(
    *,
    left_json: Path,
    right_json: Path,
    config_path: Path,
    output_dir: Path,
    candidate_prefix: str,
    minimum_confidence: float,
    maximum_frames: int | None,
    left_track_id: int | None,
    right_track_id: int | None,
) -> tuple[Path, Path, Path]:
    """Create separate left- and right-view canonical pose candidates."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
    if maximum_frames is not None:
        timestamps = timestamps[:maximum_frames]
        synced = synced[:maximum_frames]
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
        len(timestamps),
    )
    left_pose = (
        np.asarray(aligned["monocular_left_3d_mm"][..., :3])
        / 10.0
    )
    right_pose_native = (
        np.asarray(aligned["monocular_right_3d_mm"][..., :3])
        / 10.0
    )
    left_confidence = np.asarray(aligned["conf_left"], dtype=np.float64)
    right_confidence = np.asarray(
        aligned["conf_right"],
        dtype=np.float64,
    )
    left_pose[left_confidence < minimum_confidence] = np.nan
    right_pose_native[right_confidence < minimum_confidence] = np.nan
    with np.load(calibration_path) as calibration:
        rotation = np.asarray(calibration["R"], dtype=np.float64)
        translation = np.asarray(calibration["T"], dtype=np.float64)
    right_pose_left_frame = right_camera_to_left(
        right_pose_native,
        rotation,
        translation,
    )
    provenance = {
        "source_model": "NVIDIA BodyPose3DNet",
        "source_format": "official DeepStream reference-app pose JSON",
        "coordinate_unit": "cm",
        "joint_convention": "COCO-17 mapped from NVIDIA BodyPose34",
        "minimum_joint_confidence": float(minimum_confidence),
        "reference_policy": (
            "Xsens-derived reference is external comparison only; "
            "no candidate-specific alignment or parameter tuning."
        ),
        "config_sha256": sha256_file(config_path),
        "calibration_sha256": sha256_file(calibration_path),
        "left_json_sha256": sha256_file(left_json),
        "right_json_sha256": sha256_file(right_json),
    }
    left_path = output_dir / "candidate_monocular_left.npz"
    right_path = output_dir / "candidate_monocular_right.npz"
    CandidateResult(
        candidate_name=f"{candidate_prefix}_monocular_left",
        timestamps=timestamps,
        keypoints_3d=left_pose,
        keypoints_3d_raw=left_pose,
        confidence_2d=left_confidence,
        joint_quality=left_confidence,
        prior_weight=np.ones_like(left_confidence),
        metadata={
            **provenance,
            "route": "monocular_left",
            "source_view": "left",
            "coordinate_frame": "left_camera",
            "selected_track_id": selected_left,
            "track_selection": left_selection,
        },
        extra_arrays={
            "source_frame_indices": aligned["source_frame_indices"],
            "selected_object_ids": aligned["selected_object_ids"][:, 0],
        },
    ).save(left_path)
    CandidateResult(
        candidate_name=f"{candidate_prefix}_monocular_right",
        timestamps=timestamps,
        keypoints_3d=right_pose_left_frame,
        keypoints_3d_raw=right_pose_left_frame,
        confidence_2d=right_confidence,
        joint_quality=right_confidence,
        prior_weight=np.ones_like(right_confidence),
        metadata={
            **provenance,
            "route": "monocular_right",
            "source_view": "right",
            "coordinate_frame": "left_camera",
            "right_to_left_transform": (
                "X_left = R.T @ (X_right - T)"
            ),
            "selected_track_id": selected_right,
            "track_selection": right_selection,
        },
        extra_arrays={
            "source_frame_indices": aligned["source_frame_indices"],
            "selected_object_ids": aligned["selected_object_ids"][:, 1],
            "keypoints_3d_right_camera_cm": right_pose_native,
        },
    ).save(right_path)
    manifest_path = output_dir / "monocular_manifest.json"
    manifest = {
        "schema_version": "nvidia_bodypose3d_monocular_manifest_v1",
        "candidate_prefix": candidate_prefix,
        "frame_count": len(timestamps),
        "left_tracks": left_tracks,
        "right_tracks": right_tracks,
        "left_result": {
            "path": left_path.name,
            "sha256": sha256_file(left_path),
        },
        "right_result": {
            "path": right_path.name,
            "sha256": sha256_file(right_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return left_path, right_path, manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-json", type=Path, required=True)
    parser.add_argument("--right-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--left-track-id", type=int)
    parser.add_argument("--right-track-id", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the adapter."""
    args = parse_args(argv)
    paths = adapt(
        left_json=_resolve_cli_path(args.left_json),
        right_json=_resolve_cli_path(args.right_json),
        config_path=_resolve_cli_path(args.config),
        output_dir=_resolve_cli_path(args.output_dir),
        candidate_prefix=args.candidate_prefix,
        minimum_confidence=float(args.minimum_confidence),
        maximum_frames=args.max_frames,
        left_track_id=args.left_track_id,
        right_track_id=args.right_track_id,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

