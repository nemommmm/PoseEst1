"""Unified output helpers for external human-prior research candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.angles import COCO17_NAMES, SEMANTIC_ANGLE_NAMES, compute_angle_sequence


COCO_ALIASES = {
    "nose": "Nose",
    "left_eye": "LEye",
    "right_eye": "REye",
    "left_ear": "LEar",
    "right_ear": "REar",
    "left_shoulder": "LShoulder",
    "right_shoulder": "RShoulder",
    "left_elbow": "LElbow",
    "right_elbow": "RElbow",
    "left_wrist": "LWrist",
    "right_wrist": "RWrist",
    "left_hip": "LHip",
    "right_hip": "RHip",
    "left_knee": "LKnee",
    "right_knee": "RKnee",
    "left_ankle": "LAnkle",
    "right_ankle": "RAnkle",
}

BONES = {
    "left_upper_arm": (5, 7),
    "right_upper_arm": (6, 8),
    "left_forearm": (7, 9),
    "right_forearm": (8, 10),
    "left_thigh": (11, 13),
    "right_thigh": (12, 14),
    "left_shank": (13, 15),
    "right_shank": (14, 16),
}


def canonical_joint_name(name: str) -> str:
    """Normalize common external joint labels to the project COCO spelling."""
    stripped = name.strip()
    if stripped in COCO17_NAMES:
        return stripped
    key = stripped.lower().replace(" ", "_").replace("-", "_")
    return COCO_ALIASES.get(key, stripped)


def map_to_coco17(
    keypoints: np.ndarray,
    source_names: Sequence[str],
) -> np.ndarray:
    """Map an arbitrary named skeleton to a NaN-filled COCO-17 array."""
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] != len(source_names) or points.shape[2] != 3:
        raise ValueError("keypoints must have shape (frames, len(source_names), 3)")
    result = np.full((len(points), len(COCO17_NAMES), 3), np.nan, dtype=np.float64)
    target_index = {name: idx for idx, name in enumerate(COCO17_NAMES)}
    for source_idx, raw_name in enumerate(source_names):
        name = canonical_joint_name(raw_name)
        if name in target_index:
            result[:, target_index[name], :] = points[:, source_idx, :]
    return result


def convert_to_centimeters(keypoints: np.ndarray, unit: str) -> np.ndarray:
    """Convert millimeter, centimeter, or meter coordinates to centimeters."""
    factors = {"mm": 0.1, "cm": 1.0, "m": 100.0}
    normalized = unit.strip().lower()
    if normalized not in factors:
        raise ValueError(f"unsupported coordinate unit: {unit}")
    return np.asarray(keypoints, dtype=np.float64) * factors[normalized]


def transform_points(keypoints: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid homogeneous transform to a keypoint sequence."""
    points = np.asarray(keypoints, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("keypoints must have shape (frames, joints, 3)")
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    output = np.full_like(flat, np.nan)
    homogeneous = np.column_stack([flat[valid], np.ones(valid.sum())])
    output[valid] = (matrix @ homogeneous.T).T[:, :3]
    return output.reshape(points.shape)


def compute_bone_statistics(keypoints_3d: np.ndarray) -> dict[str, dict[str, float | None]]:
    """Compute robust bone-length location and temporal variation in centimeters."""
    poses = np.asarray(keypoints_3d, dtype=np.float64)
    result: dict[str, dict[str, float | None]] = {}
    for name, (joint_a, joint_b) in BONES.items():
        lengths = np.linalg.norm(poses[:, joint_a] - poses[:, joint_b], axis=1)
        finite = lengths[np.isfinite(lengths)]
        result[name] = {
            "median_cm": float(np.median(finite)) if finite.size else None,
            "std_cm": float(np.std(finite)) if finite.size else None,
            "valid_ratio": float(finite.size / len(poses)) if len(poses) else 0.0,
        }
    return result


@dataclass
class CandidateResult:
    """Canonical result representation shared by all research candidates."""

    candidate_name: str
    timestamps: np.ndarray
    keypoints_3d: np.ndarray
    angle_names: tuple[str, ...] = SEMANTIC_ANGLE_NAMES
    confidence_2d: np.ndarray | None = None
    epipolar_error_px: np.ndarray | None = None
    reprojection_error_px: np.ndarray | None = None
    stage_time_ms: Mapping[str, np.ndarray | float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate shapes and required semantic joints before serialization."""
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        keypoints = np.asarray(self.keypoints_3d, dtype=np.float64)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if keypoints.shape != (len(timestamps), 17, 3):
            raise ValueError("keypoints_3d must have shape (frames, 17, 3)")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            raise ValueError("timestamps must be non-decreasing")
        for field_name, values in (
            ("confidence_2d", self.confidence_2d),
            ("epipolar_error_px", self.epipolar_error_px),
            ("reprojection_error_px", self.reprojection_error_px),
        ):
            if values is not None and np.asarray(values).shape[:2] != (len(timestamps), 17):
                raise ValueError(f"{field_name} must begin with shape (frames, 17)")

    def save(self, path: Path) -> Path:
        """Validate and save a self-describing compressed NPZ."""
        self.validate()
        angles = compute_angle_sequence(self.keypoints_3d, list(self.angle_names))
        angle_matrix = np.column_stack([angles[name] for name in self.angle_names])
        metadata = dict(self.metadata)
        metadata["bone_statistics_cm"] = compute_bone_statistics(self.keypoints_3d)
        arrays: dict[str, Any] = {
            "schema_version": np.asarray("research_candidate_v1"),
            "candidate_name": np.asarray(self.candidate_name),
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "keypoints_3d": np.asarray(self.keypoints_3d, dtype=np.float64),
            "keypoints": np.asarray(self.keypoints_3d, dtype=np.float64),
            "angle_names": np.asarray(self.angle_names),
            "angles": angle_matrix,
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        optional = {
            "confidence_2d": self.confidence_2d,
            "epipolar_error_px": self.epipolar_error_px,
            "reprojection_error_px": self.reprojection_error_px,
        }
        arrays.update({name: np.asarray(value) for name, value in optional.items() if value is not None})
        for name, value in self.stage_time_ms.items():
            arrays[f"time_{name}_ms"] = np.asarray(value, dtype=np.float64)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path


def adapt_skt_npz(source: Path, destination: Path, candidate_name: str = "YOLOv8m-SKT") -> Path:
    """Adapt an existing deterministic SKT result to the canonical candidate schema."""
    with np.load(source, allow_pickle=False) as payload:
        keypoints = np.asarray(payload["keypoints"], dtype=np.float64)
        conf_left = np.asarray(payload["conf_left"], dtype=np.float64)
        conf_right = np.asarray(payload["conf_right"], dtype=np.float64)
        confidence = np.minimum(conf_left, conf_right)
        stage_times = {
            key.removesuffix("_ms"): payload[key]
            for key in payload.files
            if key.endswith("_time_ms") or key in {"sequence_postprocess_ms"}
        }
        result = CandidateResult(
            candidate_name=candidate_name,
            timestamps=payload["timestamps"],
            keypoints_3d=keypoints,
            confidence_2d=confidence,
            epipolar_error_px=payload["epipolar_error"],
            reprojection_error_px=payload["reprojection_error"],
            stage_time_ms=stage_times,
            metadata={
                "source": str(source),
                "coordinate_unit": "cm",
                "joint_convention": "COCO-17",
                "reference_policy": "Xsens-derived reference is external comparison only",
            },
        )
    return result.save(destination)
