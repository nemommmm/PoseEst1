"""Geometry-conditioned human-prior fitting for calibrated stereo skeletons."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as functional

from common.angles import SEMANTIC_ANGLE_NAMES, compute_angle_sequence


HUMAN_BONES = (
    ("shoulder_width", 5, 6),
    ("left_upper_arm", 5, 7),
    ("right_upper_arm", 6, 8),
    ("left_forearm", 7, 9),
    ("right_forearm", 8, 10),
    ("hip_width", 11, 12),
    ("left_torso", 5, 11),
    ("right_torso", 6, 12),
    ("left_thigh", 11, 13),
    ("right_thigh", 12, 14),
    ("left_shank", 13, 15),
    ("right_shank", 14, 16),
)

SYMMETRIC_BONES = (
    ("left_upper_arm", "right_upper_arm"),
    ("left_forearm", "right_forearm"),
    ("left_torso", "right_torso"),
    ("left_thigh", "right_thigh"),
    ("left_shank", "right_shank"),
)


@dataclass(frozen=True)
class KinematicFitConfig:
    """Configuration for lightweight calibrated-stereo kinematic fitting."""

    window_frames: int = 9
    iterations: int = 80
    learning_rate: float = 0.035
    anchor_weights: tuple[float, ...] = (0.25, 1.0)
    bone_weights: tuple[float, ...] = (1.0, 3.0)
    temporal_weights: tuple[float, ...] = (0.05, 0.2)
    high_quality_threshold: float = 0.70
    max_high_quality_correction_cm: float = 1.0
    max_reprojection_p95_px: float = 10.0
    reliable_confidence: float = 0.50
    reliable_epipolar_px: float = 3.0
    reliable_reprojection_px: float = 3.0
    min_shape_frames: int = 8
    target_shape_frames: int = 20
    device: str = "auto"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "KinematicFitConfig":
        """Build configuration from a YAML dictionary."""
        raw = raw or {}
        return cls(
            window_frames=int(raw.get("window_frames", 9)),
            iterations=int(raw.get("iterations", 80)),
            learning_rate=float(raw.get("learning_rate", 0.035)),
            anchor_weights=tuple(float(value) for value in raw.get("anchor_weights", [0.25, 1.0])),
            bone_weights=tuple(float(value) for value in raw.get("bone_weights", [1.0, 3.0])),
            temporal_weights=tuple(float(value) for value in raw.get("temporal_weights", [0.05, 0.2])),
            high_quality_threshold=float(raw.get("high_quality_threshold", 0.70)),
            max_high_quality_correction_cm=float(raw.get("max_high_quality_correction_cm", 1.0)),
            max_reprojection_p95_px=float(raw.get("max_reprojection_p95_px", 10.0)),
            reliable_confidence=float(raw.get("reliable_confidence", 0.50)),
            reliable_epipolar_px=float(raw.get("reliable_epipolar_px", 3.0)),
            reliable_reprojection_px=float(raw.get("reliable_reprojection_px", 3.0)),
            min_shape_frames=int(raw.get("min_shape_frames", 8)),
            target_shape_frames=int(raw.get("target_shape_frames", 20)),
            device=str(raw.get("device", "auto")),
        )


@dataclass
class KinematicFitResult:
    """Selected fit and geometry-only grid diagnostics."""

    keypoints_3d: np.ndarray
    joint_quality: np.ndarray
    prior_weight: np.ndarray
    reprojection_error_px: np.ndarray
    selected_weights: dict[str, float]
    bone_lengths_cm: dict[str, float]
    metrics: dict[str, Any]
    grid_results: list[dict[str, Any]]
    stage_time_ms: dict[str, float]


def compute_geometry_quality(
    confidence_left: np.ndarray,
    confidence_right: np.ndarray,
    epipolar_error_px: np.ndarray,
    reprojection_error_px: np.ndarray,
) -> np.ndarray:
    """Compute the fixed geometry-conditioned observation quality score."""
    conf_l = np.nan_to_num(np.asarray(confidence_left, dtype=np.float64), nan=0.0)
    conf_r = np.nan_to_num(np.asarray(confidence_right, dtype=np.float64), nan=0.0)
    epipolar = np.nan_to_num(np.asarray(epipolar_error_px, dtype=np.float64), nan=120.0)
    reprojection = np.nan_to_num(np.asarray(reprojection_error_px, dtype=np.float64), nan=120.0)
    quality = np.sqrt(np.clip(conf_l, 0.0, 1.0) * np.clip(conf_r, 0.0, 1.0))
    quality *= np.exp(-np.maximum(epipolar, 0.0) / 6.0)
    quality *= np.exp(-np.maximum(reprojection, 0.0) / 10.0)
    return np.clip(quality, 0.0, 1.0)


def select_gate_indices(num_frames: int, gate: str) -> np.ndarray:
    """Select one deterministic continuous interval for a validation gate."""
    limits = {"feasibility": 40, "short": 200, "full": num_frames}
    if gate not in limits:
        raise ValueError(f"unsupported gate: {gate}")
    count = min(num_frames, limits[gate])
    if gate == "full" or count == num_frames:
        return np.arange(num_frames, dtype=np.int64)
    start = max(0, (num_frames - count) // 2)
    return np.arange(start, start + count, dtype=np.int64)


def _interpolate_missing(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Temporally initialize missing 3D coordinates without changing validity metadata."""
    source = np.asarray(points, dtype=np.float64)
    output = source.copy()
    supported = np.isfinite(source).all(axis=2).any(axis=0)
    frame_index = np.arange(len(source), dtype=np.float64)
    for joint_idx in range(source.shape[1]):
        for axis in range(3):
            values = source[:, joint_idx, axis]
            valid = np.isfinite(values)
            if np.count_nonzero(valid) == 0:
                output[:, joint_idx, axis] = 0.0
            elif np.count_nonzero(valid) == 1:
                output[:, joint_idx, axis] = values[valid][0]
            else:
                output[:, joint_idx, axis] = np.interp(frame_index, frame_index[valid], values[valid])
    return output, supported


def estimate_bone_lengths(
    keypoints_3d: np.ndarray,
    confidence_left: np.ndarray,
    confidence_right: np.ndarray,
    epipolar_error_px: np.ndarray,
    reprojection_error_px: np.ndarray,
    cfg: KinematicFitConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Estimate session-specific bone lengths only from geometry-reliable frames."""
    points = np.asarray(keypoints_3d, dtype=np.float64)
    reliable = (
        (np.asarray(confidence_left) >= cfg.reliable_confidence)
        & (np.asarray(confidence_right) >= cfg.reliable_confidence)
        & (np.asarray(epipolar_error_px) <= cfg.reliable_epipolar_px)
        & (np.asarray(reprojection_error_px) <= cfg.reliable_reprojection_px)
        & np.isfinite(points).all(axis=2)
    )
    quality = compute_geometry_quality(
        confidence_left,
        confidence_right,
        epipolar_error_px,
        reprojection_error_px,
    )
    lengths: dict[str, float] = {}
    support: dict[str, int] = {}
    for name, joint_a, joint_b in HUMAN_BONES:
        distance = np.linalg.norm(points[:, joint_a] - points[:, joint_b], axis=1)
        mask = reliable[:, joint_a] & reliable[:, joint_b] & np.isfinite(distance)
        candidate_indices = np.where(mask)[0]
        if len(candidate_indices) < cfg.target_shape_frames:
            fallback = np.where(np.isfinite(distance))[0]
            pair_quality = np.minimum(quality[:, joint_a], quality[:, joint_b])
            fallback = fallback[np.argsort(pair_quality[fallback])[::-1]]
            candidate_indices = fallback[: cfg.target_shape_frames]
        if len(candidate_indices) < cfg.min_shape_frames:
            lengths[name] = math.nan
            support[name] = int(len(candidate_indices))
            continue
        values = distance[candidate_indices]
        low, high = np.percentile(values, [10.0, 90.0])
        trimmed = values[(values >= low) & (values <= high)]
        lengths[name] = float(np.median(trimmed if len(trimmed) else values))
        support[name] = int(len(candidate_indices))

    for left_name, right_name in SYMMETRIC_BONES:
        values = [lengths.get(left_name, math.nan), lengths.get(right_name, math.nan)]
        finite = [value for value in values if np.isfinite(value)]
        if finite:
            shared = float(np.mean(finite))
            lengths[left_name] = shared
            lengths[right_name] = shared
    valid_lengths = sum(np.isfinite(value) for value in lengths.values())
    if valid_lengths < 8:
        raise RuntimeError(f"insufficient reliable body shape: {valid_lengths}/{len(HUMAN_BONES)} bones")
    return lengths, {"support_frames": support, "valid_bone_count": valid_lengths}


def _torch_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic Torch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _project_torch(points: torch.Tensor, projection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project batched 3D points with a rectified 3x4 camera matrix."""
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    projected = torch.einsum("ij,...j->...i", projection, homogeneous)
    depth = projected[..., 2]
    xy = projected[..., :2] / torch.clamp(depth[..., None], min=1e-4)
    return xy, depth


def _weighted_huber(residual: torch.Tensor, weight: torch.Tensor, delta: float) -> torch.Tensor:
    """Return a stable weighted Huber mean."""
    absolute = torch.abs(residual)
    loss = torch.where(absolute <= delta, 0.5 * residual**2 / delta, absolute - 0.5 * delta)
    expanded_weight = weight
    while expanded_weight.ndim < loss.ndim:
        expanded_weight = expanded_weight.unsqueeze(-1)
    denominator = torch.clamp(expanded_weight.expand_as(loss).sum(), min=1.0)
    return (loss * expanded_weight).sum() / denominator


def _window_starts(num_frames: int, window_frames: int) -> list[tuple[int, int]]:
    """Return overlapping windows that cover the full sequence."""
    if num_frames <= window_frames:
        return [(0, num_frames)]
    stride = max(1, window_frames // 2)
    starts = list(range(0, num_frames - window_frames + 1, stride))
    last = num_frames - window_frames
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + window_frames) for start in starts]


def _fit_one_setting(
    initial_points: np.ndarray,
    raw_points: np.ndarray,
    observations_left: np.ndarray,
    observations_right: np.ndarray,
    confidence_left: np.ndarray,
    confidence_right: np.ndarray,
    quality: np.ndarray,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    bone_lengths: dict[str, float],
    cfg: KinematicFitConfig,
    anchor_weight: float,
    bone_weight: float,
    temporal_weight: float,
) -> np.ndarray:
    """Fit one geometry-only hyperparameter setting with overlapping windows."""
    device = _torch_device(cfg.device)
    dtype = torch.float32
    output_sum = np.zeros_like(initial_points, dtype=np.float64)
    output_weight = np.zeros(initial_points.shape[:2], dtype=np.float64)
    supported_joint = np.isfinite(raw_points).all(axis=2).any(axis=0)
    p_left = torch.as_tensor(projection_left, dtype=dtype, device=device)
    p_right = torch.as_tensor(projection_right, dtype=dtype, device=device)

    for start, end in _window_starts(len(initial_points), max(cfg.window_frames, 3)):
        base_np = initial_points[start:end, 5:17]
        raw_np = raw_points[start:end, 5:17]
        variable = torch.nn.Parameter(torch.as_tensor(base_np, dtype=dtype, device=device))
        raw = torch.as_tensor(np.nan_to_num(raw_np, nan=0.0), dtype=dtype, device=device)
        raw_mask = torch.as_tensor(np.isfinite(raw_np).all(axis=2), dtype=dtype, device=device)
        obs_l_np = observations_left[start:end, 5:17]
        obs_r_np = observations_right[start:end, 5:17]
        obs_l = torch.as_tensor(np.nan_to_num(obs_l_np, nan=0.0), dtype=dtype, device=device)
        obs_r = torch.as_tensor(np.nan_to_num(obs_r_np, nan=0.0), dtype=dtype, device=device)
        mask_l = torch.as_tensor(np.isfinite(obs_l_np).all(axis=2), dtype=dtype, device=device)
        mask_r = torch.as_tensor(np.isfinite(obs_r_np).all(axis=2), dtype=dtype, device=device)
        conf_l = torch.as_tensor(np.nan_to_num(confidence_left[start:end, 5:17], nan=0.0), dtype=dtype, device=device)
        conf_r = torch.as_tensor(np.nan_to_num(confidence_right[start:end, 5:17], nan=0.0), dtype=dtype, device=device)
        q = torch.as_tensor(quality[start:end, 5:17], dtype=dtype, device=device)
        optimizer = torch.optim.Adam([variable], lr=cfg.learning_rate)

        for _ in range(cfg.iterations):
            optimizer.zero_grad(set_to_none=True)
            projection_l, depth_l = _project_torch(variable, p_left)
            projection_r, depth_r = _project_torch(variable, p_right)
            valid_l = mask_l * (depth_l > 0.0).to(dtype) * conf_l
            valid_r = mask_r * (depth_r > 0.0).to(dtype) * conf_r
            loss = _weighted_huber(projection_l - obs_l, valid_l, delta=3.0)
            loss = loss + _weighted_huber(projection_r - obs_r, valid_r, delta=3.0)
            loss = loss + anchor_weight * _weighted_huber(variable - raw, raw_mask * q, delta=2.0)

            prior = 1.0 - q
            bone_loss = torch.zeros((), dtype=dtype, device=device)
            bone_terms = 0
            for name, joint_a, joint_b in HUMAN_BONES:
                expected = bone_lengths.get(name, math.nan)
                local_a, local_b = joint_a - 5, joint_b - 5
                if not np.isfinite(expected) or not (supported_joint[joint_a] and supported_joint[joint_b]):
                    continue
                distance = torch.linalg.vector_norm(variable[:, local_a] - variable[:, local_b], dim=1)
                relative = (distance - expected) / max(expected, 1e-3)
                pair_prior = 0.15 + 0.85 * 0.5 * (prior[:, local_a] + prior[:, local_b])
                bone_loss = bone_loss + _weighted_huber(relative, pair_prior, delta=0.05)
                bone_terms += 1
            if bone_terms:
                loss = loss + bone_weight * bone_loss / bone_terms

            if len(variable) >= 3:
                acceleration = variable[2:] - 2.0 * variable[1:-1] + variable[:-2]
                temporal_prior = 0.10 + 0.90 * prior[1:-1]
                loss = loss + temporal_weight * _weighted_huber(
                    acceleration,
                    temporal_prior,
                    delta=1.0,
                )
            loss.backward()
            optimizer.step()

        fitted = variable.detach().cpu().numpy().astype(np.float64)
        window_length = end - start
        weights = 1.0 + np.minimum(np.arange(window_length), np.arange(window_length)[::-1])
        output_sum[start:end, 5:17] += fitted * weights[:, None, None]
        output_weight[start:end, 5:17] += weights[:, None]

    result = np.asarray(initial_points, dtype=np.float64).copy()
    valid_weight = output_weight > 0
    result[valid_weight] = output_sum[valid_weight] / output_weight[valid_weight][:, None]
    result[:, :5] = raw_points[:, :5]
    result[:, ~supported_joint] = np.nan
    return result


def compute_reprojection_errors(
    keypoints_3d: np.ndarray,
    observations_left: np.ndarray,
    observations_right: np.ndarray,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
) -> np.ndarray:
    """Return mean two-view reprojection error for every frame and joint."""
    points = np.asarray(keypoints_3d, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=2)
    projected_left = homogeneous @ np.asarray(projection_left, dtype=np.float64).T
    projected_right = homogeneous @ np.asarray(projection_right, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        left_xy = projected_left[..., :2] / projected_left[..., 2:3]
        right_xy = projected_right[..., :2] / projected_right[..., 2:3]
    left_error = np.linalg.norm(left_xy - observations_left, axis=2)
    right_error = np.linalg.norm(right_xy - observations_right, axis=2)
    stacked = np.stack([left_error, right_error], axis=2)
    finite_count = np.isfinite(stacked).sum(axis=2)
    result = np.full(points.shape[:2], np.nan, dtype=np.float64)
    valid = finite_count > 0
    result[valid] = np.nansum(stacked, axis=2)[valid] / finite_count[valid]
    return result


def _bone_cv(keypoints: np.ndarray) -> float:
    """Return the mean coefficient of variation over supported human bones."""
    values = []
    for _, joint_a, joint_b in HUMAN_BONES:
        distances = np.linalg.norm(keypoints[:, joint_a] - keypoints[:, joint_b], axis=1)
        finite = distances[np.isfinite(distances)]
        if len(finite) >= 3 and np.mean(finite) > 1e-6:
            values.append(float(np.std(finite) / np.mean(finite)))
    return float(np.mean(values)) if values else math.inf


def _angle_jump_count(keypoints: np.ndarray, threshold_deg: float = 10.0) -> int:
    """Count finite consecutive angle changes larger than a threshold."""
    angles = compute_angle_sequence(keypoints, list(SEMANTIC_ANGLE_NAMES))
    total = 0
    for values in angles.values():
        differences = np.abs(np.diff(values))
        total += int(np.count_nonzero(np.isfinite(differences) & (differences > threshold_deg)))
    return total


def summarize_fit(
    fitted: np.ndarray,
    raw: np.ndarray,
    quality: np.ndarray,
    reprojection: np.ndarray,
    cfg: KinematicFitConfig,
) -> dict[str, Any]:
    """Build geometry-only diagnostics used for admission and grid selection."""
    correction = np.linalg.norm(fitted - raw, axis=2)
    core_correction = correction[:, 5:17]
    core_quality = quality[:, 5:17]
    core_reprojection = reprojection[:, 5:17]
    high_quality = (core_quality >= cfg.high_quality_threshold) & np.isfinite(core_correction)
    finite_reprojection = core_reprojection[np.isfinite(core_reprojection)]
    high_quality_correction = core_correction[high_quality]
    metrics = {
        "finite_ratio": float(np.isfinite(fitted[:, 5:17]).all(axis=2).mean()),
        "reprojection_p50_px": float(np.median(finite_reprojection)) if len(finite_reprojection) else math.inf,
        "reprojection_p95_px": float(np.percentile(finite_reprojection, 95.0)) if len(finite_reprojection) else math.inf,
        "bone_cv_mean": _bone_cv(fitted),
        "angle_jump_count": _angle_jump_count(fitted),
        "high_quality_correction_median_cm": (
            float(np.median(high_quality_correction)) if len(high_quality_correction) else math.inf
        ),
        "high_quality_correction_p95_cm": (
            float(np.percentile(high_quality_correction, 95.0)) if len(high_quality_correction) else math.inf
        ),
    }
    metrics["geometry_gate_pass"] = bool(
        metrics["high_quality_correction_median_cm"] <= cfg.max_high_quality_correction_cm
        and metrics["reprojection_p95_px"] <= cfg.max_reprojection_p95_px
    )
    return metrics


def _selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Return the fixed reference-free ordering for grid selection."""
    return (
        0.0 if metrics["geometry_gate_pass"] else 1.0,
        1.0 - float(metrics["finite_ratio"]),
        float(metrics["reprojection_p95_px"]),
        float(metrics["bone_cv_mean"]),
        float(metrics["angle_jump_count"]),
    )


def fit_kinematic_sequence(
    keypoints_3d: np.ndarray,
    observations_left: np.ndarray,
    observations_right: np.ndarray,
    confidence_left: np.ndarray,
    confidence_right: np.ndarray,
    epipolar_error_px: np.ndarray,
    reprojection_error_px: np.ndarray,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    cfg: KinematicFitConfig,
) -> KinematicFitResult:
    """Fit and select a lightweight human prior using geometry-only criteria."""
    total_start = time.perf_counter()
    raw = np.asarray(keypoints_3d, dtype=np.float64)
    quality = compute_geometry_quality(
        confidence_left,
        confidence_right,
        epipolar_error_px,
        reprojection_error_px,
    )
    shape_start = time.perf_counter()
    bone_lengths, shape_diagnostics = estimate_bone_lengths(
        raw,
        confidence_left,
        confidence_right,
        epipolar_error_px,
        reprojection_error_px,
        cfg,
    )
    initial, _ = _interpolate_missing(raw)
    shape_time_ms = (time.perf_counter() - shape_start) * 1000.0

    grid_results: list[dict[str, Any]] = []
    fitted_candidates: list[np.ndarray] = []
    fitting_start = time.perf_counter()
    for anchor_weight, bone_weight, temporal_weight in product(
        cfg.anchor_weights,
        cfg.bone_weights,
        cfg.temporal_weights,
    ):
        fitted = _fit_one_setting(
            initial,
            raw,
            observations_left,
            observations_right,
            confidence_left,
            confidence_right,
            quality,
            projection_left,
            projection_right,
            bone_lengths,
            cfg,
            anchor_weight,
            bone_weight,
            temporal_weight,
        )
        candidate_reprojection = compute_reprojection_errors(
            fitted,
            observations_left,
            observations_right,
            projection_left,
            projection_right,
        )
        metrics = summarize_fit(fitted, raw, quality, candidate_reprojection, cfg)
        metrics["weights"] = {
            "anchor": anchor_weight,
            "bone": bone_weight,
            "temporal": temporal_weight,
        }
        grid_results.append(metrics)
        fitted_candidates.append(fitted)
    fit_time_ms = (time.perf_counter() - fitting_start) * 1000.0

    selected_index = min(range(len(grid_results)), key=lambda index: _selection_key(grid_results[index]))
    selected = fitted_candidates[selected_index]
    selected_reprojection = compute_reprojection_errors(
        selected,
        observations_left,
        observations_right,
        projection_left,
        projection_right,
    )
    selected_metrics = dict(grid_results[selected_index])
    selected_metrics["shape_diagnostics"] = shape_diagnostics
    selected_metrics["scientific_status"] = "pass" if selected_metrics["geometry_gate_pass"] else "reject"
    total_time_ms = (time.perf_counter() - total_start) * 1000.0
    return KinematicFitResult(
        keypoints_3d=selected,
        joint_quality=quality,
        prior_weight=1.0 - quality,
        reprojection_error_px=selected_reprojection,
        selected_weights=dict(selected_metrics["weights"]),
        bone_lengths_cm=bone_lengths,
        metrics=selected_metrics,
        grid_results=grid_results,
        stage_time_ms={
            "shape_estimation": shape_time_ms,
            "grid_fitting": fit_time_ms,
            "human_prior_total": total_time_ms,
            "human_prior_per_frame": total_time_ms / max(len(raw), 1),
        },
    )
