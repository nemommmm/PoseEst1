"""Calibrated-stereo utilities and optimization for an EasyMocap SMPL prior."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


BODY25_TO_COCO17 = np.asarray(
    [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11],
    dtype=np.int64,
)


@dataclass(frozen=True)
class SmplFitConfig:
    """Fixed geometry-only configuration for the SMPL feasibility gate."""

    global_iterations: int = 80
    joint_iterations: int = 220
    frozen_shape_iterations: int = 100
    learning_rate: float = 0.025
    reprojection_weight: float = 1.0
    anchor_weight: float = 0.25
    pose_prior_weight: float = 0.002
    shape_weight: float = 0.01
    temporal_weight: float = 0.05
    max_reprojection_p95_px: float = 10.0
    max_high_quality_correction_cm: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SmplFitConfig":
        """Construct the fixed configuration from a YAML section."""
        raw = raw or {}
        values = {
            field: raw.get(field, getattr(cls(), field))
            for field in cls.__dataclass_fields__
        }
        return cls(**values)


@dataclass
class SmplFitResult:
    """Optimized SMPL outputs and geometry-only diagnostics."""

    keypoints_3d_cm: np.ndarray
    poses: np.ndarray
    betas: np.ndarray
    translation_m: np.ndarray
    joint_quality: np.ndarray
    prior_weight: np.ndarray
    reprojection_error_px: np.ndarray
    stage_time_ms: dict[str, float]


def body25_to_coco17(keypoints: np.ndarray) -> np.ndarray:
    """Map OpenPose BODY-25 joints emitted by EasyMocap to COCO-17."""
    points = np.asarray(keypoints)
    if points.shape[-2:] != (25, 3):
        raise ValueError("BODY-25 keypoints must end with shape (25, 3)")
    return points[..., BODY25_TO_COCO17, :]


def compute_geometry_quality(
    confidence_left: np.ndarray,
    confidence_right: np.ndarray,
    epipolar_error_px: np.ndarray,
    reprojection_error_px: np.ndarray,
) -> np.ndarray:
    """Compute the fixed observation-quality score defined by the plan."""
    conf_l = np.nan_to_num(confidence_left, nan=0.0)
    conf_r = np.nan_to_num(confidence_right, nan=0.0)
    epipolar = np.nan_to_num(epipolar_error_px, nan=120.0)
    reprojection = np.nan_to_num(reprojection_error_px, nan=120.0)
    quality = np.sqrt(np.clip(conf_l, 0.0, 1.0) * np.clip(conf_r, 0.0, 1.0))
    quality *= np.exp(-np.maximum(epipolar, 0.0) / 6.0)
    quality *= np.exp(-np.maximum(reprojection, 0.0) / 10.0)
    return np.clip(quality, 0.0, 1.0)


def select_gate_indices(num_frames: int, gate: str) -> np.ndarray:
    """Select one deterministic centered continuous gate interval."""
    limits = {"feasibility": 40, "short": 200, "full": num_frames}
    if gate not in limits:
        raise ValueError(f"unsupported gate: {gate}")
    count = min(num_frames, limits[gate])
    start = 0 if count == num_frames else (num_frames - count) // 2
    return np.arange(start, start + count, dtype=np.int64)


def project_points_numpy(points_cm: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Project batched centimeter points with a 3x4 camera matrix."""
    homogeneous = np.concatenate(
        [points_cm, np.ones((*points_cm.shape[:-1], 1))], axis=-1
    )
    projected = homogeneous @ np.asarray(projection).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[..., :2] / projected[..., 2:3]


def reprojection_errors(
    points_cm: np.ndarray,
    observed_left: np.ndarray,
    observed_right: np.ndarray,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
) -> np.ndarray:
    """Return the mean available two-view reprojection error per joint."""
    left = np.linalg.norm(
        project_points_numpy(points_cm, projection_left) - observed_left, axis=2
    )
    right = np.linalg.norm(
        project_points_numpy(points_cm, projection_right) - observed_right, axis=2
    )
    stacked = np.stack([left, right], axis=2)
    count = np.isfinite(stacked).sum(axis=2)
    output = np.full(points_cm.shape[:2], np.nan, dtype=np.float64)
    valid = count > 0
    output[valid] = np.nansum(stacked, axis=2)[valid] / count[valid]
    return output


def _weighted_huber(residual: torch.Tensor, weight: torch.Tensor, delta: float) -> torch.Tensor:
    """Compute a stable weighted Huber mean."""
    absolute = residual.abs()
    loss = torch.where(
        absolute <= delta,
        0.5 * residual.square() / delta,
        absolute - 0.5 * delta,
    )
    expanded = weight
    while expanded.ndim < loss.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(loss)
    return (loss * expanded).sum() / expanded.sum().clamp_min(1.0)


def _project_torch(points_cm: torch.Tensor, projection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Torch keypoints and return pixels plus depth."""
    homogeneous = torch.cat([points_cm, torch.ones_like(points_cm[..., :1])], dim=-1)
    projected = torch.einsum("ij,...j->...i", projection, homogeneous)
    depth = projected[..., 2]
    pixels = projected[..., :2] / depth[..., None].clamp_min(1e-4)
    return pixels, depth


def _external_imports() -> tuple[Any, Any]:
    """Import the pinned persistent EasyMocap implementation lazily."""
    for path in (
        Path("/workspace/external/easymocap_deps"),
        Path("/workspace/external/EasyMocap"),
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from easymocap.bodymodel.smpl import SMPLModel
    from easymocap.multistage.gmm import MaxMixturePrior

    return SMPLModel, MaxMixturePrior


def fit_smpl_sequence(
    raw_cm: np.ndarray,
    observed_left: np.ndarray,
    observed_right: np.ndarray,
    confidence_left: np.ndarray,
    confidence_right: np.ndarray,
    epipolar_error_px: np.ndarray,
    baseline_reprojection_px: np.ndarray,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    model_path: Path,
    regressor_path: Path,
    cfg: SmplFitConfig,
) -> SmplFitResult:
    """Fit one shared shape and per-frame SMPL pose using stereo geometry."""
    SMPLModel, MaxMixturePrior = _external_imports()
    device = torch.device("cuda")
    model = SMPLModel(
        model_path=str(model_path),
        regressor_path=str(regressor_path),
        device=device,
        NUM_SHAPES=10,
    )
    prior = MaxMixturePrior(num_gaussians=8, start=0, end=69).to(device)
    frames = len(raw_cm)
    quality_np = compute_geometry_quality(
        confidence_left,
        confidence_right,
        epipolar_error_px,
        baseline_reprojection_px,
    )
    raw = torch.as_tensor(np.nan_to_num(raw_cm, nan=0.0), dtype=torch.float32, device=device)
    raw_mask = torch.as_tensor(np.isfinite(raw_cm).all(axis=2), dtype=torch.float32, device=device)
    obs_l = torch.as_tensor(np.nan_to_num(observed_left, nan=0.0), dtype=torch.float32, device=device)
    obs_r = torch.as_tensor(np.nan_to_num(observed_right, nan=0.0), dtype=torch.float32, device=device)
    mask_l = torch.as_tensor(np.isfinite(observed_left).all(axis=2), dtype=torch.float32, device=device)
    mask_r = torch.as_tensor(np.isfinite(observed_right).all(axis=2), dtype=torch.float32, device=device)
    conf_l = torch.as_tensor(np.nan_to_num(confidence_left, nan=0.0), dtype=torch.float32, device=device)
    conf_r = torch.as_tensor(np.nan_to_num(confidence_right, nan=0.0), dtype=torch.float32, device=device)
    quality = torch.as_tensor(quality_np, dtype=torch.float32, device=device)
    p_left = torch.as_tensor(projection_left, dtype=torch.float32, device=device)
    p_right = torch.as_tensor(projection_right, dtype=torch.float32, device=device)

    poses = torch.nn.Parameter(torch.zeros((frames, 69), device=device))
    betas = torch.nn.Parameter(torch.zeros((1, 10), device=device))
    rh_initial = torch.zeros((frames, 3), device=device)
    rh_initial[:, 0] = math.pi
    rh = torch.nn.Parameter(rh_initial)
    th = torch.nn.Parameter(torch.zeros((frames, 3), device=device))
    with torch.no_grad():
        template_body25 = model(
            return_verts=False,
            return_tensor=True,
            poses=poses,
            shapes=betas,
            Rh=rh,
            Th=th,
        )
        template = template_body25[:, BODY25_TO_COCO17.tolist()]
        raw_pelvis = 0.5 * (raw[:, 11] + raw[:, 12]) / 100.0
        template_pelvis = 0.5 * (template[:, 11] + template[:, 12])
        pelvis_valid = raw_mask[:, 11] * raw_mask[:, 12]
        fallback = torch.nanmedian(
            torch.where(pelvis_valid[:, None] > 0, raw_pelvis, torch.nan), dim=0
        ).values
        raw_pelvis = torch.where(pelvis_valid[:, None] > 0, raw_pelvis, fallback)
        th.copy_(raw_pelvis - template_pelvis)

    def forward_points() -> torch.Tensor:
        body25 = model(
            return_verts=False,
            return_tensor=True,
            poses=poses,
            shapes=betas,
            Rh=rh,
            Th=th,
        )
        return body25[:, BODY25_TO_COCO17.tolist()] * 100.0

    def loss_value(points_cm: torch.Tensor, include_pose: bool) -> torch.Tensor:
        pixels_l, depth_l = _project_torch(points_cm, p_left)
        pixels_r, depth_r = _project_torch(points_cm, p_right)
        reprojection = _weighted_huber(
            pixels_l - obs_l,
            mask_l * conf_l * (depth_l > 0).float(),
            3.0,
        )
        reprojection += _weighted_huber(
            pixels_r - obs_r,
            mask_r * conf_r * (depth_r > 0).float(),
            3.0,
        )
        loss = cfg.reprojection_weight * reprojection
        loss += cfg.anchor_weight * _weighted_huber(
            points_cm - raw, raw_mask * quality, 2.0
        )
        loss += cfg.shape_weight * betas.square().mean()
        if include_pose:
            loss += cfg.pose_prior_weight * prior(poses)
            if frames >= 3:
                acceleration = points_cm[2:] - 2.0 * points_cm[1:-1] + points_cm[:-2]
                temporal_quality = 1.0 - quality[1:-1]
                loss += cfg.temporal_weight * _weighted_huber(
                    acceleration, 0.1 + 0.9 * temporal_quality, 1.0
                )
        return loss

    stage_time: dict[str, float] = {}

    def optimize(name: str, parameters: list[torch.nn.Parameter], iterations: int, include_pose: bool) -> None:
        start = time.perf_counter()
        optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            points = forward_points()
            loss = loss_value(points, include_pose)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite SMPL loss in {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=20.0)
            optimizer.step()
        torch.cuda.synchronize()
        stage_time[name] = (time.perf_counter() - start) * 1000.0

    optimize("global_alignment", [rh, th], cfg.global_iterations, False)
    optimize("joint_shape_pose", [poses, betas, rh, th], cfg.joint_iterations, True)
    frozen_betas = betas.detach().clone()
    betas.requires_grad_(False)
    optimize("frozen_shape_refinement", [poses, rh, th], cfg.frozen_shape_iterations, True)
    with torch.no_grad():
        fitted = forward_points().detach().cpu().numpy().astype(np.float64)
    stage_time["smpl_total"] = sum(stage_time.values())
    stage_time["smpl_per_frame"] = stage_time["smpl_total"] / max(frames, 1)
    reprojection = reprojection_errors(
        fitted,
        observed_left,
        observed_right,
        projection_left,
        projection_right,
    )
    return SmplFitResult(
        keypoints_3d_cm=fitted,
        poses=poses.detach().cpu().numpy().astype(np.float64),
        betas=frozen_betas.cpu().numpy().astype(np.float64),
        translation_m=th.detach().cpu().numpy().astype(np.float64),
        joint_quality=quality_np,
        prior_weight=1.0 - quality_np,
        reprojection_error_px=reprojection,
        stage_time_ms=stage_time,
    )
