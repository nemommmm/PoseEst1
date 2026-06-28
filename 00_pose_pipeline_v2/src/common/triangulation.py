"""Stereo triangulation utilities for SKT inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TriangulationConfig:
    """Configuration for weighted DLT triangulation and geometric gates."""

    enforce_epipolar_constraint: bool = False
    epipolar_soft_threshold_px: float = 10.0
    epipolar_soft_max_strength: float = 0.80
    epipolar_correction_decay_px: float = 3.0
    epipolar_base_px: float = 6.0
    epipolar_conf_gain_px: float = 18.0
    reprojection_base_px: float = 18.0
    reprojection_conf_gain_px: float = 42.0
    reprojection_max_px: float = 80.0
    min_pair_conf: float = 0.25
    min_disparity_px: float = 1.5

    @classmethod
    def from_skt_config(cls, skt_cfg: dict) -> "TriangulationConfig":
        """Build config from a pipeline ``skt`` section."""
        raw = skt_cfg.get("triangulation", {}) or {}
        return cls(
            enforce_epipolar_constraint=bool(raw.get("enforce_epipolar_constraint", raw.get("enabled", False))),
            epipolar_soft_threshold_px=float(raw.get("epipolar_soft_threshold_px", 10.0)),
            epipolar_soft_max_strength=float(raw.get("epipolar_soft_max_strength", 0.80)),
            epipolar_correction_decay_px=float(raw.get("epipolar_correction_decay_px", 3.0)),
            epipolar_base_px=float(raw.get("epipolar_base_px", 6.0)),
            epipolar_conf_gain_px=float(raw.get("epipolar_conf_gain_px", 18.0)),
            reprojection_base_px=float(raw.get("reprojection_base_px", 18.0)),
            reprojection_conf_gain_px=float(raw.get("reprojection_conf_gain_px", 42.0)),
            reprojection_max_px=float(raw.get("reprojection_max_px", skt_cfg.get("max_reprojection_px", 80.0))),
            min_pair_conf=float(raw.get("min_pair_conf", skt_cfg.get("min_pair_confidence", 0.25))),
            min_disparity_px=float(raw.get("min_disparity_px", 1.5)),
        )


@dataclass
class TemporalWindowConfig:
    """Configuration for rectified-2D temporal rescue before retriangulation."""

    enabled: bool = False
    radius: int = 3
    min_support: int = 3
    min_stereo_quality: float = 0.35
    decay_sec: float = 0.06
    conf_floor: float = 0.12
    support_conf_scale: float = 0.75

    @classmethod
    def from_skt_config(cls, skt_cfg: dict) -> "TemporalWindowConfig":
        """Build config from a pipeline ``skt`` section."""
        raw = skt_cfg.get("temporal_window", {}) or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            radius=int(raw.get("radius", 3)),
            min_support=int(raw.get("min_support", 3)),
            min_stereo_quality=float(raw.get("min_stereo_quality", 0.35)),
            decay_sec=float(raw.get("decay_sec", 0.06)),
            conf_floor=float(raw.get("conf_floor", 0.12)),
            support_conf_scale=float(raw.get("support_conf_scale", 0.75)),
        )


def rectify_points(
    points_xy: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rect_r: np.ndarray,
    rect_p: np.ndarray,
) -> np.ndarray:
    """Rectify 2D keypoints into stereo-rectified pixel coordinates."""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    rectified = np.full((len(points_xy), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points_xy).all(axis=1)
    if not np.any(valid):
        return rectified
    rectified_valid = cv2.undistortPoints(
        points_xy[valid].reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        R=rect_r,
        P=rect_p,
    )[:, 0, :]
    rectified[valid] = rectified_valid
    return rectified


def enforce_epipolar_constraint(
    rect_l: np.ndarray,
    rect_r: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    cfg: TriangulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Softly pull left/right rectified y-coordinates toward each other."""
    corrected_l = np.asarray(rect_l, dtype=np.float64).copy()
    corrected_r = np.asarray(rect_r, dtype=np.float64).copy()
    pre_error = np.full(rect_l.shape[0], np.nan, dtype=np.float64)
    post_error = np.full(rect_l.shape[0], np.nan, dtype=np.float64)
    shift_l = np.full(rect_l.shape[0], np.nan, dtype=np.float64)
    shift_r = np.full(rect_l.shape[0], np.nan, dtype=np.float64)
    eff_conf_l = np.clip(np.asarray(conf_l, dtype=np.float64), 0.0, 1.0)
    eff_conf_r = np.clip(np.asarray(conf_r, dtype=np.float64), 0.0, 1.0)

    for joint_idx in range(rect_l.shape[0]):
        left_pt = corrected_l[joint_idx]
        right_pt = corrected_r[joint_idx]
        if not (np.isfinite(left_pt).all() and np.isfinite(right_pt).all()):
            continue

        pre_error[joint_idx] = abs(float(left_pt[1] - right_pt[1]))
        post_error[joint_idx] = pre_error[joint_idx]
        shift_l[joint_idx] = 0.0
        shift_r[joint_idx] = 0.0
        if pre_error[joint_idx] > cfg.epipolar_soft_threshold_px:
            continue

        wl = max(float(eff_conf_l[joint_idx]) if np.isfinite(eff_conf_l[joint_idx]) else 0.01, 0.01)
        wr = max(float(eff_conf_r[joint_idx]) if np.isfinite(eff_conf_r[joint_idx]) else 0.01, 0.01)
        merged_y = (wl * left_pt[1] + wr * right_pt[1]) / (wl + wr)
        correction_ratio = 1.0 - (pre_error[joint_idx] / max(cfg.epipolar_soft_threshold_px, 1e-6))
        alpha = cfg.epipolar_soft_max_strength * max(correction_ratio, 0.0)
        corrected_l[joint_idx, 1] = left_pt[1] + alpha * (merged_y - left_pt[1])
        corrected_r[joint_idx, 1] = right_pt[1] + alpha * (merged_y - right_pt[1])
        shift_l[joint_idx] = abs(float(corrected_l[joint_idx, 1] - left_pt[1]))
        shift_r[joint_idx] = abs(float(corrected_r[joint_idx, 1] - right_pt[1]))
        post_error[joint_idx] = abs(float(corrected_l[joint_idx, 1] - corrected_r[joint_idx, 1]))

        if cfg.epipolar_correction_decay_px > 0.0:
            eff_conf_l[joint_idx] *= math.exp(-shift_l[joint_idx] / cfg.epipolar_correction_decay_px)
            eff_conf_r[joint_idx] *= math.exp(-shift_r[joint_idx] / cfg.epipolar_correction_decay_px)

    return corrected_l, corrected_r, pre_error, post_error, shift_l, shift_r, eff_conf_l, eff_conf_r


def weighted_dlt_triangulate(P1: np.ndarray, P2: np.ndarray, pt1: np.ndarray, pt2: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """Triangulate one point using confidence-weighted DLT."""
    w1 = math.sqrt(max(float(w1), 1e-4))
    w2 = math.sqrt(max(float(w2), 1e-4))
    A = np.vstack(
        [
            w1 * (pt1[0] * P1[2] - P1[0]),
            w1 * (pt1[1] * P1[2] - P1[1]),
            w2 * (pt2[0] * P2[2] - P2[0]),
            w2 * (pt2[1] * P2[2] - P2[1]),
        ]
    )
    _, _, vt = np.linalg.svd(A)
    homog = vt[-1]
    if abs(float(homog[3])) < 1e-8:
        return np.full(3, np.nan, dtype=np.float64)
    return homog[:3] / homog[3]


def project_point(P: np.ndarray, pt3d: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a 3D point with a 3x4 camera matrix."""
    homog = np.append(pt3d, 1.0)
    proj = P @ homog
    if abs(float(proj[2])) < 1e-8:
        return np.full(2, np.nan, dtype=np.float64), float("nan")
    return proj[:2] / proj[2], float(proj[2])


def compute_joint_quality(
    rect_l: np.ndarray,
    rect_r: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    cfg: TriangulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo-pair quality metrics before triangulation."""
    conf_l = np.clip(np.asarray(conf_l, dtype=np.float64), 0.0, 1.0)
    conf_r = np.clip(np.asarray(conf_r, dtype=np.float64), 0.0, 1.0)
    pair_conf = np.sqrt(conf_l * conf_r)
    epipolar_error = np.abs(rect_l[:, 1] - rect_r[:, 1])
    disparity = np.abs(rect_l[:, 0] - rect_r[:, 0])
    epipolar_scale = cfg.epipolar_base_px + cfg.epipolar_conf_gain_px * (1.0 - pair_conf)
    stereo_quality = pair_conf * np.exp(-epipolar_error / np.maximum(epipolar_scale, 1e-6))
    valid = (
        np.isfinite(rect_l).all(axis=1)
        & np.isfinite(rect_r).all(axis=1)
        & np.isfinite(pair_conf)
        & (pair_conf >= cfg.min_pair_conf)
        & np.isfinite(disparity)
        & (disparity >= cfg.min_disparity_px)
    )
    stereo_quality[~np.isfinite(stereo_quality)] = np.nan
    return pair_conf, epipolar_error, disparity, stereo_quality, valid


def triangulate_pose(
    P1: np.ndarray,
    P2: np.ndarray,
    rect_l: np.ndarray,
    rect_r: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    cfg: TriangulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate one COCO-17 pose using weighted DLT and adaptive gates."""
    num_joints = rect_l.shape[0]
    pose_3d = np.full((num_joints, 3), np.nan, dtype=np.float64)
    reprojection_error = np.full(num_joints, np.nan, dtype=np.float64)
    pair_conf, epipolar_error, disparity, stereo_quality, valid = compute_joint_quality(
        rect_l, rect_r, conf_l, conf_r, cfg
    )

    for joint_idx in np.where(valid)[0]:
        pt3d = weighted_dlt_triangulate(
            P1,
            P2,
            rect_l[joint_idx],
            rect_r[joint_idx],
            conf_l[joint_idx],
            conf_r[joint_idx],
        )
        if not np.isfinite(pt3d).all():
            continue

        proj_l, depth_l = project_point(P1, pt3d)
        proj_r, depth_r = project_point(P2, pt3d)
        if not np.isfinite(proj_l).all() or not np.isfinite(proj_r).all():
            continue
        if depth_l <= 0.0 or depth_r <= 0.0:
            continue

        err_l = np.linalg.norm(proj_l - rect_l[joint_idx])
        err_r = np.linalg.norm(proj_r - rect_r[joint_idx])
        mean_err = 0.5 * (err_l + err_r)
        reproj_threshold = min(
            cfg.reprojection_max_px,
            cfg.reprojection_base_px
            + cfg.reprojection_conf_gain_px * (1.0 - pair_conf[joint_idx])
            + 0.35 * min(epipolar_error[joint_idx], 120.0),
        )
        if mean_err > reproj_threshold:
            continue

        pose_3d[joint_idx] = pt3d
        reprojection_error[joint_idx] = mean_err

    return pose_3d, reprojection_error, pair_conf, epipolar_error, disparity, stereo_quality


def temporal_window_point_estimate(
    points_seq: np.ndarray,
    conf_seq: np.ndarray,
    timestamps: np.ndarray,
    frame_idx: int,
    joint_idx: int,
    cfg: TemporalWindowConfig,
) -> tuple[np.ndarray, float]:
    """Estimate one rectified 2D point from neighboring frames."""
    start = max(0, frame_idx - cfg.radius)
    end = min(len(points_seq), frame_idx + cfg.radius + 1)
    support_idx = np.arange(start, end, dtype=np.int64)
    support_idx = support_idx[support_idx != frame_idx]
    if support_idx.size == 0:
        return np.full(2, np.nan, dtype=np.float64), float("nan")

    support_points = points_seq[support_idx, joint_idx]
    support_conf = conf_seq[support_idx, joint_idx]
    valid = (
        np.isfinite(support_points).all(axis=1)
        & np.isfinite(support_conf)
        & (support_conf >= cfg.conf_floor)
    )
    if np.count_nonzero(valid) < cfg.min_support:
        return np.full(2, np.nan, dtype=np.float64), float("nan")

    support_idx = support_idx[valid]
    support_points = support_points[valid]
    support_conf = support_conf[valid]
    dt = np.abs(timestamps[support_idx] - timestamps[frame_idx])
    temporal_weight = np.exp(-dt / max(cfg.decay_sec, 1e-6))
    weights = support_conf * temporal_weight
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 1e-6:
        return np.full(2, np.nan, dtype=np.float64), float("nan")

    estimate = np.sum(weights[:, None] * support_points, axis=0) / weight_sum
    support_strength = float(np.sum(weights * support_conf) / weight_sum)
    return estimate, support_strength


def temporal_window_rescue_rectified(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    conf_left: np.ndarray,
    conf_right: np.ndarray,
    timestamps: np.ndarray,
    pose_3d: np.ndarray,
    stereo_quality: np.ndarray,
    cfg: TemporalWindowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repair low-quality rectified observations using neighboring 2D support."""
    rescued_left = np.asarray(rect_left, dtype=np.float64).copy()
    rescued_right = np.asarray(rect_right, dtype=np.float64).copy()
    rescued_conf_left = np.clip(np.asarray(conf_left, dtype=np.float64).copy(), 0.0, 1.0)
    rescued_conf_right = np.clip(np.asarray(conf_right, dtype=np.float64).copy(), 0.0, 1.0)
    rescue_mask_left = np.zeros(rect_left.shape[:2], dtype=bool)
    rescue_mask_right = np.zeros(rect_right.shape[:2], dtype=bool)

    num_frames, num_joints, _ = rect_left.shape
    for frame_idx in range(num_frames):
        for joint_idx in range(num_joints):
            joint_valid = np.isfinite(pose_3d[frame_idx, joint_idx]).all()
            joint_quality = stereo_quality[frame_idx, joint_idx] if np.isfinite(stereo_quality[frame_idx, joint_idx]) else 0.0
            if joint_valid and joint_quality >= cfg.min_stereo_quality:
                continue

            left_estimate, left_support = temporal_window_point_estimate(
                rect_left,
                conf_left,
                timestamps,
                frame_idx,
                joint_idx,
                cfg,
            )
            right_estimate, right_support = temporal_window_point_estimate(
                rect_right,
                conf_right,
                timestamps,
                frame_idx,
                joint_idx,
                cfg,
            )
            if not (np.isfinite(left_estimate).all() and np.isfinite(right_estimate).all()):
                continue

            blend = np.clip(joint_quality / max(cfg.min_stereo_quality, 1e-6), 0.0, 1.0)
            current_left = rect_left[frame_idx, joint_idx]
            if np.isfinite(current_left).all():
                rescued_left[frame_idx, joint_idx] = blend * current_left + (1.0 - blend) * left_estimate
            else:
                rescued_left[frame_idx, joint_idx] = left_estimate
            rescue_mask_left[frame_idx, joint_idx] = True

            current_right = rect_right[frame_idx, joint_idx]
            if np.isfinite(current_right).all():
                rescued_right[frame_idx, joint_idx] = blend * current_right + (1.0 - blend) * right_estimate
            else:
                rescued_right[frame_idx, joint_idx] = right_estimate
            rescue_mask_right[frame_idx, joint_idx] = True

            current_conf_left = conf_left[frame_idx, joint_idx] if np.isfinite(conf_left[frame_idx, joint_idx]) else 0.0
            current_conf_right = conf_right[frame_idx, joint_idx] if np.isfinite(conf_right[frame_idx, joint_idx]) else 0.0
            rescued_conf_left[frame_idx, joint_idx] = max(
                blend * current_conf_left,
                (1.0 - blend) * left_support * cfg.support_conf_scale,
            )
            rescued_conf_right[frame_idx, joint_idx] = max(
                blend * current_conf_right,
                (1.0 - blend) * right_support * cfg.support_conf_scale,
            )

    return rescued_left, rescued_right, rescued_conf_left, rescued_conf_right, rescue_mask_left, rescue_mask_right


def retriangulate_sequence(
    P1: np.ndarray,
    P2: np.ndarray,
    rect_left_seq: np.ndarray,
    rect_right_seq: np.ndarray,
    conf_left_seq: np.ndarray,
    conf_right_seq: np.ndarray,
    cfg: TriangulationConfig,
) -> dict[str, np.ndarray]:
    """Triangulate a full rectified stereo keypoint sequence."""
    num_frames, num_joints, _ = rect_left_seq.shape
    keypoints_3d = np.full((num_frames, num_joints, 3), np.nan, dtype=np.float64)
    reprojection_error = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    pair_confidence = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    epipolar_error_pre = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    epipolar_error_post = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    disparity_px = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    stereo_quality = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    rect_left_final = rect_left_seq.copy()
    rect_right_final = rect_right_seq.copy()
    triang_conf_left = np.clip(conf_left_seq.copy(), 0.0, 1.0)
    triang_conf_right = np.clip(conf_right_seq.copy(), 0.0, 1.0)
    epipolar_shift_left = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    epipolar_shift_right = np.full((num_frames, num_joints), np.nan, dtype=np.float64)

    for frame_idx in range(num_frames):
        rect_l = rect_left_seq[frame_idx].copy()
        rect_r = rect_right_seq[frame_idx].copy()
        conf_l = np.clip(conf_left_seq[frame_idx], 0.0, 1.0)
        conf_r = np.clip(conf_right_seq[frame_idx], 0.0, 1.0)

        if cfg.enforce_epipolar_constraint:
            (
                rect_l,
                rect_r,
                epi_pre,
                epi_post,
                shift_l,
                shift_r,
                conf_l,
                conf_r,
            ) = enforce_epipolar_constraint(rect_l, rect_r, conf_l, conf_r, cfg)
        else:
            epi_pre = np.abs(rect_l[:, 1] - rect_r[:, 1])
            epi_post = epi_pre.copy()
            shift_l = np.zeros(num_joints, dtype=np.float64)
            shift_r = np.zeros(num_joints, dtype=np.float64)

        pose_3d, reproj_error, pair_conf, _, disparity, quality = triangulate_pose(
            P1,
            P2,
            rect_l,
            rect_r,
            conf_l,
            conf_r,
            cfg,
        )
        keypoints_3d[frame_idx] = pose_3d
        reprojection_error[frame_idx] = reproj_error
        pair_confidence[frame_idx] = pair_conf
        epipolar_error_pre[frame_idx] = epi_pre
        epipolar_error_post[frame_idx] = epi_post
        disparity_px[frame_idx] = disparity
        stereo_quality[frame_idx] = quality
        rect_left_final[frame_idx] = rect_l
        rect_right_final[frame_idx] = rect_r
        triang_conf_left[frame_idx] = conf_l
        triang_conf_right[frame_idx] = conf_r
        epipolar_shift_left[frame_idx] = shift_l
        epipolar_shift_right[frame_idx] = shift_r

    return {
        "keypoints": keypoints_3d,
        "reprojection_error": reprojection_error,
        "pair_confidence": pair_confidence,
        "epipolar_error_pre": epipolar_error_pre,
        "epipolar_error": epipolar_error_post,
        "disparity_px": disparity_px,
        "stereo_quality": stereo_quality,
        "keypoints_left_rect": rect_left_final,
        "keypoints_right_rect": rect_right_final,
        "triang_conf_left": triang_conf_left,
        "triang_conf_right": triang_conf_right,
        "epipolar_shift_left_px": epipolar_shift_left,
        "epipolar_shift_right_px": epipolar_shift_right,
    }
