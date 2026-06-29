"""Position-domain postprocessing for stereo 3D pose sequences.

The helpers in this module avoid hard rejection where possible. Suspicious
observations are assigned lower measurement weights, then corrected with soft
bone-length priors and temporal Kalman/RTS smoothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


LIMB_CHAINS = [
    ("left_arm", 5, 7, 9, "left_upper_arm", "left_lower_arm"),
    ("right_arm", 6, 8, 10, "right_upper_arm", "right_lower_arm"),
    ("left_leg", 11, 13, 15, "left_thigh", "left_shank"),
    ("right_leg", 12, 14, 16, "right_thigh", "right_shank"),
]

DEFAULT_PRIORS_CM = {
    "left_upper_arm": 28.0,
    "right_upper_arm": 28.0,
    "left_lower_arm": 26.0,
    "right_lower_arm": 26.0,
    "left_thigh": 39.5,
    "right_thigh": 39.5,
    "left_shank": 40.5,
    "right_shank": 40.5,
}


@dataclass(frozen=True)
class PositionFlags:
    """Per-joint position quality flags and resulting measurement weights."""

    measurement_weight: np.ndarray
    velocity_flag: np.ndarray
    bone_flag: np.ndarray
    quality_flag: np.ndarray
    speed_cm_s: np.ndarray
    bone_priors_cm: dict[str, float]
    stats: dict[str, float | int]


@dataclass(frozen=True)
class AdaptiveLambda:
    """Per-frame bone-prior strength derived from stereo depth uncertainty."""

    values: np.ndarray
    sigma_z_cm: np.ndarray
    depth_cm: np.ndarray
    stats: dict[str, float | int]


def contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) ranges for contiguous True values."""
    ranges: list[tuple[int, int]] = []
    idx = 0
    mask = np.asarray(mask, dtype=bool)
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        ranges.append((start, idx))
    return ranges


def robust_median_distance(
    keypoints: np.ndarray,
    idx_a: int,
    idx_b: int,
    trim_percentile: float = 25.0,
) -> float:
    """Return a robust median distance between two joints."""
    distances = np.linalg.norm(keypoints[:, idx_a, :] - keypoints[:, idx_b, :], axis=1)
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return math.nan
    lo = np.percentile(finite, trim_percentile)
    hi = np.percentile(finite, 100.0 - trim_percentile)
    trimmed = finite[(finite >= lo) & (finite <= hi)]
    if len(trimmed) == 0:
        return math.nan
    return float(np.median(trimmed))


def estimate_limb_priors(
    keypoints: np.ndarray,
    trim_percentile: float = 25.0,
) -> dict[str, float]:
    """Estimate session-specific limb-length priors from observed 3D poses."""
    priors = dict(DEFAULT_PRIORS_CM)
    for _, prox, mid, dist, upper_name, lower_name in LIMB_CHAINS:
        upper = robust_median_distance(keypoints, prox, mid, trim_percentile)
        lower = robust_median_distance(keypoints, mid, dist, trim_percentile)
        if np.isfinite(upper):
            priors[upper_name] = upper
        if np.isfinite(lower):
            priors[lower_name] = lower
    return priors


def _safe_payload_array(payload, name: str, n_frames: int) -> np.ndarray | None:
    """Return a numeric payload array truncated to n_frames, if available."""
    if payload is None or name not in payload.files:
        return None
    return np.asarray(payload[name], dtype=np.float64)[:n_frames]


def _robust_velocity_threshold(speed: np.ndarray, absolute_threshold_cm_s: float, mad_factor: float) -> np.ndarray:
    """Estimate one robust velocity threshold per joint."""
    thresholds = np.full(speed.shape[1], float(absolute_threshold_cm_s), dtype=np.float64)
    for joint_idx in range(speed.shape[1]):
        values = speed[:, joint_idx]
        finite = values[np.isfinite(values)]
        if len(finite) < 5:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        robust_sigma = 1.4826 * mad
        thresholds[joint_idx] = max(float(absolute_threshold_cm_s), median + float(mad_factor) * robust_sigma)
    return thresholds


def flag_positions(
    keypoints: np.ndarray,
    time_s: np.ndarray,
    payload=None,
    *,
    priors: dict[str, float] | None = None,
    trim_percentile: float = 25.0,
    min_weight: float = 0.03,
    velocity_threshold_cm_s: float = 300.0,
    velocity_mad_factor: float = 8.0,
    bone_rel_tolerance: float = 0.35,
    low_quality_weight: float = 0.20,
    velocity_flag_weight: float = 0.20,
    bone_flag_weight: float = 0.20,
    min_stereo_quality: float = 0.25,
    max_epipolar_px: float = 15.0,
    max_reprojection_px: float = 15.0,
) -> PositionFlags:
    """Flag suspicious observations without deleting them.

    The returned ``measurement_weight`` is in [0, 1]. Finite but suspicious
    observations retain a small positive weight so later stages can use them
    while trusting temporal and anatomical priors more strongly.
    """
    positions = np.asarray(keypoints, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    n_frames, n_joints, _ = positions.shape
    finite = np.isfinite(positions).all(axis=2)
    priors = dict(priors or estimate_limb_priors(positions, trim_percentile))

    speed = np.full((n_frames, n_joints), np.nan, dtype=np.float64)
    if n_frames >= 2:
        dt = np.diff(time_s)
        valid_dt = np.isfinite(dt) & (dt > 0)
        displacement = np.linalg.norm(np.diff(positions, axis=0), axis=2)
        step_speed = np.full_like(displacement, np.nan)
        step_speed[valid_dt] = displacement[valid_dt] / dt[valid_dt, None]
        speed[1:] = step_speed
    thresholds = _robust_velocity_threshold(speed, velocity_threshold_cm_s, velocity_mad_factor)
    velocity_flag = np.isfinite(speed) & (speed > thresholds[None, :])
    # Mark both sides of a suspicious step so the smoother does not over-trust
    # either endpoint of a sudden jump.
    if n_frames >= 2:
        velocity_flag[:-1] |= velocity_flag[1:]

    bone_flag = np.zeros((n_frames, n_joints), dtype=bool)
    for _, prox, mid, dist, upper_name, lower_name in LIMB_CHAINS:
        for bone_name, idx_a, idx_b in [(upper_name, prox, mid), (lower_name, mid, dist)]:
            prior = float(priors.get(bone_name, math.nan))
            if not np.isfinite(prior) or prior <= 0:
                continue
            distances = np.linalg.norm(positions[:, idx_a, :] - positions[:, idx_b, :], axis=1)
            rel_error = np.abs(distances - prior) / prior
            bad = np.isfinite(rel_error) & (rel_error > float(bone_rel_tolerance))
            bone_flag[bad, idx_a] = True
            bone_flag[bad, idx_b] = True

    quality_flag = np.zeros((n_frames, n_joints), dtype=bool)
    stereo_quality = _safe_payload_array(payload, "stereo_quality", n_frames)
    if stereo_quality is not None:
        quality_flag |= np.isfinite(stereo_quality) & (stereo_quality < float(min_stereo_quality))
    epipolar = _safe_payload_array(payload, "epipolar_error", n_frames)
    if epipolar is not None:
        quality_flag |= np.isfinite(epipolar) & (epipolar > float(max_epipolar_px))
    reprojection = _safe_payload_array(payload, "reprojection_error", n_frames)
    if reprojection is not None:
        quality_flag |= np.isfinite(reprojection) & (reprojection > float(max_reprojection_px))

    weight = np.ones((n_frames, n_joints), dtype=np.float64)
    if stereo_quality is not None:
        weight *= np.where(np.isfinite(stereo_quality), np.clip(stereo_quality, min_weight, 1.0), min_weight)
    weight[quality_flag] *= float(low_quality_weight)
    weight[velocity_flag] *= float(velocity_flag_weight)
    weight[bone_flag] *= float(bone_flag_weight)
    weight[~finite] = 0.0
    finite_weight = finite & (weight > 0)
    weight[finite_weight] = np.clip(weight[finite_weight], min_weight, 1.0)

    stats = {
        "finite_position_ratio": float(np.mean(finite)),
        "quality_flag_ratio": float(np.mean(quality_flag & finite)),
        "velocity_flag_ratio": float(np.mean(velocity_flag & finite)),
        "bone_flag_ratio": float(np.mean(bone_flag & finite)),
        "mean_measurement_weight": float(np.mean(weight[finite])) if np.any(finite) else 0.0,
        "min_measurement_weight": float(np.min(weight[finite])) if np.any(finite) else 0.0,
    }
    return PositionFlags(
        measurement_weight=weight,
        velocity_flag=velocity_flag,
        bone_flag=bone_flag,
        quality_flag=quality_flag,
        speed_cm_s=speed,
        bone_priors_cm=priors,
        stats=stats,
    )


def depth_adaptive_lambda(
    keypoints: np.ndarray,
    *,
    fx_px: float,
    baseline_cm: float,
    sigma_disparity_px: float = 0.5,
    lambda_base: float = 0.3,
    sigma_z_ref_cm: float = 0.55,
    exponent: float = 1.3,
    min_lambda: float = 0.15,
    max_lambda: float = 1.8,
    joint_indices: tuple[int, ...] = tuple(range(5, 17)),
) -> AdaptiveLambda:
    """Compute per-frame bone lambda from stereo depth uncertainty.

    ``lambda_base`` is the bone strength at ``sigma_z_ref_cm``. Since this code
    uses lambda as the bone residual weight, noisier depth measurements increase
    the effective lambda.
    """
    positions = np.asarray(keypoints, dtype=np.float64)
    depth = np.nanmedian(positions[:, list(joint_indices), 2], axis=1)
    finite_depth = depth[np.isfinite(depth)]
    fallback_depth = float(np.nanmedian(finite_depth)) if len(finite_depth) else 250.0
    safe_depth = np.where(np.isfinite(depth), depth, fallback_depth)
    fx_px = max(float(fx_px), 1.0)
    baseline_cm = max(float(baseline_cm), 1.0)
    sigma_z = (safe_depth ** 2 / (fx_px * baseline_cm)) * float(sigma_disparity_px)
    scale = (sigma_z / max(float(sigma_z_ref_cm), 1e-6)) ** float(exponent)
    lam = float(lambda_base) * scale
    lam = np.clip(lam, float(min_lambda), float(max_lambda))
    finite = np.isfinite(lam)
    if np.any(finite):
        values = lam[finite]
        sigma_values = sigma_z[finite]
        depth_values = safe_depth[finite]
        stats = {
            "lambda_mean": float(np.mean(values)),
            "lambda_median": float(np.median(values)),
            "lambda_p10": float(np.percentile(values, 10)),
            "lambda_p90": float(np.percentile(values, 90)),
            "lambda_min": float(np.min(values)),
            "lambda_max": float(np.max(values)),
            "lambda_at_min_count": int(np.sum(np.isclose(values, float(min_lambda)))),
            "lambda_at_max_count": int(np.sum(np.isclose(values, float(max_lambda)))),
            "sigma_z_mean_cm": float(np.mean(sigma_values)),
            "sigma_z_median_cm": float(np.median(sigma_values)),
            "sigma_z_p10_cm": float(np.percentile(sigma_values, 10)),
            "sigma_z_p90_cm": float(np.percentile(sigma_values, 90)),
            "depth_mean_cm": float(np.mean(depth_values)),
            "depth_median_cm": float(np.median(depth_values)),
            "depth_p10_cm": float(np.percentile(depth_values, 10)),
            "depth_p90_cm": float(np.percentile(depth_values, 90)),
        }
    else:
        stats = {
            "lambda_mean": math.nan,
            "lambda_median": math.nan,
            "lambda_p10": math.nan,
            "lambda_p90": math.nan,
            "lambda_min": math.nan,
            "lambda_max": math.nan,
            "lambda_at_min_count": 0,
            "lambda_at_max_count": 0,
            "sigma_z_mean_cm": math.nan,
            "sigma_z_median_cm": math.nan,
            "sigma_z_p10_cm": math.nan,
            "sigma_z_p90_cm": math.nan,
            "depth_mean_cm": math.nan,
            "depth_median_cm": math.nan,
            "depth_p10_cm": math.nan,
            "depth_p90_cm": math.nan,
        }
    return AdaptiveLambda(values=lam, sigma_z_cm=sigma_z, depth_cm=safe_depth, stats=stats)


def solve_chain_soft_constraint(
    pose: np.ndarray,
    chain: tuple[str, int, int, int, str, str],
    priors: dict[str, float],
    lam: float,
    observation_weight: np.ndarray | None,
    flagged_prior_boost: float,
) -> np.ndarray:
    """Softly constrain one limb chain for a single frame."""
    _, prox, mid, dist, upper_name, lower_name = chain
    joint_ids = [prox, mid, dist]
    initial = np.asarray(pose[joint_ids, :], dtype=np.float64)
    if not np.isfinite(initial).all():
        return pose

    upper_prior = float(priors.get(upper_name, math.nan))
    lower_prior = float(priors.get(lower_name, math.nan))
    if not (np.isfinite(upper_prior) and np.isfinite(lower_prior)):
        return pose

    if observation_weight is None:
        weights = np.ones(3, dtype=np.float64)
    else:
        weights = np.asarray(observation_weight[joint_ids], dtype=np.float64)
        weights = np.where(np.isfinite(weights), np.clip(weights, 0.03, 1.0), 0.03)
    weights[0] = min(1.0, weights[0] * 3.0)
    local_lam = float(lam) * (1.0 + float(flagged_prior_boost) * float(np.mean(1.0 - weights)))
    x0 = initial.reshape(-1)

    def residual(x: np.ndarray) -> np.ndarray:
        pts = x.reshape(3, 3)
        obs = ((pts - initial) * np.sqrt(weights)[:, None]).reshape(-1)
        upper = np.linalg.norm(pts[0] - pts[1]) - upper_prior
        lower = np.linalg.norm(pts[1] - pts[2]) - lower_prior
        return np.concatenate([obs, [local_lam * upper, local_lam * lower]])

    result = least_squares(residual, x0, method="trf", max_nfev=50)
    if not result.success or not np.isfinite(result.x).all():
        return pose
    corrected = pose.copy()
    corrected[joint_ids, :] = result.x.reshape(3, 3)
    return corrected


def soft_bone_constrain_positions(
    keypoints: np.ndarray,
    priors: dict[str, float],
    *,
    lam: float | np.ndarray = 1.0,
    measurement_weight: np.ndarray | None = None,
    flagged_prior_boost: float = 1.0,
) -> np.ndarray:
    """Apply per-frame soft limb-length constraints to all configured chains."""
    corrected = np.asarray(keypoints, dtype=np.float64).copy()
    lambda_values = np.asarray(lam, dtype=np.float64) if np.ndim(lam) else None
    for frame_idx in range(len(corrected)):
        pose = corrected[frame_idx]
        frame_weight = None if measurement_weight is None else measurement_weight[frame_idx]
        frame_lam = float(lambda_values[frame_idx]) if lambda_values is not None else float(lam)
        for chain in LIMB_CHAINS:
            pose = solve_chain_soft_constraint(
                pose=pose,
                chain=chain,
                priors=priors,
                lam=frame_lam,
                observation_weight=frame_weight,
                flagged_prior_boost=flagged_prior_boost,
            )
        corrected[frame_idx] = pose
    return corrected


def _kalman_rts_1d(
    values: np.ndarray,
    time_s: np.ndarray,
    weights: np.ndarray,
    *,
    process_accel_std_cm_s2: float,
    measurement_std_cm: float,
    min_weight: float,
) -> np.ndarray:
    """Run a constant-velocity Kalman filter plus RTS smoother on one series."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    out = np.full_like(values, np.nan, dtype=np.float64)
    if np.count_nonzero(finite) < 2:
        out[finite] = values[finite]
        return out

    n = len(values)
    first = int(np.where(finite)[0][0])
    last = int(np.where(finite)[0][-1])
    x_filt = np.zeros((n, 2), dtype=np.float64)
    p_filt = np.zeros((n, 2, 2), dtype=np.float64)
    x_pred = np.zeros((n, 2), dtype=np.float64)
    p_pred = np.zeros((n, 2, 2), dtype=np.float64)
    f_mats = np.repeat(np.eye(2)[None, :, :], n, axis=0)

    r0 = (float(measurement_std_cm) ** 2) / max(float(weights[first]), min_weight)
    x_filt[first] = np.array([values[first], 0.0], dtype=np.float64)
    p_filt[first] = np.diag([r0, 1000.0])
    x_pred[first] = x_filt[first]
    p_pred[first] = p_filt[first]

    h = np.array([[1.0, 0.0]], dtype=np.float64)
    eye = np.eye(2, dtype=np.float64)
    accel_var = float(process_accel_std_cm_s2) ** 2

    for idx in range(first + 1, last + 1):
        dt = float(time_s[idx] - time_s[idx - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0 / 12.5
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = accel_var * np.array(
            [[0.25 * dt ** 4, 0.5 * dt ** 3], [0.5 * dt ** 3, dt ** 2]],
            dtype=np.float64,
        )
        f_mats[idx] = f
        x_pred[idx] = f @ x_filt[idx - 1]
        p_pred[idx] = f @ p_filt[idx - 1] @ f.T + q
        if finite[idx]:
            r = (float(measurement_std_cm) ** 2) / max(float(weights[idx]), min_weight)
            innovation = values[idx] - float(h @ x_pred[idx])
            s = float(h @ p_pred[idx] @ h.T + r)
            if s <= 0 or not np.isfinite(s):
                x_filt[idx] = x_pred[idx]
                p_filt[idx] = p_pred[idx]
                continue
            k = (p_pred[idx] @ h.T / s).reshape(2)
            x_filt[idx] = x_pred[idx] + k * innovation
            p_filt[idx] = (eye - np.outer(k, h.reshape(2))) @ p_pred[idx]
        else:
            x_filt[idx] = x_pred[idx]
            p_filt[idx] = p_pred[idx]

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for idx in range(last - 1, first - 1, -1):
        try:
            gain = p_filt[idx] @ f_mats[idx + 1].T @ np.linalg.inv(p_pred[idx + 1])
        except np.linalg.LinAlgError:
            gain = p_filt[idx] @ f_mats[idx + 1].T @ np.linalg.pinv(p_pred[idx + 1])
        x_smooth[idx] = x_filt[idx] + gain @ (x_smooth[idx + 1] - x_pred[idx + 1])
        p_smooth[idx] = p_filt[idx] + gain @ (p_smooth[idx + 1] - p_pred[idx + 1]) @ gain.T

    out[first:last + 1] = x_smooth[first:last + 1, 0]
    return out


def kf_rts_smooth_positions(
    keypoints: np.ndarray,
    time_s: np.ndarray,
    measurement_weight: np.ndarray,
    *,
    process_accel_std_cm_s2: float = 250.0,
    measurement_std_cm: float = 2.0,
    min_weight: float = 0.03,
) -> np.ndarray:
    """Smooth 3D positions with independent constant-velocity KF/RTS models."""
    positions = np.asarray(keypoints, dtype=np.float64)
    weights = np.asarray(measurement_weight, dtype=np.float64)
    out = np.full_like(positions, np.nan, dtype=np.float64)
    for joint_idx in range(positions.shape[1]):
        joint_weights = weights[:, joint_idx]
        for axis_idx in range(positions.shape[2]):
            out[:, joint_idx, axis_idx] = _kalman_rts_1d(
                positions[:, joint_idx, axis_idx],
                time_s,
                joint_weights,
                process_accel_std_cm_s2=process_accel_std_cm_s2,
                measurement_std_cm=measurement_std_cm,
                min_weight=min_weight,
            )
    return out
