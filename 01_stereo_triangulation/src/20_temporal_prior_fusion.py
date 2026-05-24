"""Fuse SKT 3D poses with an external temporal pose prior.

The external prior can come from MotionBERT, VideoPose3D, or another temporal
model after conversion to a simple NPZ format:

    timestamps: (N,)
    keypoints:  (N, 17, 3)

The prior is not trusted as a metric coordinate source. For each frame, it is
first similarity-aligned to reliable SKT joints. Only low-quality SKT joints are
then blended toward the aligned prior. This preserves the stereo pipeline as the
metric anchor while allowing a mature temporal model to stabilize jitter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_SKT = Path(__file__).resolve().parents[1] / "results" / "yolo_3d_optimized.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "skt_model_fusion" / "skt_temporal_prior_fused.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skt", type=Path, default=DEFAULT_SKT, help="Existing SKT NPZ file.")
    parser.add_argument("--prior", type=Path, required=True, help="External temporal-prior NPZ file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Fused output NPZ file.")
    parser.add_argument("--min-pair-conf", type=float, default=0.30, help="Bad joint if pair confidence is below this.")
    parser.add_argument("--min-stereo-quality", type=float, default=0.20, help="Bad joint if stereo quality is below this.")
    parser.add_argument("--max-epipolar-px", type=float, default=12.0, help="Bad joint if epipolar error is above this.")
    parser.add_argument("--max-reprojection-px", type=float, default=45.0, help="Bad joint if reprojection error is above this.")
    parser.add_argument("--bad-blend", type=float, default=0.65, help="Blend weight toward aligned prior for bad joints.")
    parser.add_argument("--good-blend", type=float, default=0.0, help="Blend weight toward aligned prior for good joints.")
    parser.add_argument("--min-align-joints", type=int, default=5, help="Minimum reliable joints for per-frame alignment.")
    return parser.parse_args()


def as_float_array(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, ...], fill: float) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=np.float64)
    return np.full(shape, fill, dtype=np.float64)


def interpolate_keypoints(
    source_timestamps: np.ndarray,
    source_keypoints: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    """Interpolate a keypoint sequence onto the target timeline."""
    source_timestamps = np.asarray(source_timestamps, dtype=np.float64)
    source_keypoints = np.asarray(source_keypoints, dtype=np.float64)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)
    result = np.full((len(target_timestamps),) + source_keypoints.shape[1:], np.nan, dtype=np.float64)

    for joint_idx in range(source_keypoints.shape[1]):
        for axis in range(source_keypoints.shape[2]):
            values = source_keypoints[:, joint_idx, axis]
            valid = np.isfinite(source_timestamps) & np.isfinite(values)
            if np.count_nonzero(valid) < 2:
                continue
            result[:, joint_idx, axis] = np.interp(
                target_timestamps,
                source_timestamps[valid],
                values[valid],
                left=np.nan,
                right=np.nan,
            )
    return result


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Estimate a weighted similarity transform mapping src to dst."""
    valid = (
        np.isfinite(src).all(axis=1)
        & np.isfinite(dst).all(axis=1)
        & np.isfinite(weights)
        & (weights > 1e-6)
    )
    if np.count_nonzero(valid) < 3:
        return None

    src_v = src[valid]
    dst_v = dst[valid]
    w = weights[valid].astype(np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-9:
        return None

    src_mu = np.sum(src_v * w[:, None], axis=0) / w_sum
    dst_mu = np.sum(dst_v * w[:, None], axis=0) / w_sum
    src_c = src_v - src_mu
    dst_c = dst_v - dst_mu
    cov = (src_c * w[:, None]).T @ dst_c / w_sum

    try:
        u, singular_values, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None

    # Row-vector convention: aligned = scale * (src @ rotation) + translation.
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = u @ vt

    src_var = np.sum(w * np.sum(src_c * src_c, axis=1)) / w_sum
    if src_var <= 1e-9:
        return None
    scale = float(np.sum(singular_values) / src_var)
    translation = dst_mu - scale * (src_mu @ rotation)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    return scale * (points @ rotation) + translation


def build_bad_joint_mask(
    skt_data: np.lib.npyio.NpzFile,
    shape: tuple[int, int],
    min_pair_conf: float,
    min_stereo_quality: float,
    max_epipolar_px: float,
    max_reprojection_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    pair_conf = as_float_array(skt_data, "pair_confidence", shape, fill=np.nan)
    stereo_quality = as_float_array(skt_data, "stereo_quality", shape, fill=np.nan)
    epipolar_error = as_float_array(skt_data, "epipolar_error", shape, fill=np.nan)
    reprojection_error = as_float_array(skt_data, "reprojection_error", shape, fill=np.nan)

    bad = np.zeros(shape, dtype=bool)
    observed = np.zeros(shape, dtype=bool)
    for values in (pair_conf, stereo_quality, epipolar_error, reprojection_error):
        observed |= np.isfinite(values)

    bad |= np.isfinite(pair_conf) & (pair_conf < min_pair_conf)
    bad |= np.isfinite(stereo_quality) & (stereo_quality < min_stereo_quality)
    bad |= np.isfinite(epipolar_error) & (epipolar_error > max_epipolar_px)
    bad |= np.isfinite(reprojection_error) & (reprojection_error > max_reprojection_px)
    return bad, observed


def align_prior_to_skt(
    skt_keypoints: np.ndarray,
    prior_keypoints: np.ndarray,
    bad_joint_mask: np.ndarray,
    quality_observed: np.ndarray,
    min_align_joints: int,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.full_like(prior_keypoints, np.nan, dtype=np.float64)
    alignment_support = np.zeros(len(skt_keypoints), dtype=np.int64)
    last_transform = None

    for frame_idx, (skt_pose, prior_pose) in enumerate(zip(skt_keypoints, prior_keypoints)):
        reliable = (
            np.isfinite(skt_pose).all(axis=1)
            & np.isfinite(prior_pose).all(axis=1)
            & quality_observed[frame_idx]
            & ~bad_joint_mask[frame_idx]
        )
        weights = reliable.astype(np.float64)
        transform = estimate_similarity_transform(prior_pose, skt_pose, weights)
        if transform is None and np.count_nonzero(np.isfinite(skt_pose).all(axis=1) & np.isfinite(prior_pose).all(axis=1)) >= min_align_joints:
            fallback = np.isfinite(skt_pose).all(axis=1) & np.isfinite(prior_pose).all(axis=1)
            transform = estimate_similarity_transform(prior_pose, skt_pose, fallback.astype(np.float64))
        if transform is None:
            transform = last_transform
        if transform is None:
            continue
        aligned[frame_idx] = apply_similarity(prior_pose, transform)
        alignment_support[frame_idx] = int(np.count_nonzero(reliable))
        last_transform = transform

    return aligned, alignment_support


def fuse_keypoints(
    skt_keypoints: np.ndarray,
    aligned_prior: np.ndarray,
    bad_joint_mask: np.ndarray,
    bad_blend: float,
    good_blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    fused = skt_keypoints.copy()
    blend_weight = np.full(skt_keypoints.shape[:2], np.nan, dtype=np.float64)

    for frame_idx in range(skt_keypoints.shape[0]):
        for joint_idx in range(skt_keypoints.shape[1]):
            skt_valid = np.isfinite(skt_keypoints[frame_idx, joint_idx]).all()
            prior_valid = np.isfinite(aligned_prior[frame_idx, joint_idx]).all()
            if not prior_valid:
                continue
            if not skt_valid:
                fused[frame_idx, joint_idx] = aligned_prior[frame_idx, joint_idx]
                blend_weight[frame_idx, joint_idx] = 1.0
                continue
            weight = bad_blend if bad_joint_mask[frame_idx, joint_idx] else good_blend
            weight = float(np.clip(weight, 0.0, 1.0))
            if weight <= 0.0:
                blend_weight[frame_idx, joint_idx] = 0.0
                continue
            fused[frame_idx, joint_idx] = (
                (1.0 - weight) * skt_keypoints[frame_idx, joint_idx]
                + weight * aligned_prior[frame_idx, joint_idx]
            )
            blend_weight[frame_idx, joint_idx] = weight
    return fused, blend_weight


def main() -> None:
    args = parse_args()
    if not args.skt.exists():
        raise FileNotFoundError(f"SKT file not found: {args.skt}")
    if not args.prior.exists():
        raise FileNotFoundError(f"Prior file not found: {args.prior}")

    skt_data = np.load(args.skt, allow_pickle=True)
    prior_data = np.load(args.prior, allow_pickle=True)
    skt_keypoints = np.asarray(skt_data["keypoints"], dtype=np.float64)
    skt_timestamps = np.asarray(skt_data["timestamps"], dtype=np.float64)
    prior_keypoints = np.asarray(prior_data["keypoints"], dtype=np.float64)
    prior_timestamps = np.asarray(prior_data["timestamps"], dtype=np.float64)

    prior_on_skt_timeline = interpolate_keypoints(prior_timestamps, prior_keypoints, skt_timestamps)
    bad_joint_mask, quality_observed = build_bad_joint_mask(
        skt_data,
        skt_keypoints.shape[:2],
        min_pair_conf=args.min_pair_conf,
        min_stereo_quality=args.min_stereo_quality,
        max_epipolar_px=args.max_epipolar_px,
        max_reprojection_px=args.max_reprojection_px,
    )
    aligned_prior, alignment_support = align_prior_to_skt(
        skt_keypoints,
        prior_on_skt_timeline,
        bad_joint_mask,
        quality_observed,
        min_align_joints=args.min_align_joints,
    )
    fused_keypoints, blend_weight = fuse_keypoints(
        skt_keypoints,
        aligned_prior,
        bad_joint_mask,
        bad_blend=args.bad_blend,
        good_blend=args.good_blend,
    )

    payload = {key: skt_data[key] for key in skt_data.files}
    payload["keypoints"] = fused_keypoints
    payload["keypoints_skt_before_temporal_prior"] = skt_keypoints
    payload["temporal_prior_keypoints_interpolated"] = prior_on_skt_timeline
    payload["temporal_prior_keypoints_aligned"] = aligned_prior
    payload["temporal_prior_bad_joint_mask"] = bad_joint_mask
    payload["temporal_prior_blend_weight"] = blend_weight
    payload["temporal_prior_alignment_support"] = alignment_support
    payload["temporal_prior_source"] = np.array(str(args.prior))
    payload["postprocess_variant"] = np.array("skt_plus_similarity_aligned_temporal_prior")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)

    finite_blends = blend_weight[np.isfinite(blend_weight)]
    repaired = int(np.count_nonzero(finite_blends > 0.0))
    print(f"[Info] Wrote fused SKT + temporal prior file: {args.output}")
    print(f"[Info] Repaired / blended joints: {repaired}")
    print(f"[Info] Median alignment support: {np.median(alignment_support):.1f} joints")


if __name__ == "__main__":
    main()
