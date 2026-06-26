"""Ablate SKT filter and angle-postprocess choices on an existing NPZ."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.angles import compute_angle_sequence, fill_short_gaps, moving_average, odd_window_from_ms
from common.config import get_run_dir, load_config, resolve_path, section
from common.dataset import (
    apply_depth_consistency_filter,
    apply_skt_quality_filter,
    build_pose_timeline,
    load_skt_keypoints,
)
from common.metrics import jsonable
from common.trc import load_trc_on_timeline
from eval_vs_fastsam import build_rows

RIGHT_ARM_CHAIN = (6, 8, 10)


def _contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) ranges of contiguous True values."""
    ranges: list[tuple[int, int]] = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        ranges.append((start, idx))
    return ranges


def _set_nested(config: dict, path: tuple[str, ...], value) -> dict:
    """Return a copied config with one nested value changed."""
    out = deepcopy(config)
    node = out
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    return out


def _smooth_angles(values: np.ndarray, time_s: np.ndarray, max_gap: int, smooth_radius: int) -> np.ndarray:
    """Apply the v1 angle-domain fill + moving-average chain."""
    filled, _ = fill_short_gaps(values, time_s, max_gap)
    smoothed = moving_average(filled, smooth_radius)
    smoothed[~np.isfinite(filled)] = np.nan
    return smoothed


def smooth_keypoints_savgol(
    keypoints: np.ndarray,
    time_s: np.ndarray,
    max_gap: int,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """Smooth 3D keypoints per coordinate after bounded short-gap interpolation."""
    out = np.asarray(keypoints, dtype=np.float64).copy()
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    polyorder = max(1, int(polyorder))
    for joint_idx in range(out.shape[1]):
        for axis_idx in range(out.shape[2]):
            series = out[:, joint_idx, axis_idx]
            filled, _ = fill_short_gaps(series, time_s, max_gap)
            finite = np.isfinite(filled)
            smoothed = np.full_like(filled, np.nan)
            for start, end in _contiguous_true_ranges(finite):
                segment = filled[start:end]
                seg_len = len(segment)
                if seg_len <= polyorder + 1:
                    smoothed[start:end] = segment
                    continue
                local_window = min(window, seg_len if seg_len % 2 == 1 else seg_len - 1)
                if local_window <= polyorder:
                    smoothed[start:end] = segment
                    continue
                smoothed[start:end] = savgol_filter(segment, window_length=local_window, polyorder=polyorder, mode="interp")
            out[:, joint_idx, axis_idx] = smoothed
    return out


def estimate_right_arm_bone_priors(keypoints: np.ndarray) -> dict[str, float]:
    """Estimate robust right-arm bone-length priors from finite frames."""
    priors: dict[str, float] = {}
    for name, pair in {"right_upper_arm": (6, 8), "right_forearm": (8, 10)}.items():
        distances = np.linalg.norm(keypoints[:, pair[0]] - keypoints[:, pair[1]], axis=1)
        finite = distances[np.isfinite(distances)]
        if len(finite) == 0:
            continue
        if len(finite) >= 10:
            lo, hi = np.percentile(finite, [20, 80])
            finite = finite[(finite >= lo) & (finite <= hi)]
        priors[name] = float(np.median(finite))
    return priors


def apply_right_arm_bone_constraint(
    keypoints: np.ndarray,
    priors: dict[str, float],
    bone_weight: float,
    max_nfev: int,
) -> np.ndarray:
    """Softly regularize RShoulder-RElbow-RWrist lengths per frame."""
    if "right_upper_arm" not in priors or "right_forearm" not in priors:
        return keypoints
    out = np.asarray(keypoints, dtype=np.float64).copy()
    upper_prior = float(priors["right_upper_arm"])
    fore_prior = float(priors["right_forearm"])
    obs_weight = 1.0
    bone_weight = float(bone_weight)
    for frame_idx in range(len(out)):
        s, e, w = out[frame_idx, RIGHT_ARM_CHAIN]
        if not (np.isfinite(s).all() and np.isfinite(e).all() and np.isfinite(w).all()):
            continue
        initial = np.concatenate([s, e, w])

        def residual(vec: np.ndarray) -> np.ndarray:
            ss = vec[0:3]
            ee = vec[3:6]
            ww = vec[6:9]
            return np.concatenate([
                obs_weight * (vec - initial),
                np.asarray([
                    bone_weight * (np.linalg.norm(ss - ee) - upper_prior),
                    bone_weight * (np.linalg.norm(ee - ww) - fore_prior),
                ], dtype=np.float64),
            ])

        result = least_squares(residual, initial, max_nfev=max_nfev)
        if result.success and np.isfinite(result.x).all():
            out[frame_idx, RIGHT_ARM_CHAIN, :] = result.x.reshape(3, 3)
    return out


def _prepare_skt_variant(
    config: dict,
    run_dir: Path,
    variant: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Load and process SKT angles for one ablation variant."""
    _, raw_keypoints, payload = load_skt_keypoints(config, run_dir)
    time_s, _, _, _ = build_pose_timeline(config, len(raw_keypoints))
    keypoints = raw_keypoints[: len(time_s)].copy()

    variant_cfg = deepcopy(config)
    if not variant.get("quality_filter", True):
        variant_cfg = _set_nested(variant_cfg, ("evaluation", "skt_quality_filter", "enabled"), False)
    if not variant.get("depth_filter", True):
        variant_cfg = _set_nested(variant_cfg, ("evaluation", "depth_consistency_filter", "enabled"), False)

    quality_stats: dict[str, int] = {}
    depth_stats: dict[str, int] = {}
    if variant.get("quality_filter", True):
        keypoints, quality_stats = apply_skt_quality_filter(keypoints, payload, variant_cfg)
    if variant.get("depth_filter", True):
        keypoints, depth_stats = apply_depth_consistency_filter(keypoints, variant_cfg)

    keypoint_postprocess = str(variant.get("keypoint_postprocess", "none"))
    postprocess_meta: dict[str, object] = {}
    if keypoint_postprocess == "savgol":
        keypoints = smooth_keypoints_savgol(
            keypoints,
            time_s,
            max_gap=int(variant.get("keypoint_max_gap_frames", section(config, "evaluation").get("max_gap_frames", 5))),
            window=int(variant.get("savgol_window", 7)),
            polyorder=int(variant.get("savgol_polyorder", 2)),
        )
        postprocess_meta["keypoint_postprocess"] = "savgol"
    elif keypoint_postprocess == "right_arm_bone_savgol":
        keypoints = smooth_keypoints_savgol(
            keypoints,
            time_s,
            max_gap=int(variant.get("keypoint_max_gap_frames", section(config, "evaluation").get("max_gap_frames", 5))),
            window=int(variant.get("savgol_window", 7)),
            polyorder=int(variant.get("savgol_polyorder", 2)),
        )
        priors = estimate_right_arm_bone_priors(keypoints)
        keypoints = apply_right_arm_bone_constraint(
            keypoints,
            priors,
            bone_weight=float(variant.get("bone_weight", 1.0)),
            max_nfev=int(variant.get("max_nfev", 25)),
        )
        postprocess_meta["keypoint_postprocess"] = "right_arm_bone_savgol"
        postprocess_meta["bone_priors_cm"] = priors
    elif keypoint_postprocess != "none":
        raise ValueError(f"Unsupported keypoint_postprocess: {keypoint_postprocess}")

    eval_cfg = section(config, "evaluation")
    angle_names = [str(name) for name in eval_cfg.get("angle_names", ["RightElbow"])]
    raw_angles = compute_angle_sequence(keypoints, angle_names)
    _, radius, actual_ms = odd_window_from_ms(time_s, float(eval_cfg.get("camera_smooth_window_ms", 200.0)))
    max_gap = int(eval_cfg.get("max_gap_frames", 5))

    postprocess = str(variant.get("angle_postprocess", "fill_moving_average"))
    processed = {}
    for name, values in raw_angles.items():
        if postprocess == "none":
            processed[name] = values
        elif postprocess == "moving_average_only":
            smoothed = moving_average(values, radius)
            smoothed[~np.isfinite(values)] = np.nan
            processed[name] = smoothed
        elif postprocess == "fill_moving_average":
            processed[name] = _smooth_angles(values, time_s, max_gap, radius)
        else:
            raise ValueError(f"Unsupported angle_postprocess: {postprocess}")

    meta = {
        "quality_stats": quality_stats,
        "depth_stats": depth_stats,
        **postprocess_meta,
        "angle_postprocess": postprocess,
        "smooth_radius_frames": radius,
        "smooth_actual_ms": actual_ms,
        "max_gap_frames": max_gap,
    }
    return time_s, processed, meta


def _load_fastsam_angles(config: dict, time_s: np.ndarray, synced, left_rows) -> dict[str, np.ndarray]:
    """Load FastSAM3D angles on the same video timeline."""
    refs = section(config, "references")
    fast_path = resolve_path(refs.get("fastsam_trc"), must_exist=True)
    trc_offsets = refs.get("trc_time_offsets_seconds", {}) or {}
    assert fast_path is not None
    fast_kp, _ = load_trc_on_timeline(
        "FastSAM3D",
        fast_path,
        time_s,
        synced,
        left_rows,
        source_time_offset_s=float(trc_offsets.get("FastSAM3D", 0.0)),
    )
    angle_names = [str(name) for name in section(config, "evaluation").get("angle_names", ["RightElbow"])]
    return compute_angle_sequence(fast_kp, angle_names)


def default_variants() -> list[dict[str, object]]:
    """Return the default Stage 2 filter-chain ablation variants."""
    return [
        {
            "name": "current_hard_filter_fill_ma",
            "quality_filter": True,
            "depth_filter": True,
            "angle_postprocess": "fill_moving_average",
        },
        {
            "name": "no_filter_fill_ma",
            "quality_filter": False,
            "depth_filter": False,
            "angle_postprocess": "fill_moving_average",
        },
        {
            "name": "hard_filter_raw_angle",
            "quality_filter": True,
            "depth_filter": True,
            "angle_postprocess": "none",
        },
        {
            "name": "no_filter_raw_angle",
            "quality_filter": False,
            "depth_filter": False,
            "angle_postprocess": "none",
        },
        {
            "name": "no_filter_ma_only",
            "quality_filter": False,
            "depth_filter": False,
            "angle_postprocess": "moving_average_only",
        },
        {
            "name": "hard_filter_keypoint_savgol",
            "quality_filter": True,
            "depth_filter": True,
            "keypoint_postprocess": "savgol",
            "angle_postprocess": "none",
        },
        {
            "name": "no_filter_keypoint_savgol",
            "quality_filter": False,
            "depth_filter": False,
            "keypoint_postprocess": "savgol",
            "angle_postprocess": "none",
        },
        {
            "name": "hard_filter_bone_savgol",
            "quality_filter": True,
            "depth_filter": True,
            "keypoint_postprocess": "right_arm_bone_savgol",
            "angle_postprocess": "none",
            "bone_weight": 1.0,
        },
    ]


def evaluate_filter_ablation(config: dict, run_dir: Path) -> Path:
    """Run filter-chain ablation against FastSAM3D."""
    _, raw_keypoints, _ = load_skt_keypoints(config, run_dir)
    time_s, synced, left_rows, _ = build_pose_timeline(config, len(raw_keypoints))
    fast_angles = _load_fastsam_angles(config, time_s, synced, left_rows)
    eval_cfg = section(config, "evaluation")
    variants = section(config, "filter_ablation").get("variants") or default_variants()
    angle_names = [str(name) for name in eval_cfg.get("angle_names", ["RightElbow"])]
    all_angles = {"FastSAM3D": fast_angles}
    variant_meta = {}
    for variant in variants:
        name = str(variant["name"])
        _, angles, meta = _prepare_skt_variant(config, run_dir, variant)
        all_angles[name] = angles
        variant_meta[name] = meta

    rows = build_rows(
        time_s=time_s,
        all_angles=all_angles,
        angle_names=angle_names,
        targets=[str(variant["name"]) for variant in variants],
        rula_bins=eval_cfg.get("rula_bins", {}),
        jump_threshold_deg=float(section(config, "filter_ablation").get("jump_threshold_deg", 10.0)),
    )

    out_dir = run_dir / "filter_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "target", "reference", "angle", "valid_pair_count", "valid_ratio",
        "overlap_start_s", "overlap_end_s", "mae_deg", "median_abs_error_deg",
        "rmse_deg", "bias_deg", "rula_like_agreement",
        "target_angular_acc_rms_deg_s2", "reference_angular_acc_rms_deg_s2",
        "target_jump_count", "reference_jump_count",
        "target_jump_count_full_timeline", "reference_jump_count_full_timeline",
        "jump_threshold_deg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config": {
            "reference": "FastSAM3D comparison trajectory",
            "variants": variants,
            "variant_meta": variant_meta,
        },
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    log_path = out_dir / "experiment_log.md"
    log_path.write_text(
        "# Filter-chain ablation\n\n"
        "This run compares hard quality filtering, no filtering, and angle-domain postprocess variants "
        "on the existing SKT NPZ. It does not rerun 2D detection or triangulation.\n",
        encoding="utf-8",
    )
    print(f"[filter_ablation] saved {csv_path}")
    return csv_path


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_filter_ablation(config, get_run_dir(config))


if __name__ == "__main__":
    main()
