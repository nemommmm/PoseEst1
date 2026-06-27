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


def _keypoint_max_gap_frames(variant: dict, config: dict, time_s: np.ndarray) -> int:
    """Resolve keypoint-domain repair gap from either frames or seconds."""
    if "keypoint_max_gap_seconds" in variant:
        diffs = np.diff(np.asarray(time_s, dtype=np.float64))
        finite = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(finite) == 0:
            return int(section(config, "evaluation").get("max_gap_frames", 5))
        return max(1, int(round(float(variant["keypoint_max_gap_seconds"]) / float(np.median(finite)))))
    return int(variant.get("keypoint_max_gap_frames", section(config, "evaluation").get("max_gap_frames", 5)))


def smooth_keypoints_savgol(
    keypoints: np.ndarray,
    time_s: np.ndarray,
    max_gap: int,
    window: int,
    polyorder: int,
    return_fill_flags: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Smooth 3D keypoints per coordinate after bounded short-gap interpolation."""
    out = np.asarray(keypoints, dtype=np.float64).copy()
    fill_flags = np.zeros(out.shape, dtype=bool)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    polyorder = max(1, int(polyorder))
    for joint_idx in range(out.shape[1]):
        for axis_idx in range(out.shape[2]):
            series = out[:, joint_idx, axis_idx]
            filled, flags = fill_short_gaps(series, time_s, max_gap)
            fill_flags[:, joint_idx, axis_idx] = flags
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
    if return_fill_flags:
        return out, fill_flags
    return out


def apply_quality_aware_repair_filter(
    keypoints: np.ndarray,
    payload: np.lib.npyio.NpzFile,
    config: dict,
    variant: dict,
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Mask invalid observations while preserving short repairable gaps.

    Hard filtering treats every threshold violation as unusable. This helper
    uses three quality states instead:

    - valid: normal high-quality stereo observation.
    - repairable: high-confidence observation with moderate geometry mismatch.
    - invalid: low-confidence, severe mismatch, or non-finite observation.

    Repairable points are temporarily set to NaN so bounded keypoint-domain
    interpolation can reconstruct them using neighboring valid frames.
    """
    quality_cfg = section(config, "evaluation").get("skt_quality_filter", {})
    files = set(payload.files)
    left_key = "triang_conf_left" if "triang_conf_left" in files else "conf_left"
    right_key = "triang_conf_right" if "triang_conf_right" in files else "conf_right"
    required = {left_key, right_key, "epipolar_error"}
    if not required.issubset(files):
        missing = sorted(required.difference(files))
        raise RuntimeError(f"Cannot enable quality-aware repair; missing arrays: {missing}")

    filtered = np.asarray(keypoints, dtype=np.float64).copy()
    n_frames = len(filtered)
    conf_left = np.asarray(payload[left_key], dtype=np.float64)[:n_frames]
    conf_right = np.asarray(payload[right_key], dtype=np.float64)[:n_frames]
    min_pair_conf = np.minimum(conf_left, conf_right)
    epipolar = np.asarray(payload["epipolar_error"], dtype=np.float64)[:n_frames]
    if "reprojection_error" in files:
        reprojection = np.asarray(payload["reprojection_error"], dtype=np.float64)[:n_frames]
    else:
        reprojection = np.full_like(epipolar, np.nan)

    min_conf = float(variant.get("min_conf", quality_cfg.get("min_triang_conf", 0.2)))
    repair_min_conf = float(variant.get("repair_min_conf", 0.5))
    valid_epipolar_px = float(variant.get("valid_epipolar_px", quality_cfg.get("max_epipolar_px", 10.0)))
    repair_epipolar_px = float(variant.get("repair_epipolar_px", 20.0))
    use_reprojection = bool(variant.get("use_reprojection_quality", True))
    valid_reprojection_px = float(variant.get("valid_reprojection_px", quality_cfg.get("max_reprojection_px", 10.0)))
    repair_reprojection_px = float(variant.get("repair_reprojection_px", max(20.0, 2.0 * valid_reprojection_px)))
    joint_indices = [int(v) for v in variant.get("joint_indices", quality_cfg.get("joint_indices", [5, 6, 7, 8, 9, 10]))]

    stats: dict[str, int | float | bool] = {
        "quality_mode": "quality_aware_repair",
        "min_conf": min_conf,
        "repair_min_conf": repair_min_conf,
        "valid_epipolar_px": valid_epipolar_px,
        "repair_epipolar_px": repair_epipolar_px,
        "use_reprojection_quality": use_reprojection,
        "valid_reprojection_px": valid_reprojection_px,
        "repair_reprojection_px": repair_reprojection_px,
    }
    for joint_idx in joint_indices:
        raw_finite = np.isfinite(filtered[:, joint_idx, :]).all(axis=1)
        conf = min_pair_conf[:, joint_idx]
        epi = epipolar[:, joint_idx]
        reproj = reprojection[:, joint_idx]
        valid_conf = np.isfinite(conf) & (conf >= min_conf)
        repair_conf = np.isfinite(conf) & (conf >= repair_min_conf)
        valid_geometry = np.isfinite(epi) & (epi <= valid_epipolar_px)
        repair_geometry = np.isfinite(epi) & (epi <= repair_epipolar_px)
        if use_reprojection:
            valid_geometry &= np.isfinite(reproj) & (reproj <= valid_reprojection_px)
            repair_geometry &= np.isfinite(reproj) & (reproj <= repair_reprojection_px)

        valid = raw_finite & valid_conf & valid_geometry
        repairable = raw_finite & repair_conf & repair_geometry & ~valid
        invalid = raw_finite & ~(valid | repairable)
        filtered[repairable | invalid, joint_idx, :] = np.nan

        prefix = f"joint_{joint_idx}"
        stats[f"{prefix}_valid_frames"] = int(np.sum(valid))
        stats[f"{prefix}_repairable_frames"] = int(np.sum(repairable))
        stats[f"{prefix}_invalid_frames"] = int(np.sum(invalid))
        stats[f"{prefix}_missing_frames"] = int(np.sum(~raw_finite))
        stats[f"{prefix}_masked_frames"] = int(np.sum(repairable | invalid))
    return filtered, stats


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

    quality_stats: dict[str, object] = {}
    depth_stats: dict[str, int] = {}
    quality_mode = str(variant.get("quality_mode", "hard"))
    if variant.get("quality_filter", True):
        if quality_mode == "repair":
            keypoints, quality_stats = apply_quality_aware_repair_filter(keypoints, payload, variant_cfg, variant)
        elif quality_mode == "hard":
            keypoints, quality_stats = apply_skt_quality_filter(keypoints, payload, variant_cfg)
        else:
            raise ValueError(f"Unsupported quality_mode: {quality_mode}")
    if variant.get("depth_filter", True):
        keypoints, depth_stats = apply_depth_consistency_filter(keypoints, variant_cfg)

    keypoint_postprocess = str(variant.get("keypoint_postprocess", "none"))
    keypoint_max_gap = _keypoint_max_gap_frames(variant, config, time_s)
    postprocess_meta: dict[str, object] = {}
    if keypoint_postprocess == "savgol":
        smoothed = smooth_keypoints_savgol(
            keypoints,
            time_s,
            max_gap=keypoint_max_gap,
            window=int(variant.get("savgol_window", 7)),
            polyorder=int(variant.get("savgol_polyorder", 2)),
            return_fill_flags=quality_mode == "repair",
        )
        if quality_mode == "repair":
            keypoints, fill_flags = smoothed
            joint_fill_flags = np.all(fill_flags, axis=2)
            for joint_idx in [int(v) for v in variant.get("joint_indices", section(config, "evaluation").get("skt_quality_filter", {}).get("joint_indices", []))]:
                postprocess_meta[f"joint_{joint_idx}_short_gap_repaired_frames"] = int(np.sum(joint_fill_flags[:, joint_idx]))
        else:
            keypoints = smoothed
        postprocess_meta["keypoint_postprocess"] = "savgol"
        postprocess_meta["keypoint_max_gap_frames"] = keypoint_max_gap
    elif keypoint_postprocess == "right_arm_bone_savgol":
        keypoints = smooth_keypoints_savgol(
            keypoints,
            time_s,
            max_gap=keypoint_max_gap,
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
        postprocess_meta["keypoint_max_gap_frames"] = keypoint_max_gap
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
        "quality_mode": quality_mode if variant.get("quality_filter", True) else "disabled",
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
            "name": "quality_aware_repair_keypoint_savgol",
            "quality_filter": True,
            "quality_mode": "repair",
            "depth_filter": True,
            "keypoint_postprocess": "savgol",
            "angle_postprocess": "none",
            "keypoint_max_gap_seconds": 0.5,
            "repair_min_conf": 0.5,
            "repair_epipolar_px": 20.0,
            "repair_reprojection_px": 20.0,
            "use_reprojection_quality": True,
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
