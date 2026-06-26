"""Dataset loading helpers shared by evaluation stages."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from common.config import resolve_path, section
from common.trc import load_trc_on_timeline
from stereo_loader import build_synced_timeline, truncate_timeline_to_pose


def resolve_skt_npz(config: dict, run_dir: Path) -> Path:
    """Resolve the SKT pose NPZ for downstream stages."""
    skt = section(config, "skt")
    if skt.get("use_existing_npz", False):
        path = resolve_path(skt.get("existing_npz"), must_exist=True)
        assert path is not None
        return path
    output_name = skt.get("output_npz", "skt_pose_optimized.npz")
    return run_dir / output_name


def load_skt_keypoints(config: dict, run_dir: Path) -> tuple[Path, np.ndarray, np.lib.npyio.NpzFile]:
    """Load SKT keypoints from the configured NPZ."""
    path = resolve_skt_npz(config, run_dir)
    payload = np.load(path, allow_pickle=True)
    if "keypoints" not in payload:
        raise RuntimeError(f"SKT NPZ does not contain 'keypoints': {path}")
    return path, np.asarray(payload["keypoints"], dtype=np.float64), payload


def apply_skt_quality_filter(keypoints: np.ndarray, payload: np.lib.npyio.NpzFile, config: dict) -> tuple[np.ndarray, dict[str, int]]:
    """Mask configured SKT joints whose stereo quality is too poor."""
    quality_cfg = section(config, "evaluation").get("skt_quality_filter", {})
    if not quality_cfg or not quality_cfg.get("enabled", False):
        return keypoints, {}
    files = set(payload.files)
    # Accept either legacy name (triang_conf_*) or current name (conf_*).
    left_key = "triang_conf_left" if "triang_conf_left" in files else "conf_left"
    right_key = "triang_conf_right" if "triang_conf_right" in files else "conf_right"
    required = {left_key, right_key, "epipolar_error"}
    if not required.issubset(files):
        missing = sorted(required.difference(files))
        raise RuntimeError(f"Cannot enable SKT quality filter; missing arrays: {missing}")
    filtered = np.asarray(keypoints, dtype=np.float64).copy()
    triang_left = np.asarray(payload[left_key], dtype=np.float64)[: len(filtered)]
    triang_right = np.asarray(payload[right_key], dtype=np.float64)[: len(filtered)]
    epipolar = np.asarray(payload["epipolar_error"], dtype=np.float64)[: len(filtered)]
    min_conf = float(quality_cfg.get("min_triang_conf", 0.2))
    max_epi = float(quality_cfg.get("max_epipolar_px", 10.0))
    stats: dict[str, int] = {}
    for joint_idx in [int(v) for v in quality_cfg.get("joint_indices", [5, 6, 7, 8, 9, 10])]:
        conf = np.minimum(triang_left[:, joint_idx], triang_right[:, joint_idx])
        bad = (
            ~np.isfinite(conf)
            | (conf < min_conf)
            | ~np.isfinite(epipolar[:, joint_idx])
            | (epipolar[:, joint_idx] > max_epi)
        )
        filtered[bad, joint_idx, :] = np.nan
        stats[f"joint_{joint_idx}_masked_frames"] = int(np.sum(bad))
    return filtered, stats


def apply_depth_consistency_filter(keypoints: np.ndarray, config: dict) -> tuple[np.ndarray, dict[str, int]]:
    """Mask limb joints whose 3D depth deviates anomalously from adjacent joints.

    For elbow/knee joints: checks that depth is within threshold of the midpoint
    between proximal and distal joints (shoulder-wrist, hip-ankle).
    For wrist/ankle joints: checks against the adjacent middle joint.
    Applied after apply_skt_quality_filter so already-NaN joints are excluded.
    """
    eval_cfg = section(config, "evaluation")
    depth_cfg = eval_cfg.get("depth_consistency_filter", {})
    if not depth_cfg or not depth_cfg.get("enabled", False):
        return keypoints, {}
    max_dev = float(depth_cfg.get("max_depth_deviation_cm", 15.0))
    filtered = np.asarray(keypoints, dtype=np.float64).copy()
    stats: dict[str, int] = {}
    # (proximal, middle, distal) triplets — COCO-17 indices.
    # Default to arms only; leg joints have large natural depth swings during locomotion.
    default_triplets = [[5, 7, 9], [6, 8, 10]]
    raw_triplets = depth_cfg.get("limb_triplets", default_triplets)
    limb_triplets = [tuple(int(x) for x in t) for t in raw_triplets]
    for prox, mid, dist in limb_triplets:
        zp = filtered[:, prox, 2]
        zm = filtered[:, mid, 2]
        zd = filtered[:, dist, 2]
        # Middle joint: depth should be near midpoint of proximal and distal
        both_valid = np.isfinite(zp) & np.isfinite(zd) & np.isfinite(zm)
        bad_mid = both_valid & (np.abs(zm - 0.5 * (zp + zd)) > max_dev)
        filtered[bad_mid, mid, :] = np.nan
        stats[f"joint_{mid}_depth_masked"] = int(np.sum(bad_mid))
        # Distal joint: depth should not deviate too far from middle joint
        # Re-read zm after potential masking above
        zm2 = filtered[:, mid, 2]
        valid_pair = np.isfinite(zm2) & np.isfinite(zd)
        bad_dist = valid_pair & (np.abs(zd - zm2) > max_dev)
        filtered[bad_dist, dist, :] = np.nan
        stats[f"joint_{dist}_depth_masked"] = int(np.sum(bad_dist))
    return filtered, stats


def build_pose_timeline(config: dict, n_pose_frames: int):
    """Build the shared synced-video timeline truncated to pose length."""
    dataset = section(config, "dataset")
    time_s, synced, left_rows, right_rows = build_synced_timeline(
        resolve_path(dataset.get("left_metadata"), must_exist=True),
        resolve_path(dataset.get("right_metadata"), must_exist=True),
        dataset.get("timestamp_format", "seconds_microseconds_columns"),
    )
    time_s, synced = truncate_timeline_to_pose(time_s, synced, n_pose_frames)
    return time_s, synced, left_rows, right_rows


def load_method_keypoints(config: dict, run_dir: Path) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Load SKT and optional TRC methods on a shared video timeline."""
    _, skt_keypoints, skt_payload = load_skt_keypoints(config, run_dir)
    time_s, synced, left_rows, _ = build_pose_timeline(config, len(skt_keypoints))
    skt_keypoints = skt_keypoints[: len(time_s)]
    skt_keypoints, quality_stats = apply_skt_quality_filter(skt_keypoints, skt_payload, config)
    skt_keypoints, depth_stats = apply_depth_consistency_filter(skt_keypoints, config)
    methods: dict[str, np.ndarray] = {"SKT": skt_keypoints}
    trc_summaries = {"SKT_quality_filter": quality_stats, "SKT_depth_filter": depth_stats}
    refs = section(config, "references")
    trc_offsets = refs.get("trc_time_offsets_seconds", {}) or {}
    trc_items = [
        ("FastSAM3D", refs.get("fastsam_trc")),
        ("Merge", refs.get("merge_trc")),
    ]
    for name, raw_path in trc_items:
        path = resolve_path(raw_path, must_exist=False)
        if path is None or not path.exists():
            continue
        keypoints, summary = load_trc_on_timeline(
            name,
            path,
            time_s,
            synced,
            left_rows,
            source_time_offset_s=float(trc_offsets.get(name, 0.0)),
        )
        methods[name] = keypoints
        trc_summaries[name] = summary
    return time_s, {"synced": synced, "left_rows": left_rows, "trc_summaries": trc_summaries}, methods
