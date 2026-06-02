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
    required = {"triang_conf_left", "triang_conf_right", "epipolar_error", "reprojection_error"}
    if not required.issubset(set(payload.files)):
        missing = sorted(required.difference(payload.files))
        raise RuntimeError(f"Cannot enable SKT quality filter; missing arrays: {missing}")
    filtered = np.asarray(keypoints, dtype=np.float64).copy()
    triang_left = np.asarray(payload["triang_conf_left"], dtype=np.float64)[: len(filtered)]
    triang_right = np.asarray(payload["triang_conf_right"], dtype=np.float64)[: len(filtered)]
    epipolar = np.asarray(payload["epipolar_error"], dtype=np.float64)[: len(filtered)]
    reproj = np.asarray(payload["reprojection_error"], dtype=np.float64)[: len(filtered)]
    min_conf = float(quality_cfg.get("min_triang_conf", 0.2))
    max_epi = float(quality_cfg.get("max_epipolar_px", 10.0))
    max_reproj = float(quality_cfg.get("max_reprojection_px", 10.0))
    stats: dict[str, int] = {}
    for joint_idx in [int(v) for v in quality_cfg.get("joint_indices", [5, 6, 7, 8, 9, 10])]:
        conf = np.minimum(triang_left[:, joint_idx], triang_right[:, joint_idx])
        bad = (
            ~np.isfinite(conf)
            | (conf < min_conf)
            | ~np.isfinite(epipolar[:, joint_idx])
            | (epipolar[:, joint_idx] > max_epi)
            | ~np.isfinite(reproj[:, joint_idx])
            | (reproj[:, joint_idx] > max_reproj)
        )
        filtered[bad, joint_idx, :] = np.nan
        stats[f"joint_{joint_idx}_masked_frames"] = int(np.sum(bad))
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
    methods: dict[str, np.ndarray] = {"SKT": skt_keypoints}
    trc_summaries = {"SKT_quality_filter": quality_stats}
    refs = section(config, "references")
    trc_items = [
        ("FastSAM3D", refs.get("fastsam_trc")),
        ("Merge", refs.get("merge_trc")),
    ]
    for name, raw_path in trc_items:
        path = resolve_path(raw_path, must_exist=False)
        if path is None or not path.exists():
            continue
        keypoints, summary = load_trc_on_timeline(name, path, time_s, synced, left_rows)
        methods[name] = keypoints
        trc_summaries[name] = summary
    return time_s, {"synced": synced, "left_rows": left_rows, "trc_summaries": trc_summaries}, methods
