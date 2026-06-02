"""Automatic video-to-Xsens temporal offset estimation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from common.angles import (
    SEMANTIC_ANGLE_NAMES,
    build_fair_angle_interpolators,
    build_native_angle_interpolators,
    build_xsens_coco_keypoints,
    compute_angle_sequence,
    moving_average,
    sample_interpolators,
)
from common.config import resolve_path, section
from common.dataset import load_skt_keypoints, build_pose_timeline
from common.metrics import jsonable, pearson, rmse


def kabsch_rmse(source: np.ndarray, target: np.ndarray) -> float | None:
    """Rigidly align source to target and return RMSE."""
    mask = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    src = source[mask]
    tgt = target[mask]
    if len(src) < 20:
        return None
    c_src = np.mean(src, axis=0)
    c_tgt = np.mean(tgt, axis=0)
    h = (src - c_src).T @ (tgt - c_tgt)
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    aligned = (rot @ src.T).T + (c_tgt - rot @ c_src)
    return float(np.sqrt(np.mean(np.sum((aligned - tgt) ** 2, axis=1))))


def build_xsens_keypoint_interpolator(mvnx_path: Path):
    """Build interpolators for Xsens pseudo-COCO keypoints."""
    xsens_time, xsens_kp = build_xsens_coco_keypoints(mvnx_path)
    interps = []
    for joint_idx in range(xsens_kp.shape[1]):
        axes = []
        for axis_idx in range(3):
            values = xsens_kp[:, joint_idx, axis_idx]
            finite = np.isfinite(values)
            if np.count_nonzero(finite) < 2:
                axes.append(None)
            else:
                axes.append(interp1d(xsens_time[finite], values[finite], kind="linear", bounds_error=False, fill_value=np.nan))
        interps.append(axes)

    def sample(query_time: np.ndarray) -> np.ndarray:
        out = np.full((len(query_time), xsens_kp.shape[1], 3), np.nan, dtype=np.float64)
        for joint_idx, axes in enumerate(interps):
            for axis_idx, interp in enumerate(axes):
                if interp is not None:
                    out[:, joint_idx, axis_idx] = interp(query_time)
        return out

    return sample


def delta(values: np.ndarray, k: int) -> np.ndarray:
    """K-frame difference with NaN padding."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full_like(values, np.nan)
    if len(values) > k:
        out[k:] = values[k:] - values[:-k]
    return out


def score_offset(
    offset_s: float,
    video_time: np.ndarray,
    skt_keypoints: np.ndarray,
    skt_angles: dict[str, np.ndarray],
    fair_interps: dict,
    xsens_sample_keypoints,
    angle_names: list[str],
    motion_k: int,
) -> dict[str, float | int | None]:
    """Compute position, angle, and motion scores for one candidate offset."""
    query = video_time - float(offset_s)
    reference_angles = sample_interpolators(fair_interps, query, angle_names)

    angle_scores = []
    angle_errors = []
    motion_scores = []
    for name in angle_names:
        if name not in skt_angles:
            continue
        r = pearson(reference_angles[name], skt_angles[name])
        if r is not None:
            angle_scores.append(r)
        err = rmse(reference_angles[name], skt_angles[name])
        if err is not None:
            angle_errors.append(err)
        md = pearson(delta(reference_angles[name], motion_k), delta(skt_angles[name], motion_k))
        if md is not None:
            motion_scores.append(md)

    position_score = None
    position_rmse = None
    if xsens_sample_keypoints is not None:
        xsens_kp = xsens_sample_keypoints(query)
        joints = [5, 6, 7, 8, 9, 10, 11, 12]
        src = skt_keypoints[:, joints, :].reshape(-1, 3)
        tgt = xsens_kp[:, joints, :].reshape(-1, 3)
        position_rmse = kabsch_rmse(src, tgt)
        if position_rmse is not None:
            position_score = -position_rmse

    return {
        "offset_s": float(offset_s),
        "position_score": position_score,
        "position_rmse_cm": position_rmse,
        "angle_score": float(np.mean(angle_scores)) if angle_scores else None,
        "angle_rmse_deg": float(np.mean(angle_errors)) if angle_errors else None,
        "motion_delta_score": float(np.mean(motion_scores)) if motion_scores else None,
        "angle_pair_count": int(len(angle_scores)),
        "motion_pair_count": int(len(motion_scores)),
    }


def candidate_offsets(start: float, end: float, step: float) -> np.ndarray:
    """Build an inclusive candidate offset vector."""
    count = int(np.floor((end - start) / step)) + 1
    return np.round(start + np.arange(count) * step, 8)


def best_by(rows: list[dict], key: str) -> dict | None:
    """Return row with maximum finite key."""
    valid = [row for row in rows if row.get(key) is not None and np.isfinite(row[key])]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row[key]))


def run_search(
    offsets: np.ndarray,
    video_time: np.ndarray,
    skt_keypoints: np.ndarray,
    skt_angles: dict[str, np.ndarray],
    fair_interps: dict,
    xsens_sample_keypoints,
    angle_names: list[str],
    motion_k: int,
) -> list[dict]:
    """Score all candidate offsets."""
    return [
        score_offset(offset, video_time, skt_keypoints, skt_angles, fair_interps, xsens_sample_keypoints, angle_names, motion_k)
        for offset in offsets
    ]


def plot_scores(rows: list[dict], out_path: Path) -> None:
    """Plot offset score curves."""
    offsets = np.asarray([row["offset_s"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, key, title in [
        (axes[0], "motion_delta_score", "Motion-delta Pearson score"),
        (axes[1], "angle_score", "Angle Pearson score"),
        (axes[2], "position_rmse_cm", "Position RMSE after rigid alignment"),
    ]:
        values = np.asarray([np.nan if row.get(key) is None else row[key] for row in rows], dtype=np.float64)
        ax.plot(offsets, values, linewidth=1.5)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(key)
    axes[-1].set_xlabel("Candidate offset (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def estimate_offset(config: dict, run_dir: Path) -> Path:
    """Estimate and save temporal offset summary."""
    _, skt_keypoints, _ = load_skt_keypoints(config, run_dir)
    video_time, _, _, _ = build_pose_timeline(config, len(skt_keypoints))
    skt_keypoints = skt_keypoints[: len(video_time)]

    eval_cfg = section(config, "evaluation")
    offset_cfg = section(config, "offset")
    refs = section(config, "references")
    angle_names = [name for name in eval_cfg.get("angle_names", list(SEMANTIC_ANGLE_NAMES)) if name in SEMANTIC_ANGLE_NAMES]
    skt_angles = compute_angle_sequence(skt_keypoints, angle_names)

    fair_path = resolve_path(refs.get("xsens_fair_angles"), must_exist=False)
    fair_interps = build_fair_angle_interpolators(fair_path)
    if not fair_interps:
        mvnx_path = resolve_path(refs.get("xsens_mvnx"), must_exist=True)
        fair_interps = build_native_angle_interpolators(mvnx_path)

    mvnx_path = resolve_path(refs.get("xsens_mvnx"), must_exist=False)
    xsens_sampler = build_xsens_keypoint_interpolator(mvnx_path) if mvnx_path and mvnx_path.exists() else None
    search_range = offset_cfg.get("search_range_seconds", [0.0, 30.0])
    coarse_offsets = candidate_offsets(float(search_range[0]), float(search_range[1]), float(offset_cfg.get("coarse_step_seconds", 0.1)))
    motion_k = int(offset_cfg.get("motion_k_frames", 6))

    coarse_rows = run_search(coarse_offsets, video_time, skt_keypoints, skt_angles, fair_interps, xsens_sampler, angle_names, motion_k)
    coarse_best = best_by(coarse_rows, "motion_delta_score") or best_by(coarse_rows, "angle_score") or best_by(coarse_rows, "position_score")
    center = float(coarse_best["offset_s"]) if coarse_best else float(offset_cfg.get("initial_reference_seconds") or 0.0)
    fine_window = float(offset_cfg.get("fine_window_seconds", 0.5))
    fine_start = max(float(search_range[0]), center - fine_window)
    fine_end = min(float(search_range[1]), center + fine_window)
    fine_offsets = candidate_offsets(fine_start, fine_end, float(offset_cfg.get("fine_step_seconds", 0.01)))
    fine_rows = run_search(fine_offsets, video_time, skt_keypoints, skt_angles, fair_interps, xsens_sampler, angle_names, motion_k)

    best_motion = best_by(fine_rows, "motion_delta_score")
    best_angle = best_by(fine_rows, "angle_score")
    best_position = best_by(fine_rows, "position_score")
    priority = offset_cfg.get("selection_priority", ["motion_delta", "angle", "position"])
    selected = None
    selected_source = None
    for item in priority:
        candidate = {"motion_delta": best_motion, "angle": best_angle, "position": best_position}.get(item)
        if candidate is not None:
            selected = candidate
            selected_source = item
            break
    if selected is None:
        raise RuntimeError("Offset search failed: no finite score from any method.")

    rows = coarse_rows + [dict(row, phase="fine") for row in fine_rows]
    for row in coarse_rows:
        row.setdefault("phase", "coarse")

    csv_path = run_dir / "offset_search_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["phase", "offset_s", "position_score", "position_rmse_cm", "angle_score", "angle_rmse_deg", "motion_delta_score", "angle_pair_count", "motion_pair_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_scores(fine_rows, run_dir / "offset_search_scores.png")
    summary = {
        "selected_offset_seconds": float(selected["offset_s"]),
        "selected_source": selected_source,
        "position_best_offset_seconds": None if best_position is None else float(best_position["offset_s"]),
        "angle_best_offset_seconds": None if best_angle is None else float(best_angle["offset_s"]),
        "motion_delta_best_offset_seconds": None if best_motion is None else float(best_motion["offset_s"]),
        "search_range_seconds": search_range,
        "coarse_step_seconds": float(offset_cfg.get("coarse_step_seconds", 0.1)),
        "fine_step_seconds": float(offset_cfg.get("fine_step_seconds", 0.01)),
        "initial_reference_seconds": offset_cfg.get("initial_reference_seconds"),
        "best_rows": {
            "position": best_position,
            "angle": best_angle,
            "motion_delta": best_motion,
        },
        "note": "Offset is estimated automatically. Xsens is used as a comparison/reference system, not absolute ground truth.",
    }
    out_path = run_dir / "alignment_summary.json"
    out_path.write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[offset] selected {summary['selected_offset_seconds']:.3f}s from {selected_source}")
    return out_path


def load_selected_offset(run_dir: Path) -> float:
    """Load selected offset from a run directory."""
    path = run_dir / "alignment_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Run offset first; missing {path}")
    return float(json.loads(path.read_text(encoding="utf-8"))["selected_offset_seconds"])
