"""Analyze detector angle disagreement as a function of estimated stereo depth.

The analysis compares YOLOv8m and YOLO11L on exactly the same valid frames,
uses one fixed Xsens time offset per session, and gives both models the same
YOLOv8m-derived depth coordinate. Xsens is treated only as an external,
Xsens-derived comparison system.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common.config import load_config  # noqa: E402
from common.dataset import load_method_keypoints, resolve_skt_npz  # noqa: E402
from eval_angles import prepare_angles  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RIGHT_ARM_INDICES = (6, 8, 10)
TORSO_INDICES = (5, 6, 11, 12)
MODEL_COLORS = {"YOLOv8m": "#2563EB", "YOLO11L": "#DC2626"}
SESSION_COLORS = {
    "fanbo7_a257": "#2A9D8F",
    "fanbo9_a257": "#E9C46A",
    "fanbo9_a255": "#F4A261",
    "fanbo4_a257": "#6D597A",
}


@dataclass(frozen=True)
class ModelSpec:
    """One model/config/run combination."""

    model: str
    config_path: Path
    run_dir: Path


@dataclass(frozen=True)
class SessionSpec:
    """One paired detector comparison with a shared fixed time offset."""

    name: str
    session_id: str
    fixed_offset_seconds: float
    baseline: ModelSpec
    candidate: ModelSpec


def _resolve(value: str | Path) -> Path:
    """Resolve a project-relative path without changing the source value."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    """Return the current project commit, or an explicit unknown marker."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_analysis_config(path: Path) -> tuple[dict[str, Any], list[SessionSpec]]:
    """Load and validate the distance-analysis YAML configuration."""
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    sessions: list[SessionSpec] = []
    for raw in payload.get("sessions", []):
        baseline = raw["baseline"]
        candidate = raw["candidate"]
        sessions.append(
            SessionSpec(
                name=str(raw["name"]),
                session_id=str(raw["session_id"]),
                fixed_offset_seconds=float(raw["fixed_offset_seconds"]),
                baseline=ModelSpec(
                    model=str(baseline["model"]),
                    config_path=_resolve(baseline["config"]),
                    run_dir=_resolve(baseline["run_dir"]),
                ),
                candidate=ModelSpec(
                    model=str(candidate["model"]),
                    config_path=_resolve(candidate["config"]),
                    run_dir=_resolve(candidate["run_dir"]),
                ),
            )
        )
    if not sessions:
        raise ValueError("No sessions are configured for distance analysis")
    return payload, sessions


def _torso_distance(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate optical depth and radial range from the COCO torso in metres."""
    torso = np.asarray(keypoints[:, TORSO_INDICES, :], dtype=np.float64)
    finite_joint = np.isfinite(torso).all(axis=2)
    centers = np.full((len(torso), 3), np.nan, dtype=np.float64)
    for index in range(len(torso)):
        valid = finite_joint[index]
        if int(valid.sum()) >= 2:
            centers[index] = np.nanmedian(torso[index, valid], axis=0)
    depth_m = centers[:, 2] / 100.0
    range_m = np.linalg.norm(centers, axis=1) / 100.0
    invalid = ~np.isfinite(centers).all(axis=1) | (depth_m <= 0.0)
    depth_m[invalid] = np.nan
    range_m[invalid] = np.nan
    return depth_m, range_m


def _quality_series(config: dict[str, Any], run_dir: Path, count: int) -> dict[str, np.ndarray]:
    """Read right-arm stereo quality arrays from the model NPZ."""
    npz_path = resolve_skt_npz(config, run_dir)
    with np.load(npz_path, allow_pickle=True) as payload:
        available = set(payload.files)

        def read(name: str) -> np.ndarray:
            if name not in available:
                return np.full((count, 17), np.nan, dtype=np.float64)
            return np.asarray(payload[name], dtype=np.float64)[:count]

        pair_conf = read("pair_confidence")
        if not np.isfinite(pair_conf).any():
            left_name = "triang_conf_left" if "triang_conf_left" in available else "conf_left"
            right_name = "triang_conf_right" if "triang_conf_right" in available else "conf_right"
            pair_conf = np.minimum(read(left_name), read(right_name))
        epipolar = read("epipolar_error")
        reprojection = read("reprojection_error")

    arm = list(RIGHT_ARM_INDICES)

    def finite_row_reduce(values: np.ndarray, mode: str) -> np.ndarray:
        """Reduce finite row values without emitting all-NaN warnings."""
        finite = np.isfinite(values)
        fill = np.inf if mode == "min" else -np.inf
        prepared = np.where(finite, values, fill)
        reduced = (
            np.min(prepared, axis=1)
            if mode == "min"
            else np.max(prepared, axis=1)
        )
        reduced[~finite.any(axis=1)] = np.nan
        return reduced

    return {
        "right_arm_pair_conf_min": finite_row_reduce(pair_conf[:, arm], "min"),
        "right_arm_epipolar_max_px": finite_row_reduce(epipolar[:, arm], "max"),
        "right_arm_reprojection_max_px": finite_row_reduce(
            reprojection[:, arm], "max"
        ),
    }


def _model_payload(
    model_spec: ModelSpec,
    fixed_offset_seconds: float,
    joint: str,
) -> dict[str, Any]:
    """Load processed joint angles, timeline, keypoints, and quality metrics."""
    config = load_config(model_spec.config_path)
    config.setdefault("evaluation", {})["angle_names"] = [joint]
    time_s, all_angles, info = prepare_angles(
        config, model_spec.run_dir, fixed_offset_seconds
    )
    if joint not in all_angles.get("SKT", {}):
        raise KeyError(f"{joint} is not available in {model_spec.config_path}")
    key_time_s, _, methods = load_method_keypoints(config, model_spec.run_dir)
    if len(time_s) != len(key_time_s) or not np.allclose(time_s, key_time_s, atol=1e-7):
        raise RuntimeError(f"Timeline mismatch in {model_spec.run_dir}")
    quality = _quality_series(config, model_spec.run_dir, len(time_s))
    return {
        "config": config,
        "time_s": time_s,
        "vision_angle_deg": np.asarray(all_angles["SKT"][joint], dtype=np.float64),
        "reference_angle_deg": np.asarray(all_angles["XsensFair"][joint], dtype=np.float64),
        "keypoints": np.asarray(methods["SKT"], dtype=np.float64),
        "quality": quality,
        "angle_info": info,
        "npz_path": resolve_skt_npz(config, model_spec.run_dir),
    }


def extract_rows(
    sessions: list[SessionSpec], joint: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build paired per-frame rows and a source manifest."""
    output_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for session in sessions:
        baseline = _model_payload(
            session.baseline, session.fixed_offset_seconds, joint
        )
        candidate = _model_payload(
            session.candidate, session.fixed_offset_seconds, joint
        )
        if len(baseline["time_s"]) != len(candidate["time_s"]):
            raise RuntimeError(f"Frame-count mismatch for {session.name}")
        if not np.allclose(baseline["time_s"], candidate["time_s"], atol=1e-7):
            raise RuntimeError(f"Paired timelines do not match for {session.name}")

        depth_m, range_m = _torso_distance(baseline["keypoints"])
        base_ref = baseline["reference_angle_deg"]
        cand_ref = candidate["reference_angle_deg"]
        ref_both = np.isfinite(base_ref) & np.isfinite(cand_ref)
        if np.any(ref_both):
            max_ref_difference = float(np.nanmax(np.abs(base_ref[ref_both] - cand_ref[ref_both])))
            if max_ref_difference > 1e-5:
                message = (
                    "Shared-offset reference differs by "
                    f"{max_ref_difference:.6f} deg for {session.name}"
                )
                raise RuntimeError(message)

        base_valid = np.isfinite(baseline["vision_angle_deg"]) & np.isfinite(base_ref)
        cand_valid = np.isfinite(candidate["vision_angle_deg"]) & np.isfinite(cand_ref)
        distance_valid = np.isfinite(depth_m) & np.isfinite(range_m)
        common_valid = base_valid & cand_valid & distance_valid

        model_data = ((session.baseline, baseline), (session.candidate, candidate))
        for model_spec, data in model_data:
            model_valid = np.isfinite(data["vision_angle_deg"]) & np.isfinite(
                data["reference_angle_deg"]
            )
            for frame in range(len(data["time_s"])):
                vision = data["vision_angle_deg"][frame]
                reference = data["reference_angle_deg"][frame]
                output_rows.append(
                    {
                        "session_id": session.session_id,
                        "session": session.name,
                        "frame": frame,
                        "time_s": float(data["time_s"][frame]),
                        "model": model_spec.model,
                        "joint": joint,
                        "fixed_offset_s": session.fixed_offset_seconds,
                        "optical_depth_m": depth_m[frame],
                        "radial_range_m": range_m[frame],
                        "vision_angle_deg": vision,
                        "reference_angle_deg": reference,
                        "signed_difference_deg": vision - reference,
                        "abs_difference_deg": abs(vision - reference),
                        "model_valid": bool(model_valid[frame] and distance_valid[frame]),
                        "common_valid": bool(common_valid[frame]),
                        "right_arm_pair_conf_min": data["quality"][
                            "right_arm_pair_conf_min"
                        ][frame],
                        "right_arm_epipolar_max_px": data["quality"][
                            "right_arm_epipolar_max_px"
                        ][frame],
                        "right_arm_reprojection_max_px": data["quality"][
                            "right_arm_reprojection_max_px"
                        ][frame],
                    }
                )

        for model_spec, data in model_data:
            calibration_path = _resolve(data["config"]["calibration"]["camera_params"])
            sources.append(
                {
                    "session_id": session.session_id,
                    "session": session.name,
                    "model": model_spec.model,
                    "config": str(model_spec.config_path.relative_to(PROJECT_ROOT)),
                    "config_sha256": _sha256(model_spec.config_path),
                    "run_dir": str(model_spec.run_dir.relative_to(PROJECT_ROOT)),
                    "npz": str(data["npz_path"].relative_to(PROJECT_ROOT)),
                    "npz_sha256": _sha256(data["npz_path"]),
                    "calibration": str(calibration_path.relative_to(PROJECT_ROOT)),
                    "calibration_sha256": _sha256(calibration_path),
                    "fixed_offset_seconds": session.fixed_offset_seconds,
                    "camera_smooth_window_actual_ms": data["angle_info"][
                        "camera_smooth_window_actual_ms"
                    ],
                }
            )

    frame_data = pd.DataFrame(output_rows)
    numeric_columns = [
        "optical_depth_m",
        "radial_range_m",
        "vision_angle_deg",
        "reference_angle_deg",
        "signed_difference_deg",
        "abs_difference_deg",
        "right_arm_pair_conf_min",
        "right_arm_epipolar_max_px",
        "right_arm_reprojection_max_px",
    ]
    frame_data.loc[~frame_data["model_valid"], numeric_columns[2:6]] = np.nan
    return frame_data, sources


def extract_joint_median_rows(
    sessions: list[SessionSpec],
    joints: tuple[str, ...],
    label: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Aggregate one or two joint errors into one per-frame analysis target."""
    if len(joints) == 1:
        frame_data, sources = extract_rows(sessions, joints[0])
        frame_data["joint"] = label
        frame_data["component_joints"] = joints[0]
        return frame_data, sources
    if len(joints) != 2:
        raise ValueError("A distance-analysis target must contain one or two joints")
    left, sources = extract_rows(sessions, joints[0])
    right, _ = extract_rows(sessions, joints[1])
    identity = [
        "session_id",
        "session",
        "frame",
        "time_s",
        "model",
        "fixed_offset_s",
        "optical_depth_m",
        "radial_range_m",
    ]
    metrics = [
        "vision_angle_deg",
        "reference_angle_deg",
        "signed_difference_deg",
        "abs_difference_deg",
        "model_valid",
        "common_valid",
    ]
    left_columns = {
        name: f"{joints[0]}_{name}"
        for name in metrics
    }
    right_columns = {
        name: f"{joints[1]}_{name}"
        for name in metrics
    }
    merged = left[identity + metrics].rename(columns=left_columns).merge(
        right[identity + metrics].rename(columns=right_columns),
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    valid_columns = [f"{joint}_model_valid" for joint in joints]
    common_columns = [f"{joint}_common_valid" for joint in joints]
    merged["model_valid"] = merged[valid_columns].all(axis=1)
    merged["common_valid"] = merged[common_columns].all(axis=1)
    for metric in (
        "vision_angle_deg",
        "reference_angle_deg",
        "signed_difference_deg",
        "abs_difference_deg",
    ):
        columns = [f"{joint}_{metric}" for joint in joints]
        merged[metric] = merged[columns].median(axis=1, skipna=False)
    merged["joint"] = label
    merged["component_joints"] = ",".join(joints)
    return merged, sources


def add_distance_bins(frame_data: pd.DataFrame, width_m: float) -> pd.DataFrame:
    """Attach stable half-metre distance-bin bounds and labels."""
    data = frame_data.copy()
    finite = data.loc[np.isfinite(data["optical_depth_m"]), "optical_depth_m"]
    if finite.empty:
        raise ValueError("No finite optical-depth estimates are available")
    start = math.floor(float(finite.min()) / width_m) * width_m
    stop = math.ceil(float(finite.max()) / width_m) * width_m + width_m
    edges = np.arange(start, stop + width_m * 0.1, width_m)
    labels = [f"{left:.1f}–{right:.1f}" for left, right in zip(edges[:-1], edges[1:])]
    data["distance_bin"] = pd.cut(
        data["optical_depth_m"], bins=edges, labels=labels, right=False
    )
    data["distance_bin_left_m"] = np.floor(data["optical_depth_m"] / width_m) * width_m
    return data


def _distribution(values: pd.Series) -> dict[str, float | int]:
    """Return robust and conventional distribution statistics."""
    finite = values[np.isfinite(values)].to_numpy(dtype=np.float64)
    if finite.size == 0:
        return {
            "n": 0,
            "mean_deg": math.nan,
            "median_deg": math.nan,
            "p25_deg": math.nan,
            "p75_deg": math.nan,
            "p90_deg": math.nan,
            "p95_deg": math.nan,
            "rmse_deg": math.nan,
        }
    return {
        "n": int(finite.size),
        "mean_deg": float(np.mean(finite)),
        "median_deg": float(np.median(finite)),
        "p25_deg": float(np.percentile(finite, 25)),
        "p75_deg": float(np.percentile(finite, 75)),
        "p90_deg": float(np.percentile(finite, 90)),
        "p95_deg": float(np.percentile(finite, 95)),
        "rmse_deg": float(np.sqrt(np.mean(finite**2))),
    }


def summarize_data(
    frame_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create session, distance-bin, and paired-model summaries."""
    valid = frame_data[frame_data["common_valid"]].copy()
    session_rows: list[dict[str, Any]] = []
    for (session_id, session, model), group in valid.groupby(
        ["session_id", "session", "model"], observed=True, sort=False
    ):
        all_session_rows = frame_data[
            (frame_data["session_id"] == session_id)
            & (frame_data["model"] == model)
        ]
        stats = _distribution(group["abs_difference_deg"])
        session_rows.append(
            {
                "session_id": session_id,
                "session": session,
                "model": model,
                "n_total_frames": int(len(all_session_rows)),
                "n_common_valid": stats.pop("n"),
                "common_valid_ratio": float(len(group) / len(all_session_rows)),
                "median_optical_depth_m": float(np.nanmedian(group["optical_depth_m"])),
                "p25_optical_depth_m": float(np.nanpercentile(group["optical_depth_m"], 25)),
                "p75_optical_depth_m": float(np.nanpercentile(group["optical_depth_m"], 75)),
                **stats,
            }
        )
    session_summary = pd.DataFrame(session_rows)

    bin_rows: list[dict[str, Any]] = []
    for (distance_bin, model), group in valid.groupby(
        ["distance_bin", "model"], observed=True, sort=True
    ):
        stats = _distribution(group["abs_difference_deg"])
        bin_rows.append(
            {
                "distance_bin": str(distance_bin),
                "distance_bin_left_m": float(group["distance_bin_left_m"].iloc[0]),
                "model": model,
                "median_optical_depth_m": float(np.nanmedian(group["optical_depth_m"])),
                **stats,
            }
        )
    bin_summary = pd.DataFrame(bin_rows).sort_values(
        ["distance_bin_left_m", "model"]
    )

    pivot = valid.pivot_table(
        index=[
            "session_id",
            "session",
            "frame",
            "time_s",
            "optical_depth_m",
            "radial_range_m",
            "distance_bin",
            "distance_bin_left_m",
        ],
        columns="model",
        values="abs_difference_deg",
        aggfunc="first",
        observed=True,
    ).reset_index()
    pivot["difference_11l_minus_8m_deg"] = pivot["YOLO11L"] - pivot["YOLOv8m"]
    paired_rows: list[dict[str, Any]] = []
    for distance_bin, group in pivot.groupby("distance_bin", observed=True, sort=True):
        differences = group["difference_11l_minus_8m_deg"].to_numpy(dtype=np.float64)
        paired_rows.append(
            {
                "distance_bin": str(distance_bin),
                "distance_bin_left_m": float(group["distance_bin_left_m"].iloc[0]),
                "n_common_frames": int(len(group)),
                "median_11l_minus_8m_deg": float(np.nanmedian(differences)),
                "p25_11l_minus_8m_deg": float(np.nanpercentile(differences, 25)),
                "p75_11l_minus_8m_deg": float(np.nanpercentile(differences, 75)),
                "yolo11l_lower_disagreement_ratio": float(
                    np.nanmean(differences < 0.0)
                ),
            }
        )
    paired_summary = pd.DataFrame(paired_rows).sort_values("distance_bin_left_m")

    descriptive_correlations: dict[str, Any] = {}
    for model, group in valid.groupby("model", sort=False):
        rho, p_value = spearmanr(
            group["optical_depth_m"].to_numpy(dtype=np.float64),
            group["abs_difference_deg"].to_numpy(dtype=np.float64),
            nan_policy="omit",
        )
        descriptive_correlations[model] = {
            "frame_level_spearman_rho": float(rho),
            "p_value_not_independence_adjusted": float(p_value),
            "n": int(len(group)),
            "warning": (
                "Frames are temporally autocorrelated and sessions differ in "
                "action/viewpoint; use descriptively only."
            ),
        }
    overall = {
        "common_valid_unique_frames": int(
            valid[["session_id", "frame"]].drop_duplicates().shape[0]
        ),
        "models": {
            model: _distribution(group["abs_difference_deg"])
            for model, group in valid.groupby("model", sort=False)
        },
        "descriptive_correlations": descriptive_correlations,
    }
    extras = {"overall": overall, "paired_frames": pivot}
    return session_summary, bin_summary, paired_summary, extras


def _configure_plot_style(chinese: bool) -> None:
    """Apply consistent plotting style and a Chinese-capable font when found."""
    if chinese:
        candidates = [
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
            Path.home() / "Library/Fonts/TencentSans-W7.ttf",
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
        for path in candidates:
            if path.exists():
                font_manager.fontManager.addfont(str(path))
                matplotlib.rcParams["font.family"] = font_manager.FontProperties(
                    fname=str(path)
                ).get_name()
                break
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close one publication-style figure."""
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _error_axis_max(values: pd.Series) -> float:
    """Choose a readable axis limit that still contains every finite error."""
    finite = values[np.isfinite(values)].to_numpy(dtype=np.float64)
    if finite.size == 0:
        return 50.0
    rounded = math.ceil((float(finite.max()) + 2.0) / 10.0) * 10.0
    return float(min(180.0, max(50.0, rounded)))


def plot_core_distance_curve(
    bin_summary: pd.DataFrame,
    path: Path,
    chinese: bool,
    body_region: str,
) -> None:
    """Plot the report's primary median-error versus distance curve."""
    _configure_plot_style(chinese)
    fig, axis = plt.subplots(figsize=(10.4, 5.6))
    for model in ("YOLOv8m", "YOLO11L"):
        trend = bin_summary[bin_summary["model"] == model].sort_values(
            "distance_bin_left_m"
        )
        x = trend["median_optical_depth_m"].to_numpy(dtype=float)
        median = trend["median_deg"].to_numpy(dtype=float)
        p25 = trend["p25_deg"].to_numpy(dtype=float)
        p75 = trend["p75_deg"].to_numpy(dtype=float)
        axis.fill_between(
            x,
            p25,
            p75,
            color=MODEL_COLORS[model],
            alpha=0.13,
        )
        axis.plot(
            x,
            median,
            "o-",
            color=MODEL_COLORS[model],
            linewidth=2.6,
            markersize=7,
            label=model,
        )
    axis.set_xlabel(
        "估计光轴深度（m）" if chinese else "Estimated optical depth (m)"
    )
    axis.set_ylabel(
        "中位绝对角度差（°）"
        if chinese
        else "Median absolute angular disagreement (deg)"
    )
    axis.set_title(
        (
            f"{'右肩主指标' if body_region == 'shoulder' else '双髋验证指标'}："
            "角度差随距离的变化"
        )
        if chinese
        else (
            f"{'Primary right-shoulder metric' if body_region == 'shoulder' else 'Bilateral-hip validation metric'}: "
            "angular disagreement versus distance"
        ),
        fontweight="bold",
    )
    axis.grid(True, color="#DCE3EC", linewidth=0.8, alpha=0.8)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_scatter(
    valid: pd.DataFrame, bin_summary: pd.DataFrame, path: Path, chinese: bool
) -> None:
    """Plot per-frame scatter points with median and IQR distance-bin trends."""
    _configure_plot_style(chinese)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharex=True, sharey=True)
    for axis, model in zip(axes, ("YOLOv8m", "YOLO11L")):
        model_data = valid[valid["model"] == model]
        for session_id, group in model_data.groupby("session_id", sort=False):
            axis.scatter(
                group["optical_depth_m"],
                group["abs_difference_deg"],
                s=12,
                alpha=0.26,
                color=SESSION_COLORS.get(session_id, "#777777"),
                edgecolors="none",
                label=str(group["session"].iloc[0]),
            )
        trend = bin_summary[bin_summary["model"] == model].sort_values(
            "distance_bin_left_m"
        )
        x = trend["median_optical_depth_m"].to_numpy(dtype=float)
        median = trend["median_deg"].to_numpy(dtype=float)
        p25 = trend["p25_deg"].to_numpy(dtype=float)
        p75 = trend["p75_deg"].to_numpy(dtype=float)
        axis.fill_between(x, p25, p75, color=MODEL_COLORS[model], alpha=0.16)
        axis.plot(x, median, "o-", color=MODEL_COLORS[model], linewidth=2.3, markersize=5)
        axis.set_title(model)
        axis.set_xlabel("估计光轴深度（m）" if chinese else "Estimated optical depth (m)")
        axis.set_ylim(0, _error_axis_max(valid["abs_difference_deg"]))
    axes[0].set_ylabel(
        "相对 Xsens-derived reference 的绝对角度差（°）"
        if chinese
        else "Absolute angular disagreement vs Xsens-derived reference (deg)"
    )
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "逐帧散点 + 0.5 m 分箱中位数/IQR（同一有效帧）"
        if chinese
        else "Per-frame scatter with 0.5 m median/IQR bins (same valid frames)",
        y=0.99,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    _save_figure(fig, path)


def plot_boxplots(
    valid: pd.DataFrame,
    path: Path,
    chinese: bool,
    body_region: str,
) -> None:
    """Plot side-by-side detector distributions for each distance bin."""
    _configure_plot_style(chinese)
    bins = [str(value) for value in valid["distance_bin"].dropna().cat.categories]
    bins = [value for value in bins if (valid["distance_bin"].astype(str) == value).any()]
    fig, axis = plt.subplots(figsize=(12.4, 5.3))
    positions: list[float] = []
    data: list[np.ndarray] = []
    colors: list[str] = []
    for bin_index, label in enumerate(bins):
        for model_index, model in enumerate(("YOLOv8m", "YOLO11L")):
            group = valid[
                (valid["distance_bin"].astype(str) == label)
                & (valid["model"] == model)
            ]["abs_difference_deg"].dropna()
            if group.empty:
                continue
            positions.append(bin_index + (-0.17 if model_index == 0 else 0.17))
            data.append(group.to_numpy(dtype=float))
            colors.append(MODEL_COLORS[model])
    boxes = axis.boxplot(
        data,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.18},
        medianprops={"color": "#111827", "linewidth": 1.5},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.55)
    axis.set_xticks(range(len(bins)), bins)
    finite_error = valid.loc[
        np.isfinite(valid["abs_difference_deg"]), "abs_difference_deg"
    ].to_numpy(dtype=float)
    full_max = float(np.max(finite_error))
    robust_top = math.ceil(
        (float(np.percentile(finite_error, 99)) + 5.0) / 10.0
    ) * 10.0
    axis_max = min(full_max * 1.05, max(30.0, robust_top))
    axis.set_ylim(0, axis_max)
    axis.set_xlabel(
        "估计光轴深度分箱（m）"
        if chinese
        else "Estimated optical-depth bin (m)"
    )
    axis.set_ylabel(
        "绝对角度差（°）" if chinese else "Absolute angular disagreement (deg)"
    )
    axis.set_title(
        (
            f"{'右肩主指标' if body_region == 'shoulder' else '双髋验证指标'}："
            "距离分箱误差分布"
        )
        if chinese
        else (
            f"{'Primary right-shoulder metric' if body_region == 'shoulder' else 'Bilateral-hip validation metric'}: "
            "distributions by distance bin"
        ),
        fontweight="bold",
    )
    if full_max > axis_max:
        clipped_count = int(np.sum(finite_error > axis_max))
        axis.text(
            0.99,
            0.96,
            (
                f"纵轴聚焦至 P99；{clipped_count} 个极端点高于显示范围"
                if chinese
                else (
                    f"Axis focused through P99; {clipped_count} extreme points "
                    "are above the displayed range"
                )
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#5D6B7A",
        )
    legend_handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[model], linewidth=8, alpha=0.55)
        for model in ("YOLOv8m", "YOLO11L")
    ]
    axis.legend(legend_handles, ["YOLOv8m", "YOLO11L"], frameon=False)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_session_summary(session_summary: pd.DataFrame, path: Path, chinese: bool) -> None:
    """Plot session median and P95 against median optical depth."""
    _configure_plot_style(chinese)
    fig, axis = plt.subplots(figsize=(9.7, 5.4))
    for model in ("YOLOv8m", "YOLO11L"):
        group = session_summary[session_summary["model"] == model].sort_values(
            "median_optical_depth_m"
        )
        axis.scatter(
            group["median_optical_depth_m"],
            group["median_deg"],
            color=MODEL_COLORS[model],
            marker="o",
            s=64,
            label=f"{model} median",
        )
        axis.scatter(
            group["median_optical_depth_m"],
            group["p95_deg"],
            marker="^",
            s=72,
            facecolors="none",
            edgecolors=MODEL_COLORS[model],
            linewidths=1.8,
            label=f"{model} P95",
        )
    for session, group in session_summary.groupby("session", sort=False):
        axis.annotate(
            str(session).replace(" ", "\n", 1),
            (
                float(group["median_optical_depth_m"].median()),
                float(group["median_deg"].max()),
            ),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#334155",
        )
    axis.set_xlabel(
        "序列中位光轴深度（m）"
        if chinese
        else "Session median optical depth (m)"
    )
    axis.set_ylabel("绝对角度差（°）" if chinese else "Absolute angular disagreement (deg)")
    axis.set_title(
        "每个序列：中位数与 P95"
        if chinese
        else "Per-session median and P95 disagreement",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_paired_difference(paired: pd.DataFrame, path: Path, chinese: bool) -> None:
    """Plot paired YOLO11L-minus-YOLOv8m disagreement by distance bin."""
    _configure_plot_style(chinese)
    x = np.arange(len(paired))
    medians = paired["median_11l_minus_8m_deg"].to_numpy(dtype=float)
    lower = medians - paired["p25_11l_minus_8m_deg"].to_numpy(dtype=float)
    upper = paired["p75_11l_minus_8m_deg"].to_numpy(dtype=float) - medians
    colors = ["#B42318" if value > 0 else "#087F5B" for value in medians]
    fig, axis = plt.subplots(figsize=(10.3, 4.9))
    axis.bar(x, medians, color=colors, alpha=0.82, width=0.62)
    axis.errorbar(
        x,
        medians,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#334155",
        capsize=4,
        linewidth=1.2,
    )
    axis.axhline(0.0, color="#111827", linewidth=1.2)
    tick_labels = [
        f"{row.distance_bin}\n(n={int(row.n_common_frames)})"
        for row in paired.itertuples(index=False)
    ]
    axis.set_xticks(x, tick_labels)
    axis.set_xlabel(
        "估计光轴深度分箱（m）"
        if chinese
        else "Estimated optical-depth bin (m)"
    )
    axis.set_ylabel(
        "11L - 8m 的绝对角度差（°）"
        if chinese
        else "Absolute disagreement: 11L minus 8m (deg)"
    )
    axis.set_title(
        "同一帧成对比较：负值才表示 11L 更好"
        if chinese
        else "Paired same-frame comparison: negative values favour YOLO11L",
        fontweight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, path)


def plot_mean_median(session_summary: pd.DataFrame, path: Path, chinese: bool) -> None:
    """Show how mean and median can rank the same detector result differently."""
    _configure_plot_style(chinese)
    labels = session_summary[["session", "model"]].apply(
        lambda row: f"{row['session']}\n{row['model']}", axis=1
    )
    order = session_summary.sort_values(["median_optical_depth_m", "model"]).index
    ordered = session_summary.loc[order]
    labels = labels.loc[order]
    x = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(12.7, 5.0))
    axis.vlines(x, ordered["median_deg"], ordered["mean_deg"], color="#94A3B8", linewidth=2)
    axis.scatter(x, ordered["median_deg"], color="#155EEF", s=60, label="Median")
    axis.scatter(x, ordered["mean_deg"], color="#F79009", s=60, marker="s", label="Mean")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("绝对角度差（°）" if chinese else "Absolute angular disagreement (deg)")
    axis.set_title(
        "平均值会被少数大误差帧拉高"
        if chinese
        else "The mean is pulled upward by a few large-disagreement frames",
        fontweight="bold",
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, path)


def _image_data_uri(path: Path) -> str:
    """Return a self-contained PNG data URI for an HTML report."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _fmt(value: float, digits: int = 2) -> str:
    """Format one finite numeric value for HTML."""
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def _summary_table_html(session_summary: pd.DataFrame, chinese: bool) -> str:
    """Build a compact robust-statistics table."""
    headers = (
        [
            "序列",
            "模型",
            "深度中位数(m)",
            "共同有效帧",
            "中位差(°)",
            "IQR(°)",
            "P95(°)",
            "平均差(°)",
        ]
        if chinese
        else [
            "Session",
            "Model",
            "Median depth (m)",
            "Common frames",
            "Median (deg)",
            "IQR (deg)",
            "P95 (deg)",
            "Mean (deg)",
        ]
    )
    rows = []
    ordered = session_summary.sort_values(["median_optical_depth_m", "model"])
    for _, row in ordered.iterrows():
        cells = [
            str(row["session"]),
            str(row["model"]),
            _fmt(float(row["median_optical_depth_m"])),
            str(int(row["n_common_valid"])),
            _fmt(float(row["median_deg"])),
            f"{_fmt(float(row['p25_deg']))}–{_fmt(float(row['p75_deg']))}",
            _fmt(float(row["p95_deg"])),
            _fmt(float(row["mean_deg"])),
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    header_html = "".join(f"<th>{head}</th>" for head in headers)
    return (
        f"<table><thead><tr>{header_html}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


STYLE = """
:root{--ink:#172033;--muted:#5d6b7a;--line:#dce3ec;--blue:#155eef;--green:#087f5b;--red:#b42318;--amber:#b45309;--bg:#f4f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.67 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}
header{background:linear-gradient(125deg,#102a43,#155eef);color:white;padding:46px max(6vw,28px)}header h1{max-width:1000px;margin:0 0 10px;font-size:32px}header p{max-width:960px;margin:0;opacity:.92}
.layout{display:grid;grid-template-columns:220px minmax(0,980px);gap:28px;max-width:1280px;margin:28px auto;padding:0 22px}nav{position:sticky;top:18px;align-self:start;background:white;border:1px solid var(--line);border-radius:12px;padding:17px}nav a{display:block;padding:6px 2px;text-decoration:none;color:#38506a}main{min-width:0}section{background:white;border:1px solid var(--line);border-radius:13px;padding:25px 29px;margin-bottom:20px;box-shadow:0 4px 18px #2230470a}
h2{margin:0 0 14px;color:#102a43;font-size:23px}h3{margin:20px 0 7px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin:15px 0}.card{border:1px solid var(--line);border-radius:9px;padding:13px}.metric{font-size:23px;font-weight:750;color:var(--blue)}
table{width:100%;border-collapse:collapse;margin:14px 0 18px;font-size:13.2px}th{background:#eef4ff;text-align:left}th,td{border:1px solid var(--line);padding:8px 9px;vertical-align:top}tr:nth-child(even) td{background:#fafbfd}.callout{border-left:4px solid var(--blue);background:#f1f6ff;padding:12px 15px;margin:15px 0}.warning{border-left-color:var(--amber);background:#fff8e8}.bad{color:var(--red);font-weight:700}.ok{color:var(--green);font-weight:700}
figure{margin:21px 0}figure img{width:100%;border:1px solid var(--line);border-radius:9px}figcaption{font-size:13px;color:var(--muted)}code{background:#edf1f7;padding:2px 5px;border-radius:4px}footer{color:var(--muted);font-size:13px;padding-bottom:28px}@media(max-width:850px){.layout{display:block}nav{position:static;margin-bottom:18px}.cards{grid-template-columns:1fr}section{padding:20px 17px}}
"""


def build_report(
    chinese: bool,
    session_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    overall: dict[str, Any],
    figure_paths: dict[str, Path],
) -> str:
    """Build one bilingual self-contained HTML report."""
    language_suffix = "_CN" if chinese else ""
    figures = {
        key: _image_data_uri(path.with_name(f"{path.stem}{language_suffix}{path.suffix}"))
        for key, path in figure_paths.items()
    }
    model_stats = overall["models"]
    eight = model_stats["YOLOv8m"]
    eleven = model_stats["YOLO11L"]
    bins_favouring_11l = int((paired_summary["median_11l_minus_8m_deg"] < 0).sum())
    bin_count = int(len(paired_summary))
    table = _summary_table_html(session_summary, chinese)
    if chinese:
        title = "距离、检测模型与双目角度差异：直观统计分析"
        subtitle = (
            "YOLOv8m vs YOLO11L · 同一有效帧 · 固定时间偏移 · "
            "Xsens-derived reference 仅作外部比较"
        )
        nav = ["核心图", "详细散点图", "箱线图", "模型成对比较", "平均值问题", "限制与下一步"]
        body = f"""
<section id="summary"><h2>1. 核心图：误差随距离如何变化？</h2><div class="cards"><div class="card"><div class="metric">{eight['median_deg']:.2f}°</div>YOLOv8m 右肩中位误差</div><div class="card"><div class="metric">{eleven['median_deg']:.2f}°</div>YOLO11L 右肩中位误差</div><div class="card"><div class="metric">{bins_favouring_11l}/{bin_count}</div>合格距离分箱中，11L 中位数更低</div></div><figure><img src="{figures['core']}"><figcaption><strong>主指标：</strong>右肩在四组距离数据中都有可用参考，并且现有误差明显低于肘部。曲线为每 0.5 m 距离分箱的中位绝对角度差，色带为 P25–P75。</figcaption></figure><figure><img src="{figures['hip']}"><figcaption><strong>验证指标：</strong>每帧汇总左右髋部的绝对角度差，再按距离分箱统计。只有肩部和髋部都呈现相似趋势时，才更有把握认为距离确实产生了影响。</figcaption></figure><p>两组近端关节曲线并没有呈现一致的平滑单调上升：右肩在最远距离没有恶化，双髋反而总体下降。因此当前记录不能证明“距离越远，角度误差越大”；动作、视角、遮挡和序列差异仍然混在距离因素中。右肘结果不再作为主要结论依据。</p><div class="callout">所有比较都使用同一批有效帧、每个序列一个固定时间偏移，并把 YOLOv8m 的躯干深度作为两个模型共同的横坐标。每个纳入结论的距离分箱至少包含 20 个共同有效帧。Fanbo7 的左肩参考不可用，因此主指标固定使用所有距离都可用的右肩，避免在不同距离混用单肩和双肩。下表为右肩主指标的序列级统计。</div>{table}</section>
<section id="scatter"><h2>2. 右肩主指标的详细散点图</h2><figure><img src="{figures['scatter']}"><figcaption>每个点是一帧的右肩绝对角度差。粗线是每 0.5 m 分箱的中位数，色带是 P25–P75。点很散，说明估计深度之外还有其他因素，但当前数据不能分离视角、动作和遮挡各自的影响。</figcaption></figure><figure><img src="{figures['session']}"><figcaption>将每个序列压缩为一个中位数和一个 P95 后，仍然没有形成随距离平滑上升的趋势。</figcaption></figure></section>
<section id="box"><h2>3. 箱线图：距离趋势是否稳定？</h2><figure><img src="{figures['box']}"><figcaption><strong>右肩主指标：</strong>箱体展示中间 50% 的帧，中线是中位数，散点是离群帧。相邻距离箱如果大量重叠，说明距离不能单独解释误差变化。</figcaption></figure><figure><img src="{figures['hip_box']}"><figcaption><strong>双髋验证指标：</strong>与右肩箱线图对照。只有两类近端关节都出现相似的箱体移动，距离趋势才更可信。</figcaption></figure></section>
<section id="paired"><h2>4. 同一帧直接比较两个模型</h2><figure><img src="{figures['paired']}"><figcaption>纵轴是“YOLO11L 绝对差 − YOLOv8m 绝对差”。负值才表示 11L 更好；误差线为 P25–P75。当前并未形成一个可靠的近远距离切换点。</figcaption></figure><p>因此，更稳妥的工程建议是：目前继续保留 YOLOv8m；如果以后要做动态切换，需要增加更多距离、相同动作和相同视角的受控记录，再预先定义切换阈值并在新序列上验证。</p></section>
<section id="mean"><h2>5. 为什么老师强调 Median？</h2><figure><img src="{figures['mean_median']}"><figcaption>方块是平均值，圆点是中位数。两者距离越大，说明少数非常差的帧对平均值影响越大。Fanbo7 中就出现了“平均值看起来 11L 更好，但中位数并没有更好”的情况。</figcaption></figure></section>
<section id="limits"><h2>6. 限制与下一步</h2><ul><li>横轴是双目重建得到的<strong>估计光轴深度</strong>，不是卷尺测得的独立距离。</li><li>不同距离来自不同序列，动作、视角和遮挡没有完全控制，所以这里只能说“有关联”，不能说“距离造成了全部误差”。</li><li>Xsens 只是 external comparison system / Xsens-derived reference，不是绝对 Ground Truth。</li><li>下一次可在 2.0–4.5 m 每 0.5 m 固定站位、重复相同动作，再画同样四张图；那时才适合决定是否按距离切换模型。</li></ul><p>本地已保存逐帧 CSV、分箱统计、manifest 和所有图。恢复 GPU 后再补 TensorRT/TAO 的真实速度测试，不会与本次统计混在一起。</p></section>"""
    else:
        title = "Distance, Detector Backbone, and Stereo Angle Disagreement"
        subtitle = (
            "YOLOv8m vs YOLO11L · same valid frames · fixed alignment · "
            "Xsens-derived reference used only for external comparison"
        )
        nav = ["Core chart", "Detailed scatter", "Box plots", "Paired comparison", "Mean vs median", "Limitations"]
        body = f"""
<section id="summary"><h2>1. Core chart: how does disagreement change with distance?</h2><div class="cards"><div class="card"><div class="metric">{eight['median_deg']:.2f}°</div>YOLOv8m right-shoulder median</div><div class="card"><div class="metric">{eleven['median_deg']:.2f}°</div>YOLO11L right-shoulder median</div><div class="card"><div class="metric">{bins_favouring_11l}/{bin_count}</div>eligible distance bins with a lower YOLO11L median</div></div><figure><img src="{figures['core']}"><figcaption><strong>Primary metric:</strong> the right shoulder has an available reference in all four distance records and substantially lower disagreement than the elbow. The curve shows median absolute angular disagreement in each 0.5 m distance bin; bands show P25–P75.</figcaption></figure><figure><img src="{figures['hip']}"><figcaption><strong>Validation metric:</strong> each frame aggregates the left- and right-hip absolute disagreements before distance-bin statistics are calculated. A distance interpretation is more credible only if shoulder and hip trends agree.</figcaption></figure><p>The two proximal-joint metrics do not show a consistent smooth monotonic increase: the right shoulder does not deteriorate at the farthest distance, while the bilateral-hip metric generally decreases. These recordings therefore cannot establish that angular disagreement increases with distance; action, viewpoint, occlusion, and between-session differences remain confounded with distance. Right-elbow results are no longer used as the main evidence.</p><div class="callout">Both models are evaluated on the same valid frames with one fixed offset per session. The YOLOv8m torso-depth estimate is used as their common horizontal coordinate. Each bin used for conclusions contains at least 20 common valid frames. Fanbo7 has no usable left-shoulder reference, so the primary metric is fixed to the right shoulder, which is available at every distance, instead of mixing unilateral and bilateral metrics. The table below reports the session-level right-shoulder metric.</div>{table}</section>
<section id="scatter"><h2>2. Detailed right-shoulder scatter view</h2><figure><img src="{figures['scatter']}"><figcaption>Each point is one frame's right-shoulder absolute disagreement. The thick line is the 0.5 m-bin median and the band is P25–P75. The broad scatter indicates that factors beyond estimated depth matter, but this dataset does not isolate viewpoint, action, and occlusion effects.</figcaption></figure><figure><img src="{figures['session']}"><figcaption>Session medians and P95 values still do not form a smooth increasing trend with distance.</figcaption></figure></section>
<section id="box"><h2>3. Box plots: is the distance trend stable?</h2><figure><img src="{figures['box']}"><figcaption><strong>Primary right-shoulder metric:</strong> boxes show the middle 50% of frames, the centre line is the median, and dots show outliers. Strong overlap between adjacent distance bins means that distance alone cannot explain the disagreement.</figcaption></figure><figure><img src="{figures['hip_box']}"><figcaption><strong>Bilateral-hip validation metric:</strong> this is compared with the right-shoulder distribution. A distance trend is more credible only if both proximal-joint distributions move similarly.</figcaption></figure></section>
<section id="paired"><h2>4. Same-frame detector comparison</h2><figure><img src="{figures['paired']}"><figcaption>The vertical axis is YOLO11L absolute disagreement minus YOLOv8m absolute disagreement. Only negative values favour YOLO11L; error bars show P25–P75. No reliable near/far switching threshold appears.</figcaption></figure><p>The current engineering recommendation is therefore to retain YOLOv8m. A dynamic switch would require controlled recordings at additional distances with the same action and viewpoint, followed by prospective validation on held-out sequences.</p></section>
<section id="mean"><h2>5. Why median matters</h2><figure><img src="{figures['mean_median']}"><figcaption>Squares show means and circles show medians. A large gap indicates that a small number of severe failures pulls the mean upward. Fanbo7 includes the specific case where the mean appears to favour YOLO11L while the median does not.</figcaption></figure></section>
<section id="limits"><h2>6. Limitations and next step</h2><ul><li>The horizontal coordinate is estimated optical depth from stereo reconstruction, not an independent tape-measure distance.</li><li>Distance is confounded with session, action, viewpoint, and occlusion; the evidence is associative, not causal.</li><li>Xsens is an external comparison system / Xsens-derived reference, not absolute Ground Truth.</li><li>A controlled 2.0–4.5 m experiment at 0.5 m intervals should repeat the same action before defining a switching threshold.</li></ul><p>Per-frame CSV data, summaries, a source manifest, and all figures are stored locally. TensorRT/TAO throughput tests will be added only after NVIDIA GPU access is restored.</p></section>"""
    alt_texts = {
        "core": "角度误差随距离变化的核心曲线"
        if chinese
        else "Core median angular disagreement versus distance curve",
        "hip": "髋部角度误差随距离变化的验证曲线"
        if chinese
        else "Hip angular disagreement versus distance validation curve",
        "scatter": "逐帧角度差异与估计深度散点图"
        if chinese
        else "Per-frame angular disagreement versus estimated depth",
        "session": "各序列中位数与P95对比图"
        if chinese
        else "Session median and P95 comparison",
        "box": "按距离分箱的角度差异箱线图"
        if chinese
        else "Angular disagreement box plots by depth bin",
        "hip_box": "按距离分箱的髋部角度差异箱线图"
        if chinese
        else "Hip angular disagreement box plots by depth bin",
        "paired": "YOLO11L与YOLOv8m同帧差值图"
        if chinese
        else "Paired same-frame YOLO11L versus YOLOv8m differences",
        "mean_median": "平均值与中位数对比图"
        if chinese
        else "Mean versus median comparison",
    }
    for key, alt_text in alt_texts.items():
        body = body.replace(
            f'<img src="{figures[key]}">',
            f'<img src="{figures[key]}" alt="{alt_text}">',
        )
    section_ids = ["summary", "scatter", "box", "paired", "mean", "limits"]
    links = "".join(
        f'<a href="#{section_id}">{label}</a>'
        for section_id, label in zip(section_ids, nav)
    )
    language = "zh-CN" if chinese else "en"
    footer = "本报告为自包含 HTML，图表已内嵌。" if chinese else "Self-contained HTML report with embedded figures."
    return f'<!doctype html><html lang="{language}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{STYLE}</style></head><body><header><h1>{title}</h1><p>{subtitle}</p></header><div class="layout"><nav><strong>{"目录" if chinese else "Contents"}</strong>{links}</nav><main>{body}<footer>{footer}</footer></main></div></body></html>'


def write_outputs(config_path: Path) -> Path:
    """Run the complete local analysis and write auditable outputs."""
    payload, sessions = load_analysis_config(config_path)
    analysis_cfg = payload.get("analysis", {})
    primary_joints = tuple(
        str(name)
        for name in analysis_cfg.get(
            "primary_joints", ["LeftShoulder", "RightShoulder"]
        )
    )
    validation_joints = tuple(
        str(name)
        for name in analysis_cfg.get(
            "validation_joints", ["LeftHip", "RightHip"]
        )
    )
    if len(primary_joints) not in (1, 2) or len(validation_joints) not in (1, 2):
        raise ValueError("Distance-analysis targets require one or two joints")
    width_m = float(analysis_cfg.get("distance_bin_width_m", 0.5))
    min_bin_count = int(analysis_cfg.get("min_common_frames_per_bin", 20))
    output_dir = _resolve(
        analysis_cfg.get(
            "output_dir", "00_pose_pipeline_v2/runs/distance_error_analysis"
        )
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame_data, sources = extract_joint_median_rows(
        sessions,
        primary_joints,
        "RightShoulder",
    )
    validation_frame_data, _ = extract_joint_median_rows(
        sessions,
        validation_joints,
        "BilateralHipMedian",
    )
    frame_data = add_distance_bins(frame_data, width_m)
    validation_frame_data = add_distance_bins(validation_frame_data, width_m)
    session_summary, bin_summary, paired_summary, extras = summarize_data(frame_data)
    (
        validation_session_summary,
        validation_bin_summary,
        validation_paired_summary,
        validation_extras,
    ) = summarize_data(validation_frame_data)
    valid = frame_data[frame_data["common_valid"]].copy()

    paired_summary["meets_minimum_count"] = (
        paired_summary["n_common_frames"] >= min_bin_count
    )
    validation_paired_summary["meets_minimum_count"] = (
        validation_paired_summary["n_common_frames"] >= min_bin_count
    )
    frame_data.to_csv(output_dir / "per_frame_distance_angle_data.csv", index=False)
    validation_frame_data.to_csv(
        output_dir / "per_frame_distance_hip_validation_data.csv", index=False
    )
    session_summary.to_csv(output_dir / "session_summary.csv", index=False)
    validation_session_summary.to_csv(
        output_dir / "hip_validation_session_summary.csv", index=False
    )
    bin_summary.to_csv(output_dir / "distance_bin_summary.csv", index=False)
    validation_bin_summary.to_csv(
        output_dir / "hip_validation_distance_bin_summary.csv", index=False
    )
    paired_summary.to_csv(output_dir / "paired_model_summary.csv", index=False)
    extras["paired_frames"].to_csv(output_dir / "paired_frame_differences.csv", index=False)

    eligible_paired = paired_summary[paired_summary["meets_minimum_count"]].copy()
    eligible_labels = set(eligible_paired["distance_bin"].astype(str))
    eligible_bin_summary = bin_summary[
        bin_summary["distance_bin"].astype(str).isin(eligible_labels)
    ].copy()
    plot_valid = valid[valid["distance_bin"].astype(str).isin(eligible_labels)].copy()
    eligible_validation_paired = validation_paired_summary[
        validation_paired_summary["meets_minimum_count"]
    ].copy()
    eligible_validation_labels = set(
        eligible_validation_paired["distance_bin"].astype(str)
    )
    eligible_validation_bin_summary = validation_bin_summary[
        validation_bin_summary["distance_bin"]
        .astype(str)
        .isin(eligible_validation_labels)
    ].copy()
    validation_valid = validation_frame_data[
        validation_frame_data["common_valid"]
    ].copy()
    validation_plot_valid = validation_valid[
        validation_valid["distance_bin"].astype(str).isin(
            eligible_validation_labels
        )
    ].copy()

    figure_paths = {
        "core": figures_dir / "00_core_median_error_vs_distance.png",
        "hip": figures_dir / "00b_hip_validation_error_vs_distance.png",
        "scatter": figures_dir / "01_scatter_depth_error.png",
        "box": figures_dir / "02_boxplot_distance_bins.png",
        "hip_box": figures_dir / "02b_hip_boxplot_distance_bins.png",
        "session": figures_dir / "03_session_median_p95.png",
        "paired": figures_dir / "04_paired_model_difference.png",
        "mean_median": figures_dir / "05_mean_vs_median.png",
    }
    for chinese, suffix in ((False, ""), (True, "_CN")):
        plot_core_distance_curve(
            eligible_bin_summary,
            figure_paths["core"].with_name(
                f"00_core_median_error_vs_distance{suffix}.png"
            ),
            chinese,
            "shoulder",
        )
        plot_core_distance_curve(
            eligible_validation_bin_summary,
            figure_paths["hip"].with_name(
                f"00b_hip_validation_error_vs_distance{suffix}.png"
            ),
            chinese,
            "hip",
        )
        plot_scatter(
            valid,
            eligible_bin_summary,
            figure_paths["scatter"].with_name(
                f"01_scatter_depth_error{suffix}.png"
            ),
            chinese,
        )
        plot_boxplots(
            plot_valid,
            figure_paths["box"].with_name(
                f"02_boxplot_distance_bins{suffix}.png"
            ),
            chinese,
            "shoulder",
        )
        plot_boxplots(
            validation_plot_valid,
            figure_paths["hip_box"].with_name(
                f"02b_hip_boxplot_distance_bins{suffix}.png"
            ),
            chinese,
            "hip",
        )
        plot_session_summary(
            session_summary,
            figure_paths["session"].with_name(
                f"03_session_median_p95{suffix}.png"
            ),
            chinese,
        )
        plot_paired_difference(
            eligible_paired,
            figure_paths["paired"].with_name(
                f"04_paired_model_difference{suffix}.png"
            ),
            chinese,
        )
        plot_mean_median(
            session_summary,
            figure_paths["mean_median"].with_name(
                f"05_mean_vs_median{suffix}.png"
            ),
            chinese,
        )

    thesis_figure_dir_value = analysis_cfg.get("thesis_figure_dir")
    if thesis_figure_dir_value:
        thesis_figure_dir = _resolve(thesis_figure_dir_value)
        thesis_figure_dir.mkdir(parents=True, exist_ok=True)
        thesis_names = {
            "core": "thesis_detector_distance_core_curve.png",
            "hip": "thesis_detector_distance_hip_validation.png",
            "scatter": "thesis_detector_distance_scatter.png",
            "box": "thesis_detector_distance_boxplot.png",
            "hip_box": "thesis_detector_distance_hip_boxplot.png",
            "paired": "thesis_detector_paired_difference.png",
            "mean_median": "thesis_detector_mean_vs_median.png",
        }
        for key, target_name in thesis_names.items():
            shutil.copyfile(figure_paths[key], thesis_figure_dir / target_name)

    overall = extras["overall"]
    manifest = {
        "analysis_name": analysis_cfg.get("title", "Distance-stratified detector comparison"),
        "created_from_commit": _git_commit(),
        "analysis_config": str(config_path.relative_to(PROJECT_ROOT)),
        "analysis_config_sha256": _sha256(config_path),
        "primary_metric": {
            "aggregation": (
                "single-joint absolute disagreement"
                if len(primary_joints) == 1
                else "per-frame bilateral median absolute disagreement"
            ),
            "joints": list(primary_joints),
        },
        "validation_metric": {
            "aggregation": (
                "single-joint absolute disagreement"
                if len(validation_joints) == 1
                else "per-frame bilateral median absolute disagreement"
            ),
            "joints": list(validation_joints),
        },
        "reference": analysis_cfg.get("reference_label", "Xsens-derived reference"),
        "distance_source": analysis_cfg.get("distance_source"),
        "distance_bin_width_m": width_m,
        "min_common_frames_per_bin": min_bin_count,
        "comparison_rule": (
            "Same valid frames, same fixed per-session offset, common "
            "YOLOv8m-estimated optical depth."
        ),
        "overall": overall,
        "validation_overall": validation_extras["overall"],
        "sources": sources,
        "limitations": [
            "Estimated optical depth is not an independent physical distance measurement.",
            "Distance is confounded with session, action, viewpoint, and occlusion.",
            "Frame-level observations are temporally autocorrelated.",
            "Fanbo7 has no usable left-shoulder reference; the primary metric is consistently right shoulder.",
            "Xsens is an external comparison system, not absolute Ground Truth.",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report_CN.html").write_text(
        build_report(True, session_summary, eligible_paired, overall, figure_paths),
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(
        build_report(False, session_summary, eligible_paired, overall, figure_paths),
        encoding="utf-8",
    )
    readme = f"""# Distance-stratified detector analysis

主结论：在 {overall['common_valid_unique_frames']} 个共同有效帧中，YOLOv8m 与
YOLO11L 的总体右肩中位绝对差分别为
{overall['models']['YOLOv8m']['median_deg']:.2f}° 和
{overall['models']['YOLO11L']['median_deg']:.2f}°。双髋指标作为独立的趋势验证。

- `report_CN.html` / `report.html`: 中英文图文报告
- `per_frame_distance_angle_data.csv`: 右肩主指标逐帧数据，可直接用 Excel 打开
- `per_frame_distance_hip_validation_data.csv`: 双髋验证指标逐帧数据
- `session_summary.csv`: 右肩主指标序列级 median、IQR、P95 和有效率
- `hip_validation_session_summary.csv`: 双髋验证指标序列级统计
- `distance_bin_summary.csv`: 0.5 m 分箱统计
- `hip_validation_distance_bin_summary.csv`: 双髋验证指标分箱统计
- `paired_model_summary.csv`: 同一帧 YOLO11L - YOLOv8m 成对统计
- `analysis_manifest.json`: 配置、commit、输入和标定 SHA256
- `figures/`: 中英文散点图、箱线图及成对比较图

Reproduce:

```bash
/opt/anaconda3/envs/pose/bin/python \
  00_pose_pipeline_v2/src/analyze_error_vs_distance.py \
  --config 00_pose_pipeline_v2/configs/distance_error_analysis.yaml
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(
        f"[distance-analysis] frames={len(frame_data)} "
        f"common_unique_frames={overall['common_valid_unique_frames']}"
    )
    print(f"[distance-analysis] output={output_dir}")
    return output_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("00_pose_pipeline_v2/configs/distance_error_analysis.yaml"),
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    write_outputs(config_path)


if __name__ == "__main__":
    main()
