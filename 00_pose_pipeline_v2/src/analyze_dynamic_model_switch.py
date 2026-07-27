"""Evaluate whether an explainable YOLOv8m/YOLO11L switch is learnable.

The analysis uses paired V2 stereo reconstructions, aggregates all available
semantic joint angles instead of selecting one body part, and evaluates simple
pixel-scale and distance thresholds with leave-one-capture-group-out
validation. Xsens-derived angles are used only as an external comparison
system and never as a deployment-time input.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from analyze_error_vs_distance import (  # noqa: E402
    PROJECT_ROOT,
    STYLE,
    _configure_plot_style,
    _git_commit,
    _image_data_uri,
    _save_figure,
    _sha256,
    _torso_distance,
)
from common.angles import SEMANTIC_ANGLE_NAMES  # noqa: E402
from common.config import load_config  # noqa: E402
from common.dataset import load_method_keypoints  # noqa: E402
from eval_angles import prepare_angles  # noqa: E402


@dataclass(frozen=True)
class PairSpec:
    """One paired detector result set on the same stereo timeline."""

    name: str
    session_id: str
    validation_group: str
    base_config: Path
    fixed_offset_seconds: float
    yolov8m_npz: Path
    yolo11l_npz: Path


def _resolve(value: str | Path) -> Path:
    """Resolve one project-relative path."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_switch_config(path: Path) -> tuple[dict[str, Any], list[PairSpec]]:
    """Load and validate the dynamic-switch analysis configuration."""
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    pairs: list[PairSpec] = []
    for raw in payload.get("pairs", []):
        pair = PairSpec(
            name=str(raw["name"]),
            session_id=str(raw["session_id"]),
            validation_group=str(raw["validation_group"]),
            base_config=_resolve(raw["base_config"]),
            fixed_offset_seconds=float(raw["fixed_offset_seconds"]),
            yolov8m_npz=_resolve(raw["yolov8m_npz"]),
            yolo11l_npz=_resolve(raw["yolo11l_npz"]),
        )
        for required in (
            pair.base_config,
            pair.yolov8m_npz,
            pair.yolo11l_npz,
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        pairs.append(pair)
    if not pairs:
        raise ValueError("No paired detector results are configured")
    return payload, pairs


def _config_for_npz(config_path: Path, npz_path: Path) -> dict[str, Any]:
    """Load a dataset config and redirect its SKT input to one fixed NPZ."""
    config = load_config(config_path)
    config.setdefault("skt", {})["use_existing_npz"] = True
    config["skt"]["existing_npz"] = str(npz_path)
    config.setdefault("evaluation", {})["angle_names"] = list(SEMANTIC_ANGLE_NAMES)
    return config


def _load_model_result(
    pair: PairSpec, npz_path: Path
) -> dict[str, Any]:
    """Load one detector's angles, keypoints, timeline, and raw 2D arrays."""
    config = _config_for_npz(pair.base_config, npz_path)
    time_s, all_angles, info = prepare_angles(
        config,
        npz_path.parent,
        pair.fixed_offset_seconds,
    )
    key_time_s, _, methods = load_method_keypoints(config, npz_path.parent)
    if len(time_s) != len(key_time_s) or not np.allclose(
        time_s, key_time_s, atol=1e-7
    ):
        raise RuntimeError(f"Timeline mismatch while loading {npz_path}")
    with np.load(npz_path, allow_pickle=True) as payload:
        files = set(payload.files)
        left_2d_name = (
            "keypoints_left_2d"
            if "keypoints_left_2d" in files
            else "keypoints_left_2d_raw"
        )
        left_2d = np.asarray(payload[left_2d_name], dtype=np.float64)[: len(time_s)]
        confidence = (
            np.asarray(payload["conf_left"], dtype=np.float64)[: len(time_s)]
            if "conf_left" in files
            else np.full(left_2d.shape[:2], np.nan, dtype=np.float64)
        )
        model_name = (
            str(payload["model_name"].item())
            if "model_name" in files
            else "unknown"
        )
    return {
        "config": config,
        "time_s": np.asarray(time_s, dtype=np.float64),
        "angles": all_angles["SKT"],
        "reference": all_angles["XsensFair"],
        "keypoints_3d": np.asarray(methods["SKT"], dtype=np.float64),
        "keypoints_left_2d": left_2d,
        "conf_left": confidence,
        "angle_info": info,
        "model_name": model_name,
    }


def _finite_row_median(values: np.ndarray, minimum_count: int) -> np.ndarray:
    """Return row medians only when enough finite values are present."""
    array = np.asarray(values, dtype=np.float64)
    output = np.full(array.shape[0], np.nan, dtype=np.float64)
    for index, row in enumerate(array):
        finite = row[np.isfinite(row)]
        if len(finite) >= minimum_count:
            output[index] = float(np.median(finite))
    return output


def skeleton_pixel_height(
    keypoints_2d: np.ndarray,
    confidence: np.ndarray,
    minimum_joints: int = 6,
    minimum_confidence: float = 0.2,
) -> np.ndarray:
    """Estimate person pixel scale from the robust vertical keypoint span."""
    points = np.asarray(keypoints_2d, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    output = np.full(len(points), np.nan, dtype=np.float64)
    for frame, pose in enumerate(points):
        valid = np.isfinite(pose[:, :2]).all(axis=1)
        if conf.shape == points.shape[:2] and np.isfinite(conf[frame]).any():
            valid &= np.isfinite(conf[frame]) & (conf[frame] >= minimum_confidence)
        y = pose[valid, 1]
        if len(y) >= minimum_joints:
            output[frame] = float(np.percentile(y, 95) - np.percentile(y, 5))
    return output


def reference_motion_speed(
    time_s: np.ndarray,
    reference: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute body-wide median absolute angular speed in degrees per second."""
    speed_columns: list[np.ndarray] = []
    for name in SEMANTIC_ANGLE_NAMES:
        values = np.asarray(reference[name], dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(time_s)
        speed = np.full(len(values), np.nan, dtype=np.float64)
        if int(valid.sum()) >= 3:
            interpolated = np.interp(time_s, time_s[valid], values[valid])
            gradient = np.abs(np.gradient(interpolated, time_s))
            within = (time_s >= time_s[valid][0]) & (time_s <= time_s[valid][-1])
            speed[within] = gradient[within]
        speed_columns.append(speed)
    return _finite_row_median(np.column_stack(speed_columns), 2)


def build_frame_table(
    pair: PairSpec,
    minimum_common_angles: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build paired frame-level model errors and routing features."""
    eight = _load_model_result(pair, pair.yolov8m_npz)
    eleven = _load_model_result(pair, pair.yolo11l_npz)
    if len(eight["time_s"]) != len(eleven["time_s"]) or not np.allclose(
        eight["time_s"], eleven["time_s"], atol=1e-7
    ):
        raise RuntimeError(f"Detector timelines differ for {pair.name}")

    error_eight: list[np.ndarray] = []
    error_eleven: list[np.ndarray] = []
    common_columns: list[np.ndarray] = []
    reference_columns: list[np.ndarray] = []
    for name in SEMANTIC_ANGLE_NAMES:
        ref_eight = np.asarray(eight["reference"][name], dtype=np.float64)
        ref_eleven = np.asarray(eleven["reference"][name], dtype=np.float64)
        shared_ref = np.isfinite(ref_eight) & np.isfinite(ref_eleven)
        if np.any(shared_ref):
            difference = np.nanmax(np.abs(ref_eight[shared_ref] - ref_eleven[shared_ref]))
            if float(difference) > 1e-5:
                raise RuntimeError(
                    f"Reference differs between paired models for {pair.name}: {name}"
                )
        angle_eight = np.asarray(eight["angles"][name], dtype=np.float64)
        angle_eleven = np.asarray(eleven["angles"][name], dtype=np.float64)
        common = (
            np.isfinite(angle_eight)
            & np.isfinite(angle_eleven)
            & np.isfinite(ref_eight)
            & np.isfinite(ref_eleven)
        )
        error_eight.append(np.where(common, np.abs(angle_eight - ref_eight), np.nan))
        error_eleven.append(
            np.where(common, np.abs(angle_eleven - ref_eleven), np.nan)
        )
        common_columns.append(common)
        reference_columns.append(ref_eight)

    error_eight_matrix = np.column_stack(error_eight)
    error_eleven_matrix = np.column_stack(error_eleven)
    common_matrix = np.column_stack(common_columns)
    common_count = common_matrix.sum(axis=1)
    aggregate_eight = _finite_row_median(
        error_eight_matrix, minimum_common_angles
    )
    aggregate_eleven = _finite_row_median(
        error_eleven_matrix, minimum_common_angles
    )
    depth_m, radial_range_m = _torso_distance(eight["keypoints_3d"])
    pixel_height = skeleton_pixel_height(
        eight["keypoints_left_2d"], eight["conf_left"]
    )
    reference = {
        name: reference_columns[index]
        for index, name in enumerate(SEMANTIC_ANGLE_NAMES)
    }
    motion_speed = reference_motion_speed(eight["time_s"], reference)
    table = pd.DataFrame(
        {
            "session": pair.name,
            "session_id": pair.session_id,
            "validation_group": pair.validation_group,
            "frame": np.arange(len(eight["time_s"]), dtype=int),
            "time_s": eight["time_s"],
            "error_yolov8m_deg": aggregate_eight,
            "error_yolo11l_deg": aggregate_eleven,
            "delta_11l_minus_8m_deg": aggregate_eleven - aggregate_eight,
            "common_angle_count": common_count,
            "optical_depth_m": depth_m,
            "radial_range_m": radial_range_m,
            "skeleton_height_px": pixel_height,
            "reference_motion_deg_s": motion_speed,
        }
    )
    source = {
        "session": pair.name,
        "session_id": pair.session_id,
        "validation_group": pair.validation_group,
        "base_config": str(pair.base_config.relative_to(PROJECT_ROOT)),
        "base_config_sha256": _sha256(pair.base_config),
        "fixed_offset_seconds": pair.fixed_offset_seconds,
        "yolov8m_npz": str(pair.yolov8m_npz.relative_to(PROJECT_ROOT)),
        "yolov8m_npz_sha256": _sha256(pair.yolov8m_npz),
        "yolov8m_model_name": eight["model_name"],
        "yolo11l_npz": str(pair.yolo11l_npz.relative_to(PROJECT_ROOT)),
        "yolo11l_npz_sha256": _sha256(pair.yolo11l_npz),
        "yolo11l_model_name": eleven["model_name"],
        "frame_count": int(len(table)),
    }
    return table, source


def motion_class(
    speed: pd.Series,
    thresholds: tuple[float, float],
) -> pd.Categorical:
    """Classify reference-derived angular speed into fixed interpretable bins."""
    labels = ["static", "slow", "fast"]
    return pd.cut(
        speed,
        bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
        labels=labels,
        right=False,
    )


def build_window_table(
    frame_data: pd.DataFrame,
    window_seconds: float,
    minimum_valid_frames: int,
    motion_thresholds: tuple[float, float],
) -> pd.DataFrame:
    """Aggregate temporally correlated frames into non-overlapping windows."""
    data = frame_data.copy()
    data["window_index"] = np.floor(
        (data["time_s"] - data.groupby("session_id")["time_s"].transform("min"))
        / window_seconds
    ).astype(int)
    rows: list[dict[str, Any]] = []
    group_columns = [
        "session",
        "session_id",
        "validation_group",
        "window_index",
    ]
    for keys, group in data.groupby(group_columns, sort=False):
        valid = (
            np.isfinite(group["error_yolov8m_deg"])
            & np.isfinite(group["error_yolo11l_deg"])
            & np.isfinite(group["optical_depth_m"])
            & np.isfinite(group["skeleton_height_px"])
            & np.isfinite(group["reference_motion_deg_s"])
        )
        selected = group[valid]
        if len(selected) < minimum_valid_frames:
            continue
        error_eight = float(np.median(selected["error_yolov8m_deg"]))
        error_eleven = float(np.median(selected["error_yolo11l_deg"]))
        rows.append(
            {
                "session": keys[0],
                "session_id": keys[1],
                "validation_group": keys[2],
                "window_index": int(keys[3]),
                "start_time_s": float(selected["time_s"].min()),
                "end_time_s": float(selected["time_s"].max()),
                "n_valid_frames": int(len(selected)),
                "error_yolov8m_deg": error_eight,
                "error_yolo11l_deg": error_eleven,
                "oracle_error_deg": min(error_eight, error_eleven),
                "delta_11l_minus_8m_deg": error_eleven - error_eight,
                "optical_depth_m": float(np.median(selected["optical_depth_m"])),
                "skeleton_height_px": float(
                    np.median(selected["skeleton_height_px"])
                ),
                "reference_motion_deg_s": float(
                    np.median(selected["reference_motion_deg_s"])
                ),
                "common_angle_count_median": float(
                    np.median(selected["common_angle_count"])
                ),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("No analysis windows passed the validity gate")
    output["motion_class"] = motion_class(
        output["reference_motion_deg_s"], motion_thresholds
    )
    return output


def macro_session_error(data: pd.DataFrame, column: str) -> float:
    """Return the equal-session mean of session-level median errors."""
    return float(data.groupby("session_id", observed=True)[column].median().mean())


def _threshold_candidates(values: pd.Series) -> np.ndarray:
    """Create a compact deterministic threshold grid from training values."""
    finite = values[np.isfinite(values)].to_numpy(dtype=np.float64)
    if len(finite) < 5:
        raise ValueError("Too few finite values for threshold search")
    candidates = np.unique(np.quantile(finite, np.linspace(0.05, 0.95, 37)))
    return candidates.astype(np.float64)


def apply_threshold(
    data: pd.DataFrame,
    feature: str,
    threshold: float,
    direction: str,
) -> np.ndarray:
    """Apply one explainable switch and return selected window errors."""
    if direction == "high_uses_11l":
        use_eleven = data[feature].to_numpy(dtype=float) >= threshold
    elif direction == "low_uses_11l":
        use_eleven = data[feature].to_numpy(dtype=float) <= threshold
    else:
        raise ValueError(f"Unsupported threshold direction: {direction}")
    return np.where(
        use_eleven,
        data["error_yolo11l_deg"].to_numpy(dtype=float),
        data["error_yolov8m_deg"].to_numpy(dtype=float),
    )


def fit_threshold(
    training: pd.DataFrame,
    feature: str,
    direction: str,
) -> tuple[float, float]:
    """Fit one threshold by equal-session macro median disagreement."""
    best_threshold = math.nan
    best_score = math.inf
    for threshold in _threshold_candidates(training[feature]):
        candidate = training.copy()
        candidate["candidate_error_deg"] = apply_threshold(
            candidate, feature, float(threshold), direction
        )
        score = macro_session_error(candidate, "candidate_error_deg")
        if score < best_score:
            best_threshold = float(threshold)
            best_score = score
    return best_threshold, best_score


def leave_group_out_switch(
    windows: pd.DataFrame,
    feature: str,
    direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit thresholds on other capture groups and predict each held-out group."""
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for held_out in windows["validation_group"].drop_duplicates():
        training = windows[windows["validation_group"] != held_out].copy()
        testing = windows[windows["validation_group"] == held_out].copy()
        threshold, training_score = fit_threshold(training, feature, direction)
        testing["selected_error_deg"] = apply_threshold(
            testing, feature, threshold, direction
        )
        testing["held_out_group"] = held_out
        testing["learned_threshold"] = threshold
        predictions.append(testing)
        base_error = macro_session_error(testing, "error_yolov8m_deg")
        selected_error = macro_session_error(testing, "selected_error_deg")
        fold_rows.append(
            {
                "held_out_group": held_out,
                "n_windows": int(len(testing)),
                "learned_threshold": threshold,
                "training_macro_error_deg": training_score,
                "held_out_yolov8m_error_deg": base_error,
                "held_out_yolo11l_error_deg": macro_session_error(
                    testing, "error_yolo11l_deg"
                ),
                "held_out_selected_error_deg": selected_error,
                "held_out_oracle_error_deg": macro_session_error(
                    testing, "oracle_error_deg"
                ),
                "selected_minus_yolov8m_deg": selected_error - base_error,
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows)


def summarize_methods(
    windows: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    distance_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize fixed, oracle, and cross-validated switch policies."""
    rows = [
        {
            "method": "Always YOLOv8m",
            "macro_median_error_deg": macro_session_error(
                windows, "error_yolov8m_deg"
            ),
        },
        {
            "method": "Always YOLO11L",
            "macro_median_error_deg": macro_session_error(
                windows, "error_yolo11l_deg"
            ),
        },
        {
            "method": "Oracle per window",
            "macro_median_error_deg": macro_session_error(
                windows, "oracle_error_deg"
            ),
        },
        {
            "method": "Pixel-threshold LOSO",
            "macro_median_error_deg": macro_session_error(
                pixel_predictions, "selected_error_deg"
            ),
        },
        {
            "method": "Distance-threshold LOSO",
            "macro_median_error_deg": macro_session_error(
                distance_predictions, "selected_error_deg"
            ),
        },
    ]
    summary = pd.DataFrame(rows)
    baseline = float(
        summary.loc[
            summary["method"] == "Always YOLOv8m", "macro_median_error_deg"
        ].iloc[0]
    )
    summary["gain_vs_yolov8m_percent"] = (
        100.0 * (baseline - summary["macro_median_error_deg"]) / baseline
    )
    return summary


def _boxplot_by_bins(
    windows: pd.DataFrame,
    bin_column: str,
    labels: list[str],
    path: Path,
    chinese: bool,
    title: str,
    xlabel: str,
) -> None:
    """Plot paired model advantage distributions in ordered feature bins."""
    _configure_plot_style(chinese)
    groups = [
        windows.loc[
            windows[bin_column].astype(object).map(str) == label,
            "delta_11l_minus_8m_deg",
        ].dropna()
        for label in labels
    ]
    fig, axis = plt.subplots(figsize=(11.2, 5.2))
    boxes = axis.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.25},
        medianprops={"color": "#111827", "linewidth": 1.7},
    )
    for box in boxes["boxes"]:
        box.set_facecolor("#7C9CF5")
        box.set_alpha(0.68)
    axis.axhline(0.0, color="#B42318", linewidth=1.4, linestyle="--")
    axis.set_ylabel(
        "11L误差 − 8m误差（°）"
        if chinese
        else "YOLO11L error minus YOLOv8m error (deg)"
    )
    axis.set_xlabel(xlabel)
    axis.set_title(title, fontweight="bold")
    axis.text(
        0.99,
        0.97,
        "零线以下：11L更好；零线以上：8m更好"
        if chinese
        else "Below zero: YOLO11L is better; above zero: YOLOv8m is better",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#5D6B7A",
    )
    fig.tight_layout()
    _save_figure(fig, path)


def prepare_plot_bins(
    windows: pd.DataFrame,
    pixel_bin_count: int,
    distance_width_m: float,
    minimum_windows_per_bin: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Attach ordered pixel-quantile and fixed-width distance bins."""
    data = windows.copy()
    data["pixel_bin"] = pd.qcut(
        data["skeleton_height_px"],
        q=pixel_bin_count,
        duplicates="drop",
    )
    pixel_counts = data.groupby("pixel_bin", observed=True).size()
    pixel_labels = [
        str(value)
        for value in data["pixel_bin"].cat.categories
        if int(pixel_counts.get(value, 0)) >= minimum_windows_per_bin
    ]
    minimum = math.floor(data["optical_depth_m"].min() / distance_width_m)
    maximum = math.ceil(data["optical_depth_m"].max() / distance_width_m)
    edges = np.arange(minimum, maximum + 1.0) * distance_width_m
    distance_labels = [
        f"{left:.1f}–{right:.1f}" for left, right in zip(edges[:-1], edges[1:])
    ]
    data["distance_bin"] = pd.cut(
        data["optical_depth_m"],
        bins=edges,
        labels=distance_labels,
        right=False,
    )
    used_distance_labels = [
        label
        for label in distance_labels
        if int(
            (
                data["distance_bin"].astype(object).map(str) == label
            ).sum()
        )
        >= minimum_windows_per_bin
    ]
    return data, pixel_labels, used_distance_labels


def plot_motion_heatmap(
    windows: pd.DataFrame,
    distance_labels: list[str],
    path: Path,
    chinese: bool,
    minimum_windows_per_cell: int,
) -> None:
    """Plot median model advantage by distance and motion class."""
    _configure_plot_style(chinese)
    classes = ["static", "slow", "fast"]
    pivot = windows.pivot_table(
        index="distance_bin",
        columns="motion_class",
        values="delta_11l_minus_8m_deg",
        aggfunc="median",
        observed=True,
    ).reindex(index=distance_labels, columns=classes)
    counts = windows.pivot_table(
        index="distance_bin",
        columns="motion_class",
        values="delta_11l_minus_8m_deg",
        aggfunc="count",
        observed=True,
    ).reindex(index=distance_labels, columns=classes)
    values = pivot.to_numpy(dtype=float)
    count_values = counts.fillna(0).to_numpy(dtype=int)
    values[count_values < minimum_windows_per_cell] = np.nan
    finite = np.abs(values[np.isfinite(values)])
    limit = max(2.0, float(np.percentile(finite, 90))) if finite.size else 2.0
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    image = axis.imshow(
        values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(len(classes)),
        ["静止", "慢速", "快速"] if chinese else ["Static", "Slow", "Fast"],
    )
    axis.set_yticks(np.arange(len(distance_labels)), distance_labels)
    axis.set_xlabel("运动等级" if chinese else "Motion class")
    axis.set_ylabel("估计距离（m）" if chinese else "Estimated distance (m)")
    axis.set_title(
        "距离 × 运动：模型优势中位值"
        if chinese
        else "Distance × motion: median model advantage",
        fontweight="bold",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            count = int(count_values[row, column])
            text = (
                f"—\n(n={count})"
                if not np.isfinite(value)
                else f"{value:.1f}°\n(n={count})"
            )
            axis.text(column, row, text, ha="center", va="center", fontsize=9)
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(
        "11L − 8m（负值偏向11L）"
        if chinese
        else "YOLO11L − YOLOv8m (negative favours YOLO11L)"
    )
    fig.tight_layout()
    _save_figure(fig, path)


def plot_method_comparison(
    method_summary: pd.DataFrame,
    path: Path,
    chinese: bool,
) -> None:
    """Plot macro errors for fixed, oracle, and learned policies."""
    _configure_plot_style(chinese)
    labels_cn = {
        "Always YOLOv8m": "固定8m",
        "Always YOLO11L": "固定11L",
        "Oracle per window": "Oracle上限",
        "Pixel-threshold LOSO": "像素阈值",
        "Distance-threshold LOSO": "距离阈值",
    }
    labels = [
        labels_cn[value] if chinese else value
        for value in method_summary["method"]
    ]
    values = method_summary["macro_median_error_deg"].to_numpy(dtype=float)
    colors = ["#2563EB", "#DC2626", "#087F5B", "#7C3AED", "#F79009"]
    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    bars = axis.bar(labels, values, color=colors, alpha=0.86)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value:.2f}°",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylabel(
        "等权session宏平均中位误差（°）"
        if chinese
        else "Equal-session macro median disagreement (deg)"
    )
    axis.set_title(
        "固定模型、Oracle与交叉验证切换器"
        if chinese
        else "Fixed models, oracle, and cross-validated switches",
        fontweight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, path)


def plot_fold_thresholds(
    pixel_folds: pd.DataFrame,
    distance_folds: pd.DataFrame,
    path: Path,
    chinese: bool,
) -> None:
    """Plot threshold stability and held-out gain for each capture group."""
    _configure_plot_style(chinese)
    groups = pixel_folds["held_out_group"].astype(str).tolist()
    x = np.arange(len(groups))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    axes[0].plot(
        x,
        pixel_folds["learned_threshold"],
        "o-",
        color="#7C3AED",
        linewidth=2,
        label="Pixel threshold",
    )
    axes[0].set_ylabel(
        "人物像素高度阈值（px）"
        if chinese
        else "Skeleton-height threshold (px)"
    )
    axes[0].set_title(
        "每次留出一个采集组后学到的阈值"
        if chinese
        else "Threshold learned after holding out each capture group",
        fontweight="bold",
    )
    axes_distance = axes[0].twinx()
    axes_distance.plot(
        x,
        distance_folds["learned_threshold"],
        "s--",
        color="#F79009",
        linewidth=1.8,
        label="Distance threshold",
    )
    axes_distance.set_ylabel(
        "距离阈值（m）" if chinese else "Distance threshold (m)"
    )
    width = 0.34
    axes[1].bar(
        x - width / 2,
        pixel_folds["selected_minus_yolov8m_deg"],
        width,
        color="#7C3AED",
        alpha=0.8,
        label="Pixel",
    )
    axes[1].bar(
        x + width / 2,
        distance_folds["selected_minus_yolov8m_deg"],
        width,
        color="#F79009",
        alpha=0.8,
        label="Distance",
    )
    axes[1].axhline(0.0, color="#111827", linewidth=1.1)
    axes[1].set_ylabel(
        "切换器 − 固定8m（°）"
        if chinese
        else "Switch minus always-YOLOv8m (deg)"
    )
    axes[1].set_xticks(x, groups, rotation=20, ha="right")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, path)


def _method_value(summary: pd.DataFrame, method: str, column: str) -> float:
    """Read one scalar method-summary value."""
    return float(summary.loc[summary["method"] == method, column].iloc[0])


def _fold_table(
    pixel_folds: pd.DataFrame,
    distance_folds: pd.DataFrame,
    chinese: bool,
) -> str:
    """Build a compact held-out-group performance table."""
    merged = pixel_folds[
        [
            "held_out_group",
            "n_windows",
            "learned_threshold",
            "selected_minus_yolov8m_deg",
        ]
    ].merge(
        distance_folds[
            [
                "held_out_group",
                "learned_threshold",
                "selected_minus_yolov8m_deg",
            ]
        ],
        on="held_out_group",
        suffixes=("_pixel", "_distance"),
    )
    headers = (
        [
            "留出采集组",
            "窗口数",
            "像素阈值(px)",
            "像素切换−8m(°)",
            "距离阈值(m)",
            "距离切换−8m(°)",
        ]
        if chinese
        else [
            "Held-out group",
            "Windows",
            "Pixel threshold",
            "Pixel switch−8m",
            "Distance threshold",
            "Distance switch−8m",
        ]
    )
    rows = []
    for row in merged.itertuples(index=False):
        cells = [
            str(row.held_out_group),
            str(int(row.n_windows)),
            f"{row.learned_threshold_pixel:.1f}",
            f"{row.selected_minus_yolov8m_deg_pixel:+.2f}",
            f"{row.learned_threshold_distance:.2f}",
            f"{row.selected_minus_yolov8m_deg_distance:+.2f}",
        ]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{value}</th>" for value in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_report(
    chinese: bool,
    method_summary: pd.DataFrame,
    pixel_folds: pd.DataFrame,
    distance_folds: pd.DataFrame,
    figures: dict[str, Path],
    decision: dict[str, Any],
    window_count: int,
    group_count: int,
) -> str:
    """Build one self-contained bilingual HTML report."""
    suffix = "_CN" if chinese else ""
    images = {
        key: _image_data_uri(
            path.with_name(f"{path.stem}{suffix}{path.suffix}")
        )
        for key, path in figures.items()
    }
    oracle_gain = _method_value(
        method_summary, "Oracle per window", "gain_vs_yolov8m_percent"
    )
    pixel_gain = _method_value(
        method_summary, "Pixel-threshold LOSO", "gain_vs_yolov8m_percent"
    )
    distance_gain = _method_value(
        method_summary, "Distance-threshold LOSO", "gain_vs_yolov8m_percent"
    )
    fold_table = _fold_table(pixel_folds, distance_folds, chinese)
    if chinese:
        title = "YOLOv8m / YOLO11L 动态切换模式验证"
        subtitle = (
            f"{window_count}个非重叠窗口 · {group_count}个独立采集组 · "
            "按采集组留一验证 · Xsens-derived reference仅作外部比较"
        )
        verdict = (
            "当前证据支持继续开发简单动态切换规则。"
            if decision["supported"]
            else "当前证据不支持部署一个简单的距离或像素阈值切换规则。"
        )
        body = f"""
<section id="summary"><h2>1. 结论</h2><div class="cards"><div class="card"><div class="metric">{oracle_gain:+.1f}%</div>Oracle理论改善</div><div class="card"><div class="metric">{pixel_gain:+.1f}%</div>像素阈值交叉验证改善</div><div class="card"><div class="metric">{distance_gain:+.1f}%</div>距离阈值交叉验证改善</div></div><div class="callout {'ok' if decision['supported'] else 'warning'}"><strong>{verdict}</strong> Oracle用于衡量理论上限，它在部署时不可用；真正需要关注的是按采集组留一后的像素和距离阈值结果。</div><p>{decision['reason_cn']}</p><figure><img src="{images['methods']}" alt="固定模型、Oracle与切换器比较"><figcaption>所有指标先在每个session内取窗口中位数，再对session等权平均，避免长视频支配结果。</figcaption></figure></section>
<section id="pixel"><h2>2. 人物像素大小是否形成稳定pattern？</h2><figure><img src="{images['pixel_box']}" alt="像素大小分箱箱线图"><figcaption>纵轴为11L误差减去8m误差。若“近处11L、远处8m”成立，箱体应从大像素区的零线下方，稳定移动到小像素区的零线上方。图中只显示至少含5个窗口的分组。</figcaption></figure></section>
<section id="distance"><h2>3. 距离是否形成稳定pattern？</h2><figure><img src="{images['distance_box']}" alt="距离分箱箱线图"><figcaption>不同距离箱体的符号和重叠程度比一条均值曲线更重要；大量跨越零线表示同一距离下两个模型的胜负并不稳定。图中只显示至少含5个窗口的距离区间。</figcaption></figure><figure><img src="{images['motion']}" alt="距离和运动等级热力图"><figcaption>热力图检查此前发现的混杂因素：如果同一距离在静止和快速运动中颜色相反，距离就不是足够的切换变量。少于3个窗口的格子不解读。</figcaption></figure></section>
<section id="validation"><h2>4. 按采集组留一验证</h2><figure><img src="{images['folds']}" alt="阈值稳定性和留出集收益"><figcaption>上图显示每次留出一整组数据后学到的阈值；下图小于零才表示切换器优于固定8m。Fanbo9两台相机始终一起留出，避免同一动作泄漏到训练集。</figcaption></figure>{fold_table}</section>
<section id="method"><h2>5. 测试方法与限制</h2><ul><li>每个1秒窗口使用所有可用的人体工学角度的中位绝对差，不挑选特定身体部位；每帧至少需要4个共同角度。</li><li>像素特征是YOLOv8m左图2D骨架的稳健高度，只用于探索“像素信息量”假设；真正部署时应由上一帧或统一轻量跟踪器提供。</li><li>运动等级来自Xsens-derived角速度，只用于检查混杂，不参与模型切换。</li><li>距离仍主要与session绑定，因此本分析只能筛查pattern，不能代替相同动作、相同视角的受控距离实验。</li><li>Oracle直接查看两个模型的比较误差后选较好者，只表示理论上限，不能用于部署。</li></ul></section>"""
        nav = ["结论", "像素pattern", "距离pattern", "交叉验证", "方法与限制"]
    else:
        title = "YOLOv8m / YOLO11L Dynamic-Switch Pattern Validation"
        subtitle = (
            f"{window_count} non-overlapping windows · {group_count} independent "
            "capture groups · leave-capture-group-out validation · Xsens-derived "
            "reference used only for external comparison"
        )
        verdict = (
            "Current evidence supports continued development of a simple switch."
            if decision["supported"]
            else "Current evidence does not support deploying a simple distance or pixel threshold."
        )
        body = f"""
<section id="summary"><h2>1. Conclusion</h2><div class="cards"><div class="card"><div class="metric">{oracle_gain:+.1f}%</div>Oracle potential gain</div><div class="card"><div class="metric">{pixel_gain:+.1f}%</div>Cross-validated pixel-switch gain</div><div class="card"><div class="metric">{distance_gain:+.1f}%</div>Cross-validated distance-switch gain</div></div><div class="callout {'ok' if decision['supported'] else 'warning'}"><strong>{verdict}</strong> The oracle measures only the theoretical ceiling and is unavailable at deployment; the leave-capture-group-out pixel and distance results are the relevant tests.</div><p>{decision['reason_en']}</p><figure><img src="{images['methods']}" alt="Fixed models, oracle, and learned switches"><figcaption>Metrics first take the window median within each session and then weight sessions equally, preventing long videos from dominating the result.</figcaption></figure></section>
<section id="pixel"><h2>2. Does person pixel scale form a stable pattern?</h2><figure><img src="{images['pixel_box']}" alt="Pixel-scale binned box plots"><figcaption>The vertical axis is YOLO11L error minus YOLOv8m error. Under the proposed near-11L/far-8m hypothesis, boxes should move consistently from below zero at large pixel scales to above zero at small scales. Only groups containing at least five windows are plotted.</figcaption></figure></section>
<section id="distance"><h2>3. Does distance form a stable pattern?</h2><figure><img src="{images['distance_box']}" alt="Distance-binned box plots"><figcaption>The sign and overlap of the distributions matter more than a mean curve. Boxes spanning zero show that model preference is unstable at the same distance. Only distance bins containing at least five windows are plotted.</figcaption></figure><figure><img src="{images['motion']}" alt="Distance and motion heatmap"><figcaption>The heatmap checks the previously identified confounder: if static and fast motion have opposite colours at the same distance, distance is not a sufficient routing variable. Cells with fewer than three windows are not interpreted.</figcaption></figure></section>
<section id="validation"><h2>4. Leave-capture-group-out validation</h2><figure><img src="{images['folds']}" alt="Threshold stability and held-out gain"><figcaption>The upper panel shows thresholds learned after holding out an entire capture group; only negative bars in the lower panel mean that switching beats always using YOLOv8m. Both Fanbo9 cameras are held out together to prevent action leakage.</figcaption></figure>{fold_table}</section>
<section id="method"><h2>5. Method and limitations</h2><ul><li>Each one-second window uses the median absolute disagreement across all available ergonomic angles, without selecting a body part; each frame requires at least four common angles.</li><li>Pixel scale is the robust height of the YOLOv8m left-view 2D skeleton and is used only to test the information-content hypothesis. Deployment should obtain it from the preceding frame or a shared lightweight tracker.</li><li>Xsens-derived angular speed is used only to inspect motion confounding and never as a switching input.</li><li>Distance remains largely tied to session, so this analysis screens for a pattern but cannot replace a controlled same-action, same-viewpoint distance experiment.</li><li>The oracle directly observes both comparison errors before selecting a model; it is a theoretical ceiling, not a deployable method.</li></ul></section>"""
        nav = ["Conclusion", "Pixel pattern", "Distance pattern", "Cross-validation", "Method"]
    section_ids = ["summary", "pixel", "distance", "validation", "method"]
    links = "".join(
        f'<a href="#{section_id}">{label}</a>'
        for section_id, label in zip(section_ids, nav)
    )
    language = "zh-CN" if chinese else "en"
    footer = (
        "本报告为自包含HTML，图表已内嵌。"
        if chinese
        else "Self-contained HTML report with embedded figures."
    )
    return (
        f'<!doctype html><html lang="{language}"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{STYLE}</style></head><body>"
        f"<header><h1>{title}</h1><p>{subtitle}</p></header>"
        f'<div class="layout"><nav><strong>{"目录" if chinese else "Contents"}</strong>'
        f"{links}</nav><main>{body}<footer>{footer}</footer></main></div></body></html>"
    )


def write_outputs(config_path: Path) -> Path:
    """Run the complete dynamic-switch pattern analysis."""
    payload, pairs = load_switch_config(config_path)
    analysis = payload.get("analysis", {})
    output_dir = _resolve(
        analysis.get(
            "output_dir",
            "00_pose_pipeline_v2/runs/dynamic_model_switch_analysis",
        )
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    window_seconds = float(analysis.get("window_seconds", 1.0))
    minimum_frames = int(analysis.get("min_valid_frames_per_window", 5))
    minimum_angles = int(analysis.get("min_common_angles_per_frame", 4))
    motion_thresholds = tuple(
        float(value)
        for value in analysis.get("motion_thresholds_deg_s", [5.0, 20.0])
    )
    if len(motion_thresholds) != 2:
        raise ValueError("Exactly two motion thresholds are required")

    frame_tables: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for pair in pairs:
        table, source = build_frame_table(pair, minimum_angles)
        frame_tables.append(table)
        sources.append(source)
    frame_data = pd.concat(frame_tables, ignore_index=True)
    windows = build_window_table(
        frame_data,
        window_seconds,
        minimum_frames,
        (motion_thresholds[0], motion_thresholds[1]),
    )
    windows, pixel_labels, distance_labels = prepare_plot_bins(
        windows,
        int(analysis.get("pixel_bin_count", 4)),
        float(analysis.get("distance_bin_width_m", 0.5)),
        int(analysis.get("minimum_windows_per_plot_bin", 5)),
    )

    pixel_predictions, pixel_folds = leave_group_out_switch(
        windows,
        "skeleton_height_px",
        "high_uses_11l",
    )
    distance_predictions, distance_folds = leave_group_out_switch(
        windows,
        "optical_depth_m",
        "low_uses_11l",
    )
    method_summary = summarize_methods(
        windows, pixel_predictions, distance_predictions
    )

    baseline = _method_value(
        method_summary, "Always YOLOv8m", "macro_median_error_deg"
    )
    oracle_gain = _method_value(
        method_summary, "Oracle per window", "gain_vs_yolov8m_percent"
    )
    pixel_gain = _method_value(
        method_summary, "Pixel-threshold LOSO", "gain_vs_yolov8m_percent"
    )
    distance_gain = _method_value(
        method_summary, "Distance-threshold LOSO", "gain_vs_yolov8m_percent"
    )
    minimum_oracle_gain = float(
        analysis.get("minimum_oracle_gain_percent", 5.0)
    )
    minimum_switch_gain = float(
        analysis.get("minimum_switch_gain_percent", 5.0)
    )
    maximum_allowed_degradation = float(
        analysis.get("maximum_group_degradation_deg", 1.0)
    )
    if pixel_gain >= distance_gain:
        best_switch_name = "pixel"
        best_switch_gain = pixel_gain
        best_switch_folds = pixel_folds
    else:
        best_switch_name = "distance"
        best_switch_gain = distance_gain
        best_switch_folds = distance_folds
    maximum_degradation = float(
        best_switch_folds["selected_minus_yolov8m_deg"].max()
    )
    supported = bool(
        oracle_gain >= minimum_oracle_gain
        and best_switch_gain >= minimum_switch_gain
        and maximum_degradation <= maximum_allowed_degradation
    )
    reason_cn = (
        f"Oracle相对固定8m的理论改善为{oracle_gain:.1f}%，说明两个模型之间"
        f"{'存在可利用的互补性' if oracle_gain >= minimum_oracle_gain else '互补空间有限'}；"
        f"但真正留组验证的最佳简单阈值改善为{best_switch_gain:.1f}%，"
        f"最差留出组相对8m变化为{maximum_degradation:+.2f}°。"
    )
    reason_en = (
        f"The oracle improves on always using YOLOv8m by {oracle_gain:.1f}%, "
        f"indicating {'potential complementarity' if oracle_gain >= minimum_oracle_gain else 'limited complementarity'}; "
        f"however, the best genuinely held-out simple threshold changes the "
        f"macro error by {best_switch_gain:.1f}%, and the worst held-out group "
        f"changes by {maximum_degradation:+.2f} deg relative to YOLOv8m."
    )
    decision = {
        "supported": supported,
        "reason_cn": reason_cn,
        "reason_en": reason_en,
        "baseline_macro_error_deg": baseline,
        "oracle_gain_percent": oracle_gain,
        "pixel_switch_gain_percent": pixel_gain,
        "distance_switch_gain_percent": distance_gain,
        "best_simple_switch": best_switch_name,
        "maximum_held_out_degradation_deg": maximum_degradation,
        "criteria": {
            "minimum_oracle_gain_percent": minimum_oracle_gain,
            "minimum_switch_gain_percent": minimum_switch_gain,
            "maximum_group_degradation_deg": maximum_allowed_degradation,
        },
    }

    figure_paths = {
        "pixel_box": figures_dir / "01_pixel_scale_advantage_boxplot.png",
        "distance_box": figures_dir / "02_distance_advantage_boxplot.png",
        "motion": figures_dir / "03_distance_motion_heatmap.png",
        "methods": figures_dir / "04_method_comparison.png",
        "folds": figures_dir / "05_fold_thresholds_and_gain.png",
    }
    for chinese, suffix in ((False, ""), (True, "_CN")):
        _boxplot_by_bins(
            windows,
            "pixel_bin",
            pixel_labels,
            figure_paths["pixel_box"].with_name(
                f"01_pixel_scale_advantage_boxplot{suffix}.png"
            ),
            chinese,
            "人物像素大小与模型优势"
            if chinese
            else "Person pixel scale and model advantage",
            "2D骨架高度分位组（px）"
            if chinese
            else "2D skeleton-height quantile bin (px)",
        )
        _boxplot_by_bins(
            windows,
            "distance_bin",
            distance_labels,
            figure_paths["distance_box"].with_name(
                f"02_distance_advantage_boxplot{suffix}.png"
            ),
            chinese,
            "估计距离与模型优势"
            if chinese
            else "Estimated distance and model advantage",
            "估计光轴深度（m）"
            if chinese
            else "Estimated optical depth (m)",
        )
        plot_motion_heatmap(
            windows,
            distance_labels,
            figure_paths["motion"].with_name(
                f"03_distance_motion_heatmap{suffix}.png"
            ),
            chinese,
            int(analysis.get("minimum_windows_per_heatmap_cell", 3)),
        )
        plot_method_comparison(
            method_summary,
            figure_paths["methods"].with_name(
                f"04_method_comparison{suffix}.png"
            ),
            chinese,
        )
        plot_fold_thresholds(
            pixel_folds,
            distance_folds,
            figure_paths["folds"].with_name(
                f"05_fold_thresholds_and_gain{suffix}.png"
            ),
            chinese,
        )

    frame_data.to_csv(output_dir / "paired_frame_metrics.csv", index=False)
    export_windows = windows.copy()
    for column in ("motion_class", "pixel_bin", "distance_bin"):
        export_windows[column] = export_windows[column].astype(object)
    export_windows.to_csv(output_dir / "window_metrics.csv", index=False)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)
    pixel_folds.to_csv(output_dir / "pixel_threshold_folds.csv", index=False)
    distance_folds.to_csv(
        output_dir / "distance_threshold_folds.csv", index=False
    )
    manifest = {
        "analysis_name": analysis.get("title"),
        "created_from_commit": _git_commit(),
        "analysis_config": str(config_path.relative_to(PROJECT_ROOT)),
        "analysis_config_sha256": _sha256(config_path),
        "window_seconds": window_seconds,
        "minimum_valid_frames_per_window": minimum_frames,
        "minimum_common_angles_per_frame": minimum_angles,
        "angle_names": list(SEMANTIC_ANGLE_NAMES),
        "motion_thresholds_deg_s": list(motion_thresholds),
        "reference": analysis.get(
            "reference_label", "Xsens-derived reference"
        ),
        "window_count": int(len(windows)),
        "validation_groups": windows["validation_group"].drop_duplicates().tolist(),
        "decision": decision,
        "sources": sources,
        "limitations": [
            "Distance remains confounded with session, action, and viewpoint.",
            "Skeleton pixel height is a common YOLOv8m-derived exploratory feature.",
            "Frames are aggregated into non-overlapping windows; capture groups are held out intact.",
            "Xsens is an external comparison system, not absolute Ground Truth.",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    group_count = int(windows["validation_group"].nunique())
    (output_dir / "report_CN.html").write_text(
        build_report(
            True,
            method_summary,
            pixel_folds,
            distance_folds,
            figure_paths,
            decision,
            len(windows),
            group_count,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(
        build_report(
            False,
            method_summary,
            pixel_folds,
            distance_folds,
            figure_paths,
            decision,
            len(windows),
            group_count,
        ),
        encoding="utf-8",
    )
    print(
        f"[dynamic-switch] windows={len(windows)} groups={group_count} "
        f"oracle_gain={oracle_gain:.2f}% pixel_gain={pixel_gain:.2f}% "
        f"distance_gain={distance_gain:.2f}% supported={supported}"
    )
    print(f"[dynamic-switch] output={output_dir}")
    return output_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "00_pose_pipeline_v2/configs/dynamic_model_switch_analysis.yaml"
        ),
    )
    args = parser.parse_args()
    write_outputs(_resolve(args.config))


if __name__ == "__main__":
    main()
