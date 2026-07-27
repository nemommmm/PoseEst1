"""Plot bilateral hip disagreement versus stereo distance for two YOLO models.

Only paired frames where YOLOv8m, YOLO11L, both hip angles, FastSAM3D, and
the shared YOLOv8m-derived torso depth are all finite enter the comparison.
FastSAM3D is treated as a comparison trajectory, not absolute ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from analyze_dynamic_model_switch import (  # noqa: E402
    PairSpec,
    _config_for_npz,
    load_switch_config,
)
from analyze_error_vs_distance import (  # noqa: E402
    MODEL_COLORS,
    PROJECT_ROOT,
    STYLE,
    _configure_plot_style,
    _git_commit,
    _image_data_uri,
    _save_figure,
    _sha256,
    _torso_distance,
)
from common.config import resolve_path, section  # noqa: E402
from common.dataset import load_method_keypoints  # noqa: E402
from eval_angles import prepare_angles  # noqa: E402


def _resolve(value: str | Path) -> Path:
    """Resolve one project-relative path."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_analysis_config(path: Path) -> tuple[dict[str, Any], list[PairSpec]]:
    """Load the plot configuration and its paired detector-result catalogue."""
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    analysis = payload.get("analysis", {})
    paired_config = _resolve(analysis["paired_results_config"])
    _, pairs = load_switch_config(paired_config)
    return payload, pairs


def _model_angles(
    pair: PairSpec,
    npz_path: Path,
    hip_joints: tuple[str, str],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load one model and FastSAM3D angles on the stereo-video timeline."""
    config = _config_for_npz(pair.base_config, npz_path)
    config.setdefault("evaluation", {})["angle_names"] = list(hip_joints)
    time_s, all_angles, info = prepare_angles(
        config,
        npz_path.parent,
        pair.fixed_offset_seconds,
    )
    return np.asarray(time_s, dtype=np.float64), all_angles, {
        "config": config,
        "angle_info": info,
    }


def extract_pair_rows(
    pair: PairSpec,
    hip_joints: tuple[str, str],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Extract common-frame bilateral hip disagreement for one paired result."""
    time_eight, angles_eight, meta_eight = _model_angles(
        pair,
        pair.yolov8m_npz,
        hip_joints,
    )
    time_eleven, angles_eleven, meta_eleven = _model_angles(
        pair,
        pair.yolo11l_npz,
        hip_joints,
    )
    if len(time_eight) != len(time_eleven) or not np.allclose(
        time_eight,
        time_eleven,
        atol=1e-7,
    ):
        raise RuntimeError(f"Detector timelines differ for {pair.name}")
    if "FastSAM3D" not in angles_eight or "FastSAM3D" not in angles_eleven:
        return None, {
            "session": pair.name,
            "session_id": pair.session_id,
            "status": "skipped_missing_fastsam3d",
        }

    config_eight = meta_eight["config"]
    _, _, methods = load_method_keypoints(
        config_eight,
        pair.yolov8m_npz.parent,
    )
    depth_m, radial_range_m = _torso_distance(methods["SKT"])
    reference_columns: list[np.ndarray] = []
    eight_errors: list[np.ndarray] = []
    eleven_errors: list[np.ndarray] = []
    common = np.isfinite(depth_m) & np.isfinite(radial_range_m)
    maximum_reference_difference = 0.0
    for joint in hip_joints:
        reference_eight = np.asarray(
            angles_eight["FastSAM3D"][joint],
            dtype=np.float64,
        )
        reference_eleven = np.asarray(
            angles_eleven["FastSAM3D"][joint],
            dtype=np.float64,
        )
        shared_reference = np.isfinite(reference_eight) & np.isfinite(
            reference_eleven
        )
        if np.any(shared_reference):
            maximum_reference_difference = max(
                maximum_reference_difference,
                float(
                    np.max(
                        np.abs(
                            reference_eight[shared_reference]
                            - reference_eleven[shared_reference]
                        )
                    )
                ),
            )
        angle_eight = np.asarray(angles_eight["SKT"][joint], dtype=np.float64)
        angle_eleven = np.asarray(
            angles_eleven["SKT"][joint],
            dtype=np.float64,
        )
        common &= (
            np.isfinite(reference_eight)
            & np.isfinite(reference_eleven)
            & np.isfinite(angle_eight)
            & np.isfinite(angle_eleven)
        )
        reference_columns.append(reference_eight)
        eight_errors.append(np.abs(angle_eight - reference_eight))
        eleven_errors.append(np.abs(angle_eleven - reference_eight))
    if maximum_reference_difference > 1e-7:
        raise RuntimeError(
            f"FastSAM3D references differ between paired models for {pair.name}"
        )

    valid_indices = np.flatnonzero(common)
    if valid_indices.size == 0:
        return None, {
            "session": pair.name,
            "session_id": pair.session_id,
            "status": "skipped_no_common_hip_frames",
        }
    reference_matrix = np.column_stack(reference_columns)
    error_matrices = {
        "YOLOv8m": np.column_stack(eight_errors),
        "YOLO11L": np.column_stack(eleven_errors),
    }
    rows: list[dict[str, Any]] = []
    for model, errors in error_matrices.items():
        for frame in valid_indices:
            rows.append(
                {
                    "session": pair.name,
                    "session_id": pair.session_id,
                    "validation_group": pair.validation_group,
                    "frame": int(frame),
                    "time_s": float(time_eight[frame]),
                    "model": model,
                    "optical_depth_m": float(depth_m[frame]),
                    "radial_range_m": float(radial_range_m[frame]),
                    "left_hip_fastsam_deg": float(reference_matrix[frame, 0]),
                    "right_hip_fastsam_deg": float(reference_matrix[frame, 1]),
                    "left_hip_abs_disagreement_deg": float(errors[frame, 0]),
                    "right_hip_abs_disagreement_deg": float(errors[frame, 1]),
                    "bilateral_hip_abs_disagreement_deg": float(
                        np.median(errors[frame])
                    ),
                }
            )

    references = section(config_eight, "references")
    trc_path = resolve_path(
        references.get("fastsam_trc"),
        must_exist=True,
    )
    assert trc_path is not None
    camera_path = _resolve(config_eight["calibration"]["camera_params"])
    source = {
        "session": pair.name,
        "session_id": pair.session_id,
        "status": "included",
        "common_frame_count": int(valid_indices.size),
        "base_config": str(pair.base_config.relative_to(PROJECT_ROOT)),
        "base_config_sha256": _sha256(pair.base_config),
        "yolov8m_npz": str(pair.yolov8m_npz.relative_to(PROJECT_ROOT)),
        "yolov8m_npz_sha256": _sha256(pair.yolov8m_npz),
        "yolo11l_npz": str(pair.yolo11l_npz.relative_to(PROJECT_ROOT)),
        "yolo11l_npz_sha256": _sha256(pair.yolo11l_npz),
        "fastsam_trc": str(trc_path.relative_to(PROJECT_ROOT)),
        "fastsam_trc_sha256": _sha256(trc_path),
        "fastsam_source_offset_seconds": float(
            (references.get("trc_time_offsets_seconds", {}) or {}).get(
                "FastSAM3D",
                0.0,
            )
        ),
        "camera_calibration": str(camera_path.relative_to(PROJECT_ROOT)),
        "camera_calibration_sha256": _sha256(camera_path),
        "camera_smooth_window_actual_ms": meta_eight["angle_info"][
            "camera_smooth_window_actual_ms"
        ],
    }
    return pd.DataFrame(rows), source


def add_distance_bins(
    frame_data: pd.DataFrame,
    width_m: float,
) -> pd.DataFrame:
    """Attach fixed-width optical-depth bins to common-frame rows."""
    data = frame_data.copy()
    finite = data.loc[
        np.isfinite(data["optical_depth_m"]),
        "optical_depth_m",
    ]
    start = math.floor(float(finite.min()) / width_m) * width_m
    stop = math.ceil(float(finite.max()) / width_m) * width_m
    edges = np.arange(start, stop + width_m * 1.01, width_m)
    labels = [
        f"{left:.1f}–{right:.1f}"
        for left, right in zip(edges[:-1], edges[1:])
    ]
    data["distance_bin"] = pd.cut(
        data["optical_depth_m"],
        bins=edges,
        labels=labels,
        right=False,
    )
    data["distance_bin_left_m"] = (
        np.floor(data["optical_depth_m"] / width_m) * width_m
    )
    return data


def summarize_bins(
    frame_data: pd.DataFrame,
    minimum_common_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Summarize eligible distance bins and each source session."""
    bin_rows: list[dict[str, Any]] = []
    for (distance_bin, model), group in frame_data.groupby(
        ["distance_bin", "model"],
        observed=True,
        sort=True,
    ):
        values = group["bilateral_hip_abs_disagreement_deg"].to_numpy(
            dtype=np.float64
        )
        bin_rows.append(
            {
                "distance_bin": str(distance_bin),
                "distance_bin_left_m": float(
                    group["distance_bin_left_m"].iloc[0]
                ),
                "model": model,
                "n_common_frames": int(len(values)),
                "n_sessions": int(group["session_id"].nunique()),
                "median_optical_depth_m": float(
                    np.median(group["optical_depth_m"])
                ),
                "median_deg": float(np.median(values)),
                "p25_deg": float(np.percentile(values, 25)),
                "p75_deg": float(np.percentile(values, 75)),
                "p95_deg": float(np.percentile(values, 95)),
            }
        )
    bin_summary = pd.DataFrame(bin_rows).sort_values(
        ["distance_bin_left_m", "model"]
    )
    counts = bin_summary.pivot(
        index="distance_bin",
        columns="model",
        values="n_common_frames",
    )
    eligible = counts[
        (counts.get("YOLOv8m", 0) >= minimum_common_frames)
        & (counts.get("YOLO11L", 0) >= minimum_common_frames)
    ].index.tolist()
    eligible_set = set(eligible)
    eligible_labels = (
        bin_summary[
            bin_summary["distance_bin"].isin(eligible_set)
        ]
        .sort_values("distance_bin_left_m")["distance_bin"]
        .drop_duplicates()
        .tolist()
    )

    session_rows: list[dict[str, Any]] = []
    for (session, session_id, model), group in frame_data.groupby(
        ["session", "session_id", "model"],
        observed=True,
        sort=False,
    ):
        values = group["bilateral_hip_abs_disagreement_deg"].to_numpy(
            dtype=np.float64
        )
        session_rows.append(
            {
                "session": session,
                "session_id": session_id,
                "model": model,
                "n_common_frames": int(len(values)),
                "median_optical_depth_m": float(
                    np.median(group["optical_depth_m"])
                ),
                "p25_optical_depth_m": float(
                    np.percentile(group["optical_depth_m"], 25)
                ),
                "p75_optical_depth_m": float(
                    np.percentile(group["optical_depth_m"], 75)
                ),
                "median_deg": float(np.median(values)),
                "p25_deg": float(np.percentile(values, 25)),
                "p75_deg": float(np.percentile(values, 75)),
                "p95_deg": float(np.percentile(values, 95)),
            }
        )
    return (
        bin_summary,
        pd.DataFrame(session_rows),
        eligible_labels,
    )


def plot_grouped_boxplot(
    frame_data: pd.DataFrame,
    bin_summary: pd.DataFrame,
    eligible_labels: list[str],
    path: Path,
    chinese: bool,
) -> None:
    """Plot paired model box plots in each eligible distance interval."""
    _configure_plot_style(chinese)
    model_specs = (
        ("YOLOv8m", -0.19),
        ("YOLO11L", 0.19),
    )
    centres = np.arange(len(eligible_labels), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(12.4, 6.4))
    for model, offset in model_specs:
        groups = [
            frame_data.loc[
                (frame_data["distance_bin"].astype(object).map(str) == label)
                & (frame_data["model"] == model),
                "bilateral_hip_abs_disagreement_deg",
            ].to_numpy(dtype=np.float64)
            for label in eligible_labels
        ]
        boxes = axis.boxplot(
            groups,
            positions=centres + offset,
            widths=0.32,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
            flierprops={
                "marker": ".",
                "markersize": 2.4,
                "alpha": 0.16,
                "markeredgecolor": MODEL_COLORS[model],
            },
            medianprops={"color": "#111827", "linewidth": 1.8},
            whiskerprops={
                "color": MODEL_COLORS[model],
                "linewidth": 1.15,
            },
            capprops={
                "color": MODEL_COLORS[model],
                "linewidth": 1.15,
            },
        )
        for box in boxes["boxes"]:
            box.set_facecolor(MODEL_COLORS[model])
            box.set_edgecolor(MODEL_COLORS[model])
            box.set_alpha(0.58)
        for index, values in enumerate(groups):
            median = float(np.median(values))
            axis.text(
                centres[index] + offset,
                median,
                f"{median:.1f}°",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#111827",
            )
        axis.plot(
            [],
            [],
            color=MODEL_COLORS[model],
            linewidth=9,
            alpha=0.58,
            label=model,
        )

    counts = {
        str(row.distance_bin): int(row.n_common_frames)
        for row in bin_summary[
            bin_summary["model"] == "YOLOv8m"
        ].itertuples(index=False)
    }
    ticks = [
        f"{label}\n(n={counts[label]})"
        for label in eligible_labels
    ]
    axis.set_xticks(centres, ticks)
    axis.set_xlabel(
        "YOLOv8m 双目躯干光轴深度分箱（m）"
        if chinese
        else "YOLOv8m stereo torso-depth bin (m)"
    )
    axis.set_ylabel(
        "双髋绝对角度差（°）"
        if chinese
        else "Bilateral hip absolute angular disagreement (deg)"
    )
    axis.set_title(
        "YOLOv8m 与 YOLO11L：髋部角度差随距离的箱线图"
        if chinese
        else "YOLOv8m vs YOLO11L: hip disagreement by distance",
        fontweight="bold",
    )
    axis.text(
        0.99,
        0.98,
        "参考：FastSAM3D；每帧取左右髋中位数"
        if chinese
        else "Reference: FastSAM3D; median of left/right hips per frame",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#5D6B7A",
        fontsize=9,
    )
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    _save_figure(fig, path)


def _table_html(summary: pd.DataFrame, chinese: bool) -> str:
    """Render one compact distance-bin summary table."""
    pivot = summary.pivot(
        index="distance_bin",
        columns="model",
        values=["n_common_frames", "median_deg", "p25_deg", "p75_deg", "p95_deg"],
    )
    order = (
        summary.sort_values("distance_bin_left_m")["distance_bin"]
        .drop_duplicates()
        .tolist()
    )
    headers = (
        [
            "距离(m)",
            "共同帧",
            "8m中位数",
            "11L中位数",
            "8m IQR",
            "11L IQR",
            "8m P95",
            "11L P95",
        ]
        if chinese
        else [
            "Distance (m)",
            "Common frames",
            "8m median",
            "11L median",
            "8m IQR",
            "11L IQR",
            "8m P95",
            "11L P95",
        ]
    )
    rows: list[str] = []
    for label in order:
        n = int(pivot.loc[label, ("n_common_frames", "YOLOv8m")])
        values = [
            label,
            str(n),
            f"{pivot.loc[label, ('median_deg', 'YOLOv8m')]:.2f}°",
            f"{pivot.loc[label, ('median_deg', 'YOLO11L')]:.2f}°",
            (
                f"{pivot.loc[label, ('p25_deg', 'YOLOv8m')]:.2f}–"
                f"{pivot.loc[label, ('p75_deg', 'YOLOv8m')]:.2f}°"
            ),
            (
                f"{pivot.loc[label, ('p25_deg', 'YOLO11L')]:.2f}–"
                f"{pivot.loc[label, ('p75_deg', 'YOLO11L')]:.2f}°"
            ),
            f"{pivot.loc[label, ('p95_deg', 'YOLOv8m')]:.2f}°",
            f"{pivot.loc[label, ('p95_deg', 'YOLO11L')]:.2f}°",
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{value}</td>" for value in values)
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_report(
    chinese: bool,
    figure_path: Path,
    bin_summary: pd.DataFrame,
    eligible_labels: list[str],
    included_sessions: int,
    skipped_sources: list[dict[str, Any]],
) -> str:
    """Build a concise self-contained HTML report around the requested plot."""
    suffix = "_CN" if chinese else ""
    image_path = figure_path.with_name(
        f"{figure_path.stem}{suffix}{figure_path.suffix}"
    )
    eligible = bin_summary[
        bin_summary["distance_bin"].isin(set(eligible_labels))
    ].copy()
    pivot = eligible.pivot(
        index="distance_bin",
        columns="model",
        values="median_deg",
    )
    bins_favouring_eleven = int(
        (pivot["YOLO11L"] < pivot["YOLOv8m"]).sum()
    )
    table = _table_html(eligible, chinese)
    skipped_2025 = any(
        source["session"] == "2025 Ergonomics"
        for source in skipped_sources
    )
    if chinese:
        title = "FastSAM3D参照下的髋部误差—距离箱线图"
        summary = (
            f"共纳入{included_sessions}组同步双目结果和"
            f"{len(eligible_labels)}个合格距离区间。"
            f"YOLO11L在{bins_favouring_eleven}/{len(eligible_labels)}个区间中"
            "具有更低的中位髋部角度差。"
        )
        limitation = (
            "2025数据未纳入，因为配置指向的FastSAM3D TRC文件目前本地缺失。"
            if skipped_2025
            else ""
        )
        body = f"""
<section><h2>结果</h2><div class="callout"><strong>{summary}</strong> 但各距离区间仍对应不同动作和session，因此这张图用于观察pattern，不能单独证明距离造成模型优劣变化。</div><figure><img src="{_image_data_uri(image_path)}" alt="按距离分箱的双髋误差箱线图"><figcaption>每帧分别计算左髋和右髋相对FastSAM3D comparison trajectory的绝对角度差，再取两侧中位数。蓝色为YOLOv8m，红色为YOLO11L；横轴括号内n为每个模型的共同有效帧数。只有两个模型、双髋、FastSAM3D和距离同时有效的相同帧才进入统计。</figcaption></figure>{table}</section>
<section><h2>如何理解</h2><ul><li>约2.0–2.5 m区间主要来自Fanbo7，YOLOv8m明显更低。</li><li>2.5–3.5 m以及4.0–4.5 m区间中，YOLO11L的中位数更低。</li><li>结果没有表现出“近距离11L、远距离8m”的平滑切换规律；动作和视角差异仍是重要混杂因素。</li><li>FastSAM3D是外部比较轨迹，不是绝对Ground Truth。{limitation}</li></ul></section>"""
        language = "zh-CN"
    else:
        title = "Hip disagreement versus distance using FastSAM3D"
        summary = (
            f"{included_sessions} synchronized stereo result sets and "
            f"{len(eligible_labels)} eligible distance bins are included. "
            f"YOLO11L has the lower median hip disagreement in "
            f"{bins_favouring_eleven}/{len(eligible_labels)} bins."
        )
        limitation = (
            "The 2025 data are excluded because the configured FastSAM3D TRC "
            "file is currently absent locally."
            if skipped_2025
            else ""
        )
        body = f"""
<section><h2>Result</h2><div class="callout"><strong>{summary}</strong> Distance intervals still correspond to different actions and sessions, so this chart is exploratory evidence of a pattern rather than proof of a causal distance effect.</div><figure><img src="{_image_data_uri(image_path)}" alt="Bilateral hip disagreement box plots by distance"><figcaption>For every frame, the absolute left- and right-hip angular disagreements against the FastSAM3D comparison trajectory are calculated and their median is used. Blue denotes YOLOv8m and red denotes YOLO11L; n on the horizontal axis is the common valid frame count per model. A frame enters only when both models, both hips, FastSAM3D, and the shared distance are finite.</figcaption></figure>{table}</section>
<section><h2>Interpretation</h2><ul><li>The approximately 2.0–2.5 m bin is dominated by Fanbo7 and clearly favours YOLOv8m.</li><li>YOLO11L has a lower median in the 2.5–3.5 m and 4.0–4.5 m intervals.</li><li>The result does not show a smooth near-11L/far-8m switching pattern; action and viewpoint remain important confounders.</li><li>FastSAM3D is an external comparison trajectory, not absolute Ground Truth. {limitation}</li></ul></section>"""
        language = "en"
    return (
        f'<!doctype html><html lang="{language}"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{STYLE}</style></head><body>"
        f"<header><h1>{title}</h1><p>{summary}</p></header>"
        f'<div class="layout"><main>{body}</main></div></body></html>'
    )


def write_outputs(config_path: Path) -> Path:
    """Run the complete FastSAM3D-referenced hip-distance analysis."""
    payload, pairs = load_analysis_config(config_path)
    analysis = payload.get("analysis", {})
    output_dir = _resolve(analysis["output_dir"])
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    hip_joints = tuple(analysis.get("hip_joints", ["LeftHip", "RightHip"]))
    if len(hip_joints) != 2:
        raise ValueError("Exactly two hip joints are required")

    tables: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for pair in pairs:
        table, source = extract_pair_rows(
            pair,
            (str(hip_joints[0]), str(hip_joints[1])),
        )
        sources.append(source)
        if table is not None:
            tables.append(table)
    if not tables:
        raise RuntimeError("No paired result contains a usable FastSAM3D trajectory")
    frame_data = add_distance_bins(
        pd.concat(tables, ignore_index=True),
        float(analysis.get("distance_bin_width_m", 0.5)),
    )
    minimum_frames = int(
        analysis.get("minimum_common_frames_per_bin", 20)
    )
    bin_summary, session_summary, eligible_labels = summarize_bins(
        frame_data,
        minimum_frames,
    )
    if not eligible_labels:
        raise RuntimeError("No distance bin passes the common-frame threshold")
    eligible_data = frame_data[
        frame_data["distance_bin"].astype(object).map(str).isin(
            set(eligible_labels)
        )
    ].copy()
    eligible_summary = bin_summary[
        bin_summary["distance_bin"].isin(set(eligible_labels))
    ].copy()

    figure_path = figures_dir / "hip_error_boxplot_by_distance.png"
    plot_grouped_boxplot(
        eligible_data,
        eligible_summary,
        eligible_labels,
        figure_path,
        False,
    )
    plot_grouped_boxplot(
        eligible_data,
        eligible_summary,
        eligible_labels,
        figure_path.with_name("hip_error_boxplot_by_distance_CN.png"),
        True,
    )

    export_data = frame_data.copy()
    export_data["distance_bin"] = export_data["distance_bin"].astype(object)
    export_data.to_csv(output_dir / "per_frame_hip_distance_data.csv", index=False)
    bin_summary.to_csv(output_dir / "distance_bin_summary.csv", index=False)
    session_summary.to_csv(output_dir / "session_summary.csv", index=False)
    included_sessions = int(
        frame_data["session_id"].nunique()
    )
    skipped_sources = [
        source for source in sources if source["status"] != "included"
    ]
    (output_dir / "report_CN.html").write_text(
        build_report(
            True,
            figure_path,
            bin_summary,
            eligible_labels,
            included_sessions,
            skipped_sources,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(
        build_report(
            False,
            figure_path,
            bin_summary,
            eligible_labels,
            included_sessions,
            skipped_sources,
        ),
        encoding="utf-8",
    )
    manifest = {
        "analysis_name": analysis.get("title"),
        "created_from_commit": _git_commit(),
        "analysis_config": str(config_path.relative_to(PROJECT_ROOT)),
        "analysis_config_sha256": _sha256(config_path),
        "reference": analysis.get(
            "reference_label",
            "FastSAM3D comparison trajectory",
        ),
        "reference_is_absolute_ground_truth": False,
        "distance_source": analysis.get("distance_source"),
        "hip_joints": list(hip_joints),
        "bilateral_aggregation": "per-frame median of left/right absolute angular disagreement",
        "common_frame_policy": "both models, both hips, FastSAM3D, and distance must be finite",
        "distance_bin_width_m": float(
            analysis.get("distance_bin_width_m", 0.5)
        ),
        "minimum_common_frames_per_bin": minimum_frames,
        "eligible_distance_bins": eligible_labels,
        "included_session_count": included_sessions,
        "sources": sources,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[fastsam-hip-distance] sessions={included_sessions} "
        f"bins={len(eligible_labels)} rows={len(eligible_data)}"
    )
    print(f"[fastsam-hip-distance] output={output_dir}")
    return output_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "00_pose_pipeline_v2/configs/fastsam_hip_distance_boxplot.yaml"
        ),
    )
    args = parser.parse_args()
    write_outputs(_resolve(args.config))


if __name__ == "__main__":
    main()
