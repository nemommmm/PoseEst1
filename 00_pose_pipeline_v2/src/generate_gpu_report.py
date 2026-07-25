"""Generate bilingual, data-driven GPU deployment reports.

The public reports intentionally omit DeepStream SDK throughput, latency, and
competitive benchmark figures. The NVIDIA BodyPose3DNet section is limited to
the project-specific stereo geometry and external-comparison accuracy gate.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT / "00_pose_pipeline_v2/runs/gpu_rebuild_20260725"
)

COLORS = {
    "pytorch": "#087f5b",
    "trt_fp32": "#155eef",
    "trt_fp16": "#d97706",
    "grid": "#d8dee8",
    "muted": "#65758b",
    "bad": "#b42318",
}


@dataclass(frozen=True)
class BackendResult:
    """One repeated GPU backend benchmark and its equivalence result."""

    key: str
    dataset: str
    backend: str
    family: str
    repeat_fps: tuple[float, ...]
    fps_median: float
    fps_p95: float
    fps_min: float
    fps_max: float
    pose_p50_ms: float
    pose_p95_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    repeat_deterministic: bool
    equivalence_passed: bool
    keypoint_median_cm: float
    keypoint_p95_cm: float
    angle_mae_deg: float
    angle_median_deg: float
    angle_p95_deg: float
    rula_bin_agreement: float


@dataclass(frozen=True)
class NvdecResult:
    """Public decode-only and proxy-equivalence evidence."""

    cpu_paired_fps_median: float
    nvdec_paired_fps_median: float
    decode_speedup: float
    luma_validation_passed: bool
    original_nvdec_validation_rejected: bool
    proxy_quality_label: str
    proxy_equivalence_passed: bool
    keypoint_median_cm: float
    keypoint_p95_cm: float
    angle_mae_deg: float
    angle_median_deg: float
    angle_p95_deg: float
    rula_bin_agreement: float


@dataclass(frozen=True)
class NvidiaAccuracyResult:
    """Permitted project-specific BodyPose3DNet accuracy-gate values."""

    frame_count: int
    matched_frame_count: int
    candidate_mae_deg: float
    candidate_median_deg: float
    candidate_p95_deg: float
    candidate_rula_agreement: float
    control_mae_deg: float
    control_median_deg: float
    control_p95_deg: float
    control_rula_agreement: float
    mae_improvement_percent: float
    epipolar_median_px: float
    epipolar_p95_px: float
    epipolar_median_limit_px: float
    epipolar_p95_limit_px: float
    finite_3d_joint_ratio: float
    geometry_gate_passed: bool
    decision: str


@dataclass(frozen=True)
class ReportData:
    """All structured evidence used by the two public HTML reports."""

    backends: tuple[BackendResult, ...]
    nvdec: NvdecResult
    nvidia_accuracy: NvidiaAccuracyResult
    nvidia_performance_decision: str
    pytorch_environment: Mapping[str, Any]
    nvidia_environment: Mapping[str, Any]
    pytorch_artifact_manifest: Mapping[str, Any]
    tensorrt_artifact_manifest: Mapping[str, Any]
    tensorrt_export_manifest: Mapping[str, Any]
    run_root: Path


def configure_fonts(language: str) -> None:
    """Use a readable local font for the requested report language."""
    matplotlib.rcParams["font.style"] = "normal"
    matplotlib.rcParams["font.weight"] = "normal"
    if language == "en":
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["axes.unicode_minus"] = False
        return
    candidates = (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path.home() / "Library/Fonts/TencentSans-W7.ttf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            matplotlib.rcParams["font.family"] = family
            matplotlib.rcParams["axes.unicode_minus"] = False
            break


def read_json(path: Path) -> Mapping[str, Any]:
    """Read a required JSON object with a useful failure message."""
    if not path.is_file():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    """Return a required nested value."""
    value: Any = mapping
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(value, Mapping) or key not in value:
            dotted = ".".join(traversed)
            raise KeyError(f"Missing required report field: {dotted}")
        value = value[key]
    return value


def as_float(value: Any) -> float:
    """Convert a required numeric value to float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric report value, got {value!r}")
    return float(value)


def as_int(value: Any) -> int:
    """Convert a required integer-like value to int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer report value, got {value!r}")
    return int(value)


def backend_identity(key: str, family: str) -> tuple[str, str, str]:
    """Return dataset, public backend name, and chart family."""
    dataset = "Fanbo7 A257" if key.startswith("fanbo7") else "Fanbo4 A257"
    if family == "pytorch":
        return dataset, "PyTorch FP32", "pytorch"
    if key.endswith("fp16"):
        return dataset, "TensorRT FP16", "trt_fp16"
    return dataset, "TensorRT FP32", "trt_fp32"


def load_backend_suite(path: Path, family: str) -> list[BackendResult]:
    """Load one repeated benchmark suite."""
    suite = read_json(path)
    datasets = nested(suite, "datasets")
    if not isinstance(datasets, Mapping):
        raise TypeError(f"Expected datasets object in {path}")

    results: list[BackendResult] = []
    for key, payload_any in datasets.items():
        if not isinstance(key, str) or not isinstance(payload_any, Mapping):
            raise TypeError(f"Invalid dataset entry in {path}")
        payload = payload_any
        dataset, backend, chart_family = backend_identity(key, family)
        timing = nested(payload, "timing_aggregate")
        fps = nested(timing, "online_fps_across_repeats")
        stages = nested(timing, "stages")
        pose = nested(stages, "pose_inference_stereo", "pooled_frame_ms")
        e2e = nested(stages, "end_to_end_online", "pooled_frame_ms")
        records = nested(payload, "records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"No benchmark records for {key}")
        repeat_fps = tuple(
            as_float(nested(record, "benchmark", "online_fps"))
            for record in records
        )
        comparison = nested(records[0], "historical_comparison")
        keypoint = nested(comparison, "keypoints", "distance_cm")
        angle = nested(comparison, "angle", "absolute_difference_deg")

        results.append(
            BackendResult(
                key=key,
                dataset=dataset,
                backend=backend,
                family=chart_family,
                repeat_fps=repeat_fps,
                fps_median=as_float(nested(fps, "median")),
                fps_p95=as_float(nested(fps, "p95")),
                fps_min=as_float(nested(fps, "min")),
                fps_max=as_float(nested(fps, "max")),
                pose_p50_ms=as_float(nested(pose, "median")),
                pose_p95_ms=as_float(nested(pose, "p95")),
                e2e_p50_ms=as_float(nested(e2e, "median")),
                e2e_p95_ms=as_float(nested(e2e, "p95")),
                repeat_deterministic=bool(
                    nested(payload, "all_repeats_deterministic")
                ),
                equivalence_passed=bool(
                    nested(payload, "all_historical_comparisons_pass")
                ),
                keypoint_median_cm=as_float(nested(keypoint, "median")),
                keypoint_p95_cm=as_float(nested(keypoint, "p95")),
                angle_mae_deg=as_float(nested(angle, "mean")),
                angle_median_deg=as_float(nested(angle, "median")),
                angle_p95_deg=as_float(nested(angle, "p95")),
                rula_bin_agreement=as_float(
                    nested(comparison, "angle", "rula_bin_agreement")
                ),
            )
        )
    return results


def load_nvdec(run_root: Path) -> NvdecResult:
    """Load the valid decode-only test and the rejected proxy gate."""
    decode = read_json(run_root / "decode/fanbo7_proxy_cpu_nvdec_v2.json")
    original = read_json(run_root / "decode/fanbo7_cpu_nvdec.json")
    equivalence = read_json(
        run_root
        / "nvdec_proxy_equivalence/fanbo7_pt_240proxy/source_equivalence.json"
    )
    proxy_manifest = read_json(
        run_root
        / "nvdec_proxy_manifests/cap_5_0_nvdec_240.mp4.manifest.json"
    )

    backends = nested(decode, "backends")
    if not isinstance(backends, list):
        raise TypeError("NVDEC benchmark backends must be a list")
    by_name = {
        str(nested(item, "backend")): item
        for item in backends
        if isinstance(item, Mapping)
    }
    cpu = by_name["cpu"]
    nvdec = by_name["nvdec"]
    luma_status = str(nested(decode, "luma_sample_validation", "status"))
    original_validation = str(
        nested(original, "rgb_sample_validation", "status")
    )

    keypoint = nested(equivalence, "keypoints", "distance_cm")
    angle = nested(equivalence, "angle", "absolute_difference_deg")
    return NvdecResult(
        cpu_paired_fps_median=as_float(
            nested(cpu, "paired_effective_fps", "median")
        ),
        nvdec_paired_fps_median=as_float(
            nested(nvdec, "paired_effective_fps", "median")
        ),
        decode_speedup=as_float(
            nested(decode, "speedup", "nvdec_over_cpu_paired_median")
        ),
        luma_validation_passed=luma_status == "passed",
        original_nvdec_validation_rejected=original_validation == "rejected",
        proxy_quality_label=str(
            nested(proxy_manifest, "quality_classification", "label")
        ),
        proxy_equivalence_passed=bool(
            nested(equivalence, "passes_thresholds")
        ),
        keypoint_median_cm=as_float(nested(keypoint, "median")),
        keypoint_p95_cm=as_float(nested(keypoint, "p95")),
        angle_mae_deg=as_float(nested(angle, "mean")),
        angle_median_deg=as_float(nested(angle, "median")),
        angle_p95_deg=as_float(nested(angle, "p95")),
        rula_bin_agreement=as_float(
            nested(equivalence, "angle", "rula_bin_agreement")
        ),
    )


def load_nvidia_accuracy(path: Path) -> NvidiaAccuracyResult:
    """Load only the permitted NVIDIA accuracy and geometry fields."""
    metrics = read_json(path)
    same_input = nested(metrics, "same_input_yolo_control")
    candidate = nested(same_input, "candidate_right_elbow_matched")
    control = nested(same_input, "control_right_elbow_matched")
    candidate_diff = nested(candidate, "absolute_difference")
    control_diff = nested(control, "absolute_difference")
    epipolar = nested(metrics, "geometry", "epipolar_pre_px")
    gate = nested(metrics, "geometry", "geometry_gate")
    limits = nested(gate, "thresholds")
    return NvidiaAccuracyResult(
        frame_count=as_int(nested(metrics, "scope", "frame_count")),
        matched_frame_count=as_int(
            nested(same_input, "matched_common_finite_count")
        ),
        candidate_mae_deg=as_float(nested(candidate_diff, "mean")),
        candidate_median_deg=as_float(nested(candidate_diff, "median")),
        candidate_p95_deg=as_float(nested(candidate_diff, "p95")),
        candidate_rula_agreement=as_float(
            nested(candidate, "rula_bin_agreement")
        ),
        control_mae_deg=as_float(nested(control_diff, "mean")),
        control_median_deg=as_float(nested(control_diff, "median")),
        control_p95_deg=as_float(nested(control_diff, "p95")),
        control_rula_agreement=as_float(
            nested(control, "rula_bin_agreement")
        ),
        mae_improvement_percent=as_float(
            nested(same_input, "candidate_mae_improvement_percent_matched")
        ),
        epipolar_median_px=as_float(nested(epipolar, "median")),
        epipolar_p95_px=as_float(nested(epipolar, "p95")),
        epipolar_median_limit_px=as_float(
            nested(limits, "maximum_median_epipolar_px")
        ),
        epipolar_p95_limit_px=as_float(
            nested(limits, "maximum_p95_epipolar_px")
        ),
        finite_3d_joint_ratio=as_float(
            nested(metrics, "geometry", "finite_3d_joint_ratio")
        ),
        geometry_gate_passed=bool(nested(gate, "passed")),
        decision=str(nested(metrics, "decision")),
    )


def load_report_data(run_root: Path) -> ReportData:
    """Load and validate all formal report inputs."""
    pytorch_root = run_root / "pytorch_formal"
    tensorrt_root = run_root / "tensorrt_formal"
    nvidia_root = run_root / "nvidia_bodypose3d_feasibility"

    backends = load_backend_suite(
        pytorch_root / "suite_summary.json", "pytorch"
    )
    backends.extend(
        load_backend_suite(
            tensorrt_root / "suite_summary.json", "tensorrt"
        )
    )
    order = {
        ("Fanbo7 A257", "PyTorch FP32"): 0,
        ("Fanbo7 A257", "TensorRT FP32"): 1,
        ("Fanbo7 A257", "TensorRT FP16"): 2,
        ("Fanbo4 A257", "PyTorch FP32"): 3,
        ("Fanbo4 A257", "TensorRT FP32"): 4,
        ("Fanbo4 A257", "TensorRT FP16"): 5,
    }
    backends.sort(key=lambda item: order[(item.dataset, item.backend)])

    performance_metrics = read_json(
        nvidia_root
        / "performance_full433_formal_fixed_reference/metrics.json"
    )
    return ReportData(
        backends=tuple(backends),
        nvdec=load_nvdec(run_root),
        nvidia_accuracy=load_nvidia_accuracy(
            nvidia_root
            / "accuracy_full433_formal_fixed_reference/metrics.json"
        ),
        nvidia_performance_decision=str(
            nested(performance_metrics, "decision")
        ),
        pytorch_environment=read_json(pytorch_root / "environment.json"),
        nvidia_environment=read_json(
            nvidia_root / "nvidia_environment_manifest.json"
        ),
        pytorch_artifact_manifest=read_json(
            pytorch_root / "artifact_manifest.json"
        ),
        tensorrt_artifact_manifest=read_json(
            tensorrt_root / "artifact_manifest.json"
        ),
        tensorrt_export_manifest=read_json(
            run_root / "tensorrt_exports/export_manifest.json"
        ),
        run_root=run_root,
    )


def figure_uri(fig: plt.Figure) -> str:
    """Return a PNG figure encoded as a data URI."""
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def dataset_results(
    data: ReportData, dataset: str
) -> tuple[BackendResult, ...]:
    """Return results for one dataset in public backend order."""
    return tuple(item for item in data.backends if item.dataset == dataset)


def repeat_fps_chart(data: ReportData, language: str) -> str:
    """Plot every repeat and its median rather than hiding n=3."""
    labels = {
        "en": {
            "ylabel": "Paired-file processing throughput (fps)",
            "median": "median",
        },
        "zh": {
            "ylabel": "双路文件处理吞吐率（fps）",
            "median": "中位数",
        },
    }[language]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    all_repeat_fps = [
        fps for item in data.backends for fps in item.repeat_fps
    ]
    for ax, dataset in zip(
        axes, ("Fanbo7 A257", "Fanbo4 A257"), strict=True
    ):
        items = dataset_results(data, dataset)
        for x_pos, item in enumerate(items):
            offsets = (-0.09, 0.0, 0.09)
            for index, fps in enumerate(item.repeat_fps):
                offset = offsets[index] if index < len(offsets) else 0.0
                ax.scatter(
                    x_pos + offset,
                    fps,
                    s=46,
                    color=COLORS[item.family],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
            ax.plot(
                [x_pos - 0.22, x_pos + 0.22],
                [item.fps_median, item.fps_median],
                color="#172033",
                linewidth=2.2,
                zorder=4,
            )
            ax.annotate(
                f"{item.fps_median:.2f}",
                (x_pos, item.fps_median),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_title(dataset)
        ax.set_xticks(
            range(len(items)),
            [item.backend.replace("TensorRT ", "TRT\n") for item in items],
        )
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(min(all_repeat_fps) - 0.65, max(all_repeat_fps) + 0.65)
    axes[0].set_ylabel(labels["ylabel"])
    fig.tight_layout()
    return figure_uri(fig)


def latency_chart(data: ReportData, language: str) -> str:
    """Plot pooled p50-to-p95 pose and file-path latency intervals."""
    labels = {
        "en": {
            "xlabel": "Warm-excluded latency (ms), pooled over 570 frames",
            "pose": "stereo pose",
            "e2e": "paired-file path",
        },
        "zh": {
            "xlabel": "去除 warm-up 后的延迟（ms，570 帧合并统计）",
            "pose": "双路姿态",
            "e2e": "双路文件路径",
        },
    }[language]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharex=True)
    for ax, dataset in zip(
        axes, ("Fanbo7 A257", "Fanbo4 A257"), strict=True
    ):
        items = dataset_results(data, dataset)
        for y_pos, item in enumerate(items):
            for offset, p50, p95, color, marker in (
                (
                    -0.12,
                    item.pose_p50_ms,
                    item.pose_p95_ms,
                    "#4c78a8",
                    "o",
                ),
                (
                    0.12,
                    item.e2e_p50_ms,
                    item.e2e_p95_ms,
                    "#2a9d8f",
                    "s",
                ),
            ):
                ax.plot(
                    [p50, p95],
                    [y_pos + offset, y_pos + offset],
                    color=color,
                    linewidth=2.4,
                    solid_capstyle="round",
                )
                ax.scatter(
                    p50,
                    y_pos + offset,
                    s=42,
                    color=color,
                    marker=marker,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.scatter(
                    p95,
                    y_pos + offset,
                    s=42,
                    facecolor="white",
                    edgecolor=color,
                    marker=marker,
                    linewidth=1.4,
                    zorder=3,
                )
        ax.set_title(dataset)
        ax.set_yticks(range(len(items)), [item.backend for item in items])
        ax.invert_yaxis()
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlabel(labels["xlabel"])
    axes[1].set_xlabel(labels["xlabel"])
    pose_handle = axes[1].plot(
        [], [], color="#4c78a8", marker="o", label=labels["pose"]
    )[0]
    file_handle = axes[1].plot(
        [], [], color="#2a9d8f", marker="s", label=labels["e2e"]
    )[0]
    fig.legend(
        handles=[pose_handle, file_handle],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return figure_uri(fig)


def speed_equivalence_chart(data: ReportData, language: str) -> str:
    """Plot speed against backend-output disagreement with PyTorch."""
    labels = {
        "en": {
            "xlabel": "Median paired-file throughput (fps)",
            "ylabel": "Right-elbow MAE vs deterministic PyTorch (°)",
            "accepted": "accepted reference",
            "rejected": "equivalence rejected",
        },
        "zh": {
            "xlabel": "双路文件吞吐率中位数（fps）",
            "ylabel": "相对确定性 PyTorch 的右肘 MAE（°）",
            "accepted": "已接受参考",
            "rejected": "等价性拒绝",
        },
    }[language]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    dataset_markers = {"Fanbo7 A257": "o", "Fanbo4 A257": "s"}
    for item in data.backends:
        accepted = item.family == "pytorch" and item.equivalence_passed
        marker = dataset_markers[item.dataset] if accepted else "X"
        color = COLORS[item.family] if accepted else COLORS["bad"]
        size = 95 if accepted else 110
        ax.scatter(
            item.fps_median,
            item.angle_mae_deg,
            s=size,
            marker=marker,
            color=color,
            edgecolor="white" if marker != "X" else color,
            linewidth=0.9,
            zorder=3,
        )
        short_dataset = item.dataset.split()[0]
        short_backend = item.backend.replace("TensorRT ", "TRT ")
        if accepted:
            dx = -8
            dy = 10 if item.dataset.startswith("Fanbo7") else 24
            ha = "right"
        else:
            dx = 6 if item.dataset.startswith("Fanbo7") else -6
            dy = 6
            ha = "left" if dx > 0 else "right"
        ax.annotate(
            f"{short_dataset} {short_backend}",
            (item.fps_median, item.angle_mae_deg),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=8,
        )
    ax.axhline(0, color=COLORS["pytorch"], linewidth=1.2)
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], marker="o", color=COLORS["pytorch"], label=labels["accepted"])
    ax.scatter([], [], marker="X", color=COLORS["bad"], label=labels["rejected"])
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return figure_uri(fig)


def fmt(value: float, digits: int = 3) -> str:
    """Format a report number consistently."""
    return f"{value:.{digits}f}"


def artifact_table(data: ReportData, language: str) -> str:
    """Return a concise public provenance table."""
    pt = data.pytorch_artifact_manifest
    trt = data.tensorrt_artifact_manifest
    export = data.tensorrt_export_manifest
    source_sha = str(nested(export, "source_model", "sha256"))
    if language == "en":
        return f"""
        <div class="table-wrap"><table>
          <thead><tr><th>Evidence</th><th>Files</th><th>Recorded commit / hash</th></tr></thead>
          <tbody>
            <tr><td>Repeated PyTorch suite</td><td>{as_int(nested(pt, "file_count"))}</td><td><code>{str(nested(pt, "git_commit"))[:7]}</code></td></tr>
            <tr><td>Repeated TensorRT suite</td><td>{as_int(nested(trt, "file_count"))}</td><td><code>{str(nested(trt, "git_commit"))[:7]}</code></td></tr>
            <tr><td>YOLOv8m source weight</td><td>manifest only</td><td><code>{source_sha[:16]}…</code></td></tr>
            <tr><td>NVIDIA feasibility route</td><td>models/engines excluded locally</td><td><code>{str(nested(data.nvidia_environment, "project", "evaluation_commit"))[:7]}</code></td></tr>
          </tbody>
        </table></div>"""
    return f"""
    <div class="table-wrap"><table>
      <thead><tr><th>证据</th><th>文件数</th><th>记录的 commit / hash</th></tr></thead>
      <tbody>
        <tr><td>PyTorch 三次重复套件</td><td>{as_int(nested(pt, "file_count"))}</td><td><code>{str(nested(pt, "git_commit"))[:7]}</code></td></tr>
        <tr><td>TensorRT 三次重复套件</td><td>{as_int(nested(trt, "file_count"))}</td><td><code>{str(nested(trt, "git_commit"))[:7]}</code></td></tr>
        <tr><td>YOLOv8m 源权重</td><td>仅保留 manifest</td><td><code>{source_sha[:16]}…</code></td></tr>
        <tr><td>NVIDIA 可行性路线</td><td>本地排除模型与 engine</td><td><code>{str(nested(data.nvidia_environment, "project", "evaluation_commit"))[:7]}</code></td></tr>
      </tbody>
    </table></div>"""


def benchmark_rows(
    results: Sequence[BackendResult], language: str, pytorch_only: bool
) -> str:
    """Render repeated benchmark rows."""
    rows: list[str] = []
    for item in results:
        if pytorch_only != (item.family == "pytorch"):
            continue
        repeats = " / ".join(f"{value:.2f}" for value in item.repeat_fps)
        if item.equivalence_passed:
            decision = (
                '<span class="status accepted">Exact</span>'
                if language == "en"
                else '<span class="status accepted">完全一致</span>'
            )
        else:
            decision = (
                '<span class="status rejected">Rejected</span>'
                if language == "en"
                else '<span class="status rejected">拒绝</span>'
            )
        if pytorch_only:
            rows.append(
                f"""<tr>
                <td>{item.dataset}</td><td>{repeats}</td>
                <td>{item.fps_median:.3f}</td>
                <td>{item.pose_p50_ms:.3f} / {item.pose_p95_ms:.3f}</td>
                <td>{item.e2e_p50_ms:.3f} / {item.e2e_p95_ms:.3f}</td>
                <td>{decision}</td></tr>"""
            )
        else:
            rows.append(
                f"""<tr>
                <td>{item.dataset}</td><td>{item.backend}</td>
                <td>{item.fps_median:.3f}</td>
                <td>{item.pose_p50_ms:.3f} / {item.pose_p95_ms:.3f}</td>
                <td>{item.e2e_p50_ms:.3f} / {item.e2e_p95_ms:.3f}</td>
                <td>{item.keypoint_median_cm:.3f} / {item.keypoint_p95_cm:.3f}</td>
                <td>{item.angle_mae_deg:.3f} / {item.angle_median_deg:.3f} / {item.angle_p95_deg:.3f}</td>
                <td>{item.rula_bin_agreement:.3f}</td><td>{decision}</td></tr>"""
            )
    return "\n".join(rows)


def common_css() -> str:
    """Return the self-contained report stylesheet."""
    return """
    :root{
      --ink:#172033;--muted:#5d6b7a;--line:#dce3ec;--blue:#155eef;
      --navy:#102a43;--green:#087f5b;--red:#b42318;--amber:#b45309;
      --bg:#f3f6fb;--panel:#ffffff;--soft:#f6f8fb
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{margin:0;background:var(--bg);color:var(--ink);
      font:15px/1.68 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}
    header{background:linear-gradient(128deg,#102a43 0%,#155eef 72%,#4f7cff 100%);
      color:#fff;padding:50px max(6vw,28px) 44px}
    header h1{max-width:1050px;margin:0 0 10px;font-size:34px;line-height:1.24}
    header p{max-width:940px;margin:0;opacity:.92}
    .layout{display:grid;grid-template-columns:225px minmax(0,980px);gap:28px;
      max-width:1280px;margin:28px auto;padding:0 22px}
    nav{position:sticky;top:18px;align-self:start;background:var(--panel);
      border:1px solid var(--line);border-radius:12px;padding:16px}
    nav strong{display:block;margin-bottom:7px;color:var(--navy)}
    nav a{display:block;color:#40566d;text-decoration:none;padding:5px 3px}
    nav a:hover{color:var(--blue)}
    main{min-width:0}
    section{background:var(--panel);border:1px solid var(--line);border-radius:14px;
      padding:26px 30px;margin-bottom:20px;box-shadow:0 5px 20px #26354d0a}
    h2{margin:0 0 14px;color:var(--navy);font-size:23px}
    h3{margin:22px 0 7px;font-size:18px;color:#25364b}
    p{margin:8px 0 14px}
    .lead{font-size:17px}
    .cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin:18px 0}
    .card{border:1px solid var(--line);border-radius:10px;padding:14px;background:#fbfcfe}
    .metric{font-size:24px;font-weight:750;color:var(--blue);line-height:1.25;margin-bottom:4px}
    .card small{color:var(--muted)}
    .callout{border-left:4px solid var(--blue);background:#f1f6ff;padding:13px 16px;margin:16px 0}
    .callout.warning{border-left-color:var(--amber);background:#fff8e8}
    .callout.danger{border-left-color:var(--red);background:#fff3f2}
    .status{display:inline-block;border-radius:999px;padding:2px 8px;font-weight:700;font-size:12px}
    .status.accepted{color:#086b4c;background:#dff7eb}
    .status.rejected{color:#9f2118;background:#fee4e2}
    .status.limited{color:#8c4d04;background:#fff0cf}
    .table-wrap{overflow-x:auto;margin:13px 0 19px}
    table{width:100%;border-collapse:collapse;min-width:690px;font-size:13.5px}
    th{background:#eef4ff;text-align:left;color:#25364b}
    th,td{border:1px solid var(--line);padding:8px 9px;vertical-align:top}
    tbody tr:nth-child(even) td{background:#fafbfd}
    figure{margin:22px 0}
    figure img{display:block;width:100%;height:auto;border:1px solid var(--line);
      border-radius:10px;background:#fff}
    figcaption{color:var(--muted);font-size:13px;margin-top:7px}
    code{background:#edf1f7;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}
    ul,ol{padding-left:22px}
    li{margin:5px 0}
    a{color:var(--blue)}
    footer{color:var(--muted);font-size:13px;padding:3px 0 28px}
    .note{color:var(--muted);font-size:13px}
    @media(max-width:870px){
      .layout{display:block}.layout nav{position:static;margin-bottom:18px}
      .cards{grid-template-columns:1fr}section{padding:21px 17px}
      header h1{font-size:28px}
    }
    @media print{
      body{background:#fff}.layout{display:block;max-width:none;margin:0;padding:0}
      nav{display:none}section{box-shadow:none;break-inside:avoid}
      header{background:#102a43!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}
    }
    """


def build_html(data: ReportData, language: str) -> str:
    """Build one self-contained public report."""
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported report language: {language}")
    configure_fonts(language)
    repeat_chart = repeat_fps_chart(data, language)
    latency = latency_chart(data, language)
    pareto = speed_equivalence_chart(data, language)

    pt_results = [item for item in data.backends if item.family == "pytorch"]
    trt_results = [item for item in data.backends if item.family != "pytorch"]
    pt_low = min(item.fps_median for item in pt_results)
    pt_high = max(item.fps_median for item in pt_results)
    exact_runs = sum(
        len(item.repeat_fps)
        for item in pt_results
        if item.equivalence_passed and item.repeat_deterministic
    )
    accuracy = data.nvidia_accuracy
    nvdec = data.nvdec
    gpu = nested(data.pytorch_environment, "nvidia_smi")[0]
    if not isinstance(gpu, Mapping):
        raise TypeError("Expected one GPU environment object")

    if language == "en":
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Data-driven A6000 GPU deployment and stereo-pose feasibility report">
<title>Stereo Pose GPU Deployment and Feasibility Report</title><style>{common_css()}</style></head>
<body><header><h1>Stereo Pose Pipeline: A6000 Deployment and Model Admission</h1>
<p>Formal evidence refreshed 2026-07-25 · Fanbo7 near view and Fanbo4 far view · joint angles and RULA are the deployment priorities.<br>
Xsens is treated only as an external comparison system / Xsens-derived reference, never as absolute ground truth.</p></header>
<div class="layout"><nav><strong>Contents</strong>
<a href="#summary">1. Executive summary</a><a href="#scope">2. Scope and environment</a>
<a href="#pytorch">3. Repeated PyTorch baseline</a><a href="#tensorrt">4. TensorRT gate</a>
<a href="#nvdec">5. Decode-only NVDEC test</a><a href="#nvidia">6. NVIDIA pose feasibility</a>
<a href="#decision">7. Decision and provenance</a></nav><main>

<section id="summary"><h2>1. Executive summary</h2>
<p class="lead">The accepted remote backend remains <strong>deterministic PyTorch FP32 + YOLOv8m-pose + SKT</strong>. It reproduced the accepted historical output exactly in all repeated tests and sustained approximately 30 fps while processing two stored video files.</p>
<div class="cards">
  <div class="card"><div class="metric">{pt_low:.2f}–{pt_high:.2f} fps</div><small>median paired-file throughput across Fanbo7/Fanbo4</small></div>
  <div class="card"><div class="metric">{exact_runs}/{exact_runs} exact</div><small>PyTorch repeats versus the deterministic historical reference</small></div>
  <div class="card"><div class="metric">{accuracy.candidate_mae_deg:.2f}° vs {accuracy.control_mae_deg:.2f}°</div><small>preliminary same-input BodyPose3DNet Accuracy signal; geometry gate still failed</small></div>
</div>
<div class="callout"><strong>Important scope:</strong> the reported pipeline FPS is offline throughput for two stored files, including decode, pose, tracking, and per-frame geometry. It is not a completed live system with camera capture, display, transport, and downstream output.</div>
<ul>
  <li><strong>TensorRT FP32/FP16:</strong> deterministic across repeats, but every candidate failed output equivalence and showed no stable end-to-end throughput gain.</li>
  <li><strong>NVDEC:</strong> faster in a decode-only test on a compatible proxy, but that proxy changed the reconstructed pose and was rejected as a formal-input replacement.</li>
  <li><strong>NVIDIA BodyPose3DNet Accuracy:</strong> showed lower right-elbow disagreement on the matched same-input frames, but failed the stereo epipolar gate; it is not admitted.</li>
</ul></section>

<section id="scope"><h2>2. Scope, measurement rules, and environment</h2>
<div class="table-wrap"><table><tbody>
<tr><th>GPU</th><td>{nested(gpu, "name")}, {nested(gpu, "memory_total_mib")} MiB, driver {nested(gpu, "driver")}, compute capability {nested(gpu, "compute_capability")}</td></tr>
<tr><th>Accepted pipeline environment</th><td>Python {str(nested(data.pytorch_environment, "python")).split()[0]}; PyTorch {nested(data.pytorch_environment, "packages", "torch")}; CUDA {nested(data.pytorch_environment, "cuda", "runtime")}; Ultralytics {nested(data.pytorch_environment, "packages", "ultralytics")}; OpenCV {nested(data.pytorch_environment, "packages", "opencv-python")}</td></tr>
<tr><th>NVIDIA feasibility environment</th><td>Ubuntu {nested(data.nvidia_environment, "software", "os").replace("Ubuntu ", "")}; DeepStream {nested(data.nvidia_environment, "software", "deepstream")}; CUDA {nested(data.nvidia_environment, "software", "cuda_runtime")}; TensorRT {nested(data.nvidia_environment, "software", "tensorrt")}; GStreamer {nested(data.nvidia_environment, "software", "gstreamer")}</td></tr>
<tr><th>Repeated timing</th><td>200 frames per run, 10 warm-up frames, three independent repeats. FPS is the median across repeats; p50/p95 latency pools 570 warm-excluded frames.</td></tr>
<tr><th>Accuracy terminology</th><td>Backend equivalence is measured against deterministic PyTorch. BodyPose3DNet angle values are agreement with the Xsens-derived external comparison on one fixed, matched frame set; no candidate-specific alignment retuning was used.</td></tr>
</tbody></table></div>
<p class="note">PyTorch and CUDA came from the RunPod base image; application packages and all observed versions were recorded. The PyTorch suite recorded a dirty worktree at commit <code>{str(nested(data.pytorch_environment, "git_commit"))[:7]}</code>, but the benchmark script, configuration, videos, calibration, weights, and reference NPZ were separately SHA256-recorded in the input manifest.</p></section>

<section id="pytorch"><h2>3. Repeated deterministic PyTorch baseline</h2>
<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Repeat FPS (1 / 2 / 3)</th><th>Median FPS</th><th>Pose p50 / p95 ms</th><th>File-path p50 / p95 ms</th><th>Historical equivalence</th></tr></thead>
<tbody>{benchmark_rows(data.backends, "en", True)}</tbody></table></div>
<figure><img src="{repeat_chart}" alt="Three throughput repeats for PyTorch and TensorRT on Fanbo7 and Fanbo4"><figcaption>Figure 1. All three repeat values are shown; the short dark line is the median. These are paired-file processing results, not live-camera FPS.</figcaption></figure>
<p>Both PyTorch datasets were deterministic across all repeats. Key arrays, 3D joints, the right-elbow trajectory, and RULA bins were exactly equal to the accepted 2026-07-12 deterministic PyTorch reference.</p>
<figure><img src="{latency}" alt="Pooled p50 to p95 latency intervals by backend"><figcaption>Figure 2. Filled marks are p50 and hollow marks are p95. Pose and paired-file intervals are shown separately because percentiles must not be added.</figcaption></figure></section>

<section id="tensorrt"><h2>4. TensorRT FP32/FP16 equivalence gate</h2>
<div class="callout danger"><strong>Decision: reject both TensorRT precisions.</strong> All 12 candidate executions completed and were repeat-deterministic, but all four dataset/precision candidates failed equivalence with the accepted deterministic PyTorch output.</div>
<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Backend</th><th>Median FPS</th><th>Pose p50 / p95 ms</th><th>File-path p50 / p95 ms</th><th>3D delta med / p95 cm</th><th>Elbow delta MAE / med / p95 °</th><th>RULA-bin agreement</th><th>Decision</th></tr></thead>
<tbody>{benchmark_rows(data.backends, "en", False)}</tbody></table></div>
<figure><img src="{pareto}" alt="Throughput versus right-elbow output equivalence"><figcaption>Figure 3. The vertical metric is backend disagreement with deterministic PyTorch, not disagreement with Xsens. Lower is better; the PyTorch reference is zero by construction.</figcaption></figure>
<p>FP16 lowered median pose latency, but its p95 latency remained high and the full paired-file throughput did not improve consistently. The crop-tracked sequence amplified small inference differences into centimetre-scale 3D and multi-degree angle changes.</p></section>

<section id="nvdec"><h2>5. NVDEC: decode-only benefit, rejected proxy</h2>
<div class="table-wrap"><table><thead><tr><th>Check</th><th>Formal result</th><th>Interpretation</th></tr></thead><tbody>
<tr><td>Original QP0 input through NVDEC</td><td><span class="status rejected">Pixel validation rejected</span></td><td>The source uses an incompatible H.264 profile. Its apparent speed-up is invalid and is intentionally not reported.</td></tr>
<tr><td>Compatible near-lossless proxy, CPU decode</td><td>{nvdec.cpu_paired_fps_median:.3f} paired fps median</td><td rowspan="2"><strong>Decode-only</strong> FFmpeg test. It excludes timestamp synchronization, tensor preprocessing, pose inference, stereo geometry, angles, and RULA.</td></tr>
<tr><td>Compatible near-lossless proxy, NVDEC</td><td>{nvdec.nvdec_paired_fps_median:.3f} paired fps median ({nvdec.decode_speedup:.3f}×); luma check {"passed" if nvdec.luma_validation_passed else "failed"}</td></tr>
<tr><td>Proxy versus formal QP0 pose output</td><td>3D delta {nvdec.keypoint_median_cm:.3f}/{nvdec.keypoint_p95_cm:.3f} cm med/p95; elbow delta {nvdec.angle_mae_deg:.3f}/{nvdec.angle_median_deg:.3f}/{nvdec.angle_p95_deg:.3f}° MAE/med/p95</td><td><span class="status rejected">Equivalence failed</span>; RULA-bin agreement {nvdec.rula_bin_agreement:.3f} alone was insufficient.</td></tr>
</tbody></table></div>
<p>The compatible proxy is explicitly classified as <strong>{nvdec.proxy_quality_label}</strong>, not lossless. A future live implementation should use a sensor-native NVDEC-compatible stream or a direct raw/GPU upload path, followed by the same complete pose-equivalence gate.</p></section>

<section id="nvidia"><h2>6. NVIDIA BodyPose3DNet stereo feasibility</h2>
<div class="callout warning"><strong>Licensing boundary:</strong> this public report intentionally contains no DeepStream SDK throughput, latency, FPS, or competitive performance benchmark numbers. Such disclosure requires checking the applicable NVIDIA DeepStream EULA and obtaining any required written permission.</div>
<h3>Accuracy variant: promising angle signal, geometry gate failed</h3>
<p>The Accuracy variant and the YOLO control used the same upright near-lossless proxies, the same fixed synchronization, and the same pre-existing alignment to the Xsens-derived external comparison. The matched set contains {accuracy.matched_frame_count} frames where both candidates and the external comparison were finite.</p>
<div class="table-wrap"><table><thead><tr><th>Same-input matched metric</th><th>BodyPose3DNet Accuracy</th><th>YOLOv8m + SKT control</th></tr></thead><tbody>
<tr><td>Right-elbow absolute disagreement MAE</td><td>{accuracy.candidate_mae_deg:.3f}°</td><td>{accuracy.control_mae_deg:.3f}°</td></tr>
<tr><td>Median / p95 disagreement</td><td>{accuracy.candidate_median_deg:.3f}° / {accuracy.candidate_p95_deg:.3f}°</td><td>{accuracy.control_median_deg:.3f}° / {accuracy.control_p95_deg:.3f}°</td></tr>
<tr><td>RULA-bin agreement</td><td>{accuracy.candidate_rula_agreement:.3f}</td><td>{accuracy.control_rula_agreement:.3f}</td></tr>
</tbody></table></div>
<p>This is a {accuracy.mae_improvement_percent:.1f}% lower MAE on the matched subset, but it is only a preliminary signal—not a model admission or a ground-truth claim.</p>
<div class="callout danger"><strong>Stereo geometry failed:</strong> pre-correction epipolar median/p95 were {accuracy.epipolar_median_px:.3f}/{accuracy.epipolar_p95_px:.3f} px, above the {accuracy.epipolar_median_limit_px:.1f}/{accuracy.epipolar_p95_limit_px:.1f} px admission limits. The finite 3D-joint ratio was {accuracy.finite_3d_joint_ratio:.3f}. Therefore the Accuracy variant did not proceed to the far-view gate.</div>
<h3>Performance variant</h3>
<p><span class="status rejected">Rejected before the far-view gate</span> The formal near-view evaluation did not pass the combined angle and stereo-geometry admission criteria. Its DeepStream SDK performance numbers are not disclosed here.</p>
</section>

<section id="decision"><h2>7. Deployment decision, limitations, and provenance</h2>
<ol>
<li><strong>Keep the accepted baseline:</strong> deterministic PyTorch FP32 + YOLOv8m + SKT.</li>
<li><strong>Do not deploy the tested TensorRT engines:</strong> median model latency alone did not translate into reliable paired-file speed, and the reconstructed output changed materially.</li>
<li><strong>Treat NVDEC as an input-path engineering task:</strong> the valid 1.512× result is decode-only; the tested compatible proxy is not pose-equivalent.</li>
<li><strong>Keep BodyPose3DNet Accuracy as a research lead only:</strong> improve cross-view association and epipolar consistency before any far-view, full-data, or deployment claim.</li>
<li><strong>Do not extrapolate A6000 results to RTX 30-series or Jetson hardware:</strong> target devices require direct measurement.</li>
</ol>
<h3>Evidence index</h3>{artifact_table(data, "en")}
<p class="note">Formal inputs: <code>pytorch_formal/</code>, <code>tensorrt_formal/</code>, <code>decode/</code>, <code>nvdec_proxy_equivalence/</code>, and <code>nvidia_bodypose3d_feasibility/*_formal_fixed_reference/</code> under <code>00_pose_pipeline_v2/runs/gpu_rebuild_20260725/</code>. Licensed weights, TensorRT engines, and proxy videos were excluded from the local result package.</p>
<h3>Study limitations</h3><ul>
<li>Repeated deployment timing covers 200-frame windows from two stored stereo sequences, not a long-running live camera system.</li>
<li>The NVIDIA feasibility result covers one near-view Fanbo7 recording and near-lossless technical proxies; its stereo geometry did not pass.</li>
<li>The Xsens-derived reference has calibration and offset limitations, especially around the elbow. Reported angle values measure system agreement, not absolute truth.</li>
</ul></section>
<footer>Generated from formal structured artifacts by <code>00_pose_pipeline_v2/src/generate_gpu_report.py</code>. Chinese counterpart: <code>docs/gpu_deployment_note_CN.html</code>.</footer>
</main></div></body></html>"""

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="数据驱动的 A6000 GPU 部署与双目姿态可行性报告">
<title>双目姿态 GPU 部署与可行性报告</title><style>{common_css()}</style></head>
<body><header><h1>双目姿态 Pipeline：A6000 部署与模型准入</h1>
<p>正式证据更新于 2026-07-25 · Fanbo7 近距与 Fanbo4 远距 · 部署优先指标为关节角度与 RULA。<br>
Xsens 始终只作为 external comparison system / Xsens-derived reference，绝不视为绝对 Ground Truth。</p></header>
<div class="layout"><nav><strong>目录</strong>
<a href="#summary">1. 执行摘要</a><a href="#scope">2. 范围与环境</a>
<a href="#pytorch">3. PyTorch 三次重复</a><a href="#tensorrt">4. TensorRT 准入</a>
<a href="#nvdec">5. NVDEC 解码测试</a><a href="#nvidia">6. NVIDIA 姿态可行性</a>
<a href="#decision">7. 决策与证据</a></nav><main>

<section id="summary"><h2>1. 执行摘要</h2>
<p class="lead">当前接受的远程后端仍然是 <strong>确定性 PyTorch FP32 + YOLOv8m-pose + SKT</strong>。它在所有重复测试中完全复现了已接受的历史输出，并在处理两路存储视频文件时维持约 30 fps。</p>
<div class="cards">
  <div class="card"><div class="metric">{pt_low:.2f}–{pt_high:.2f} fps</div><small>Fanbo7/Fanbo4 双路文件吞吐率中位数</small></div>
  <div class="card"><div class="metric">{exact_runs}/{exact_runs} 完全一致</div><small>PyTorch 重复结果相对确定性历史参考</small></div>
  <div class="card"><div class="metric">{accuracy.candidate_mae_deg:.2f}° vs {accuracy.control_mae_deg:.2f}°</div><small>BodyPose3DNet Accuracy 同输入初步信号；几何门槛仍失败</small></div>
</div>
<div class="callout"><strong>范围提醒：</strong>这里的 Pipeline FPS 是两路存储文件的离线处理吞吐率，包含解码、姿态、tracking 和逐帧几何；它不等于已经包含相机采集、显示、传输和下游输出的完整 live system。</div>
<ul>
  <li><strong>TensorRT FP32/FP16：</strong>重复运行本身确定，但四个候选都未通过输出等价性，而且没有稳定的端到端吞吐收益。</li>
  <li><strong>NVDEC：</strong>在兼容代理上的纯解码测试更快，但代理改变了重建结果，不能替代正式输入。</li>
  <li><strong>NVIDIA BodyPose3DNet Accuracy：</strong>在同输入共同帧上显示了更低的右肘一致性差异，但未通过双目极线门槛，因此不准入。</li>
</ul></section>

<section id="scope"><h2>2. 范围、统计口径与环境</h2>
<div class="table-wrap"><table><tbody>
<tr><th>GPU</th><td>{nested(gpu, "name")}，{nested(gpu, "memory_total_mib")} MiB，driver {nested(gpu, "driver")}，compute capability {nested(gpu, "compute_capability")}</td></tr>
<tr><th>已接受 Pipeline 环境</th><td>Python {str(nested(data.pytorch_environment, "python")).split()[0]}；PyTorch {nested(data.pytorch_environment, "packages", "torch")}；CUDA {nested(data.pytorch_environment, "cuda", "runtime")}；Ultralytics {nested(data.pytorch_environment, "packages", "ultralytics")}；OpenCV {nested(data.pytorch_environment, "packages", "opencv-python")}</td></tr>
<tr><th>NVIDIA 可行性环境</th><td>{nested(data.nvidia_environment, "software", "os")}；DeepStream {nested(data.nvidia_environment, "software", "deepstream")}；CUDA {nested(data.nvidia_environment, "software", "cuda_runtime")}；TensorRT {nested(data.nvidia_environment, "software", "tensorrt")}；GStreamer {nested(data.nvidia_environment, "software", "gstreamer")}</td></tr>
<tr><th>重复测速</th><td>每次 200 帧、warm-up 10 帧、独立重复 3 次。FPS 取三次重复的中位数；p50/p95 延迟合并 570 个去除 warm-up 的帧。</td></tr>
<tr><th>精度术语</th><td>后端等价性只和确定性 PyTorch 比较。BodyPose3DNet 的角度数字是在一个固定共同帧集合上与 Xsens-derived external comparison 的一致性；没有为候选单独重调对齐。</td></tr>
</tbody></table></div>
<p class="note">PyTorch 与 CUDA 来自 RunPod base image；应用依赖和实际版本均有记录。PyTorch 套件在 commit <code>{str(nested(data.pytorch_environment, "git_commit"))[:7]}</code> 记录到 dirty worktree，但 benchmark 脚本、配置、视频、标定、权重和参考 NPZ 都在 input manifest 中分别记录了 SHA256。</p></section>

<section id="pytorch"><h2>3. 确定性 PyTorch 三次重复基线</h2>
<div class="table-wrap"><table><thead><tr><th>数据集</th><th>三次 FPS（1 / 2 / 3）</th><th>FPS 中位数</th><th>姿态 p50 / p95 ms</th><th>文件路径 p50 / p95 ms</th><th>历史等价性</th></tr></thead>
<tbody>{benchmark_rows(data.backends, "zh", True)}</tbody></table></div>
<figure><img src="{repeat_chart}" alt="Fanbo7 和 Fanbo4 的三次 PyTorch 与 TensorRT 吞吐重复点"><figcaption>图 1：展示全部三次结果，深色短线表示中位数。这是双路文件处理结果，不是 live camera FPS。</figcaption></figure>
<p>两个 PyTorch 数据集的三次重复都完全确定。关键数组、3D 关节、右肘轨迹和 RULA 分箱与 2026-07-12 已接受的确定性 PyTorch 参考完全一致。</p>
<figure><img src="{latency}" alt="各后端合并帧的 p50 到 p95 延迟区间"><figcaption>图 2：实心点为 p50，空心点为 p95。姿态阶段和双路文件路径分别展示，因为不同分位数不能直接相加。</figcaption></figure></section>

<section id="tensorrt"><h2>4. TensorRT FP32/FP16 等价性门槛</h2>
<div class="callout danger"><strong>决定：FP32 与 FP16 都拒绝。</strong>12 次候选执行全部完成并在各自重复间保持确定，但四个数据集/精度候选都未通过与已接受 PyTorch 输出的等价性检查。</div>
<div class="table-wrap"><table><thead><tr><th>数据集</th><th>后端</th><th>FPS 中位数</th><th>姿态 p50 / p95 ms</th><th>文件路径 p50 / p95 ms</th><th>3D 差 med / p95 cm</th><th>右肘差 MAE / med / p95 °</th><th>RULA 分箱一致率</th><th>决定</th></tr></thead>
<tbody>{benchmark_rows(data.backends, "zh", False)}</tbody></table></div>
<figure><img src="{pareto}" alt="吞吐率与右肘输出等价性的速度精度图"><figcaption>图 3：纵轴是相对确定性 PyTorch 的后端差异，不是与 Xsens 的差异。越低越好；PyTorch 作为参考按定义为 0。</figcaption></figure>
<p>FP16 降低了姿态阶段的中位延迟，但 p95 仍然偏高，完整双路文件吞吐也没有稳定改善。crop tracking 会把很小的推理差异逐帧放大，最终形成厘米级 3D 变化和数度的角度变化。</p></section>

<section id="nvdec"><h2>5. NVDEC：纯解码有收益，代理输入被拒绝</h2>
<div class="table-wrap"><table><thead><tr><th>检查</th><th>正式结果</th><th>解释</th></tr></thead><tbody>
<tr><td>原 QP0 输入直接走 NVDEC</td><td><span class="status rejected">像素验证失败</span></td><td>源文件使用了不兼容的 H.264 profile；其表面加速无效，因此不在报告中披露。</td></tr>
<tr><td>兼容 near-lossless 代理，CPU 解码</td><td>{nvdec.cpu_paired_fps_median:.3f} paired fps 中位数</td><td rowspan="2"><strong>仅为 FFmpeg decode-only</strong>。不包含时间戳同步、tensor 预处理、姿态推理、双目几何、角度与 RULA。</td></tr>
<tr><td>兼容 near-lossless 代理，NVDEC</td><td>{nvdec.nvdec_paired_fps_median:.3f} paired fps 中位数（{nvdec.decode_speedup:.3f}×）；亮度检查{"通过" if nvdec.luma_validation_passed else "失败"}</td></tr>
<tr><td>代理与正式 QP0 姿态输出</td><td>3D 差 med/p95 为 {nvdec.keypoint_median_cm:.3f}/{nvdec.keypoint_p95_cm:.3f} cm；右肘差 MAE/med/p95 为 {nvdec.angle_mae_deg:.3f}/{nvdec.angle_median_deg:.3f}/{nvdec.angle_p95_deg:.3f}°</td><td><span class="status rejected">等价性失败</span>；仅有 RULA 分箱一致率 {nvdec.rula_bin_agreement:.3f} 不足以准入。</td></tr>
</tbody></table></div>
<p>兼容代理被明确标为 <strong>{nvdec.proxy_quality_label}</strong>，不是 lossless。未来 live implementation 应使用传感器原生的 NVDEC-compatible stream，或直接走 raw/GPU upload，并重新通过完整姿态等价性门槛。</p></section>

<section id="nvidia"><h2>6. NVIDIA BodyPose3DNet 双目可行性</h2>
<div class="callout warning"><strong>许可边界：</strong>本公开报告有意不包含任何 DeepStream SDK 吞吐、延迟、FPS 或竞争性 performance benchmark 数字。公开此类数字前必须核查适用的 NVIDIA DeepStream EULA，并取得任何所需的书面许可。</div>
<h3>Accuracy 版本：角度信号有希望，但几何门槛失败</h3>
<p>Accuracy 版本与 YOLO control 使用相同的 upright near-lossless 代理、相同的固定同步，以及同一个既有的 Xsens-derived external comparison 对齐。共同集合包含 {accuracy.matched_frame_count} 帧，其中候选、control 和外部比较均为有限值。</p>
<div class="table-wrap"><table><thead><tr><th>同输入共同帧指标</th><th>BodyPose3DNet Accuracy</th><th>YOLOv8m + SKT control</th></tr></thead><tbody>
<tr><td>右肘绝对一致性差 MAE</td><td>{accuracy.candidate_mae_deg:.3f}°</td><td>{accuracy.control_mae_deg:.3f}°</td></tr>
<tr><td>一致性差 median / p95</td><td>{accuracy.candidate_median_deg:.3f}° / {accuracy.candidate_p95_deg:.3f}°</td><td>{accuracy.control_median_deg:.3f}° / {accuracy.control_p95_deg:.3f}°</td></tr>
<tr><td>RULA 分箱一致率</td><td>{accuracy.candidate_rula_agreement:.3f}</td><td>{accuracy.control_rula_agreement:.3f}</td></tr>
</tbody></table></div>
<p>候选在共同子集上的 MAE 低 {accuracy.mae_improvement_percent:.1f}%，但这只能视为初步信号，不能当作模型准入或 Ground Truth 结论。</p>
<div class="callout danger"><strong>双目几何失败：</strong>修正前极线误差 median/p95 为 {accuracy.epipolar_median_px:.3f}/{accuracy.epipolar_p95_px:.3f} px，超过 {accuracy.epipolar_median_limit_px:.1f}/{accuracy.epipolar_p95_limit_px:.1f} px 的准入上限；有限 3D 关节比例为 {accuracy.finite_3d_joint_ratio:.3f}。因此 Accuracy 版本没有进入远距门槛。</div>
<h3>Performance 版本</h3>
<p><span class="status rejected">在远距门槛前拒绝</span> 正式近距评估未通过角度与双目几何的联合准入条件。这里不披露其 DeepStream SDK performance 数字。</p>
</section>

<section id="decision"><h2>7. 部署决定、局限与证据</h2>
<ol>
<li><strong>保留已接受基线：</strong>确定性 PyTorch FP32 + YOLOv8m + SKT。</li>
<li><strong>不部署本次 TensorRT engines：</strong>仅看模型中位延迟并没有带来可靠的双路文件加速，而且重建输出发生了明显变化。</li>
<li><strong>把 NVDEC 视为输入链路工程问题：</strong>有效的 1.512× 数字只属于 decode-only；测试代理不具备姿态等价性。</li>
<li><strong>BodyPose3DNet Accuracy 只保留为研究线索：</strong>必须先改善左右关联和极线一致性，再谈远距、全数据或部署结论。</li>
<li><strong>不要把 A6000 结果直接换算到 RTX 30 系或 Jetson：</strong>目标硬件必须实测。</li>
</ol>
<h3>证据索引</h3>{artifact_table(data, "zh")}
<p class="note">正式输入位于 <code>00_pose_pipeline_v2/runs/gpu_rebuild_20260725/</code> 下的 <code>pytorch_formal/</code>、<code>tensorrt_formal/</code>、<code>decode/</code>、<code>nvdec_proxy_equivalence/</code> 和 <code>nvidia_bodypose3d_feasibility/*_formal_fixed_reference/</code>。许可权重、TensorRT engines 与代理视频均未进入本地结果包。</p>
<h3>研究局限</h3><ul>
<li>重复部署测速只覆盖两个双目存储序列的 200 帧窗口，不是长时间 live camera system。</li>
<li>NVIDIA 可行性结果只覆盖一个 Fanbo7 近距记录和 near-lossless 技术代理，而且双目几何没有通过。</li>
<li>Xsens-derived reference 存在校准和 offset 局限，肘部尤其需要谨慎；角度数字表示系统间一致性，而不是真实误差上限。</li>
</ul></section>
<footer>由 <code>00_pose_pipeline_v2/src/generate_gpu_report.py</code> 从正式结构化结果生成。英文对应版本：<code>docs/gpu_deployment_note.html</code>。</footer>
</main></div></body></html>"""


def main() -> None:
    """Generate both public reports from formal structured artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--output-en",
        type=Path,
        default=PROJECT_ROOT / "docs/gpu_deployment_note.html",
    )
    parser.add_argument(
        "--output-cn",
        type=Path,
        default=PROJECT_ROOT / "docs/gpu_deployment_note_CN.html",
    )
    args = parser.parse_args()

    data = load_report_data(args.run_root.resolve())
    outputs = (
        (args.output_en, build_html(data, "en")),
        (args.output_cn, build_html(data, "zh")),
    )
    for output, content in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(f"Report written to {output.resolve()}")


if __name__ == "__main__":
    main()
