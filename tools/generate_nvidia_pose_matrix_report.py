#!/opt/anaconda3/envs/pose/bin/python
"""Generate bilingual HTML reports for one NVIDIA pose-matrix run."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STYLE = """
:root{--ink:#172033;--muted:#607080;--line:#dce3ec;--blue:#155eef;--green:#087f5b;--red:#b42318;--amber:#b45309;--bg:#f5f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}
header{background:linear-gradient(125deg,#102a43,#155eef);color:#fff;padding:45px max(6vw,28px)}header h1{margin:0 0 9px;font-size:32px}header p{margin:0;opacity:.9}
.layout{display:grid;grid-template-columns:220px minmax(0,980px);gap:28px;max-width:1270px;margin:28px auto;padding:0 22px}
nav{position:sticky;top:18px;align-self:start;background:#fff;border:1px solid var(--line);border-radius:12px;padding:17px}
nav a{display:block;padding:6px 2px;text-decoration:none;color:#38506a}main{min-width:0}
section{background:#fff;border:1px solid var(--line);border-radius:13px;padding:25px 29px;margin-bottom:20px;box-shadow:0 4px 18px #2230470a}
h2{margin:0 0 14px;color:#102a43;font-size:23px}h3{margin:20px 0 8px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin:15px 0}
.card{border:1px solid var(--line);border-radius:9px;padding:13px}.metric{font-size:23px;font-weight:750;color:var(--blue)}
table{width:100%;border-collapse:collapse;margin:13px 0 19px;font-size:13px}th{background:#eef4ff;text-align:left}th,td{border:1px solid var(--line);padding:8px;vertical-align:top}tr:nth-child(even) td{background:#fafbfd}
.ok{color:var(--green);font-weight:700}.bad{color:var(--red);font-weight:700}.blocked{color:var(--amber);font-weight:700}
.callout{border-left:4px solid var(--blue);background:#f1f6ff;padding:12px 15px;margin:15px 0}
figure{margin:18px 0}figure img{width:100%;border:1px solid var(--line);border-radius:9px}figcaption{font-size:13px;color:var(--muted)}
details{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:9px 0}summary{cursor:pointer;font-weight:650}
code{background:#edf1f7;padding:2px 5px;border-radius:4px}a{color:var(--blue)}footer{color:var(--muted);font-size:13px;padding-bottom:28px}
@media(max-width:850px){.layout{display:block}nav{position:static;margin-bottom:18px}.cards{grid-template-columns:1fr}section{padding:20px 17px}}
"""


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON mapping."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping: {path}")
    return payload


def mean_finite(values: Sequence[float | None]) -> float | None:
    """Average finite numeric values."""
    array = np.asarray(
        [value for value in values if value is not None],
        dtype=np.float64,
    )
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def display_dataset(value: str) -> str:
    """Map the configured dataset name to its concise session name."""
    text = value.lower()
    for name in ("fanbo3", "fanbo4", "fanbo7"):
        if name in text:
            return name.capitalize()
    return value


def collect_evaluations(run_dir: Path) -> list[dict[str, Any]]:
    """Collect robust metric rows from all successful route evaluations."""
    records: list[dict[str, Any]] = []
    metric_paths = [
        path
        for path in run_dir.rglob("metrics.json")
        if "evaluation" in path.relative_to(run_dir).parts
    ]
    for path in sorted(metric_paths):
        metrics = load_json(path)
        rows = metrics.get("rows", [])
        candidate = metrics.get("candidate", path.parent.parent.name)
        candidate_median = mean_finite(
            row["candidate"]["absolute_error_deg"]["median"]
            for row in rows
        )
        baseline_median = mean_finite(
            row["baseline"]["absolute_error_deg"]["median"]
            for row in rows
        )
        candidate_p95 = mean_finite(
            row["candidate"]["absolute_error_deg"]["p95"]
            for row in rows
        )
        valid_ratio = mean_finite(
            row["candidate"]["valid_ratio"] for row in rows
        )
        rula = mean_finite(
            row["candidate"].get("rula_like_agreement") for row in rows
        )
        jumps = int(
            sum(
                int(row["candidate"].get("jump_count", 0))
                for row in rows
            )
        )
        records.append(
            {
                "candidate": candidate,
                "dataset": display_dataset(str(metrics.get("dataset", ""))),
                "median": candidate_median,
                "baseline_median": baseline_median,
                "p95": candidate_p95,
                "valid_ratio": valid_ratio,
                "rula": rula,
                "jumps": jumps,
                "improvement": metrics.get(
                    "aggregate_median_improvement_ratio"
                ),
                "metrics_path": path,
                "plot_path": path.parent / "angle_summary.png",
                "reference": metrics.get("reference", {}),
            }
        )
    return records


def collect_timings(run_dir: Path) -> dict[str, list[float]]:
    """Collect publishable open-model pipeline FPS values."""
    result: dict[str, list[float]] = {}
    for path in sorted((run_dir / "baseline").glob("*/benchmark.json")):
        payload = load_json(path)
        result.setdefault("YOLOv8m+SKT", []).append(
            float(payload["online_fps"])
        )
    for path in sorted(run_dir.glob("dense_stereo/**/timing_internal.json")):
        payload = load_json(path)
        metadata_path = path.parent / "run_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.is_file() else {}
        name = str(metadata.get("candidate_name", path.parent.name))
        fps = payload.get("complete_pipeline_fps")
        if fps is not None:
            result.setdefault(name, []).append(float(fps))
    return result


def aggregate_methods(
    evaluations: Sequence[dict[str, Any]],
    timings: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Aggregate dataset-level results by candidate name."""
    names = sorted({str(row["candidate"]) for row in evaluations})
    output: list[dict[str, Any]] = []
    for name in names:
        rows = [row for row in evaluations if row["candidate"] == name]
        output.append(
            {
                "candidate": name,
                "datasets": len(rows),
                "median": mean_finite(row["median"] for row in rows),
                "p95": mean_finite(row["p95"] for row in rows),
                "valid_ratio": mean_finite(
                    row["valid_ratio"] for row in rows
                ),
                "rula": mean_finite(row["rula"] for row in rows),
                "improvement": mean_finite(
                    row["improvement"] for row in rows
                ),
                "jumps": sum(int(row["jumps"]) for row in rows),
                "fps": mean_finite(timings.get(name, [])),
            }
        )
    return output


def save_charts(
    run_dir: Path,
    methods: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    """Save compact Pareto and validity charts."""
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pareto_path = figure_dir / "accuracy_speed_pareto.png"
    validity_path = figure_dir / "validity_comparison.png"

    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    plotted = 0
    for row in methods:
        if row["fps"] is None or row["median"] is None:
            continue
        axis.scatter(row["fps"], row["median"], s=70)
        axis.annotate(
            row["candidate"],
            (row["fps"], row["median"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
        plotted += 1
    axis.axvline(12.5, color="#b42318", linestyle="--", label="12.5 fps")
    axis.set_xlabel("Complete pipeline FPS (higher is better)")
    axis.set_ylabel("Mean dataset median absolute difference (deg)")
    axis.grid(alpha=0.2)
    if plotted:
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(pareto_path, dpi=170)
    plt.close(figure)

    valid_methods = [
        row for row in methods if row["valid_ratio"] is not None
    ]
    figure, axis = plt.subplots(figsize=(8.2, max(3.8, len(valid_methods) * 0.45)))
    axis.barh(
        [row["candidate"] for row in valid_methods],
        [row["valid_ratio"] for row in valid_methods],
        color="#2878b5",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Common valid-frame ratio")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(validity_path, dpi=170)
    plt.close(figure)
    return {"pareto": pareto_path, "validity": validity_path}


def find_candidate_npz(run_dir: Path, candidate_name: str) -> Path | None:
    """Find the canonical NPZ whose embedded name matches one evaluation."""
    for path in sorted(run_dir.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "candidate_name" not in payload.files:
                    continue
                name = str(np.asarray(payload["candidate_name"]).item())
        except (OSError, ValueError, KeyError):
            continue
        if name == candidate_name:
            return path
    return None


def ensure_best_preview(
    run_dir: Path,
    evaluations: Sequence[dict[str, Any]],
    methods: Sequence[dict[str, Any]],
) -> Path | None:
    """Render the best aggregate method's strongest dataset locally."""
    eligible = [row for row in methods if row["median"] is not None]
    if not eligible:
        return None
    best_method = min(eligible, key=lambda row: row["median"])
    cells = [
        row
        for row in evaluations
        if row["candidate"] == best_method["candidate"]
        and row["median"] is not None
    ]
    if not cells:
        return None
    cell = min(cells, key=lambda row: row["median"])
    candidate = find_candidate_npz(run_dir, str(cell["candidate"]))
    selection = run_dir / "input_selection.json"
    if candidate is None or not selection.is_file():
        return None
    dataset = str(cell["dataset"]).lower()
    configs = {
        "fanbo3": (
            "00_pose_pipeline_v2/configs/ablation_pipeline_model/"
            "fanbo3_v2_yolov8m.yaml"
        ),
        "fanbo4": (
            "00_pose_pipeline_v2/configs/ablation_pipeline_model/"
            "fanbo4_v2_yolov8m.yaml"
        ),
        "fanbo7": (
            "00_pose_pipeline_v2/configs/ablation_pipeline_model/"
            "fanbo7_v2_yolov8m.yaml"
        ),
    }
    if dataset not in configs:
        return None
    output_dir = run_dir / "best_candidate_evidence"
    output = output_dir / "best_candidate_preview_h264.mp4"
    if output.is_file():
        return output
    subprocess.run(
        [
            "/opt/anaconda3/envs/pose/bin/python",
            "tools/render_nvidia_candidate_preview.py",
            "--candidate",
            str(candidate),
            "--config",
            configs[dataset],
            "--selection",
            str(selection),
            "--dataset",
            dataset,
            "--output-dir",
            str(output_dir),
            "--evaluation-csv",
            str(Path(cell["metrics_path"]).parent / "angle_timeseries.csv"),
            "--duration-seconds",
            "20",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return output


def fmt(value: float | None, digits: int = 2) -> str:
    """Format a nullable metric."""
    return "—" if value is None else f"{value:.{digits}f}"


def status_rows(run_dir: Path, chinese: bool) -> str:
    """Render SDK availability classifications."""
    path = run_dir / "sdk_probes" / "sdk_route_status.json"
    if not path.is_file():
        return ""
    probe = load_json(path)
    rows = []
    for key in ("bodyposenet", "maxine"):
        item = probe[key]
        status = str(item["status"])
        class_name = "blocked" if "blocked" in status else "ok"
        meaning = (
            "运行环境/授权受阻，不是精度失败"
            if chinese and "blocked" in status
            else "Runtime/access blocker, not an accuracy failure"
            if "blocked" in status
            else "Available"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['candidate']))}</td>"
            f"<td class='{class_name}'>{html.escape(status)}</td>"
            f"<td>{meaning}</td></tr>"
        )
    return "".join(rows)


def metric_table(evaluations: Sequence[dict[str, Any]]) -> str:
    """Render dataset-level robust results."""
    rows = []
    for row in evaluations:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['candidate']))}</td>"
            f"<td>{html.escape(str(row['dataset']))}</td>"
            f"<td>{fmt(row['median'])}</td>"
            f"<td>{fmt(row['p95'])}</td>"
            f"<td>{fmt(row['baseline_median'])}</td>"
            f"<td>{fmt(row['valid_ratio'], 3)}</td>"
            f"<td>{fmt(row['rula'], 3)}</td>"
            f"<td>{row['jumps']}</td>"
            f"<td>{fmt(row['improvement'] * 100 if row['improvement'] is not None else None, 1)}%</td>"
            "</tr>"
        )
    return "".join(rows)


def method_table(methods: Sequence[dict[str, Any]]) -> str:
    """Render cross-dataset method summaries."""
    rows = []
    for row in methods:
        meets_speed = row["fps"] is not None and row["fps"] >= 12.5
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['candidate']))}</td>"
            f"<td>{row['datasets']}</td>"
            f"<td>{fmt(row['median'])}</td>"
            f"<td>{fmt(row['p95'])}</td>"
            f"<td>{fmt(row['valid_ratio'], 3)}</td>"
            f"<td>{fmt(row['fps'])}</td>"
            f"<td class='{'ok' if meets_speed else 'bad'}'>"
            f"{'yes' if meets_speed else 'no / unavailable'}</td>"
            "</tr>"
        )
    return "".join(rows)


def evidence_gallery(run_dir: Path, evaluations: Sequence[dict[str, Any]], chinese: bool) -> str:
    """Render expandable relative links to all evaluation plots."""
    blocks = []
    for row in evaluations:
        plot = Path(row["plot_path"])
        if not plot.is_file():
            continue
        relative = plot.relative_to(run_dir).as_posix()
        title = f"{row['candidate']} · {row['dataset']}"
        caption = (
            "时间曲线、参考散点和绝对差箱线图"
            if chinese
            else "Angle trace, reference scatter, and absolute-difference box plot"
        )
        blocks.append(
            f"<details><summary>{html.escape(title)}</summary>"
            f"<figure><img src='{html.escape(relative)}'>"
            f"<figcaption>{caption}</figcaption></figure></details>"
        )
    return "".join(blocks)


def preview_gallery(run_dir: Path, chinese: bool) -> str:
    """Render the best-candidate video and representative-frame gallery."""
    root = run_dir / "best_candidate_evidence"
    video = root / "best_candidate_preview_h264.mp4"
    if not video.is_file():
        return ""
    video_relative = video.relative_to(run_dir).as_posix()
    images = "".join(
        (
            f"<a href='{path.relative_to(run_dir).as_posix()}'>"
            f"<img src='{path.relative_to(run_dir).as_posix()}' "
            "style='width:31%;margin:1%;border-radius:6px'></a>"
        )
        for path in sorted((root / "representative_frames").glob("*.jpg"))
    )
    caption = (
        "最佳候选的 20 秒 H.264 左图—3D 重建预览"
        if chinese
        else "20-second H.264 left-view–3D preview for the best candidate"
    )
    return (
        f"<h3>{caption}</h3><video controls style='width:100%' "
        f"src='{video_relative}'></video><div>{images}</div>"
    )


def build_report(
    run_dir: Path,
    evaluations: Sequence[dict[str, Any]],
    methods: Sequence[dict[str, Any]],
    charts: dict[str, Path],
    chinese: bool,
) -> str:
    """Build one language version."""
    blocked = status_rows(run_dir, chinese)
    pareto = charts["pareto"].relative_to(run_dir).as_posix()
    validity = charts["validity"].relative_to(run_dir).as_posix()
    completed = len(evaluations)
    best = min(
        (row for row in methods if row["median"] is not None),
        key=lambda row: row["median"],
        default=None,
    )
    best_name = str(best["candidate"]) if best else "—"
    if chinese:
        title = "Fanbo3/4/7 NVIDIA 单目—双目 GPU 对比"
        subtitle = "固定标定 · 固定参考偏移 · 中位数/IQR/p95 优先 · RTX A6000"
        body = f"""
<section id="summary"><h2>1. 结论摘要</h2>
<div class="cards"><div class="card"><div class="metric">{completed}</div>成功的数据集—候选评估单元</div><div class="card"><div class="metric">{html.escape(best_name)}</div>当前最低聚合中位绝对差</div><div class="card"><div class="metric">12.5 fps</div>完整 Pipeline 实时门槛</div></div>
<div class="callout">Xsens 仅作为 Xsens-derived reference / external comparison system；Fanbo4/7 使用 FastSAM3D comparison trajectory。没有为候选重新搜索时间偏移或手动挑选较好视角。</div></section>
<section id="availability"><h2>2. NVIDIA 路线可用性</h2><table><tr><th>路线</th><th>状态</th><th>含义</th></tr>{blocked}</table>
<p>BodyPoseNet 2D 和 Maxine 的阻塞会被单独记录，不会被写成“模型精度失败”。BodyPose3DNet、FoundationStereo 与 Fast-FoundationStereo 使用官方仓库/权重实测。</p></section>
<section id="results"><h2>3. 逐数据集稳健指标</h2><table><tr><th>候选</th><th>数据</th><th>Median °</th><th>p95 °</th><th>SKT median °</th><th>有效率</th><th>RULA一致率</th><th>&gt;10°跳变</th><th>改善</th></tr>{metric_table(evaluations)}</table></section>
<section id="aggregate"><h2>4. 精度—速度汇总</h2><table><tr><th>候选</th><th>数据集数</th><th>Median °</th><th>p95 °</th><th>有效率</th><th>完整 FPS</th><th>实时</th></tr>{method_table(methods)}</table>
<figure><img src="{pareto}"><figcaption>精度—速度 Pareto。DeepStream/Maxine 受许可约束的精确 timing 不在对外 HTML 中展开。</figcaption></figure>
<figure><img src="{validity}"><figcaption>共同有效帧比例。</figcaption></figure></section>
<section id="evidence"><h2>5. 直观图表</h2>{preview_gallery(run_dir, True)}{evidence_gallery(run_dir, evaluations, True)}</section>
<section id="rules"><h2>6. 判定规则与可复现性</h2><ul><li>固定 Fanbo3 2.00 s、Fanbo4 5.4 s、Fanbo7 7.3 s 参考偏移。</li><li>主要展示 median、IQR、p95、RULA 和有效率；MAE/RMSE仅作辅助。</li><li>实时要求完整 Pipeline ≥12.5 fps 且 p95 ≤80 ms。</li><li>许可权重、TensorRT engine、视频和凭据均不进入本地结果包。</li></ul></section>"""
        nav = ["摘要", "可用性", "结果", "精度—速度", "图表", "规则"]
    else:
        title = "Fanbo3/4/7 NVIDIA Monocular–Stereo GPU Comparison"
        subtitle = "Frozen calibration · fixed reference offsets · median/IQR/p95 priority · RTX A6000"
        body = f"""
<section id="summary"><h2>1. Executive summary</h2>
<div class="cards"><div class="card"><div class="metric">{completed}</div>successful dataset–candidate evaluation cells</div><div class="card"><div class="metric">{html.escape(best_name)}</div>lowest current aggregate median absolute difference</div><div class="card"><div class="metric">12.5 fps</div>complete-pipeline real-time gate</div></div>
<div class="callout">Xsens is treated only as an Xsens-derived reference / external comparison system. Fanbo4/7 use FastSAM3D comparison trajectories. Candidate-specific offset search and post-hoc view selection were prohibited.</div></section>
<section id="availability"><h2>2. NVIDIA route availability</h2><table><tr><th>Route</th><th>Status</th><th>Meaning</th></tr>{blocked}</table>
<p>BodyPoseNet 2D and Maxine blockers are reported separately from model accuracy. BodyPose3DNet, FoundationStereo, and Fast-FoundationStereo were run from official repositories and checkpoints.</p></section>
<section id="results"><h2>3. Robust dataset-level metrics</h2><table><tr><th>Candidate</th><th>Dataset</th><th>Median °</th><th>p95 °</th><th>SKT median °</th><th>Valid ratio</th><th>RULA agreement</th><th>&gt;10° jumps</th><th>Improvement</th></tr>{metric_table(evaluations)}</table></section>
<section id="aggregate"><h2>4. Accuracy–throughput summary</h2><table><tr><th>Candidate</th><th>Datasets</th><th>Median °</th><th>p95 °</th><th>Valid ratio</th><th>Complete FPS</th><th>Real-time</th></tr>{method_table(methods)}</table>
<figure><img src="{pareto}"><figcaption>Accuracy–speed Pareto view. Exact proprietary DeepStream/Maxine timing is not expanded in the outward-facing HTML.</figcaption></figure>
<figure><img src="{validity}"><figcaption>Common valid-frame ratio.</figcaption></figure></section>
<section id="evidence"><h2>5. Visual evidence</h2>{preview_gallery(run_dir, False)}{evidence_gallery(run_dir, evaluations, False)}</section>
<section id="rules"><h2>6. Decision rules and reproducibility</h2><ul><li>Frozen reference offsets: Fanbo3 2.00 s, Fanbo4 5.4 s, Fanbo7 7.3 s.</li><li>Primary metrics are median, IQR, p95, RULA agreement, and validity; MAE/RMSE are secondary.</li><li>Real-time requires a complete pipeline ≥12.5 fps and p95 latency ≤80 ms.</li><li>Licensed weights, TensorRT engines, videos, and credentials are excluded from the local result bundle.</li></ul></section>"""
        nav = ["Summary", "Availability", "Results", "Accuracy–speed", "Evidence", "Rules"]
    ids = ["summary", "availability", "results", "aggregate", "evidence", "rules"]
    links = "".join(
        f'<a href="#{target}">{label}</a>'
        for target, label in zip(ids, nav)
    )
    return (
        f'<!doctype html><html lang="{"zh-CN" if chinese else "en"}">'
        f"<head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>{title}</title>"
        f"<style>{STYLE}</style></head><body><header><h1>{title}</h1>"
        f"<p>{subtitle}</p></header><div class='layout'><nav><strong>"
        f"{'目录' if chinese else 'Contents'}</strong>{links}</nav><main>"
        f"{body}<footer>{'生成时间' if chinese else 'Generated'}: "
        f"{datetime.now(timezone.utc).isoformat()}</footer></main></div>"
        "</body></html>"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(run_dir: Path) -> Path:
    """Write a checksum manifest while excluding forbidden bulky assets."""
    forbidden = {
        ".avi",
        ".mp4",
        ".mkv",
        ".pt",
        ".pth",
        ".onnx",
        ".engine",
        ".pkl",
    }
    output = run_dir / "artifact_manifest.json"
    records = []
    for path in sorted(run_dir.rglob("*")):
        if (
            not path.is_file()
            or path == output
            or path.suffix.lower() in forbidden
            or "credential" in path.name.lower()
        ):
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output.write_text(
        json.dumps(
            {
                "schema_version": "nvidia_pose_artifact_manifest_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "records": records,
                "excluded_suffixes": sorted(forbidden),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def write_experiment_log(
    run_dir: Path,
    evaluations: Sequence[dict[str, Any]],
    methods: Sequence[dict[str, Any]],
) -> Path:
    """Write a concise, durable experiment log for the formal run."""
    status_path = run_dir / "candidate_matrix_status.json"
    status = load_json(status_path) if status_path.is_file() else {}
    lines = [
        "# NVIDIA pose matrix experiment log",
        "",
        f"- Finalized UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Run directory: `{run_dir}`",
        f"- Successful evaluation cells: {len(evaluations)}",
        "- Reference policy: Xsens-derived reference is an external "
        "comparison system, not absolute Ground Truth.",
        "- Candidate-specific time-offset search: disabled.",
        "",
        "## Method summary",
        "",
        "| Candidate | Datasets | Median deg | p95 deg | Valid ratio | FPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in methods:
        lines.append(
            f"| {row['candidate']} | {row['datasets']} | "
            f"{fmt(row['median'])} | {fmt(row['p95'])} | "
            f"{fmt(row['valid_ratio'], 3)} | {fmt(row['fps'])} |"
        )
    lines.extend(["", "## Command records", ""])
    for record in status.get("records", []):
        command_value = record.get("command")
        command_text = (
            " ".join(str(value) for value in command_value)
            if isinstance(command_value, list)
            else str(command_value or "")
        )
        lines.append(
            f"- `{record.get('status', 'unknown')}` "
            f"{record.get('route', '')} "
            f"{record.get('dataset', '')}: `{command_text}`"
        )
    output = run_dir / "experiment_log.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate both reports and the artifact checksum manifest."""
    args = parse_args(argv)
    run_dir = project_path(args.run_dir)
    evaluations = collect_evaluations(run_dir)
    timings = collect_timings(run_dir)
    methods = aggregate_methods(evaluations, timings)
    charts = save_charts(run_dir, methods)
    ensure_best_preview(run_dir, evaluations, methods)
    (run_dir / "report.html").write_text(
        build_report(run_dir, evaluations, methods, charts, False),
        encoding="utf-8",
    )
    (run_dir / "report_CN.html").write_text(
        build_report(run_dir, evaluations, methods, charts, True),
        encoding="utf-8",
    )
    write_experiment_log(run_dir, evaluations, methods)
    manifest = write_manifest(run_dir)
    print(run_dir / "report.html")
    print(run_dir / "report_CN.html")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
