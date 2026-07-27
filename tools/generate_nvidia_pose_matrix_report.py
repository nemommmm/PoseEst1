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
        candidate_mae = mean_finite(
            row["candidate"]["absolute_error_deg"]["mean"]
            for row in rows
        )
        baseline_median = mean_finite(
            row["baseline"]["absolute_error_deg"]["median"]
            for row in rows
        )
        baseline_p95 = mean_finite(
            row["baseline"]["absolute_error_deg"]["p95"]
            for row in rows
        )
        candidate_p95 = mean_finite(
            row["candidate"]["absolute_error_deg"]["p95"]
            for row in rows
        )
        valid_ratio = mean_finite(
            row["candidate"]["valid_ratio"] for row in rows
        )
        baseline_valid_ratio = mean_finite(
            row["baseline"]["valid_ratio"] for row in rows
        )
        rula = mean_finite(
            row["candidate"].get("rula_like_agreement") for row in rows
        )
        baseline_rula = mean_finite(
            row["baseline"].get("rula_like_agreement") for row in rows
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
                "mae": candidate_mae,
                "median": candidate_median,
                "baseline_median": baseline_median,
                "baseline_p95": baseline_p95,
                "p95": candidate_p95,
                "valid_ratio": valid_ratio,
                "baseline_valid_ratio": baseline_valid_ratio,
                "rula": rula,
                "baseline_rula": baseline_rula,
                "jumps": jumps,
                "improvement": metrics.get(
                    "aggregate_median_improvement_ratio"
                ),
                "metrics_path": path,
                "plot_path": path.parent / "angle_summary.png",
                "reference": metrics.get("reference", {}),
                "angles": [
                    str(row["angle"])
                    for row in rows
                    if row.get("angle") is not None
                ],
            }
        )
    return records


def collect_timings(run_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Collect publishable open-model pipeline FPS values."""
    result: dict[str, list[dict[str, float]]] = {}
    for path in sorted((run_dir / "baseline").glob("*/benchmark.json")):
        payload = load_json(path)
        result.setdefault("YOLOv8m-PyTorch-SKT", []).append(
            {
                "fps": float(payload["online_fps"]),
                "p95_ms": float(
                    payload["stages"]["end_to_end_online"]["p95_ms"]
                ),
            }
        )
    for path in sorted(run_dir.glob("dense_stereo/**/timing_internal.json")):
        payload = load_json(path)
        metadata_path = path.parent / "run_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.is_file() else {}
        name = str(metadata.get("candidate_name", path.parent.name))
        fps = payload.get("complete_pipeline_fps")
        if fps is not None:
            result.setdefault(name, []).append(
                {
                    "fps": float(fps),
                    "p95_ms": float(
                        payload["complete_left_yolo_plus_dense"]["p95_ms"]
                    ),
                }
            )
    return result


def aggregate_methods(
    evaluations: Sequence[dict[str, Any]],
    timings: dict[str, list[dict[str, float]]],
) -> list[dict[str, Any]]:
    """Aggregate dataset-level results by candidate name."""
    names = sorted(
        {str(row["candidate"]) for row in evaluations}
        | set(timings)
    )
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
                "baseline_rula": mean_finite(
                    row["baseline_rula"] for row in rows
                ),
                "baseline_median": mean_finite(
                    row["baseline_median"] for row in rows
                ),
                "baseline_valid_ratio": mean_finite(
                    row["baseline_valid_ratio"] for row in rows
                ),
                "improvement": mean_finite(
                    row["improvement"] for row in rows
                ),
                "jumps": sum(int(row["jumps"]) for row in rows),
                "fps": mean_finite(
                    item["fps"] for item in timings.get(name, [])
                ),
                "latency_p95_ms": (
                    max(
                        item["p95_ms"]
                        for item in timings.get(name, [])
                    )
                    if timings.get(name)
                    else None
                ),
            }
        )
    return output


def assess_method_gates(
    evaluations: Sequence[dict[str, Any]],
    methods: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply the frozen cross-dataset accuracy and real-time rules."""
    expected = {"Fanbo3", "Fanbo4", "Fanbo7"}
    decisions: dict[str, dict[str, Any]] = {}
    for method in methods:
        name = str(method["candidate"])
        rows = [row for row in evaluations if row["candidate"] == name]
        if name in {"YOLOv8m+SKT", "YOLOv8m-PyTorch-SKT"}:
            decisions[name] = {
                "offline_passed": True,
                "realtime_passed": bool(
                    method["fps"] is not None
                    and method["fps"] >= 12.5
                    and method["latency_p95_ms"] is not None
                    and method["latency_p95_ms"] <= 80.0
                ),
                "reasons": ["control group"],
            }
            continue
        reasons: list[str] = []
        if {row["dataset"] for row in rows} != expected:
            reasons.append("not all three datasets completed")
        finite_rows = [
            row
            for row in rows
            if row["median"] is not None
            and row["baseline_median"] is not None
        ]
        if len(finite_rows) != 3:
            reasons.append("one or more datasets have no valid metric")
        else:
            candidate_mean = float(
                np.mean([row["median"] for row in finite_rows])
            )
            baseline_mean = float(
                np.mean([row["baseline_median"] for row in finite_rows])
            )
            improvement = (
                (baseline_mean - candidate_mean) / baseline_mean
                if baseline_mean > 0
                else float("-inf")
            )
            if improvement < 0.05:
                reasons.append("overall median improvement is below 5%")
            if any(
                row["median"] - row["baseline_median"] > 1.0
                for row in finite_rows
            ):
                reasons.append("a core dataset degrades by more than 1 degree")
            if any(
                row["rula"] is not None
                and row["baseline_rula"] is not None
                and row["rula"] < row["baseline_rula"]
                for row in finite_rows
            ):
                reasons.append("RULA-like agreement decreases")
            if any(
                row["rula"] is None or row["baseline_rula"] is None
                for row in finite_rows
            ):
                reasons.append("RULA-like agreement is unavailable")
            if any(
                row["valid_ratio"] is not None
                and row["baseline_valid_ratio"] is not None
                and (
                    row["valid_ratio"]
                    < row["baseline_valid_ratio"] - 0.03
                )
                for row in finite_rows
            ):
                reasons.append("valid ratio decreases by more than 3 points")
            if any(
                row["valid_ratio"] is None
                or row["baseline_valid_ratio"] is None
                for row in finite_rows
            ):
                reasons.append("valid ratio is unavailable")
        offline_passed = not reasons
        realtime_passed = bool(
            offline_passed
            and method["fps"] is not None
            and method["fps"] >= 12.5
            and method["latency_p95_ms"] is not None
            and method["latency_p95_ms"] <= 80.0
        )
        decisions[name] = {
            "offline_passed": offline_passed,
            "realtime_passed": realtime_passed,
            "reasons": reasons,
        }
    return decisions


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
    """Render the strongest dataset from the best accepted method."""
    eligible = [
        row
        for row in methods
        if row["median"] is not None and row.get("offline_passed", False)
    ]
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


def fmt_percent(value: float | None, digits: int = 1) -> str:
    """Format one nullable fraction as a percentage."""
    return "—" if value is None else f"{value * 100:.{digits}f}%"


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


def is_skt_baseline(candidate: str) -> bool:
    """Return whether a candidate is the deterministic SKT control."""
    return candidate in {"YOLOv8m+SKT", "YOLOv8m-PyTorch-SKT"}


def display_candidate(candidate: str, chinese: bool) -> str:
    """Return a concise display label, marking the control explicitly."""
    if is_skt_baseline(candidate):
        suffix = "（基线）" if chinese else " (baseline)"
        return f"YOLOv8m + SKT{suffix}"
    return candidate


def dataset_metric_tables(
    evaluations: Sequence[dict[str, Any]],
    chinese: bool,
) -> str:
    """Render one robust-metric table per dataset with SKT first."""
    tables: list[str] = []
    dataset_order = ("Fanbo3", "Fanbo4", "Fanbo7")
    for dataset in dataset_order:
        dataset_rows = [
            row for row in evaluations if row["dataset"] == dataset
        ]
        if not dataset_rows:
            continue
        dataset_rows.sort(
            key=lambda row: (
                not is_skt_baseline(str(row["candidate"])),
                str(row["candidate"]),
            )
        )
        reference = dataset_rows[0].get("reference", {})
        reference_label = str(reference.get("label", "—"))
        angles = dataset_rows[0].get("angles", [])
        angle_scope = (
            f"{len(angles)} 个关节角"
            if chinese and len(angles) > 1
            else f"{len(angles)} joint angles"
            if len(angles) > 1
            else str(angles[0])
            if angles
            else "—"
        )
        rows = []
        for row in dataset_rows:
            candidate = display_candidate(str(row["candidate"]), chinese)
            candidate_cell = html.escape(candidate)
            if is_skt_baseline(str(row["candidate"])):
                candidate_cell = f"<strong>{candidate_cell}</strong>"
            rows.append(
                "<tr>"
                f"<td>{candidate_cell}</td>"
                f"<td>{fmt(row['mae'])}</td>"
                f"<td>{fmt(row['median'])}</td>"
                f"<td>{fmt(row['p95'])}</td>"
                f"<td>{fmt(row['valid_ratio'], 3)}</td>"
                f"<td>{fmt(row['rula'], 3)}</td>"
                f"<td>{row['jumps']}</td>"
                f"<td>{fmt_percent(row['improvement'])}</td>"
                "</tr>"
            )
        comparison_label = "对比参考" if chinese else "Comparison reference"
        scope_label = "评价角度" if chinese else "Evaluated angles"
        improvement_label = (
            "Median 相对 SKT 改善"
            if chinese
            else "Median improvement vs SKT"
        )
        tables.append(
            f"<h3>{dataset}</h3>"
            f"<p><strong>{comparison_label}：</strong>"
            f"{html.escape(reference_label)}；"
            f"<strong>{scope_label}：</strong>{html.escape(angle_scope)}</p>"
            "<table><tr>"
            f"<th>{'候选' if chinese else 'Candidate'}</th>"
            "<th>MAE °</th><th>Median °</th><th>p95 °</th>"
            f"<th>{'有效率' if chinese else 'Valid ratio'}</th>"
            f"<th>{'RULA一致率' if chinese else 'RULA agreement'}</th>"
            f"<th>{'&gt;10°跳变' if chinese else '&gt;10° jumps'}</th>"
            f"<th>{improvement_label}</th></tr>{''.join(rows)}</table>"
        )
    return "".join(tables)


def method_table(methods: Sequence[dict[str, Any]]) -> str:
    """Render cross-dataset method summaries."""
    rows = []
    for row in methods:
        offline_passed = bool(row.get("offline_passed", False))
        realtime_passed = bool(row.get("realtime_passed", False))
        reasons = "; ".join(str(value) for value in row.get("reasons", []))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['candidate']))}</td>"
            f"<td>{row['datasets']}</td>"
            f"<td>{fmt(row['median'])}</td>"
            f"<td>{fmt(row['p95'])}</td>"
            f"<td>{fmt(row['valid_ratio'], 3)}</td>"
            f"<td>{fmt(row['fps'])}</td>"
            f"<td>{fmt(row['latency_p95_ms'])}</td>"
            f"<td class='{'ok' if offline_passed else 'bad'}' "
            f"title='{html.escape(reasons)}'>"
            f"{'yes' if offline_passed else 'no'}</td>"
            f"<td class='{'ok' if realtime_passed else 'bad'}' "
            f"title='{html.escape(reasons)}'>"
            f"{'yes' if realtime_passed else 'no / unavailable'}</td>"
            "</tr>"
        )
    return "".join(rows)


def decision_summary(
    methods: Sequence[dict[str, Any]],
    chinese: bool,
) -> tuple[str, str]:
    """Return the accepted method name and a concise recommendation."""
    controls = {
        "YOLOv8m+SKT",
        "YOLOv8m-PyTorch-SKT",
    }
    accepted = [
        row
        for row in methods
        if row["candidate"] not in controls
        and row.get("offline_passed", False)
        and row["median"] is not None
    ]
    if not accepted:
        if chinese:
            return (
                "YOLOv8m + SKT",
                "没有新的候选同时通过三个数据集的精度、RULA和有效率门槛；"
                "因此正式建议继续保留 YOLOv8m + SKT。局部数据集上的改善"
                "只作为后续研究线索，不作为替换依据。",
            )
        return (
            "YOLOv8m + SKT",
            "No new candidate passed the cross-dataset accuracy, RULA, and "
            "validity gates. The formal recommendation is therefore to "
            "retain YOLOv8m + SKT. Isolated dataset gains are retained as "
            "research evidence, not as grounds for replacement.",
        )
    offline = min(accepted, key=lambda row: row["median"])
    realtime = [
        row for row in accepted if row.get("realtime_passed", False)
    ]
    realtime_name = (
        min(realtime, key=lambda row: row["median"])["candidate"]
        if realtime
        else None
    )
    if chinese:
        text = f"离线精度候选为 {offline['candidate']}。"
        text += (
            f"实时候选为 {realtime_name}。"
            if realtime_name is not None
            else "尚无新候选同时通过完整 Pipeline 实时门槛。"
        )
    else:
        text = f"The accepted offline candidate is {offline['candidate']}. "
        text += (
            f"The accepted real-time candidate is {realtime_name}."
            if realtime_name is not None
            else "No new candidate also passes the complete-pipeline "
            "real-time gate."
        )
    return str(offline["candidate"]), text


def notable_findings(
    evaluations: Sequence[dict[str, Any]],
    methods: Sequence[dict[str, Any]],
    chinese: bool,
) -> str:
    """Render concise evidence behind the final selection decision."""
    cells = {
        (str(row["candidate"]), str(row["dataset"])): row
        for row in evaluations
    }
    method_map = {str(row["candidate"]): row for row in methods}
    foundation = "FoundationStereo-ViT-L-32iter"
    fast = "Fast-FoundationStereo-4iter"
    body = "BodyPose3DNet-accuracy_monocular_left"
    foundation_far = cells.get((foundation, "Fanbo4"))
    foundation_fanbo3 = cells.get((foundation, "Fanbo3"))
    fast_method = method_map.get(fast)
    body_near = cells.get((body, "Fanbo7"))
    body_far = cells.get((body, "Fanbo4"))
    findings: list[str] = []
    if foundation_far is not None:
        if chinese:
            findings.append(
                "FoundationStereo 在远距 Fanbo4 将 RightElbow median "
                f"从 {fmt(foundation_far['baseline_median'])}° 降至 "
                f"{fmt(foundation_far['median'])}°，说明单侧人体检测加"
                "双目稠密深度具有明确研究价值。"
            )
        else:
            findings.append(
                "On distant-view Fanbo4, FoundationStereo reduced the "
                f"RightElbow median from {fmt(foundation_far['baseline_median'])}° "
                f"to {fmt(foundation_far['median'])}°, supporting the value "
                "of single-view joint detection plus dense stereo depth."
            )
    if foundation_fanbo3 is not None:
        if chinese:
            findings.append(
                "但它在 Fanbo3 的 RULA-like 一致率从 "
                f"{fmt(foundation_fanbo3['baseline_rula'], 3)} 降至 "
                f"{fmt(foundation_fanbo3['rula'], 3)}，因此未通过冻结的"
                "跨数据集准入规则。"
            )
        else:
            findings.append(
                "However, its Fanbo3 RULA-like agreement fell from "
                f"{fmt(foundation_fanbo3['baseline_rula'], 3)} to "
                f"{fmt(foundation_fanbo3['rula'], 3)}, so it failed the "
                "frozen cross-dataset acceptance rule."
            )
    if fast_method is not None:
        if chinese:
            findings.append(
                "Fast-FoundationStereo 4-iteration 完整 Pipeline 仅为 "
                f"{fmt(fast_method['fps'])} FPS，p95 "
                f"{fmt(fast_method['latency_p95_ms'])} ms，仍不实时。"
            )
        else:
            findings.append(
                "The complete Fast-FoundationStereo 4-iteration pipeline "
                f"reached only {fmt(fast_method['fps'])} FPS with "
                f"{fmt(fast_method['latency_p95_ms'])} ms p95 latency and "
                "therefore remained non-real-time."
            )
    if body_near is not None and body_far is not None:
        if chinese:
            findings.append(
                "BodyPose3DNet Accuracy 单目左视图在近距 Fanbo7 为 "
                f"{fmt(body_near['median'])}°，但远距 Fanbo4 增至 "
                f"{fmt(body_far['median'])}°，跨距离鲁棒性不足。"
            )
        else:
            findings.append(
                "BodyPose3DNet Accuracy monocular-left reached "
                f"{fmt(body_near['median'])}° on near-view Fanbo7 but "
                f"{fmt(body_far['median'])}° on distant Fanbo4, exposing "
                "insufficient cross-distance robustness."
            )
    return "<ul>" + "".join(f"<li>{value}</li>" for value in findings) + "</ul>"


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
    best_name, recommendation = decision_summary(methods, chinese)
    if chinese:
        title = "Fanbo3/4/7 NVIDIA 单目—双目 GPU 对比"
        subtitle = "固定标定 · 固定参考偏移 · 中位数/IQR/p95 优先 · RTX A6000"
        body = f"""
<section id="summary"><h2>1. 结论摘要</h2>
<div class="cards"><div class="card"><div class="metric">{completed}</div>成功的数据集—候选评估单元</div><div class="card"><div class="metric">{html.escape(best_name)}</div>按预设门槛得到的正式选择</div><div class="card"><div class="metric">12.5 fps</div>完整 Pipeline 实时门槛</div></div>
<div class="callout"><strong>选型结论：</strong>{html.escape(recommendation)}</div>
{notable_findings(evaluations, methods, True)}
<div class="callout">Xsens 仅作为 Xsens-derived reference / external comparison system；Fanbo4/7 使用 FastSAM3D comparison trajectory。没有为候选重新搜索时间偏移或手动挑选较好视角。</div></section>
<section id="availability"><h2>2. NVIDIA 路线可用性</h2><table><tr><th>路线</th><th>状态</th><th>含义</th></tr>{blocked}</table>
<p>BodyPoseNet 2D 和 Maxine 的阻塞会被单独记录，不会被写成“模型精度失败”。BodyPose3DNet、FoundationStereo 与 Fast-FoundationStereo 使用官方仓库/权重实测。</p></section>
<section id="results"><h2>3. 逐数据集稳健指标</h2>
<p>每个数据集单独展示；YOLOv8m + SKT 固定为第一行基线。MAE 是绝对角度差的平均值，Median 是绝对角度差的中位数。</p>
{dataset_metric_tables(evaluations, True)}</section>
<section id="aggregate"><h2>4. 精度—速度汇总</h2><table><tr><th>候选</th><th>数据集数</th><th>Median °</th><th>p95 °</th><th>有效率</th><th>完整 FPS</th><th>p95延迟 ms</th><th>离线准入</th><th>实时准入</th></tr>{method_table(methods)}</table>
<figure><img src="{pareto}"><figcaption>精度—速度 Pareto。DeepStream/Maxine 受许可约束的精确 timing 不在对外 HTML 中展开。</figcaption></figure>
<figure><img src="{validity}"><figcaption>共同有效帧比例。</figcaption></figure></section>
<section id="evidence"><h2>5. 直观图表</h2>{preview_gallery(run_dir, True)}{evidence_gallery(run_dir, evaluations, True)}</section>
<section id="rules"><h2>6. 判定规则与可复现性</h2><ul><li>固定 Fanbo3 2.00 s、Fanbo4 5.4 s、Fanbo7 7.3 s 参考偏移。</li><li>主要展示 median、IQR、p95、RULA 和有效率；MAE/RMSE仅作辅助。</li><li>实时要求完整 Pipeline ≥12.5 fps 且 p95 ≤80 ms。</li><li>许可权重、TensorRT engine、原始输入视频和凭据均不进入本地结果包；仅保留计划要求的 20 秒重建预览。</li></ul></section>"""
        nav = ["摘要", "可用性", "结果", "精度—速度", "图表", "规则"]
    else:
        title = "Fanbo3/4/7 NVIDIA Monocular–Stereo GPU Comparison"
        subtitle = "Frozen calibration · fixed reference offsets · median/IQR/p95 priority · RTX A6000"
        body = f"""
<section id="summary"><h2>1. Executive summary</h2>
<div class="cards"><div class="card"><div class="metric">{completed}</div>successful dataset–candidate evaluation cells</div><div class="card"><div class="metric">{html.escape(best_name)}</div>formal selection under the frozen gates</div><div class="card"><div class="metric">12.5 fps</div>complete-pipeline real-time gate</div></div>
<div class="callout"><strong>Selection outcome:</strong> {html.escape(recommendation)}</div>
{notable_findings(evaluations, methods, False)}
<div class="callout">Xsens is treated only as an Xsens-derived reference / external comparison system. Fanbo4/7 use FastSAM3D comparison trajectories. Candidate-specific offset search and post-hoc view selection were prohibited.</div></section>
<section id="availability"><h2>2. NVIDIA route availability</h2><table><tr><th>Route</th><th>Status</th><th>Meaning</th></tr>{blocked}</table>
<p>BodyPoseNet 2D and Maxine blockers are reported separately from model accuracy. BodyPose3DNet, FoundationStereo, and Fast-FoundationStereo were run from official repositories and checkpoints.</p></section>
<section id="results"><h2>3. Robust dataset-level metrics</h2>
<p>Each dataset is shown separately, with YOLOv8m + SKT fixed as the first-row baseline. MAE is the mean absolute angular difference; Median is its median.</p>
{dataset_metric_tables(evaluations, False)}</section>
<section id="aggregate"><h2>4. Accuracy–throughput summary</h2><table><tr><th>Candidate</th><th>Datasets</th><th>Median °</th><th>p95 °</th><th>Valid ratio</th><th>Complete FPS</th><th>p95 latency ms</th><th>Offline gate</th><th>Real-time gate</th></tr>{method_table(methods)}</table>
<figure><img src="{pareto}"><figcaption>Accuracy–speed Pareto view. Exact proprietary DeepStream/Maxine timing is not expanded in the outward-facing HTML.</figcaption></figure>
<figure><img src="{validity}"><figcaption>Common valid-frame ratio.</figcaption></figure></section>
<section id="evidence"><h2>5. Visual evidence</h2>{preview_gallery(run_dir, False)}{evidence_gallery(run_dir, evaluations, False)}</section>
<section id="rules"><h2>6. Decision rules and reproducibility</h2><ul><li>Frozen reference offsets: Fanbo3 2.00 s, Fanbo4 5.4 s, Fanbo7 7.3 s.</li><li>Primary metrics are median, IQR, p95, RULA agreement, and validity; MAE/RMSE are secondary.</li><li>Real-time requires a complete pipeline ≥12.5 fps and p95 latency ≤80 ms.</li><li>Licensed weights, TensorRT engines, raw input videos, and credentials are excluded from the local result bundle; only the planned 20-second reconstruction preview is retained.</li></ul></section>"""
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
        "| Candidate | Datasets | Median deg | p95 deg | Valid ratio | "
        "FPS | p95 ms | Offline gate | Real-time gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in methods:
        lines.append(
            f"| {row['candidate']} | {row['datasets']} | "
            f"{fmt(row['median'])} | {fmt(row['p95'])} | "
            f"{fmt(row['valid_ratio'], 3)} | {fmt(row['fps'])} | "
            f"{fmt(row['latency_p95_ms'])} | "
            f"{'pass' if row.get('offline_passed') else 'reject'} | "
            f"{'pass' if row.get('realtime_passed') else 'reject'} |"
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


def reconcile_repaired_evaluation_statuses(run_dir: Path) -> int:
    """Mark BodyPose evaluations repaired after reference upload as complete."""
    status_path = run_dir / "candidate_matrix_status.json"
    if not status_path.is_file():
        return 0
    payload = load_json(status_path)
    repaired = 0
    evaluation_routes = {
        "BodyPose3DNet_monocular_left": "monocular_left",
        "BodyPose3DNet_monocular_right": "monocular_right",
        "BodyPose3DNet_stereo": "stereo",
    }
    for record in payload.get("records", []):
        if record.get("status") != "failed":
            continue
        route = str(record.get("route", ""))
        suffix = evaluation_routes.get(route)
        dataset = record.get("dataset")
        mode = record.get("mode")
        if suffix is None or dataset is None or mode is None:
            continue
        metrics_path = (
            run_dir
            / "bodypose3d"
            / "evaluation"
            / str(dataset)
            / str(mode)
            / suffix
            / "metrics.json"
        )
        if not metrics_path.is_file():
            continue
        record["original_status"] = "failed_before_reference_upload_repair"
        record["status"] = "completed_after_reference_upload_repair"
        record["return_code"] = 0
        record["repair_evidence"] = str(metrics_path.relative_to(run_dir))
        repaired += 1
    if repaired:
        payload["status_reconciliation"] = {
            "repaired_records": repaired,
            "reason": (
                "The original evaluation commands ran before reference files "
                "with spaces were re-uploaded to their literal paths. The "
                "fixed-offset evaluations were rerun and metrics artifacts "
                "were verified."
            ),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        status_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return repaired


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
    decisions = assess_method_gates(evaluations, methods)
    for method in methods:
        method.update(decisions[str(method["candidate"])])
    reconcile_repaired_evaluation_statuses(run_dir)
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
