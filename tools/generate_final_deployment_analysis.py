#!/opt/anaconda3/envs/pose/bin/python
"""Generate the final CPU, GPU deployment, and compression summary."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "00_pose_pipeline_v2/runs/final_deployment_analysis_20260805"
)
GPU_ROOT = Path("00_pose_pipeline_v2/runs/gpu_rebuild_20260725")
PROXY_ROOT = Path("00_pose_pipeline_v2/runs/nvidia_pose_matrix")


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def median(values: Sequence[float]) -> float:
    """Return a floating-point median."""
    return float(statistics.median(float(value) for value in values))


def gpu_deployment_rows() -> list[dict[str, Any]]:
    """Collect formal A6000 rows from existing three-repeat suites."""
    rows: list[dict[str, Any]] = []
    suites = [
        (
            "PyTorch",
            "FP32",
            project_path(GPU_ROOT / "pytorch_formal/suite_summary.json"),
        ),
        (
            "TensorRT",
            "FP32",
            project_path(GPU_ROOT / "tensorrt_formal/suite_summary.json"),
        ),
        (
            "TensorRT",
            "FP16",
            project_path(GPU_ROOT / "tensorrt_formal/suite_summary.json"),
        ),
    ]
    for backend, precision, suite_path in suites:
        suite = load_json(suite_path)
        for dataset_label in ("fanbo7", "fanbo4"):
            key = dataset_label
            if backend == "TensorRT":
                key = f"{dataset_label}_trt_{precision.lower()}"
            dataset = suite["datasets"][key]
            timing = dataset["timing_aggregate"]
            pose = timing["stages"]["pose_inference_stereo"][
                "pooled_frame_ms"
            ]
            online = timing["stages"]["end_to_end_online"][
                "pooled_frame_ms"
            ]
            comparison = dataset["records"][0]["historical_comparison"]
            angle = comparison["angle"]["absolute_difference_deg"]
            distance = comparison["keypoints"]["distance_cm"]
            rows.append(
                {
                    "hardware": "NVIDIA RTX A6000",
                    "backend": backend,
                    "precision": precision,
                    "dataset": dataset_label.capitalize(),
                    "frames": 200,
                    "repeats": 3,
                    "steady_fps": timing["online_fps_across_repeats"][
                        "median"
                    ],
                    "pose_median_ms": pose["median"],
                    "pose_p95_ms": pose["p95"],
                    "end_to_end_median_ms": online["median"],
                    "end_to_end_p95_ms": online["p95"],
                    "keypoint_3d_median_cm": distance["median"],
                    "angle_median_deg": angle["median"],
                    "angle_p95_deg": angle["p95"],
                    "rula_agreement": comparison["angle"][
                        "rula_bin_agreement"
                    ],
                    "accepted": bool(comparison["passes_thresholds"]),
                }
            )
    return rows


def cpu_deployment_row(output_root: Path) -> dict[str, Any]:
    """Collect the explicit CPU-only benchmark row."""
    benchmark = load_json(output_root / "cpu_fanbo7/benchmark.json")
    pose = benchmark["stages"]["pose_inference_stereo"]
    online = benchmark["stages"]["end_to_end_online"]
    startup = benchmark["startup"]
    return {
        "hardware": "Intel Core i7-9750H CPU @ 2.60 GHz",
        "backend": "PyTorch CPU",
        "precision": "FP32",
        "dataset": "Fanbo7",
        "frames": benchmark["frames"],
        "repeats": benchmark["repeats"],
        "steady_fps": benchmark["online_fps"],
        "pose_median_ms": pose["median_ms"],
        "pose_p95_ms": pose["p95_ms"],
        "end_to_end_median_ms": online["median_ms"],
        "end_to_end_p95_ms": online["p95_ms"],
        "model_constructor_median_ms": median(
            startup["model_constructor_ms"]
        ),
        "first_frame_median_ms": median(
            startup["first_end_to_end_frame_ms"]
        ),
        "requested_device": benchmark["requested_device"],
        "runtime_device": benchmark["runtime_device"],
        "accepted": None,
    }


def proxy_comparison_files(label: str) -> list[Path]:
    """Find comparison files for one previously tested proxy ladder row."""
    root = project_path(PROXY_ROOT)
    patterns = {
        "CRF 14": "proxy_gate_crf14__*/proxy_equivalence_fanbo*.json",
        "CRF 12": "proxy_gate_crf12__*/proxy_equivalence_fanbo*.json",
        "CRF 10": "proxy_gate_crf10_screen200__*/proxy_equivalence_fanbo*.json",
        "CRF 8": "proxy_gate_crf8_screen200__*/comparison/fanbo*.json",
        "QP0 lossless": (
            "proxy_gate_lossless_qp0_full__*/comparison/fanbo*.json"
        ),
    }
    files = sorted(root.glob(patterns[label]))
    if len(files) != 3:
        raise FileNotFoundError(
            f"Expected three comparison files for {label}, found {files}"
        )
    return files


def aggregate_proxy_gate(label: str) -> dict[str, Any]:
    """Aggregate worst-case metrics across Fanbo3, Fanbo4, and Fanbo7."""
    payloads = [load_json(path) for path in proxy_comparison_files(label)]
    angles = [
        angle
        for payload in payloads
        for angle in payload.get("angles", [])
    ]
    rula = [
        float(angle["rula_like_agreement"])
        for angle in angles
        if angle.get("rula_like_agreement") is not None
    ]
    return {
        "variant": label,
        "scope": "Fanbo3/4/7",
        "keypoint_2d_median_px": max(
            float(payload["keypoint_2d_distance_px"]["median"])
            for payload in payloads
        ),
        "keypoint_2d_p95_px": max(
            float(payload["keypoint_2d_distance_px"]["p95"])
            for payload in payloads
        ),
        "keypoint_3d_median_cm": max(
            float(payload["keypoint_3d_distance_cm"]["median"])
            for payload in payloads
        ),
        "keypoint_3d_p95_cm": max(
            float(payload["keypoint_3d_distance_cm"]["p95"])
            for payload in payloads
        ),
        "angle_median_deg": max(
            float(angle["absolute_difference_deg"]["median"])
            for angle in angles
        ),
        "angle_p95_deg": max(
            float(angle["absolute_difference_deg"]["p95"])
            for angle in angles
        ),
        "rula_agreement": min(rula) if rula else None,
        "passed": all(
            bool(payload["gates"]["passed"]) for payload in payloads
        ),
    }


def fanbo7_refinement_rows(output_root: Path) -> list[dict[str, Any]]:
    """Build the equal-frame Fanbo7 compression-boundary table."""
    refinement = load_json(
        output_root / "compression_refinement_summary.json"
    )
    qp0_path = project_path(
        PROXY_ROOT
        / "proxy_gate_lossless_qp0_screen200__20260726__90bbd51"
        / "comparison/fanbo7.json"
    )
    qp0 = load_json(qp0_path)
    source = refinement["source"]
    rows = [
        {
            "variant": "QP0 lossless",
            "total_bytes": int(source["left_bytes"])
            + int(source["right_bytes"]),
            "keypoint_2d_median_px": qp0["keypoint_2d_distance_px"][
                "median"
            ],
            "keypoint_2d_p95_px": qp0["keypoint_2d_distance_px"]["p95"],
            "keypoint_3d_median_cm": qp0["keypoint_3d_distance_cm"][
                "median"
            ],
            "keypoint_3d_p95_cm": qp0["keypoint_3d_distance_cm"]["p95"],
            "angle_median_deg": qp0["angles"][0][
                "absolute_difference_deg"
            ]["median"],
            "angle_p95_deg": qp0["angles"][0][
                "absolute_difference_deg"
            ]["p95"],
            "rula_agreement": qp0["angles"][0]["rula_like_agreement"],
            "passed": True,
        }
    ]
    for variant in refinement["variants"]:
        rows.append(
            {
                "variant": variant["name"],
                "total_bytes": int(variant["left_bytes"])
                + int(variant["right_bytes"]),
                **{
                    key: variant[key]
                    for key in (
                        "keypoint_2d_median_px",
                        "keypoint_2d_p95_px",
                        "keypoint_3d_median_cm",
                        "keypoint_3d_p95_cm",
                        "angle_median_deg",
                        "angle_p95_deg",
                        "rula_agreement",
                        "passed",
                    )
                },
            }
        )
    return rows


def lossless_size_rows() -> list[dict[str, Any]]:
    """Return full-sequence raw-to-QP0 size changes for all datasets."""
    manifest = load_json(project_path(PROXY_ROOT / "inputs/proxy_manifest.json"))
    rows: list[dict[str, Any]] = []
    for dataset in ("fanbo3", "fanbo4", "fanbo7"):
        records = [
            record
            for record in manifest["records"]
            if record.get("dataset") == dataset
            and record.get("variant") == "lossless_qp0"
            and record.get("source_bytes") is not None
        ]
        if len(records) != 2:
            raise ValueError(f"Missing full QP0 manifest records for {dataset}")
        raw_bytes = sum(int(record["source_bytes"]) for record in records)
        proxy_bytes = sum(
            int(record["destination_bytes"]) for record in records
        )
        rows.append(
            {
                "dataset": dataset.capitalize(),
                "raw_bytes": raw_bytes,
                "proxy_bytes": proxy_bytes,
                "ratio": proxy_bytes / raw_bytes,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows using a stable union of field names."""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 2) -> str:
    """Format an optional numeric table cell."""
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def mib(value: int) -> str:
    """Format bytes as MiB."""
    return f"{value / (1024 ** 2):.1f}"


def result_badge(value: bool | None, cn: bool) -> str:
    """Return a styled pass/fail/not-applicable label."""
    if value is None:
        label = "仅测速" if cn else "Timing only"
        return f'<span class="badge neutral">{label}</span>'
    if value:
        label = "通过" if cn else "Pass"
        return f'<span class="badge pass">{label}</span>'
    label = "未通过" if cn else "Fail"
    return f'<span class="badge fail">{label}</span>'


def render_report(
    output_root: Path,
    cpu: dict[str, Any],
    gpu: list[dict[str, Any]],
    broad_compression: list[dict[str, Any]],
    refinement: list[dict[str, Any]],
    sizes: list[dict[str, Any]],
    *,
    cn: bool,
) -> str:
    """Render one self-contained report language."""
    title = (
        "CPU、GPU 部署与视频压缩最终核查"
        if cn
        else "Final CPU, GPU Deployment, and Video Compression Check"
    )
    gpu_fanbo7 = next(
        row
        for row in gpu
        if row["backend"] == "PyTorch" and row["dataset"] == "Fanbo7"
    )
    speedup = gpu_fanbo7["steady_fps"] / cpu["steady_fps"]
    deployment_rows = [cpu, *gpu]
    deployment_html = "".join(
        f"""
        <tr>
          <td>{html.escape(str(row['hardware']))}</td>
          <td>{html.escape(str(row['backend']))} {row['precision']}</td>
          <td>{row['dataset']}</td>
          <td>{row['frames']} × {row['repeats']}</td>
          <td>{fmt(row['steady_fps'])}</td>
          <td>{fmt(row['pose_median_ms'])} / {fmt(row['pose_p95_ms'])}</td>
          <td>{fmt(row['end_to_end_median_ms'])} / {fmt(row['end_to_end_p95_ms'])}</td>
          <td>{fmt(row.get('angle_median_deg'))} / {fmt(row.get('angle_p95_deg'))}</td>
          <td>{result_badge(row.get('accepted'), cn)}</td>
        </tr>
        """
        for row in deployment_rows
    )
    broad_html = "".join(
        f"""
        <tr>
          <td>{row['variant']}</td>
          <td>{fmt(row['keypoint_2d_median_px'])} / {fmt(row['keypoint_2d_p95_px'])}</td>
          <td>{fmt(row['keypoint_3d_median_cm'])} / {fmt(row['keypoint_3d_p95_cm'])}</td>
          <td>{fmt(row['angle_median_deg'])} / {fmt(row['angle_p95_deg'])}</td>
          <td>{fmt(100 * row['rula_agreement'], 1)}%</td>
          <td>{result_badge(row['passed'], cn)}</td>
        </tr>
        """
        for row in broad_compression
    )
    qp0_bytes = refinement[0]["total_bytes"]
    refinement_html = "".join(
        f"""
        <tr>
          <td>{row['variant']}</td>
          <td>{mib(row['total_bytes'])}</td>
          <td>{100 * (row['total_bytes'] / qp0_bytes - 1):+.1f}%</td>
          <td>{fmt(row['keypoint_2d_median_px'])} / {fmt(row['keypoint_2d_p95_px'])}</td>
          <td>{fmt(row['keypoint_3d_median_cm'])} / {fmt(row['keypoint_3d_p95_cm'])}</td>
          <td>{fmt(row['angle_median_deg'])} / {fmt(row['angle_p95_deg'])}</td>
          <td>{fmt(100 * row['rula_agreement'], 1)}%</td>
          <td>{result_badge(row['passed'], cn)}</td>
        </tr>
        """
        for row in refinement
    )
    sizes_html = "".join(
        f"""
        <tr>
          <td>{row['dataset']}</td>
          <td>{mib(row['raw_bytes'])}</td>
          <td>{mib(row['proxy_bytes'])}</td>
          <td>{100 * row['ratio']:.1f}%</td>
        </tr>
        """
        for row in sizes
    )
    if cn:
        lead = (
            f"本机纯 CPU 稳定速度为 <strong>{cpu['steady_fps']:.2f} FPS</strong>；"
            f"A6000 PyTorch FP32 在同一 Fanbo7 路线上约为 "
            f"<strong>{gpu_fanbo7['steady_fps']:.2f} FPS</strong>，约快 {speedup:.1f} 倍。"
            "视频压缩的严格结论是：有损 CRF 1–14 均未保持重建等价，"
            "只有 H.264 QP0 灰度无损通过。"
        )
        startup_text = (
            f"YOLO 对象创建中位时间为 {cpu['model_constructor_median_ms']:.0f} ms；"
            f"第一对双目帧中位时间为 {cpu['first_frame_median_ms']:.0f} ms。"
            "稳定 FPS 从第二帧开始统计，因此没有把一次性启动成本平均到每帧。"
        )
        deployment_note = (
            "CPU 与 GPU 的软件环境、样本长度不同，所以该表用于部署量级判断，"
            "不是严格的硬件微基准。TensorRT 的角度差表示相对已接受 PyTorch 输出的变化。"
        )
        compression_note = (
            "表中是跨 Fanbo3/4/7 的最差值。CRF 10 和 8 使用 200 帧筛选；"
            "CRF 12、14 与 QP0 使用完整序列。门槛为 2D 0.5/2 px、3D 0.5/2 cm、"
            "角度 0.5°/2°（median/p95）以及 RULA ≥99%。"
        )
        refinement_note = (
            "为进一步定位边界，Fanbo7 又补测了 CRF 1/2/4。CRF 1 和 2 甚至比"
            "灰度 QP0 文件更大，却仍改变了关键点结果；CRF 4 只节省约 4%，也未通过。"
        )
        recommendation = (
            "正式论文实验和远程 GPU 重建使用 H.264 QP0、灰度、MKV。"
            "它把完整双目视频压到原始大小约 35%，但通常仍是每路 400–700 MiB，"
            "不能压到 50–60 MB。50–60 MB 版本可以用于快速查看或预览，"
            "但必须标成有损代理，不能默认与原视频的 3D 结果等价。"
        )
        source_title = "测试定义"
        deployment_title = "1. CPU 与 GPU 部署对比"
        compression_title = "2. 已有三数据集压缩门槛"
        refinement_title = "3. Fanbo7 近无损边界补测（200帧双目）"
        size_title = "4. 推荐无损代理的完整序列大小"
        conclusion_title = "最终建议"
        col = {
            "hardware": "硬件",
            "route": "运行方式",
            "data": "数据",
            "samples": "帧×重复",
            "fps": "稳定 FPS",
            "pose": "双目2D p50/p95 (ms)",
            "online": "端到端 p50/p95 (ms)",
            "angle": "角度变化 med/p95 (°)",
            "gate": "结论",
        }
    else:
        lead = (
            f"The forced CPU-only steady rate is <strong>{cpu['steady_fps']:.2f} FPS</strong>. "
            f"The A6000 PyTorch FP32 Fanbo7 rate is <strong>{gpu_fanbo7['steady_fps']:.2f} FPS</strong>, "
            f"about {speedup:.1f}× faster. Under the strict reconstruction-equivalence gate, "
            "all lossy CRF 1–14 variants failed; only H.264 QP0 grayscale lossless passed."
        )
        startup_text = (
            f"Median YOLO object construction was {cpu['model_constructor_median_ms']:.0f} ms, "
            f"and the first stereo frame took {cpu['first_frame_median_ms']:.0f} ms. "
            "Steady FPS starts at the second frame, so one-off startup cost is not amortized into frame time."
        )
        deployment_note = (
            "CPU and GPU software environments and sample lengths differ, so this table gives deployment-scale "
            "evidence rather than a controlled hardware microbenchmark. TensorRT angle change is measured against "
            "the accepted PyTorch output."
        )
        compression_note = (
            "Values are the worst result across Fanbo3/4/7. CRF 10 and 8 use 200-frame screening; "
            "CRF 12, 14, and QP0 use full sequences. Gates are 2D 0.5/2 px, 3D 0.5/2 cm, "
            "angle 0.5°/2° (median/p95), and RULA ≥99%."
        )
        refinement_note = (
            "Fanbo7 CRF 1/2/4 tests refined the near-lossless boundary. CRF 1 and 2 were even larger "
            "than grayscale QP0 while still changing keypoints; CRF 4 saved only about 4% and also failed."
        )
        recommendation = (
            "Use H.264 QP0 grayscale MKV for formal thesis experiments and remote reconstruction. "
            "It reduces a full stereo sequence to about 35% of the raw size, but each camera file remains "
            "roughly 400–700 MiB. A 50–60 MB copy is suitable for visual review only; it must be labelled as a "
            "lossy proxy and cannot be assumed to preserve 3D reconstruction."
        )
        source_title = "Test definition"
        deployment_title = "1. CPU and GPU deployment comparison"
        compression_title = "2. Existing three-dataset compression gate"
        refinement_title = "3. Fanbo7 near-lossless boundary test (200 stereo frames)"
        size_title = "4. Full-sequence size of the recommended lossless proxy"
        conclusion_title = "Final recommendation"
        col = {
            "hardware": "Hardware",
            "route": "Runtime",
            "data": "Dataset",
            "samples": "Frames × repeats",
            "fps": "Steady FPS",
            "pose": "Stereo 2D p50/p95 (ms)",
            "online": "End-to-end p50/p95 (ms)",
            "angle": "Angle change med/p95 (°)",
            "gate": "Decision",
        }
    return f"""<!doctype html>
<html lang="{'zh-CN' if cn else 'en'}">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--ink:#172033;--muted:#657086;--line:#dfe5ed;--blue:#2457d6;--pale:#f4f7fb;--pass:#0d7a48;--fail:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f7;color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:28px auto;padding:34px;background:white;border-radius:18px;box-shadow:0 8px 28px #1a2b4912}}
h1{{font-size:30px;margin:0 0 8px}} h2{{margin:32px 0 10px;font-size:21px;border-left:5px solid var(--blue);padding-left:11px}}
p{{margin:8px 0}} .lead{{font-size:17px;background:#edf4ff;border-radius:12px;padding:16px 18px}}
.note{{color:var(--muted);font-size:13px}} .callout{{background:#f0faf5;border:1px solid #bce6cf;border-radius:12px;padding:15px 18px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}} table{{border-collapse:collapse;width:100%;min-width:900px}}
th,td{{padding:10px 11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{background:var(--pale);font-size:13px}} tr:last-child td{{border-bottom:0}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-weight:650;font-size:12px}} .pass{{color:var(--pass);background:#e8f7ef}} .fail{{color:var(--fail);background:#fff0ee}} .neutral{{color:#566174;background:#eef1f5}}
code{{background:#eef1f5;border-radius:5px;padding:2px 5px}} .command{{white-space:pre-wrap;background:#172033;color:#e9eef8;padding:12px 14px;border-radius:9px;overflow:auto}}
</style></head>
<body><main>
<h1>{title}</h1><p class="note">2026-08-05 · YOLOv8m + SKT · fixed stereo calibration</p>
<p class="lead">{lead}</p>
<h2>{source_title}</h2><p>{startup_text}</p>
<h2>{deployment_title}</h2>
<div class="table-wrap"><table><thead><tr>
<th>{col['hardware']}</th><th>{col['route']}</th><th>{col['data']}</th><th>{col['samples']}</th><th>{col['fps']}</th><th>{col['pose']}</th><th>{col['online']}</th><th>{col['angle']}</th><th>{col['gate']}</th>
</tr></thead><tbody>{deployment_html}</tbody></table></div><p class="note">{deployment_note}</p>
<h2>{compression_title}</h2>
<div class="table-wrap"><table><thead><tr><th>Codec</th><th>2D med/p95 (px)</th><th>3D med/p95 (cm)</th><th>Angle med/p95 (°)</th><th>RULA</th><th>{col['gate']}</th></tr></thead><tbody>{broad_html}</tbody></table></div><p class="note">{compression_note}</p>
<h2>{refinement_title}</h2>
<div class="table-wrap"><table><thead><tr><th>Codec</th><th>Total MiB</th><th>vs QP0</th><th>2D med/p95 (px)</th><th>3D med/p95 (cm)</th><th>Angle med/p95 (°)</th><th>RULA</th><th>{col['gate']}</th></tr></thead><tbody>{refinement_html}</tbody></table></div><p class="note">{refinement_note}</p>
<h2>{size_title}</h2>
<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Raw stereo MiB</th><th>QP0 stereo MiB</th><th>Remaining size</th></tr></thead><tbody>{sizes_html}</tbody></table></div>
<p class="command">ffmpeg -i input.avi -map 0:v:0 -an -fps_mode passthrough -vf hflip,vflip -c:v libx264 -preset slow -qp 0 -pix_fmt gray output.mkv</p>
<h2>{conclusion_title}</h2><div class="callout">{recommendation}</div>
<p class="note">CSV: deployment_comparison.csv · compression_screening.csv</p>
</main></body></html>"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate CSV and bilingual HTML deliverables."""
    args = parse_args(argv)
    output_root = project_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cpu = cpu_deployment_row(output_root)
    gpu = gpu_deployment_rows()
    broad = [
        aggregate_proxy_gate(label)
        for label in (
            "QP0 lossless",
            "CRF 14",
            "CRF 12",
            "CRF 10",
            "CRF 8",
        )
    ]
    refinement = fanbo7_refinement_rows(output_root)
    sizes = lossless_size_rows()
    write_csv(output_root / "deployment_comparison.csv", [cpu, *gpu])
    write_csv(output_root / "compression_screening.csv", [*broad, *refinement])
    (output_root / "report.html").write_text(
        render_report(output_root, cpu, gpu, broad, refinement, sizes, cn=False),
        encoding="utf-8",
    )
    (output_root / "report_CN.html").write_text(
        render_report(output_root, cpu, gpu, broad, refinement, sizes, cn=True),
        encoding="utf-8",
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
