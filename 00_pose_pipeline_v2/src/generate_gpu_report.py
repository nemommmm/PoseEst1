"""Generate the Chinese GPU migration and real-time evaluation report."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def configure_cjk_font() -> None:
    """Use an available CJK font for chart labels."""
    candidates = (
        Path.home() / "Library/Fonts/TencentSans-W7.ttf",
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(
                fname=str(font_path)
            ).get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


CPU_GPU = {
    "Fanbo7 A257": (9.65, 12.26),
    "Fanbo4 A257": (10.63, 14.61),
    "Fanbo9 A255": (18.13, 24.83),
    "Fanbo9 A257": (13.10, 17.88),
}


def figure_uri(fig: plt.Figure) -> str:
    """Return a PNG figure encoded as a data URI."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def speedup_chart() -> str:
    """Create the CPU-to-GPU speedup chart."""
    labels = list(CPU_GPU)
    frame = [CPU_GPU[label][0] for label in labels]
    pose = [CPU_GPU[label][1] for label in labels]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar([i - 0.18 for i in x], frame, 0.36, label="端到端帧处理", color="#277da1")
    ax.bar([i + 0.18 for i in x], pose, 0.36, label="YOLO 推理", color="#f8961e")
    ax.axhline(1, color="#6b7280", linewidth=1)
    ax.set_ylabel("相对本地 CPU 加速比")
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    return figure_uri(fig)


def backend_chart() -> str:
    """Create a deployment speed and accuracy trade-off chart."""
    points = [
        ("PyTorch FP32", 28.88, 10.43, "#2a9d8f"),
        ("ONNX CUDA", 17.02, 12.36, "#e9c46a"),
        ("TensorRT FP16", 30.21, 12.81, "#e76f51"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for name, fps, mae, color in points:
        ax.scatter(fps, mae, s=130, color=color, edgecolor="white", linewidth=1.2)
        ax.annotate(name, (fps, mae), xytext=(7, 7), textcoords="offset points")
    ax.axvline(12.5, linestyle="--", color="#6b7280", label="相机帧率 12.5 fps")
    ax.set_xlabel("Fanbo7 真实端到端吞吐（fps，越高越好）")
    ax.set_ylabel("RightElbow MAE（°，越低越好）")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return figure_uri(fig)


def build_html() -> str:
    """Build a self-contained Chinese HTML report."""
    configure_cjk_font()
    speedup = speedup_chart()
    backend = backend_chart()
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双目姿态 Pipeline：A6000 GPU 迁移与实时评估</title>
<style>
:root{{--ink:#172033;--muted:#5f6b7a;--line:#dce3ec;--blue:#155eef;--navy:#102a43;--green:#087f5b;--amber:#b45309;--red:#b42318;--bg:#f5f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}}
header{{background:linear-gradient(125deg,#102a43,#155eef);color:white;padding:54px max(6vw,28px) 46px}} header h1{{max-width:980px;margin:0 0 12px;font-size:34px;line-height:1.25}} header p{{max-width:900px;margin:0;opacity:.9}}
.layout{{display:grid;grid-template-columns:230px minmax(0,920px);gap:32px;max-width:1240px;margin:30px auto;padding:0 24px}} nav{{position:sticky;top:20px;align-self:start;background:white;border:1px solid var(--line);border-radius:12px;padding:18px}} nav a{{display:block;color:#38506a;text-decoration:none;padding:6px 4px}} nav a:hover{{color:var(--blue)}} main{{min-width:0}} section{{background:white;border:1px solid var(--line);border-radius:14px;padding:26px 30px;margin-bottom:22px;box-shadow:0 4px 18px #2230470a}}
h2{{margin:0 0 16px;color:var(--navy);font-size:24px}} h3{{margin:22px 0 8px;font-size:18px}} p{{margin:8px 0 14px}} .lead{{font-size:17px}} .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}} .card{{border:1px solid var(--line);border-radius:10px;padding:14px}} .metric{{font-size:25px;font-weight:750;color:var(--blue)}} .ok,.warn,.bad{{font-weight:700}} .ok{{color:var(--green)}} .warn{{color:var(--amber)}} .bad{{color:var(--red)}}
table{{width:100%;border-collapse:collapse;margin:14px 0 20px;font-size:14px}} th{{background:#eef4ff;text-align:left}} th,td{{border:1px solid var(--line);padding:9px 10px;vertical-align:top}} tr:nth-child(even) td{{background:#fafbfd}} figure{{margin:22px 0}} figure img{{width:100%;border:1px solid var(--line);border-radius:10px}} figcaption{{color:var(--muted);font-size:13px;margin-top:7px}} .callout{{border-left:4px solid var(--blue);background:#f1f6ff;padding:13px 16px;margin:16px 0}} code{{background:#edf1f7;padding:2px 5px;border-radius:4px}} a{{color:var(--blue)}} ul,ol{{padding-left:22px}} footer{{color:var(--muted);font-size:13px;padding:4px 0 30px}}
@media(max-width:850px){{.layout{{display:block}} nav{{position:static;margin-bottom:20px}} .cards{{grid-template-columns:1fr}} header h1{{font-size:28px}} section{{padding:22px 18px}}}}
</style></head><body>
<header><h1>双目姿态 Pipeline：A6000 GPU 迁移、新模型准入与实时评估</h1><p>PoseEst1 · Fanbo4 / Fanbo7 / Fanbo9 · 2026-07-12<br>评价重点：关节角度与 RULA；Xsens 仅作为 Xsens-derived reference / comparison system。</p></header>
<div class="layout"><nav><strong>目录</strong><a href="#summary">1. 执行摘要</a><a href="#method">2. 方法与环境</a><a href="#migration">3. 迁移与复现</a><a href="#realtime">4. 实时后端</a><a href="#models">5. 新模型准入</a><a href="#recommend">6. 建议与限制</a></nav><main>
<section id="summary"><h2>1. 执行摘要</h2><p class="lead">当前最可靠的远程方案是 <strong>RTX A6000 + PyTorch FP32 + YOLOv8m-pose + 确定性 CUDA</strong>。它在测试数据上保持与本地 CPU 几乎一致的角度结果，同时稳定超过 12.5 fps 相机输入要求。</p>
<div class="cards"><div class="card"><div class="metric">28.9–29.9 fps</div>PyTorch 真实在线吞吐</div><div class="card"><div class="metric">9.7×–18.1×</div>端到端相对 CPU 加速</div><div class="card"><div class="metric">≈0.0004°</div>确定性 GPU 与本地角度差</div></div>
<div class="callout"><span class="ok">接受：</span>PyTorch FP32。<span class="bad">暂不接受：</span>ONNX CUDA、TensorRT FP16、RTMPose、RTMO。TensorRT 的模型推理更快，但完整角度精度未通过等价性门槛。</div></section>

<section id="method"><h2>2. 方法与环境</h2><table><tr><th>项目</th><th>设置</th></tr><tr><td>远程硬件</td><td>NVIDIA RTX A6000 48 GB；RunPod 持久卷位于 <code>/workspace</code></td></tr><tr><td>软件</td><td>PyTorch 2.8 / CUDA 12.8；Ultralytics 8.3.235；ONNX Runtime GPU 1.22；TensorRT 10.13</td></tr><tr><td>数据</td><td>Fanbo4 远距、Fanbo7 近距、Fanbo9 双相机 A255/A257；2048×1536 双目</td></tr><tr><td>实时门槛</td><td>端到端 ≥12.5 fps；不仅统计模型 forward，还包含解码、双路姿态、几何处理</td></tr><tr><td>精度原则</td><td>优先关节角度和 RULA；Xsens 输出只作为外部比较参考，不视为绝对 Ground Truth</td></tr></table>
<p>所有 GPU 复现实验启用确定性计算：关闭 TF32 与 cuDNN benchmark，并请求确定性算法。这样消除了默认 CUDA 算法选择造成的约 0.79° 跨平台角度偏差。</p>
<h3>视频传输门槛</h3><p>H.265 CRF 18 虽把单文件从约 1.3 GB 压至约 13 MB，但造成 RightElbow 原始/压缩轨迹 MAE 2.28°，因此拒绝。最终采用 H.264 QP 0 无损压缩：抽样像素误差为 0，关键点与角度完全一致；Fanbo7 双路由约 2.6 GB 降至 938 MB。</p></section>

<section id="migration"><h2>3. GPU 复现与 CPU 加速</h2><table><tr><th>数据</th><th>参考比较结果</th><th>有效率 / RULA 一致率</th><th>GPU 稳态</th></tr><tr><td>Fanbo7 A257</td><td>RightElbow MAE 10.431°</td><td>0.427 / 0.941</td><td>YOLO 41.9 fps；在线 29.2 fps</td></tr><tr><td>Fanbo4 A257</td><td>RightElbow MAE 9.820°（FastSAM reference）</td><td>0.603 / —</td><td>29.7 fps</td></tr><tr><td>Fanbo9 A255</td><td>8 关节平均 MAE 10.700°</td><td>0.681 / —</td><td>29.9 fps</td></tr><tr><td>Fanbo9 A257</td><td>8 关节平均 MAE 6.420°</td><td>0.857 / —</td><td>29.9 fps</td></tr></table>
<figure><img src="{speedup}" alt="CPU GPU speedup"><figcaption>图 1：相对本地 Mac CPU 的加速比。A6000 的收益在不同视频与本地解码负载下有所变化。</figcaption></figure>
<p>A257 在 Fanbo9 上同时取得更低的平均角度差与更高有效率，适合作为后续部署与消融实验的优先相机。这里的“MAE”均表示与对应外部比较系统的一致性误差，不能解释为绝对真实误差。</p></section>

<section id="realtime"><h2>4. 实时后端与模型阶梯</h2><h3>真实端到端阶段计时（前 200 帧）</h3><table><tr><th>配置</th><th>数据</th><th>解码 mean / median</th><th>双路姿态</th><th>在线总延迟 / fps</th><th>判定</th></tr><tr><td>PyTorch FP32</td><td>Fanbo7</td><td>10.97 / 9.41 ms</td><td>23.44 ms</td><td>34.63 ms / 28.88</td><td class="ok">接受</td></tr><tr><td>PyTorch FP32</td><td>Fanbo4</td><td>10.21 / 9.22 ms</td><td>23.71 ms</td><td>34.21 ms / 29.23</td><td class="ok">接受</td></tr><tr><td>ONNX CUDA</td><td>Fanbo7</td><td>33.0 / 9.07 ms</td><td>25.53 ms</td><td>58.74 ms / 17.02</td><td class="bad">精度拒绝</td></tr><tr><td>TensorRT FP16</td><td>Fanbo7</td><td>抖动显著</td><td>17.49 ms</td><td>33.11 ms / 30.21</td><td class="bad">精度拒绝</td></tr></table>
<figure><img src="{backend}" alt="Backend speed accuracy"><figcaption>图 2：Fanbo7 后端速度—角度一致性。达到 12.5 fps 只是必要条件，未满足精度等价性的点不能进入部署。</figcaption></figure>
<p>TensorRT FP16 将姿态模型阶段缩短约 26%–31%，但 Fanbo7 RightElbow MAE 从 10.43° 增至 12.81°，且与 PyTorch 前 200 帧角度轨迹相差约 3.44°。ONNX CUDA 也出现约 3.72° 的轨迹差。因此当前生产基线保留 PyTorch FP32。</p>
<h3>模型大小与双目 batch</h3><p>YOLO11 n/s/m/l 的模型阶段约为 21.96–33.23 ms；网络盘解码抖动会使短窗口端到端 fps 非单调，故选型应优先看姿态阶段和完整精度。YOLOv8m 左右图 batch=2 的模型吞吐仅提升 1.11×；现有 crop tracker 对两侧使用不同 ROI，不能直接替换为等价的生产 batch。</p></section>

<section id="models"><h2>5. 新模型调研与双目准入</h2><p>调研聚焦 <a href="https://arxiv.org/abs/2312.07526">RTMO</a>、<a href="https://arxiv.org/abs/2303.07399">RTMPose</a>、<a href="https://arxiv.org/abs/2407.08634">RTMW</a>、<a href="https://github.com/ViTAE-Transformer/ViTPose">ViTPose</a>，以及面向遮挡的人机协作多视角方法 <a href="https://arxiv.org/abs/2408.15810">Multi-view Pose Fusion</a>。候选必须同时通过 Fanbo7 近距与 Fanbo4 远距的时延和校正后极线一致性门槛。</p>
<table><tr><th>候选</th><th>Fanbo7 ms</th><th>Fanbo7 极线 median / p95</th><th>Fanbo4 ms</th><th>Fanbo4 极线 median / p95</th><th>结果</th></tr><tr><td>RTMPose-S</td><td>86.3</td><td>121.3 / 289.1 px</td><td>75.3</td><td>0.58 / 3.87 px</td><td class="bad">近距几何拒绝</td></tr><tr><td>RTMPose-M</td><td>92.6</td><td>4.83 / 323.6 px</td><td>118.9</td><td>0.41 / 94.7 px</td><td class="bad">时延与离群拒绝</td></tr><tr><td>RTMPose-X</td><td>126.2</td><td>2.99 / 123.0 px</td><td>180.8</td><td>0.53 / 121.1 px</td><td class="bad">时延与离群拒绝</td></tr><tr><td>RTMO-S</td><td>65.9</td><td>196.2 / 357.0 px</td><td>44.4</td><td>0.87 / 385.9 px</td><td class="bad">左右语义不一致</td></tr><tr><td>RTMO-M</td><td>45.8</td><td>305.6 / 426.5 px</td><td>41.9</td><td>1.16 / 406.2 px</td><td class="bad">左右语义不一致</td></tr></table>
<p>RTMO 达到模型级速度门槛，却在部分帧产生灾难性的左右关节对应错误；RTMPose-X 改善中位数但仍有大离群且超过 80 ms 双目预算。为避免从错误的 2D 对应生成看似合理的 3D 数字，本轮没有继续做完整角度评估。更重的 ViTPose-L/H 与 RTMW-L 因较轻的 RTMPose-X 已经失败，未在本轮占用额外 GPU 时间。</p></section>

<section id="recommend"><h2>6. 部署建议、论文表述与限制</h2><ol><li><strong>立即可用：</strong>保持 PyTorch FP32 / YOLOv8m / 确定性 CUDA；以 A257 为优先输入，保留每次运行的模型、库版本和确定性标志。</li><li><strong>下一阶段：</strong>把视频解码移出网络盘敏感路径，验证本地 NVMe、NVDEC 与 GStreamer/DeepStream；用长时间流和队列背压报告 p50/p95/p99，而不仅是平均 fps。</li><li><strong>TensorRT：</strong>先逐层定位导出前后关键点偏差、预处理和 NMS 差异；只有完整关节角与 RULA 等价性通过后才能部署，不能仅依据 forward 加速。</li><li><strong>新模型：</strong>若继续 RTMO，必须增加左右实例/关节关联和极线约束，而不是直接独立推理后按置信度三角化。精度上限模型应在这个关联层稳定后再测试。</li><li><strong>论文：</strong>使用 “compared against Xsens-derived reference” 或 “agreement with Xsens comparison system”；讨论 Xsens 校准 offset，尤其是肘部，避免误差转移。</li></ol>
<h3>限制</h3><ul><li>A6000 结果不能直接代表 Jetson 等边缘设备；需要目标硬件复测。</li><li>解码来自 RunPod 持久网络卷，偶发 I/O 抖动抬高 p95；模型阶段比短窗口总 fps 更适合后端比较。</li><li>数据覆盖三个 session，尚不足以代表全部遮挡、人员和服装条件。</li><li>Xsens 不是绝对基准；角度 MAE 是系统间一致性，而非真实误差上界。</li></ul></section>
<footer>生成脚本：<code>00_pose_pipeline_v2/src/generate_gpu_report.py</code>。本报告为计划要求的首版中文报告；英文正式版待内容确认后生成。</footer>
</main></div></body></html>"""


def main() -> None:
    """Write the self-contained report to disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("00_pose_pipeline_v2/runs/gpu_realtime_eval/report_CN.html"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(), encoding="utf-8")
    print(f"Report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
