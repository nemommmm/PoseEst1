"""Generate bilingual reports for stereo human-prior candidate validation."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def configure_font() -> None:
    """Select a local font that supports Chinese chart labels."""
    candidates = [
        Path.home() / "Library/Fonts/TencentSans-W7.ttf",
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            break


def image_uri(fig: plt.Figure) -> str:
    """Encode one Matplotlib figure as an inline PNG."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def pose2sim_chart(chinese: bool) -> str:
    """Plot the Pose2Sim angle gate."""
    labels = ["Fanbo7", "Fanbo4"]
    baseline = [10.47, 15.47]
    candidate = [52.36, 24.98]
    x = range(2)
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.bar([i - 0.18 for i in x], baseline, 0.36, label="SKT 基线" if chinese else "SKT baseline", color="#2878b5")
    ax.bar([i + 0.18 for i in x], candidate, 0.36, label="Pose2Sim/OpenSim", color="#d1495b")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("RightElbow MAE（°）" if chinese else "RightElbow MAE (deg)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    return image_uri(fig)


def metrabs_chart(chinese: bool) -> str:
    """Plot MeTRAbs speed and stereo consistency gate results."""
    labels = ["Fanbo7", "Fanbo4"]
    fps = [4.76, 4.96]
    consistency = [69.45, 66.38]
    x = range(2)
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.bar([i - 0.18 for i in x], fps, 0.36, color="#2a9d8f")
    ax.axhline(12.5, color="#b42318", linestyle="--")
    ax.set_ylabel("fps")
    ax.set_xticks(list(x), labels)
    other = ax.twinx()
    other.bar([i + 0.18 for i in x], consistency, 0.36, color="#f4a261")
    other.set_ylabel("cm")
    handles = [
        Patch(facecolor="#2a9d8f"),
        Patch(facecolor="#f4a261"),
        Line2D([0], [0], color="#b42318", linestyle="--"),
    ]
    labels_legend = (
        ["端到端 fps", "左右 3D 差", "12.5 fps 门槛"]
        if chinese
        else ["End-to-end fps", "Stereo 3D disagreement", "12.5 fps gate"]
    )
    ax.legend(handles, labels_legend, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    other.spines["top"].set_visible(False)
    return image_uri(fig)


def pareto_chart(chinese: bool) -> str:
    """Plot the comparable accuracy-throughput admission results."""
    baseline_fps = [28.88, 29.23]
    baseline_mae = [10.47, 15.47]
    pose2sim_fps_upper = [1000.0 / 102.6, 1000.0 / 101.4]
    pose2sim_mae = [52.36, 24.98]
    datasets = ["Fanbo7", "Fanbo4"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for index, dataset in enumerate(datasets):
        ax.plot(
            [pose2sim_fps_upper[index], baseline_fps[index]],
            [pose2sim_mae[index], baseline_mae[index]],
            color="#9aa5b1",
            linestyle=":",
            linewidth=1.2,
        )
        ax.annotate(dataset, (baseline_fps[index], baseline_mae[index]), xytext=(5, 5), textcoords="offset points")
    ax.scatter(baseline_fps, baseline_mae, s=75, color="#2878b5", label="SKT baseline")
    ax.scatter(
        pose2sim_fps_upper,
        pose2sim_mae,
        s=75,
        marker="X",
        color="#d1495b",
        label="Pose2Sim optimistic upper bound" if not chinese else "Pose2Sim 乐观速度上界",
    )
    ax.axvline(12.5, color="#b42318", linestyle="--", linewidth=1.4, label="12.5 fps gate" if not chinese else "12.5 fps 门槛")
    ax.set_xlabel(
        "Equivalent throughput (fps; higher is better)"
        if not chinese
        else "等效吞吐率（fps，越高越好）"
    )
    ax.set_ylabel("RightElbow MAE (deg; lower is better)" if not chinese else "RightElbow MAE（°，越低越好）")
    ax.text(
        0.02,
        0.96,
        "MeTRAbs and kinematic prior excluded: geometry gates failed"
        if not chinese
        else "MeTRAbs 与轻量运动学先验：几何门槛失败，无有效精度点",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#b42318",
    )
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    return image_uri(fig)


STYLE = """
:root{--ink:#172033;--muted:#607080;--line:#dce3ec;--blue:#155eef;--green:#087f5b;--red:#b42318;--amber:#b45309;--bg:#f5f7fb}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.68 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}
header{background:linear-gradient(125deg,#102a43,#155eef);color:white;padding:48px max(6vw,28px)} header h1{max-width:960px;margin:0 0 10px;font-size:33px} header p{max-width:900px;margin:0;opacity:.9}
.layout{display:grid;grid-template-columns:225px minmax(0,930px);gap:30px;max-width:1230px;margin:28px auto;padding:0 22px} nav{position:sticky;top:18px;align-self:start;background:white;border:1px solid var(--line);border-radius:12px;padding:17px} nav a{display:block;padding:6px 3px;text-decoration:none;color:#38506a} main{min-width:0} section{background:white;border:1px solid var(--line);border-radius:13px;padding:25px 29px;margin-bottom:20px;box-shadow:0 4px 18px #2230470a}
h2{margin:0 0 14px;color:#102a43;font-size:23px} h3{margin:20px 0 7px} p{margin:8px 0 13px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin:16px 0}.card{border:1px solid var(--line);border-radius:9px;padding:13px}.metric{font-size:24px;font-weight:750;color:var(--blue)}
table{width:100%;border-collapse:collapse;margin:13px 0 19px;font-size:13.7px}th{background:#eef4ff;text-align:left}th,td{border:1px solid var(--line);padding:8px 9px;vertical-align:top}tr:nth-child(even) td{background:#fafbfd}.ok{color:var(--green);font-weight:700}.bad{color:var(--red);font-weight:700}.pending{color:var(--amber);font-weight:700}.callout{border-left:4px solid var(--blue);background:#f1f6ff;padding:12px 15px;margin:15px 0}
figure{margin:20px 0}figure img{width:100%;border:1px solid var(--line);border-radius:9px}figcaption{font-size:13px;color:var(--muted)}a{color:var(--blue)}code{background:#edf1f7;padding:2px 5px;border-radius:4px}footer{color:var(--muted);font-size:13px;padding-bottom:28px}@media(max-width:850px){.layout{display:block}nav{position:static;margin-bottom:18px}.cards{grid-template-columns:1fr}section{padding:20px 17px}}
"""


def build_report(chinese: bool) -> str:
    """Build one self-contained language version."""
    configure_font()
    p2s = pose2sim_chart(chinese)
    met = metrabs_chart(chinese)
    pareto = pareto_chart(chinese)
    if chinese:
        title = "双目人体先验与 3D 重建：深度调研及 A6000 准入验证"
        subtitle = "PoseEst1 · Fanbo4 / Fanbo7 · 更新于 2026-07-13 · 关节角与 RULA 优先"
        nav = ["执行摘要", "调研矩阵", "Pose2Sim", "轻量运动学先验", "MeTRAbs", "EasyMocap", "结论与下一步"]
        body = f"""
<section id="summary"><h2>1. 执行摘要</h2><p>本轮验证的核心发现是：对当前严格双目和工效学角度目标，<strong>加入人体先验并不会自动提高精度</strong>。Pose2Sim/OpenSim、MeTRAbs 和新实现的几何条件运动学先验均在预先规定的准入门槛失败，未进入全量评估；EasyMocap/SMPL 技术环境已就绪，但因许可资产尚未提供而未执行。</p><div class="cards"><div class="card"><div class="metric">3 个拒绝</div>Pose2Sim、MeTRAbs、轻量运动学先验</div><div class="card"><div class="metric">1 个待资产</div>EasyMocap/SMPL</div><div class="card"><div class="metric">仍推荐</div>确定性 YOLOv8m + SKT</div></div><div class="callout">Xsens 始终仅作为 Xsens-derived reference / external comparison system；未使用 Xsens 初始帧或原生角度校准任何候选。</div></section>
<section id="research"><h2>2. 近年方法与代码库筛选</h2><table><tr><th>方向</th><th>代表方法</th><th>与本项目的关系</th><th>决定</th></tr><tr><td>运动学逆解</td><td><a href="https://github.com/perfanalytics/pose2sim">Pose2Sim/OpenSim</a></td><td>两相机支持，直接输出关节角，最贴近工效学</td><td class="bad">已实测拒绝</td></tr><tr><td>参数人体模型</td><td><a href="https://github.com/zju3dv/EasyMocap">EasyMocap/SMPL</a></td><td>固定形状、关节活动范围和时间先验，适合离线精度上限</td><td class="pending">等待许可资产</td></tr><tr><td>直接 metric 3D</td><td><a href="https://github.com/isarandi/metrabs">MeTRAbs</a></td><td>标定后输出世界坐标，内置姿态合理性约束</td><td class="bad">已实测拒绝</td></tr><tr><td>端到端多视角</td><td><a href="https://github.com/XunshanMan/MVGFormer">MVGFormer</a></td><td>CVPR 2024；官方预训练配置支持 3–7 视角，不支持严格双目</td><td>不实测</td></tr><tr><td>遮挡融合</td><td><a href="https://arxiv.org/abs/2408.15810">Multi-view Pose Fusion</a></td><td>面向人机协作遮挡，但公开实现和双视角迁移成熟度不足</td><td>后续研究</td></tr><tr><td>稠密视差先验</td><td><a href="https://github.com/NVlabs/FoundationStereo">FoundationStereo</a></td><td>强零样本 stereo depth，但不能直接保证关节语义和角度</td><td>仅作深度辅助候选</td></tr><tr><td>单目人体网格</td><td><a href="https://github.com/naver/multi-hmr">Multi-HMR</a>、<a href="https://github.com/yufu-wang/PromptHMR">PromptHMR</a></td><td>形状先验强，但不是双目几何模型；需额外跨视图一致性层</td><td>不替换主线</td></tr></table><p>筛选依据不是论文 MPJPE 排名，而是：两相机可用性、相机标定接口、关节语义、角度/RULA 可解释性、代码和权重可获得性，以及 A6000 实际延迟。</p></section>
<section id="pose2sim"><h2>3. Pose2Sim / OpenSim 准入结果</h2><p>官方 Pose2Sim 0.10.48 与 OpenSim 4.6 成功运行。输入复用确定性 YOLOv8m/SKT COCO-17 轨迹，未更换 detector。完整序列插值后截取连续 200 帧，按序列缩放 simple OpenSim 模型并运行 IK。</p><table><tr><th>数据</th><th>系统</th><th>RightElbow MAE</th><th>RULA 一致率</th><th>跳变</th><th>先验阶段</th></tr><tr><td>Fanbo7</td><td>SKT</td><td>10.47°</td><td>0.939</td><td>0</td><td>—</td></tr><tr><td>Fanbo7</td><td>Pose2Sim</td><td>52.36°</td><td>0.344</td><td>0</td><td>102.6 ms/帧</td></tr><tr><td>Fanbo4</td><td>SKT</td><td>15.47°</td><td>0.948</td><td>7</td><td>—</td></tr><tr><td>Fanbo4</td><td>Pose2Sim</td><td>24.98°</td><td>0.895</td><td>19</td><td>101.4 ms/帧</td></tr></table><figure><img src="{p2s}"><figcaption>图 1：OpenSim IK 给出了人体可行解，但没有保留当前观测的实际肘部运动。</figcaption></figure><p class="bad">结论：角度和实时门槛同时失败。稀疏 COCO-17 与缺失左臂使 IK 得到“人体可行但动作不一致”的解。</p></section>
<section id="kinematic"><h2>4. 几何条件轻量运动学先验</h2><p>在 Fanbo7 A257 连续 40 帧上，按预定 8 组固定参数网格联合优化双视图 Huber 重投影、质量加权 SKT anchor、固定骨长和时间二阶连续性。参数选择完全依据几何和时间稳定性，未使用 Xsens-derived reference。</p><table><tr><th>指标</th><th>原始 SKT</th><th>运动学先验</th></tr><tr><td>核心关节有效率</td><td>0.8375</td><td>0.9167</td></tr><tr><td>重投影 p50 / p95</td><td>1.366 / 7.788 px</td><td>1.611 / 13.545 px</td></tr><tr><td>骨长 CV</td><td>0.1116</td><td>0.0971</td></tr><tr><td>&gt;10° 跳变数</td><td>53</td><td>42</td></tr><tr><td>高质量点修正 median / p95</td><td>—</td><td>0.206 / 0.631 cm</td></tr><tr><td>先验耗时 / 估算端到端</td><td>—</td><td>5028.9 ms/帧 / 0.198 fps</td></tr></table><p class="bad">结论：虽然缺失点、骨长稳定性和跳变有所改善，但重投影 p95 超过预设 10 px 门槛，速度也远低于 12.5 fps，因此在 Fanbo7 即拒绝。失败适配器已按规则回滚，重建 NPZ、12 张诊断图、预览、GPU 元数据及 SHA256 清单已下载本地。</p></section>
<section id="metrabs"><h2>5. MeTRAbs 双视图融合准入结果</h2><p>使用官方实验性 PyTorch EfficientNetV2-S 权重进行 A6000 forward；官方 MobileNet 权重只有旧 TensorFlow 导出，因此用官方 S 版作为可复现轻量替代。左右图以一个 batch 推理，并通过标定外参转入左相机世界坐标。</p><table><tr><th>数据</th><th>端到端 fps</th><th>推理 ms/双目</th><th>右臂左右 3D 差 median/p95</th><th>2D 差 median/p95</th></tr><tr><td>Fanbo7</td><td>4.76</td><td>86.8</td><td>69.45 / 75.68 cm</td><td>564.8 / 735.1 px</td></tr><tr><td>Fanbo4</td><td>4.96</td><td>82.1</td><td>66.38 / 109.02 cm</td><td>854.6 / 1028.0 px</td></tr></table><figure><img src="{met}"><figcaption>图 2：MeTRAbs-S 同时未通过 12.5 fps 与跨视图几何门槛。</figcaption></figure><p class="bad">结论：通用单目 metric 3D 预测不能直接在当前工业双目上形成一致的世界坐标骨架。更大 L 模型和更多 TTA 不再运行。</p></section>
<section id="smpl"><h2>6. EasyMocap / SMPL 状态</h2><p>EasyMocap SMPL CUDA 模块已在 Torch 2.8 / CUDA 12.8 成功导入，环境重建脚本和资产验证器已完成。正式拟合尚未执行，原因是缺少受许可保护的 <code>SMPL_NEUTRAL.pkl</code>。</p><p class="pending">该路线不是精度失败，而是许可资产门槛未满足。请按 <code>docs/smpl_asset_setup.md</code> 从 SMPL 官方网站获取并上传；不能从非官方镜像代替。</p></section>
<section id="conclusion"><h2>7. 结论与下一步</h2><figure><img src="{pareto}"><figcaption>图 3：可比较候选的精度—速度 Pareto 图。Pose2Sim 横坐标仅按 IK 阶段耗时换算，是未计入检测与三角化的乐观上界；即便如此仍被 SKT 基线支配。MeTRAbs 和轻量运动学先验均因几何门槛失败，没有可报告的有效角度精度点。</figcaption></figure><ol><li>当前正式基线继续使用确定性 PyTorch FP32 YOLOv8m + SKT。</li><li>完成 SMPL 官方资产上传后，只先跑 Fanbo7/Fanbo4 40 帧重投影与关节语义门槛；通过后才跑 200 帧。</li><li>若 SMPL 仍失败，不再重复已拒绝的运动学网格；优先改善遮挡下的双目 2D 语义一致性。</li><li>FoundationStereo 可用于遮挡区深度诊断，但不能替代关键点语义。</li><li>所有调参继续只依赖几何和时间稳定性；Xsens 仅用于最终 agreement 报告，避免误差转移。</li></ol></section>"""
    else:
        title = "Stereo Human Priors and 3D Reconstruction: Research Review and A6000 Gates"
        subtitle = "PoseEst1 · Fanbo4 / Fanbo7 · updated 2026-07-13 · Joint-angle and RULA priority"
        nav = ["Executive summary", "Research matrix", "Pose2Sim", "Kinematic prior", "MeTRAbs", "EasyMocap", "Conclusions"]
        body = f"""
<section id="summary"><h2>1. Executive summary</h2><p>Human priors did not automatically improve the current calibrated stereo ergonomics pipeline. Pose2Sim/OpenSim, MeTRAbs, and the project-specific geometry-conditioned kinematic prior failed predefined admission gates and did not advance to full-sequence evaluation. EasyMocap/SMPL is technically prepared but was not executed because the licensed SMPL asset is not yet available.</p><div class="cards"><div class="card"><div class="metric">3 rejected</div>Pose2Sim, MeTRAbs, kinematic prior</div><div class="card"><div class="metric">1 pending</div>EasyMocap/SMPL asset</div><div class="card"><div class="metric">Retained</div>Deterministic YOLOv8m + SKT</div></div><div class="callout">Xsens is treated only as an Xsens-derived reference / external comparison system. No candidate was calibrated from an Xsens initial frame or native angle.</div></section>
<section id="research"><h2>2. Research and repository screening</h2><table><tr><th>Track</th><th>Representative method</th><th>Project relevance</th><th>Decision</th></tr><tr><td>Kinematic IK</td><td><a href="https://github.com/perfanalytics/pose2sim">Pose2Sim/OpenSim</a></td><td>Supports two cameras and directly produces joint angles</td><td class="bad">Tested, rejected</td></tr><tr><td>Parametric body</td><td><a href="https://github.com/zju3dv/EasyMocap">EasyMocap/SMPL</a></td><td>Shape, joint-range, and temporal priors for the offline ceiling</td><td class="pending">Awaiting asset</td></tr><tr><td>Direct metric 3D</td><td><a href="https://github.com/isarandi/metrabs">MeTRAbs</a></td><td>World-space predictions with calibration and plausibility filtering</td><td class="bad">Tested, rejected</td></tr><tr><td>End-to-end multiview</td><td><a href="https://github.com/XunshanMan/MVGFormer">MVGFormer</a></td><td>CVPR 2024; official pretrained configurations require 3–7 views</td><td>Not applicable to strict stereo</td></tr><tr><td>Occlusion fusion</td><td><a href="https://arxiv.org/abs/2408.15810">Multi-view Pose Fusion</a></td><td>Human–robot collaboration focus; insufficient released stereo path</td><td>Future research</td></tr><tr><td>Dense stereo prior</td><td><a href="https://github.com/NVlabs/FoundationStereo">FoundationStereo</a></td><td>Strong zero-shot depth, but no joint semantics</td><td>Depth auxiliary only</td></tr><tr><td>Monocular mesh</td><td><a href="https://github.com/naver/multi-hmr">Multi-HMR</a>, <a href="https://github.com/yufu-wang/PromptHMR">PromptHMR</a></td><td>Strong shape priors but require an added cross-view consistency layer</td><td>Not a direct replacement</td></tr></table></section>
<section id="pose2sim"><h2>3. Pose2Sim / OpenSim gate</h2><table><tr><th>Dataset</th><th>System</th><th>RightElbow MAE</th><th>RULA agreement</th><th>Jumps</th><th>Prior stage</th></tr><tr><td>Fanbo7</td><td>SKT</td><td>10.47°</td><td>0.939</td><td>0</td><td>—</td></tr><tr><td>Fanbo7</td><td>Pose2Sim</td><td>52.36°</td><td>0.344</td><td>0</td><td>102.6 ms/frame</td></tr><tr><td>Fanbo4</td><td>SKT</td><td>15.47°</td><td>0.948</td><td>7</td><td>—</td></tr><tr><td>Fanbo4</td><td>Pose2Sim</td><td>24.98°</td><td>0.895</td><td>19</td><td>101.4 ms/frame</td></tr></table><figure><img src="{p2s}"><figcaption>Figure 1. IK produced anatomically feasible solutions that did not preserve the observed elbow motion.</figcaption></figure><p class="bad">Rejected on both angle accuracy and real-time criteria.</p></section>
<section id="kinematic"><h2>4. Geometry-conditioned kinematic prior</h2><p>The fixed eight-setting grid combined calibrated two-view Huber reprojection, a quality-weighted SKT anchor, fixed bone lengths, and temporal second-order continuity on 40 continuous Fanbo7 A257 frames. Selection used geometry and temporal stability only.</p><table><tr><th>Metric</th><th>Raw SKT</th><th>Kinematic prior</th></tr><tr><td>Finite core-joint ratio</td><td>0.8375</td><td>0.9167</td></tr><tr><td>Reprojection p50 / p95</td><td>1.366 / 7.788 px</td><td>1.611 / 13.545 px</td></tr><tr><td>Bone-length CV</td><td>0.1116</td><td>0.0971</td></tr><tr><td>Angle jumps above 10°</td><td>53</td><td>42</td></tr><tr><td>High-quality correction median / p95</td><td>—</td><td>0.206 / 0.631 cm</td></tr><tr><td>Prior time / estimated end-to-end</td><td>—</td><td>5028.9 ms/frame / 0.198 fps</td></tr></table><p class="bad">Rejected because reprojection p95 exceeded the predefined 10 px gate and throughput was far below 12.5 fps. The adapter was reverted; the checksummed negative-result bundle remains available locally.</p></section>
<section id="metrabs"><h2>5. MeTRAbs calibrated stereo gate</h2><table><tr><th>Dataset</th><th>End-to-end fps</th><th>Inference ms/pair</th><th>Right-arm stereo 3D median/p95</th><th>2D disagreement median/p95</th></tr><tr><td>Fanbo7</td><td>4.76</td><td>86.8</td><td>69.45 / 75.68 cm</td><td>564.8 / 735.1 px</td></tr><tr><td>Fanbo4</td><td>4.96</td><td>82.1</td><td>66.38 / 109.02 cm</td><td>854.6 / 1028.0 px</td></tr></table><figure><img src="{met}"><figcaption>Figure 2. MeTRAbs-S failed both throughput and cross-view geometry gates.</figcaption></figure><p class="bad">The generic monocular metric-3D prior did not yield a consistent stereo world-space skeleton. The larger model and additional TTA were therefore not run.</p></section>
<section id="smpl"><h2>6. EasyMocap / SMPL status</h2><p>The EasyMocap CUDA SMPL module imports successfully under Torch 2.8 / CUDA 12.8. Reproducible setup, private-asset ignore rules, and a CUDA validator are ready. Formal fitting has not run because <code>SMPL_NEUTRAL.pkl</code> is license-controlled and absent.</p><p class="pending">This is an external asset gate, not an accuracy failure. Follow <code>docs/smpl_asset_setup.md</code> and obtain the model only from the official SMPL website.</p></section>
<section id="conclusion"><h2>7. Conclusions and next steps</h2><figure><img src="{pareto}"><figcaption>Figure 3. Accuracy-throughput Pareto view for comparable candidates. Pose2Sim throughput is an optimistic upper bound derived from IK-stage time alone; it still remains dominated by the SKT baseline. MeTRAbs and the lightweight kinematic prior have no valid accuracy points because they failed geometry gates.</figcaption></figure><ol><li>Retain deterministic PyTorch FP32 YOLOv8m + SKT as the formal baseline.</li><li>After official SMPL upload, run only the 40-frame Fanbo4/Fanbo7 reprojection and semantic gate before any longer job.</li><li>If SMPL also fails, do not repeat the rejected kinematic grid; prioritize stereo 2D semantic consistency under occlusion.</li><li>Use FoundationStereo only as an occlusion/depth diagnostic, not as a joint-semantic replacement.</li><li>Select parameters from geometry and temporal stability; use Xsens only for final agreement reporting.</li></ol></section>"""
    links = "".join(f'<a href="#{target}">{label}</a>' for target, label in zip(["summary", "research", "pose2sim", "kinematic", "metrabs", "smpl", "conclusion"], nav))
    return f'<!doctype html><html lang="{"zh-CN" if chinese else "en"}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{STYLE}</style></head><body><header><h1>{title}</h1><p>{subtitle}</p></header><div class="layout"><nav><strong>{"目录" if chinese else "Contents"}</strong>{links}</nav><main>{body}<footer>{"本报告为自包含 HTML；图表已内嵌。" if chinese else "Self-contained HTML report with embedded figures."}</footer></main></div></body></html>'


def main() -> None:
    """Write both required language versions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("00_pose_pipeline_v2/runs/human_prior_eval"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report_CN.html").write_text(build_report(True), encoding="utf-8")
    (args.output_dir / "report.html").write_text(build_report(False), encoding="utf-8")
    print((args.output_dir / "report_CN.html").resolve())
    print((args.output_dir / "report.html").resolve())


if __name__ == "__main__":
    main()
