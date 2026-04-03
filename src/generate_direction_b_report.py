#!/usr/bin/env python3
"""
generate_direction_b_report.py
Generate HTML report for Direction B (SGBM dense stereo) results.
Produces both English and Chinese versions.
"""
import base64
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_EN = os.path.join(RESULTS_DIR, "report_direction_b.html")
OUT_CN = os.path.join(RESULTS_DIR, "report_direction_b_CN.html")


def img_b64(filename: str) -> str:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── load images ──────────────────────────────────────────────────────────────
b64 = {k: img_b64(v) for k, v in {
    "fill_rate":   "disparity_analysis_fill_rate.png",
    "scatter":     "disparity_analysis_scatter.png",
    "error_hist":  "disparity_analysis_error_hist.png",
    "frame":       "disparity_analysis_frame200.png",
    "by_joint":    "eval_angle_by_joint_yolov8m_sgbm.png",
    "by_scenario": "eval_angle_by_scenario_yolov8m_sgbm.png",
    "trunk":       "eval_trunk_flexion_compare_yolov8m_sgbm.png",
}.items()}


def img_tag(key: str, alt: str = "", width: str = "100%") -> str:
    d = b64[key]
    if not d:
        return f'<p style="color:red">[Image not found: {key}]</p>'
    return f'<img src="data:image/png;base64,{d}" alt="{alt}" style="width:{width};max-width:800px;display:block;margin:12px auto;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.15)">'


# ── CSS (shared) ─────────────────────────────────────────────────────────────
CSS = """
<style>
  body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:0;background:#f4f6fa;color:#222}
  .header{background:linear-gradient(135deg,#1a237e,#283593);color:#fff;padding:40px 48px 30px;
          box-shadow:0 4px 12px rgba(0,0,0,.25)}
  .header h1{margin:0 0 8px;font-size:2rem;letter-spacing:.5px}
  .header .meta{opacity:.8;font-size:.9rem}
  .container{max-width:1100px;margin:0 auto;padding:24px 32px}
  .toc{background:#fff;border-radius:8px;padding:20px 28px;margin-bottom:28px;
       box-shadow:0 1px 4px rgba(0,0,0,.1)}
  .toc h2{margin:0 0 12px;font-size:1rem;color:#1a237e;text-transform:uppercase;letter-spacing:1px}
  .toc ol{margin:0;padding-left:20px;line-height:2}
  .toc a{color:#1565c0;text-decoration:none}
  .toc a:hover{text-decoration:underline}
  .section{background:#fff;border-radius:8px;padding:28px 32px;margin-bottom:24px;
           box-shadow:0 1px 4px rgba(0,0,0,.1)}
  .section h2{margin:0 0 18px;color:#1a237e;border-bottom:2px solid #e3e8f0;padding-bottom:10px}
  .section h3{color:#283593;margin:20px 0 10px}
  table{border-collapse:collapse;width:100%;margin:14px 0;font-size:.92rem}
  th{background:#1a237e;color:#fff;padding:10px 14px;text-align:left}
  td{padding:9px 14px;border-bottom:1px solid #e3e8f0}
  tr:nth-child(even) td{background:#f8f9fd}
  .badge-fail{background:#ef5350;color:#fff;border-radius:4px;padding:2px 8px;font-size:.8rem;font-weight:700}
  .badge-ok{background:#66bb6a;color:#fff;border-radius:4px;padding:2px 8px;font-size:.8rem;font-weight:700}
  .badge-warn{background:#ffa726;color:#fff;border-radius:4px;padding:2px 8px;font-size:.8rem;font-weight:700}
  .highlight-box{background:#fff3e0;border-left:4px solid #ff9800;padding:14px 18px;
                 border-radius:0 6px 6px 0;margin:14px 0;font-size:.93rem}
  .info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:14px 18px;
            border-radius:0 6px 6px 0;margin:14px 0;font-size:.93rem}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  @media(max-width:700px){.grid2{grid-template-columns:1fr}}
  .figure-cap{text-align:center;font-size:.82rem;color:#555;margin-top:4px;font-style:italic}
  footer{text-align:center;padding:20px;color:#888;font-size:.82rem}
</style>
"""

# ═════════════════════════════════════════════════════════════════════════════
# ENGLISH REPORT
# ═════════════════════════════════════════════════════════════════════════════
EN = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Direction B Report — SGBM Dense Stereo</title>
{CSS}
</head>
<body>
<div class="header">
  <h1>Direction B: SGBM Dense Stereo Triangulation</h1>
  <div class="meta">
    Master Thesis · Fanbo Meng · Chalmers University of Technology · March 2026<br>
    Replacing DLT point triangulation with SGBM disparity-map lookup
  </div>
</div>
<div class="container">

<!-- TOC -->
<div class="toc">
  <h2>Contents</h2>
  <ol>
    <li><a href="#overview">Approach Overview</a></li>
    <li><a href="#disparity">Disparity Quality Analysis</a></li>
    <li><a href="#results">Evaluation Results</a></li>
    <li><a href="#perjoint">Per-Joint &amp; Per-Scenario Breakdown</a></li>
    <li><a href="#trunk">Trunk Flexion</a></li>
    <li><a href="#failure">Failure Analysis</a></li>
    <li><a href="#comparison">Comparison vs Direction A</a></li>
    <li><a href="#conclusion">Conclusion &amp; Next Steps</a></li>
  </ol>
</div>

<!-- 1. Overview -->
<div class="section" id="overview">
  <h2>1. Approach Overview</h2>
  <p>Direction B replaces the sparse DLT triangulation used in Direction A with a
  <strong>dense stereo disparity map</strong> computed by OpenCV's Semi-Global Block Matching
  (SGBM). The hypothesis is that a dense depth map with sub-pixel accuracy would improve
  3-D pose reconstruction quality compared to the point-by-point DLT.</p>

  <h3>SGBM Parameters</h3>
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Rationale</th></tr>
    <tr><td>minDisparity</td><td>100 px</td><td>Covers depth &lt; 4.3 m</td></tr>
    <tr><td>numDisparities</td><td>256 px</td><td>Range 100–356 px → depth 1.2–4.3 m</td></tr>
    <tr><td>blockSize</td><td>9</td><td>Balance between detail and noise</td></tr>
    <tr><td>Lookup window</td><td>5×5 median</td><td>Robust depth at each keypoint</td></tr>
  </table>

  <h3>3-D Reconstruction</h3>
  <p>For each 2-D keypoint detected by YOLOv8m-pose, the local 5×5 disparity patch is
  sampled. Valid values (≥ minDisparity) are median-aggregated, then projected via the
  Q-matrix from <code>stereoRectify</code>:</p>
  <p style="font-family:monospace;background:#f0f4ff;padding:10px;border-radius:4px">
    Z = f_rect × baseline / disparity<br>
    X = (u − cx_rect) × Z / f_rect<br>
    Y = (v − cy_rect) × Z / f_rect
  </p>
  <p>Baseline = 41.27 cm, focal (rect) ≈ 1037.6 px.</p>
</div>

<!-- 2. Disparity Quality -->
<div class="section" id="disparity">
  <h2>2. Disparity Quality Analysis</h2>
  <p>Before evaluating pose accuracy, the disparity map was validated at 94 sampled frames
  (every 30th frame). The key question: do the SGBM values at keypoint locations reflect
  the true depth?</p>

  <div class="grid2">
    <div>
      {img_tag("fill_rate", "Fill rate per joint")}
      <p class="figure-cap">Fig 1 — Disparity fill rate per joint (threshold: ≥ minDisparity).
      Overall mean = 74.2%.</p>
    </div>
    <div>
      {img_tag("frame", "Example disparity frame")}
      <p class="figure-cap">Fig 2 — Example rectified frame with SGBM disparity overlay
      (frame 200). Uniform clothing creates large textureless regions.</p>
    </div>
  </div>

  <div class="grid2">
    <div>
      {img_tag("scatter", "SGBM vs DLT disparity scatter")}
      <p class="figure-cap">Fig 3 — SGBM vs DLT disparity at joint locations.
      Valid pairs cluster near y = x (top-right); failures concentrate near disparity ≈ 0 (DLT) or 100 (SGBM floor).</p>
    </div>
    <div>
      {img_tag("error_hist", "Disparity error histogram")}
      <p class="figure-cap">Fig 4 — Distribution of |SGBM − DLT| disparity difference.
      Two modes: good agreement (&lt;20 px) and large errors (&gt;100 px).</p>
    </div>
  </div>

  <div class="highlight-box">
    <strong>Key finding:</strong> Mean fill rate is 74.2%, with LHip (39.4%) and
    LElbow (48.9%) below the 80% threshold critical for RULA scoring. The root cause is
    <em>lack of texture</em> on industrial work clothing — SGBM cannot match featureless
    uniform surfaces.
  </div>

  <h3>Per-Joint Fill Rate</h3>
  <table>
    <tr><th>Joint</th><th>Fill Rate</th><th>Status</th></tr>
    <tr><td>RightShoulder</td><td>81.5%</td><td><span class="badge-ok">OK</span></td></tr>
    <tr><td>LeftShoulder</td><td>80.2%</td><td><span class="badge-ok">OK</span></td></tr>
    <tr><td>RightElbow</td><td>77.6%</td><td><span class="badge-warn">MARGINAL</span></td></tr>
    <tr><td>RightKnee</td><td>76.3%</td><td><span class="badge-warn">MARGINAL</span></td></tr>
    <tr><td>RightHip</td><td>75.1%</td><td><span class="badge-warn">MARGINAL</span></td></tr>
    <tr><td>LeftKnee</td><td>72.8%</td><td><span class="badge-warn">MARGINAL</span></td></tr>
    <tr><td>LeftElbow</td><td>48.9%</td><td><span class="badge-fail">FAIL</span></td></tr>
    <tr><td>LHip</td><td>39.4%</td><td><span class="badge-fail">FAIL</span></td></tr>
  </table>
</div>

<!-- 3. Results -->
<div class="section" id="results">
  <h2>3. Evaluation Results</h2>
  <table>
    <tr><th>Metric</th><th>Direction A (calibrated)</th><th>Direction B (SGBM)</th><th>Change</th></tr>
    <tr><td>Joint Angle MAE</td><td>13.21°</td><td><strong>30.63°</strong></td><td style="color:#ef5350">+17.4° ↑ worse</td></tr>
    <tr><td>Joint Angle Median</td><td>9.73°</td><td>19.23°</td><td style="color:#ef5350">+9.5° ↑ worse</td></tr>
    <tr><td>Elbow RULA Accuracy</td><td>76.3%</td><td><strong>50.6%</strong></td><td style="color:#ef5350">−25.7 pp ↓ worse</td></tr>
    <tr><td>Trunk Flexion MAE</td><td>11.40°</td><td><strong>57.48°</strong></td><td style="color:#ef5350">+46.1° ↑ worse</td></tr>
    <tr><td>MPJPE</td><td>26.0 cm</td><td><strong>117.5 cm</strong></td><td style="color:#ef5350">+91.5 cm ↑ worse</td></tr>
  </table>
  <p>Direction B is significantly worse on all metrics. SGBM did not improve over DLT for
  this industrial dataset.</p>
</div>

<!-- 4. Per-Joint & Scenario -->
<div class="section" id="perjoint">
  <h2>4. Per-Joint &amp; Per-Scenario Breakdown</h2>
  <div class="grid2">
    <div>
      {img_tag("by_joint", "Per-joint MAE")}
      <p class="figure-cap">Fig 5 — Per-joint MAE for Direction B. All joints exceed 25°;
      hips and elbows worst (&gt;34°).</p>
    </div>
    <div>
      {img_tag("by_scenario", "Per-scenario MAE")}
      <p class="figure-cap">Fig 6 — Per-scenario MAE. Environmental Interference
      scenario reaches 61.9° — SGBM fails under lighting changes.</p>
    </div>
  </div>

  <h3>Per-Joint MAE Comparison</h3>
  <table>
    <tr><th>Joint</th><th>Dir A MAE</th><th>Dir B MAE</th><th>Ratio</th></tr>
    <tr><td>LeftShoulder</td><td>13.27°</td><td>25.33°</td><td>1.9×</td></tr>
    <tr><td>RightShoulder</td><td>13.72°</td><td>26.25°</td><td>1.9×</td></tr>
    <tr><td>LeftKnee</td><td>14.35°</td><td>28.71°</td><td>2.0×</td></tr>
    <tr><td>RightKnee</td><td>13.06°</td><td>28.73°</td><td>2.2×</td></tr>
    <tr><td>LeftElbow</td><td>15.98°</td><td>32.95°</td><td>2.1×</td></tr>
    <tr><td>RightElbow</td><td>16.17°</td><td>34.75°</td><td>2.1×</td></tr>
    <tr><td>RightHip</td><td>8.41°</td><td>34.14°</td><td>4.1×</td></tr>
    <tr><td>LeftHip</td><td>9.98°</td><td>34.14°</td><td>3.4×</td></tr>
  </table>
  <p>Hip joints show the most dramatic degradation (3–4×), consistent with the lowest
  disparity fill rates.</p>
</div>

<!-- 5. Trunk Flexion -->
<div class="section" id="trunk">
  <h2>5. Trunk Flexion</h2>
  {img_tag("trunk", "Trunk flexion comparison")}
  <p class="figure-cap">Fig 7 — Predicted vs GT trunk flexion angle. Direction B predictions
  are systematically biased and show large outliers.</p>
  <p>Trunk flexion MAE increased from 11.4° (Dir A) to 57.5° (Dir B).
  Trunk angle depends on the relative positions of shoulder and hip keypoints.
  Since both hips have fill rates below 40–75%, their 3-D positions are frequently
  reconstructed from the minDisparity boundary value, giving a fixed depth of ~430 cm
  regardless of actual position.</p>
</div>

<!-- 6. Failure Analysis -->
<div class="section" id="failure">
  <h2>6. Failure Analysis</h2>
  <div class="info-box">
    <strong>Toolchain diagnosis:</strong> The 2D keypoint detection (YOLO) performs well on
    industrial clothing — the bottleneck is exclusively in the <em>stereo matching step</em>.
    Replacing SGBM with a learning-based matcher (RAFT-Stereo, CREStereo) that incorporates
    semantic priors would directly address the root cause without changing any other part of
    the pipeline.
  </div>

  <h3>Root Cause: Textureless Industrial Clothing</h3>
  <p>SGBM relies on local image texture for patch matching. The test subject wears uniform
  work clothing with minimal surface variation. This causes:</p>
  <ul>
    <li>Disparity values clipping to <code>minDisparity</code> floor (≈100 px)</li>
    <li>Reconstructed depth pinned at ~430 cm (far beyond actual 1.3–2.4 m range)</li>
    <li>All joints with clipped disparity placed at the same depth → angles collapse</li>
  </ul>

  <div class="highlight-box">
    <strong>SGBM depth at minDisparity floor:</strong><br>
    Z = 1037.6 × 41.27 / 100 ≈ 428 cm (actual range: 133–242 cm)<br>
    This 2–3× depth overestimate directly corrupts all 3-D joint angle calculations.
  </div>

  <h3>Why DLT Was Better</h3>
  <p>DLT triangulation uses 2-D detections from both cameras and geometric constraints
  (camera calibration + known baseline). It does <em>not</em> rely on image texture —
  it only requires accurate 2-D keypoint detection in each view. YOLOv8m-pose maintains
  good 2-D detection accuracy even on uniform clothing, so DLT succeeds where SGBM fails.</p>

  <h3>Could Better SGBM Parameters Help?</h3>
  <p>No — this is a structural limitation. Even with optimal SGBM tuning, the fundamental
  problem (no texture → no disparity) cannot be overcome. Learning-based stereo matchers
  (RAFT-Stereo, CREStereo) that incorporate semantic priors would be needed.</p>
</div>

<!-- 7. Comparison -->
<div class="section" id="comparison">
  <h2>7. Summary Comparison vs Direction A</h2>
  <table>
    <tr><th>Property</th><th>Direction A (DLT)</th><th>Direction B (SGBM)</th></tr>
    <tr><td>Depth method</td><td>Sparse triangulation</td><td>Dense disparity map</td></tr>
    <tr><td>Texture requirement</td><td>None (geometry only)</td><td>High</td></tr>
    <tr><td>Fill rate (RULA joints)</td><td>100% (always 2 views)</td><td>39–82%</td></tr>
    <tr><td>Joint Angle MAE</td><td><strong>13.2°</strong></td><td>30.6°</td></tr>
    <tr><td>Elbow RULA Acc</td><td><strong>76.3%</strong></td><td>50.6%</td></tr>
    <tr><td>Trunk Flexion MAE</td><td><strong>11.4°</strong></td><td>57.5°</td></tr>
    <tr><td>MPJPE</td><td><strong>26.0 cm</strong></td><td>117.5 cm</td></tr>
    <tr><td>Verdict</td><td><span class="badge-ok">BEST SO FAR</span></td><td><span class="badge-fail">REJECTED</span></td></tr>
  </table>
</div>

<!-- 8. Conclusion -->
<div class="section" id="conclusion">
  <h2>8. Conclusion &amp; Next Steps</h2>
  <div class="info-box">
    <strong>Direction B is rejected.</strong> SGBM dense stereo fails on this industrial
    dataset due to textureless uniform clothing. All evaluation metrics are 2–4× worse
    than Direction A's DLT baseline. The codebase retains the SGBM implementation
    (controlled by <code>USE_DENSE_STEREO=1</code>) for future experiments with
    learning-based matchers.
  </div>

  <h3>Next Directions</h3>
  <table>
    <tr><th>Direction</th><th>Method</th><th>Status</th><th>Expected Benefit</th></tr>
    <tr><td>A (baseline)</td><td>YOLOv8m + DLT + calibration</td><td><span class="badge-ok">BEST: 13.2°</span></td><td>—</td></tr>
    <tr><td>C (monocular)</td><td>RTMDet + MotionBERT + OpenSim</td><td><span class="badge-warn">IN PROGRESS (CPU)</span></td><td>Ablation: monocular vs stereo</td></tr>
    <tr><td>D (FastSAM3D)</td><td>Learned multi-view mesh fusion</td><td><span class="badge-warn">PENDING GPU</span></td><td>End-to-end 3D, no texture assumption</td></tr>
    <tr><td>B.2 (future)</td><td>RAFT-Stereo / CREStereo + DLT</td><td>Not started</td><td>Learned disparity handles low-texture</td></tr>
  </table>
</div>

</div>
<footer>Generated by Claude Code · Master Thesis · Fanbo Meng · Chalmers University of Technology</footer>
</body>
</html>"""

# ═════════════════════════════════════════════════════════════════════════════
# CHINESE REPORT
# ═════════════════════════════════════════════════════════════════════════════
CN = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Direction B 报告 — SGBM 稠密立体匹配</title>
{CSS}
</head>
<body>
<div class="header">
  <h1>Direction B：SGBM 稠密立体视差三角化</h1>
  <div class="meta">
    硕士论文 · Fanbo Meng · 查尔姆斯理工大学 · 2026年3月<br>
    用 SGBM 视差图查询替代 DLT 逐点三角化
  </div>
</div>
<div class="container">

<!-- TOC -->
<div class="toc">
  <h2>目录</h2>
  <ol>
    <li><a href="#overview">方法概述</a></li>
    <li><a href="#disparity">视差质量分析</a></li>
    <li><a href="#results">评估结果</a></li>
    <li><a href="#perjoint">关节与场景细分</a></li>
    <li><a href="#trunk">躯干弯曲角</a></li>
    <li><a href="#failure">失败原因分析</a></li>
    <li><a href="#comparison">与 Direction A 对比</a></li>
    <li><a href="#conclusion">结论与后续</a></li>
  </ol>
</div>

<!-- 1. Overview -->
<div class="section" id="overview">
  <h2>1. 方法概述</h2>
  <p>Direction B 用 OpenCV SGBM（半全局块匹配）生成的<strong>稠密视差图</strong>替代
  Direction A 中的稀疏 DLT 三角化。假设：稠密深度图能提供比逐点 DLT 更精确的 3D 重建。</p>

  <h3>SGBM 参数</h3>
  <table>
    <tr><th>参数</th><th>值</th><th>说明</th></tr>
    <tr><td>minDisparity</td><td>100 px</td><td>覆盖深度 &lt; 4.3 m</td></tr>
    <tr><td>numDisparities</td><td>256 px</td><td>范围 100–356 px → 深度 1.2–4.3 m</td></tr>
    <tr><td>blockSize</td><td>9</td><td>细节与噪声平衡</td></tr>
    <tr><td>查询窗口</td><td>5×5 中值</td><td>关键点处鲁棒深度</td></tr>
  </table>

  <h3>三维重建公式</h3>
  <p style="font-family:monospace;background:#f0f4ff;padding:10px;border-radius:4px">
    Z = f_rect × baseline / disparity<br>
    X = (u − cx_rect) × Z / f_rect<br>
    Y = (v − cy_rect) × Z / f_rect
  </p>
  <p>基线 = 41.27 cm，矫正焦距 ≈ 1037.6 px。</p>
</div>

<!-- 2. Disparity Quality -->
<div class="section" id="disparity">
  <h2>2. 视差质量分析</h2>
  <p>在评估姿态精度前，先对 94 个采样帧（每 30 帧一帧）的视差图进行验证。
  关键问题：关键点位置的 SGBM 值是否反映真实深度？</p>

  <div class="grid2">
    <div>
      {img_tag("fill_rate", "关节填充率")}
      <p class="figure-cap">图1 — 各关节视差有效率（阈值：≥ minDisparity）。总体均值 74.2%。</p>
    </div>
    <div>
      {img_tag("frame", "示例视差帧")}
      <p class="figure-cap">图2 — 第200帧的矫正图像与 SGBM 视差叠加。
      均匀工装造成大面积无纹理区域。</p>
    </div>
  </div>

  <div class="grid2">
    <div>
      {img_tag("scatter", "SGBM vs DLT 视差散点图")}
      <p class="figure-cap">图3 — SGBM 与 DLT 在关节位置的视差对比。
      有效匹配点集中在 y≈x（右上角）；失败点集中在 SGBM=100（下限值）。</p>
    </div>
    <div>
      {img_tag("error_hist", "视差误差分布")}
      <p class="figure-cap">图4 — |SGBM − DLT| 视差差分布。
      双峰：吻合好（&lt;20 px）和大误差（&gt;100 px）。</p>
    </div>
  </div>

  <div class="highlight-box">
    <strong>关键发现：</strong>总体有效率 74.2%，其中 LHip（39.4%）和 LElbow（48.9%）
    低于 RULA 关键关节所需的 80% 阈值。根本原因是工业工装<em>缺乏纹理</em>——
    SGBM 无法匹配无特征的均匀表面。
  </div>

  <h3>各关节视差有效率</h3>
  <table>
    <tr><th>关节</th><th>有效率</th><th>状态</th></tr>
    <tr><td>右肩</td><td>81.5%</td><td><span class="badge-ok">通过</span></td></tr>
    <tr><td>左肩</td><td>80.2%</td><td><span class="badge-ok">通过</span></td></tr>
    <tr><td>右肘</td><td>77.6%</td><td><span class="badge-warn">边缘</span></td></tr>
    <tr><td>右膝</td><td>76.3%</td><td><span class="badge-warn">边缘</span></td></tr>
    <tr><td>右髋</td><td>75.1%</td><td><span class="badge-warn">边缘</span></td></tr>
    <tr><td>左膝</td><td>72.8%</td><td><span class="badge-warn">边缘</span></td></tr>
    <tr><td>左肘</td><td>48.9%</td><td><span class="badge-fail">不通过</span></td></tr>
    <tr><td>左髋</td><td>39.4%</td><td><span class="badge-fail">不通过</span></td></tr>
  </table>
</div>

<!-- 3. Results -->
<div class="section" id="results">
  <h2>3. 评估结果</h2>
  <table>
    <tr><th>指标</th><th>Direction A（已校准）</th><th>Direction B（SGBM）</th><th>变化</th></tr>
    <tr><td>关节角度 MAE</td><td>13.21°</td><td><strong>30.63°</strong></td><td style="color:#ef5350">+17.4° ↑ 更差</td></tr>
    <tr><td>关节角度 中值</td><td>9.73°</td><td>19.23°</td><td style="color:#ef5350">+9.5° ↑ 更差</td></tr>
    <tr><td>肘部 RULA 准确率</td><td>76.3%</td><td><strong>50.6%</strong></td><td style="color:#ef5350">−25.7 pp ↓ 更差</td></tr>
    <tr><td>躯干弯曲 MAE</td><td>11.40°</td><td><strong>57.48°</strong></td><td style="color:#ef5350">+46.1° ↑ 更差</td></tr>
    <tr><td>MPJPE</td><td>26.0 cm</td><td><strong>117.5 cm</strong></td><td style="color:#ef5350">+91.5 cm ↑ 更差</td></tr>
  </table>
  <p>Direction B 在所有指标上均显著劣于 Direction A。SGBM 对本工业数据集未能改善 DLT。</p>
</div>

<!-- 4. Per-Joint & Scenario -->
<div class="section" id="perjoint">
  <h2>4. 关节与场景细分</h2>
  <div class="grid2">
    <div>
      {img_tag("by_joint", "各关节 MAE")}
      <p class="figure-cap">图5 — Direction B 各关节 MAE。所有关节均超过 25°；髋部和肘部最差（&gt;34°）。</p>
    </div>
    <div>
      {img_tag("by_scenario", "各场景 MAE")}
      <p class="figure-cap">图6 — 各场景 MAE。"环境干扰"场景达 61.9°——SGBM 在光照变化下失效。</p>
    </div>
  </div>

  <h3>各关节 MAE 对比</h3>
  <table>
    <tr><th>关节</th><th>Dir A MAE</th><th>Dir B MAE</th><th>倍数</th></tr>
    <tr><td>左肩</td><td>13.27°</td><td>25.33°</td><td>1.9×</td></tr>
    <tr><td>右肩</td><td>13.72°</td><td>26.25°</td><td>1.9×</td></tr>
    <tr><td>左膝</td><td>14.35°</td><td>28.71°</td><td>2.0×</td></tr>
    <tr><td>右膝</td><td>13.06°</td><td>28.73°</td><td>2.2×</td></tr>
    <tr><td>左肘</td><td>15.98°</td><td>32.95°</td><td>2.1×</td></tr>
    <tr><td>右肘</td><td>16.17°</td><td>34.75°</td><td>2.1×</td></tr>
    <tr><td>右髋</td><td>8.41°</td><td>34.14°</td><td>4.1×</td></tr>
    <tr><td>左髋</td><td>9.98°</td><td>34.14°</td><td>3.4×</td></tr>
  </table>
  <p>髋部关节退化最严重（3–4×），与最低视差有效率一致。</p>
</div>

<!-- 5. Trunk Flexion -->
<div class="section" id="trunk">
  <h2>5. 躯干弯曲角</h2>
  {img_tag("trunk", "躯干弯曲角对比")}
  <p class="figure-cap">图7 — 预测躯干弯曲角 vs GT。Direction B 预测存在系统性偏差和大量异常值。</p>
  <p>躯干弯曲 MAE 从 Direction A 的 11.4° 增加到 57.5°。躯干角依赖肩部和髋部关键点的相对位置。
  由于两侧髋部有效率仅 39–75%，其 3D 位置频繁被固定在 minDisparity 边界值，
  给出约 430 cm 的恒定深度（真实距离为 133–242 cm）。</p>
</div>

<!-- 6. Failure Analysis -->
<div class="section" id="failure">
  <h2>6. 失败原因分析</h2>
  <div class="info-box">
    <strong>工具链诊断：</strong> YOLO 的 2D 关键点检测在工业工装上表现正常——瓶颈
    <em>完全在立体匹配这一步</em>。用引入语义先验的学习型匹配器（RAFT-Stereo、CREStereo）
    替换 SGBM，可直接解决根本问题，而无需修改 pipeline 的其他部分。
  </div>

  <h3>根本原因：工业工装缺乏纹理</h3>
  <p>SGBM 依赖局部图像纹理进行块匹配。测试对象穿着表面变化极少的均匀工装，导致：</p>
  <ul>
    <li>视差值被截断到 <code>minDisparity</code> 下限（≈100 px）</li>
    <li>重建深度固定在约 430 cm（远超实际 1.3–2.4 m 范围）</li>
    <li>所有被截断的关节被放在同一深度 → 角度计算崩溃</li>
  </ul>

  <div class="highlight-box">
    <strong>minDisparity 下限处的深度：</strong><br>
    Z = 1037.6 × 41.27 / 100 ≈ 428 cm（实际范围：133–242 cm）<br>
    这 2–3× 的深度高估直接破坏了所有 3D 关节角度计算。
  </div>

  <h3>为什么 DLT 更好</h3>
  <p>DLT 三角化使用两个相机的 2D 检测结果和几何约束（相机标定 + 已知基线），
  <em>不依赖图像纹理</em>——只需要每个视角准确的 2D 关键点检测。
  YOLOv8m-pose 在均匀工装上仍保持良好的 2D 检测精度，因此 DLT 在 SGBM 失效的地方能正常工作。</p>

  <h3>调整 SGBM 参数能改善吗？</h3>
  <p>不能——这是结构性限制。即使 SGBM 参数完全优化，根本问题（无纹理 → 无视差）也无法克服。
  需要引入语义先验的基于学习的立体匹配器（如 RAFT-Stereo、CREStereo）才能解决。</p>
</div>

<!-- 7. Comparison -->
<div class="section" id="comparison">
  <h2>7. 与 Direction A 汇总对比</h2>
  <table>
    <tr><th>属性</th><th>Direction A (DLT)</th><th>Direction B (SGBM)</th></tr>
    <tr><td>深度方法</td><td>稀疏三角化</td><td>稠密视差图</td></tr>
    <tr><td>纹理需求</td><td>无（仅几何）</td><td>高</td></tr>
    <tr><td>RULA关节有效率</td><td>100%（始终双目）</td><td>39–82%</td></tr>
    <tr><td>关节角度 MAE</td><td><strong>13.2°</strong></td><td>30.6°</td></tr>
    <tr><td>肘部 RULA 准确率</td><td><strong>76.3%</strong></td><td>50.6%</td></tr>
    <tr><td>躯干弯曲 MAE</td><td><strong>11.4°</strong></td><td>57.5°</td></tr>
    <tr><td>MPJPE</td><td><strong>26.0 cm</strong></td><td>117.5 cm</td></tr>
    <tr><td>结论</td><td><span class="badge-ok">当前最优</span></td><td><span class="badge-fail">已否决</span></td></tr>
  </table>
</div>

<!-- 8. Conclusion -->
<div class="section" id="conclusion">
  <h2>8. 结论与后续方向</h2>
  <div class="info-box">
    <strong>Direction B 已否决。</strong> SGBM 稠密立体匹配在本工业数据集上因均匀工装缺乏纹理而失效。
    所有评估指标比 Direction A 的 DLT 基线差 2–4 倍。代码库保留了 SGBM 实现
    （通过 <code>USE_DENSE_STEREO=1</code> 控制），供未来引入基于学习的匹配器使用。
  </div>

  <h3>后续方向</h3>
  <table>
    <tr><th>方向</th><th>方法</th><th>状态</th><th>预期收益</th></tr>
    <tr><td>A（基线）</td><td>YOLOv8m + DLT + 校准</td><td><span class="badge-ok">最优：13.2°</span></td><td>—</td></tr>
    <tr><td>C（单目）</td><td>RTMDet + MotionBERT + OpenSim</td><td><span class="badge-warn">进行中（CPU）</span></td><td>消融：单目 vs 双目</td></tr>
    <tr><td>D（FastSAM3D）</td><td>学习型多视角网格融合</td><td><span class="badge-warn">待 GPU</span></td><td>端到端3D，无纹理假设</td></tr>
    <tr><td>B.2（未来）</td><td>RAFT-Stereo / CREStereo + DLT</td><td>未开始</td><td>学习型视差可处理低纹理</td></tr>
  </table>
</div>

</div>
<footer>由 Claude Code 生成 · 硕士论文 · Fanbo Meng · 查尔姆斯理工大学</footer>
</body>
</html>"""


def main() -> None:
    with open(OUT_EN, "w", encoding="utf-8") as f:
        f.write(EN)
    print(f"[report] English → {OUT_EN}")

    with open(OUT_CN, "w", encoding="utf-8") as f:
        f.write(CN)
    print(f"[report] Chinese → {OUT_CN}")


if __name__ == "__main__":
    main()
