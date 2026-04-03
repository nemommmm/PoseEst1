"""Generate weekly progress report (ZH + EN) as self-contained HTML files."""
import os, base64

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_METHOD_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(_METHOD_DIR)
RESULTS_DIR = os.path.join(_METHOD_DIR, "results")

imgs_zh = {
    "evolution":    os.path.join(RESULTS_DIR, "report_error_evolution.png"),
    "pipeline":     os.path.join(RESULTS_DIR, "report_pipeline_diagram.png"),
    "cal_compare":  os.path.join(RESULTS_DIR, "report_calibration_comparison.png"),
    "joint_cal":    os.path.join(RESULTS_DIR, "eval_angle_by_joint_calibrated.png"),
    "scenario_cal": os.path.join(RESULTS_DIR, "eval_angle_by_scenario_calibrated.png"),
    "rula_conf":    os.path.join(RESULTS_DIR, "eval_rula_confusion.png"),
    "trunk":        os.path.join(RESULTS_DIR, "eval_trunk_flexion_compare_calibrated.png"),
}
imgs_en = {**imgs_zh,
    "evolution":   os.path.join(RESULTS_DIR, "report_error_evolution_en.png"),
    "cal_compare": os.path.join(RESULTS_DIR, "report_calibration_comparison_en.png"),
}
imgs = imgs_zh  # default; switched per language below

def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def make_img_tag(img_dict):
    def img_tag(key, caption="", height_px=None):
        path = img_dict[key]
        b64 = img_b64(path)
        ext = os.path.splitext(path)[1].lstrip(".")
        style = "width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"
        if height_px:
            style += f"height:{height_px}px;object-fit:contain;background:#fff;"
        img = f'<img src="data:image/{ext};base64,{b64}" style="{style}">'
        if caption:
            return f'<figure style="margin:12px 0;">{img}<figcaption>{caption}</figcaption></figure>'
        return f'<div style="margin:12px 0;">{img}</div>'
    return img_tag

# ─────────── shared CSS ───────────
CSS = """
body{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;margin:0;background:#f5f6fa;color:#222;}
.container{max-width:980px;margin:40px auto;background:#fff;padding:48px 56px;border-radius:10px;
  box-shadow:0 4px 24px rgba(0,0,0,.10);}
h1{font-size:26px;border-bottom:3px solid #1565c0;padding-bottom:10px;color:#1565c0;}
h2{font-size:20px;margin-top:40px;color:#1a237e;border-left:4px solid #1565c0;padding-left:10px;}
h3{font-size:16px;color:#333;margin-top:22px;}
p,li{line-height:1.85;font-size:15px;}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;}
th{background:#1565c0;color:#fff;padding:8px 12px;text-align:left;}
td{padding:7px 12px;border-bottom:1px solid #e0e0e0;}
tr:nth-child(even){background:#f5f8ff;}
.hl{background:#e8f5e9;border-left:4px solid #4caf50;padding:12px 16px;border-radius:4px;margin:16px 0;}
.warn{background:#fff8e1;border-left:4px solid #ff9800;padding:12px 16px;border-radius:4px;margin:16px 0;}
.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0;}
.mc{background:#f0f4ff;border-radius:8px;padding:16px;text-align:center;}
.mc .val{font-size:26px;font-weight:bold;color:#1565c0;}
.mc .lbl{font-size:12px;color:#666;margin-top:4px;}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;}
figcaption{font-size:12.5px;color:#666;text-align:center;margin-top:5px;}
figure{margin:12px 0;}
code{background:#f0f4ff;padding:2px 6px;border-radius:3px;font-size:13px;}
hr{margin-top:40px;border:none;border-top:1px solid #ddd;}
.foot{text-align:center;color:#aaa;font-size:12px;}
"""

# ══════════════════════════════════════════════════════
#  CHINESE VERSION
# ══════════════════════════════════════════════════════
img_tag = make_img_tag(imgs_zh)
zh = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<title>每周进展报告 – 双目立体姿态估计 (2026-03-24)</title>
<style>{CSS}</style></head>
<body><div class="container">

<h1>每周进展报告
<br><span style="font-size:15px;font-weight:normal;color:#666;">
双目立体视觉 3D 姿态估计 · 人机工效学 RULA 评估 · 2026-03-24</span></h1>

<!-- ══════════════════ 一 ══════════════════ -->
<h2>一、当前关键结果数据</h2>

<div class="mg">
  <div class="mc"><div class="val">13.21°</div><div class="lbl">整体角度 MAE（校准后）</div></div>
  <div class="mc"><div class="val">76.3%</div><div class="lbl">肘部 RULA 类别准确率</div></div>
  <div class="mc"><div class="val">84.9%</div><div class="lbl">RULA Grand Score ±1 容差</div></div>
  <div class="mc"><div class="val">18.59°</div><div class="lbl">整体角度 MAE（未校准）</div></div>
  <div class="mc"><div class="val">65.8%</div><div class="lbl">肘部 RULA（未校准）</div></div>
  <div class="mc"><div class="val">26.02 cm</div><div class="lbl">MPJPE（空间参考，校准前后相同）</div></div>
</div>

<p>MPJPE（关节位置均方误差）在角度校准前后保持不变（均为 26.02 cm），因为 MPJPE 直接由 3D 关键点计算，角度校准仅作用于从关键点推导出的角度值。</p>

<table>
  <tr><th>场景</th><th>角度 MAE（校准后）</th><th>样本数</th></tr>
  <tr><td>Baseline（自由行走）</td><td>14.2°</td><td>2,241</td></tr>
  <tr><td>Environmental Interference</td><td>16.8°</td><td>8,028</td></tr>
  <tr><td>Occlusion（遮挡）</td><td>17.4°</td><td>7,812</td></tr>
</table>

<div class="two">
  {img_tag("joint_cal", "图1：各关节角度 MAE（校准后）", height_px=320)}
  {img_tag("rula_conf", "图2：RULA Grand Score 混淆矩阵", height_px=320)}
</div>
{img_tag("scenario_cal", "图3：按场景分类的角度误差（校准后）")}


<!-- ══════════════════ 二 ══════════════════ -->
<h2>二、当前 Pipeline 架构</h2>

{img_tag("pipeline", "图4：双目立体姿态估计 Pipeline（当前版本）")}

<table>
  <tr><th>维度</th><th>早期版本</th><th>当前版本</th></tr>
  <tr><td>主优化指标</td><td>MPJPE</td><td><b>Joint Angle MAE</b></td></tr>
  <tr><td>检测策略</td><td>全帧 YOLO，无追踪</td><td>Crop 动态追踪 + 置信度评分</td></tr>
  <tr><td>三角测量</td><td>无质量门控</td><td>置信度 / 视差 / 重投影三重门控</td></tr>
  <tr><td>后处理</td><td>无</td><td>骨骼长度约束 + 3D OneEuroFilter</td></tr>
  <tr><td>角度语义</td><td>定义混乱，比错了量</td><td>统一语义，GT 精确对应解剖量</td></tr>
  <tr><td>系统性偏差</td><td>未处理</td><td>分段线性角度校准（10 bins）</td></tr>
  <tr><td>典型角度误差</td><td>40+°</td><td><b>13.21°（校准后）</b></td></tr>
</table>


<!-- ══════════════════ 三 ══════════════════ -->
<h2>三、为什么 Pipeline 要做出这些调整</h2>

<h3>3.1 从 MPJPE 转向角度 MAE 为主指标</h3>
<p>MPJPE 衡量 Kabsch 全局刚体对齐后骨架点云的空间距离，会掩盖局部关节方向错误。腕/肘部 3–4 cm 的绝对误差对 25 cm 长的肢段可造成 10°+ 的角度偏差，而 MPJPE 对此感知迟钝。改为以角度 MAE 为核心后，优化方向聚焦到 RULA 评估真正关心的"关节方向准确性"。</p>

<h3>3.2 Crop 追踪</h3>
<p>全帧推理在多人场景下容易混入错误的人体检测。Crop 追踪基于前帧检测框裁剪局部图像，提高目标正确率，并降低背景干扰引起的关键点跳变。</p>

<h3>3.3 三重质量门控</h3>
<p>原始三角测量对所有 2D 检测同等信任，遮挡帧和低置信度帧引入大量噪声。引入<b>双目置信度 × 视差 × 重投影误差</b>联合门控后，不满足条件的关键点标记为 NaN 而非传入错误的 3D 值。</p>

<h3>3.4 骨骼长度约束</h3>
<p>三角测量对每帧独立计算，允许同一骨段在相邻帧长度剧烈变化。骨骼约束从稳定帧估计先验骨段长度，将子关节投影至先验距离，抑制深度方向噪声。</p>


<!-- ══════════════════ 四 ══════════════════ -->
<h2>四、关于角度校准</h2>

<h3>4.1 为什么需要角度校准</h3>
<p>相机标定（重投影误差 &lt;1px）解决"给定 2D 点算出正确 3D 坐标"的问题。角度校准解决的是另一个问题：<b>YOLO 检测到的关键点与解剖学关节旋转中心不是同一个点</b>，存在 3–8mm 的系统性偏移。此外，绝对角度计算存在"regression to mean"效应：</p>
<ul>
  <li>真实角度接近 0° → 噪声只能向大方向推 → 系统性<b>高估</b></li>
  <li>真实角度接近 90° → 噪声双向抵消 → 无系统偏差</li>
  <li>真实角度较大 → 噪声倾向于向小方向拉 → 系统性<b>低估</b></li>
</ul>
<p>分段线性校准（10 bins）学习并反转每个关节的偏差曲线，将 MAE 从 18.59° 降至 13.21°，RULA ±1 准确率从 ~75% 提升至 84.9%。</p>

{img_tag("cal_compare", "图5：各关节校准前后对比（YOLOv8m）")}

<h3>4.2 存在的问题：过拟合风险</h3>
<div class="warn">
<b>当前校准在被试者 Aitor 的单一视频上完成训练与测试</b>，存在明显局限：
<ul>
  <li>校准曲线同时记住了"系统性偏差"和"Aitor 的个体特征"，泛化到不同被试者的能力未经验证</li>
  <li>角度范围稀疏的 bin（如极端屈曲动作）校正可能不可靠</li>
  <li>依赖 Xsens 离线标定，不符合工业部署的成本约束</li>
</ul>
</div>

<h3>4.3 工业场景的改进方向</h3>
<table>
  <tr><th>方案</th><th>成本</th><th>预期改善</th><th>适用场景</th></tr>
  <tr><td>身高输入</td><td>极低</td><td>~1–2°</td><td>快速部署，精度要求不高</td></tr>
  <tr><td>标准姿势校准（30 秒）</td><td>低</td><td>~3–4°</td><td>工业现场，有操作规范</td></tr>
  <tr><td><b>骨骼比例自动估计（推荐）</b></td><td><b>零成本</b></td><td>~2–3°</td><td>全自动，无人工干预</td></tr>
  <tr><td>Xsens GT 校准（当前）</td><td>高</td><td>~5.4°</td><td>研究/高精度场景</td></tr>
</table>
<p>推荐方案：系统首次使用时录制约 30 秒自然站立视频，自动估计各肢段比例，生成每人专属校准参数，后续无需任何额外输入。</p>


<!-- ══════════════════ 五 ══════════════════ -->
<h2>五、相机标定的问题与重新标定</h2>

<h3>5.1 发现问题的过程</h3>
<p>在早期评估中，观察到重投影误差在 <b>Y 方向（极线方向）异常偏大</b>：经过立体矫正后，左右图像中对应关键点的 Y 坐标仍存在明显不一致（epipolar error 高），说明初始标定采集的图像质量或覆盖范围不足，导致畸变参数估计不准，矫正后极线仍有偏移。</p>

<h3>5.2 重新标定的主要调整</h3>
<ul>
  <li><b>标定板类型</b>：改用非对称圆形网格（5×9），在亮度不均场景下比棋盘格检测更稳定</li>
  <li><b>图像质量过滤</b>：每帧重投影误差 &gt; 1.0px 的标定图像直接剔除</li>
  <li><b>畸变模型升级</b>：从 5 参数升级为 OpenCV <b>14 参数有理畸变模型</b></li>
</ul>

<h3>5.3 当前标定核心参数</h3>
<table>
  <tr><th>参数</th><th>左相机</th><th>右相机</th></tr>
  <tr><td>焦距 f (px)</td><td>1130.3</td><td>1129.0</td></tr>
  <tr><td>主点 (cx, cy)</td><td>(1019.3, 825.7)</td><td>(1028.3, 826.3)</td></tr>
  <tr><td>基线 B</td><td colspan="2"><b>41.27 mm</b></td></tr>
  <tr><td>矫正后 cy 差值</td><td colspan="2"><b>0.0000 px</b>（极线完全对齐）</td></tr>
  <tr><td>畸变模型</td><td colspan="2">14 参数有理模型</td></tr>
</table>

<div class="hl">
<b>关于左右相机 k1 数值不对称（k1≈40 vs. k1≈9）的说明：</b>在 14 参数有理模型中，k1 是分子系数，k4 是分母系数，两者在校正时相互抵消。实际计算两相机在肩部区域的有效像素校正量分别为 105px（左）和 104px（右），几乎相同。k1 不对称是有理模型的正常现象，标定没有问题。
</div>


<!-- ══════════════════ 六 ══════════════════ -->
<h2>六、Q&amp;A：早期 40+° 角度误差是怎么回事</h2>

{img_tag("evolution", "图6：角度误差演化路径，从 42° 到 13.21°")}

<h3>第一层（最大影响）：角度语义对不上</h3>
<p>早期估计端算的是"几何三点夹角"，GT 端拿的是不同解剖定义的角度。以肩部为例：估计端用 <code>shoulder→elbow</code> 向量的俯仰角，GT 端用 Xsens 的解剖欧拉角——两个量根本不是同一个物理量。修正后把 "40+" 直接拉到了 "20 多"。</p>

<h3>第二层（中等影响）：MPJPE 掩盖了角度问题</h3>
<p>"整体位置准、局部关节方向偏"的结果，MPJPE 可以不差但角度 MAE 很高。以 MPJPE 为优化目标时，局部关节方向错误没有被充分暴露和修复。</p>

<h3>第三层（真实误差）：三角测量噪声</h3>
<p>遮挡、低置信度关键点、无门控的三角测量贡献了真实的几何误差，通过后续质量门控、骨骼约束和时序平滑逐步压缩。</p>

<table>
  <tr><th>改进步骤</th><th>主要作用</th><th>效果估计</th></tr>
  <tr><td>角度语义统一 + GT 对齐修正</td><td>消除假高误差</td><td>约 −15°</td></tr>
  <tr><td>主指标切换为角度 MAE</td><td>优化方向聚焦</td><td>间接影响后续所有优化</td></tr>
  <tr><td>三角测量质量门控</td><td>减少真实几何噪声</td><td>约 −3°</td></tr>
  <tr><td>骨骼约束 + 时序平滑</td><td>降低帧间抖动</td><td>约 −1°</td></tr>
  <tr><td>分段线性角度校准</td><td>修正系统性偏差</td><td><b>−5.4°</b></td></tr>
</table>


<!-- ══════════════════ 七 ══════════════════ -->
<h2>七、下一步计划</h2>

<h3>方向 A：替换 2D 关键点检测模型</h3>
<p>当前使用 YOLOv8m-pose，后续可尝试在工业/人体姿态估计基准上精度更高的模型：</p>
<ul>
  <li><b>RTMPose-x（performance 模式）</b>：在 COCO-Pose 上 AP 约 75–77%，高于 YOLOv8m（~65%）。本次初步测试的 RTMPose-m 结果较差，主要原因是置信度分布与现有三角测量门控参数不兼容——换用 RTMPose-x 并重新调优置信度阈值后仍有潜力</li>
  <li><b>ViTPose</b>：基于 Vision Transformer，关键点定位精度更高，对遮挡场景更鲁棒</li>
  <li><b>HRNet（High-Resolution Net）</b>：在高分辨率特征图上估计关键点，对细粒度关节（肘/腕）精度更好</li>
</ul>
<p>替换 2D 检测器的集成成本低——所有后续步骤（矫正、三角测量、后处理）无需修改，只需调整置信度相关参数。</p>

<h3>方向 B：稠密立体匹配（Dense Stereo）★★</h3>
<p>与当前"稀疏关键点 + 三角测量"不同，稠密立体匹配基于整张图像的视差估计深度，不依赖关键点检测精度。</p>
<ul>
  <li><b>优势</b>：对遮挡更鲁棒；不受 YOLO 关键点与解剖中心偏差影响；可结合人体分割 mask 提取关节深度</li>
  <li><b>实现路径</b>：用 RAFT-Stereo / SGBM 生成视差图，将 YOLO 2D 关键点坐标查表至视差值，替代 DLT 三角测量</li>
  <li><b>需验证</b>：室内均匀光照下纹理少的区域（如布料）稠密匹配质量是否足够</li>
</ul>

<h3>方向 C：端到端 3D 姿态估计 Pipeline</h3>
<p>用预训练视频级 3D 人体姿态模型（MotionBERT、VideoPose3D）直接从视频预测 3D 关节位置，利用时序上下文和运动先验。工程复杂度较高（需解决坐标系对齐），可作为中长期方向。</p>

<div class="hl">
<b>优先推荐顺序</b>：方向 A（替换 2D 检测器）实施成本最低，效果可快速验证；方向 B 充分利用双目硬件优势，是核心技术路线；方向 C 长期潜力最大但工程量最重。
</div>

{img_tag("trunk", "图7：躯干屈曲角度估计 vs Xsens GT（校准后）")}

<hr><p class="foot">自动生成 · 2026-03-24 · 双目立体视觉人机工效学评估项目</p>
</div></body></html>"""

# ══════════════════════════════════════════════════════
#  ENGLISH VERSION
# ══════════════════════════════════════════════════════
img_tag = make_img_tag(imgs_en)
en = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Weekly Progress Report – Stereo 3D Pose Estimation (2026-03-24)</title>
<style>{CSS.replace("PingFang SC,","")}</style></head>
<body><div class="container">

<h1>Weekly Progress Report
<br><span style="font-size:15px;font-weight:normal;color:#666;">
Stereo Vision 3D Pose Estimation · Ergonomic RULA Assessment · 2026-03-24</span></h1>

<!-- ══════════════════ 1 ══════════════════ -->
<h2>1. Key Results Summary</h2>

<div class="mg">
  <div class="mc"><div class="val">13.21°</div><div class="lbl">Overall Angle MAE (calibrated)</div></div>
  <div class="mc"><div class="val">76.3%</div><div class="lbl">Elbow RULA Category Accuracy</div></div>
  <div class="mc"><div class="val">84.9%</div><div class="lbl">RULA Grand Score within ±1</div></div>
  <div class="mc"><div class="val">18.59°</div><div class="lbl">Overall Angle MAE (uncalibrated)</div></div>
  <div class="mc"><div class="val">65.8%</div><div class="lbl">Elbow RULA (uncalibrated)</div></div>
  <div class="mc"><div class="val">26.02 cm</div><div class="lbl">MPJPE (same before/after calibration)</div></div>
</div>

<p>Note: MPJPE (Mean Per-Joint Position Error) is identical with and without angle calibration (26.02 cm), since it is computed directly from 3D keypoints — angle calibration only affects the angle values derived from those keypoints.</p>

<table>
  <tr><th>Scenario</th><th>Angle MAE (calibrated)</th><th>Samples</th></tr>
  <tr><td>Baseline (free walking)</td><td>14.2°</td><td>2,241</td></tr>
  <tr><td>Environmental Interference</td><td>16.8°</td><td>8,028</td></tr>
  <tr><td>Occlusion</td><td>17.4°</td><td>7,812</td></tr>
</table>

<div class="two">
  {img_tag("joint_cal", "Fig. 1: Per-joint angle MAE (calibrated)", height_px=320)}
  {img_tag("rula_conf", "Fig. 2: RULA Grand Score confusion matrix", height_px=320)}
</div>
{img_tag("scenario_cal", "Fig. 3: Angle error by scenario (calibrated)")}


<!-- ══════════════════ 2 ══════════════════ -->
<h2>2. Current Pipeline Architecture</h2>

{img_tag("pipeline", "Fig. 4: Stereo 3D Pose Estimation Pipeline (current version)")}

<table>
  <tr><th>Aspect</th><th>Early Version</th><th>Current Version</th></tr>
  <tr><td>Primary Metric</td><td>MPJPE</td><td><b>Joint Angle MAE</b></td></tr>
  <tr><td>Detection Strategy</td><td>Full-frame YOLO, no tracking</td><td>Adaptive crop tracking + confidence scoring</td></tr>
  <tr><td>Triangulation</td><td>No quality gating</td><td>Confidence / disparity / reprojection triple gate</td></tr>
  <tr><td>Post-processing</td><td>None</td><td>Bone length constraint + 3D OneEuroFilter</td></tr>
  <tr><td>Angle Semantics</td><td>Inconsistent — estimating different quantities than GT</td><td>Unified semantic definition aligned to GT anatomy</td></tr>
  <tr><td>Systematic Bias</td><td>Unaddressed</td><td>Piecewise linear angle calibration (10 bins)</td></tr>
  <tr><td>Typical Angle Error</td><td>40+°</td><td><b>13.21° (calibrated)</b></td></tr>
</table>


<!-- ══════════════════ 3 ══════════════════ -->
<h2>3. Rationale for Pipeline Changes</h2>

<h3>3.1 Switching Primary Metric from MPJPE to Angle MAE</h3>
<p>MPJPE measures the spatial distance of the whole-body skeleton after global Kabsch alignment, which can mask local joint orientation errors. A 3–4 cm absolute error at the wrist/elbow is sufficient to cause 10°+ of angle error on a 25 cm limb segment — yet MPJPE is insensitive to this. Switching to angle MAE as the primary metric immediately aligned the optimization target with what RULA scoring actually requires: correct joint directions.</p>

<h3>3.2 Crop Tracking</h3>
<p>Full-frame inference is easily confused by multiple persons in the scene. Crop tracking crops a local image region around the previous detection box, improving target identity consistency and reducing keypoint jumps caused by background detections.</p>

<h3>3.3 Triple Quality Gate</h3>
<p>Naive triangulation trusted all 2D detections equally, letting occluded and low-confidence frames inject large amounts of noise. The <b>confidence × disparity × reprojection error</b> gate marks invalid triangulations as NaN rather than propagating erroneous 3D values.</p>

<h3>3.4 Bone Length Constraint</h3>
<p>Frame-independent triangulation allowed the same limb segment to change length drastically between frames (especially along the depth axis). The bone constraint estimates prior limb lengths from stable frames and projects each child joint back to the prior distance, suppressing depth-axis noise.</p>


<!-- ══════════════════ 4 ══════════════════ -->
<h2>4. Angle Calibration — Necessity, Limitations, and Industrial Path</h2>

<h3>4.1 Why Angle Calibration is Needed</h3>
<p>Camera calibration (reprojection error &lt;1 px) solves "given 2D points, compute correct 3D coordinates." Angle calibration addresses a different problem: <b>YOLO keypoints and true anatomical joint rotation centres are not the same point</b>, with a systematic offset of 3–8 mm. There is also a mathematical "regression-to-mean" effect on absolute angle values:</p>
<ul>
  <li>True angle near 0° → noise can only push it up → systematic <b>overestimation</b></li>
  <li>True angle near 90° → noise cancels symmetrically → no systematic bias</li>
  <li>True angle large → noise tends to pull it down → systematic <b>underestimation</b></li>
</ul>
<p>A 10-bin piecewise linear calibration learns and inverts this per-joint bias curve, reducing MAE from 18.59° to 13.21° and improving RULA ±1 accuracy from ~75% to 84.9%.</p>

{img_tag("cal_compare", "Fig. 5: Per-joint angle error before and after calibration (YOLOv8m)")}

<h3>4.2 Limitations: Risk of Overfitting</h3>
<div class="warn">
<b>The current calibration was trained and tested on a single video of one subject (Aitor)</b>, with notable limitations:
<ul>
  <li>The calibration curve encodes both systematic bias and Aitor's individual characteristics (body proportions, clothing, movement habits) — generalisation to different subjects is unvalidated</li>
  <li>Bins with sparse angle samples (extreme flexion) may produce unreliable corrections</li>
  <li>Requires expensive Xsens hardware for offline calibration — incompatible with industrial deployment costs</li>
</ul>
</div>

<h3>4.3 Industrial Deployment Alternatives</h3>
<table>
  <tr><th>Approach</th><th>Cost</th><th>Expected Improvement</th><th>Use Case</th></tr>
  <tr><td>Height input</td><td>Minimal</td><td>~1–2°</td><td>Rapid deployment, moderate accuracy</td></tr>
  <tr><td>Reference pose calibration (30 s)</td><td>Low</td><td>~3–4°</td><td>Industrial site with standardised procedures</td></tr>
  <tr><td><b>Automatic bone proportion estimation (recommended)</b></td><td><b>Zero</b></td><td>~2–3°</td><td>Fully automated, no manual input</td></tr>
  <tr><td>Xsens GT calibration (current)</td><td>High</td><td>~5.4°</td><td>Research / high-accuracy scenarios</td></tr>
</table>
<p>Recommended approach: record ~30 s of natural standing at first use; automatically estimate limb proportions; generate per-person calibration parameters for all subsequent assessments — zero additional cost, no user input required.</p>


<!-- ══════════════════ 5 ══════════════════ -->
<h2>5. Camera Calibration Issue and Re-calibration</h2>

<h3>5.1 How the Problem was Found</h3>
<p>During early evaluation, the <b>Y-direction (epipolar) error was unusually large</b>: after stereo rectification — which should eliminate Y-axis misalignment — corresponding keypoints in the left and right images still showed significant Y-coordinate discrepancy. This indicated that the initial calibration images lacked sufficient quality or spatial coverage, causing inaccurate distortion parameter estimation.</p>

<h3>5.2 Re-calibration Changes</h3>
<ul>
  <li><b>Calibration target</b>: Switched to an asymmetric circle grid (5×9), which is more stable than a checkerboard under uneven illumination</li>
  <li><b>Frame filtering</b>: Calibration frames with per-camera reprojection error &gt; 1.0 px were discarded</li>
  <li><b>Distortion model</b>: Upgraded from 5-parameter to OpenCV's <b>14-parameter rational distortion model</b></li>
</ul>

<h3>5.3 Key Calibration Parameters</h3>
<table>
  <tr><th>Parameter</th><th>Left Camera</th><th>Right Camera</th></tr>
  <tr><td>Focal length f (px)</td><td>1130.3</td><td>1129.0</td></tr>
  <tr><td>Principal point (cx, cy)</td><td>(1019.3, 825.7)</td><td>(1028.3, 826.3)</td></tr>
  <tr><td>Baseline B</td><td colspan="2"><b>41.27 mm</b></td></tr>
  <tr><td>Post-rectification cy difference</td><td colspan="2"><b>0.0000 px</b> (epipolar lines perfectly aligned)</td></tr>
  <tr><td>Distortion model</td><td colspan="2">14-parameter rational model</td></tr>
</table>

<div class="hl">
<b>Note on asymmetric k1 values (k1≈40 left vs. k1≈9 right):</b> In the 14-parameter rational model, k1 is the numerator polynomial coefficient and k4 is the denominator coefficient; they partially cancel during correction. Computing the effective pixel correction at the shoulder region gives 105 px (left) and 104 px (right) — nearly identical. The numerical asymmetry is normal for rational distortion models and does not indicate a calibration error.
</div>


<!-- ══════════════════ 6 ══════════════════ -->
<h2>6. Q&amp;A: Why Was the Early Angle Error 40+°?</h2>

{img_tag("evolution", "Fig. 6: Error evolution from 42° to 13.21°")}

<h3>Layer 1 (Dominant): Mismatched Angle Semantics</h3>
<p>The early estimation computed "geometric three-point angles" while GT used different anatomical definitions. For the shoulder, estimation used the elevation of the <code>shoulder→elbow</code> vector while GT used a specific anatomical Euler axis from Xsens — fundamentally different quantities. Aligning the semantics immediately reduced the error from "40+" to "~20".</p>

<h3>Layer 2 (Moderate): MPJPE Hid Joint Orientation Errors</h3>
<p>A skeleton that is globally well-positioned but has incorrect local joint orientations can score well on MPJPE while having high angle MAE. Optimising for MPJPE meant joint direction errors were never properly exposed or fixed.</p>

<h3>Layer 3 (Real Error): Triangulation Noise</h3>
<p>Occlusion, low-confidence detections, and ungated triangulation contributed genuine geometric noise, progressively reduced by quality gating, bone constraints, and temporal smoothing.</p>

<table>
  <tr><th>Improvement Step</th><th>Effect</th><th>Estimated Contribution</th></tr>
  <tr><td>Angle semantics + GT alignment fix</td><td>Eliminate false-high error</td><td>≈ −15°</td></tr>
  <tr><td>Switch primary metric to angle MAE</td><td>Correct optimisation direction</td><td>Indirect — enables all subsequent gains</td></tr>
  <tr><td>Triangulation quality gate</td><td>Reduce real geometric noise</td><td>≈ −3°</td></tr>
  <tr><td>Bone constraint + temporal smoothing</td><td>Reduce inter-frame jitter</td><td>≈ −1°</td></tr>
  <tr><td>Piecewise angle calibration</td><td>Correct systematic bias</td><td><b>−5.4°</b></td></tr>
</table>


<!-- ══════════════════ 7 ══════════════════ -->
<h2>7. Next Steps</h2>

<h3>Direction A: Alternative 2D Keypoint Detectors</h3>
<p>The current YOLOv8m-pose can be replaced with models that achieve higher accuracy on human pose benchmarks:</p>
<ul>
  <li><b>RTMPose-x (performance mode)</b>: COCO-Pose AP ~75–77%, significantly higher than YOLOv8m (~65%). An initial test with RTMPose-m showed poor results primarily due to confidence score distribution mismatch with the existing triangulation gate parameters — re-tuning these thresholds for RTMPose-x remains worth exploring</li>
  <li><b>ViTPose</b>: Vision Transformer-based; stronger on occluded poses and fine-grained joints</li>
  <li><b>HRNet (High-Resolution Net)</b>: Maintains high-resolution feature maps throughout; better accuracy for elbow/wrist keypoints</li>
</ul>
<p>Swapping the 2D detector carries low integration cost — all downstream steps (rectification, triangulation, post-processing) remain unchanged; only confidence-related thresholds require re-tuning.</p>

<h3>Direction B: Dense Stereo Matching ★★</h3>
<p>Unlike the current sparse keypoint + triangulation approach, dense stereo matching estimates depth from the full image disparity map, independent of keypoint detection accuracy.</p>
<ul>
  <li><b>Advantages</b>: More robust under occlusion; unaffected by the YOLO-to-anatomical-centre offset; can leverage human segmentation masks to extract joint-region depth</li>
  <li><b>Implementation path</b>: Generate disparity maps with RAFT-Stereo or SGBM; look up 2D keypoint positions in the disparity map to replace DLT triangulation</li>
  <li><b>To validate</b>: Whether dense matching quality is sufficient in low-texture indoor regions (clothing, uniform surfaces)</li>
</ul>

<h3>Direction C: End-to-End 3D Pose Estimation</h3>
<p>Pre-trained video-level 3D pose models (MotionBERT, VideoPose3D) predict 3D joint positions directly from video using temporal context and human motion priors. Engineering complexity is higher (coordinate system alignment required); suitable as a medium-to-long-term direction.</p>

<div class="hl">
<b>Recommended priority:</b> Direction A (alternative 2D detector) has the lowest integration cost and fastest feedback; Direction B fully leverages existing stereo hardware and is the core technical path; Direction C offers the highest long-term ceiling but carries the most engineering effort.
</div>

{img_tag("trunk", "Fig. 7: Trunk flexion estimation vs. Xsens GT (calibrated)")}

<hr><p class="foot">Auto-generated · 2026-03-24 · Stereo Vision Ergonomic Assessment Project</p>
</div></body></html>"""

# ── Write files ──
zh_path = os.path.join(PROJECT_ROOT, "weekly_report_zh_20260324.html")
en_path = os.path.join(PROJECT_ROOT, "weekly_report_en_20260324.html")

with open(zh_path, "w", encoding="utf-8") as f:
    f.write(zh)
print(f"Chinese report: {zh_path}")

with open(en_path, "w", encoding="utf-8") as f:
    f.write(en)
print(f"English report: {en_path}")
