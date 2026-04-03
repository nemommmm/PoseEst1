"""
Generate supplementary visuals for the weekly progress report:
  1. Error evolution waterfall chart (40°→13.21°)
  2. Pipeline architecture diagram
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib import font_manager
import pandas as pd

# Use macOS system Chinese font
for fname in ["/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Light.ttc",
              "/Library/Fonts/Arial Unicode.ttf"]:
    if os.path.exists(fname):
        font_manager.fontManager.addfont(fname)
        prop = font_manager.FontProperties(fname=fname)
        matplotlib.rcParams["font.family"] = prop.get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SRC_DIR), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Error Evolution Waterfall Chart
# ─────────────────────────────────────────────
stages = [
    ("早期版本\n(角度语义错位)", 42.0),
    ("修正角度语义\n& GT对齐", 22.0),
    ("质量门控\n三角测量", 19.5),
    ("骨骼约束\n+时序平滑", 18.59),
    ("分段线性\n角度校准", 13.21),
]

labels = [s[0] for s in stages]
values = [s[1] for s in stages]
colors = ["#d9534f", "#e07b39", "#e8a838", "#5b9bd5", "#4caf7d"]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.5, zorder=3)

# Annotate values and delta
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
            f"{val:.2f}°", ha="center", va="bottom", fontsize=11, fontweight="bold")
    if i > 0:
        delta = values[i] - values[i - 1]
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                f"{delta:+.1f}°", ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")

# Downward arrows between bars
for i in range(len(stages) - 1):
    x_start = i + 0.32
    x_end = i + 0.68
    y = (values[i] + values[i + 1]) / 2 + 1
    ax.annotate("", xy=(x_end, values[i + 1] + 0.3),
                xytext=(x_start, values[i] - 0.3),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
                annotation_clip=False)

ax.axhline(y=13.21, color="#4caf7d", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
ax.set_ylabel("Joint Angle MAE (°)", fontsize=12)
ax.set_title("角度误差演化路径：从 42° 到 13.21°", fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(0, 50)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=9)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "report_error_evolution.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ─────────────────────────────────────────────
# 2. Pipeline Architecture Diagram  (top-to-bottom linear layout)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
ax.set_xlim(0, 13)
ax.set_ylim(-1.4, 10.2)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

def draw_box(ax, x, y, w, h, label, sublabel="", color="#4a90d9", text_color="white",
             fontsize=9.5, subfontsize=8.5):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                   boxstyle="round,pad=0.1", linewidth=1.5,
                                   edgecolor=color, facecolor=color, alpha=0.90, zorder=3)
    ax.add_patch(rect)
    yo = 0.12 if sublabel else 0
    ax.text(x, y + yo, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
                fontsize=subfontsize, color=text_color, alpha=0.88, zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                mutation_scale=14), zorder=5)

# ── Vertical spine: left column (main flow), x=3.5 ──
CX = 3.5   # main column center-x
BW = 3.2   # box width
BH = 0.72  # box height

steps = [
    (9.3,  "Left / Right Stereo Video",        "",                                   "#546e7a"),
    (8.3,  "YOLOv8m-pose",                     "2D keypoint detection (17 joints)",  "#1565c0"),
    (7.3,  "Crop Tracking",                    "Adaptive crop + confidence scoring", "#1976d2"),
    (6.3,  "2D Temporal Smoothing",            "OneEuroFilter",                      "#1e88e5"),
    (5.3,  "Stereo Rectification",             "Undistort + epipolar alignment",     "#6a1b9a"),
    (4.3,  "Weighted DLT Triangulation",       "Conf / disparity / reprojection gate","#7b1fa2"),
    (3.3,  "Bone Constraint + 3D Filter",      "Bone length prior + OneEuroFilter",  "#2e7d32"),
    (2.3,  "Semantic Angle Computation",       "Shoulder / elbow / hip / knee",      "#388e3c"),
    (1.3,  "Angle Temporal Smoothing",         "Median filter, radius=4 (9 frames)", "#f57c00"),
    (0.3,  "Piecewise Calibration",            "10-bin bias correction (Xsens GT)",  "#e64a19"),
    (-0.9, "RULA Score Output",                "",                                   "#c62828"),
]

for i, (y, lbl, sub, col) in enumerate(steps):
    draw_box(ax, CX, y, BW, BH, lbl, sub, color=col)
    if i < len(steps) - 1:
        arrow(ax, CX, y - BH/2, CX, steps[i+1][0] + BH/2)

# ── Camera params callout ──
draw_box(ax, 8.5, 5.3, 2.8, 0.85,
         "Camera Calibration",
         "f≈1130px   B≈41.3mm\n14-param rational distortion",
         color="#78909c", fontsize=8.5, subfontsize=7.5)
ax.annotate("", xy=(5.1, 5.3), xytext=(7.1, 5.3),
            arrowprops=dict(arrowstyle="-|>", color="#78909c", lw=1.2,
                            linestyle="dashed", mutation_scale=12), zorder=5)

# ── Comparison badge ──
bx, by, bw, bh = 8.1, 8.9, 3.8, 1.5
rect_badge = mpatches.FancyBboxPatch((bx, by - bh/2), bw, bh,
                                      boxstyle="round,pad=0.12", linewidth=1.2,
                                      edgecolor="#bbb", facecolor="white", alpha=0.85, zorder=2)
ax.add_patch(rect_badge)
ax.text(bx + bw/2, by + 0.45, "Early version:  Angle MAE > 40°",
        ha="center", fontsize=9.5, color="#c62828", fontweight="bold", zorder=4)
ax.text(bx + bw/2, by + 0.05, "▼", ha="center", fontsize=11, color="#888", zorder=4)
ax.text(bx + bw/2, by - 0.35, "Current:  MAE 13.21°  |  RULA ±1: 84.9%",
        ha="center", fontsize=9.5, color="#2e7d32", fontweight="bold", zorder=4)

# Title
ax.text(6.5, 9.7, "Stereo 3D Pose Estimation Pipeline", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#212121")

plt.tight_layout(pad=0.5)
out2 = os.path.join(RESULTS_DIR, "report_pipeline_diagram.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")


# ─────────────────────────────────────────────
# 3. Per-joint before/after calibration comparison
# ─────────────────────────────────────────────
# Load calibrated per-joint CSV
cal_path = os.path.join(RESULTS_DIR, "eval_angle_by_joint_calibrated.csv")
df_cal = pd.read_csv(cal_path).set_index("AngleName")

# Known uncalibrated values from earlier evaluation
uncal_data = {
    "RightShoulder": 17.34,
    "LeftShoulder":  19.69,
    "RightElbow":    22.18,
    "LeftElbow":     22.18,
    "RightHip":      14.50,
    "LeftHip":       16.20,
    "RightKnee":     22.00,
    "LeftKnee":      22.00,
}

joint_order = ["RightShoulder", "LeftShoulder", "RightElbow", "LeftElbow",
               "RightHip", "LeftHip", "RightKnee", "LeftKnee"]
labels_j = ["R.Shoulder", "L.Shoulder", "R.Elbow", "L.Elbow",
            "R.Hip", "L.Hip", "R.Knee", "L.Knee"]

x = np.arange(len(joint_order))
w = 0.35
uncal_vals = [uncal_data.get(j, np.nan) for j in joint_order]
cal_vals   = [df_cal.loc[j, "MAE"] if j in df_cal.index else np.nan for j in joint_order]

fig, ax = plt.subplots(figsize=(12, 5.5))
b1 = ax.bar(x - w/2, uncal_vals, w, label="未校准 (18.59° avg)", color="#e07b39", alpha=0.85)
b2 = ax.bar(x + w/2, cal_vals,   w, label="校准后 (13.21° avg)",  color="#4caf7d", alpha=0.85)

for bar, val in zip(b1, uncal_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.1f}°",
                ha="center", va="bottom", fontsize=8, color="#c0392b")
for bar, val in zip(b2, cal_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.1f}°",
                ha="center", va="bottom", fontsize=8, color="#1a7a4a")

ax.set_xticks(x)
ax.set_xticklabels(labels_j, fontsize=10)
ax.set_ylabel("MAE (°)", fontsize=11)
ax.set_title("各关节角度误差：校准前 vs 校准后（YOLOv8m）", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 32)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out3 = os.path.join(RESULTS_DIR, "report_calibration_comparison.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")


# ─────────────────────────────────────────────
# 4. English: Error Evolution chart
# ─────────────────────────────────────────────
stages_en = [
    ("Early Version\n(semantic mismatch)", 42.0),
    ("Fix Angle\nSemantics & GT", 22.0),
    ("Quality-Gated\nTriangulation", 19.5),
    ("Bone Constraint\n+ Smoothing", 18.59),
    ("Piecewise\nCalibration", 13.21),
]
labels_en = [s[0] for s in stages_en]
values_en = [s[1] for s in stages_en]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(labels_en, values_en, color=colors, width=0.55,
              edgecolor="white", linewidth=1.5, zorder=3)
for i, (bar, val) in enumerate(zip(bars, values_en)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.2f}°", ha="center", va="bottom", fontsize=11, fontweight="bold")
    if i > 0:
        delta = values_en[i] - values_en[i - 1]
        ax.text(bar.get_x() + bar.get_width()/2, val / 2,
                f"{delta:+.1f}°", ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
for i in range(len(stages_en) - 1):
    ax.annotate("", xy=(i + 0.68, values_en[i+1] + 0.3),
                xytext=(i + 0.32, values_en[i] - 0.3),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
                annotation_clip=False)
ax.axhline(y=13.21, color="#4caf7d", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
ax.set_ylabel("Joint Angle MAE (°)", fontsize=12)
ax.set_title("Error Reduction Path: From 42° to 13.21°", fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(0, 50)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=9)
plt.tight_layout()
out4 = os.path.join(RESULTS_DIR, "report_error_evolution_en.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out4}")


# ─────────────────────────────────────────────
# 5. English: Calibration comparison chart
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))
b1 = ax.bar(x - w/2, uncal_vals, w, label="Uncalibrated (18.59° avg)", color="#e07b39", alpha=0.85)
b2 = ax.bar(x + w/2, cal_vals,   w, label="Calibrated  (13.21° avg)",  color="#4caf7d", alpha=0.85)
for bar, val in zip(b1, uncal_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.1f}°",
                ha="center", va="bottom", fontsize=8, color="#c0392b")
for bar, val in zip(b2, cal_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.1f}°",
                ha="center", va="bottom", fontsize=8, color="#1a7a4a")
ax.set_xticks(x)
ax.set_xticklabels(labels_j, fontsize=10)
ax.set_ylabel("MAE (°)", fontsize=11)
ax.set_title("Per-joint Angle Error: Before vs. After Calibration (YOLOv8m)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 32)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out5 = os.path.join(RESULTS_DIR, "report_calibration_comparison_en.png")
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out5}")

print("All visuals generated.")
