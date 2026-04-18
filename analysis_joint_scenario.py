#!/usr/bin/env python3
"""
Per-joint and per-scenario angle MAE analysis for SKT (Dir A) and MTL (Dir C).

Outputs saved to analysis_charts/:
  joint_mae_comparison.png    - per-joint MAE bar chart, SKT vs MTL
  scenario_mae_skt.png        - per-scenario MAE for SKT
  scenario_mae_mtl.png        - per-scenario MAE for MTL
  joint_coverage_skt.png      - NaN/fill rate per joint for SKT
  error_dist_comparison.png   - boxplot of error distribution per joint
  overview_summary.png        - 2x2 summary figure for report
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "01_stereo_triangulation", "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path

from utils_mvnx import MvnxParser
from pose_angle_utils import (
    DEFAULT_ANGLE_SMOOTH_RADIUS,
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
    median_filter_angle_sequence,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")
SKT_NPZ   = os.path.join(PROJECT_ROOT,
    "01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz")
MTL_NPZ   = os.path.join(PROJECT_ROOT,
    "03_mono_motionbert/results_mono/eval_results_mono.npz")
ALIGN_JSON = os.path.join(PROJECT_ROOT,
    "01_stereo_triangulation/results/alignment_summary.json")

OUT_DIR = Path(PROJECT_ROOT) / "analysis_charts"
OUT_DIR.mkdir(exist_ok=True)

# ── Temporal offsets ───────────────────────────────────────────────────────────
with open(ALIGN_JSON) as f:
    SKT_OFFSET = float(json.load(f).get("best_offset_seconds", 17.25))
MTL_OFFSET = 17.40   # confirmed from skeleton_comparison_dirC.json

# ── Activity segments (in video time, seconds from video start) ────────────────
ACTIVITY_SEGMENTS = {
    "Walking (Normal)":           (17,  32),
    "Walking (Late)":             (220, 240),
    "Sitting (Lower Occluded)":   (32,  62),
    "Walking (Upper Occluded)":   (87,  97),
    "Walking (Lower Occluded 1)": (130, 140),
    "Walking (Lower Occluded 2)": (164, 170),
    "Chair Interaction (Complex)":(140, 160),
    "Lifting Box (Near Chair)":   (214, 218),
    "Squatting":                  (66,  69),
    "Squatting (Check)":          (156, 160),
}
SCENARIO_MAPPING = {
    "Walking (Normal)":            "Baseline",
    "Walking (Late)":              "Baseline",
    "Sitting (Lower Occluded)":    "Occlusion",
    "Walking (Upper Occluded)":    "Occlusion",
    "Walking (Lower Occluded 1)":  "Occlusion",
    "Walking (Lower Occluded 2)":  "Occlusion",
    "Chair Interaction (Complex)": "Env. Interference",
    "Lifting Box (Near Chair)":    "Env. Interference",
    "Squatting":                   "Dynamic Action",
    "Squatting (Check)":           "Dynamic Action",
}

SCENARIO_ORDER  = ["Baseline", "Occlusion", "Env. Interference", "Dynamic Action", "Unclassified"]
SCENARIO_COLORS = {
    "Baseline":          "#4C72B0",
    "Occlusion":         "#DD8452",
    "Env. Interference": "#55A868",
    "Dynamic Action":    "#C44E52",
    "Unclassified":      "#8172B2",
}

# Joint display names (canonical 8-joint ergonomic set, L/R pairs merged for display)
JOINT_PAIRS = [
    ("RightShoulder", "LeftShoulder",  "Shoulder"),
    ("RightElbow",    "LeftElbow",     "Elbow"),
    ("RightHip",      "LeftHip",       "Hip"),
    ("RightKnee",     "LeftKnee",      "Knee"),
]
JOINT_DISPLAY_ORDER = [
    "RightShoulder", "LeftShoulder",
    "RightElbow",    "LeftElbow",
    "RightHip",      "LeftHip",
    "RightKnee",     "LeftKnee",
]
JOINT_LABELS = {
    "RightShoulder": "R.Shoulder",
    "LeftShoulder":  "L.Shoulder",
    "RightElbow":    "R.Elbow",
    "LeftElbow":     "L.Elbow",
    "RightHip":      "R.Hip",
    "LeftHip":       "L.Hip",
    "RightKnee":     "R.Knee",
    "LeftKnee":      "L.Knee",
}

SKT_COLOR = "#4C72B0"   # blue
MTL_COLOR = "#DD8452"   # orange


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_scenario(t: float) -> str:
    """Map a video-time timestamp (seconds) to a scenario label."""
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t <= end:
            return SCENARIO_MAPPING.get(label, "Unclassified")
    return "Unclassified"


def load_xsens_gt(mvnx_path: str):
    """Return (xsens_ts, gt_angle_interp) — timestamps normalised to start at 0."""
    mvnx = MvnxParser(mvnx_path)
    mvnx.parse()
    ts = mvnx.timestamps.copy()
    ts, xidx = np.unique(ts, return_index=True)
    ts -= ts[0]
    return ts, build_gt_angle_interpolators(mvnx, ts, xidx)


def angle_mae_per_frame(
    est_ts: np.ndarray,
    est_angles: np.ndarray,          # (N, n_joints)
    angle_names: list[str],
    gt_interp: dict,
    video_offset: float,
) -> pd.DataFrame:
    """Build a DataFrame with per-frame angle errors and scenario labels."""
    rows = []
    for i, t in enumerate(est_ts):
        gt_t   = t - video_offset
        scenario = get_scenario(t)
        for j, name in enumerate(angle_names):
            if name not in gt_interp:
                continue
            est_val = float(est_angles[i, j])
            gt_val  = float(gt_interp[name](gt_t))
            if not np.isfinite(est_val) or not np.isfinite(gt_val):
                continue
            rows.append({
                "AngleName": name,
                "Scenario":  scenario,
                "Time_s":    t,
                "Estimated": est_val,
                "GT":        gt_val,
                "Error":     abs(est_val - gt_val),
            })
    return pd.DataFrame(rows)


# ── Load Xsens GT ──────────────────────────────────────────────────────────────
print("[1/5] Loading Xsens GT …")
xsens_ts, gt_interp = load_xsens_gt(MVNX_PATH)


# ── SKT data ───────────────────────────────────────────────────────────────────
print("[2/5] Computing SKT (Dir A) per-frame angle errors …")
skt_data  = np.load(SKT_NPZ)
skt_kpts  = skt_data["keypoints"]   # (N, 17, 3)
skt_ts_raw = skt_data["timestamps"].copy()
skt_ts_raw -= skt_ts_raw[0]          # normalise to 0

# Filter out frames with invalid hip centre (same as 05_detailed_evaluation.py)
hip_centre = (skt_kpts[:, 11] + skt_kpts[:, 12]) / 2.0
valid = (
    (hip_centre[:, 2] > 10) &
    (hip_centre[:, 2] < 1000) &
    np.isfinite(hip_centre).all(axis=1)
)
skt_kpts_v = skt_kpts[valid]
skt_ts_v   = skt_ts_raw[valid]
skt_ts_v, uidx = np.unique(skt_ts_v, return_index=True)
skt_kpts_v = skt_kpts_v[uidx]

# Compute angles (uncalibrated — matches relative per-joint ranking)
# Returns: (list_of_names, ndarray shape (N, n_angles))
skt_angle_names, skt_angle_vals = compute_semantic_angle_sequence(skt_kpts_v)
skt_angle_vals = median_filter_angle_sequence(skt_angle_vals, radius=DEFAULT_ANGLE_SMOOTH_RADIUS)

skt_df = angle_mae_per_frame(skt_ts_v, skt_angle_vals, skt_angle_names, gt_interp, SKT_OFFSET)
print(f"   SKT valid samples: {len(skt_df)}")
print(f"   SKT overall MAE (uncalibrated): {skt_df['Error'].mean():.2f}°")


# ── MTL data ───────────────────────────────────────────────────────────────────
print("[3/5] Computing MTL (Dir C) per-frame angle errors …")
mtl_data  = np.load(MTL_NPZ, allow_pickle=True)
mtl_ts    = mtl_data["timestamps"].copy()      # video time (already normalised)
mtl_anames = [str(x) for x in mtl_data["angle_names"]]
mtl_avals  = mtl_data["angle_vals"]            # (N, 9)

# Exclude TrunkFlex from the 8-joint comparison
mtl_8joint_mask = [n != "TrunkFlex" for n in mtl_anames]
mtl_8names = [n for n in mtl_anames if n != "TrunkFlex"]
mtl_8vals  = mtl_avals[:, mtl_8joint_mask]

mtl_df = angle_mae_per_frame(mtl_ts, mtl_8vals, mtl_8names, gt_interp, MTL_OFFSET)
print(f"   MTL valid samples: {len(mtl_df)}")
print(f"   MTL overall MAE: {mtl_df['Error'].mean():.2f}°")


# ── Aggregate summaries ────────────────────────────────────────────────────────
print("[4/5] Aggregating …")

def summarise_by_joint(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("AngleName")["Error"]
        .agg(MAE="mean", Median="median", Std="std", N="count")
        .reindex(JOINT_DISPLAY_ORDER)
        .reset_index()
    )

def summarise_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    s = (
        df.groupby("Scenario")["Error"]
        .agg(MAE="mean", Median="median", Std="std", N="count")
        .reindex(SCENARIO_ORDER)
        .dropna(how="all")
        .reset_index()
    )
    return s

skt_joint_df = summarise_by_joint(skt_df)
mtl_joint_df = summarise_by_joint(mtl_df)
skt_scen_df  = summarise_by_scenario(skt_df)
mtl_scen_df  = summarise_by_scenario(mtl_df)

# Per-joint NaN rates for SKT (all frames, not just valid hip ones)
COCO_NAMES = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder",
              "LElbow","RElbow","LWrist","RWrist","LHip","RHip",
              "LKnee","RKnee","LAnkle","RAnkle"]
skt_all_kpts = np.load(SKT_NPZ)["keypoints"]
nan_rates = {}
for i, name in enumerate(COCO_NAMES):
    nan_rates[name] = np.mean(~np.isfinite(skt_all_kpts[:, i, :]).all(axis=1)) * 100


# ── Plotting ───────────────────────────────────────────────────────────────────
print("[5/5] Generating charts …")

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "figure.dpi":   150,
})

# ── Chart 1: Per-joint MAE comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(JOINT_DISPLAY_ORDER))
width  = 0.38
xlabels = [JOINT_LABELS[n] for n in JOINT_DISPLAY_ORDER]

skt_vals = skt_joint_df["MAE"].values
mtl_vals = mtl_joint_df["MAE"].values

bars1 = ax.bar(x - width/2, skt_vals, width, label="SKT — Dir A (uncalibrated)", color=SKT_COLOR, alpha=0.88, edgecolor="white")
bars2 = ax.bar(x + width/2, mtl_vals, width, label="MTL — Dir C",                color=MTL_COLOR, alpha=0.88, edgecolor="white")

# Value labels on bars
for bar, val in zip(bars1, skt_vals):
    if np.isfinite(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}°", ha="center", va="bottom", fontsize=8.5, color=SKT_COLOR)
for bar, val in zip(bars2, mtl_vals):
    if np.isfinite(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}°", ha="center", va="bottom", fontsize=8.5, color=MTL_COLOR)

ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=20, ha="right")
ax.set_ylabel("Angle MAE (°)")
ax.set_title("Per-Joint Angle MAE: SKT vs MTL", fontweight="bold", pad=12)
ax.legend(loc="upper left")
ax.axhline(skt_df["Error"].mean(), color=SKT_COLOR, linestyle="--", linewidth=1,
           alpha=0.6, label=f"SKT overall {skt_df['Error'].mean():.1f}°")
ax.axhline(mtl_df["Error"].mean(), color=MTL_COLOR, linestyle="--", linewidth=1,
           alpha=0.6, label=f"MTL overall {mtl_df['Error'].mean():.1f}°")
ax.set_ylim(0, max(skt_vals.max(), mtl_vals.max()) * 1.18)
fig.tight_layout()
fig.savefig(OUT_DIR / "joint_mae_comparison.png", bbox_inches="tight")
plt.close(fig)
print("   Saved: joint_mae_comparison.png")


# ── Chart 2: Per-scenario MAE — SKT ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [SCENARIO_COLORS.get(s, "#999") for s in skt_scen_df["Scenario"]]
bars = ax.barh(skt_scen_df["Scenario"], skt_scen_df["MAE"], color=colors, alpha=0.88, edgecolor="white", height=0.55)
for bar, row in zip(bars, skt_scen_df.itertuples()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{row.MAE:.1f}°  (n={int(row.N)})",
            va="center", fontsize=9.5)
ax.set_xlabel("Angle MAE (°)")
ax.set_title("SKT (Dir A) — Angle MAE by Scenario\n(uncalibrated, 8-joint set)",
             fontweight="bold", pad=10)
ax.axvline(skt_df["Error"].mean(), color="grey", linestyle="--", linewidth=1.2,
           label=f"Overall mean {skt_df['Error'].mean():.1f}°")
ax.legend(fontsize=9)
ax.set_xlim(0, skt_scen_df["MAE"].max() * 1.30)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT_DIR / "scenario_mae_skt.png", bbox_inches="tight")
plt.close(fig)
print("   Saved: scenario_mae_skt.png")


# ── Chart 3: Per-scenario MAE — MTL ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [SCENARIO_COLORS.get(s, "#999") for s in mtl_scen_df["Scenario"]]
bars = ax.barh(mtl_scen_df["Scenario"], mtl_scen_df["MAE"], color=colors, alpha=0.88, edgecolor="white", height=0.55)
for bar, row in zip(bars, mtl_scen_df.itertuples()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{row.MAE:.1f}°  (n={int(row.N)})",
            va="center", fontsize=9.5)
ax.set_xlabel("Angle MAE (°)")
ax.set_title("MTL (Dir C) — Angle MAE by Scenario\n(8-joint set)",
             fontweight="bold", pad=10)
ax.axvline(mtl_df["Error"].mean(), color="grey", linestyle="--", linewidth=1.2,
           label=f"Overall mean {mtl_df['Error'].mean():.1f}°")
ax.legend(fontsize=9)
ax.set_xlim(0, mtl_scen_df["MAE"].max() * 1.30)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT_DIR / "scenario_mae_mtl.png", bbox_inches="tight")
plt.close(fig)
print("   Saved: scenario_mae_mtl.png")


# ── Chart 4: SKT joint NaN / coverage rate ────────────────────────────────────
# Show only ergonomically relevant keypoints
ergo_kpts = {
    "LShoulder": "conf_left",  # will use nan_rates
    "RShoulder": None, "LElbow": None, "RElbow": None,
    "LHip": None, "RHip": None, "LKnee": None, "RKnee": None,
    "LAnkle": None, "RAnkle": None,
}
coco_map = {  # COCO name → display label
    "LShoulder": "L.Shoulder", "RShoulder": "R.Shoulder",
    "LElbow":    "L.Elbow",    "RElbow":    "R.Elbow",
    "LHip":      "L.Hip",      "RHip":      "R.Hip",
    "LKnee":     "L.Knee",     "RKnee":     "R.Knee",
    "LAnkle":    "L.Ankle",    "RAnkle":    "R.Ankle",
}
coco_keys = list(coco_map.keys())
nan_vals  = [nan_rates.get(k, 0.0) for k in coco_keys]
coverage  = [100.0 - v for v in nan_vals]

fig, ax = plt.subplots(figsize=(9, 4))
xc = np.arange(len(coco_keys))
bar_skt = ax.bar(xc - 0.2, coverage, 0.38, label="SKT coverage %", color=SKT_COLOR, alpha=0.82, edgecolor="white")
ax.axhline(100, color=MTL_COLOR, linewidth=1.5, linestyle="--", label="MTL coverage 100% (always)")
for bar, val in zip(bar_skt, coverage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5, color=SKT_COLOR)
ax.set_xticks(xc)
ax.set_xticklabels([coco_map[k] for k in coco_keys], rotation=22, ha="right")
ax.set_ylabel("Frame coverage (%)")
ax.set_ylim(0, 108)
ax.set_title("Joint Coverage Rate: SKT vs MTL\n(100% = no missing detections in any frame)",
             fontweight="bold", pad=10)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "joint_coverage_skt.png", bbox_inches="tight")
plt.close(fig)
print("   Saved: joint_coverage_skt.png")


# ── Chart 5: Error distribution boxplot per joint ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, df, color, title in [
    (axes[0], skt_df, SKT_COLOR, f"SKT (Dir A) — overall {skt_df['Error'].mean():.1f}° uncal."),
    (axes[1], mtl_df, MTL_COLOR, f"MTL (Dir C) — overall {mtl_df['Error'].mean():.1f}°"),
]:
    data_by_joint = [
        df.loc[df["AngleName"] == jn, "Error"].dropna().values
        for jn in JOINT_DISPLAY_ORDER
    ]
    bp = ax.boxplot(
        data_by_joint,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        widths=0.55,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks(range(1, len(JOINT_DISPLAY_ORDER) + 1))
    ax.set_xticklabels([JOINT_LABELS[n] for n in JOINT_DISPLAY_ORDER], rotation=22, ha="right")
    ax.set_ylabel("Angle error (°)")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.axhline(df["Error"].mean(), color=color, linewidth=1.2, linestyle="--", alpha=0.7)
fig.suptitle("Angle Error Distribution by Joint", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "error_dist_comparison.png", bbox_inches="tight")
plt.close(fig)
print("   Saved: error_dist_comparison.png")


# ── Chart 6: 2×2 Overview summary (report-ready) ──────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# Top-left: per-joint comparison
ax_jt = fig.add_subplot(gs[0, 0])
x     = np.arange(len(JOINT_DISPLAY_ORDER))
w     = 0.38
ax_jt.bar(x - w/2, skt_joint_df["MAE"].values, w, label="SKT", color=SKT_COLOR, alpha=0.85, edgecolor="white")
ax_jt.bar(x + w/2, mtl_joint_df["MAE"].values, w, label="MTL", color=MTL_COLOR, alpha=0.85, edgecolor="white")
ax_jt.set_xticks(x)
ax_jt.set_xticklabels([JOINT_LABELS[n] for n in JOINT_DISPLAY_ORDER], rotation=28, ha="right", fontsize=9)
ax_jt.set_ylabel("MAE (°)", fontsize=10)
ax_jt.set_title("(a) Per-Joint Angle MAE", fontweight="bold")
ax_jt.legend(fontsize=9)
ax_jt.axhline(skt_df["Error"].mean(), color=SKT_COLOR, linestyle=":", linewidth=1, alpha=0.8)
ax_jt.axhline(mtl_df["Error"].mean(), color=MTL_COLOR, linestyle=":", linewidth=1, alpha=0.8)
ax_jt.spines["top"].set_visible(False); ax_jt.spines["right"].set_visible(False)

# Top-right: per-scenario comparison (grouped)
ax_sc = fig.add_subplot(gs[0, 1])
sc_both = pd.merge(
    skt_scen_df[["Scenario", "MAE"]].rename(columns={"MAE": "SKT"}),
    mtl_scen_df[["Scenario", "MAE"]].rename(columns={"MAE": "MTL"}),
    on="Scenario", how="outer",
).set_index("Scenario").reindex(SCENARIO_ORDER).dropna(how="all")
xs    = np.arange(len(sc_both))
w2    = 0.38
ax_sc.bar(xs - w2/2, sc_both["SKT"].values, w2, label="SKT", color=SKT_COLOR, alpha=0.85, edgecolor="white")
ax_sc.bar(xs + w2/2, sc_both["MTL"].values, w2, label="MTL", color=MTL_COLOR, alpha=0.85, edgecolor="white")
ax_sc.set_xticks(xs)
ax_sc.set_xticklabels(sc_both.index, rotation=22, ha="right", fontsize=9)
ax_sc.set_ylabel("MAE (°)", fontsize=10)
ax_sc.set_title("(b) Per-Scenario Angle MAE", fontweight="bold")
ax_sc.legend(fontsize=9)
ax_sc.spines["top"].set_visible(False); ax_sc.spines["right"].set_visible(False)

# Bottom-left: joint coverage
ax_cv = fig.add_subplot(gs[1, 0])
ax_cv.bar(np.arange(len(coco_keys)), coverage, 0.6, color=SKT_COLOR, alpha=0.82, edgecolor="white", label="SKT")
ax_cv.axhline(100, color=MTL_COLOR, linewidth=1.8, linestyle="--", label="MTL (always 100%)")
ax_cv.set_xticks(range(len(coco_keys)))
ax_cv.set_xticklabels([coco_map[k] for k in coco_keys], rotation=28, ha="right", fontsize=9)
ax_cv.set_ylim(0, 108)
ax_cv.set_ylabel("Coverage (%)", fontsize=10)
ax_cv.set_title("(c) Joint Frame Coverage", fontweight="bold")
ax_cv.legend(fontsize=9)
ax_cv.spines["top"].set_visible(False); ax_cv.spines["right"].set_visible(False)

# Bottom-right: overall comparison summary text
ax_tx = fig.add_subplot(gs[1, 1])
ax_tx.axis("off")
summary_lines = [
    "Summary — Key Metrics",
    "",
    f"SKT (Dir A, uncalibrated)",
    f"  Overall MAE:  {skt_df['Error'].mean():.2f}°",
    f"  Calibrated:   13.21°",
    f"  MPJPE:        26.02 cm",
    f"  Best joint:   {skt_joint_df.loc[skt_joint_df['MAE'].idxmin(), 'AngleName']}  "
    f"({skt_joint_df['MAE'].min():.1f}°)",
    f"  Worst joint:  {skt_joint_df.loc[skt_joint_df['MAE'].idxmax(), 'AngleName']}  "
    f"({skt_joint_df['MAE'].max():.1f}°)",
    f"  Best scenario:  {skt_scen_df.loc[skt_scen_df['MAE'].idxmin(), 'Scenario']}",
    f"  Worst scenario: {skt_scen_df.loc[skt_scen_df['MAE'].idxmax(), 'Scenario']}",
    "",
    f"MTL (Dir C)",
    f"  Overall MAE:  {mtl_df['Error'].mean():.2f}°",
    f"  MPJPE:        218 cm (Kabsch-aligned)",
    f"  Best joint:   {mtl_joint_df.loc[mtl_joint_df['MAE'].idxmin(), 'AngleName']}  "
    f"({mtl_joint_df['MAE'].min():.1f}°)",
    f"  Worst joint:  {mtl_joint_df.loc[mtl_joint_df['MAE'].idxmax(), 'AngleName']}  "
    f"({mtl_joint_df['MAE'].max():.1f}°)",
    f"  Best scenario:  {mtl_scen_df.loc[mtl_scen_df['MAE'].idxmin(), 'Scenario']}",
    f"  Worst scenario: {mtl_scen_df.loc[mtl_scen_df['MAE'].idxmax(), 'Scenario']}",
]
ax_tx.text(0.05, 0.97, "\n".join(summary_lines),
           transform=ax_tx.transAxes, va="top", ha="left",
           fontsize=10.5, fontfamily="monospace",
           bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC"})

fig.suptitle("Joint & Scenario Analysis — SKT vs MTL\n(Ergonomic Pose Estimation Pipeline Comparison)",
             fontsize=13, fontweight="bold", y=1.02)
fig.savefig(OUT_DIR / "overview_summary.png", bbox_inches="tight", dpi=160)
plt.close(fig)
print("   Saved: overview_summary.png")


# ── Save CSVs ─────────────────────────────────────────────────────────────────
skt_joint_df.to_csv(OUT_DIR / "skt_per_joint_mae.csv", index=False)
mtl_joint_df.to_csv(OUT_DIR / "mtl_per_joint_mae.csv", index=False)
skt_scen_df.to_csv(OUT_DIR / "skt_per_scenario_mae.csv", index=False)
mtl_scen_df.to_csv(OUT_DIR / "mtl_per_scenario_mae.csv", index=False)
print("   Saved: *.csv")

print(f"\nDone. All outputs in: {OUT_DIR}/")
print(f"\nSKT uncalibrated overall:  {skt_df['Error'].mean():.2f}°")
print(f"MTL overall:               {mtl_df['Error'].mean():.2f}°")
print("\nSKT per-joint MAE:")
print(skt_joint_df[["AngleName","MAE","N"]].to_string(index=False))
print("\nMTL per-joint MAE:")
print(mtl_joint_df[["AngleName","MAE","N"]].to_string(index=False))
print("\nSKT per-scenario:")
print(skt_scen_df.to_string(index=False))
print("\nMTL per-scenario:")
print(mtl_scen_df.to_string(index=False))
