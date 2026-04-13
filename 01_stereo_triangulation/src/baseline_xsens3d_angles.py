#!/usr/bin/env python3
"""
baseline_xsens3d_angles.py
==========================
Baseline Test 2: Feed Xsens ground-truth 3D segment positions directly into
our geometric angle calculation pipeline and compare with Xsens GT angles.

This measures the "angle definition gap" (②) — the irreducible error caused by
the difference between Xsens joint conventions and our COCO-skeleton geometry:

    total_error (13.2°) = ① 3D reconstruction error  +  ② angle definition gap

If ② is large, no improvement in 3D reconstruction can close the gap.

Xsens segment → COCO keypoint mapping:
  Xsens "RightUpperArm"  (origin = glenohumeral joint)  → COCO idx 6  (RShoulder)
  Xsens "RightForeArm"   (origin = elbow joint)         → COCO idx 8  (RElbow)
  Xsens "RightHand"      (origin = wrist joint)         → COCO idx 10 (RWrist)
  Xsens "LeftUpperArm"                                  → COCO idx 5  (LShoulder)
  Xsens "LeftForeArm"                                   → COCO idx 7  (LElbow)
  Xsens "LeftHand"                                      → COCO idx 9  (LWrist)
  Xsens "RightUpperLeg"  (origin = hip joint)           → COCO idx 12 (RHip)
  Xsens "RightLowerLeg"  (origin = knee joint)          → COCO idx 14 (RKnee)
  Xsens "RightFoot"      (origin = ankle joint)         → COCO idx 16 (RAnkle)
  Xsens "LeftUpperLeg"                                  → COCO idx 11 (LHip)
  Xsens "LeftLowerLeg"                                  → COCO idx 13 (LKnee)
  Xsens "LeftFoot"                                      → COCO idx 15 (LAnkle)

Usage:
    cd 01_stereo_triangulation
    /opt/anaconda3/envs/pose/bin/python src/baseline_xsens3d_angles.py
"""

import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR   = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(METHOD_DIR)
SHARED_DIR   = os.path.join(PROJECT_ROOT, "shared")
RESULTS_DIR  = os.path.join(METHOD_DIR, "results")
MVNX_PATH    = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")

sys.path.insert(0, SHARED_DIR)

from utils_mvnx import MvnxParser
from pose_angle_utils import (
    build_gt_angle_interpolators,
    compute_semantic_joint_angles,
    SEMANTIC_ANGLE_NAMES,
)

# ── Xsens segment → COCO 17-keypoint index ────────────────────────────────────
# Segment origin = proximal joint center, which maps to the corresponding COCO joint.
# Indices follow the COCO 17-keypoint convention used in pose_postprocess.py.
XSENS_TO_COCO: dict[str, int] = {
    "LeftUpperArm":  5,   # LShoulder (glenohumeral joint)
    "RightUpperArm": 6,   # RShoulder
    "LeftForeArm":   7,   # LElbow
    "RightForeArm":  8,   # RElbow
    "LeftHand":      9,   # LWrist
    "RightHand":     10,  # RWrist
    "LeftUpperLeg":  11,  # LHip
    "RightUpperLeg": 12,  # RHip
    "LeftLowerLeg":  13,  # LKnee
    "RightLowerLeg": 14,  # RKnee
    "LeftFoot":      15,  # LAnkle
    "RightFoot":     16,  # RAnkle
}


def build_xsens_coco_poses(mvnx: MvnxParser) -> np.ndarray:
    """Build (N_frames, 17, 3) array of pseudo-COCO poses from Xsens segments.

    Joints 0-4 (nose/eyes/ears) are set to NaN — not used in angle calculation.
    All positions are in cm (matching Xsens parser output).
    """
    n_frames = mvnx.data.shape[0]
    poses = np.full((n_frames, 17, 3), np.nan, dtype=np.float64)

    for seg_name, coco_idx in XSENS_TO_COCO.items():
        seg_data = mvnx.get_segment_data(seg_name)
        if seg_data is None:
            print(f"  [Warning] Segment '{seg_name}' not found in MVNX.")
            continue
        poses[:, coco_idx, :] = seg_data

    return poses


def compute_angles_from_poses(poses: np.ndarray) -> dict[str, np.ndarray]:
    """Run compute_semantic_joint_angles on every frame.

    Returns dict: angle_name → (N_frames,) array of degrees.
    """
    n = len(poses)
    result: dict[str, np.ndarray] = {name: np.full(n, np.nan) for name in SEMANTIC_ANGLE_NAMES}

    for i, pose in enumerate(poses):
        frame_angles = compute_semantic_joint_angles(pose)
        for name, val in frame_angles.items():
            result[name][i] = val

    return result


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Parse MVNX ─────────────────────────────────────────────────────────
    print("\n[Step 1] Parsing MVNX ...")
    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()

    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]

    # ── 2. Build GT angle interpolators (Xsens native angles) ─────────────────
    print("\n[Step 2] Building GT angle interpolators ...")
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, xidx)
    print(f"  GT joints available: {list(gt_interps.keys())}")

    # ── 3. Build pseudo-COCO poses from Xsens segments ────────────────────────
    print("\n[Step 3] Mapping Xsens segments → COCO poses ...")
    all_poses = build_xsens_coco_poses(mvnx)
    # Apply same unique-frame indexing as GT
    all_poses = all_poses[xidx]
    n_frames = len(xsens_ts)
    print(f"  Frames: {n_frames}  |  Pose shape: {all_poses.shape}")

    # Check mapping coverage
    for seg, idx in XSENS_TO_COCO.items():
        valid = np.isfinite(all_poses[:, idx, 0]).sum()
        print(f"  {seg:20s} → idx {idx:2d}  valid={valid}/{n_frames}")

    # ── 4. Compute geometric angles from Xsens 3D ─────────────────────────────
    print("\n[Step 4] Computing geometric angles from Xsens 3D positions ...")
    geom_angles = compute_angles_from_poses(all_poses)

    for name, arr in geom_angles.items():
        valid = np.isfinite(arr).sum()
        print(f"  {name:15s}  valid={valid}/{n_frames}  "
              f"mean={np.nanmean(arr):.1f}°  std={np.nanstd(arr):.1f}°")

    # ── 5. Compare with GT (Xsens native angles) ──────────────────────────────
    print("\n[Step 5] Computing MAE vs Xsens GT ...")

    mae_results: dict[str, dict] = {}
    output_rows = []

    for name in SEMANTIC_ANGLE_NAMES:
        if name not in gt_interps:
            print(f"  [Skip] No GT interpolator for {name}")
            continue

        gt_vals  = gt_interps[name](xsens_ts)          # GT on Xsens timeline
        geo_vals = geom_angles[name]                    # our calculation

        valid = np.isfinite(gt_vals) & np.isfinite(geo_vals)
        n_valid = valid.sum()
        if n_valid < 10:
            print(f"  {name:15s}  insufficient valid samples ({n_valid})")
            continue

        mae    = float(np.mean(np.abs(gt_vals[valid] - geo_vals[valid])))
        median = float(np.median(np.abs(gt_vals[valid] - geo_vals[valid])))
        bias   = float(np.mean(geo_vals[valid] - gt_vals[valid]))
        std    = float(np.std(geo_vals[valid] - gt_vals[valid]))

        mae_results[name] = {
            "MAE": mae, "Median": median, "Bias": bias, "Std": std, "N": int(n_valid)
        }
        output_rows.append(f"  {name:15s}  MAE={mae:5.2f}°  Median={median:5.2f}°  "
                           f"Bias={bias:+6.2f}°  Std={std:5.2f}°  N={n_valid}")
        print(output_rows[-1])

    if mae_results:
        mean_mae = float(np.mean([v["MAE"] for v in mae_results.values()]))
        print(f"\n  → Mean MAE across all joints: {mean_mae:.2f}°")
        print(f"  → This is the irreducible angle-definition gap (②)")
        print(f"  → Our pipeline's total error (①+②) = 13.2°")
        if mean_mae < 13.2:
            print(f"  → 3D reconstruction error (①) ≈ {13.2 - mean_mae:.2f}°")

    # ── 6. Save JSON summary ───────────────────────────────────────────────────
    summary = {
        "description": "Xsens GT 3D positions → our angle calculator vs Xsens GT angles",
        "interpretation": "MAE here = irreducible angle-definition gap (②). "
                          "Pipeline total error 13.2° = ① + ②.",
        "mean_mae_deg": float(np.mean([v["MAE"] for v in mae_results.values()])) if mae_results else None,
        "per_joint": mae_results,
    }
    json_path = os.path.join(RESULTS_DIR, "baseline_xsens3d_angles.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {json_path}")

    # ── Save fair GT angles to shared/ for reuse across all Directions ─────────
    fair_gt_path = os.path.join(SHARED_DIR, "fair_gt_angles.npz")
    np.savez(
        fair_gt_path,
        timestamps=xsens_ts,
        **{name: geom_angles[name] for name in SEMANTIC_ANGLE_NAMES},
    )
    print(f"[Saved] Fair GT angles → {fair_gt_path}")

    # ── 7. Plot: time series comparison for each joint ─────────────────────────
    print("\n[Step 6] Generating plots ...")
    joint_list = [j for j in SEMANTIC_ANGLE_NAMES if j in mae_results]
    n_joints = len(joint_list)
    if n_joints == 0:
        print("  No valid joints to plot.")
        return

    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 2.8 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    t = xsens_ts
    for ax, name in zip(axes, joint_list):
        gt   = gt_interps[name](t)
        geom = geom_angles[name]
        mae  = mae_results[name]["MAE"]
        bias = mae_results[name]["Bias"]

        ax.plot(t, gt,   color="#1565C0", lw=1.2, label="Xsens GT (native)")
        ax.plot(t, geom, color="#E53935", lw=0.9, alpha=0.85,
                label=f"Our calc from Xsens 3D  (MAE={mae:.2f}°, bias={bias:+.2f}°)")
        ax.set_ylabel("Degrees", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Baseline Test 2: Xsens 3D → Our Angle Calculator vs Xsens GT\n"
        f"Mean MAE = {summary['mean_mae_deg']:.2f}°  "
        f"(= irreducible angle-definition gap ②)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "baseline_xsens3d_angles.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {plot_path}")

    # ── 8. Bar chart: MAE per joint ────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    names  = list(mae_results.keys())
    maes   = [mae_results[n]["MAE"] for n in names]
    colors = ["#EF5350" if m > 5 else "#42A5F5" for m in maes]

    bars = ax2.bar(names, maes, color=colors, edgecolor="white")
    ax2.axhline(summary["mean_mae_deg"], color="black", linestyle="--", lw=1.5,
                label=f"Mean = {summary['mean_mae_deg']:.2f}°")
    ax2.axhline(13.2, color="#FF6F00", linestyle=":", lw=1.5,
                label="Pipeline total error = 13.2°")
    for bar, mae in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{mae:.1f}°", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("MAE (degrees)")
    ax2.set_title("Baseline Test 2: Angle Definition Gap (②) per Joint\n"
                  "Blue = within tolerance (<5°), Red = significant gap (>5°)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(maes) * 1.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, "baseline_xsens3d_angles_bar.png")
    fig2.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Saved] {bar_path}")

    print("\n[Done] Baseline Test 2 complete.")


if __name__ == "__main__":
    main()
