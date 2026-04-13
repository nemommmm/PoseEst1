"""
Out-of-sample calibration evaluation — leave-one-scenario-out cross-validation.

For each fold, calibration is fit on two scenarios and evaluated on the held-out
third. This gives an honest estimate of calibration generalization.

Usage:
    cd 01_stereo_triangulation/src
    /opt/anaconda3/envs/pose/bin/python oos_calibration_eval.py
"""

import os
import sys
import numpy as np
from scipy.interpolate import interp1d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))
sys.path.insert(0, os.path.dirname(__file__))

from pose_angle_utils import (
    SEMANTIC_ANGLE_NAMES,
    fit_piecewise_calibration,
    apply_piecewise_calibration,
    build_gt_angle_interpolators,
    build_fair_gt_interpolators,
    median_filter_angle_sequence,
    compute_semantic_angle_sequence,
)
from utils_mvnx import MvnxParser

# ── Config ──────────────────────────────────────────────────────────────────
POSE_NPZ = os.path.join(
    PROJECT_ROOT,
    "01_stereo_triangulation", "results",
    "historical_best_20260324", "recovered_baseline", "optimized_pose.npz",
)
GT_MVNX = os.path.join(
    PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx"
)
FAIR_GT_NPZ = os.path.join(PROJECT_ROOT, "shared", "fair_gt_angles.npz")
TEMPORAL_OFFSET = 17.40   # seconds: pose t=0 → Xsens t=17.40
SMOOTH_RADIUS   = int(os.environ.get("POSE_ANGLE_SMOOTH_RADIUS", "8"))
CAL_BINS        = int(os.environ.get("POSE_CALIBRATION_BINS", "10"))

# Activity segments (seconds, from 05_detailed_evaluation.py)
ACTIVITY_SEGMENTS = {
    "Walking (Normal)":           [17,  32],
    "Walking (Late)":             [220, 240],
    "Sitting (Lower Occluded)":   [32,  62],
    "Walking (Upper Occluded)":   [87,  97],
    "Walking (Lower Occluded 1)": [130, 140],
    "Walking (Lower Occluded 2)": [164, 170],
    "Chair Interaction (Complex)":[140, 160],
    "Lifting Box (Near Chair)":   [214, 218],
}
SCENARIO_MAPPING = {
    "Walking (Normal)":            "Baseline",
    "Walking (Late)":              "Baseline",
    "Sitting (Lower Occluded)":    "Occlusion",
    "Walking (Upper Occluded)":    "Occlusion",
    "Walking (Lower Occluded 1)":  "Occlusion",
    "Walking (Lower Occluded 2)":  "Occlusion",
    "Chair Interaction (Complex)": "Environmental Interference",
    "Lifting Box (Near Chair)":    "Environmental Interference",
}

CV_SCENARIOS = ["Baseline", "Occlusion", "Environmental Interference"]


def get_scenario(t: float) -> str | None:
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t < end:
            return SCENARIO_MAPPING.get(label)
    return None


def load_pose_angles(npz_path: str, smooth_radius: int) -> tuple:
    """Return (timestamps_pose in seconds from start, angle_matrix [N x n_angles])."""
    data = np.load(npz_path)
    ts   = data["timestamps"].astype(float)
    ts   = ts - ts[0]   # normalize to seconds from video start
    kpts = data["keypoints"]  # (N, 17, 3)
    angle_names, angle_vals = compute_semantic_angle_sequence(kpts)
    # angle_vals: (N, n_angles), angle_names: list of names
    # Reorder to SEMANTIC_ANGLE_NAMES
    name_to_idx = {n: i for i, n in enumerate(angle_names)}
    angles = np.stack([
        median_filter_angle_sequence(angle_vals[:, name_to_idx[n]], radius=smooth_radius)
        for n in SEMANTIC_ANGLE_NAMES
    ], axis=1)   # (N, 8)
    return ts, angles


def main() -> None:
    print(f"SR={SMOOTH_RADIUS}  bins={CAL_BINS}")
    print(f"Pose:    {POSE_NPZ}")

    # ── 1. Load pose estimates ───────────────────────────────────────────────
    ts_pose, ang_est = load_pose_angles(POSE_NPZ, SMOOTH_RADIUS)
    # Shift to Xsens time
    ts_xsens_eq = ts_pose + TEMPORAL_OFFSET

    # ── 2. Load GT interpolators ─────────────────────────────────────────────
    mvnx     = MvnxParser(GT_MVNX)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps   = build_gt_angle_interpolators(mvnx, xsens_ts, xidx)
    fair_interps = build_fair_gt_interpolators(FAIR_GT_NPZ)
    if not gt_interps:
        raise RuntimeError("GT interpolators empty — check MVNX path")
    print(f"GT loaded ({len(gt_interps)} joints)  Fair GT: {bool(fair_interps)}")

    # ── 3. Build frame-level records ─────────────────────────────────────────
    # Each record: est angles, gt angles, fair_gt angles, scenario label
    records = []
    for i, (t_x, est_row) in enumerate(zip(ts_xsens_eq, ang_est)):
        scenario = get_scenario(ts_pose[i])
        if scenario is None:
            continue
        gt_row   = np.array([gt_interps[n](t_x)   if n in gt_interps   else np.nan
                             for n in SEMANTIC_ANGLE_NAMES])
        fair_row = np.array([fair_interps[n](t_x)  if n in fair_interps else np.nan
                             for n in SEMANTIC_ANGLE_NAMES])
        if not np.any(np.isfinite(gt_row)):
            continue
        records.append({
            "scenario": scenario,
            "est":      est_row,
            "gt":       gt_row,
            "fair_gt":  fair_row,
        })

    print(f"Scenario frames: {len(records)}")
    scenario_counts = {s: sum(1 for r in records if r["scenario"] == s)
                       for s in CV_SCENARIOS}
    for s, n in scenario_counts.items():
        print(f"  {s}: {n} frames")

    # ── 4. Leave-one-scenario-out CV ─────────────────────────────────────────
    all_oos_errors_e2e  = []   # end-to-end (vs xsens native)
    all_oos_errors_fair = []   # fair GT (vs same formula)

    print("\n── Leave-one-scenario-out results ──────────────────────────────")
    for held_out in CV_SCENARIOS:
        train_recs = [r for r in records if r["scenario"] != held_out]
        test_recs  = [r for r in records if r["scenario"] == held_out]

        # Build train arrays per angle
        calibrations = {}
        for j, name in enumerate(SEMANTIC_ANGLE_NAMES):
            est_col = np.array([r["est"][j] for r in train_recs])
            gt_col  = np.array([r["gt"][j]  for r in train_recs])
            fin     = np.isfinite(est_col) & np.isfinite(gt_col)
            if fin.sum() < CAL_BINS * 2:
                continue
            calibrations[name] = fit_piecewise_calibration(
                est_col[fin], gt_col[fin], n_bins=CAL_BINS
            )

        # Evaluate on held-out scenario
        fold_e2e_errors  = []
        fold_fair_errors = []
        for r in test_recs:
            for j, name in enumerate(SEMANTIC_ANGLE_NAMES):
                est_raw = r["est"][j]
                gt_val  = r["gt"][j]
                fair_val= r["fair_gt"][j]
                if not (np.isfinite(est_raw) and np.isfinite(gt_val)):
                    continue
                # Apply calibration
                if name in calibrations:
                    est_cal = apply_piecewise_calibration(
                        np.array([est_raw]), calibrations[name]
                    )[0]
                else:
                    est_cal = est_raw

                fold_e2e_errors.append(abs(est_cal - gt_val))
                if np.isfinite(fair_val):
                    fold_fair_errors.append(abs(est_cal - fair_val))

        fold_mae_e2e  = np.mean(fold_e2e_errors)  if fold_e2e_errors  else np.nan
        fold_mae_fair = np.mean(fold_fair_errors) if fold_fair_errors else np.nan
        print(f"  Held-out [{held_out:30s}] "
              f"end-to-end {fold_mae_e2e:.2f}°  "
              f"fair GT {fold_mae_fair:.2f}°  "
              f"(n={len(test_recs)})")

        all_oos_errors_e2e.extend(fold_e2e_errors)
        all_oos_errors_fair.extend(fold_fair_errors)

    # ── 5. Overall OOS result ─────────────────────────────────────────────────
    oos_mae_e2e  = np.mean(all_oos_errors_e2e)
    oos_mae_fair = np.mean(all_oos_errors_fair)
    gap = oos_mae_e2e - oos_mae_fair

    print("\n── Overall out-of-sample MAE ───────────────────────────────────")
    print(f"  End-to-end (vs Xsens native):  {oos_mae_e2e:.2f}°")
    print(f"  Fair GT (pure 3D error ①):     {oos_mae_fair:.2f}°")
    print(f"  Definition gap ②:              {gap:+.2f}°")

    # ── 6. In-sample baseline for reference ──────────────────────────────────
    calibrations_full = {}
    for j, name in enumerate(SEMANTIC_ANGLE_NAMES):
        est_col = np.array([r["est"][j] for r in records])
        gt_col  = np.array([r["gt"][j]  for r in records])
        fin     = np.isfinite(est_col) & np.isfinite(gt_col)
        if fin.sum() < CAL_BINS * 2:
            continue
        calibrations_full[name] = fit_piecewise_calibration(
            est_col[fin], gt_col[fin], n_bins=CAL_BINS
        )

    insample_errors = []
    for r in records:
        for j, name in enumerate(SEMANTIC_ANGLE_NAMES):
            est_raw = r["est"][j];  gt_val = r["gt"][j]
            if not (np.isfinite(est_raw) and np.isfinite(gt_val)):
                continue
            est_cal = apply_piecewise_calibration(
                np.array([est_raw]), calibrations_full[name]
            )[0] if name in calibrations_full else est_raw
            insample_errors.append(abs(est_cal - gt_val))

    print(f"\n  (In-sample MAE for reference:  {np.mean(insample_errors):.2f}°)")
    print(f"  Overfitting gap (in→OOS):      "
          f"{oos_mae_e2e - np.mean(insample_errors):+.2f}°")


if __name__ == "__main__":
    main()
