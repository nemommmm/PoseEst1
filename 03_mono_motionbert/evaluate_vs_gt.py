"""
evaluate_vs_gt.py

Evaluates Aitor Iriondo's monocular pipeline output against the Xsens
motion capture ground-truth.

All joint angles are computed geometrically from TRC 3D markers, matching
the approach used in the stereo pipeline. OpenSim IK arm DOFs are unreliable
due to the Pose2Sim_Simple model's unconstrained rotation accumulation.

Input:
  MONO_TRC_PATH  — path to the TRC file (auto-detected if not set)

Outputs:
  results_mono/eval_angle_mono.png    — joint angle MAE bar chart vs Xsens GT
  results_mono/eval_rula_mono.png     — RULA timeline (estimated vs GT)
  results_mono/eval_results_mono.npz  — numeric results for further analysis
"""

import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SHARED_DIR   = os.path.join(PROJECT_ROOT, "shared")

MVNX_PATH   = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")
ALIGN_JSON  = os.environ.get(
    "MONO_ALIGNMENT_JSON",
    os.path.join(PROJECT_ROOT, "01_stereo_triangulation", "results", "alignment_summary.json"),
)
OUT_DIR     = os.path.join(SCRIPT_DIR, "results_mono")

DEFAULT_TRC = os.path.join(OUT_DIR, "markers_results_mono.trc")
TRC_PATH    = os.environ.get("MONO_TRC_PATH", DEFAULT_TRC)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BEST_OFFSET_DEFAULT = 17.20

ACTIVITY_SEGMENTS = {
    "Walking (Normal)":             [17, 32],
    "Walking (Late)":               [220, 240],
    "Sitting (Lower Occluded)":     [32, 62],
    "Walking (Upper Occluded)":     [87, 97],
    "Walking (Lower Occluded 1)":   [130, 140],
    "Walking (Lower Occluded 2)":   [164, 170],
    "Chair Interaction (Complex)":  [140, 160],
    "Lifting Box (Near Chair)":     [214, 218],
    "Squatting":                    [66, 69],
    "Squatting (Check)":            [156, 160],
}

# ---------------------------------------------------------------------------
# TRC parsing
# ---------------------------------------------------------------------------

def load_trc(trc_path: str) -> tuple:
    """Parse TRC file, return (timestamps, marker_names, positions (N,M,3))."""
    if not os.path.isfile(trc_path):
        raise FileNotFoundError(f"[eval] .trc file not found: {trc_path}")
    with open(trc_path) as f:
        lines = f.readlines()

    raw_names = lines[3].strip().split("\t")[2:]
    marker_names = [n for n in raw_names if n.strip()]
    data_lines = [l.strip() for l in lines[6:] if l.strip()]

    frames, timestamps = [], []
    for line in data_lines:
        vals = line.split("\t")
        timestamps.append(float(vals[1]))
        coords = [float(v) if v else np.nan for v in vals[2:]]
        frames.append(coords)

    arr = np.array(frames)
    n_markers = len(marker_names)
    positions = arr[:, :n_markers * 3].reshape(-1, n_markers, 3)
    print(f"[eval] .trc loaded: {len(timestamps)} frames, {n_markers} markers")
    return np.array(timestamps), marker_names, positions


# ---------------------------------------------------------------------------
# Geometric angle computation from 3D markers
# ---------------------------------------------------------------------------

def _vec_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two 3D vectors in degrees."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def _elbow_angle(shoulder: np.ndarray, elbow: np.ndarray,
                 wrist: np.ndarray) -> float:
    """Elbow flexion = angle at the elbow joint (180° - angle between
    upper arm and forearm vectors)."""
    upper = shoulder - elbow
    fore = wrist - elbow
    return _vec_angle_deg(upper, fore)


def _median_smooth_positions(positions, radius=3):
    """Apply temporal median filter to 3D marker positions."""
    if radius <= 0:
        return positions
    smoothed = positions.copy()
    n_frames = len(positions)
    for i in range(n_frames):
        lo = max(0, i - radius)
        hi = min(n_frames, i + radius + 1)
        window = positions[lo:hi]
        for m in range(positions.shape[1]):
            for ax in range(3):
                vals = window[:, m, ax]
                finite = vals[np.isfinite(vals)]
                if len(finite) > 0:
                    smoothed[i, m, ax] = np.median(finite)
    return smoothed


def compute_geometric_angles(marker_names, positions):
    """Compute all RULA-relevant angles geometrically from TRC markers.

    Applies median smoothing to 3D positions before angle computation
    to reduce MotionBERT noise.

    Returns dict of angle arrays (degrees):
      RightShoulder, LeftShoulder  — upper arm elevation (angle from torso-down)
      RightElbow, LeftElbow        — elbow flexion
      RightKnee, LeftKnee          — knee flexion
      TrunkFlex                    — trunk inclination from vertical
    """
    # Smooth 3D positions to reduce MotionBERT noise
    positions = _median_smooth_positions(positions, radius=3)

    idx = {n: i for i, n in enumerate(marker_names)}
    n = len(positions)

    angles = {
        "RightShoulder": np.full(n, np.nan),
        "LeftShoulder":  np.full(n, np.nan),
        "RightElbow":    np.full(n, np.nan),
        "LeftElbow":     np.full(n, np.nan),
        "RightKnee":     np.full(n, np.nan),
        "LeftKnee":      np.full(n, np.nan),
        "TrunkFlex":     np.full(n, np.nan),
    }

    for i in range(n):
        p = positions[i]

        r_shoulder = p[idx["RShoulder"]]
        l_shoulder = p[idx["LShoulder"]]
        r_elbow    = p[idx["RElbow"]]
        l_elbow    = p[idx["LElbow"]]
        r_wrist    = p[idx["RWrist"]]
        l_wrist    = p[idx["LWrist"]]
        r_hip      = p[idx["RHip"]]
        l_hip      = p[idx["LHip"]]
        r_knee     = p[idx["RKnee"]]
        l_knee     = p[idx["LKnee"]]
        r_ankle    = p[idx["RAnkle"]]
        l_ankle    = p[idx["LAnkle"]]

        hip_mid = 0.5 * (l_hip + r_hip)
        shoulder_mid = 0.5 * (l_shoulder + r_shoulder)
        torso_down = hip_mid - shoulder_mid

        # Shoulder elevation: angle between upper arm and torso-down vector
        angles["RightShoulder"][i] = _vec_angle_deg(r_elbow - r_shoulder, torso_down)
        angles["LeftShoulder"][i]  = _vec_angle_deg(l_elbow - l_shoulder, torso_down)

        # Elbow flexion: angle at elbow joint
        angles["RightElbow"][i] = _elbow_angle(r_shoulder, r_elbow, r_wrist)
        angles["LeftElbow"][i]  = _elbow_angle(l_shoulder, l_elbow, l_wrist)

        # Knee flexion: angle at knee joint
        angles["RightKnee"][i] = 180.0 - _vec_angle_deg(r_hip - r_knee, r_ankle - r_knee)
        angles["LeftKnee"][i]  = 180.0 - _vec_angle_deg(l_hip - l_knee, l_ankle - l_knee)

        # Trunk flexion: deviation from vertical (Y-up in OpenSim)
        vertical = np.array([0.0, 1.0, 0.0])
        angles["TrunkFlex"][i] = _vec_angle_deg(shoulder_mid - hip_mid, vertical)

    return angles


def best_offset() -> float:
    if os.path.isfile(ALIGN_JSON):
        with open(ALIGN_JSON) as f:
            return float(json.load(f).get("best_offset_seconds", BEST_OFFSET_DEFAULT))
    return BEST_OFFSET_DEFAULT


# ---------------------------------------------------------------------------
# RULA scoring (mirrors src/08_ergonomic_scoring.py)
# ---------------------------------------------------------------------------

def _rula_grand(shoulder, elbow, trunk, knee):
    if any(not np.isfinite(v) for v in [shoulder, elbow, trunk, knee]):
        return np.nan

    def ua(a): return 1 if abs(a)<=20 else 2 if abs(a)<=45 else 3 if abs(a)<=90 else 4
    def la(a): return 1 if 60<=abs(a)<=100 else 2
    def tr(a): return 1 if abs(a)<=10 else 2 if abs(a)<=20 else 3 if abs(a)<=60 else 4
    def lg(a): return 2 if abs(a)>30 else 1

    tA = {(1,1):1,(1,2):2,(2,1):2,(2,2):3,(3,1):3,(3,2):4,(4,1):4,(4,2):5}
    tB = {(1,1):1,(1,2):2,(2,1):2,(2,2):3,(3,1):3,(3,2):4,(4,1):4,(4,2):5}
    tC = {(1,1):1,(1,2):2,(1,3):3,(1,4):3,(1,5):4,
          (2,1):2,(2,2):2,(2,3):3,(2,4):4,(2,5):4,
          (3,1):3,(3,2):3,(3,3):3,(3,4):4,(3,5):5,
          (4,1):3,(4,2):4,(4,3):4,(4,4):5,(4,5):6,
          (5,1):4,(5,2):4,(5,3):5,(5,4):6,(5,5):7}

    sa = tA.get((ua(shoulder), la(elbow)), max(ua(shoulder), la(elbow)))
    sb = tB.get((tr(trunk),    lg(knee)),  max(tr(trunk),    lg(knee)))
    return float(tC.get((sa, sb), max(sa, sb)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if SHARED_DIR not in sys.path:
        sys.path.insert(0, SHARED_DIR)
    from utils_mvnx import MvnxParser
    from pose_angle_utils import build_gt_angle_interpolators

    offset = best_offset()
    print(f"[eval] Temporal offset: {offset:.2f} s")
    print(f"[eval] .trc path:       {TRC_PATH}")

    # ------------------------------------------------------------------ #
    # 1. Load TRC and compute geometric angles
    # ------------------------------------------------------------------ #
    trc_ts, marker_names, positions = load_trc(TRC_PATH)
    trc_ts = trc_ts - trc_ts[0]

    angles = compute_geometric_angles(marker_names, positions)

    for k, v in angles.items():
        fin = v[np.isfinite(v)]
        if len(fin) > 0:
            print(f"[eval] {k:<22s}: range [{fin.min():.1f}, {fin.max():.1f}]°  "
                  f"mean={fin.mean():.1f}°")

    # Worst-case side for bilateral joints
    shoulder_deg = np.nanmax(np.stack([angles["RightShoulder"],
                                       angles["LeftShoulder"]], axis=1), axis=1)
    elbow_deg    = np.nanmax(np.stack([angles["RightElbow"],
                                       angles["LeftElbow"]], axis=1), axis=1)
    knee_deg     = np.nanmax(np.stack([angles["RightKnee"],
                                       angles["LeftKnee"]], axis=1), axis=1)
    trunk_deg    = angles["TrunkFlex"]

    rula_est = np.array([
        _rula_grand(shoulder_deg[i], elbow_deg[i], trunk_deg[i], knee_deg[i])
        for i in range(len(trc_ts))
    ])

    print(f"[eval] RULA — mean: {np.nanmean(rula_est):.2f}  "
          f"max: {np.nanmax(rula_est):.0f}  "
          f"% high-risk (>=4): {100*np.nanmean(rula_est>=4):.1f}%")

    # ------------------------------------------------------------------ #
    # 2. Load Xsens GT
    # ------------------------------------------------------------------ #
    if not os.path.isfile(MVNX_PATH):
        print(f"[eval] WARNING: Xsens file not found ({MVNX_PATH}). Skipping GT comparison.")
        _save_results(trc_ts, angles, rula_est, {})
        return

    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_interps = build_gt_angle_interpolators(mvnx, xsens_ts, xidx)

    # ------------------------------------------------------------------ #
    # 3. Time-align and compute MAE
    # ------------------------------------------------------------------ #
    aligned_ts = trc_ts + offset
    clip_start, clip_end = xsens_ts[0], xsens_ts[-1]
    mask = (aligned_ts >= clip_start) & (aligned_ts <= clip_end)
    print(f"[eval] Overlap: {mask.sum()} frames "
          f"({aligned_ts[mask][0]:.1f}-{aligned_ts[mask][-1]:.1f}s)")

    DOF_TO_GT = {
        "RightShoulder": "RightShoulder",
        "LeftShoulder":  "LeftShoulder",
        "RightElbow":    "RightElbow",
        "LeftElbow":     "LeftElbow",
        "RightKnee":     "RightKnee",
        "LeftKnee":      "LeftKnee",
    }

    mae_results: dict[str, float] = {}
    for label, gt_key in DOF_TO_GT.items():
        if gt_key not in gt_interps or not mask.any():
            continue
        est = angles[label][mask]
        gt  = gt_interps[gt_key](aligned_ts[mask])
        fin = np.isfinite(est) & np.isfinite(gt)
        if fin.sum() < 5:
            continue
        mae_results[label] = float(np.mean(np.abs(est[fin] - gt[fin])))

    print("\n[eval] Joint angle MAE vs Xsens GT:")
    for k, v in sorted(mae_results.items()):
        print(f"       {k:<22s}: {v:.2f}°")
    if mae_results:
        print(f"       {'Overall mean':<22s}: "
              f"{np.mean(list(mae_results.values())):.2f}°")

    # GT RULA — trunk from Xsens ergo angle Pelvis_T8 axis=0
    def _gt(key):
        return gt_interps[key](xsens_ts) if key in gt_interps \
            else np.full(len(xsens_ts), np.nan)

    trunk_ergo = mvnx.get_ergo_angle_data("Pelvis_T8")
    gt_trunk_vals = np.full(len(xsens_ts), np.nan)
    if trunk_ergo is not None:
        gt_trunk_interp = interp1d(
            xsens_ts, trunk_ergo[xidx, 0], kind='linear',
            bounds_error=False, fill_value=np.nan)
        gt_trunk_vals = gt_trunk_interp(xsens_ts)

    rula_gt = np.array([
        _rula_grand(
            np.nanmax([abs(_gt("RightShoulder")[i]),
                       abs(_gt("LeftShoulder")[i])]),
            np.nanmax([_gt("RightElbow")[i], _gt("LeftElbow")[i]]),
            abs(gt_trunk_vals[i]),
            np.nanmax([abs(_gt("RightKnee")[i]),
                       abs(_gt("LeftKnee")[i])]),
        )
        for i in range(len(xsens_ts))
    ])

    # ------------------------------------------------------------------ #
    # 4. Save and plot
    # ------------------------------------------------------------------ #
    _save_results(trc_ts, angles, rula_est, mae_results)
    _plot_mae(mae_results)
    _plot_rula(trc_ts, rula_est, xsens_ts, rula_gt)
    _plot_angle_timeseries(trc_ts, angles, gt_interps, aligned_ts, mask, xsens_ts)


def _save_results(timestamps, angles, rula_scores, mae_results):
    out = os.path.join(OUT_DIR, "eval_results_mono.npz")
    np.savez_compressed(
        out,
        timestamps      = timestamps,
        angle_names     = np.array(list(angles.keys())),
        angle_vals      = np.column_stack(list(angles.values())),
        rula_scores     = rula_scores,
        mae_joint_names = np.array(list(mae_results.keys())),
        mae_values      = np.array(list(mae_results.values())),
        overall_mae     = np.float64(
            np.mean(list(mae_results.values())) if mae_results else np.nan
        ),
    )
    print(f"[eval] Saved -> {out}")


def _plot_mae(mae_results: dict) -> None:
    if not mae_results:
        return
    names  = list(mae_results.keys())
    values = [mae_results[n] for n in names]
    overall = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#e74c3c" if v > 15 else "#f39c12" if v > 10 else "#2ecc71"
              for v in values]
    bars = ax.bar(names, values, color=colors, edgecolor="white", lw=0.8)
    ax.axhline(overall, color="#2c3e50", ls="--", lw=1.4,
               label=f"Mean = {overall:.1f}")
    ax.set_ylabel("MAE (degrees)")
    ax.set_title("Monocular Baseline — Joint Angle MAE vs Xsens GT "
                 "(geometric from TRC)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eval_angle_mono.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[eval] Plot -> {out}")


def _plot_rula(mot_ts, rula_est, xsens_ts, rula_gt) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

    ax = axes[0]
    valid = np.isfinite(rula_est)
    ax.plot(mot_ts[valid], rula_est[valid], color="#3498db", lw=0.9,
            label="Monocular (Aitor's pipeline)")
    for label, (s, e) in ACTIVITY_SEGMENTS.items():
        ax.axvspan(s, e, alpha=0.07, color="gray")
    ax.set_ylim(0.5, 7.5)
    ax.set_ylabel("RULA Grand Score")
    ax.set_xlabel("Video time (s)")
    ax.set_title("RULA Timeline — Monocular Baseline")
    ax.legend(loc="upper right")

    ax2 = axes[1]
    vgt = np.isfinite(rula_gt)
    if vgt.sum() > 0:
        ax2.plot(xsens_ts[vgt], rula_gt[vgt], color="#e74c3c", lw=0.9,
                 label="Xsens GT")
    ax2.set_ylim(0.5, 7.5)
    ax2.set_ylabel("RULA Grand Score (GT)")
    ax2.set_xlabel("Xsens time (s)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eval_rula_mono.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[eval] Plot -> {out}")


def _plot_angle_timeseries(trc_ts, angles, gt_interps, aligned_ts, mask,
                           xsens_ts) -> None:
    """Plot per-joint time series: estimated vs GT."""
    joints = ["RightShoulder", "LeftShoulder", "RightElbow", "LeftElbow",
              "RightKnee", "LeftKnee"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=False)

    for j, name in enumerate(joints):
        ax = axes[j // 2, j % 2]
        est = angles[name]
        valid_est = np.isfinite(est)
        ax.plot(trc_ts[valid_est], est[valid_est], color="#3498db", lw=0.7,
                alpha=0.8, label="Monocular")

        if name in gt_interps:
            gt_vals = gt_interps[name](xsens_ts)
            valid_gt = np.isfinite(gt_vals)
            ax.plot(xsens_ts[valid_gt], gt_vals[valid_gt], color="#e74c3c",
                    lw=0.7, alpha=0.8, label="Xsens GT")

        ax.set_title(name)
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=7, loc="upper right")

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle("Monocular Baseline — Joint Angle Time Series vs Xsens GT",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eval_timeseries_mono.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[eval] Plot -> {out}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[eval] Done in {time.time()-t0:.1f}s")
