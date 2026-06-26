"""Stage 0 baseline diagnostics — characterises the v1 fanbo7 failure mode.

Reads v1's SKT NPZ + angle_timeseries.csv. Produces:
  1. Baseline metrics (SKT vs FastSAM3D) — MAE / smoothness / valid ratio / jump count
  2. Jitter-source bucket breakdown — assigns each angle-jump frame a dominant cause
  3. Five diagnostic figures saved to <output-dir>/figures/

This script ONLY reads v1 output — does not modify v1 in any way.

Usage::

    /opt/anaconda3/envs/pose/bin/python 00_pose_pipeline_v2/src/diagnostics_baseline.py \
        --v1-run-dir 00_pose_pipeline/runs/assar2026_fanbo7_a257_elbow_test \
        --output-dir 00_pose_pipeline_v2/runs/baseline_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COCO_ARM_JOINTS = {5: "L.Shldr", 6: "R.Shldr", 7: "L.Elbow", 8: "R.Elbow", 9: "L.Wrist", 10: "R.Wrist"}
RIGHT_ARM = [6, 8, 10]
LEFT_ARM = [5, 7, 9]
RIGHT_ELBOW_TRIPLET = (6, 8, 10)


def load_angle_timeseries(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load angle_timeseries.csv → (time_s, {system: array}). Right elbow only here."""
    times: list[float] = []
    cols: dict[str, list[float]] = {}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                times.append(float(row["Time_s"]))
            except (KeyError, ValueError):
                continue
            for key, value in row.items():
                if key in ("Frame", "Time_s") or not key.endswith("_RightElbow_deg"):
                    continue
                system = key[: -len("_RightElbow_deg")]
                cols.setdefault(system, []).append(float(value) if value.strip() else math.nan)
    time_s = np.asarray(times, dtype=np.float64)
    angles = {sys: np.asarray(arr, dtype=np.float64) for sys, arr in cols.items()}
    return time_s, angles


def angular_acc_rms(angle: np.ndarray, time_s: np.ndarray) -> float:
    """RMS of second-difference / dt² — smoothness proxy in deg/s²."""
    finite_mask = np.isfinite(angle)
    if finite_mask.sum() < 3:
        return float("nan")
    dt = np.median(np.diff(time_s))
    if dt <= 0 or not np.isfinite(dt):
        return float("nan")
    acc = np.diff(angle, n=2) / (dt ** 2)
    finite = acc[np.isfinite(acc)]
    if len(finite) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite ** 2)))


def count_jumps(angle: np.ndarray, threshold_deg: float = 10.0) -> int:
    """Count consecutive-frame absolute differences > threshold."""
    d = np.diff(angle)
    return int(np.sum(np.abs(d[np.isfinite(d)]) > threshold_deg))


def classify_jitter_source(
    frame_idx: int,
    epi: np.ndarray,
    reproj: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    disp: np.ndarray,
    kp3d: np.ndarray,
    bone_priors: dict[str, float] | None,
) -> str:
    """Return dominant root-cause bucket label for a single frame on the right-arm chain.

    Buckets (in priority order):
      person_match_fail      — negative disparity → left/right see different people
      epi_residual_spike     — epipolar y-residual > 15 px (large 2D mismatch)
      low_one_side_conf      — at least one of L/R has conf < 0.4
      reproj_high            — triangulation reproj > 15 px
      depth_jitter           — R.Elbow Z changes > 4 cm but 2D keypoints stable → triangulation depth noise
      bone_length_anomaly    — R.upper-arm or R.forearm deviation > 15% from prior
      near_straight_geom     — flexion < 15° (geometry sensitivity amplifies noise)
      all_valid_but_jump     — none of the above triggered
    """
    cf_min_arm = np.minimum(conf_l[frame_idx, RIGHT_ARM], conf_r[frame_idx, RIGHT_ARM])
    epi_arm = epi[frame_idx, RIGHT_ARM]
    rep_arm = reproj[frame_idx, RIGHT_ARM]
    disp_arm = disp[frame_idx, RIGHT_ARM]

    if np.any(disp_arm < 0):
        return "person_match_fail"
    if np.any(epi_arm > 15):
        return "epi_residual_spike"
    if np.any(cf_min_arm < 0.4):
        return "low_one_side_conf"
    if np.any(rep_arm > 15):
        return "reproj_high"

    # Depth jitter: R.Elbow Z changed > 4 cm but 2D position changed < 5 px
    if frame_idx > 0:
        re_curr = kp3d[frame_idx, 8]
        re_prev = kp3d[frame_idx - 1, 8]
        if np.isfinite(re_curr).all() and np.isfinite(re_prev).all():
            dz = abs(re_curr[2] - re_prev[2])
            dxy = float(np.linalg.norm(re_curr[:2] - re_prev[:2]))
            if dz > 4.0 and dxy < 5.0:
                return "depth_jitter"

    # Bone-length check (relax to 15%)
    s, e, w = kp3d[frame_idx, 6], kp3d[frame_idx, 8], kp3d[frame_idx, 10]
    if np.isfinite(s).all() and np.isfinite(e).all() and np.isfinite(w).all():
        upper = float(np.linalg.norm(s - e))
        forearm = float(np.linalg.norm(e - w))
        if bone_priors is not None:
            up_dev = abs(upper - bone_priors.get("right_upper_arm", upper)) / max(bone_priors.get("right_upper_arm", 1e-6), 1e-6)
            fa_dev = abs(forearm - bone_priors.get("right_lower_arm", forearm)) / max(bone_priors.get("right_lower_arm", 1e-6), 1e-6)
            if up_dev > 0.15 or fa_dev > 0.15:
                return "bone_length_anomaly"

        # Near-straight geometry: flexion < 15° makes angle hyper-sensitive to depth noise
        v1 = s - e
        v2 = w - e
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = float(np.clip((v1 @ v2) / (n1 * n2), -1.0, 1.0))
            interior_deg = math.degrees(math.acos(cos_a))
            flex_deg = 180.0 - interior_deg
            if flex_deg < 15.0:
                return "near_straight_geom"

    return "all_valid_but_jump"


def estimate_bone_priors_simple(kp3d: np.ndarray) -> dict[str, float]:
    """Trimmed-median bone priors for right arm (used only for jitter classification)."""
    out = {}
    pairs = {"right_upper_arm": (6, 8), "right_lower_arm": (8, 10), "left_upper_arm": (5, 7), "left_lower_arm": (7, 9)}
    for name, (a, b) in pairs.items():
        d = np.linalg.norm(kp3d[:, a] - kp3d[:, b], axis=1)
        d = d[np.isfinite(d)]
        if len(d) < 10:
            continue
        lo, hi = np.percentile(d, [20, 80])
        trimmed = d[(d >= lo) & (d <= hi)]
        out[name] = float(np.median(trimmed)) if len(trimmed) > 0 else float(np.median(d))
    return out


def plot_right_elbow_with_jumps(
    time_s: np.ndarray,
    angles: dict[str, np.ndarray],
    jump_frames: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"SKT": "#e05c5c", "FastSAM3D": "#4a90d9", "XsensFair": "#2ca02c", "XsensNative": "#999999"}
    for sys, arr in angles.items():
        ls = "--" if sys == "XsensNative" else "-"
        lw = 1.5 if sys == "SKT" else 1.3
        ax.plot(time_s, arr, label=sys, color=colors.get(sys, "k"), linestyle=ls, linewidth=lw, zorder=2 if sys != "SKT" else 3)
    if len(jump_frames) > 0:
        valid_jumps = jump_frames[(jump_frames >= 0) & (jump_frames < len(time_s))]
        ax.scatter(time_s[valid_jumps], angles["SKT"][valid_jumps], s=30, c="red", marker="x", label=f"SKT jump >10° (n={len(valid_jumps)})", zorder=4)
    ax.set_title("Right elbow angle — baseline v1 timeseries with SKT jump positions")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Right elbow angle (°)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_jitter_bucket_histogram(buckets: dict[str, int], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(buckets.keys())
    counts = [buckets[k] for k in labels]
    colors_map = {
        "person_match_fail": "#d62728",
        "epi_residual_spike": "#ff7f0e",
        "low_one_side_conf": "#9467bd",
        "reproj_high": "#1f77b4",
        "depth_jitter": "#e377c2",
        "bone_length_anomaly": "#8c564b",
        "near_straight_geom": "#bcbd22",
        "all_valid_but_jump": "#7f7f7f",
    }
    bar_colors = [colors_map.get(k, "k") for k in labels]
    ax.barh(labels, counts, color=bar_colors)
    for i, v in enumerate(counts):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("Number of frames")
    ax.set_title("Right elbow jump (>10°) frames — dominant root-cause distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_frame_quality(
    time_s: np.ndarray,
    epi: np.ndarray,
    reproj: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for j, name in COCO_ARM_JOINTS.items():
        if j not in (6, 8, 10):
            continue
        axes[0].plot(time_s, np.clip(epi[:, j], 0, 50), label=name, alpha=0.7, linewidth=1.0)
        axes[1].plot(time_s, np.clip(reproj[:, j], 0, 50), label=name, alpha=0.7, linewidth=1.0)
        axes[2].plot(time_s, np.minimum(conf_l[:, j], conf_r[:, j]), label=name, alpha=0.7, linewidth=1.0)
    axes[0].axhline(10, color="r", linestyle="--", alpha=0.5, label="epi threshold 10px")
    axes[0].set_ylabel("Epipolar (px, clipped 50)")
    axes[0].set_title("Right arm joint quality vs time (clipped for visibility)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(15, color="r", linestyle="--", alpha=0.5, label="reproj threshold 15px")
    axes[1].set_ylabel("Reprojection (px, clipped 50)")
    axes[1].legend(loc="best", fontsize=8)
    axes[2].axhline(0.4, color="r", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("min(conf_L, conf_R)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="best", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_bone_length_drift(time_s: np.ndarray, kp3d: np.ndarray, priors: dict[str, float], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax_idx, (label, (a, b), prior_key) in enumerate(
        [("Right upper arm", (6, 8), "right_upper_arm"), ("Right forearm", (8, 10), "right_lower_arm")]
    ):
        d = np.linalg.norm(kp3d[:, a] - kp3d[:, b], axis=1)
        axes[ax_idx].plot(time_s, d, color="#1f77b4", linewidth=1.0)
        if prior_key in priors:
            axes[ax_idx].axhline(priors[prior_key], color="g", linestyle="--", alpha=0.7, label=f"prior median = {priors[prior_key]:.1f} cm")
            axes[ax_idx].axhline(priors[prior_key] * 1.25, color="r", linestyle=":", alpha=0.5, label="±25%")
            axes[ax_idx].axhline(priors[prior_key] * 0.75, color="r", linestyle=":", alpha=0.5)
        axes[ax_idx].set_ylabel(f"{label} length (cm)")
        axes[ax_idx].legend(loc="best", fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Right arm bone length vs time — sanity check for triangulation stability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_disparity_sign_anomalies(time_s: np.ndarray, disp: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for j in (6, 8, 10):
        ax.plot(time_s, disp[:, j], label=COCO_ARM_JOINTS[j], linewidth=1.0, alpha=0.8)
    ax.axhline(0, color="r", linestyle="--", alpha=0.5, label="Δ disparity = 0 (impossible region for forward scene)")
    ax.fill_between(time_s, -100, 0, alpha=0.1, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Disparity = u_L - u_R (px)")
    ax.set_title("Right arm disparity — negative values indicate left/right mismatch")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-run-dir", type=Path, required=True, help="v1 run directory containing skt_pose_optimized.npz + angle_eval/")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write baseline_metrics.json + figures/")
    parser.add_argument("--jump-threshold-deg", type=float, default=10.0)
    args = parser.parse_args()

    v1_dir = args.v1_run_dir
    out_dir = args.output_dir
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print(f"[stage0] reading v1 outputs from {v1_dir}")
    npz_path = v1_dir / "skt_pose_optimized.npz"
    csv_path = v1_dir / "angle_eval" / "angle_timeseries.csv"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    npz = np.load(npz_path, allow_pickle=True)
    time_s_kp = np.asarray(npz["timestamps"], dtype=np.float64)
    kp3d = np.asarray(npz["keypoints"], dtype=np.float64)
    epi = np.asarray(npz["epipolar_error"], dtype=np.float64)
    reproj = np.asarray(npz["reprojection_error"], dtype=np.float64)
    disp = np.asarray(npz["disparity_px"], dtype=np.float64)
    files = set(npz.files)
    conf_l = np.asarray(npz["triang_conf_left" if "triang_conf_left" in files else "conf_left"], dtype=np.float64)
    conf_r = np.asarray(npz["triang_conf_right" if "triang_conf_right" in files else "conf_right"], dtype=np.float64)

    time_s, angles = load_angle_timeseries(csv_path)
    print(f"[stage0] angle systems: {list(angles.keys())}, n_frames={len(time_s)}")

    # Trim to shared length
    n = min(len(time_s), len(time_s_kp))
    time_s = time_s[:n]
    kp3d = kp3d[:n]
    epi = epi[:n]
    reproj = reproj[:n]
    disp = disp[:n]
    conf_l = conf_l[:n]
    conf_r = conf_r[:n]
    for sys in list(angles):
        angles[sys] = angles[sys][:n]

    skt = angles.get("SKT")
    fast = angles.get("FastSAM3D")

    # Triple-overlap window
    overlap_mask = np.isfinite(skt) & np.isfinite(fast)
    if "XsensFair" in angles:
        overlap_mask &= np.isfinite(angles["XsensFair"])
    valid_idx = np.where(overlap_mask)[0]
    t_start = float(time_s[valid_idx[0]]) if len(valid_idx) else float("nan")
    t_end = float(time_s[valid_idx[-1]]) if len(valid_idx) else float("nan")

    metrics: dict[str, object] = {
        "source": str(v1_dir),
        "n_frames": int(n),
        "duration_s": float(time_s[-1] - time_s[0]) if n > 0 else 0.0,
        "triple_overlap": {
            "t_start_s": t_start,
            "t_end_s": t_end,
            "n_overlap_frames": int(overlap_mask.sum()),
        },
    }

    # MAE SKT vs FastSAM3D in overlap (= primary user-preferred metric)
    if overlap_mask.any():
        diff = np.abs(skt[overlap_mask] - fast[overlap_mask])
        metrics["mae_deg_SKT_vs_FastSAM3D"] = float(np.mean(diff))
        metrics["median_abs_err_deg_SKT_vs_FastSAM3D"] = float(np.median(diff))
        metrics["bias_deg_SKT_vs_FastSAM3D"] = float(np.mean(skt[overlap_mask] - fast[overlap_mask]))

    # Smoothness (angular acc RMS) in overlap window for each system
    for sys, arr in angles.items():
        a = arr.copy()
        if overlap_mask.any():
            a_window = a[valid_idx[0]:valid_idx[-1] + 1]
            t_window = time_s[valid_idx[0]:valid_idx[-1] + 1]
            metrics[f"angular_acc_rms_deg_per_s2_{sys}"] = angular_acc_rms(a_window, t_window)
            metrics[f"jump_count_{sys}_gt{int(args.jump_threshold_deg)}"] = count_jumps(a_window, args.jump_threshold_deg)

    # Valid ratio in overlap window
    for sys, arr in angles.items():
        if overlap_mask.any():
            window = arr[valid_idx[0]:valid_idx[-1] + 1]
            metrics[f"valid_ratio_{sys}"] = float(np.isfinite(window).mean())

    # Identify SKT jump frames (within full timeline, finite SKT pair)
    skt_diff = np.diff(skt)
    jump_mask = np.isfinite(skt_diff) & (np.abs(skt_diff) > args.jump_threshold_deg)
    jump_frames = np.where(jump_mask)[0] + 1  # the second frame of the jump pair
    print(f"[stage0] SKT jumps >{args.jump_threshold_deg}°: {len(jump_frames)} frames")

    # Bucket each jump frame by dominant cause
    bone_priors = estimate_bone_priors_simple(kp3d)
    buckets = {"person_match_fail": 0, "epi_residual_spike": 0, "low_one_side_conf": 0,
               "reproj_high": 0, "depth_jitter": 0, "bone_length_anomaly": 0,
               "near_straight_geom": 0, "all_valid_but_jump": 0}
    for fi in jump_frames:
        if fi < n:
            buckets[classify_jitter_source(fi, epi, reproj, conf_l, conf_r, disp, kp3d, bone_priors)] += 1
    metrics["jitter_source_buckets"] = buckets
    metrics["bone_priors_cm"] = bone_priors

    # Save figures
    plot_right_elbow_with_jumps(time_s, angles, jump_frames, figs / "01_right_elbow_with_jumps.png")
    plot_jitter_bucket_histogram(buckets, figs / "02_jitter_source_buckets.png")
    plot_per_frame_quality(time_s_kp[:n], epi, reproj, conf_l, conf_r, figs / "03_right_arm_quality.png")
    plot_bone_length_drift(time_s_kp[:n], kp3d, bone_priors, figs / "04_bone_length_drift.png")
    plot_disparity_sign_anomalies(time_s_kp[:n], disp, figs / "05_disparity_anomalies.png")

    # Save metrics
    metrics_path = out_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[stage0] wrote {metrics_path}")
    print(f"[stage0] figures in {figs}")
    print()
    print("=" * 70)
    print("Baseline (v1) summary — fanbo7_a257 right elbow")
    print("=" * 70)
    print(f"  triple-overlap: {t_start:.1f}s ~ {t_end:.1f}s  ({int(overlap_mask.sum())} frames)")
    if "mae_deg_SKT_vs_FastSAM3D" in metrics:
        print(f"  MAE   SKT vs FastSAM3D = {metrics['mae_deg_SKT_vs_FastSAM3D']:.2f}°")
        print(f"  Med|err| SKT vs FastSAM3D = {metrics['median_abs_err_deg_SKT_vs_FastSAM3D']:.2f}°")
        print(f"  Bias  SKT vs FastSAM3D = {metrics['bias_deg_SKT_vs_FastSAM3D']:+.2f}°")
    for sys in ["SKT", "FastSAM3D"]:
        rms = metrics.get(f"angular_acc_rms_deg_per_s2_{sys}", None)
        jumps = metrics.get(f"jump_count_{sys}_gt{int(args.jump_threshold_deg)}", None)
        vr = metrics.get(f"valid_ratio_{sys}", None)
        print(f"  {sys:<10} angular_acc_RMS = {rms:.0f} °/s²,  jumps >{int(args.jump_threshold_deg)}° = {jumps},  valid_ratio = {vr:.2%}" if rms is not None else f"  {sys}: n/a")
    print()
    print("Jitter source breakdown (SKT jumps):")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        pct = v / max(len(jump_frames), 1) * 100
        print(f"  {k:<24}  {v:>4}   ({pct:5.1f}%)")
    print()
    print("Bone priors (cm):")
    for k, v in bone_priors.items():
        print(f"  {k:<22}  {v:6.1f}")


if __name__ == "__main__":
    main()
