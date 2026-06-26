"""Per-joint quality analysis for a single video frame.

Loads the SKT NPZ quality arrays, extracts 2D keypoints, 3D triangulated
positions, and per-joint quality metrics for a given frame, then renders
an annotated figure showing:
  - Left panel : cropped video frame with skeleton drawn, joints color-coded
                 green (good) / orange (marginal) / red (bad)
  - Right panel: table of per-joint metrics (2D pixel, 3D depth, epi, reproj, conf)

Usage::

    /opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/analyze_frame_joints.py \
        --frame 147 \
        --run-dir 00_pose_pipeline/runs/assar2026_fanbo1_a255_angle_test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from common.config import load_config, resolve_path, section
from stereo_loader import build_synced_timeline

# ── constants ──────────────────────────────────────────────────────────────────
JOINT_NAMES = {
    0: "Nose", 1: "L.Eye", 2: "R.Eye", 3: "L.Ear", 4: "R.Ear",
    5: "L.Shoulder", 6: "R.Shoulder", 7: "L.Elbow", 8: "R.Elbow",
    9: "L.Wrist", 10: "R.Wrist",
    11: "L.Hip", 12: "R.Hip", 13: "L.Knee", 14: "R.Knee",
    15: "L.Ankle", 16: "R.Ankle",
}
ARM_JOINTS = [5, 6, 7, 8, 9, 10]   # focus joints
SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
EPI_WARN = 10.0   # px – SKT quality filter threshold
EPI_BAD  = 20.0   # px – visually flagged as bad


def load_npz(config: dict) -> dict:
    """Load SKT NPZ and return a dict of arrays."""
    skt_cfg = section(config, "skt")
    if skt_cfg.get("use_existing_npz", False):
        npz_path = resolve_path(skt_cfg["existing_npz"], must_exist=True)
    else:
        run_dir = REPO_ROOT / section(config, "outputs")["runs_dir"] / section(config, "outputs")["run_tag"]
        npz_path = run_dir / skt_cfg.get("output_npz", "skt_pose_optimized.npz")
    payload = np.load(npz_path, allow_pickle=True)
    return dict(payload), npz_path


def load_video_frame(config: dict, frame_idx: int) -> tuple[np.ndarray, int]:
    """Return the BGR video frame corresponding to SKT frame `frame_idx`."""
    dataset = section(config, "dataset")
    left_meta = resolve_path(dataset["left_metadata"], must_exist=True)
    right_meta = resolve_path(dataset["right_metadata"], must_exist=True)
    left_video = resolve_path(dataset["left_video"], must_exist=True)
    rotate = dataset.get("rotate_180", False)
    ts_fmt = dataset.get("timestamp_format", "seconds_microseconds_columns")

    _, synced, _, _ = build_synced_timeline(left_meta, right_meta, ts_fmt)

    if frame_idx >= len(synced):
        raise IndexError(f"Frame {frame_idx} out of range (synced has {len(synced)} pairs)")

    video_frame_idx = synced[frame_idx].left_idx
    cap = cv2.VideoCapture(str(left_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {video_frame_idx} from {left_video}")
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame, video_frame_idx


def joint_color(epi: float) -> tuple[float, float, float]:
    """Return an (R,G,B) matplotlib color based on epipolar error."""
    if epi > EPI_BAD:
        return (0.90, 0.15, 0.15)   # red
    if epi > EPI_WARN:
        return (0.95, 0.55, 0.05)   # orange
    return (0.15, 0.75, 0.30)       # green


def draw_skeleton(frame_bgr: np.ndarray, kp2d: np.ndarray, epi: np.ndarray) -> np.ndarray:
    """Return annotated copy of frame_bgr with skeleton and colored keypoints."""
    img = frame_bgr.copy()
    n_joints = kp2d.shape[0]

    # Draw limb lines (gray)
    for i, j in SKELETON_PAIRS:
        if i < n_joints and j < n_joints:
            x1, y1 = int(kp2d[i, 0]), int(kp2d[i, 1])
            x2, y2 = int(kp2d[j, 0]), int(kp2d[j, 1])
            cv2.line(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

    # Draw joints
    for ji in range(n_joints):
        x, y = int(kp2d[ji, 0]), int(kp2d[ji, 1])
        epi_val = float(epi[ji]) if ji < len(epi) else 0.0
        r, g, b = joint_color(epi_val)
        color_bgr = (int(b * 255), int(g * 255), int(r * 255))
        radius = 7 if ji in ARM_JOINTS else 4
        cv2.circle(img, (x, y), radius, color_bgr, -1)
        cv2.circle(img, (x, y), radius + 1, (255, 255, 255), 1)

    return img


def make_figure(
    frame_idx: int,
    frame_bgr: np.ndarray,
    kp2d: np.ndarray,
    kp3d: np.ndarray,
    epi: np.ndarray,
    reproj: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    out_path: Path,
) -> None:
    """Render and save the annotated figure."""
    annotated = draw_skeleton(frame_bgr, kp2d, epi)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax_img, ax_tbl = axes

    # ── Left: annotated frame ─────────────────────────────────────────────────
    ax_img.imshow(annotated_rgb)

    # Arm joint labels on image
    for ji in ARM_JOINTS:
        x, y = float(kp2d[ji, 0]), float(kp2d[ji, 1])
        epi_val = float(epi[ji])
        r, g, b = joint_color(epi_val)
        short = JOINT_NAMES[ji].replace("L.", "L.").replace("R.", "R.")
        ax_img.annotate(
            f"{short}\nepi={epi_val:.1f}px",
            xy=(x, y), xytext=(x + 25, y - 20),
            fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=(r, g, b), alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="white", lw=0.8),
        )

    ax_img.set_title(f"Frame {frame_idx} — left camera (A255)\nSkeleton with per-joint quality color coding", fontsize=10)
    ax_img.axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=(0.15, 0.75, 0.30), label=f"Good  (epi ≤ {EPI_WARN:.0f}px)"),
        mpatches.Patch(color=(0.95, 0.55, 0.05), label=f"Warn  ({EPI_WARN:.0f}px < epi ≤ {EPI_BAD:.0f}px)"),
        mpatches.Patch(color=(0.90, 0.15, 0.15), label=f"Bad   (epi > {EPI_BAD:.0f}px)"),
    ]
    ax_img.legend(handles=patches, loc="lower right", fontsize=8)

    # ── Right: per-joint table ────────────────────────────────────────────────
    ax_tbl.axis("off")

    headers = ["Joint", "2D-x", "2D-y", "3D-z(cm)", "epi(px)", "reproj(px)", "conf_L", "conf_R"]
    rows = []
    for ji in ARM_JOINTS:
        x2d, y2d = float(kp2d[ji, 0]), float(kp2d[ji, 1])
        z3d = float(kp3d[ji, 2]) if ji < len(kp3d) else float("nan")  # already in cm
        e = float(epi[ji]) if ji < len(epi) else float("nan")
        rp = float(reproj[ji]) if ji < len(reproj) else float("nan")
        cl = float(conf_l[ji]) if ji < len(conf_l) else float("nan")
        cr = float(conf_r[ji]) if ji < len(conf_r) else float("nan")
        rows.append([
            JOINT_NAMES[ji],
            f"{x2d:.0f}", f"{y2d:.0f}",
            f"{z3d:.1f}",
            f"{e:.2f}", f"{rp:.2f}",
            f"{cl:.3f}", f"{cr:.3f}",
        ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.0)

    # Color rows by epi severity
    for row_idx, ji in enumerate(ARM_JOINTS):
        e = float(epi[ji]) if ji < len(epi) else 0.0
        r, g, b = joint_color(e)
        for col_idx in range(len(headers)):
            cell = tbl[(row_idx + 1, col_idx)]
            cell.set_facecolor((r, g, b, 0.25))

    ax_tbl.set_title(
        f"Per-joint quality metrics — arm joints\n"
        f"(3D-z is depth from camera; epi threshold = {EPI_WARN:.0f}px)",
        fontsize=10,
    )

    # Depth anomaly annotation
    z_vals = [float(kp3d[ji, 2]) for ji in [6, 8, 10] if ji < len(kp3d)]
    if len(z_vals) == 3:
        note = (
            f"R.Shoulder→R.Elbow depth: Δ={z_vals[1]-z_vals[0]:+.1f}cm\n"
            f"R.Elbow→R.Wrist depth:    Δ={z_vals[2]-z_vals[1]:+.1f}cm"
        )
        ax_tbl.text(
            0.5, 0.02, note,
            transform=ax_tbl.transAxes,
            ha="center", va="bottom",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8),
        )

    fig.suptitle(
        f"SKT triangulation quality — Frame {frame_idx}\n"
        "Root cause: high epipolar error on R.Wrist/R.Elbow causes false 3D depth displacement",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze_frame_joints] Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=147, help="SKT frame index to analyze")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "00_pose_pipeline_v2/configs/assar2026_fanbo1_a255.yaml")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run directory for output (default: derived from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    payload, npz_path = load_npz(config)
    print(f"[analyze_frame_joints] NPZ: {npz_path}")
    print(f"[analyze_frame_joints] Available arrays: {list(payload.keys())}")

    frame_idx = args.frame
    kp2d = np.asarray(payload["keypoints_left_2d"], dtype=np.float64)   # (T, J, 2) or (T, J, 3)
    kp3d = np.asarray(payload["keypoints"], dtype=np.float64)           # (T, J, 3) in metres

    # Confidence arrays
    files = set(payload.keys())
    left_key = "triang_conf_left" if "triang_conf_left" in files else "conf_left"
    right_key = "triang_conf_right" if "triang_conf_right" in files else "conf_right"
    conf_l = np.asarray(payload[left_key], dtype=np.float64)[frame_idx]
    conf_r = np.asarray(payload[right_key], dtype=np.float64)[frame_idx]
    epi    = np.asarray(payload["epipolar_error"], dtype=np.float64)[frame_idx]
    reproj = np.asarray(payload["reprojection_error"], dtype=np.float64)[frame_idx]

    kp2d_f = kp2d[frame_idx, :, :2]
    kp3d_f = kp3d[frame_idx]

    # Print summary
    print(f"\nFrame {frame_idx} — arm joint quality:")
    print(f"{'Joint':<14} {'2D-x':>7} {'2D-y':>7} {'3D-z(cm)':>10} {'epi(px)':>9} {'reproj':>8} {'conf_L':>8} {'conf_R':>8}")
    for ji in ARM_JOINTS:
        z = kp3d_f[ji, 2]  # already in cm
        flag = " ← BAD" if epi[ji] > EPI_BAD else (" ← warn" if epi[ji] > EPI_WARN else "")
        print(f"{JOINT_NAMES[ji]:<14} {kp2d_f[ji,0]:>7.1f} {kp2d_f[ji,1]:>7.1f} {z:>10.1f} "
              f"{epi[ji]:>9.2f} {reproj[ji]:>8.2f} {conf_l[ji]:>8.3f} {conf_r[ji]:>8.3f}{flag}")

    print("\nLoading video frame …")
    frame_bgr, vid_idx = load_video_frame(config, frame_idx)
    print(f"  Video frame index: {vid_idx}, shape: {frame_bgr.shape}")

    # Output path
    if args.run_dir:
        out_dir = args.run_dir / "elbow_viz"
    else:
        run_tag = section(config, "outputs")["run_tag"]
        out_dir = REPO_ROOT / section(config, "outputs")["runs_dir"] / run_tag / "elbow_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame{frame_idx:04d}_joint_analysis.png"

    make_figure(frame_idx, frame_bgr, kp2d_f, kp3d_f, epi, reproj, conf_l, conf_r, out_path)


if __name__ == "__main__":
    main()
