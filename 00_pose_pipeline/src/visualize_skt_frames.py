"""Visualize selected SKT frames with 2D skeleton overlay.

Shows why certain frames succeed or fail the quality filter.
Usage:
    python 00_pose_pipeline/src/visualize_skt_frames.py \
        --config 00_pose_pipeline/configs/assar2026_fanbo7_a257.yaml \
        [--frames 107 149 354]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from common.config import resolve_path, section
import yaml

# COCO-17 skeleton connections
SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),       # legs
    (0, 1), (0, 2), (1, 3), (2, 4),               # head
]

RIGHT_ARM_JOINTS = {6, 8, 10}  # right shoulder, elbow, wrist

JOINT_NAMES = {
    0: "nose", 1: "l_eye", 2: "r_eye", 3: "l_ear", 4: "r_ear",
    5: "l_sho", 6: "r_sho", 7: "l_elb", 8: "r_elb", 9: "l_wri", 10: "r_wri",
    11: "l_hip", 12: "r_hip", 13: "l_kne", 14: "r_kne", 15: "l_ank", 16: "r_ank",
}


def read_frame(cap: cv2.VideoCapture, idx: int, rotate_180: bool = False) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_skeleton(
    ax,
    img: np.ndarray,
    kp2d_this: np.ndarray,        # (17, 2) pixel coords for this camera
    kp2d_other: np.ndarray | None, # (17, 2) pixel coords for other camera (for epi line)
    epi: np.ndarray,               # (17,) epipolar errors
    reproj: np.ndarray,            # (17,) reprojection errors
    kp3d: np.ndarray,              # (17, 3) 3D coords in cm
    title: str,
    draw_epi_line: bool = False,   # draw horizontal epipolar mismatch lines on right elbow
) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=9, pad=4)
    ax.axis("off")

    h, w = img.shape[:2]

    # Draw skeleton connections
    for j1, j2 in SKELETON_PAIRS:
        p1, p2 = kp2d_this[j1], kp2d_this[j2]
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        is_right = j1 in RIGHT_ARM_JOINTS or j2 in RIGHT_ARM_JOINTS
        color = "red" if is_right else "lime"
        lw = 2.0 if is_right else 1.0
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, alpha=0.85)

    # Draw joints
    for j in range(17):
        p = kp2d_this[j]
        if not np.isfinite(p).all():
            continue
        is_right = j in RIGHT_ARM_JOINTS
        color = "red" if is_right else "lime"
        ms = 5 if is_right else 3
        ax.plot(p[0], p[1], "o", color=color, markersize=ms)

    # Draw epipolar lines for all 3 right arm joints when requested.
    # One full-width horizontal line per joint in each camera image.
    # Comparing left vs right image: lines at the same y → epipolar ok; gap → error.
    if draw_epi_line and kp2d_other is not None:
        # (joint_index, color, short_label)
        arm_epi_joints = [
            (6, "#FFD700", "r_sho"),   # shoulder – yellow
            (8, "#FF4444", "r_elb"),   # elbow    – red
            (10, "#00FFFF", "r_wri"),  # wrist    – cyan
        ]
        for j, color, label in arm_epi_joints:
            p_this = kp2d_this[j]
            p_other = kp2d_other[j]
            if not (np.isfinite(p_this).all() and np.isfinite(p_other).all()):
                continue
            y_this = p_this[1]
            y_other = p_other[1]
            y_diff = abs(y_this - y_other)
            epi_val = epi[j]

            # Full-width solid line at this camera's detected y
            ax.axhline(y=y_this, color=color, lw=1.5, alpha=0.85, zorder=5)

            # Label on the right edge: joint name + y value + epi
            epi_flag = f"  epi={epi_val:.1f}px ✗" if epi_val > 10.0 else f"  epi={epi_val:.1f}px ✓"
            ax.text(
                w - 4, y_this - 3,
                f"{label} y={y_this:.0f}{epi_flag}",
                fontsize=6.5, color=color, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55),
                zorder=7,
            )

            # For the elbow only, add a double-headed arrow showing Δy vs the other camera
            if j == 8 and y_diff > 3:
                x_arr = 0.06 * w
                ax.annotate(
                    "", xy=(x_arr, y_other), xytext=(x_arr, y_this),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1.8),
                    zorder=8,
                )
                y_mid = 0.5 * (y_this + y_other)
                ax.text(
                    x_arr + 8, y_mid,
                    f"Δy={y_diff:.1f}px",
                    fontsize=7, color="white", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#333", alpha=0.7),
                    zorder=8,
                )

    # Annotation box for right arm metrics
    arm_j = [6, 8, 10]
    lines = []
    for j in arm_j:
        name = JOINT_NAMES[j]
        e_val = epi[j] if np.isfinite(epi[j]) else float("nan")
        r_val = reproj[j] if np.isfinite(reproj[j]) else float("nan")
        z_val = kp3d[j, 2] if np.isfinite(kp3d[j, 2]) else float("nan")
        e_flag = " ✗" if e_val > 10.0 else " ✓"
        r_flag = " ✗" if r_val > 10.0 else ""
        lines.append(f"{name}: epi={e_val:.1f}{e_flag}  reproj={r_val:.1f}{r_flag}  z={z_val:.0f}cm")

    text = "\n".join(lines)
    ax.text(
        0.01, 0.01, text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        fontfamily="monospace",
    )


def visualize_frames(config_path: str, frame_indices: list[int]) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = section(config, "dataset")
    skt_cfg = section(config, "skt")
    rotate = bool(dataset.get("rotate_180", False))

    left_video = resolve_path(dataset["left_video"], must_exist=True)
    right_video = resolve_path(dataset["right_video"], must_exist=True)
    left_meta = resolve_path(dataset["left_metadata"], must_exist=True)
    right_meta = resolve_path(dataset["right_metadata"], must_exist=True)

    # Build synced timeline to get correct video frame indices
    from stereo_loader import build_synced_timeline
    _, synced, _, _ = build_synced_timeline(
        left_meta, right_meta, dataset.get("timestamp_format", "seconds_microseconds_columns")
    )

    # Determine NPZ path
    if skt_cfg.get("use_existing_npz", False):
        npz_path = resolve_path(skt_cfg["existing_npz"], must_exist=True)
    else:
        run_dir = Path(section(config, "outputs")["runs_dir"]) / section(config, "outputs")["run_tag"]
        npz_path = run_dir / skt_cfg.get("output_npz", "skt_pose_optimized.npz")

    d = np.load(npz_path, allow_pickle=True)
    kp3d_all = np.asarray(d["keypoints"])       # (N, 17, 3) in cm
    kp2d_l_all = np.asarray(d["keypoints_left_2d"])  # (N, 17, 2)
    kp2d_r_all = np.asarray(d["keypoints_right_2d"])  # (N, 17, 2)
    epi_all = np.asarray(d["epipolar_error"])    # (N, 17)
    reproj_all = np.asarray(d["reprojection_error"])  # (N, 17)
    ts_all = np.asarray(d["timestamps"])

    n_frames = len(frame_indices)
    fig, axes = plt.subplots(n_frames, 2, figsize=(14, 5 * n_frames))
    if n_frames == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("SKT Frame Diagnostics — Fanbo7 A257", fontsize=12, y=0.995)

    cap_l = cv2.VideoCapture(str(left_video))
    cap_r = cv2.VideoCapture(str(right_video))

    for row, fr in enumerate(frame_indices):
        assert fr < len(synced), f"Frame {fr} out of range (synced has {len(synced)} frames)"
        left_vid_idx = synced[fr].left_idx
        right_vid_idx = synced[fr].right_idx
        t_s = ts_all[fr]

        img_l = read_frame(cap_l, left_vid_idx, rotate_180=rotate)
        img_r = read_frame(cap_r, right_vid_idx, rotate_180=rotate)

        if img_l is None or img_r is None:
            print(f"WARNING: could not read frame {fr} from video")
            continue

        kp2d_l = kp2d_l_all[fr]
        kp2d_r = kp2d_r_all[fr]
        kp3d = kp3d_all[fr]
        epi = epi_all[fr]
        reproj = reproj_all[fr]

        # Determine label for this frame
        e8 = epi[8]
        r8 = reproj[8]
        z6, z8, z10 = kp3d[6, 2], kp3d[8, 2], kp3d[10, 2]
        if np.isfinite(z6) and np.isfinite(z8) and np.isfinite(z10):
            mid = 0.5 * (z6 + z10)
            dev = abs(z8 - mid)
        else:
            dev = float("nan")

        if e8 > 100:
            status = "FAIL: wrong person (epi>>100px)"
        elif e8 > 10:
            status = f"FAIL: epi={e8:.1f}px > 10px threshold"
        elif r8 > 10:
            status = f"FAIL: reproj={r8:.1f}px > 10px threshold"
        elif np.isfinite(dev) and dev > 15.0:
            status = f"FAIL: depth_dev={dev:.1f}cm > 15cm threshold"
        else:
            status = "VALID ✓"

        is_fail = e8 > 10 or r8 > 10 or (np.isfinite(dev) and dev > 15.0)

        title_l = f"fr={fr}  t={t_s:.2f}s  LEFT — {status}"
        title_r = f"fr={fr}  t={t_s:.2f}s  RIGHT"

        # For failing frames: draw epipolar mismatch lines on LEFT image;
        # right image shows the other-camera detected position for reference.
        draw_skeleton(axes[row, 0], img_l, kp2d_l, kp2d_r, epi, reproj, kp3d, title_l,
                      draw_epi_line=is_fail)
        draw_skeleton(axes[row, 1], img_r, kp2d_r, kp2d_l, epi, reproj, kp3d, title_r,
                      draw_epi_line=is_fail)

    cap_l.release()
    cap_r.release()

    plt.tight_layout()

    # Output path
    run_dir = Path(section(config, "outputs")["runs_dir"]) / section(config, "outputs")["run_tag"]
    out_dir = run_dir / "angle_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"skeleton_frames_{'_'.join(str(f) for f in frame_indices)}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", nargs="+", type=int, default=[107, 135, 149, 167])
    args = ap.parse_args()
    visualize_frames(args.config, args.frames)


if __name__ == "__main__":
    main()
