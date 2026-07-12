#!/opt/anaconda3/envs/pose/bin/python
"""Render specific single-frame snapshots from SKT vs Xsens comparison
for the weekly report (good-alignment frame + turning/hip-offset frame)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_ROOT / "shared"
sys.path.insert(0, str(SHARED_DIR))

from skeleton_video_utils import (  # noqa: E402
    load_xsens_skeleton, xsens_pose_at, kabsch_transform,
)

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

XSENS_TO_COCO_IDX = {
    "LeftUpperArm": 5, "RightUpperArm": 6,
    "LeftForeArm": 7, "RightForeArm": 8,
    "LeftHand": 9, "RightHand": 10,
    "LeftUpperLeg": 11, "RightUpperLeg": 12,
    "LeftLowerLeg": 13, "RightLowerLeg": 14,
    "LeftFoot": 15, "RightFoot": 16,
}


def xsens_to_coco17(xsens_pose):
    out = np.full((17, 3), np.nan, dtype=np.float64)
    for seg, idx in XSENS_TO_COCO_IDX.items():
        if seg in xsens_pose:
            out[idx] = xsens_pose[seg]
    return out


def align_per_frame(skt_kp, xsens_kp):
    valid_mask = np.zeros(17, dtype=bool)
    for idx in XSENS_TO_COCO_IDX.values():
        valid_mask[idx] = (
            np.isfinite(skt_kp[idx]).all() and np.isfinite(xsens_kp[idx]).all()
        )
    if valid_mask.sum() < 4:
        return np.full_like(xsens_kp, np.nan)
    src = xsens_kp[valid_mask]
    tgt = skt_kp[valid_mask]
    rot, trans = kabsch_transform(src, tgt)
    return (rot @ xsens_kp.T).T + trans


def draw_skeleton(ax, kp, color, label, lw=2.8):
    finite = np.isfinite(kp).all(axis=1)
    xs = kp[:, 0]
    ys = kp[:, 2]
    zs = -kp[:, 1]
    for a, b in COCO_EDGES:
        if finite[a] and finite[b]:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]],
                    color=color, lw=lw, alpha=0.92)
    if finite.any():
        ax.scatter(xs[finite], ys[finite], zs[finite], c=color, s=24, alpha=0.95)
    ax.plot([], [], [], color=color, lw=lw, label=label)


def render_snapshot(out_path, frame_idx, title_extra,
                    skt_kp_all, ts, xsens, offset_s, cap, cam_target_h=720):
    """Render a single side-by-side snapshot."""
    cam_w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_target_w = int(round(cam_w_in * cam_target_h / cam_h_in))
    plot_w, plot_h = 880, 720

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    frame = cv2.flip(frame, -1)
    cam = cv2.resize(frame, (cam_target_w, cam_target_h))
    cv2.putText(cam, f"t={ts[frame_idx]:.2f}s  frame={frame_idx}",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(cam, title_extra, (16, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    xsens_pose = xsens_pose_at(xsens, float(ts[frame_idx] - offset_s))
    xsens_kp = xsens_to_coco17(xsens_pose) if xsens_pose else np.full((17, 3), np.nan)
    skt_frame = skt_kp_all[frame_idx].copy()
    mid_pel_skt = 0.5 * (skt_frame[11] + skt_frame[12])
    skt_disp = skt_frame - mid_pel_skt if np.isfinite(mid_pel_skt).all() else skt_frame

    if np.isfinite(xsens_kp).any():
        xsens_aligned = align_per_frame(skt_disp, xsens_kp)
    else:
        xsens_aligned = np.full_like(xsens_kp, np.nan)

    fig = plt.figure(figsize=(plot_w / 100, plot_h / 100), dpi=110)
    ax3d = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.04, top=0.92)
    ax3d.set_facecolor("white")
    draw_skeleton(ax3d, skt_disp, "tab:red", "SKT (smoothed)")
    draw_skeleton(ax3d, xsens_aligned, "tab:blue", "Xsens (per-frame rigid aligned)")
    ax3d.set_xlim(-60, 60)
    ax3d.set_ylim(-60, 60)
    ax3d.set_zlim(-80, 55)
    ax3d.set_xlabel("X cam right (cm)")
    ax3d.set_ylabel("Z cam fwd (cm)")
    ax3d.set_zlabel("anatomical up (cm)")
    ax3d.view_init(elev=10, azim=-90)
    ax3d.set_title(f"SKT vs Xsens (rigid-aligned)  t={ts[frame_idx]:.2f}s",
                   fontsize=11, pad=8)
    ax3d.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plot_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if plot_img.shape[:2] != (cam_target_h, plot_w):
        plot_img = cv2.resize(plot_img, (plot_w, cam_target_h))
    plt.close(fig)

    combined = np.concatenate([cam, plot_img], axis=1)
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  saved: {out_path}  shape={combined.shape}")


def main():
    skt_npz = PROJECT_ROOT / "00_pose_pipeline_v2/runs/assar2026_fanbo3_a255_walking/skt_pose_smoothed.npz"
    raw_npz = PROJECT_ROOT / "00_pose_pipeline_v2/runs/assar2026_fanbo3_a255_walking/skt_pose_optimized.npz"
    xsens_mvnx = PROJECT_ROOT / "2026_Assar_Data/Xsens MVNX/Fanbo-003.mvnx"
    left_video = PROJECT_ROOT / "2026_Assar_Data/A255/Video/cap_2_0.avi"
    out_dir = PROJECT_ROOT / "weekly_reports/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    offset_s = 1.98

    smo = np.load(skt_npz, allow_pickle=True)
    raw = np.load(raw_npz, allow_pickle=True)
    skt_kp = np.asarray(smo["keypoints"], dtype=np.float64)
    ts = np.asarray(raw["timestamps"], dtype=np.float64)
    ts -= ts[0]

    xsens = load_xsens_skeleton(xsens_mvnx)
    cap = cv2.VideoCapture(str(left_video))

    # Render several candidates spanning the walking sequence so we can pick.
    # User wants: one "明显好" (good alignment) + one "不好" (hip/turning offset).
    candidates = [
        (40,  "candidate_t03.2s_walking_forward"),
        (80,  "candidate_t06.4s_walking_forward"),
        (120, "candidate_t09.6s_approaching"),
        (150, "candidate_t12.0s_near_turn"),
        (170, "candidate_t13.6s_turning"),
        (190, "candidate_t15.2s_post_turn"),
        (220, "candidate_t17.6s_walking_back"),
        (250, "candidate_t20.0s_returning"),
    ]
    for frame_idx, label in candidates:
        render_snapshot(out_dir / f"{label}.jpg", frame_idx, "",
                        skt_kp, ts, xsens, offset_s, cap)

    cap.release()


if __name__ == "__main__":
    main()
