#!/opt/anaconda3/envs/pose/bin/python
"""Side-by-side video for Fanbo3: left = raw A255 camera, right = SKT 3D skeleton.

SKT keypoints are rotated into a per-frame canonical anatomy frame (pelvis at
origin, +Z up, +X to subject right, +Y forward) so the skeleton orientation is
independent of the subject's facing direction. The original walking motion is
visible in the left camera panel.

Single-skeleton variant of ``11_aitor_comparison_video.py``. SKT NPZ and video
frames are already 1:1 aligned, so no metadata-based resync is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
LSHO, RSHO, LHIP, RHIP = 5, 6, 11, 12


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skt-npz",
        default=str(
            PROJECT_ROOT
            / "00_pose_pipeline_v2"
            / "runs"
            / "assar2026_fanbo3_a255_walking"
            / "skt_pose_optimized.npz"
        ),
    )
    parser.add_argument(
        "--left-video",
        default=str(PROJECT_ROOT / "2026_Assar_Data" / "A255" / "Video" / "cap_2_0.avi"),
    )
    parser.add_argument(
        "--output-mp4",
        default=str(
            PROJECT_ROOT
            / "00_pose_pipeline_v2"
            / "runs"
            / "assar2026_fanbo3_a255_walking"
            / "fanbo3_skt_skeleton_comparison.mp4"
        ),
    )
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=22.0)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--no-camera-flip", action="store_true",
                        help="Skip 180-degree rotation; fanbo3 yaml has rotate_180:true so flip is default.")
    return parser.parse_args()


def canonicalize(kp: np.ndarray) -> np.ndarray:
    """Rotate per-frame skeleton into anatomical canonical frame.

    Convention: pelvis at origin, +Z up (pelvis→neck), +X to subject's right
    (LHip→RHip projected), +Y forward (up × right).
    """
    out = np.full_like(kp, np.nan, dtype=np.float64)
    n = kp.shape[0]
    for i in range(n):
        lsh, rsh, lhip, rhip = kp[i, LSHO], kp[i, RSHO], kp[i, LHIP], kp[i, RHIP]
        if not (np.isfinite(lsh).all() and np.isfinite(rsh).all()
                and np.isfinite(lhip).all() and np.isfinite(rhip).all()):
            continue
        pelvis = 0.5 * (lhip + rhip)
        neck = 0.5 * (lsh + rsh)
        up = neck - pelvis
        nz = np.linalg.norm(up)
        if nz < 1e-3:
            continue
        up /= nz
        right_raw = rhip - lhip
        right_proj = right_raw - np.dot(right_raw, up) * up
        nr = np.linalg.norm(right_proj)
        if nr < 1e-3:
            continue
        right = right_proj / nr
        forward = np.cross(up, right)
        nf = np.linalg.norm(forward)
        if nf < 1e-3:
            continue
        forward /= nf
        rot = np.stack([right, forward, up], axis=0)
        centered = kp[i] - pelvis
        out[i] = centered @ rot.T
    return out


def render_skeleton_panel(ax, skt_pose, t_subject):
    """Draw SKT skeleton in the canonical frame."""
    ax.cla()
    ax.set_facecolor("white")
    finite = np.isfinite(skt_pose).all(axis=1)
    for a, b in COCO_EDGES:
        if finite[a] and finite[b]:
            ax.plot(
                [skt_pose[a, 0], skt_pose[b, 0]],
                [skt_pose[a, 1], skt_pose[b, 1]],
                [skt_pose[a, 2], skt_pose[b, 2]],
                color="tab:red", linewidth=2.5, alpha=0.9,
            )
    if finite.any():
        ax.scatter(skt_pose[finite, 0], skt_pose[finite, 1], skt_pose[finite, 2],
                   c="tab:red", s=18, alpha=0.9)

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-90, 60)
    ax.set_xlabel("X right (cm)")
    ax.set_ylabel("Y forward (cm)")
    ax.set_zlabel("Z up (cm)")
    ax.view_init(elev=15, azim=-65)
    ax.set_title(f"SKT 3D skeleton (canonical anatomy frame)  t={t_subject:.2f}s",
                 fontsize=11, pad=12)
    ax.plot([], [], [], color="tab:red", linewidth=2.5, label="SKT (stereo)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)


def main() -> None:
    """Build side-by-side comparison video for fanbo3 SKT."""
    args = parse_args()

    print(f"[load] SKT: {args.skt_npz}")
    skt = np.load(args.skt_npz, allow_pickle=True)
    skt_kp = np.asarray(skt["keypoints"], dtype=np.float64)
    skt_ts = np.asarray(skt["timestamps"], dtype=np.float64)
    skt_ts = skt_ts - skt_ts[0]
    n_frames = len(skt_kp)
    print(f"[load] SKT frames: {n_frames}, duration: {skt_ts[-1]:.2f}s")

    print("[canonicalize] rotating SKT pose into anatomical frame")
    skt_canon = canonicalize(skt_kp)
    valid_torso = np.isfinite(skt_canon[:, [LSHO, RSHO, LHIP, RHIP], :]).all(axis=(1, 2))
    print(f"[canonicalize] torso valid: {valid_torso.mean()*100:.1f}% ({valid_torso.sum()}/{n_frames})")

    start_idx = int(np.searchsorted(skt_ts, args.start_time))
    end_idx = int(np.searchsorted(skt_ts, args.end_time))
    end_idx = min(end_idx, n_frames)
    print(f"[range] frames {start_idx}..{end_idx} ({(end_idx-start_idx)/args.fps:.1f} s)")

    cap = cv2.VideoCapture(args.left_video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.left_video}")
    cam_w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cam_target_h = 720
    cam_scale = cam_target_h / cam_h_in
    cam_target_w = int(round(cam_w_in * cam_scale))
    plot_w, plot_h = 800, 720
    out_w = cam_target_w + plot_w
    out_h = cam_target_h
    print(f"[output] {out_w}x{out_h} @ {args.fps} fps -> {args.output_mp4}")

    Path(args.output_mp4).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (out_w, out_h),
    )

    fig = plt.figure(figsize=(plot_w / 100, plot_h / 100), dpi=100)
    ax3d = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.92)

    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    for i in tqdm(range(start_idx, end_idx), desc="render"):
        ok, frame = cap.read()
        if not ok:
            continue
        if not args.no_camera_flip:
            frame = cv2.flip(frame, -1)
        cam = cv2.resize(frame, (cam_target_w, cam_target_h))
        cv2.putText(cam, f"A255 left camera  t={skt_ts[i]:.2f}s  frame={i}",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        render_skeleton_panel(ax3d, skt_canon[i], float(skt_ts[i]))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plot_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if plot_img.shape[:2] != (out_h, plot_w):
            plot_img = cv2.resize(plot_img, (plot_w, out_h))

        out_frame = np.concatenate([cam, plot_img], axis=1)
        writer.write(out_frame)
        written += 1

    writer.release()
    cap.release()
    plt.close(fig)
    print(f"[saved] {args.output_mp4} ({written} frames)")


if __name__ == "__main__":
    main()
