#!/opt/anaconda3/envs/pose/bin/python
"""Side-by-side video: left = raw camera frame, right = SKT (smoothed) + Xsens
both rendered in the SKT camera coordinate frame (rectified-left).

Xsens is per-frame rigidly aligned (Kabsch) onto the SKT skeleton so the two
share pelvis position and orientation — the residual is the pose-shape
disagreement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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


def xsens_to_coco17(xsens_pose: dict) -> np.ndarray:
    """Build a COCO-17 array (NaN for unmapped joints) from Xsens segment dict."""
    out = np.full((17, 3), np.nan, dtype=np.float64)
    for seg, idx in XSENS_TO_COCO_IDX.items():
        if seg in xsens_pose:
            out[idx] = xsens_pose[seg]
    return out


def align_per_frame(skt_kp: np.ndarray, xsens_kp: np.ndarray) -> np.ndarray:
    """Kabsch align Xsens onto SKT for this single frame.

    Uses only joints valid in BOTH skeletons.
    """
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


def draw_skeleton(ax, kp: np.ndarray, color: str, label: str, lw: float = 2.5) -> None:
    """Render a skeleton.

    Axis mapping to matplotlib 3D (matplotlib always renders its Z as vertical):
      mpl X = camera X (right)
      mpl Y = camera Z (depth, forward)
      mpl Z = -camera Y (anatomical up; camera Y is image-down)

    With view_init(elev=10, azim=-90) this looks from the -depth direction,
    i.e. from the camera side, so +X is right and mpl Z (= anatomical up)
    appears as vertical on screen.
    """
    finite = np.isfinite(kp).all(axis=1)
    xs = kp[:, 0]          # camera X → mpl X
    ys = kp[:, 2]          # camera Z (depth) → mpl Y
    zs = -kp[:, 1]         # -camera Y → mpl Z (vertical in rendered view)
    for a, b in COCO_EDGES:
        if finite[a] and finite[b]:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]],
                    color=color, lw=lw, alpha=0.9)
    if finite.any():
        ax.scatter(xs[finite], ys[finite], zs[finite],
                   c=color, s=18, alpha=0.95)
    ax.plot([], [], [], color=color, lw=lw, label=label)


def main() -> None:
    skt_npz = PROJECT_ROOT / "00_pose_pipeline_v2/runs/assar2026_fanbo3_a255_walking/skt_pose_smoothed.npz"
    raw_npz = PROJECT_ROOT / "00_pose_pipeline_v2/runs/assar2026_fanbo3_a255_walking/skt_pose_optimized.npz"
    xsens_mvnx = PROJECT_ROOT / "2026_Assar_Data/Xsens MVNX/Fanbo-003.mvnx"
    left_video = PROJECT_ROOT / "2026_Assar_Data/A255/Video/cap_2_0.avi"
    out_path = PROJECT_ROOT / "00_pose_pipeline_v2/runs/assar2026_fanbo3_a255_walking/fanbo3_skt_smoothed_vs_xsens_cam_v2.mp4"
    offset_s = 1.98
    start_frame, end_frame = 0, 275  # ~21s window

    smo = np.load(skt_npz, allow_pickle=True)
    raw = np.load(raw_npz, allow_pickle=True)
    skt_kp = np.asarray(smo["keypoints"], dtype=np.float64)
    ts = np.asarray(raw["timestamps"], dtype=np.float64)
    ts -= ts[0]

    xsens = load_xsens_skeleton(xsens_mvnx)
    print(f"Loaded Xsens with {len(xsens.interpolators)} segments")

    # Camera coordinate frame for SKT (rectified left):
    # X right, Y down (into image bottom), Z forward (depth into scene)
    # Pre-translate everything to mid-pelvis at origin for visualization, BUT
    # preserve relative orientation (camera frame). We'll use mid-pelvis per frame.

    cap = cv2.VideoCapture(str(left_video))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_target_h = 720
    cam_target_w = int(round(cam_w * cam_target_h / cam_h))
    plot_w, plot_h = 880, 720
    out_w = cam_target_w + plot_w
    fps = 12.5

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (out_w, cam_target_h))

    fig = plt.figure(figsize=(plot_w / 100, plot_h / 100), dpi=100)
    ax3d = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.04, top=0.94)

    align_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in tqdm(range(start_frame, end_frame), desc="render"):
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, -1)
        cam = cv2.resize(frame, (cam_target_w, cam_target_h))
        cv2.putText(cam, f"t={ts[i]:.2f}s frame={i}",
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Sample Xsens at corresponding time
        xsens_pose = xsens_pose_at(xsens, float(ts[i] - offset_s))
        xsens_kp = xsens_to_coco17(xsens_pose) if xsens_pose else np.full((17, 3), np.nan)
        skt_frame = skt_kp[i].copy()

        # Translate SKT so mid-pelvis at origin (camera-frame relative)
        mid_pel_skt = 0.5 * (skt_frame[11] + skt_frame[12])
        if np.isfinite(mid_pel_skt).all():
            skt_disp = skt_frame - mid_pel_skt
        else:
            skt_disp = skt_frame

        # Align Xsens to SKT (rigid Kabsch, per frame) in centered frame
        if np.isfinite(xsens_kp).any():
            xsens_aligned = align_per_frame(skt_disp, xsens_kp)
            if np.isfinite(xsens_aligned).any():
                align_count += 1
        else:
            xsens_aligned = np.full_like(xsens_kp, np.nan)

        ax3d.cla()
        ax3d.set_facecolor("white")
        draw_skeleton(ax3d, skt_disp, "tab:red", "SKT (smoothed)")
        draw_skeleton(ax3d, xsens_aligned, "tab:blue", "Xsens (per-frame rigid aligned)")

        # Display axes (after Y negation in draw_skeleton):
        #   X right (X_cam), Y up (-Y_cam = anatomical up), Z forward (Z_cam)
        # mpl X=cam_X (right), mpl Y=cam_Z (depth), mpl Z=anatomical_up
        ax3d.set_xlim(-60, 60)
        ax3d.set_ylim(-60, 60)   # depth relative to pelvis
        ax3d.set_zlim(-80, 55)   # feet ~-75 to head ~+50 relative to pelvis
        ax3d.set_xlabel("X cam right (cm)")
        ax3d.set_ylabel("Z cam fwd (cm)")
        ax3d.set_zlabel("anatomical up (cm)")
        # azim=-90 → look from -mpl_Y direction = from -depth = from camera side.
        # mpl Z (anatomical up) is rendered as the screen vertical → head at top.
        ax3d.view_init(elev=10, azim=-90)
        ax3d.set_title(
            f"SKT vs Xsens (rigid-aligned, camera frame)  t={ts[i]:.2f}s",
            fontsize=11, pad=8,
        )
        ax3d.legend(loc="upper right", fontsize=9, framealpha=0.9)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plot_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if plot_img.shape[:2] != (cam_target_h, plot_w):
            plot_img = cv2.resize(plot_img, (plot_w, cam_target_h))

        out_frame = np.concatenate([cam, plot_img], axis=1)
        writer.write(out_frame)

    writer.release()
    cap.release()
    plt.close(fig)
    print(f"Frames with Xsens alignment: {align_count}/{end_frame - start_frame}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
