#!/opt/anaconda3/envs/pose/bin/python
"""Side-by-side video: left = raw camera frame, right = SKT (smoothed) + FastSAM3D
both rendered in the SKT camera coordinate frame (rectified-left).

FastSAM3D is per-frame rigidly aligned (Kabsch) onto the SKT skeleton so both
share pelvis position and orientation — residual is the pose-shape disagreement.

TRC offset: FastSAM3D t=0 → stereo t=3.0 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHARED_DIR = PROJECT_ROOT / "shared"
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "00_pose_pipeline_v2" / "src"))

from skeleton_video_utils import kabsch_transform  # noqa: E402

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COCO17_NAMES = (
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
)


def load_fastsam_trc(trc_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load FastSAM3D TRC and return (timestamps_s, keypoints_cm [N,17,3]).

    Marker names match COCO17_NAMES exactly — direct index mapping works.
    Units in TRC are mm; converted to cm here.
    """
    lines = trc_path.read_text(encoding="utf-8").splitlines()
    header = lines[2].strip().split("\t")
    fps = float(header[0])
    n_markers = int(header[3])
    units = header[4].strip().lower()
    to_cm = 0.1 if units == "mm" else (100.0 if units == "m" else 1.0)

    marker_names = [n.strip() for n in lines[3].rstrip("\n").split("\t")[2:] if n.strip()]
    name_to_col = {n: i for i, n in enumerate(marker_names)}

    timestamps, frames = [], []
    for line in lines[6:]:
        if not line.strip():
            continue
        vals = line.rstrip("\n").split("\t")
        timestamps.append(float(vals[1]))
        coords = []
        for j in range(n_markers):
            coords.append([float(vals[2 + j * 3]) if vals[2 + j * 3] else np.nan,
                           float(vals[3 + j * 3]) if vals[3 + j * 3] else np.nan,
                           float(vals[4 + j * 3]) if vals[4 + j * 3] else np.nan])
        frames.append(coords)

    positions_cm = np.asarray(frames, dtype=np.float64) * to_cm  # (N, n_markers, 3)
    ts = np.asarray(timestamps, dtype=np.float64)
    ts -= ts[0]

    kp = np.full((len(ts), 17, 3), np.nan, dtype=np.float64)
    for coco_idx, name in enumerate(COCO17_NAMES):
        col = name_to_col.get(name)
        if col is not None:
            kp[:, coco_idx, :] = positions_cm[:, col, :]
    return ts, kp


def build_fastsam_interpolators(ts: np.ndarray, kp: np.ndarray):
    """Build per-joint per-axis scipy interpolators for FastSAM3D."""
    interps = {}
    for j in range(17):
        for ax in range(3):
            vals = kp[:, j, ax]
            finite = np.isfinite(vals)
            if finite.sum() < 2:
                continue
            interps[(j, ax)] = interp1d(
                ts[finite], vals[finite],
                kind="linear", bounds_error=False, fill_value=np.nan,
            )
    return interps


def fastsam_at(interps: dict, query_t: float) -> np.ndarray:
    """Sample FastSAM3D keypoints at one time point."""
    kp = np.full((17, 3), np.nan, dtype=np.float64)
    for (j, ax), fn in interps.items():
        kp[j, ax] = float(fn(query_t))
    return kp


def align_per_frame(skt_kp: np.ndarray, other_kp: np.ndarray) -> np.ndarray:
    """Kabsch-align other_kp onto skt_kp using shared finite joints."""
    # Use upper body + hips (joints 5-12) which are most reliably present in both
    shared_idx = [5, 6, 7, 8, 9, 10, 11, 12]
    valid = np.array([
        np.isfinite(skt_kp[i]).all() and np.isfinite(other_kp[i]).all()
        for i in shared_idx
    ])
    use = [shared_idx[k] for k in range(len(shared_idx)) if valid[k]]
    if len(use) < 4:
        return np.full_like(other_kp, np.nan)
    src = other_kp[use]
    tgt = skt_kp[use]
    rot, trans = kabsch_transform(src, tgt)
    return (rot @ other_kp.T).T + trans


def draw_skeleton(ax, kp: np.ndarray, color: str, label: str, lw: float = 2.5) -> None:
    """Render skeleton in camera frame.

    Axis mapping (matplotlib Z is rendered as screen vertical):
      mpl X = camera X (right)
      mpl Y = camera Z (depth, forward)
      mpl Z = -camera Y (anatomical up)
    """
    finite = np.isfinite(kp).all(axis=1)
    xs = kp[:, 0]
    ys = kp[:, 2]
    zs = -kp[:, 1]
    for a, b in COCO_EDGES:
        if finite[a] and finite[b]:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]],
                    color=color, lw=lw, alpha=0.9)
    if finite.any():
        ax.scatter(xs[finite], ys[finite], zs[finite], c=color, s=18, alpha=0.95)
    ax.plot([], [], [], color=color, lw=lw, label=label)


def main() -> None:
    skt_npz = PROJECT_ROOT / "00_pose_pipeline_v2/runs/assar2026_fanbo3_a255_walking/skt_pose_smoothed.npz"
    raw_npz = PROJECT_ROOT / "00_pose_pipeline_v2/runs/assar2026_fanbo3_a255_walking/skt_pose_optimized.npz"
    fastsam_trc = PROJECT_ROOT / "2026_Assar_Data/TRC FastSAM3D/markers_Fanbo3_2026-06-16 15-19-59_e65b2c80c6a540a9bed422da7bf58765.trc"
    left_video = PROJECT_ROOT / "2026_Assar_Data/A255/Video/cap_2_0.avi"
    out_path = PROJECT_ROOT / "00_pose_pipeline_v2/runs/assar2026_fanbo3_a255_walking/fanbo3_skt_smoothed_vs_fastsam_cam_v2.mp4"

    # FastSAM3D starts at stereo t=3.0 s
    fastsam_offset_s = 3.0
    start_frame, end_frame = 0, 275  # ~21 s window

    smo = np.load(skt_npz, allow_pickle=True)
    raw = np.load(raw_npz, allow_pickle=True)
    skt_kp = np.asarray(smo["keypoints"], dtype=np.float64)
    ts = np.asarray(raw["timestamps"], dtype=np.float64)
    ts -= ts[0]

    print("Loading FastSAM3D TRC ...")
    fs_ts, fs_kp = load_fastsam_trc(fastsam_trc)
    fs_interps = build_fastsam_interpolators(fs_ts, fs_kp)
    print(f"FastSAM3D: {len(fs_ts)} frames, duration {fs_ts[-1]:.2f}s")

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

        # Center SKT at mid-pelvis
        skt_frame = skt_kp[i].copy()
        mid_pel = 0.5 * (skt_frame[11] + skt_frame[12])
        skt_disp = skt_frame - mid_pel if np.isfinite(mid_pel).all() else skt_frame

        # Sample FastSAM3D at corresponding stereo time
        fs_query_t = float(ts[i]) - fastsam_offset_s
        fs_frame = fastsam_at(fs_interps, fs_query_t)

        # FastSAM3D uses Y-UP world frame; SKT is in Y-DOWN camera frame.
        # Negate Y so Kabsch can find a proper rotation (det=+1) instead of
        # a reflection, which would produce a mirrored skeleton.
        if np.isfinite(fs_frame).any():
            fs_frame_conv = fs_frame.copy()
            fs_frame_conv[:, 1] = -fs_frame[:, 1]  # Y-UP → Y-DOWN
            fs_aligned = align_per_frame(skt_disp, fs_frame_conv)
            if np.isfinite(fs_aligned).any():
                align_count += 1
        else:
            fs_aligned = np.full_like(fs_frame, np.nan)
            fs_frame_conv = fs_aligned

        ax3d.cla()
        ax3d.set_facecolor("white")
        draw_skeleton(ax3d, skt_disp, "tab:red", "SKT (smoothed)")
        draw_skeleton(ax3d, fs_aligned, "tab:blue", "FastSAM3D (per-frame rigid aligned)")

        ax3d.set_xlim(-60, 60)
        ax3d.set_ylim(-60, 60)
        ax3d.set_zlim(-80, 55)
        ax3d.set_xlabel("X cam right (cm)")
        ax3d.set_ylabel("Z cam fwd (cm)")
        ax3d.set_zlabel("anatomical up (cm)")
        ax3d.view_init(elev=10, azim=-90)
        ax3d.set_title(
            f"SKT vs FastSAM3D (rigid-aligned, camera frame)  t={ts[i]:.2f}s",
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
    print(f"Frames with FastSAM3D alignment: {align_count}/{end_frame - start_frame}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
