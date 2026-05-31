#!/opt/anaconda3/envs/pose/bin/python
"""Side-by-side video: left = raw camera, right = SKT vs FastSAM3D skeleton overlay.

Both skeletons are rotated into a per-frame canonical anatomy frame (pelvis at
origin, +Z up, +X to subject right, +Y forward) so the two skeletons share a
unified orientation regardless of their source coordinate conventions. Style
follows ``skeleton_comparison_dirA.mp4`` (white background, single 3D axes).
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
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

COCO17_NAMES = (
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
)
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
            / "01_stereo_triangulation"
            / "results"
            / "historical_best_20260324"
            / "recovered_baseline"
            / "optimized_pose.npz"
        ),
    )
    parser.add_argument(
        "--fastsam-trc",
        default=str(PROJECT_ROOT.parent / "10 Aitor" / "fastsam3d_2.trc"),
    )
    parser.add_argument(
        "--left-video",
        default=str(PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.avi"),
    )
    parser.add_argument(
        "--left-meta",
        default=str(PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.txt"),
    )
    parser.add_argument(
        "--right-meta",
        default=str(PROJECT_ROOT / "2025_Ergonomics_Data" / "1_video_right.txt"),
    )
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--start-time", type=float, default=100.0)
    parser.add_argument("--end-time", type=float, default=160.0)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--combined-csv", default=str(
        PROJECT_ROOT / "04_frame_delta_eval" / "results" / "phase4_skt_baseline_default" / "elbow_delta_combined.csv"
    ), help="Optional combined CSV for overlay angles (skipped if missing).")
    parser.add_argument("--no-camera-flip", action="store_true",
                        help="Skip 180-degree rotation of left camera frames.")
    parser.add_argument("--no-scale-fs", action="store_true",
                        help="Skip FastSAM3D->SKT torso-length scaling.")
    return parser.parse_args()


def torso_length(kp: np.ndarray) -> np.ndarray:
    """Per-frame mid-pelvis to mid-shoulder distance (cm)."""
    mask = (
        np.isfinite(kp[:, LSHO, :]).all(axis=1)
        & np.isfinite(kp[:, RSHO, :]).all(axis=1)
        & np.isfinite(kp[:, LHIP, :]).all(axis=1)
        & np.isfinite(kp[:, RHIP, :]).all(axis=1)
    )
    out = np.full(len(kp), np.nan, dtype=np.float64)
    if not mask.any():
        return out
    neck = 0.5 * (kp[mask, LSHO] + kp[mask, RSHO])
    pelvis = 0.5 * (kp[mask, LHIP] + kp[mask, RHIP])
    out[mask] = np.linalg.norm(neck - pelvis, axis=1)
    return out


def parse_meta(path: Path) -> list:
    """Parse stereo meta txt."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append({"id": int(parts[0]), "ts": int(parts[1]) + int(parts[2]) * 1e-6})
            except ValueError:
                continue
    return rows


def build_synced(left_rows, right_rows):
    """Match left/right by hardware frame id; preserve left_idx for video seek."""
    out = []
    li, ri = 0, 0
    while li < len(left_rows) and ri < len(right_rows):
        l, r = left_rows[li], right_rows[ri]
        if l["id"] == r["id"]:
            out.append({"frame_id": l["id"], "left_idx": li, "ts": l["ts"]})
            li += 1
            ri += 1
        elif l["id"] < r["id"]:
            li += 1
        else:
            ri += 1
    return out


def load_trc(path: Path):
    """Load TRC into (timestamps, marker_names, positions[F,M,3], units)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[2].strip().split("\t")
    if len(header) < 5:
        header = lines[2].strip().split()
    n_markers = int(header[3])
    units = header[4]
    raw_names = lines[3].rstrip("\n").split("\t")[2:]
    marker_names = [n.strip() for n in raw_names if n.strip()]
    if len(marker_names) != n_markers:
        marker_names = [n.strip() for n in lines[3].strip().split()[2:] if n.strip()]
    timestamps, frames = [], []
    expected = n_markers * 3
    for line in lines[6:]:
        if not line.strip():
            continue
        vals = line.rstrip("\n").split("\t")
        if len(vals) < 2:
            vals = line.strip().split()
        timestamps.append(float(vals[1]))
        coords_raw = vals[2:]
        if len(coords_raw) < expected:
            coords_raw = coords_raw + [""] * (expected - len(coords_raw))
        frames.append([float(v) if v else np.nan for v in coords_raw[:expected]])
    pos = np.asarray(frames, dtype=np.float64).reshape(-1, n_markers, 3)
    return np.asarray(timestamps, dtype=np.float64), marker_names, pos, units


def unit_to_cm(units: str) -> float:
    """Convert unit string to centimeter scale factor."""
    u = units.strip().lower()
    if u == "cm":
        return 1.0
    if u == "mm":
        return 0.1
    if u in {"m", "meter", "meters"}:
        return 100.0
    raise ValueError(f"unsupported unit: {units}")


def trc_to_coco17(marker_names, positions_cm) -> np.ndarray:
    """Pick COCO-17 subset from TRC."""
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    out = np.full((positions_cm.shape[0], 17, 3), np.nan, dtype=np.float64)
    for coco_idx, name in enumerate(COCO17_NAMES):
        if name in name_to_idx:
            out[:, coco_idx, :] = positions_cm[:, name_to_idx[name], :]
    return out


def canonicalize(kp: np.ndarray) -> np.ndarray:
    """Rotate a per-frame skeleton into anatomical canonical frame.

    Convention: pelvis at origin, +Z up (pelvis→neck), +X to subject's right
    (LHip→RHip projected), +Y forward (right × up).
    """
    out = np.full_like(kp, np.nan, dtype=np.float64)
    n = kp.shape[0]
    for i in range(n):
        lsh, rsh, lhip, rhip = kp[i, LSHO], kp[i, RSHO], kp[i, LHIP], kp[i, RHIP]
        if not (np.isfinite(lsh).all() and np.isfinite(rsh).all() and np.isfinite(lhip).all() and np.isfinite(rhip).all()):
            continue
        pelvis = 0.5 * (lhip + rhip)
        neck = 0.5 * (lsh + rsh)
        up = neck - pelvis
        nz = np.linalg.norm(up)
        if nz < 1e-3:
            continue
        up /= nz
        right_raw = rhip - lhip  # from subject's left hip → right hip = subject's +X
        right_proj = right_raw - np.dot(right_raw, up) * up
        nr = np.linalg.norm(right_proj)
        if nr < 1e-3:
            continue
        right = right_proj / nr
        forward = np.cross(up, right)  # right-hand: forward = up × right (subject anterior)
        nf = np.linalg.norm(forward)
        if nf < 1e-3:
            continue
        forward /= nf
        rot = np.stack([right, forward, up], axis=0)  # rows: maps world→canonical
        centered = kp[i] - pelvis
        out[i] = centered @ rot.T
    return out


def render_skeleton_panel(ax, skt_pose, fs_pose, t_subject):
    """Draw both skeletons in the same canonical frame."""
    ax.cla()
    ax.set_facecolor("white")
    for edges_color_label_pose in (
        (skt_pose, "tab:red", "SKT", 2.5, 0.9),
        (fs_pose, "tab:blue", "FastSAM3D", 2.0, 0.85),
    ):
        pose, color, _, lw, alpha = edges_color_label_pose
        finite = np.isfinite(pose).all(axis=1)
        for a, b in COCO_EDGES:
            if finite[a] and finite[b]:
                ax.plot(
                    [pose[a, 0], pose[b, 0]],
                    [pose[a, 1], pose[b, 1]],
                    [pose[a, 2], pose[b, 2]],
                    color=color, linewidth=lw, alpha=alpha,
                )
        if finite.any():
            ax.scatter(pose[finite, 0], pose[finite, 1], pose[finite, 2],
                       c=color, s=16, alpha=alpha)

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-90, 60)
    ax.set_xlabel("X right (cm)")
    ax.set_ylabel("Y forward (cm)")
    ax.set_zlabel("Z up (cm)")
    ax.view_init(elev=15, azim=-65)
    ax.set_title(f"SKT vs FastSAM3D (canonical anatomy frame, torso-scaled)  t={t_subject:.2f}s",
                 fontsize=11, pad=12)
    ax.plot([], [], [], color="tab:red", linewidth=2.5, label="SKT (stereo)")
    ax.plot([], [], [], color="tab:blue", linewidth=2.0, label="FastSAM3D (mono, scaled to SKT torso)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)


def main() -> None:
    """Build side-by-side comparison video."""
    args = parse_args()

    print(f"[load] SKT: {args.skt_npz}")
    skt = np.load(args.skt_npz, allow_pickle=True)
    skt_kp = np.asarray(skt["keypoints"], dtype=np.float64)
    skt_ts = np.asarray(skt["timestamps"], dtype=np.float64)
    skt_ts = skt_ts - skt_ts[0]
    n_synced = len(skt_kp)
    print(f"[load] SKT frames: {n_synced}")

    print(f"[load] FastSAM3D TRC: {args.fastsam_trc}")
    fs_ts, fs_names, fs_pos, fs_units = load_trc(Path(args.fastsam_trc))
    fs_pos_cm = fs_pos * unit_to_cm(fs_units)
    fs_coco = trc_to_coco17(fs_names, fs_pos_cm)
    print(f"[load] FastSAM3D frames: {len(fs_coco)} ({fs_units})")

    print("[align] mapping FastSAM3D to SKT synced timeline by left_idx")
    left_rows = parse_meta(Path(args.left_meta))
    right_rows = parse_meta(Path(args.right_meta))
    synced = build_synced(left_rows, right_rows)[:n_synced]
    if len(fs_coco) == n_synced:
        fs_aligned = fs_coco.copy()
    elif len(fs_coco) == len(left_rows):
        left_indices = np.asarray([row["left_idx"] for row in synced], dtype=np.int64)
        fs_aligned = fs_coco[left_indices]
    else:
        raise RuntimeError(
            f"FastSAM3D frame count {len(fs_coco)} matches neither SKT timeline "
            f"({n_synced}) nor left metadata ({len(left_rows)})."
        )
    print(f"[align] FastSAM3D aligned shape: {fs_aligned.shape}")

    print("[canonicalize] rotating both poses into anatomical frame")
    skt_canon = canonicalize(skt_kp)
    fs_canon = canonicalize(fs_aligned)
    valid_skt = np.isfinite(skt_canon).all(axis=2).all(axis=1).mean()
    valid_fs = np.isfinite(fs_canon).all(axis=2).all(axis=1).mean()
    print(f"[canonicalize] valid fully: SKT={valid_skt:.3f}, FS={valid_fs:.3f}")

    if args.no_scale_fs:
        scale_fs = 1.0
    else:
        skt_torso_med = float(np.nanmedian(torso_length(skt_kp)))
        fs_torso_med = float(np.nanmedian(torso_length(fs_aligned)))
        scale_fs = skt_torso_med / fs_torso_med if fs_torso_med > 1e-6 else 1.0
        fs_canon = fs_canon * scale_fs
        print(f"[scale] torso-length scale FS->SKT: {scale_fs:.4f}  "
              f"(SKT torso {skt_torso_med:.1f} cm, FS torso {fs_torso_med:.1f} cm)")

    # Frame range
    start_idx = int(np.searchsorted(skt_ts, args.start_time))
    end_idx = int(np.searchsorted(skt_ts, args.end_time))
    end_idx = min(end_idx, n_synced)
    print(f"[range] frames {start_idx}..{end_idx} ({(end_idx-start_idx)/args.fps:.1f} s)")

    cap = cv2.VideoCapture(args.left_video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.left_video}")

    # Output size: camera scaled to 720p height, 3D plot 800x720
    cam_target_h = 720
    cam_scale = cam_target_h / 1536
    cam_target_w = int(round(2048 * cam_scale))
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

    last_left_idx = -1
    written = 0
    for i in tqdm(range(start_idx, end_idx), desc="render"):
        row = synced[i]
        left_idx = row["left_idx"]
        if left_idx != last_left_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, left_idx)
        last_left_idx = left_idx
        ok, frame = cap.read()
        if not ok:
            continue
        if not args.no_camera_flip:
            frame = cv2.flip(frame, -1)  # 180-deg rotation: camera is mounted upside-down
        # Letter-box overlay on camera frame
        cam = cv2.resize(frame, (cam_target_w, cam_target_h))
        cv2.putText(cam, f"Left camera  t={skt_ts[i]:.2f}s  frame={left_idx}",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 3D plot
        render_skeleton_panel(ax3d, skt_canon[i], fs_canon[i], float(skt_ts[i]))
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
