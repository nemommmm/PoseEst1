"""Visualize elbow skeleton on video frames for diagnostic purposes.

Finds frames where SKT elbow detection has HIGH confidence but HIGH angle error
vs XsensFair/FastSAM3D, and draws the 2D SKT skeleton on the actual video frame.
FastSAM3D and Xsens angles are annotated as text (cannot be projected to A255 frame).

Usage::

    /opt/anaconda3/envs/pose/bin/python 00_pose_pipeline_v2/src/visualize_elbow_frames.py \
        --config 00_pose_pipeline_v2/configs/assar2026_fanbo1_a255.yaml \
        --run-dir 00_pose_pipeline_v2/runs/assar2026_fanbo1_a255_angle_test \
        --n-frames 6 \
        --out-dir 00_pose_pipeline_v2/runs/assar2026_fanbo1_a255_angle_test/elbow_viz
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "00_pose_pipeline" / "src"))

from common.config import resolve_path, section, load_config
from stereo_loader import build_synced_timeline

# COCO-17 skeleton links for drawing
_SKELETON = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 6),            # shoulders
    (5, 11), (6, 12),  # torso sides
    (11, 12),          # hips
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16),# right leg
    (0, 5), (0, 6),    # neck to shoulders (approx)
]
_JOINT_COLORS = {
    7: (0, 200, 0),    # left elbow  — green
    8: (0, 100, 255),  # right elbow — orange-ish
    9: (0, 200, 0),    # left wrist
    10: (0, 100, 255), # right wrist
    5: (200, 200, 0),  # left shoulder
    6: (200, 200, 0),  # right shoulder
}
_DEFAULT_JOINT_COLOR = (180, 180, 180)
_LINK_COLOR = (200, 200, 200)
_ELBOW_LINK_COLOR_LEFT  = (0, 200, 0)
_ELBOW_LINK_COLOR_RIGHT = (0, 100, 255)
_ELBOW_LINKS = {(5, 7), (7, 9), (6, 8), (8, 10)}


def _draw_skeleton(img: np.ndarray, kps_2d: np.ndarray, conf: np.ndarray, min_conf: float = 0.2) -> np.ndarray:
    """Draw COCO-17 skeleton on img. kps_2d: (17,2), conf: (17,)."""
    out = img.copy()
    h, w = out.shape[:2]
    for j1, j2 in _SKELETON:
        if conf[j1] < min_conf or conf[j2] < min_conf:
            continue
        x1, y1 = int(kps_2d[j1, 0]), int(kps_2d[j1, 1])
        x2, y2 = int(kps_2d[j2, 0]), int(kps_2d[j2, 1])
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue
        pair = (min(j1, j2), max(j1, j2))
        color = _ELBOW_LINK_COLOR_LEFT if pair in {(5, 7), (7, 9)} else \
                _ELBOW_LINK_COLOR_RIGHT if pair in {(6, 8), (8, 10)} else _LINK_COLOR
        cv2.line(out, (x1, y1), (x2, y2), color, 2)
    for j in range(17):
        if conf[j] < min_conf:
            continue
        x, y = int(kps_2d[j, 0]), int(kps_2d[j, 1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        color = _JOINT_COLORS.get(j, _DEFAULT_JOINT_COLOR)
        r = 6 if j in (7, 8) else 4
        cv2.circle(out, (x, y), r, color, -1)
        cv2.circle(out, (x, y), r, (255, 255, 255), 1)
    return out


def _angle_from_kps(kps: np.ndarray, a: int, b: int, c: int) -> float | None:
    """Compute interior angle at joint b using 2D points a-b-c."""
    pa, pb, pc = kps[a], kps[b], kps[c]
    if not (np.isfinite(pa).all() and np.isfinite(pb).all() and np.isfinite(pc).all()):
        return None
    v1 = pa - pb
    v2 = pc - pb
    cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))


def _load_angle_timeseries(csv_path: Path) -> dict[str, list]:
    """Load angle_timeseries.csv into column arrays."""
    data: dict[str, list] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(v)
    return data


def find_candidate_frames(
    npz: "np.lib.npyio.NpzFile",
    angle_data: dict[str, list],
    n: int,
    min_conf: float = 0.5,
    min_error_deg: float = 15.0,
) -> list[dict]:
    """Find frames with high SKT confidence AND large elbow error vs XsensFair."""
    kps_2d = np.asarray(npz["keypoints_left_2d"], dtype=np.float64)   # (N,17,2)
    conf_l = np.asarray(npz["conf_left"],  dtype=np.float64)           # (N,17)
    conf_r = np.asarray(npz["conf_right"], dtype=np.float64)           # (N,17)
    n_frames = len(kps_2d)

    def _col(name: str) -> np.ndarray:
        vals = angle_data.get(name, [])
        arr = np.full(n_frames, np.nan)
        for i, v in enumerate(vals[:n_frames]):
            if isinstance(v, str) and v.strip():
                try:
                    arr[i] = float(v)
                except ValueError:
                    pass
        return arr

    skt_le  = _col("SKT_LeftElbow_deg")
    skt_re  = _col("SKT_RightElbow_deg")
    xsens_le = _col("XsensFair_LeftElbow_deg")
    xsens_re = _col("XsensFair_RightElbow_deg")
    fs_le   = _col("FastSAM3D_LeftElbow_deg")
    fs_re   = _col("FastSAM3D_RightElbow_deg")

    candidates = []
    for i in range(n_frames):
        # elbow confidence (min of left/right for each elbow joint)
        conf_le = min(conf_l[i, 7], conf_r[i, 7])
        conf_re = min(conf_l[i, 8], conf_r[i, 8])
        high_conf = (conf_le >= min_conf) or (conf_re >= min_conf)
        if not high_conf:
            continue

        # Compute errors where reference is available
        err_le = abs(skt_le[i] - xsens_le[i]) if np.isfinite(skt_le[i]) and np.isfinite(xsens_le[i]) else np.nan
        err_re = abs(skt_re[i] - xsens_re[i]) if np.isfinite(skt_re[i]) and np.isfinite(xsens_re[i]) else np.nan
        max_err = max(e for e in [err_le, err_re] if np.isfinite(e)) if any(np.isfinite(e) for e in [err_le, err_re]) else np.nan

        candidates.append({
            "frame_idx": i,
            "conf_le": float(conf_le),
            "conf_re": float(conf_re),
            "skt_le": float(skt_le[i]) if np.isfinite(skt_le[i]) else None,
            "skt_re": float(skt_re[i]) if np.isfinite(skt_re[i]) else None,
            "xsens_le": float(xsens_le[i]) if np.isfinite(xsens_le[i]) else None,
            "xsens_re": float(xsens_re[i]) if np.isfinite(xsens_re[i]) else None,
            "fs_le": float(fs_le[i]) if np.isfinite(fs_le[i]) else None,
            "fs_re": float(fs_re[i]) if np.isfinite(fs_re[i]) else None,
            "err_le": float(err_le) if np.isfinite(err_le) else None,
            "err_re": float(err_re) if np.isfinite(err_re) else None,
            "max_err": float(max_err) if np.isfinite(max_err) else None,
        })

    # Sort: frames WITH Xsens reference by descending error first, then high-conf-only frames
    with_ref = [c for c in candidates if c["max_err"] is not None and c["max_err"] >= min_error_deg]
    without_ref = [c for c in candidates if c["max_err"] is None]

    with_ref.sort(key=lambda c: c["max_err"], reverse=True)
    without_ref.sort(key=lambda c: max(c["conf_le"], c["conf_re"]), reverse=True)

    # Take top with-ref frames, fill remaining slots with high-conf no-ref frames
    selected = with_ref[:n]
    if len(selected) < n:
        selected += without_ref[: n - len(selected)]

    # Ensure variety: if all from same region, spread them out
    return selected[:n]


def render_frame(
    video_path: Path,
    video_frame_idx: int,
    skt_kps2d: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    meta: dict,
    rotate_180: bool = True,
) -> np.ndarray:
    """Read a video frame and draw the SKT skeleton on it."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    conf_min = np.minimum(conf_l, conf_r)  # (17,)
    frame = _draw_skeleton(frame, skt_kps2d, conf_min)

    # Annotate angle values
    h, w = frame.shape[:2]
    lines = []
    if meta.get("skt_le") is not None:
        lines.append(f"SKT  L.Elbow: {meta['skt_le']:.0f}deg")
    if meta.get("skt_re") is not None:
        lines.append(f"SKT  R.Elbow: {meta['skt_re']:.0f}deg")
    if meta.get("xsens_le") is not None:
        lines.append(f"Xsens L.Elbow: {meta['xsens_le']:.0f}deg")
    if meta.get("xsens_re") is not None:
        lines.append(f"Xsens R.Elbow: {meta['xsens_re']:.0f}deg")
    if meta.get("fs_le") is not None:
        lines.append(f"FS3D L.Elbow: {meta['fs_le']:.0f}deg")
    if meta.get("fs_re") is not None:
        lines.append(f"FS3D R.Elbow: {meta['fs_re']:.0f}deg")
    if meta.get("err_le") is not None:
        lines.append(f"Error L: {meta['err_le']:.0f}deg")
    if meta.get("err_re") is not None:
        lines.append(f"Error R: {meta['err_re']:.0f}deg")

    # Draw text box top-left
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale, thick = 0.55, 1
    pad = 6
    box_h = len(lines) * 22 + pad * 2
    box_w = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for li, txt in enumerate(lines):
        color = (255, 90, 90) if txt.startswith("SKT") else \
                (90, 200, 90) if txt.startswith("Xsens") else \
                (90, 160, 255) if txt.startswith("FS3D") else \
                (255, 220, 80)
        cv2.putText(frame, txt, (pad, pad + 18 + li * 22), font, fscale, color, thick, cv2.LINE_AA)

    # Frame index & conf
    info = f"SKT#{meta['frame_idx']}  conf_L={meta['conf_le']:.2f} conf_R={meta['conf_re']:.2f}"
    cv2.putText(frame, info, (pad, h - 10), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=6)
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="Minimum per-joint confidence (min of left/right)")
    parser.add_argument("--min-error", type=float, default=10.0,
                        help="Min angle error (deg) vs XsensFair to count as 'high error'")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.config.is_absolute():
        args.config = Path.cwd() / args.config
    if not args.run_dir.is_absolute():
        args.run_dir = Path.cwd() / args.run_dir

    config = load_config(args.config)
    dataset_cfg = section(config, "dataset")
    rotate_180 = bool(dataset_cfg.get("rotate_180", False))

    left_video = resolve_path(dataset_cfg.get("left_video"), must_exist=True)
    left_meta  = resolve_path(dataset_cfg.get("left_metadata"), must_exist=True)
    right_meta = resolve_path(dataset_cfg.get("right_metadata"), must_exist=True)
    ts_fmt = dataset_cfg.get("timestamp_format", "seconds_microseconds_columns")

    skt_cfg = section(config, "skt")
    if skt_cfg.get("use_existing_npz", False):
        npz_path = resolve_path(skt_cfg.get("existing_npz"), must_exist=True)
    else:
        npz_path = args.run_dir / skt_cfg.get("output_npz", "skt_pose_optimized.npz")

    print(f"[viz] Loading SKT NPZ: {npz_path}")
    npz = np.load(str(npz_path), allow_pickle=False)
    kps_2d  = np.asarray(npz["keypoints_left_2d"], dtype=np.float64)
    conf_l  = np.asarray(npz["conf_left"],  dtype=np.float64)
    conf_r  = np.asarray(npz["conf_right"], dtype=np.float64)

    # Build synced timeline to map SKT frame index → video frame position
    print("[viz] Parsing metadata for frame mapping …")
    _, synced, _, _ = build_synced_timeline(left_meta, right_meta, ts_fmt)
    n_skt = len(kps_2d)
    if len(synced) < n_skt:
        n_skt = len(synced)

    angle_csv = args.run_dir / "angle_eval" / "angle_timeseries.csv"
    angle_data = _load_angle_timeseries(angle_csv)

    print(f"[viz] Selecting up to {args.n_frames} candidate frames …")
    candidates = find_candidate_frames(npz, angle_data, args.n_frames,
                                       min_conf=args.min_conf,
                                       min_error_deg=args.min_error)
    if not candidates:
        print("[viz] No candidates found — try lowering --min-conf or --min-error")
        return
    print(f"[viz] Found {len(candidates)} candidate frame(s)")

    out_dir = args.out_dir or (args.run_dir / "elbow_viz")
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cols = min(3, len(candidates))
    n_rows = math.ceil(len(candidates) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))
    axes_flat = np.array(axes).flatten()

    for ci, cand in enumerate(candidates):
        i = cand["frame_idx"]
        video_idx = synced[i].left_idx if i < len(synced) else i
        print(f"  frame {i} (video_frame={video_idx}): "
              f"conf_R={cand['conf_re']:.2f}  "
              f"skt_re={cand['skt_re']}  xsens_re={cand['xsens_re']}  "
              f"err_re={cand['err_re']}")
        img = render_frame(
            left_video, video_idx,
            kps_2d[i], conf_l[i], conf_r[i],
            cand, rotate_180=rotate_180,
        )
        ax = axes_flat[ci]
        ax.imshow(img)
        title_parts = [f"SKT frame {i}"]
        if cand["max_err"] is not None:
            title_parts.append(f"max_err={cand['max_err']:.0f}deg")
        else:
            title_parts.append("(no Xsens ref)")
        ax.set_title("  ".join(title_parts), fontsize=9)
        ax.axis("off")

    for ci in range(len(candidates), len(axes_flat)):
        axes_flat[ci].axis("off")

    fig.suptitle("SKT Elbow Visualization — high confidence frames\n"
                 "(green=left elbow, blue=right elbow; angles annotated from all systems)",
                 fontsize=11)
    plt.tight_layout()
    out_path = out_dir / "elbow_viz_grid.png"
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")

    # Also save individual frames
    for ci, cand in enumerate(candidates):
        i = cand["frame_idx"]
        video_idx = synced[i].left_idx if i < len(synced) else i
        img = render_frame(
            left_video, video_idx,
            kps_2d[i], conf_l[i], conf_r[i],
            cand, rotate_180=rotate_180,
        )
        single_path = out_dir / f"frame_{i:04d}_confl{cand['conf_le']:.2f}_confr{cand['conf_re']:.2f}.png"
        cv2.imwrite(str(single_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[viz] Individual frames saved to {out_dir}/")


if __name__ == "__main__":
    main()
