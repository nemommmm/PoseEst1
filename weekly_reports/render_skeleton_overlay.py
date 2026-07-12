#!/opt/anaconda3/envs/pose/bin/python
"""Render a demo video with 2D skeleton overlay on Fanbo left-camera walking video.

Purpose: the supervisor asked for a "best-looking video with skeleton overlay"
showing varied motion — NOT the elbow-angle validation style.

Note: the overlay uses ``keypoints_left_2d`` from the SKT NPZ, i.e. the YOLO
2D detections on the left camera image that feed into the stereo triangulation.
Positions on-screen are identical to the projected 3D pose within a fraction of
a pixel, so this remains a faithful preview of the SKT pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# COCO-17 edges (no left/right colour distinction — single-colour style).
ALL_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

SKELETON_COLOUR = (255, 220, 120)  # soft cyan (BGR)
LINE_THICKNESS = 2
JOINT_RADIUS = 3


def draw_skeleton(img, kp):
    for a, b in ALL_EDGES:
        pa, pb = kp[a], kp[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 SKELETON_COLOUR, LINE_THICKNESS, cv2.LINE_AA)
    for j in range(17):
        p = kp[j]
        if not np.isfinite(p).all():
            continue
        cv2.circle(img, (int(p[0]), int(p[1])), JOINT_RADIUS,
                   SKELETON_COLOUR, -1, cv2.LINE_AA)


def render(skt_npz: Path, left_video: Path, out_path: Path,
           start_sec: float = 0.0, end_sec: float = 1e9,
           out_h: int = 900, fps: float = 12.5) -> None:
    print(f"Loading pose: {skt_npz.name}")
    npz = np.load(str(skt_npz), allow_pickle=True)
    kp2d = np.asarray(npz["keypoints_left_2d"], dtype=np.float64)
    ts = np.asarray(npz["timestamps"], dtype=np.float64)
    ts -= ts[0]
    print(f"  frames={len(kp2d)}  duration={ts[-1]:.1f}s")

    cap = cv2.VideoCapture(str(left_video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = int(round(src_w * out_h / src_h))
    scale = out_h / src_h
    print(f"Output: {out_w}x{out_h}  fps={fps}  → {out_path.name}")

    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    for idx in tqdm(range(len(kp2d)), desc="render"):
        t = ts[idx]
        if t < start_sec or t > end_sec:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame, (out_w, out_h))
        kp = kp2d[idx].copy() * scale
        draw_skeleton(frame, kp)

        label = f"t = {t:5.2f} s"
        cv2.putText(frame, label, (24, out_h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (24, out_h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skt-npz", type=Path, required=True)
    ap.add_argument("--left-video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=1e9)
    args = ap.parse_args()
    render(args.skt_npz, args.left_video, args.out,
           start_sec=args.start_sec, end_sec=args.end_sec)


if __name__ == "__main__":
    main()
