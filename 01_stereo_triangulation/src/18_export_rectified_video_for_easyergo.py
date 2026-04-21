#!/opt/anaconda3/envs/pose/bin/python
"""Export an upright + rectified monocular video for EasyErgo experiments.

The current EasyErgo upload uses an upright-only left video. This helper exports
an explicitly rectified monocular view using the same stereo calibration maps as
the SKT pipeline, which makes it easy to test whether rectification helps the
external workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent

DEFAULT_INPUT_LEFT = PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.avi"
DEFAULT_INPUT_RIGHT = PROJECT_ROOT / "2025_Ergonomics_Data" / "1_video_right.avi"
DEFAULT_OUTPUT_LEFT = METHOD_DIR / "results" / "easyergo_ready" / "0_video_left_rectified.avi"
DEFAULT_OUTPUT_RIGHT = METHOD_DIR / "results" / "easyergo_ready" / "1_video_right_rectified.avi"
DEFAULT_PARAM_PATH = PROJECT_ROOT / "shared" / "camera_params.npz"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export an upright + rectified stereo video for EasyErgo.")
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Which camera stream to export.",
    )
    parser.add_argument(
        "--input-video",
        default=None,
        help="Optional override input video path.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional override output video path.",
    )
    parser.add_argument(
        "--camera-params",
        default=str(DEFAULT_PARAM_PATH),
        help="Stereo calibration npz.",
    )
    parser.add_argument(
        "--keep-upside-down",
        action="store_true",
        help="Do not apply the 180-degree upright rotation before rectification.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    parser.add_argument(
        "--codec",
        default="XVID",
        help="FourCC codec for AVI output.",
    )
    return parser.parse_args()


def build_rectification_maps(param_path: Path, frame_size: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Build stereo rectification maps from the saved calibration."""
    payload = np.load(param_path)
    mtx_l = payload["mtx_l"]
    dist_l = payload["dist_l"]
    mtx_r = payload["mtx_r"]
    dist_r = payload["dist_r"]
    rot = payload["R"]
    trans = payload["T"]
    width, height = frame_size
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        (width, height),
        rot,
        trans,
        alpha=0,
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, r1, p1, (width, height), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, r2, p2, (width, height), cv2.CV_32FC1)
    return map1_l, map2_l, map1_r, map2_r


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve input/output paths for the chosen side."""
    if args.side == "left":
        default_input = DEFAULT_INPUT_LEFT
        default_output = DEFAULT_OUTPUT_LEFT
    else:
        default_input = DEFAULT_INPUT_RIGHT
        default_output = DEFAULT_OUTPUT_RIGHT
    input_path = Path(args.input_video).resolve() if args.input_video else default_input
    output_path = Path(args.output_video).resolve() if args.output_video else default_output
    return input_path, output_path


def main() -> None:
    """Export one rectified video."""
    args = parse_args()
    input_path, output_path = resolve_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 12.5
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame from {input_path}")
    if not args.keep_upside_down:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    height, width = frame.shape[:2]

    map1_l, map2_l, map1_r, map2_r = build_rectification_maps(Path(args.camera_params), (width, height))
    map1, map2 = (map1_l, map2_l) if args.side == "left" else (map1_r, map2_r)

    fourcc = cv2.VideoWriter_fourcc(*args.codec[:4])
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for: {output_path}")

    processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if not args.keep_upside_down:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        rectified = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        writer.write(rectified)
        processed += 1
        if args.max_frames is not None and processed >= args.max_frames:
            break

    cap.release()
    writer.release()

    print(f"[saved] {output_path}")
    print(f"[info] side={args.side}, frames={processed}, fps={fps:.4f}, upright_applied={not args.keep_upside_down}")


if __name__ == "__main__":
    main()
