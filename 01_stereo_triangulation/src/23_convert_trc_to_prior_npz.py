"""Convert an external TRC skeleton file into a timeline-aligned prior NPZ.

The converter intentionally reuses the frame-delta evaluation timeline logic:
when a TRC file has the same length as the left-camera metadata, synchronized
stereo frame pairs select the matching left-frame rows. This keeps FastSAM3D
priors aligned to the same corrected SKT timeline used in the 05 evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAME_DELTA_SCRIPT = PROJECT_ROOT / "04_frame_delta_eval" / "src" / "01_compute_elbow_deltas.py"
DEFAULT_SKT = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
DEFAULT_LEFT_META = PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.txt"
DEFAULT_RIGHT_META = PROJECT_ROOT / "2025_Ergonomics_Data" / "1_video_right.txt"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "skt_model_fusion"
    / "fastsam3d_unfiltered_prior.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trc", type=Path, required=True, help="Input TRC skeleton file.")
    parser.add_argument("--name", default="FastSAM3D", help="Source name stored in the NPZ metadata.")
    parser.add_argument("--skt-npz", type=Path, default=DEFAULT_SKT, help="SKT NPZ used only for target length.")
    parser.add_argument("--left-meta", type=Path, default=DEFAULT_LEFT_META, help="Left camera metadata txt.")
    parser.add_argument("--right-meta", type=Path, default=DEFAULT_RIGHT_META, help="Right camera metadata txt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output prior NPZ path.")
    return parser.parse_args()


def load_frame_delta_module() -> ModuleType:
    """Load the 05 frame-delta script as a helper module."""
    spec = importlib.util.spec_from_file_location("frame_delta_compute", FRAME_DELTA_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frame-delta helper from {FRAME_DELTA_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if not args.trc.exists():
        raise FileNotFoundError(f"TRC file not found: {args.trc}")
    if not args.skt_npz.exists():
        raise FileNotFoundError(f"SKT NPZ not found: {args.skt_npz}")

    helper = load_frame_delta_module()
    skt_payload = np.load(args.skt_npz, allow_pickle=True)
    skt_frame_count = int(np.asarray(skt_payload["keypoints"]).shape[0])

    corrected_time, synced_meta = helper.build_synced_video_timeline(args.left_meta, args.right_meta)
    corrected_time, synced_meta = helper.truncate_to_pose_length(corrected_time, synced_meta, skt_frame_count)
    left_rows = helper.parse_stereo_meta(args.left_meta)
    source = helper.TRCSource(args.name, args.trc)
    keypoints, summary = helper.load_trc_keypoints_on_timeline(
        source,
        corrected_time=corrected_time,
        synced_meta=synced_meta,
        left_rows=left_rows,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        timestamps=corrected_time,
        keypoints=keypoints,
        source_name=np.array(args.name),
        source_path=np.array(str(args.trc)),
        source_format=np.array("trc"),
        coordinate_units=np.array("cm"),
        alignment_summary_json=np.array(json.dumps(summary, indent=2)),
    )

    valid_ratio = float(np.mean(np.isfinite(keypoints).all(axis=2))) if keypoints.size else 0.0
    print(f"[Info] Wrote timeline-aligned prior: {args.output}")
    print(f"[Info] Frames: {len(keypoints)} | valid joint ratio: {valid_ratio:.3f}")
    print(f"[Info] Alignment mode: {summary['alignment_mode']}")


if __name__ == "__main__":
    main()
