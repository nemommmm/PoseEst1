#!/opt/anaconda3/envs/pose/bin/python
"""Apply One-Euro Filter to SKT 3D keypoints sequence (post-process baseline).

Loads an existing SKT pose NPZ, applies temporal One-Euro smoothing to the
3D keypoint sequence, and writes a new NPZ preserving all original arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from pose_postprocess import OneEuroFilter  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-npz",
        default=str(
            PROJECT_ROOT
            / "01_stereo_triangulation"
            / "results"
            / "historical_best_20260324"
            / "recovered_baseline"
            / "optimized_pose.npz"
        ),
        help="Path to source SKT pose NPZ containing 'keypoints' and 'timestamps'.",
    )
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Path to write the smoothed NPZ.",
    )
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=1.0,
        help="One-Euro low-pass minimum cutoff frequency (Hz).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="One-Euro speed coefficient.",
    )
    parser.add_argument(
        "--d-cutoff",
        type=float,
        default=1.0,
        help="One-Euro derivative cutoff frequency (Hz).",
    )
    return parser.parse_args()


def apply_one_euro_to_keypoints(
    keypoints: np.ndarray,
    timestamps: np.ndarray,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    """Apply One-Euro filter along the time axis of a (T, J, 3) keypoints sequence."""
    if keypoints.ndim != 3:
        raise ValueError(f"keypoints must be (T, J, 3); got {keypoints.shape}")
    shape = keypoints.shape[1:]
    filt = OneEuroFilter(shape=shape, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    smoothed = np.full_like(keypoints, np.nan, dtype=np.float64)
    for idx in range(len(keypoints)):
        smoothed[idx] = filt(float(timestamps[idx]), keypoints[idx])
    return smoothed


def main() -> None:
    """Entry point."""
    args = parse_args()
    input_path = Path(args.input_npz)
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = np.load(input_path, allow_pickle=True)
    if "keypoints" not in payload.files:
        raise RuntimeError(f"{input_path} does not contain 'keypoints'.")
    if "timestamps" not in payload.files:
        raise RuntimeError(f"{input_path} does not contain 'timestamps'.")

    keypoints = np.asarray(payload["keypoints"], dtype=np.float64)
    timestamps = np.asarray(payload["timestamps"], dtype=np.float64)

    print(f"[input] {input_path}")
    print(f"[shape] keypoints={keypoints.shape}, timestamps={timestamps.shape}")
    print(f"[oneeuro] min_cutoff={args.min_cutoff}, beta={args.beta}, d_cutoff={args.d_cutoff}")

    smoothed = apply_one_euro_to_keypoints(
        keypoints=keypoints,
        timestamps=timestamps,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        d_cutoff=args.d_cutoff,
    )

    valid_before = float(np.mean(np.isfinite(keypoints).all(axis=2)))
    valid_after = float(np.mean(np.isfinite(smoothed).all(axis=2)))
    diff = smoothed - keypoints
    diff_mag = np.linalg.norm(diff, axis=2)
    finite_diff = diff_mag[np.isfinite(diff_mag)]
    print(f"[stats] valid frame fraction before={valid_before:.4f}, after={valid_after:.4f}")
    if finite_diff.size:
        print(
            "[stats] keypoint shift (cm): "
            f"mean={np.mean(finite_diff):.3f}, median={np.median(finite_diff):.3f}, "
            f"p95={np.percentile(finite_diff, 95):.3f}, max={np.max(finite_diff):.3f}"
        )

    out_payload = {key: payload[key] for key in payload.files}
    out_payload["keypoints"] = smoothed
    out_payload["one_euro_min_cutoff"] = np.array(args.min_cutoff, dtype=np.float64)
    out_payload["one_euro_beta"] = np.array(args.beta, dtype=np.float64)
    out_payload["one_euro_d_cutoff"] = np.array(args.d_cutoff, dtype=np.float64)
    np.savez_compressed(output_path, **out_payload)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
