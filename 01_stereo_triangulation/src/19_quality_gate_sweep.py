"""Scan SKT quality gates for elbow high-delta outlier diagnostics.

This script does not rerun 2D inference. It consumes an existing SKT NPZ file
and asks a narrower question: which available quality signal best identifies
frame pairs where the elbow angle delta is suspiciously large?

The output is intended to guide RTMPose/RTMO retuning and future model-prior
fusion. It keeps Xsens out of the loop and focuses only on internal SKT
stability, which makes the diagnostic easier to interpret.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_METHOD_DIR = _SRC_DIR.parent
PROJECT_ROOT = _METHOD_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from pose_angle_utils import compute_semantic_angle_sequence  # noqa: E402


DEFAULT_INPUT = _METHOD_DIR / "results" / "historical_best_20260324" / "recovered_baseline" / "optimized_pose.npz"
DEFAULT_OUTPUT_DIR = _METHOD_DIR / "results" / "skt_model_fusion"
ANGLE_JOINTS = {
    "LeftElbow": np.array([5, 7, 9], dtype=np.int64),
    "RightElbow": np.array([6, 8, 10], dtype=np.int64),
}


@dataclass(frozen=True)
class GateSpec:
    signal: str
    bad_tail: str
    thresholds: tuple[float, ...]


GATE_SPECS = (
    GateSpec("detect_conf_min", "low", (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)),
    GateSpec("pair_conf_min", "low", (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)),
    GateSpec("stereo_quality_min", "low", (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)),
    GateSpec("epipolar_error_max", "high", (3.0, 6.0, 10.0, 15.0, 20.0, 30.0)),
    GateSpec("reprojection_error_max", "high", (10.0, 20.0, 30.0, 40.0, 60.0, 80.0)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="SKT pose NPZ file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/MD output.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 6], help="Frame gaps used for angle deltas.")
    parser.add_argument(
        "--high-delta-deg",
        type=float,
        default=35.0,
        help="Angle-delta threshold used to define suspicious frame pairs.",
    )
    return parser.parse_args()


def finite_min(values: np.ndarray, joints: np.ndarray) -> np.ndarray:
    subset = values[:, joints]
    finite = np.isfinite(subset)
    result = np.full(len(values), np.nan, dtype=np.float64)
    for idx in range(len(values)):
        if np.any(finite[idx]):
            result[idx] = float(np.nanmin(subset[idx]))
    return result


def finite_max(values: np.ndarray, joints: np.ndarray) -> np.ndarray:
    subset = values[:, joints]
    finite = np.isfinite(subset)
    result = np.full(len(values), np.nan, dtype=np.float64)
    for idx in range(len(values)):
        if np.any(finite[idx]):
            result[idx] = float(np.nanmax(subset[idx]))
    return result


def get_array(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int]) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=np.float64)
    return np.full(shape, np.nan, dtype=np.float64)


def build_quality_signals(data: np.lib.npyio.NpzFile, angle_joints: np.ndarray) -> dict[str, np.ndarray]:
    keypoints = np.asarray(data["keypoints"], dtype=np.float64)
    num_frames, num_joints = keypoints.shape[:2]
    shape = (num_frames, num_joints)

    conf_left = get_array(data, "conf_left", shape)
    conf_right = get_array(data, "conf_right", shape)
    pair_conf = get_array(data, "pair_confidence", shape)
    if not np.isfinite(pair_conf).any():
        pair_conf = np.sqrt(np.clip(conf_left, 0.0, 1.0) * np.clip(conf_right, 0.0, 1.0))

    detect_conf = np.minimum(conf_left, conf_right)
    stereo_quality = get_array(data, "stereo_quality", shape)
    epipolar_error = get_array(data, "epipolar_error", shape)
    reprojection_error = get_array(data, "reprojection_error", shape)

    return {
        "detect_conf_min": finite_min(detect_conf, angle_joints),
        "pair_conf_min": finite_min(pair_conf, angle_joints),
        "stereo_quality_min": finite_min(stereo_quality, angle_joints),
        "epipolar_error_max": finite_max(epipolar_error, angle_joints),
        "reprojection_error_max": finite_max(reprojection_error, angle_joints),
    }


def evaluate_gate(
    angle_values: np.ndarray,
    bad_frame_mask: np.ndarray,
    k: int,
    high_delta_deg: float,
) -> dict[str, float]:
    if k <= 0 or len(angle_values) <= k:
        return {}

    delta = np.abs(angle_values[k:] - angle_values[:-k])
    valid = np.isfinite(delta)
    if not np.any(valid):
        return {}

    bad_pair = bad_frame_mask[k:] | bad_frame_mask[:-k]
    bad_pair = bad_pair[valid]
    delta = delta[valid]
    high_delta = delta >= high_delta_deg

    n_pairs = int(delta.size)
    n_high = int(np.count_nonzero(high_delta))
    n_flagged = int(np.count_nonzero(bad_pair))
    n_retained = n_pairs - n_flagged
    n_captured = int(np.count_nonzero(high_delta & bad_pair))
    n_high_retained = int(np.count_nonzero(high_delta & ~bad_pair))
    n_high_flagged = int(np.count_nonzero(high_delta & bad_pair))

    high_rate_all = n_high / n_pairs if n_pairs else np.nan
    high_rate_retained = n_high_retained / n_retained if n_retained else np.nan
    high_rate_flagged = n_high_flagged / n_flagged if n_flagged else np.nan

    return {
        "n_pairs": n_pairs,
        "high_delta_count": n_high,
        "high_delta_rate_all": high_rate_all,
        "flagged_pair_ratio": n_flagged / n_pairs if n_pairs else np.nan,
        "retained_pair_ratio": n_retained / n_pairs if n_pairs else np.nan,
        "captured_high_delta_ratio": n_captured / n_high if n_high else np.nan,
        "high_delta_rate_retained": high_rate_retained,
        "high_delta_rate_flagged": high_rate_flagged,
        "precision_lift": high_rate_flagged / high_rate_all if high_rate_all > 0 and np.isfinite(high_rate_flagged) else np.nan,
    }


def iter_gate_rows(
    data: np.lib.npyio.NpzFile,
    angle_names: Iterable[str],
    angles: np.ndarray,
    k_values: Iterable[int],
    high_delta_deg: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    name_to_idx = {name: idx for idx, name in enumerate(angle_names)}

    for angle_name, joints in ANGLE_JOINTS.items():
        if angle_name not in name_to_idx:
            continue
        angle_values = angles[:, name_to_idx[angle_name]]
        signals = build_quality_signals(data, joints)

        for spec in GATE_SPECS:
            signal_values = signals[spec.signal]
            for threshold in spec.thresholds:
                if spec.bad_tail == "low":
                    bad_frame_mask = np.isfinite(signal_values) & (signal_values < threshold)
                else:
                    bad_frame_mask = np.isfinite(signal_values) & (signal_values > threshold)

                for k in k_values:
                    metrics = evaluate_gate(angle_values, bad_frame_mask, int(k), high_delta_deg)
                    if not metrics:
                        continue
                    rows.append(
                        {
                            "angle": angle_name,
                            "k": int(k),
                            "signal": spec.signal,
                            "bad_tail": spec.bad_tail,
                            "threshold": float(threshold),
                            **metrics,
                        }
                    )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def write_markdown(rows: list[dict[str, object]], output_path: Path, input_path: Path, high_delta_deg: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eligible = [
        row
        for row in rows
        if float(row["retained_pair_ratio"]) >= 0.50
        and np.isfinite(float(row["captured_high_delta_ratio"]))
    ]
    top_rows = sorted(
        eligible,
        key=lambda row: (
            float(row["captured_high_delta_ratio"]),
            float(row["precision_lift"]) if np.isfinite(float(row["precision_lift"])) else -1.0,
            float(row["retained_pair_ratio"]),
        ),
        reverse=True,
    )[:12]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# SKT Quality-Gate Sweep for Elbow High-Delta Outliers\n\n")
        handle.write(f"- Input: `{input_path}`\n")
        handle.write(f"- High-delta threshold: `{high_delta_deg:.1f} deg`\n")
        handle.write("- Scope: LeftElbow and RightElbow only; no Xsens reference is used here.\n\n")
        handle.write("## Best candidate gates\n\n")
        handle.write(
            "| Angle | K | Signal | Bad if | Threshold | Retained | Captured high-delta | "
            "High-delta retained | Precision lift |\n"
        )
        handle.write("|---|---:|---|---|---:|---:|---:|---:|---:|\n")
        for row in top_rows:
            comparator = "<" if row["bad_tail"] == "low" else ">"
            handle.write(
                f"| {row['angle']} | {row['k']} | {row['signal']} | {comparator} | "
                f"{fmt_float(row['threshold'], 2)} | {fmt_float(row['retained_pair_ratio'])} | "
                f"{fmt_float(row['captured_high_delta_ratio'])} | "
                f"{fmt_float(row['high_delta_rate_retained'])} | {fmt_float(row['precision_lift'])} |\n"
            )
        handle.write("\n## How to read this\n\n")
        handle.write(
            "- `Captured high-delta` means how many suspicious large angle jumps would be flagged by this gate.\n"
        )
        handle.write(
            "- `Retained` means how much data remains if flagged frame pairs are removed or sent to a prior-assisted repair path.\n"
        )
        handle.write(
            "- A useful gate should capture many high-delta pairs without discarding most of the sequence.\n"
        )
        handle.write(
            "- This sweep is a diagnostic layer for SKT jitter; it should be paired with the frame-delta comparison report later.\n"
        )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input pose file not found: {args.input}")

    data = np.load(args.input, allow_pickle=True)
    keypoints = np.asarray(data["keypoints"], dtype=np.float64)
    angle_names, angles = compute_semantic_angle_sequence(keypoints, wrist_smooth_radius=0)
    rows = iter_gate_rows(data, angle_names, angles, args.k_values, args.high_delta_deg)

    csv_path = args.output_dir / "quality_gate_sweep_elbow.csv"
    md_path = args.output_dir / "quality_gate_sweep_elbow.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args.input, args.high_delta_deg)
    print(f"[Info] Wrote {csv_path}")
    print(f"[Info] Wrote {md_path}")


if __name__ == "__main__":
    main()
