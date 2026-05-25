"""Run reference-free temporal smoothing ablations on an SKT raw NPZ file.

This script separates simple temporal/bone-prior post-processing effects from
the detector itself. It is intentionally reference-free: it reports internal
stability signals such as elbow high-delta rates, not accuracy against Xsens.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_METHOD_DIR = _SRC_DIR.parent
PROJECT_ROOT = _METHOD_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from pose_angle_utils import compute_semantic_angle_sequence  # noqa: E402
from pose_postprocess import estimate_bone_priors, postprocess_sequence  # noqa: E402


DEFAULT_INPUT = (
    _METHOD_DIR
    / "results"
    / "skt_model_fusion"
    / "yolo_120"
    / "skt_3d_raw_yolo_yolov8m_pose.npz"
)
DEFAULT_OUTPUT_DIR = _METHOD_DIR / "results" / "skt_model_fusion" / "temporal_smoothing_ablation"
DEFAULT_LEFT_META = PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.txt"
DEFAULT_RIGHT_META = PROJECT_ROOT / "2025_Ergonomics_Data" / "1_video_right.txt"
ELBOW_CHAINS = {
    "LeftElbow": np.array([5, 7, 9], dtype=np.int64),
    "RightElbow": np.array([6, 8, 10], dtype=np.int64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input raw SKT NPZ.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--left-meta", type=Path, default=DEFAULT_LEFT_META, help="Left camera metadata txt.")
    parser.add_argument("--right-meta", type=Path, default=DEFAULT_RIGHT_META, help="Right camera metadata txt.")
    parser.add_argument(
        "--timeline-source",
        choices=("corrected", "npz"),
        default="corrected",
        help="Timeline for temporal filters; corrected uses synchronized stereo metadata.",
    )
    parser.add_argument("--high-delta-deg", type=float, default=35.0, help="High-delta threshold.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 6], help="Delta K values.")
    parser.add_argument(
        "--write-variant-npz",
        action="store_true",
        help="Also save one NPZ per smoothing variant for downstream frame-delta evaluation.",
    )
    return parser.parse_args()


def parse_meta_timestamp(parts: list[str]) -> float:
    """Parse metadata seconds + microseconds without losing leading zeros."""
    return int(parts[1]) + int(parts[2]) * 1e-6


def parse_stereo_meta(path: Path) -> list[dict[str, float | int]]:
    """Parse a stereo metadata txt file."""
    rows: list[dict[str, float | int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append({"id": int(parts[0]), "ts": parse_meta_timestamp(parts)})
            except ValueError:
                continue
    return rows


def build_synced_video_timeline(left_meta: Path, right_meta: Path) -> np.ndarray:
    """Build the corrected relative timeline from synchronized hardware frame IDs."""
    left_rows = parse_stereo_meta(left_meta)
    right_rows = parse_stereo_meta(right_meta)
    synced_ts: list[float] = []
    ptr_l = 0
    ptr_r = 0
    while ptr_l < len(left_rows) and ptr_r < len(right_rows):
        left_id = int(left_rows[ptr_l]["id"])
        right_id = int(right_rows[ptr_r]["id"])
        if left_id == right_id:
            synced_ts.append(float(left_rows[ptr_l]["ts"]))
            ptr_l += 1
            ptr_r += 1
        elif left_id < right_id:
            ptr_l += 1
        else:
            ptr_r += 1
    if not synced_ts:
        raise RuntimeError("No synchronized stereo metadata pairs found.")
    timeline = np.asarray(synced_ts, dtype=np.float64)
    timeline = timeline - timeline[0]
    if np.any(np.diff(timeline) <= 0):
        raise RuntimeError("Corrected stereo timeline is not strictly monotonic.")
    return timeline


def resolve_timestamps(
    data: np.lib.npyio.NpzFile,
    n_frames: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    """Resolve the temporal-filter timeline and report diagnostics."""
    original = np.asarray(data["timestamps"], dtype=np.float64)
    diffs = np.diff(original)
    diagnostics: dict[str, object] = {
        "timeline_source": args.timeline_source,
        "original_nonpositive_diff_count": int(np.sum(diffs <= 0)),
        "original_diff_median_s": float(np.nanmedian(diffs)) if len(diffs) else np.nan,
    }
    if args.timeline_source == "npz":
        timestamps = original.copy()
    else:
        corrected = build_synced_video_timeline(args.left_meta, args.right_meta)
        if len(corrected) < n_frames:
            raise RuntimeError(f"Corrected timeline has {len(corrected)} frames but pose has {n_frames}.")
        timestamps = corrected[:n_frames]
        diagnostics["corrected_duration_s"] = float(timestamps[-1] - timestamps[0]) if len(timestamps) else 0.0
        diagnostics["corrected_median_dt_s"] = float(np.nanmedian(np.diff(timestamps))) if len(timestamps) > 1 else np.nan
    return timestamps, diagnostics


def optional_array(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int]) -> np.ndarray | None:
    if key not in data:
        return None
    values = np.asarray(data[key], dtype=np.float64)
    if values.shape[:2] != shape:
        return None
    return values


def run_kalman_filter(
    keypoints: np.ndarray,
    timestamps: np.ndarray,
    process_var: float = 4.0,
    measurement_var: float = 25.0,
) -> np.ndarray:
    """Apply a simple constant-velocity Kalman filter per joint axis."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    filtered = np.full_like(keypoints, np.nan, dtype=np.float64)
    state = np.full((keypoints.shape[1], keypoints.shape[2], 2), np.nan, dtype=np.float64)
    cov = np.zeros((keypoints.shape[1], keypoints.shape[2], 2, 2), dtype=np.float64)

    for frame_idx, pose in enumerate(keypoints):
        if frame_idx == 0:
            dt = 1.0 / 12.5
        else:
            dt = max(float(timestamps[frame_idx] - timestamps[frame_idx - 1]), 1e-3)
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        process = process_var * np.array(
            [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]],
            dtype=np.float64,
        )

        for joint_idx in range(keypoints.shape[1]):
            for axis_idx in range(keypoints.shape[2]):
                measurement = pose[joint_idx, axis_idx]
                current_state = state[joint_idx, axis_idx]
                current_cov = cov[joint_idx, axis_idx]

                if not np.isfinite(current_state).all():
                    if np.isfinite(measurement):
                        state[joint_idx, axis_idx] = np.array([measurement, 0.0], dtype=np.float64)
                        cov[joint_idx, axis_idx] = np.eye(2, dtype=np.float64) * measurement_var
                        filtered[frame_idx, joint_idx, axis_idx] = measurement
                    continue

                predicted_state = transition @ current_state
                predicted_cov = transition @ current_cov @ transition.T + process

                if np.isfinite(measurement):
                    innovation = measurement - predicted_state[0]
                    innovation_cov = predicted_cov[0, 0] + measurement_var
                    gain = predicted_cov[:, 0] / max(float(innovation_cov), 1e-9)
                    updated_state = predicted_state + gain * innovation
                    updated_cov = (np.eye(2, dtype=np.float64) - np.outer(gain, [1.0, 0.0])) @ predicted_cov
                else:
                    updated_state = predicted_state
                    updated_cov = predicted_cov

                state[joint_idx, axis_idx] = updated_state
                cov[joint_idx, axis_idx] = updated_cov
                filtered[frame_idx, joint_idx, axis_idx] = updated_state[0]

    return filtered


def build_variants(data: np.lib.npyio.NpzFile, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    """Generate temporal smoothing variants from the same raw keypoints."""
    keypoints = np.asarray(data["keypoints"], dtype=np.float64)
    priors = estimate_bone_priors(keypoints, timestamps=timestamps)
    shape = keypoints.shape[:2]
    reprojection = optional_array(data, "reprojection_error", shape)
    pair_conf = optional_array(data, "pair_confidence", shape)

    variants = {
        "raw": keypoints,
        "bone_only": postprocess_sequence(
            keypoints,
            timestamps,
            priors,
            reprojection_errors=reprojection,
            pair_confidence=pair_conf,
            enable_bone_constraint=True,
            enable_quality_blend=False,
            enable_one_euro=False,
        ),
        "one_euro_only": postprocess_sequence(
            keypoints,
            timestamps,
            priors,
            reprojection_errors=reprojection,
            pair_confidence=pair_conf,
            enable_bone_constraint=False,
            enable_quality_blend=False,
            enable_one_euro=True,
        ),
        "bone_plus_one_euro": postprocess_sequence(
            keypoints,
            timestamps,
            priors,
            reprojection_errors=reprojection,
            pair_confidence=pair_conf,
            enable_bone_constraint=True,
            enable_quality_blend=False,
            enable_one_euro=True,
        ),
        "kalman_only": run_kalman_filter(keypoints, timestamps),
    }
    return variants


def summarize_variant(
    label: str,
    keypoints: np.ndarray,
    k_values: list[int],
    high_delta_deg: float,
) -> list[dict[str, object]]:
    valid_joint = np.isfinite(keypoints).all(axis=2)
    angle_names, angle_values = compute_semantic_angle_sequence(keypoints, wrist_smooth_radius=0)
    name_to_idx = {name: idx for idx, name in enumerate(angle_names)}
    rows: list[dict[str, object]] = []
    for angle_name, joints in ELBOW_CHAINS.items():
        angle_idx = name_to_idx[angle_name]
        chain_valid = np.all(valid_joint[:, joints], axis=1)
        values = angle_values[:, angle_idx]
        for k in k_values:
            delta = np.abs(values[k:] - values[:-k]) if len(values) > k else np.array([])
            finite_delta = delta[np.isfinite(delta)]
            rows.append(
                {
                    "variant": label,
                    "angle": angle_name,
                    "k": int(k),
                    "valid_joint_ratio": float(np.mean(valid_joint)),
                    "elbow_chain_valid_ratio": float(np.mean(chain_valid)),
                    "n_delta_pairs": int(finite_delta.size),
                    "mean_delta_deg": float(np.mean(finite_delta)) if finite_delta.size else np.nan,
                    "p90_delta_deg": float(np.percentile(finite_delta, 90)) if finite_delta.size else np.nan,
                    "p95_delta_deg": float(np.percentile(finite_delta, 95)) if finite_delta.size else np.nan,
                    "high_delta_rate": float(np.mean(finite_delta >= high_delta_deg)) if finite_delta.size else np.nan,
                }
            )
    return rows


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def write_outputs(
    rows: list[dict[str, object]],
    output_dir: Path,
    high_delta_deg: float,
    timeline_info: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "temporal_smoothing_ablation.csv"
    md_path = output_dir / "temporal_smoothing_ablation.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        ("variant", "Variant"),
        ("angle", "Angle"),
        ("k", "K"),
        ("elbow_chain_valid_ratio", "Elbow valid"),
        ("mean_delta_deg", "Mean delta"),
        ("p95_delta_deg", "P95 delta"),
        ("high_delta_rate", "High-delta rate"),
    ]
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Temporal Smoothing Ablation\n\n")
        handle.write(f"- High-delta threshold: `{high_delta_deg:.1f} deg`\n")
        handle.write("- Reference-free diagnostic: Xsens is not used here.\n\n")
        handle.write(
            f"- Timeline: `{timeline_info['timeline_source']}` "
            f"(original non-positive timestamp diffs: `{timeline_info['original_nonpositive_diff_count']}`).\n\n"
        )
        handle.write("| " + " | ".join(title for _, title in columns) + " |\n")
        handle.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row.get(key, "nan")) for key, _ in columns) + " |\n")


def write_variant_npz_files(
    data: np.lib.npyio.NpzFile,
    variants: dict[str, np.ndarray],
    timestamps: np.ndarray,
    output_dir: Path,
    timeline_info: dict[str, object],
) -> None:
    """Write each variant as an NPZ compatible with downstream evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    original_timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    for label, keypoints in variants.items():
        payload = {key: data[key] for key in data.files}
        payload["timestamps_original_before_temporal_smoothing_ablation"] = original_timestamps
        payload["timestamps"] = timestamps
        payload["keypoints"] = keypoints
        payload["temporal_smoothing_ablation_variant"] = np.array(label)
        payload["temporal_smoothing_ablation_timeline_source"] = np.array(str(timeline_info["timeline_source"]))
        payload["postprocess_variant"] = np.array(f"skt_temporal_smoothing_ablation_{label}")
        np.savez(output_dir / f"{label}.npz", **payload)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input NPZ not found: {args.input}")
    data = np.load(args.input, allow_pickle=True)
    keypoints = np.asarray(data["keypoints"], dtype=np.float64)
    timestamps, timeline_info = resolve_timestamps(data, len(keypoints), args)
    variants = build_variants(data, timestamps)
    rows: list[dict[str, object]] = []
    for label, keypoints in variants.items():
        rows.extend(summarize_variant(label, keypoints, args.k_values, args.high_delta_deg))
    write_outputs(rows, args.output_dir, args.high_delta_deg, timeline_info)
    if args.write_variant_npz:
        write_variant_npz_files(data, variants, timestamps, args.output_dir, timeline_info)
    print(f"[Info] Wrote temporal smoothing ablation: {args.output_dir}")


if __name__ == "__main__":
    main()
