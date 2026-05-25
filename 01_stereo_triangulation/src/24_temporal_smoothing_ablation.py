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
ELBOW_CHAINS = {
    "LeftElbow": np.array([5, 7, 9], dtype=np.int64),
    "RightElbow": np.array([6, 8, 10], dtype=np.int64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input raw SKT NPZ.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--high-delta-deg", type=float, default=35.0, help="High-delta threshold.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 6], help="Delta K values.")
    return parser.parse_args()


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


def build_variants(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Generate temporal smoothing variants from the same raw keypoints."""
    keypoints = np.asarray(data["keypoints"], dtype=np.float64)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
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


def write_outputs(rows: list[dict[str, object]], output_dir: Path, high_delta_deg: float) -> None:
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
        handle.write("| " + " | ".join(title for _, title in columns) + " |\n")
        handle.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row.get(key, "nan")) for key, _ in columns) + " |\n")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input NPZ not found: {args.input}")
    data = np.load(args.input, allow_pickle=True)
    variants = build_variants(data)
    rows: list[dict[str, object]] = []
    for label, keypoints in variants.items():
        rows.extend(summarize_variant(label, keypoints, args.k_values, args.high_delta_deg))
    write_outputs(rows, args.output_dir, args.high_delta_deg)
    print(f"[Info] Wrote temporal smoothing ablation: {args.output_dir}")


if __name__ == "__main__":
    main()
