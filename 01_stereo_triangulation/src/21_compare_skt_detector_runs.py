"""Compare small SKT detector-backend runs.

This utility summarizes detector-backend experiments before we commit to a full
sequence run. It is intentionally reference-free: it compares internal SKT
stability and quality signals only.
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


DEFAULT_OUTPUT_DIR = _METHOD_DIR / "results" / "skt_model_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Detector run in LABEL=path/to/pose.npz format. Can be repeated.",
    )
    parser.add_argument("--limit-frames", type=int, default=0, help="Compare only the first N frames if >0.")
    parser.add_argument("--high-delta-deg", type=float, default=35.0, help="High-delta threshold.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be LABEL=path, got: {spec}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path)


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def optional_array(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int]) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=np.float64)
    return np.full(shape, np.nan, dtype=np.float64)


def high_delta_rate(angle_values: np.ndarray, angle_idx: int, k: int, high_delta_deg: float) -> float:
    if len(angle_values) <= k:
        return float("nan")
    values = angle_values[:, angle_idx]
    delta = np.abs(values[k:] - values[:-k])
    valid = np.isfinite(delta)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(delta[valid] >= high_delta_deg))


def summarize_run(label: str, path: Path, limit_frames: int, high_delta_deg: float) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")
    data = np.load(path, allow_pickle=True)
    keypoints = np.asarray(data["keypoints"], dtype=np.float64)
    if limit_frames > 0:
        keypoints = keypoints[:limit_frames]
    n_frames, n_joints = keypoints.shape[:2]
    shape = (n_frames, n_joints)

    epipolar = optional_array(data, "epipolar_error", data["keypoints"].shape[:2])[:n_frames]
    reprojection = optional_array(data, "reprojection_error", data["keypoints"].shape[:2])[:n_frames]
    stereo_quality = optional_array(data, "stereo_quality", data["keypoints"].shape[:2])[:n_frames]
    pair_conf = optional_array(data, "pair_confidence", data["keypoints"].shape[:2])[:n_frames]

    valid_joint = np.isfinite(keypoints).all(axis=2)
    left_elbow_chain = np.all(valid_joint[:, [5, 7, 9]], axis=1)
    right_elbow_chain = np.all(valid_joint[:, [6, 8, 10]], axis=1)
    angle_names, angle_values = compute_semantic_angle_sequence(keypoints, wrist_smooth_radius=0)
    name_to_idx = {name: idx for idx, name in enumerate(angle_names)}

    row: dict[str, object] = {
        "label": label,
        "path": str(path),
        "n_frames": n_frames,
        "valid_joint_ratio": float(np.mean(valid_joint)),
        "left_elbow_chain_valid_ratio": float(np.mean(left_elbow_chain)),
        "right_elbow_chain_valid_ratio": float(np.mean(right_elbow_chain)),
        "epipolar_p50_px": finite_percentile(epipolar, 50),
        "epipolar_p90_px": finite_percentile(epipolar, 90),
        "epipolar_p95_px": finite_percentile(epipolar, 95),
        "reprojection_p50_px": finite_percentile(reprojection, 50),
        "reprojection_p90_px": finite_percentile(reprojection, 90),
        "reprojection_p95_px": finite_percentile(reprojection, 95),
        "stereo_quality_mean": finite_mean(stereo_quality),
        "pair_conf_mean": finite_mean(pair_conf),
    }

    for angle_name in ("LeftElbow", "RightElbow"):
        angle_idx = name_to_idx.get(angle_name)
        if angle_idx is None:
            continue
        key = "left" if angle_name == "LeftElbow" else "right"
        row[f"{key}_k1_high_delta_rate"] = high_delta_rate(angle_values, angle_idx, 1, high_delta_deg)
        row[f"{key}_k6_high_delta_rate"] = high_delta_rate(angle_values, angle_idx, 6, high_delta_deg)
    return row


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def write_outputs(rows: list[dict[str, object]], output_dir: Path, high_delta_deg: float, limit_frames: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "detector_smoke_comparison.csv"
    md_path = output_dir / "detector_smoke_comparison.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        ("label", "Run"),
        ("n_frames", "Frames"),
        ("valid_joint_ratio", "Valid joints"),
        ("left_elbow_chain_valid_ratio", "L elbow chain"),
        ("right_elbow_chain_valid_ratio", "R elbow chain"),
        ("epipolar_p90_px", "Epi p90 px"),
        ("reprojection_p90_px", "Reproj p90 px"),
        ("stereo_quality_mean", "Stereo quality"),
        ("pair_conf_mean", "Pair conf"),
        ("left_k1_high_delta_rate", "L K1 high"),
        ("right_k1_high_delta_rate", "R K1 high"),
        ("left_k6_high_delta_rate", "L K6 high"),
        ("right_k6_high_delta_rate", "R K6 high"),
    ]

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# SKT Detector Smoke Comparison\n\n")
        handle.write(f"- High-delta threshold: `{high_delta_deg:.1f} deg`\n")
        if limit_frames > 0:
            handle.write(f"- Compared first `{limit_frames}` frames from each run.\n")
        handle.write("- Reference-free summary: Xsens is not used in this diagnostic.\n\n")
        handle.write("| " + " | ".join(title for _, title in columns) + " |\n")
        handle.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row.get(key, "nan")) for key, _ in columns) + " |\n")
        handle.write("\n## Notes\n\n")
        handle.write(
            "- Lower epipolar/reprojection percentiles and higher stereo-quality values generally indicate cleaner stereo geometry.\n"
        )
        handle.write(
            "- High-delta rates are only a quick smoke-test signal on short clips; full-sequence frame-delta evaluation is still required.\n"
        )
    print(f"[Info] Wrote {csv_path}")
    print(f"[Info] Wrote {md_path}")


def main() -> None:
    args = parse_args()
    runs = [parse_run_spec(spec) for spec in args.run]
    rows = [
        summarize_run(label, path, args.limit_frames, args.high_delta_deg)
        for label, path in runs
    ]
    write_outputs(rows, args.output_dir, args.high_delta_deg, args.limit_frames)


if __name__ == "__main__":
    main()
