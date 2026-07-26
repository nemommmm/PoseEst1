#!/opt/anaconda3/envs/pose/bin/python
"""Gate one compressed-video SKT result against its raw-video reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = PROJECT_ROOT / "00_pose_pipeline_v2" / "src"

import sys

sys.path.insert(0, str(PIPELINE_SRC))

from common.candidate_evaluation import (  # noqa: E402
    finite_distribution,
    process_camera_angles,
)
from common.config import load_config, section  # noqa: E402
from common.metrics import rula_bin  # noqa: E402


def project_path(value: str | Path) -> Path:
    """Resolve one path against the project root."""
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def load_matrix(path: Path) -> dict[str, Any]:
    """Load matrix configuration."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping: {path}")
    return payload


def load_skt(path: Path) -> dict[str, np.ndarray]:
    """Load arrays required by the proxy gate."""
    required = (
        "timestamps",
        "keypoints",
        "keypoints_left_2d_raw",
        "keypoints_right_2d_raw",
    )
    with np.load(path, allow_pickle=True) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        return {
            key: np.asarray(payload[key], dtype=np.float64)
            for key in required
        }


def paired_distance(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return Euclidean distance wherever both arrays are finite."""
    count = min(len(candidate), len(reference))
    candidate_values = np.asarray(candidate[:count], dtype=np.float64)
    reference_values = np.asarray(reference[:count], dtype=np.float64)
    distance = np.linalg.norm(candidate_values - reference_values, axis=-1)
    valid = (
        np.isfinite(candidate_values).all(axis=-1)
        & np.isfinite(reference_values).all(axis=-1)
    )
    return np.where(valid, distance, np.nan)


def compare(
    matrix_path: Path,
    dataset_name: str,
    candidate_path: Path,
    reference_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compute all scientific equivalence gates."""
    matrix = load_matrix(matrix_path)
    dataset_spec = matrix["datasets"][dataset_name]
    config = load_config(project_path(dataset_spec["config"]))
    thresholds = matrix["proxy"]["thresholds"]
    angle_names = list(dataset_spec["angle_names"])
    candidate = load_skt(candidate_path)
    reference = load_skt(reference_path)
    count = min(len(candidate["timestamps"]), len(reference["timestamps"]))
    timestamp_match = bool(
        np.allclose(
            candidate["timestamps"][:count],
            reference["timestamps"][:count],
            atol=1e-6,
            rtol=0.0,
        )
    )
    distance_2d = np.concatenate(
        [
            paired_distance(
                candidate["keypoints_left_2d_raw"],
                reference["keypoints_left_2d_raw"],
            ).reshape(-1),
            paired_distance(
                candidate["keypoints_right_2d_raw"],
                reference["keypoints_right_2d_raw"],
            ).reshape(-1),
        ]
    )
    distance_3d = paired_distance(
        candidate["keypoints"],
        reference["keypoints"],
    )
    stats_2d = finite_distribution(distance_2d)
    stats_3d = finite_distribution(distance_3d)
    candidate_angles = process_camera_angles(
        candidate["keypoints"][:count],
        candidate["timestamps"][:count],
        config,
        angle_names,
    )
    reference_angles = process_camera_angles(
        reference["keypoints"][:count],
        reference["timestamps"][:count],
        config,
        angle_names,
    )
    bins_by_angle = section(config, "evaluation").get("rula_bins", {})
    angle_rows: list[dict[str, Any]] = []
    angle_pass = True
    rula_pass = True
    valid_pass = True
    for name in angle_names:
        candidate_values = candidate_angles[name]
        reference_values = reference_angles[name]
        common = np.isfinite(candidate_values) & np.isfinite(reference_values)
        absolute = np.where(
            common,
            np.abs(candidate_values - reference_values),
            np.nan,
        )
        error = finite_distribution(absolute)
        candidate_valid = float(np.mean(np.isfinite(candidate_values)))
        reference_valid = float(np.mean(np.isfinite(reference_values)))
        valid_change = abs(candidate_valid - reference_valid)
        bins = bins_by_angle.get(name)
        rula_agreement = None
        if bins and np.any(common):
            rula_agreement = float(
                np.mean(
                    rula_bin(candidate_values[common], bins)
                    == rula_bin(reference_values[common], bins)
                )
            )
            rula_pass &= (
                rula_agreement
                >= float(thresholds["minimum_rula_agreement"])
            )
        angle_pass &= bool(
            error["median"] is not None
            and error["p95"] is not None
            and float(error["median"])
            <= float(thresholds["angle_median_deg"])
            and float(error["p95"])
            <= float(thresholds["angle_p95_deg"])
        )
        valid_pass &= (
            valid_change
            <= float(thresholds["maximum_valid_ratio_change"])
        )
        angle_rows.append(
            {
                "angle": name,
                "absolute_difference_deg": error,
                "candidate_valid_ratio": candidate_valid,
                "reference_valid_ratio": reference_valid,
                "absolute_valid_ratio_change": valid_change,
                "rula_like_agreement": rula_agreement,
            }
        )
    geometry_pass = bool(
        stats_2d["median"] is not None
        and stats_2d["p95"] is not None
        and stats_3d["median"] is not None
        and stats_3d["p95"] is not None
        and float(stats_2d["median"])
        <= float(thresholds["keypoint_2d_median_px"])
        and float(stats_2d["p95"])
        <= float(thresholds["keypoint_2d_p95_px"])
        and float(stats_3d["median"])
        <= float(thresholds["keypoint_3d_median_cm"])
        and float(stats_3d["p95"])
        <= float(thresholds["keypoint_3d_p95_cm"])
    )
    passed = bool(
        timestamp_match
        and geometry_pass
        and angle_pass
        and rula_pass
        and valid_pass
    )
    result = {
        "schema_version": "pose_proxy_equivalence_v1",
        "dataset": dataset_name,
        "candidate": str(candidate_path),
        "reference": str(reference_path),
        "frame_count": count,
        "timestamp_match": timestamp_match,
        "keypoint_2d_distance_px": stats_2d,
        "keypoint_3d_distance_cm": stats_3d,
        "angles": angle_rows,
        "gates": {
            "geometry_passed": geometry_pass,
            "angle_passed": angle_pass,
            "rula_passed": rula_pass,
            "valid_ratio_passed": valid_pass,
            "passed": passed,
            "thresholds": thresholds,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "00_pose_pipeline_v2/configs/nvidia_pose_matrix.yaml"
        ),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison."""
    args = parse_args(argv)
    result = compare(
        project_path(args.matrix),
        args.dataset,
        project_path(args.candidate),
        project_path(args.reference),
        project_path(args.output),
    )
    print(json.dumps(result["gates"], indent=2))
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

