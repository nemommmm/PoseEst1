#!/usr/bin/env python
"""Run and audit repeated deterministic GPU benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from common.angles import compute_angle_sequence
from common.metrics import jsonable, rula_bin


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(
    "00_pose_pipeline_v2/configs/gpu_benchmark_repeats.yaml"
)
TIMING_KEYS = {
    "decode": "decode_time_ms",
    "pose_inference_stereo": "yolo_time_ms",
    "per_frame_geometry": "geometry_time_ms",
    "end_to_end_online": "frame_time_ms",
}


def parse_args() -> argparse.Namespace:
    """Parse suite configuration and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the timestamped output directory.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and hash all inputs without running inference.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete repeat outputs and run only missing repeats.",
    )
    return parser.parse_args()


def utc_now() -> str:
    """Return a timezone-aware ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a project-relative path without requiring it to exist."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    """Read a Git value or return an explicit unavailable marker."""
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def package_version(distribution: str) -> str:
    """Return one installed distribution version."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def collect_environment() -> dict[str, Any]:
    """Collect software, Git, CUDA, and GPU metadata."""
    metadata: dict[str, Any] = {
        "created_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "opencv-python",
                "opencv-python-headless",
                "torch",
                "ultralytics",
                "onnxruntime-gpu",
                "PyYAML",
            )
        },
        "selected_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "YOLO_CONFIG_DIR",
            )
        },
    }
    try:
        import cv2

        metadata["opencv_module_version"] = cv2.__version__
    except ImportError:
        metadata["opencv_module_version"] = "unavailable"
    try:
        import torch

        metadata["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
    except ImportError:
        metadata["cuda"] = {"available": False, "runtime": "unavailable"}
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu=name,uuid,memory.total,driver_version,"
                    "compute_cap"
                ),
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        metadata["nvidia_smi"] = [
            {
                "name": values[0],
                "uuid": values[1],
                "memory_total_mib": int(values[2]),
                "driver": values[3],
                "compute_capability": values[4],
            }
            for line in query.splitlines()
            if line.strip()
            for values in ([part.strip() for part in line.split(",")],)
        ]
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        IndexError,
        ValueError,
    ):
        metadata["nvidia_smi"] = []
    return metadata


def write_pip_freeze(output: Path) -> None:
    """Save the complete installed-package list for reproduction."""
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    content = completed.stdout
    if completed.returncode != 0:
        content += f"\n# pip freeze failed\n{completed.stderr}"
    output.write_text(content, encoding="utf-8")


def load_suite_config(path: Path) -> dict[str, Any]:
    """Load and validate the benchmark-suite YAML."""
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    suite = config.get("suite", {})
    datasets = config.get("datasets", [])
    if not isinstance(suite, dict) or not isinstance(datasets, list):
        raise ValueError("suite must be a mapping and datasets must be a list")
    for key in ("name", "benchmark_script", "repeats", "max_frames"):
        if key not in suite:
            raise ValueError(f"missing suite.{key}")
    if int(suite["repeats"]) < 1 or int(suite["max_frames"]) < 1:
        raise ValueError("repeats and max_frames must be positive")
    if int(suite.get("warmup_frames", 0)) >= int(suite["max_frames"]):
        raise ValueError("warmup_frames must be smaller than max_frames")
    names = [str(item.get("name", "")) for item in datasets]
    if not names or any(not name for name in names):
        raise ValueError("each dataset needs a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    for item in datasets:
        for key in ("config", "model", "historical_reference_npz"):
            if key not in item:
                raise ValueError(f"dataset {item['name']} is missing {key}")
    return config


def pipeline_input_paths(dataset: dict[str, Any]) -> dict[str, Path]:
    """Resolve every formal input used by one dataset benchmark."""
    pipeline_config = resolve_project_path(dataset["config"])
    payload = yaml.safe_load(pipeline_config.read_text(encoding="utf-8")) or {}
    data = payload.get("dataset", {})
    calibration = payload.get("calibration", {})
    paths = {
        "pipeline_config": pipeline_config,
        "left_video": resolve_project_path(data["left_video"]),
        "right_video": resolve_project_path(data["right_video"]),
        "left_metadata": resolve_project_path(data["left_metadata"]),
        "right_metadata": resolve_project_path(data["right_metadata"]),
        "camera_params": resolve_project_path(calibration["camera_params"]),
        "model": resolve_project_path(dataset["model"]),
        "historical_reference_npz": resolve_project_path(
            dataset["historical_reference_npz"]
        ),
    }
    return paths


def build_input_manifest(
    suite_config_path: Path,
    datasets: list[dict[str, Any]],
    benchmark_script: Path | None = None,
) -> dict[str, Any]:
    """Hash every configuration, video, model, calibration, and reference."""
    missing: list[str] = []
    records: list[dict[str, Any]] = []
    all_inputs: list[tuple[str, str, Path]] = [
        ("suite", "suite_config", suite_config_path)
    ]
    if benchmark_script is not None:
        all_inputs.append(("suite", "benchmark_script", benchmark_script))
    for dataset in datasets:
        all_inputs.extend(
            (str(dataset["name"]), role, path)
            for role, path in pipeline_input_paths(dataset).items()
        )
    for dataset_name, role, path in all_inputs:
        if not path.is_file():
            missing.append(str(path))
            continue
        records.append(
            {
                "dataset": dataset_name,
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "created_utc": utc_now(),
        "files": records,
        "missing": sorted(set(missing)),
        "ok": not missing,
    }


def build_benchmark_command(
    benchmark_script: Path,
    dataset: dict[str, Any],
    repeat_dir: Path,
    max_frames: int,
    warmup_frames: int,
) -> list[str]:
    """Build one invocation of the existing benchmark entry point."""
    return [
        sys.executable,
        str(benchmark_script),
        "--config",
        str(resolve_project_path(dataset["config"])),
        "--model",
        str(resolve_project_path(dataset["model"])),
        "--max-frames",
        str(max_frames),
        "--warmup-frames",
        str(warmup_frames),
        "--run-dir",
        str(repeat_dir),
        "--output-json",
        str(repeat_dir / "benchmark.json"),
    ]


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize finite values with central and tail statistics."""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if not clean.size:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(clean.size),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "p95": float(np.percentile(clean, 95)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def compare_array(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    """Compare one numeric, boolean, or string NPZ array."""
    left = np.asarray(candidate)
    right = np.asarray(reference)
    result: dict[str, Any] = {
        "candidate_shape": list(left.shape),
        "reference_shape": list(right.shape),
        "shape_match": left.shape == right.shape,
        "candidate_dtype": str(left.dtype),
        "reference_dtype": str(right.dtype),
    }
    if left.shape != right.shape:
        result.update({"exact": False, "agreement_ratio": None})
        return result
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(
        right.dtype, np.number
    ):
        left_float = left.astype(np.float64)
        right_float = right.astype(np.float64)
        finite_left = np.isfinite(left_float)
        finite_right = np.isfinite(right_float)
        finite_agreement = float(np.mean(finite_left == finite_right))
        common = finite_left & finite_right
        diff = np.abs(left_float[common] - right_float[common])
        exact = bool(
            np.array_equal(left_float, right_float, equal_nan=True)
        )
        result.update(
            {
                "exact": exact,
                "finite_mask_agreement": finite_agreement,
                "common_finite_count": int(np.count_nonzero(common)),
                "mean_abs_diff": float(np.mean(diff)) if diff.size else None,
                "max_abs_diff": float(np.max(diff)) if diff.size else None,
                "agreement_ratio": (
                    float(np.mean(left_float[common] == right_float[common]))
                    if diff.size
                    else None
                ),
            }
        )
        return result
    equality = left == right
    result.update(
        {
            "exact": bool(np.all(equality)),
            "agreement_ratio": float(np.mean(equality)),
            "finite_mask_agreement": 1.0,
        }
    )
    return result


def keypoint_distance_summary(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    """Summarize paired 3D keypoint distances in centimetres."""
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 3:
        return {
            "shape_match": False,
            "finite_mask_agreement": 0.0,
            "distance_cm": distribution(np.asarray([])),
        }
    finite_left = np.isfinite(left).all(axis=-1)
    finite_right = np.isfinite(right).all(axis=-1)
    common = finite_left & finite_right
    distances = np.linalg.norm(left[common] - right[common], axis=-1)
    return {
        "shape_match": True,
        "finite_mask_agreement": float(
            np.mean(finite_left == finite_right)
        ),
        "common_joint_count": int(np.count_nonzero(common)),
        "distance_cm": distribution(distances),
    }


def angle_and_rula_summary(
    candidate_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
    angle_name: str,
    bins: list[float],
) -> dict[str, Any]:
    """Compare one derived joint-angle trajectory and its RULA-like bins."""
    candidate = compute_angle_sequence(
        np.asarray(candidate_keypoints), [angle_name]
    )[angle_name]
    reference = compute_angle_sequence(
        np.asarray(reference_keypoints), [angle_name]
    )[angle_name]
    if candidate.shape != reference.shape:
        return {
            "shape_match": False,
            "common_finite_count": 0,
            "absolute_difference_deg": distribution(np.asarray([])),
            "rula_bin_agreement": None,
        }
    common = np.isfinite(candidate) & np.isfinite(reference)
    difference = np.abs(candidate[common] - reference[common])
    agreement = None
    if np.any(common):
        agreement = float(
            np.mean(
                rula_bin(candidate[common], bins)
                == rula_bin(reference[common], bins)
            )
        )
    return {
        "shape_match": True,
        "common_finite_count": int(np.count_nonzero(common)),
        "finite_mask_agreement": float(
            np.mean(np.isfinite(candidate) == np.isfinite(reference))
        ),
        "absolute_difference_deg": distribution(difference),
        "rula_bin_agreement": agreement,
    }


def comparison_passes(
    result: dict[str, Any],
    thresholds: dict[str, Any],
) -> bool:
    """Evaluate configured deterministic or historical tolerances."""
    key_arrays = result["key_arrays"]
    if thresholds.get("require_key_arrays_exact", False):
        if not all(item.get("exact", False) for item in key_arrays.values()):
            return False
    minimum_mask = float(
        thresholds.get("min_finite_mask_agreement", 0.0)
    )
    mask_values = [
        float(item["finite_mask_agreement"])
        for item in key_arrays.values()
        if item.get("finite_mask_agreement") is not None
    ]
    keypoint_mask = result["keypoints"].get("finite_mask_agreement")
    if keypoint_mask is not None:
        mask_values.append(float(keypoint_mask))
    if mask_values and min(mask_values) < minimum_mask:
        return False
    keypoint_p95 = result["keypoints"]["distance_cm"].get("p95")
    if (
        keypoint_p95 is None
        or float(keypoint_p95)
        > float(thresholds["max_keypoint_p95_distance_cm"])
    ):
        return False
    angle_mae = result["angle"]["absolute_difference_deg"].get("mean")
    if (
        angle_mae is None
        or float(angle_mae)
        > float(thresholds["max_right_elbow_mae_deg"])
    ):
        return False
    rula_agreement = result["angle"].get("rula_bin_agreement")
    if (
        rula_agreement is None
        or float(rula_agreement)
        < float(thresholds["min_rula_bin_agreement"])
    ):
        return False
    return True


def compare_npz(
    candidate_path: Path,
    reference_path: Path,
    key_arrays: Sequence[str],
    angle_name: str,
    rula_bins: list[float],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Compare key arrays, 3D joints, angle, and RULA-like bins."""
    with np.load(candidate_path, allow_pickle=True) as candidate, np.load(
        reference_path, allow_pickle=True
    ) as reference:
        array_results: dict[str, Any] = {}
        for name in key_arrays:
            if name not in candidate.files or name not in reference.files:
                array_results[name] = {
                    "present_in_candidate": name in candidate.files,
                    "present_in_reference": name in reference.files,
                    "exact": False,
                    "shape_match": False,
                }
                continue
            array_results[name] = compare_array(
                candidate[name], reference[name]
            )
        if "keypoints" not in candidate.files or "keypoints" not in reference.files:
            raise KeyError("both NPZ files must contain keypoints")
        keypoints = keypoint_distance_summary(
            candidate["keypoints"], reference["keypoints"]
        )
        angle = angle_and_rula_summary(
            candidate["keypoints"],
            reference["keypoints"],
            angle_name,
            rula_bins,
        )
    result = {
        "candidate": str(candidate_path),
        "reference": str(reference_path),
        "key_arrays": array_results,
        "key_arrays_all_exact": all(
            item.get("exact", False) for item in array_results.values()
        ),
        "keypoints": keypoints,
        "angle_name": angle_name,
        "angle": angle,
        "thresholds": thresholds,
    }
    result["passes_thresholds"] = comparison_passes(result, thresholds)
    return result


def pooled_timing_summary(
    npz_paths: Sequence[Path],
    warmup_frames: int,
    repeat_benchmarks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-frame latency and per-repeat FPS distributions."""
    stages: dict[str, Any] = {}
    for stage, key in TIMING_KEYS.items():
        pooled: list[np.ndarray] = []
        repeat_means: list[float] = []
        for path in npz_paths:
            with np.load(path, allow_pickle=True) as payload:
                if key not in payload.files:
                    continue
                values = np.asarray(payload[key], dtype=np.float64)
            clean = values[warmup_frames:]
            clean = clean[np.isfinite(clean)]
            if clean.size:
                pooled.append(clean)
                repeat_means.append(float(np.mean(clean)))
        combined = (
            np.concatenate(pooled) if pooled else np.asarray([], dtype=float)
        )
        stage_summary = {
            "pooled_frame_ms": distribution(combined),
            "repeat_mean_ms": distribution(np.asarray(repeat_means)),
        }
        if stage == "end_to_end_online":
            pooled_mean = stage_summary["pooled_frame_ms"]["mean"]
            stage_summary["pooled_fps"] = (
                float(1000.0 / pooled_mean)
                if pooled_mean not in (None, 0)
                else None
            )
        stages[stage] = stage_summary
    fps_values = np.asarray(
        [
            float(item["online_fps"])
            for item in repeat_benchmarks
            if item.get("online_fps") is not None
        ]
    )
    return {
        "successful_repeats": len(npz_paths),
        "online_fps_across_repeats": distribution(fps_values),
        "stages": stages,
    }


def write_repeat_csv(
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    """Write one compact spreadsheet-ready row per repeat."""
    fieldnames = [
        "dataset",
        "repeat",
        "return_code",
        "frames",
        "warmup_frames",
        "online_fps",
        "decode_mean_ms",
        "decode_median_ms",
        "decode_p95_ms",
        "pose_mean_ms",
        "pose_median_ms",
        "pose_p95_ms",
        "geometry_mean_ms",
        "geometry_median_ms",
        "geometry_p95_ms",
        "online_mean_ms",
        "online_median_ms",
        "online_p95_ms",
        "repeat_deterministic",
        "historical_match",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_row(
    dataset_name: str,
    repeat_index: int,
    return_code: int,
    benchmark: dict[str, Any] | None,
    repeat_match: bool | None,
    historical_match: bool | None,
) -> dict[str, Any]:
    """Flatten one benchmark record for CSV."""
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "repeat": repeat_index,
        "return_code": return_code,
        "repeat_deterministic": repeat_match,
        "historical_match": historical_match,
    }
    if benchmark is None:
        return row
    row.update(
        {
            "frames": benchmark.get("frames"),
            "warmup_frames": benchmark.get("warmup_frames"),
            "online_fps": benchmark.get("online_fps"),
        }
    )
    stage_columns = {
        "decode": "decode",
        "pose_inference_stereo": "pose",
        "per_frame_geometry": "geometry",
        "end_to_end_online": "online",
    }
    for stage, prefix in stage_columns.items():
        values = benchmark.get("stages", {}).get(stage, {})
        for statistic in ("mean_ms", "median_ms", "p95_ms"):
            row[f"{prefix}_{statistic}"] = values.get(statistic)
    return row


def write_output_manifest(run_dir: Path, command: str) -> Path:
    """Checksum every generated suite artifact."""
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "gpu_benchmark_suite_v1",
        "created_utc": utc_now(),
        "command": command,
        "git_commit": git_value("rev-parse", "HEAD"),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    output = run_dir / "artifact_manifest.json"
    output.write_text(
        json.dumps(jsonable(manifest), indent=2), encoding="utf-8"
    )
    return output


def benchmark_one_repeat(
    command: list[str],
    repeat_dir: Path,
) -> dict[str, Any]:
    """Execute one benchmark and retain its exact command and logs."""
    repeat_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    wall_start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    record = {
        "command": command,
        "command_text": shlex.join(command),
        "started_utc": started,
        "ended_utc": utc_now(),
        "wall_time_seconds": time.perf_counter() - wall_start,
        "return_code": completed.returncode,
    }
    (repeat_dir / "stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (repeat_dir / "stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    (repeat_dir / "execution.json").write_text(
        json.dumps(jsonable(record), indent=2), encoding="utf-8"
    )
    return record


def reusable_repeat_outputs(repeat_dir: Path) -> bool:
    """Return whether a repeat has both readable formal output files."""
    benchmark_path = repeat_dir / "benchmark.json"
    npz_path = repeat_dir / "skt_pose_optimized.npz"
    if not benchmark_path.is_file() or not npz_path.is_file():
        return False
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=True) as data:
            required_npz = {
                "timestamps",
                "keypoints",
                "frame_time_ms",
                "yolo_time_ms",
            }
            return (
                isinstance(payload.get("stages"), dict)
                and payload.get("online_fps") is not None
                and required_npz.issubset(data.files)
            )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def suite_output_dir(
    suite: dict[str, Any],
    override: Path | None,
) -> Path:
    """Build a descriptive timestamped output directory."""
    if override is not None:
        return resolve_project_path(override)
    root = resolve_project_path(suite["output_root"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = git_value("rev-parse", "--short=7", "HEAD")
    return root / f"{suite['name']}__{stamp}__{commit}"


def run_suite(
    config_path: Path,
    output_override: Path | None = None,
    preflight_only: bool = False,
    resume: bool = False,
) -> Path:
    """Run the configured datasets and write an auditable result bundle."""
    config_path = resolve_project_path(config_path)
    config = load_suite_config(config_path)
    suite = config["suite"]
    comparison = config.get("comparison", {})
    datasets = config["datasets"]
    run_dir = suite_output_dir(suite, output_override)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(jsonable(collect_environment()), indent=2),
        encoding="utf-8",
    )
    write_pip_freeze(run_dir / "pip_freeze.txt")
    benchmark_script = resolve_project_path(suite["benchmark_script"])
    inputs = build_input_manifest(
        config_path,
        datasets,
        benchmark_script=benchmark_script,
    )
    (run_dir / "input_manifest.json").write_text(
        json.dumps(jsonable(inputs), indent=2), encoding="utf-8"
    )
    invocation = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path)]
    )
    if output_override is not None:
        invocation += f" --output-dir {shlex.quote(str(output_override))}"
    if resume:
        invocation += " --resume"
    if preflight_only:
        (run_dir / "suite_summary.json").write_text(
            json.dumps(
                {
                    "status": "preflight_completed" if inputs["ok"] else "blocked",
                    "input_manifest_ok": inputs["ok"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_output_manifest(run_dir, invocation + " --preflight-only")
        if not inputs["ok"]:
            raise FileNotFoundError("\n".join(inputs["missing"]))
        return run_dir
    if not inputs["ok"]:
        write_output_manifest(run_dir, invocation)
        raise FileNotFoundError("\n".join(inputs["missing"]))

    repeats = int(suite["repeats"])
    max_frames = int(suite["max_frames"])
    warmup_frames = int(suite.get("warmup_frames", 0))
    key_arrays = [
        str(name) for name in comparison.get("key_arrays", ["keypoints"])
    ]
    angle_name = str(comparison.get("angle_name", "RightElbow"))
    bins = [float(value) for value in comparison.get("rula_bins", [60, 100])]
    repeat_thresholds = comparison.get("repeat_thresholds", {})
    historical_thresholds = comparison.get("historical_thresholds", {})

    suite_result: dict[str, Any] = {
        "schema_version": "gpu_benchmark_suite_v1",
        "created_utc": utc_now(),
        "status": "completed",
        "datasets": {},
    }
    csv_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset in datasets:
        name = str(dataset["name"])
        dataset_dir = run_dir / name
        reference = resolve_project_path(dataset["historical_reference_npz"])
        records: list[dict[str, Any]] = []
        npz_paths: list[Path] = []
        successful_repeat_indices: list[int] = []
        benchmark_payloads: list[dict[str, Any]] = []
        for repeat_index in range(1, repeats + 1):
            repeat_dir = dataset_dir / f"repeat_{repeat_index:02d}"
            command = build_benchmark_command(
                benchmark_script,
                dataset,
                repeat_dir,
                max_frames,
                warmup_frames,
            )
            benchmark_path = repeat_dir / "benchmark.json"
            npz_path = repeat_dir / "skt_pose_optimized.npz"
            if resume and reusable_repeat_outputs(repeat_dir):
                execution = {
                    "command": command,
                    "command_text": shlex.join(command),
                    "started_utc": None,
                    "ended_utc": utc_now(),
                    "wall_time_seconds": 0.0,
                    "return_code": 0,
                    "resumed": True,
                }
                (repeat_dir / "execution.json").write_text(
                    json.dumps(jsonable(execution), indent=2),
                    encoding="utf-8",
                )
            else:
                execution = benchmark_one_repeat(command, repeat_dir)
            benchmark = None
            if execution["return_code"] == 0:
                if not benchmark_path.is_file() or not npz_path.is_file():
                    execution["return_code"] = 98
                    failures.append(f"{name} repeat {repeat_index}: missing output")
                else:
                    benchmark = json.loads(
                        benchmark_path.read_text(encoding="utf-8")
                    )
                    benchmark_payloads.append(benchmark)
                    npz_paths.append(npz_path)
                    successful_repeat_indices.append(repeat_index)
            else:
                failures.append(
                    f"{name} repeat {repeat_index}: return code "
                    f"{execution['return_code']}"
                )
            historical = None
            if benchmark is not None:
                historical = compare_npz(
                    npz_path,
                    reference,
                    key_arrays,
                    angle_name,
                    bins,
                    historical_thresholds,
                )
                (repeat_dir / "historical_comparison.json").write_text(
                    json.dumps(jsonable(historical), indent=2),
                    encoding="utf-8",
                )
            record = {
                "repeat": repeat_index,
                "execution": execution,
                "benchmark": benchmark,
                "historical_comparison": historical,
            }
            records.append(record)

        pairwise = []
        repeat_match_by_index: dict[int, bool | None] = {
            index: (
                True if index in successful_repeat_indices else None
            )
            for index in range(1, repeats + 1)
        }
        for left_index, right_index in combinations(
            range(len(npz_paths)), 2
        ):
            comparison_result = compare_npz(
                npz_paths[right_index],
                npz_paths[left_index],
                key_arrays,
                angle_name,
                bins,
                repeat_thresholds,
            )
            candidate_repeat = successful_repeat_indices[right_index]
            reference_repeat = successful_repeat_indices[left_index]
            comparison_result["candidate_repeat"] = candidate_repeat
            comparison_result["reference_repeat"] = reference_repeat
            pairwise.append(comparison_result)
            if not comparison_result["passes_thresholds"]:
                repeat_match_by_index[reference_repeat] = False
                repeat_match_by_index[candidate_repeat] = False
        (dataset_dir / "repeat_determinism.json").write_text(
            json.dumps(jsonable(pairwise), indent=2), encoding="utf-8"
        )
        timing = pooled_timing_summary(
            npz_paths, warmup_frames, benchmark_payloads
        )
        dataset_result = {
            "repeats_requested": repeats,
            "repeats_successful": len(npz_paths),
            "records": records,
            "repeat_pairwise_comparisons": pairwise,
            "all_repeats_deterministic": (
                len(npz_paths) == repeats
                and all(item["passes_thresholds"] for item in pairwise)
            ),
            "all_historical_comparisons_pass": (
                len(npz_paths) == repeats
                and all(
                    bool(record["historical_comparison"])
                    and record["historical_comparison"]["passes_thresholds"]
                    for record in records
                )
            ),
            "timing_aggregate": timing,
        }
        suite_result["datasets"][name] = dataset_result
        for record in records:
            repeat_index = int(record["repeat"])
            historical = record["historical_comparison"]
            csv_rows.append(
                csv_row(
                    name,
                    repeat_index,
                    int(record["execution"]["return_code"]),
                    record["benchmark"],
                    repeat_match_by_index.get(repeat_index),
                    (
                        bool(historical["passes_thresholds"])
                        if historical is not None
                        else None
                    ),
                )
            )
        if not dataset_result["all_repeats_deterministic"]:
            failures.append(f"{name}: repeat determinism gate failed")
        if not dataset_result["all_historical_comparisons_pass"]:
            failures.append(f"{name}: historical comparison gate failed")

    if failures:
        suite_result["status"] = "failed"
    suite_result["failures"] = failures
    suite_result["ended_utc"] = utc_now()
    (run_dir / "suite_summary.json").write_text(
        json.dumps(jsonable(suite_result), indent=2), encoding="utf-8"
    )
    write_repeat_csv(csv_rows, run_dir / "repeat_summary.csv")
    write_output_manifest(run_dir, invocation)
    if failures:
        raise RuntimeError("; ".join(failures))
    return run_dir


def main() -> None:
    """Run the configured suite and print the result path."""
    args = parse_args()
    output = run_suite(
        args.config,
        output_override=args.output_dir,
        preflight_only=args.preflight_only,
        resume=args.resume,
    )
    print(f"[gpu-benchmark-suite] {output}")


if __name__ == "__main__":
    main()
