#!/opt/anaconda3/envs/pose/bin/python
"""Run NVIDIA's official BodyPose3DNet app on selected left/right videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = Path(
    "00_pose_pipeline_v2/configs/nvidia_pose_matrix.yaml"
)
DEFAULT_APP_ROOT = Path(
    "/workspace/official_nvidia/deepstream_reference_apps/"
    "deepstream-bodypose-3d"
)


def project_path(value: str | Path) -> Path:
    """Resolve a path against the project root."""
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> str:
    """Run one command and return combined text output."""
    process = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return process.stdout


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping: {path}")
    return payload


def load_focal_length(config_path: Path) -> float:
    """Read the fixed left-camera focal length from dataset calibration."""
    config = load_yaml(config_path)
    calibration_path = project_path(
        config["calibration"]["camera_params"]
    )
    with np.load(calibration_path) as calibration:
        return float(calibration["mtx_l"][0, 0])


def configure_mode(app_root: Path, mode: str) -> Path:
    """Point the official secondary-inference config at one official model."""
    if mode not in {"accuracy", "performance"}:
        raise ValueError(f"Unsupported BodyPose3DNet mode: {mode}")
    config_path = (
        app_root / "configs" / "config_infer_secondary_bodypose3dnet.txt"
    )
    model_name = f"bodypose3dnet_{mode}"
    content = config_path.read_text(encoding="utf-8")
    content = re.sub(
        r"^model-engine-file=.*$",
        (
            "model-engine-file=../models/bodypose3dnet/"
            f"{model_name}.onnx_b8_gpu0_fp16.engine"
        ),
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^onnx-file=.*$",
        f"onnx-file=../models/bodypose3dnet/{model_name}.onnx",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^#model-engine-file=.*\n?",
        "",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^#onnx-file=.*\n?",
        "",
        content,
        flags=re.MULTILINE,
    )
    config_path.write_text(content, encoding="utf-8")
    return config_path


def app_command(
    binary: Path,
    video: Path,
    focal_length: float,
    pose_json: Path | None,
) -> list[str]:
    """Build the official reference-app command for one camera stream."""
    command = [
        str(binary),
        "--input",
        video.resolve().as_uri(),
        "--output",
        "fakesink",
        "--tracker",
        "perf",
        "--width",
        "2048",
        "--height",
        "1536",
        "--focal",
        f"{focal_length:.9f}",
        "--fps",
    ]
    if pose_json is not None:
        command.extend(["--save-pose", str(pose_json)])
    return command


def create_warmup_video(
    source: Path,
    output: Path,
    frame_count: int,
) -> Path:
    """Create a short, lossless, same-resolution video for model warm-up."""
    if frame_count <= 0:
        raise ValueError("Warm-up frame count must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        str(output),
    ]
    run_text(command)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Warm-up video was not created: {output}")
    return output


def execute_one(
    *,
    app_root: Path,
    video: Path,
    focal_length: float,
    pose_json: Path,
    output_dir: Path,
    repeats: int,
    warmup_frames: int,
) -> dict[str, Any]:
    """Warm the engine, then run repeatable end-to-end application trials."""
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = app_root / "sources" / "deepstream-pose-estimation-app"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    warmup_video = create_warmup_video(
        video,
        output_dir / "warmup_input.mkv",
        warmup_frames,
    )
    warmup_command = app_command(
        binary,
        warmup_video,
        focal_length,
        None,
    )
    warmup_started = time.perf_counter()
    warmup_output = run_text(
        warmup_command,
        cwd=app_root / "sources",
    )
    warmup_elapsed = time.perf_counter() - warmup_started
    (output_dir / "warmup.log").write_text(
        f"$ {shlex.join(warmup_command)}\n{warmup_output}",
        encoding="utf-8",
    )
    trials: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        target_json = pose_json if repeat_index == 0 else None
        command = app_command(
            binary,
            video,
            focal_length,
            target_json,
        )
        started = time.perf_counter()
        output = run_text(command, cwd=app_root / "sources")
        elapsed = time.perf_counter() - started
        log_path = output_dir / f"repeat_{repeat_index + 1}.log"
        log_path.write_text(
            f"$ {shlex.join(command)}\n{output}",
            encoding="utf-8",
        )
        trials.append(
            {
                "repeat": repeat_index + 1,
                "elapsed_seconds": elapsed,
                "log": log_path.name,
            }
        )
    if not pose_json.is_file():
        raise RuntimeError(f"Pose JSON was not created: {pose_json}")
    return {
        "status": "completed",
        "input_video": str(video),
        "input_sha256": sha256_file(video),
        "pose_json": str(pose_json),
        "pose_json_sha256": sha256_file(pose_json),
        "warmup_frames": int(warmup_frames),
        "warmup_input_sha256": sha256_file(warmup_video),
        "warmup_elapsed_seconds": warmup_elapsed,
        "trials": trials,
    }


def environment_manifest(app_root: Path) -> dict[str, Any]:
    """Record exact internal hardware and runtime provenance."""
    binary = app_root / "sources" / "deepstream-pose-estimation-app"
    return {
        "platform": platform.platform(),
        "gpu": run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ).strip(),
        "deepstream": run_text(
            ["deepstream-app", "--version-all"],
            check=False,
        ).strip(),
        "project_commit": run_text(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
        ).strip(),
        "reference_app_commit": run_text(
            ["git", "rev-parse", "HEAD"],
            cwd=app_root.parent,
        ).strip(),
        "binary_sha256": sha256_file(binary),
        "license_policy": (
            "Internal timing evidence. Do not publish proprietary SDK "
            "competitive benchmark values without checking the applicable "
            "NVIDIA license and obtaining any required permission."
        ),
    }


def run_matrix(args: argparse.Namespace) -> Path:
    """Run both official model modes over all selected datasets and views."""
    matrix_path = project_path(args.matrix)
    selection_path = project_path(args.selection)
    matrix = load_yaml(matrix_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    app_root = args.app_root.expanduser().resolve()
    output_root = project_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for mode in args.modes:
        model_config = configure_mode(app_root, mode)
        for dataset_name in args.datasets or list(matrix["datasets"]):
            config_path = project_path(
                matrix["datasets"][dataset_name]["config"]
            )
            focal_length = load_focal_length(config_path)
            selected = selection["accepted"][dataset_name]
            for side in ("left", "right"):
                video = project_path(selected[side])
                cell_dir = output_root / "raw" / dataset_name / mode / side
                pose_json = cell_dir / "pose.json"
                record: dict[str, Any] = {
                    "dataset": dataset_name,
                    "mode": mode,
                    "side": side,
                    "official_model_config_sha256": sha256_file(
                        model_config
                    ),
                }
                try:
                    record.update(
                        execute_one(
                            app_root=app_root,
                            video=video,
                            focal_length=focal_length,
                            pose_json=pose_json,
                            output_dir=cell_dir,
                            repeats=args.repeats,
                            warmup_frames=args.warmup_frames,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    record.update(
                        {
                            "status": "failed",
                            "error": repr(error),
                        }
                    )
                results.append(record)
                summary_path = output_root / "bodypose3d_run_summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "schema_version": (
                                "nvidia_bodypose3d_run_matrix_v1"
                            ),
                            "environment": environment_manifest(app_root),
                            "results": results,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    return output_root / "bodypose3d_run_summary.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--app-root",
        type=Path,
        default=DEFAULT_APP_ROOT,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["accuracy", "performance"],
        default=["accuracy", "performance"],
    )
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the matrix and print its durable summary path."""
    args = parse_args(argv)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmup_frames <= 0:
        raise ValueError("--warmup-frames must be positive")
    result = run_matrix(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
