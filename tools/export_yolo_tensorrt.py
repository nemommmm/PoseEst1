#!/usr/bin/env python
"""Export reproducible fixed-shape Ultralytics TensorRT engines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PRECISIONS = ("fp32", "fp16")
YoloFactory = Callable[[str], Any]


def parse_args() -> argparse.Namespace:
    """Parse model, precision, fixed-shape, and device arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=SUPPORTED_PRECISIONS,
        default=list(SUPPORTED_PRECISIONS),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing copied weight and engine.",
    )
    return parser.parse_args()


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    """Return an auditable file path, byte size, and SHA256 digest."""
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def package_version(distribution: str) -> str:
    """Return one installed distribution version."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def git_value(*args: str) -> str:
    """Return a Git value or an explicit unavailable marker."""
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def collect_environment() -> dict[str, Any]:
    """Collect software, CUDA, Git, and GPU export metadata."""
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
                "torch",
                "ultralytics",
                "tensorrt",
                "tensorrt-cu12",
                "onnx",
                "onnxslim",
            )
        },
    }
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


def load_yolo_factory() -> YoloFactory:
    """Import Ultralytics lazily so unit tests do not require a GPU."""
    from ultralytics import YOLO

    return YOLO


def validate_precisions(precisions: Sequence[str]) -> list[str]:
    """Return unique supported precisions while preserving their order."""
    unique: list[str] = []
    for precision in precisions:
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported precision: {precision}")
        if precision not in unique:
            unique.append(precision)
    if not unique:
        raise ValueError("at least one precision is required")
    return unique


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON manifest."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_exported_engine(
    exported: str | Path,
    expected_engine: Path,
    precision_dir: Path,
) -> Path:
    """Resolve and validate the engine returned by Ultralytics."""
    returned = Path(exported).expanduser()
    if not returned.is_absolute():
        returned = (Path.cwd() / returned).resolve()
    else:
        returned = returned.resolve()
    if not returned.is_file() and expected_engine.is_file():
        returned = expected_engine.resolve()
    if not returned.is_file():
        raise FileNotFoundError(f"Ultralytics did not create an engine: {returned}")
    if returned.suffix.lower() != ".engine":
        raise ValueError(f"unexpected export suffix: {returned}")
    if returned.parent != precision_dir.resolve():
        raise ValueError(
            "engine was created outside its precision directory: "
            f"{returned}"
        )
    return returned


def export_precision(
    source_model: Path,
    output_root: Path,
    precision: str,
    imgsz: int,
    batch: int,
    device: int,
    overwrite: bool = False,
    yolo_factory: YoloFactory | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export one precision and always retain its status manifest."""
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported precision: {precision}")
    source_model = source_model.resolve()
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    precision_dir = (output_root / precision).resolve()
    precision_dir.mkdir(parents=True, exist_ok=True)
    copied_model = precision_dir / source_model.name
    expected_engine = copied_model.with_suffix(".engine")
    parameters = {
        "format": "engine",
        "imgsz": int(imgsz),
        "batch": int(batch),
        "dynamic": False,
        "half": precision == "fp16",
        "device": int(device),
    }
    manifest: dict[str, Any] = {
        "schema_version": "ultralytics_tensorrt_export_v1",
        "precision": precision,
        "status": "started",
        "started_utc": utc_now(),
        "ended_utc": None,
        "duration_seconds": None,
        "export_parameters": parameters,
        "environment": environment or collect_environment(),
        "source_model": file_metadata(source_model),
        "copied_model": None,
        "engine": None,
        "ultralytics_returned_path": None,
        "overwrite": bool(overwrite),
    }
    manifest_path = precision_dir / "manifest.json"
    started = time.perf_counter()
    try:
        if copied_model.exists():
            copied_matches = (
                copied_model.stat().st_size == source_model.stat().st_size
                and sha256_file(copied_model) == sha256_file(source_model)
            )
            if not copied_matches and not overwrite:
                manifest["status"] = "blocked_existing_model"
                manifest["copied_model"] = file_metadata(copied_model)
                return manifest
        if not copied_model.exists() or overwrite:
            shutil.copy2(source_model, copied_model)
        manifest["copied_model"] = file_metadata(copied_model)

        if expected_engine.exists() and not overwrite:
            manifest["status"] = "skipped_existing"
            manifest["engine"] = file_metadata(expected_engine)
            return manifest
        if expected_engine.exists():
            expected_engine.unlink()

        factory = yolo_factory or load_yolo_factory()
        model = factory(str(copied_model))
        exported = model.export(**parameters)
        manifest["ultralytics_returned_path"] = str(exported)
        engine_path = resolve_exported_engine(
            exported,
            expected_engine,
            precision_dir,
        )
        manifest["engine"] = file_metadata(engine_path)
        manifest["status"] = "exported"
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        (precision_dir / "error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return manifest
    finally:
        manifest["ended_utc"] = utc_now()
        manifest["duration_seconds"] = time.perf_counter() - started
        write_manifest(manifest_path, manifest)


def run_exports(
    source_model: Path,
    output_root: Path,
    precisions: Sequence[str],
    imgsz: int,
    batch: int,
    device: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export all requested precisions and write the aggregate manifest."""
    source_model = source_model.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    if imgsz <= 0 or batch <= 0 or device < 0:
        raise ValueError("imgsz and batch must be positive; device must be >= 0")
    requested = validate_precisions(precisions)
    output_root.mkdir(parents=True, exist_ok=True)
    environment = collect_environment()
    command = shlex.join(sys.argv)
    results = [
        export_precision(
            source_model=source_model,
            output_root=output_root,
            precision=precision,
            imgsz=imgsz,
            batch=batch,
            device=device,
            overwrite=overwrite,
            environment=environment,
        )
        for precision in requested
    ]
    accepted_statuses = {"exported", "skipped_existing"}
    overall_status = (
        "completed"
        if all(result["status"] in accepted_statuses for result in results)
        else "failed"
    )
    manifest = {
        "schema_version": "ultralytics_tensorrt_export_suite_v1",
        "status": overall_status,
        "created_utc": utc_now(),
        "command": command,
        "source_model": file_metadata(source_model),
        "output_root": str(output_root),
        "requested_precisions": requested,
        "environment": environment,
        "results": results,
    }
    write_manifest(output_root / "export_manifest.json", manifest)
    return manifest


def main() -> None:
    """Export requested engines and return non-zero on any failed precision."""
    args = parse_args()
    manifest = run_exports(
        source_model=args.model,
        output_root=args.output_root,
        precisions=args.precisions,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
