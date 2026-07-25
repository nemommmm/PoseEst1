#!/usr/bin/env python3
"""Create an auditable near-lossless NVDEC-compatible H.264 proxy.

The source videos used by this project can be H.264 High 4:4:4 Predictive,
which is outside NVIDIA's published NVDEC H.264 profile set on the RTX A6000.
This utility converts one source video to H.264 High with yuv420p chroma using
the already validated NVENC settings. An optional exact 180-degree geometric
rotation can be applied before encoding for sensors that store upside-down
frames. The proxy is explicitly classified as ``near-lossless``: QP 0
minimizes encoder loss, but profile/pixel-format conversion and re-encoding
mean pixel identity is not guaranteed.

The utility never overwrites an existing output or manifest. It records the
exact command, hashes, probe metadata, software/GPU metadata, and timestamps
so that downstream decode benchmarks can identify the precise proxy used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "nvdec_compatible_proxy_v2"
QUALITY_LABEL = "near-lossless"


class TranscodeError(RuntimeError):
    """Raised when proxy creation or validation cannot be completed."""


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def command_text(command: Sequence[str]) -> str:
    """Render an argument vector as an auditable shell command."""

    return shlex.join(str(part) for part in command)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA256 without loading the full video into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest_path(output: Path) -> Path:
    """Return the sidecar manifest path for an output video."""

    return output.with_name(f"{output.name}.manifest.json")


def build_probe_command(ffprobe: str, path: Path) -> list[str]:
    """Build the FFprobe command used for source and output metadata."""

    return [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,profile,width,height,pix_fmt,color_range,"
            "color_space,avg_frame_rate,r_frame_rate,nb_frames,duration:"
            "format=format_name,duration,size"
        ),
        "-of",
        "json",
        str(path),
    ]


def parse_probe_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Extract stable video fields from an FFprobe JSON payload."""

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise TranscodeError(f"FFprobe found no video stream in {path}")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise TranscodeError(f"Malformed FFprobe stream metadata for {path}")
    format_payload = payload.get("format")
    if not isinstance(format_payload, dict):
        format_payload = {}
    return {
        "codec_name": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pixel_format": stream.get("pix_fmt"),
        "color_range": stream.get("color_range"),
        "color_space": stream.get("color_space"),
        "average_frame_rate": stream.get("avg_frame_rate"),
        "reported_frame_rate": stream.get("r_frame_rate"),
        "reported_frame_count": stream.get("nb_frames"),
        "stream_duration_seconds": stream.get("duration"),
        "container_format": format_payload.get("format_name"),
        "container_duration_seconds": format_payload.get("duration"),
        "container_size_bytes": format_payload.get("size"),
    }


def probe_video(
    ffprobe: str,
    path: Path,
    timeout: float | None,
) -> dict[str, Any]:
    """Probe one video and return both metadata and the exact command."""

    command = build_probe_command(ffprobe, path)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise TranscodeError(
            f"FFprobe failed for {path}: {completed.stderr.strip()[-1000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise TranscodeError(f"Invalid FFprobe JSON for {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscodeError(f"Unexpected FFprobe payload for {path}")
    return {
        "command": command,
        "command_text": command_text(command),
        "metadata": parse_probe_payload(payload, path),
    }


def build_transcode_command(
    ffmpeg: str,
    source: Path,
    output: Path,
    max_frames: int | None = None,
    rotate_180: bool = False,
) -> list[str]:
    """Build the validated H.264 NVENC near-lossless proxy command."""

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    video_filter = (
        "hflip,vflip,format=nv12" if rotate_180 else "format=nv12"
    )
    command.extend(
        [
            "-vf",
            video_filter,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "constqp",
            "-qp",
            "0",
            "-profile:v",
            "high",
            "-bf",
            "0",
            str(output),
        ]
    )
    return command


def parse_gpu_query(output: str) -> list[dict[str, Any]]:
    """Parse the selected ``nvidia-smi`` CSV query."""

    gpus: list[dict[str, Any]] = []
    reader = csv.reader(line for line in output.splitlines() if line.strip())
    for row in reader:
        if len(row) != 5:
            continue
        try:
            index: int | None = int(row[0].strip())
        except ValueError:
            index = None
        try:
            memory_mib: int | None = int(row[4].strip())
        except ValueError:
            memory_mib = None
        gpus.append(
            {
                "index": index,
                "name": row[1].strip(),
                "uuid": row[2].strip(),
                "driver_version": row[3].strip(),
                "memory_total_mib": memory_mib,
            }
        )
    return gpus


def query_gpus(timeout: float | None) -> dict[str, Any]:
    """Collect GPU metadata while producing a useful no-GPU result."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found", "gpus": []}
    command = [
        executable,
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": False,
            "reason": str(exc),
            "command": command,
            "gpus": [],
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": completed.stderr.strip(),
            "command": command,
            "gpus": [],
        }
    gpus = parse_gpu_query(completed.stdout)
    return {
        "available": bool(gpus),
        "command": command,
        "gpus": gpus,
    }


def first_line(command: Sequence[str], timeout: float | None) -> str:
    """Return the first stdout line of a version command."""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    if completed.returncode != 0:
        return f"unavailable: {completed.stderr.strip()}"
    return completed.stdout.splitlines()[0] if completed.stdout else "unknown"


def verify_output_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return compatibility validation errors for a transcoded output."""

    errors: list[str] = []
    if str(metadata.get("codec_name") or "").casefold() != "h264":
        errors.append("output codec is not H.264")
    profile = str(metadata.get("profile") or "")
    if profile.casefold() != "high":
        errors.append(f"output profile is {profile!r}, expected 'High'")
    pixel_format = str(metadata.get("pixel_format") or "")
    if pixel_format.casefold() != "yuv420p":
        errors.append(
            f"output pixel format is {pixel_format!r}, expected 'yuv420p'"
        )
    return errors


def validate_paths(
    source: Path,
    output: Path,
    manifest: Path,
    ffmpeg: str,
    ffprobe: str,
) -> None:
    """Validate files and enforce the no-overwrite contract."""

    if not source.is_file():
        raise TranscodeError(f"Source video does not exist: {source}")
    if source.resolve() == output.resolve():
        raise TranscodeError("Source and output paths must differ")
    if output.exists():
        raise TranscodeError(f"Refusing to overwrite existing output: {output}")
    if manifest.exists():
        raise TranscodeError(
            f"Refusing to overwrite existing manifest: {manifest}"
        )
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).is_file():
        raise TranscodeError(f"FFmpeg executable not found: {ffmpeg}")
    if shutil.which(ffprobe) is None and not Path(ffprobe).is_file():
        raise TranscodeError(f"FFprobe executable not found: {ffprobe}")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a JSON manifest atomically without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        if path.exists():
            raise TranscodeError(
                f"Refusing to overwrite existing manifest: {path}"
            )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def positive_int(value: str) -> int:
    """Argparse type for strictly positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    """Argparse type for non-negative finite floats."""

    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a near-lossless H.264 High yuv420p compatibility proxy "
            "for NVIDIA NVDEC. Pixel identity with the source is not assumed."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Default: <output>.manifest.json",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        help="Encode only the first N frames for a short validation proxy.",
    )
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help=(
            "Rotate each decoded frame by exactly 180 degrees before the "
            "near-lossless compatibility encode."
        ),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--timeout-seconds",
        type=non_negative_float,
        default=0,
        help="Per-command timeout; zero disables it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Transcode one source and write its reproducibility manifest."""

    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else default_manifest_path(output)
    )
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    try:
        validate_paths(
            source,
            output,
            manifest_path,
            args.ffmpeg,
            args.ffprobe,
        )
    except TranscodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started_counter = time.perf_counter()
    command = build_transcode_command(
        args.ffmpeg,
        source,
        output,
        args.max_frames,
        rotate_180=args.rotate_180,
    )
    source_probe: dict[str, Any] | None = None
    output_probe: dict[str, Any] | None = None
    completed: subprocess.CompletedProcess[str] | None = None
    failure: str | None = None
    status = "pending"

    try:
        source_probe = probe_video(args.ffprobe, source, timeout)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise TranscodeError(
                "FFmpeg transcode failed with "
                f"{completed.returncode}: {completed.stderr.strip()[-1000:]}"
            )
        if not output.is_file():
            raise TranscodeError("FFmpeg reported success but output is missing")
        output_probe = probe_video(args.ffprobe, output, timeout)
        verification_errors = verify_output_metadata(output_probe["metadata"])
        if verification_errors:
            raise TranscodeError("; ".join(verification_errors))
        status = "completed"
    except (TranscodeError, subprocess.TimeoutExpired) as exc:
        failure = str(exc)
        status = "failed"

    ended_at = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "quality_classification": {
            "label": QUALITY_LABEL,
            "pixel_identity_with_source_guaranteed": False,
            "reason": (
                "QP 0 minimizes encoder loss, while profile/pixel-format "
                "conversion and re-encoding can change decoded pixels."
            ),
            "intended_use": (
                "NVDEC-compatible decode and pose-pipeline benchmarking"
            ),
        },
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_seconds": time.perf_counter() - started_counter,
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "probe": source_probe,
        },
        "output": {
            "path": str(output),
            "exists": output.is_file(),
            "size_bytes": output.stat().st_size if output.is_file() else None,
            "sha256": sha256_file(output) if output.is_file() else None,
            "probe": output_probe,
        },
        "transcode": {
            "command": command,
            "command_text": command_text(command),
            "max_frames": args.max_frames,
            "rotate_180": args.rotate_180,
            "return_code": (
                completed.returncode if completed is not None else None
            ),
            "stderr_tail": (
                completed.stderr.strip()[-4000:]
                if completed is not None
                else None
            ),
        },
        "environment": {
            "ffmpeg_version": first_line(
                [args.ffmpeg, "-version"], timeout
            ),
            "ffprobe_version": first_line(
                [args.ffprobe, "-version"], timeout
            ),
            "gpu": query_gpus(timeout),
        },
        "failure": failure,
    }

    try:
        write_manifest(manifest_path, manifest)
    except (OSError, TranscodeError) as exc:
        print(f"error: could not write manifest: {exc}", file=sys.stderr)
        return 2

    if status != "completed":
        print(f"error: {failure}", file=sys.stderr)
        print(f"Wrote failure manifest: {manifest_path}", file=sys.stderr)
        return 3
    print(f"Wrote proxy: {output}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
