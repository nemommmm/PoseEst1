#!/usr/bin/env python3
"""Benchmark FFmpeg software decoding against NVIDIA NVDEC.

This utility intentionally measures video demuxing and decoding into a null
sink. It does not include pose inference, stereo synchronization,
triangulation, angle calculation, or RULA scoring, so its throughput must not
be reported as end-to-end pipeline FPS.

The optional validation path decodes a few frames to a common luma plane with
both backends and records checksums, pixel differences, and spatial
correlation. Luma avoids mistaking a full-range YUV-to-RGB conversion mismatch
for a decoder mismatch. The validation path is separate from the timed
null-sink benchmark because GPU-to-host transfer changes the workload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "decode_benchmark_v2"
VALID_BACKENDS = ("cpu", "nvdec")


@dataclass(frozen=True)
class VideoInfo:
    """Describe the first video stream in an input file."""

    path: str
    codec_name: str
    profile: str | None
    width: int
    height: int
    pixel_format: str | None
    color_range: str | None
    color_space: str | None
    frame_rate: float | None
    duration_seconds: float | None
    frame_count: int
    frame_count_source: str


class BenchmarkError(RuntimeError):
    """Raised when a benchmark prerequisite or subprocess fails."""


def parse_ratio(value: object) -> float | None:
    """Parse an FFprobe ratio such as ``25/1`` into a positive float."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", maxsplit=1)
            numerator = float(numerator_text)
            denominator = float(denominator_text)
            if denominator == 0:
                return None
            result = numerator / denominator
        else:
            result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) and result > 0 else None


def parse_optional_float(value: object) -> float | None:
    """Parse a finite non-negative float from FFprobe output."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) and result >= 0 else None


def parse_positive_int(value: object) -> int | None:
    """Parse a strictly positive integer from loosely typed metadata."""

    if value is None:
        return None
    try:
        result = int(str(value).strip())
    except ValueError:
        return None
    return result if result > 0 else None


def parse_probe_payload(payload: dict[str, Any], path: Path) -> VideoInfo:
    """Convert FFprobe JSON into a validated :class:`VideoInfo`."""

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise BenchmarkError(f"FFprobe found no video stream in {path}")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise BenchmarkError(f"Malformed FFprobe stream metadata for {path}")

    width = parse_positive_int(stream.get("width"))
    height = parse_positive_int(stream.get("height"))
    if width is None or height is None:
        raise BenchmarkError(f"Missing video dimensions for {path}")

    frame_rate = parse_ratio(stream.get("avg_frame_rate"))
    if frame_rate is None:
        frame_rate = parse_ratio(stream.get("r_frame_rate"))

    stream_duration = parse_optional_float(stream.get("duration"))
    format_payload = payload.get("format", {})
    format_duration = (
        parse_optional_float(format_payload.get("duration"))
        if isinstance(format_payload, dict)
        else None
    )
    duration = stream_duration if stream_duration is not None else format_duration

    frame_count = parse_positive_int(stream.get("nb_read_frames"))
    frame_count_source = "ffprobe_nb_read_frames"
    if frame_count is None:
        frame_count = parse_positive_int(stream.get("nb_frames"))
        frame_count_source = "stream_nb_frames"
    if frame_count is None and duration is not None and frame_rate is not None:
        frame_count = max(1, int(round(duration * frame_rate)))
        frame_count_source = "duration_times_frame_rate_estimate"
    if frame_count is None:
        raise BenchmarkError(
            f"Could not determine a frame count for {path}; "
            "FFprobe returned neither a count nor usable duration/FPS."
        )

    return VideoInfo(
        path=str(path.resolve()),
        codec_name=str(stream.get("codec_name") or "unknown"),
        profile=(
            str(stream["profile"]) if stream.get("profile") is not None else None
        ),
        width=width,
        height=height,
        pixel_format=(
            str(stream["pix_fmt"]) if stream.get("pix_fmt") is not None else None
        ),
        color_range=(
            str(stream["color_range"])
            if stream.get("color_range") is not None
            else None
        ),
        color_space=(
            str(stream["color_space"])
            if stream.get("color_space") is not None
            else None
        ),
        frame_rate=frame_rate,
        duration_seconds=duration,
        frame_count=frame_count,
        frame_count_source=frame_count_source,
    )


def build_probe_command(ffprobe: str, video: Path) -> list[str]:
    """Build an exact-frame-count FFprobe command."""

    return [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,profile,width,height,pix_fmt,color_range,"
            "color_space,avg_frame_rate,"
            "r_frame_rate,nb_frames,nb_read_frames,duration:"
            "format=duration"
        ),
        "-of",
        "json",
        str(video),
    ]


def probe_video(ffprobe: str, video: Path, timeout: float | None) -> VideoInfo:
    """Inspect one input video and count its decoded frames outside timing."""

    command = build_probe_command(ffprobe, video)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise BenchmarkError(
            f"FFprobe failed for {video}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid FFprobe JSON for {video}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Unexpected FFprobe payload for {video}")
    return parse_probe_payload(payload, video)


def nvdec_compatibility_issues(
    video_info: Sequence[VideoInfo],
) -> list[dict[str, str]]:
    """Identify stream profiles known to be unsupported by NVIDIA NVDEC.

    NVIDIA's published NVDEC matrix lists H.264 Baseline, Main, and High
    profiles for GA10x/Ampere. H.264 High 4:4:4 Predictive is not in that
    supported set. FFmpeg can still initialize NVDEC for such a stream, so a
    successful process exit is not sufficient evidence of correct pixels.
    """

    issues: list[dict[str, str]] = []
    for info in video_info:
        profile = (info.profile or "").casefold()
        if info.codec_name.casefold() == "h264" and "4:4:4" in profile:
            issues.append(
                {
                    "path": info.path,
                    "code": "unsupported_h264_high_444_profile",
                    "message": (
                        f"H.264 profile {info.profile!r} is outside the "
                        "published NVDEC H.264 Baseline/Main/High support "
                        "set. Timing from this decode path is not valid unless "
                        "decoded luma first passes content validation."
                    ),
                }
            )
    return issues


def build_decode_command(
    ffmpeg: str,
    videos: Sequence[Path],
    backend: str,
    frame_limit: int | None,
    gpu_id: int = 0,
) -> list[str]:
    """Build a concurrent multi-input null-sink decode command."""

    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    if not videos:
        raise ValueError("At least one input video is required")
    if frame_limit is not None and frame_limit <= 0:
        raise ValueError("frame_limit must be positive")

    command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error"]
    for video in videos:
        if backend == "cpu":
            command.extend(["-hwaccel", "none"])
        else:
            command.extend(
                [
                    "-hwaccel",
                    "cuda",
                    "-hwaccel_device",
                    str(gpu_id),
                    "-hwaccel_output_format",
                    "cuda",
                ]
            )
        command.extend(["-i", str(video)])

    for index in range(len(videos)):
        command.extend(["-map", f"{index}:v:0", "-an", "-sn", "-dn"])
        if frame_limit is not None:
            command.extend(["-frames:v", str(frame_limit)])
        command.extend(["-fps_mode", "passthrough", "-f", "null", "-"])
    return command


def build_validation_command(
    ffmpeg: str,
    video: Path,
    backend: str,
    frames: int,
    gpu_id: int = 0,
) -> list[str]:
    """Build a command that emits a common 8-bit luma sample to stdout."""

    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    if frames <= 0:
        raise ValueError("frames must be positive")

    command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error"]
    if backend == "cpu":
        command.extend(["-hwaccel", "none"])
    else:
        command.extend(
            [
                "-hwaccel",
                "cuda",
                "-hwaccel_device",
                str(gpu_id),
                "-hwaccel_output_format",
                "cuda",
            ]
        )
    command.extend(["-i", str(video), "-map", "0:v:0"])
    if backend == "nvdec":
        command.extend(["-vf", "hwdownload,format=nv12,format=gray"])
    else:
        command.extend(["-vf", "format=gray"])
    command.extend(
        [
            "-frames:v",
            str(frames),
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    return command


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Summarize repeated measurements with robust statistics."""

    if not values:
        raise ValueError("values must not be empty")
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p95": percentile(numeric, 0.95),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def frame_budget(video: VideoInfo, frame_limit: int | None) -> int:
    """Return the number of frames expected from one benchmark output."""

    return (
        video.frame_count
        if frame_limit is None
        else min(video.frame_count, frame_limit)
    )


def command_text(command: Sequence[str]) -> str:
    """Render a command for an auditable JSON report."""

    return shlex.join(str(part) for part in command)


def run_timed_command(
    command: Sequence[str],
    total_frames: int,
    paired_frames: int,
    timeout: float | None,
) -> dict[str, Any]:
    """Run one timed FFmpeg command and calculate effective throughput."""

    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    wall_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "return_code": completed.returncode,
        "wall_seconds": wall_seconds,
        "total_frames": total_frames,
        "paired_frames": paired_frames,
        "aggregate_effective_fps": (
            total_frames / wall_seconds if wall_seconds > 0 else None
        ),
        "paired_effective_fps": (
            paired_frames / wall_seconds if wall_seconds > 0 else None
        ),
        "stderr_tail": completed.stderr.strip()[-4000:],
    }
    if completed.returncode != 0:
        raise BenchmarkError(
            f"FFmpeg exited with {completed.returncode}: "
            f"{completed.stderr.strip()[-1000:]}"
        )
    return result


def run_backend(
    ffmpeg: str,
    videos: Sequence[Path],
    video_info: Sequence[VideoInfo],
    backend: str,
    warmup_runs: int,
    warmup_frames: int,
    repeats: int,
    frame_limit: int | None,
    gpu_id: int,
    timeout: float | None,
    force_unsupported_nvdec: bool = False,
) -> dict[str, Any]:
    """Run warm-up and repeated measurements for one decode backend."""

    compatibility_issues = (
        nvdec_compatibility_issues(video_info) if backend == "nvdec" else []
    )
    warmup_limit = warmup_frames
    warmup_total = sum(frame_budget(info, warmup_limit) for info in video_info)
    warmup_paired = min(frame_budget(info, warmup_limit) for info in video_info)
    warmup_command = build_decode_command(
        ffmpeg, videos, backend, warmup_limit, gpu_id
    )

    output: dict[str, Any] = {
        "backend": backend,
        "status": "pending",
        "warmup_runs": warmup_runs,
        "warmup_frames_per_video": warmup_frames,
        "warmup_command": warmup_command,
        "warmup_command_text": command_text(warmup_command),
        "warmups": [],
        "repeat_count": repeats,
        "compatibility_issues": compatibility_issues,
    }
    if compatibility_issues and not force_unsupported_nvdec:
        output["status"] = "unsupported"
        output["error"] = (
            "NVDEC timing was not run because the input codec profile is "
            "outside NVIDIA's published support set."
        )
        return output
    try:
        for run_index in range(warmup_runs):
            warmup = run_timed_command(
                warmup_command,
                total_frames=warmup_total,
                paired_frames=warmup_paired,
                timeout=timeout,
            )
            warmup["run_index"] = run_index
            output["warmups"].append(warmup)

        measured_total = sum(
            frame_budget(info, frame_limit) for info in video_info
        )
        measured_paired = min(
            frame_budget(info, frame_limit) for info in video_info
        )
        measured_command = build_decode_command(
            ffmpeg, videos, backend, frame_limit, gpu_id
        )
        output["command"] = measured_command
        output["command_text"] = command_text(measured_command)
        output["measured_frames_per_video"] = [
            frame_budget(info, frame_limit) for info in video_info
        ]
        measurements: list[dict[str, Any]] = []
        for run_index in range(repeats):
            measurement = run_timed_command(
                measured_command,
                total_frames=measured_total,
                paired_frames=measured_paired,
                timeout=timeout,
            )
            measurement["run_index"] = run_index
            measurements.append(measurement)
        output["measurements"] = measurements
        output["wall_seconds"] = summarize(
            [item["wall_seconds"] for item in measurements]
        )
        output["aggregate_effective_fps"] = summarize(
            [item["aggregate_effective_fps"] for item in measurements]
        )
        output["paired_effective_fps"] = summarize(
            [item["paired_effective_fps"] for item in measurements]
        )
        output["status"] = (
            "invalid_unsupported_input"
            if compatibility_issues
            else "completed"
        )
    except (BenchmarkError, subprocess.TimeoutExpired) as exc:
        output["status"] = "failed"
        output["error"] = str(exc)
    return output


def decode_validation_sample(
    command: Sequence[str],
    expected_bytes: int,
    timeout: float | None,
) -> bytes:
    """Decode a short luma sample and verify its byte length."""

    completed = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise BenchmarkError(
            f"Validation decode failed with {completed.returncode}: "
            f"{stderr.strip()[-1000:]}"
        )
    if len(completed.stdout) != expected_bytes:
        raise BenchmarkError(
            "Validation decode returned "
            f"{len(completed.stdout)} bytes; expected {expected_bytes}."
        )
    return completed.stdout


def compare_luma_bytes(
    cpu_bytes: bytes,
    nvdec_bytes: bytes,
    tolerance: int,
    minimum_within_tolerance: float,
) -> dict[str, Any]:
    """Calculate checksum and pixel differences for two luma samples."""

    if len(cpu_bytes) != len(nvdec_bytes):
        raise BenchmarkError("CPU and NVDEC validation buffers differ in size")
    if not 0 <= tolerance <= 255:
        raise ValueError("tolerance must be between zero and 255")
    if not 0 <= minimum_within_tolerance <= 1:
        raise ValueError("minimum_within_tolerance must be between zero and one")

    try:
        import numpy as np
    except ImportError as exc:
        raise BenchmarkError(
            "NumPy is required only for optional pixel-difference validation"
        ) from exc

    cpu_array = np.frombuffer(cpu_bytes, dtype=np.uint8)
    nvdec_array = np.frombuffer(nvdec_bytes, dtype=np.uint8)
    difference = np.abs(
        cpu_array.astype(np.int16) - nvdec_array.astype(np.int16)
    )
    within_fraction = float(np.mean(difference <= tolerance))
    exact_fraction = float(np.mean(difference == 0))
    return {
        "cpu_sha256": hashlib.sha256(cpu_bytes).hexdigest(),
        "nvdec_sha256": hashlib.sha256(nvdec_bytes).hexdigest(),
        "exact_match": bool(np.array_equal(cpu_array, nvdec_array)),
        "mean_absolute_pixel_difference": float(np.mean(difference)),
        "p95_absolute_pixel_difference": float(
            np.percentile(difference, 95)
        ),
        "p99_absolute_pixel_difference": float(
            np.percentile(difference, 99)
        ),
        "maximum_absolute_pixel_difference": int(np.max(difference)),
        "exact_pixel_fraction": exact_fraction,
        "within_tolerance_fraction": within_fraction,
        "tolerance": tolerance,
        "minimum_within_tolerance": minimum_within_tolerance,
        "comparison_passed": within_fraction >= minimum_within_tolerance,
    }


def validate_video(
    ffmpeg: str,
    video: Path,
    info: VideoInfo,
    requested_frames: int,
    gpu_id: int,
    tolerance: int,
    minimum_within_tolerance: float,
    maximum_bytes: int,
    timeout: float | None,
) -> dict[str, Any]:
    """Compare a small CPU/NVDEC luma sample for one input."""

    frames = min(requested_frames, info.frame_count)
    expected_bytes = info.width * info.height * frames
    result: dict[str, Any] = {
        "video": str(video.resolve()),
        "status": "pending",
        "frames": frames,
        "width": info.width,
        "height": info.height,
        "pixel_format": "gray",
        "expected_bytes_per_backend": expected_bytes,
    }
    if expected_bytes > maximum_bytes:
        result["status"] = "skipped"
        result["reason"] = (
            f"Sample requires {expected_bytes} bytes per backend, above "
            f"the configured limit of {maximum_bytes}."
        )
        return result

    cpu_command = build_validation_command(
        ffmpeg, video, "cpu", frames, gpu_id
    )
    nvdec_command = build_validation_command(
        ffmpeg, video, "nvdec", frames, gpu_id
    )
    result["cpu_command"] = cpu_command
    result["cpu_command_text"] = command_text(cpu_command)
    result["nvdec_command"] = nvdec_command
    result["nvdec_command_text"] = command_text(nvdec_command)
    try:
        cpu_bytes = decode_validation_sample(
            cpu_command, expected_bytes, timeout
        )
        nvdec_bytes = decode_validation_sample(
            nvdec_command, expected_bytes, timeout
        )
        result.update(
            compare_luma_bytes(
                cpu_bytes,
                nvdec_bytes,
                tolerance=tolerance,
                minimum_within_tolerance=minimum_within_tolerance,
            )
        )
        result["status"] = (
            "passed" if result["comparison_passed"] else "rejected"
        )
    except (BenchmarkError, subprocess.TimeoutExpired) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
    return result


def first_line(command: Sequence[str], timeout: float | None) -> str:
    """Return the first stdout line from a version command."""

    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        return f"unavailable: {completed.stderr.strip()}"
    return completed.stdout.splitlines()[0] if completed.stdout else "unknown"


def parse_gpu_query(output: str) -> list[dict[str, Any]]:
    """Parse CSV output from the selected ``nvidia-smi`` query."""

    gpus: list[dict[str, Any]] = []
    reader = csv.reader(line for line in output.splitlines() if line.strip())
    for row in reader:
        if len(row) != 4:
            continue
        memory_text = row[3].strip()
        try:
            memory_mib: int | None = int(memory_text)
        except ValueError:
            memory_mib = None
        gpus.append(
            {
                "name": row[0].strip(),
                "uuid": row[1].strip(),
                "driver_version": row[2].strip(),
                "memory_total_mib": memory_mib,
            }
        )
    return gpus


def query_gpus(timeout: float | None) -> dict[str, Any]:
    """Collect GPU metadata without making GPU availability mandatory."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found", "gpus": []}
    command = [
        executable,
        "--query-gpu=name,uuid,driver_version,memory.total",
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
        return {"available": False, "reason": str(exc), "gpus": []}
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": completed.stderr.strip(),
            "gpus": [],
        }
    gpus = parse_gpu_query(completed.stdout)
    return {"available": bool(gpus), "gpus": gpus, "command": command}


def git_commit(project_root: Path, timeout: float | None) -> str | None:
    """Return the current project commit when the script runs in a clone."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def speedup_summary(backends: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Calculate NVDEC/CPU median throughput ratios when both succeeded."""

    by_name = {item["backend"]: item for item in backends}
    cpu = by_name.get("cpu")
    nvdec = by_name.get("nvdec")
    if (
        cpu is None
        or nvdec is None
        or cpu.get("status") != "completed"
        or nvdec.get("status") != "completed"
    ):
        return None
    cpu_aggregate = cpu["aggregate_effective_fps"]["median"]
    nvdec_aggregate = nvdec["aggregate_effective_fps"]["median"]
    cpu_paired = cpu["paired_effective_fps"]["median"]
    nvdec_paired = nvdec["paired_effective_fps"]["median"]
    return {
        "nvdec_over_cpu_aggregate_median": (
            nvdec_aggregate / cpu_aggregate if cpu_aggregate > 0 else None
        ),
        "nvdec_over_cpu_paired_median": (
            nvdec_paired / cpu_paired if cpu_paired > 0 else None
        ),
    }


def positive_int(value: str) -> int:
    """Argparse type for strictly positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    """Argparse type for non-negative integers."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def fraction(value: str) -> float:
    """Argparse type for values in the inclusive unit interval."""

    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark concurrent FFmpeg CPU/NVDEC decoding. Reported FPS is "
            "decode-only throughput, not pose-pipeline FPS."
        )
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=VALID_BACKENDS,
        default=list(VALID_BACKENDS),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--gpu-id", type=non_negative_int, default=0)
    parser.add_argument("--warmup-runs", type=non_negative_int, default=1)
    parser.add_argument("--warmup-frames", type=positive_int, default=30)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        help="Limit each input stream during measured repetitions.",
    )
    parser.add_argument(
        "--validate-frames",
        type=non_negative_int,
        default=3,
        help="CPU/NVDEC luma frames to compare per input; zero disables it.",
    )
    parser.add_argument(
        "--pixel-tolerance",
        type=non_negative_int,
        default=2,
        help="Allowed absolute difference for one luma pixel.",
    )
    parser.add_argument(
        "--minimum-within-tolerance",
        type=fraction,
        default=0.999,
    )
    parser.add_argument(
        "--maximum-validation-mib",
        type=positive_int,
        default=256,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0,
        help="Per-command timeout; zero disables the timeout.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def validate_cli_inputs(args: argparse.Namespace) -> None:
    """Fail early on invalid paths and executable names."""

    if args.pixel_tolerance > 255:
        raise BenchmarkError("--pixel-tolerance cannot exceed 255")
    if args.timeout_seconds < 0:
        raise BenchmarkError("--timeout-seconds cannot be negative")
    for video in args.videos:
        if not video.is_file():
            raise BenchmarkError(f"Input video does not exist: {video}")
    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).is_file():
        raise BenchmarkError(f"FFmpeg executable not found: {args.ffmpeg}")
    if shutil.which(args.ffprobe) is None and not Path(args.ffprobe).is_file():
        raise BenchmarkError(f"FFprobe executable not found: {args.ffprobe}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and optionally save its JSON report."""

    args = parse_args(argv)
    try:
        validate_cli_inputs(args)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    videos = [video.resolve() for video in args.videos]
    try:
        video_info = [
            probe_video(args.ffprobe, video, timeout) for video in videos
        ]
    except (BenchmarkError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = [
        run_backend(
            ffmpeg=args.ffmpeg,
            videos=videos,
            video_info=video_info,
            backend=backend,
            warmup_runs=args.warmup_runs,
            warmup_frames=args.warmup_frames,
            repeats=args.repeats,
            frame_limit=args.max_frames,
            gpu_id=args.gpu_id,
            timeout=timeout,
        )
        for backend in dict.fromkeys(args.backends)
    ]

    validation: dict[str, Any]
    if args.validate_frames == 0:
        validation = {"status": "disabled", "videos": []}
    elif "cpu" not in args.backends or "nvdec" not in args.backends:
        validation = {
            "status": "skipped",
            "reason": "Luma comparison requires both cpu and nvdec backends.",
            "videos": [],
        }
    else:
        validation_items = [
            validate_video(
                ffmpeg=args.ffmpeg,
                video=video,
                info=info,
                requested_frames=args.validate_frames,
                gpu_id=args.gpu_id,
                tolerance=args.pixel_tolerance,
                minimum_within_tolerance=args.minimum_within_tolerance,
                maximum_bytes=args.maximum_validation_mib * 1024 * 1024,
                timeout=timeout,
            )
            for video, info in zip(videos, video_info, strict=True)
        ]
        statuses = {item["status"] for item in validation_items}
        if statuses == {"skipped"}:
            validation_status = "skipped"
        elif statuses <= {"passed", "skipped"}:
            validation_status = "passed"
        elif "failed" in statuses:
            validation_status = "failed"
        else:
            validation_status = "rejected"
        validation = {"status": validation_status, "videos": validation_items}

    project_root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_scope": {
            "name": "FFmpeg concurrent video decode benchmark",
            "included": [
                "FFmpeg process startup",
                "container demuxing",
                "CPU software decode or NVIDIA NVDEC",
                "null output sink",
            ],
            "excluded": [
                "frame timestamp synchronization",
                "RGB tensor preprocessing in the pose pipeline",
                "pose inference",
                "stereo matching and triangulation",
                "joint angles and RULA",
            ],
            "warning": (
                "These measurements are decode-only throughput and must not "
                "be reported as end-to-end stereo pose FPS."
            ),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "ffmpeg_path": shutil.which(args.ffmpeg) or args.ffmpeg,
            "ffprobe_path": shutil.which(args.ffprobe) or args.ffprobe,
            "ffmpeg_version": first_line(
                [args.ffmpeg, "-version"], timeout
            ),
            "ffprobe_version": first_line(
                [args.ffprobe, "-version"], timeout
            ),
            "gpu": query_gpus(timeout),
            "git_commit": git_commit(project_root, timeout),
        },
        "configuration": {
            "videos": [str(video) for video in videos],
            "backends": list(dict.fromkeys(args.backends)),
            "gpu_id": args.gpu_id,
            "warmup_runs": args.warmup_runs,
            "warmup_frames_per_video": args.warmup_frames,
            "repeats": args.repeats,
            "max_frames_per_video": args.max_frames,
            "validate_frames_per_video": args.validate_frames,
            "pixel_tolerance": args.pixel_tolerance,
            "minimum_within_tolerance": args.minimum_within_tolerance,
            "timeout_seconds": args.timeout_seconds,
        },
        "inputs": [asdict(info) for info in video_info],
        "backends": backends,
        "speedup": speedup_summary(backends),
        "luma_sample_validation": validation,
    }

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(serialized)

    backend_failed = any(item["status"] != "completed" for item in backends)
    validation_failed = validation["status"] in {"failed", "rejected"}
    if backend_failed:
        return 2
    if validation_failed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
