#!/usr/bin/env python
"""Compare serial and batch=2 full-frame stereo pose inference latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from common.config import load_config, resolve_path, section
from common.metrics import jsonable
from skt_inference import configure_deterministic_cuda
from stereo_loader import StereoFrameReader, build_synced_timeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def latency_stats(values: list[float], warmup: int) -> dict[str, float]:
    """Return latency statistics for values after warm-up."""
    data = np.asarray(values[warmup:], dtype=np.float64)
    return {
        "mean_ms": float(np.mean(data)),
        "median_ms": float(np.median(data)),
        "p95_ms": float(np.percentile(data, 95)),
        "stereo_pair_fps": float(1000.0 / np.mean(data)),
    }


def run_mode(
    model: YOLO,
    reader: StereoFrameReader,
    n_frames: int,
    batch: bool,
) -> list[float]:
    """Measure pose inference for serial calls or one batch=2 call."""
    timings: list[float] = []
    for idx in range(n_frames):
        ok, left, right = reader.read_synced_sequential(idx)
        if not ok or left is None or right is None:
            break
        start = time.perf_counter()
        if batch:
            model([left, right], verbose=False)
        else:
            model(left, verbose=False)
            model(right, verbose=False)
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def main() -> None:
    """Benchmark both inference modes and save JSON."""
    args = parse_args()
    config = load_config(args.config)
    dataset = section(config, "dataset")
    left_video = resolve_path(dataset.get("left_video"), must_exist=True)
    right_video = resolve_path(dataset.get("right_video"), must_exist=True)
    left_meta = resolve_path(dataset.get("left_metadata"), must_exist=True)
    right_meta = resolve_path(dataset.get("right_metadata"), must_exist=True)
    model_path = resolve_path(args.model, must_exist=True)
    assert left_video and right_video and left_meta and right_meta and model_path
    _, synced, _, _ = build_synced_timeline(
        left_meta,
        right_meta,
        dataset.get("timestamp_format", "seconds_microseconds_columns"),
    )
    n_frames = min(len(synced), args.max_frames)
    model = YOLO(str(model_path))
    deterministic = configure_deterministic_cuda(True)
    results: dict[str, object] = {
        "model": str(model_path),
        "frames": n_frames,
        "warmup_frames": args.warmup_frames,
        "deterministic_cuda": deterministic,
    }
    for label, batch in (("serial", False), ("batch_2", True)):
        reader = StereoFrameReader(
            left_video,
            right_video,
            synced,
            rotate_180=bool(dataset.get("rotate_180", False)),
        )
        timings = run_mode(model, reader, n_frames, batch)
        reader.release()
        results[label] = latency_stats(timings, args.warmup_frames)
    serial_ms = results["serial"]["mean_ms"]  # type: ignore[index]
    batch_ms = results["batch_2"]["mean_ms"]  # type: ignore[index]
    results["batch_speedup"] = float(serial_ms / batch_ms)
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(results), indent=2), encoding="utf-8")
    print(json.dumps(jsonable(results), indent=2))


if __name__ == "__main__":
    main()
