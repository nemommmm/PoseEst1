#!/usr/bin/env python
"""Benchmark the real SKT pipeline with reproducible stage-level timing."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from common.config import load_config, resolve_path
from common.metrics import jsonable
from skt_inference import run_skt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Override PT, ONNX, or TensorRT engine path.")
    parser.add_argument(
        "--left-video",
        help="Override the configured left video for deployment-input tests.",
    )
    parser.add_argument(
        "--right-video",
        help="Override the configured right video for deployment-input tests.",
    )
    parser.add_argument(
        "--input-upright",
        action="store_true",
        help="The video override is already rotated upright; disable config rotation.",
    )
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--allow-nondeterministic-cuda",
        action="store_true",
        help="Enable TF32/autotuning for a labelled throughput-only run.",
    )
    return parser.parse_args()


def summarize(values: np.ndarray, warmup: int) -> dict[str, float]:
    """Summarize a per-frame latency array after warm-up."""
    clean = np.asarray(values, dtype=np.float64)[warmup:]
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"mean_ms": float("nan"), "median_ms": float("nan"), "p95_ms": float("nan")}
    return {
        "mean_ms": float(np.mean(clean)),
        "median_ms": float(np.median(clean)),
        "p95_ms": float(np.percentile(clean, 95)),
    }


def main() -> None:
    """Run the configured pipeline and save timing statistics as JSON."""
    args = parse_args()
    config = deepcopy(load_config(args.config))
    if bool(args.left_video) != bool(args.right_video):
        raise ValueError("--left-video and --right-video must be provided together")
    dataset = config.setdefault("dataset", {})
    if args.left_video and args.right_video:
        left_video = resolve_path(args.left_video, must_exist=True)
        right_video = resolve_path(args.right_video, must_exist=True)
        assert left_video is not None and right_video is not None
        dataset["left_video"] = str(left_video)
        dataset["right_video"] = str(right_video)
    if args.input_upright:
        if not args.left_video:
            raise ValueError("--input-upright requires video overrides")
        dataset["rotate_180"] = False
    skt = config.setdefault("skt", {})
    skt["use_existing_npz"] = False
    skt["max_frames"] = args.max_frames
    skt["deterministic_cuda"] = not args.allow_nondeterministic_cuda
    if args.model:
        model_path = resolve_path(args.model, must_exist=True)
        assert model_path is not None
        skt["model_path"] = str(model_path)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    npz_path = run_skt(config, run_dir)
    data = np.load(npz_path, allow_pickle=True)
    warmup = min(max(args.warmup_frames, 0), max(len(data["timestamps"]) - 1, 0))

    stages = {
        "decode": summarize(data["decode_time_ms"], warmup),
        "pose_inference_stereo": summarize(data["yolo_time_ms"], warmup),
        "per_frame_geometry": summarize(data["geometry_time_ms"], warmup),
        "end_to_end_online": summarize(data["frame_time_ms"], warmup),
    }
    online_mean_ms = stages["end_to_end_online"]["mean_ms"]
    summary = {
        "config": str(Path(args.config)),
        "left_video": str(dataset.get("left_video")),
        "right_video": str(dataset.get("right_video")),
        "model": str(np.asarray(data["model_name"]).item()),
        "frames": int(len(data["timestamps"])),
        "warmup_frames": int(warmup),
        "deterministic_cuda": bool(np.asarray(data["deterministic_cuda"]).item()),
        "stages": stages,
        "sequence_postprocess_total_ms": float(np.asarray(data["sequence_postprocess_ms"]).item()),
        "sequence_postprocess_ms_per_frame": float(
            np.asarray(data["sequence_postprocess_ms"]).item() / len(data["timestamps"])
        ),
        "online_fps": float(1000.0 / online_mean_ms),
        "meets_12_5_fps": bool(online_mean_ms <= 80.0),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(jsonable(summary), indent=2))
    print(f"[benchmark] saved {output}")


if __name__ == "__main__":
    main()
