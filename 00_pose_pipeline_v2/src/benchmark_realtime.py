#!/usr/bin/env python
"""Benchmark the real SKT pipeline with reproducible stage-level timing."""

from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

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
    parser.add_argument(
        "--device",
        help="Force the inference device, for example cpu, mps, or 0.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the complete run and aggregate per-frame timing.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--allow-nondeterministic-cuda",
        action="store_true",
        help="Enable TF32/autotuning for a labelled throughput-only run.",
    )
    return parser.parse_args()


def summarize(values: list[np.ndarray], warmup: int) -> dict[str, float]:
    """Summarize a per-frame latency array after warm-up."""
    clean = np.concatenate(
        [
            np.asarray(repeat, dtype=np.float64)[warmup:]
            for repeat in values
        ]
    )
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
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
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
    if args.device:
        skt["device"] = args.device
    if args.model:
        model_path = resolve_path(args.model, must_exist=True)
        assert model_path is not None
        skt["model_path"] = str(model_path)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    repeat_payloads: list[dict[str, np.ndarray]] = []
    primary_data: dict[str, np.ndarray] | None = None
    for repeat_index in range(args.repeats):
        repeat_dir = (
            run_dir
            if repeat_index == 0
            else run_dir / "timing_repeats" / f"repeat_{repeat_index + 1}"
        )
        repeat_dir.mkdir(parents=True, exist_ok=True)
        npz_path = run_skt(deepcopy(config), repeat_dir)
        with np.load(npz_path, allow_pickle=True) as data:
            payload = {
                key: np.asarray(data[key])
                for key in (
                    "timestamps",
                    "decode_time_ms",
                    "yolo_time_ms",
                    "geometry_time_ms",
                    "frame_time_ms",
                )
            }
            payload["model_init_ms"] = np.asarray(
                data["model_init_ms"]
            )
            payload["requested_device"] = np.asarray(
                data["requested_device"]
            )
            payload["runtime_device"] = np.asarray(
                data["runtime_device"]
            )
            if repeat_index == 0:
                primary_data = {
                    key: np.asarray(data[key])
                    for key in data.files
                }
        repeat_payloads.append(payload)
    assert primary_data is not None
    warmup = min(
        max(args.warmup_frames, 0),
        max(len(primary_data["timestamps"]) - 1, 0),
    )

    stages = {
        "decode": summarize(
            [payload["decode_time_ms"] for payload in repeat_payloads],
            warmup,
        ),
        "pose_inference_stereo": summarize(
            [payload["yolo_time_ms"] for payload in repeat_payloads],
            warmup,
        ),
        "per_frame_geometry": summarize(
            [payload["geometry_time_ms"] for payload in repeat_payloads],
            warmup,
        ),
        "end_to_end_online": summarize(
            [payload["frame_time_ms"] for payload in repeat_payloads],
            warmup,
        ),
    }
    online_mean_ms = stages["end_to_end_online"]["mean_ms"]
    startup = {
        "model_constructor_ms": [
            float(np.asarray(payload["model_init_ms"]).item())
            for payload in repeat_payloads
        ],
        "first_stereo_pose_ms": [
            float(payload["yolo_time_ms"][0])
            for payload in repeat_payloads
        ],
        "first_end_to_end_frame_ms": [
            float(payload["frame_time_ms"][0])
            for payload in repeat_payloads
        ],
    }
    summary = {
        "config": str(Path(args.config)),
        "left_video": str(dataset.get("left_video")),
        "right_video": str(dataset.get("right_video")),
        "model": str(np.asarray(primary_data["model_name"]).item()),
        "frames": int(len(primary_data["timestamps"])),
        "warmup_frames": int(warmup),
        "repeats": int(args.repeats),
        "requested_device": str(
            np.asarray(primary_data["requested_device"]).item()
        ),
        "runtime_device": str(
            np.asarray(primary_data["runtime_device"]).item()
        ),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ),
        },
        "startup": startup,
        "deterministic_cuda": bool(
            np.asarray(primary_data["deterministic_cuda"]).item()
        ),
        "stages": stages,
        "sequence_postprocess_total_ms": float(
            np.asarray(primary_data["sequence_postprocess_ms"]).item()
        ),
        "sequence_postprocess_ms_per_frame": float(
            np.asarray(primary_data["sequence_postprocess_ms"]).item()
            / len(primary_data["timestamps"])
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
