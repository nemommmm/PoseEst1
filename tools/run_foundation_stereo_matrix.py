#!/opt/anaconda3/envs/pose/bin/python
"""Run official FoundationStereo models and sample depth at left-view joints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = PROJECT_ROOT / "00_pose_pipeline_v2" / "src"
sys.path.insert(0, str(PIPELINE_SRC))

from adapt_foundation_stereo_joint_depth import (  # noqa: E402
    adapt,
    rectify_points_sequence,
    restore_full_resolution_disparity,
    sample_joint_disparity,
)
from common.config import load_config, resolve_path, section  # noqa: E402
from common.metrics import jsonable  # noqa: E402


@dataclass(frozen=True)
class StereoRectification:
    """Image maps and matrices for one fixed stereo calibration."""

    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    rectification_left: np.ndarray
    projection_left: np.ndarray
    image_size: tuple[int, int]
    calibration_path: Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str | Path) -> Path:
    """Resolve one path relative to the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def summarize(values: np.ndarray, warmup: int) -> dict[str, float | None]:
    """Summarize finite latency values after warm-up."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    clean = array[:, min(max(warmup, 0), array.shape[1]) :]
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None}
    return {
        "mean_ms": float(np.mean(clean)),
        "median_ms": float(np.median(clean)),
        "p95_ms": float(np.percentile(clean, 95)),
    }


def load_rectification(
    config_path: Path,
    image_size: tuple[int, int],
) -> StereoRectification:
    """Build full-resolution stereo rectification maps."""
    config = load_config(config_path)
    calibration_path = resolve_path(
        section(config, "calibration").get("camera_params"),
        must_exist=True,
    )
    assert calibration_path is not None
    with np.load(calibration_path) as payload:
        matrix_left = np.asarray(payload["mtx_l"], dtype=np.float64)
        distortion_left = np.asarray(payload["dist_l"], dtype=np.float64)
        matrix_right = np.asarray(payload["mtx_r"], dtype=np.float64)
        distortion_right = np.asarray(payload["dist_r"], dtype=np.float64)
        rotation = np.asarray(payload["R"], dtype=np.float64)
        translation = np.asarray(payload["T"], dtype=np.float64)
    (
        rectification_left,
        rectification_right,
        projection_left,
        projection_right,
        _,
        _,
        _,
    ) = cv2.stereoRectify(
        matrix_left,
        distortion_left,
        matrix_right,
        distortion_right,
        image_size,
        rotation,
        translation,
        alpha=0,
    )
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        matrix_left,
        distortion_left,
        rectification_left,
        projection_left,
        image_size,
        cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        matrix_right,
        distortion_right,
        rectification_right,
        projection_right,
        image_size,
        cv2.CV_32FC1,
    )
    return StereoRectification(
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        rectification_left=rectification_left,
        projection_left=projection_left,
        image_size=image_size,
        calibration_path=calibration_path,
    )


def add_repository(repository: Path) -> None:
    """Put exactly one official model repository first on sys.path."""
    repository_string = str(repository.resolve())
    if repository_string in sys.path:
        sys.path.remove(repository_string)
    sys.path.insert(0, repository_string)


def load_foundation_model(
    repository: Path,
    model_path: Path,
    valid_iters: int,
) -> tuple[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Load the official accuracy-focused FoundationStereo checkpoint."""
    add_repository(repository)
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo

    config = OmegaConf.load(str(model_path.parent / "cfg.yaml"))
    if "vit_size" not in config:
        config["vit_size"] = "vitl"
    config["valid_iters"] = int(valid_iters)
    model = FoundationStereo(config)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()

    def infer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            return model.forward(
                left,
                right,
                iters=int(valid_iters),
                test_mode=True,
            )

    return model, infer


def load_fast_model(
    repository: Path,
    model_path: Path,
    valid_iters: int,
    maximum_disparity: int,
) -> tuple[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Load the official Fast-FoundationStereo serialized checkpoint."""
    add_repository(repository)
    from Utils import AMP_DTYPE

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.args.valid_iters = int(valid_iters)
    model.args.max_disp = int(maximum_disparity)
    model.cuda().eval()

    def infer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            return model.forward(
                left,
                right,
                iters=int(valid_iters),
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

    return model, infer


def prepare_tensor_pair(
    left: np.ndarray,
    right: np.ndarray,
    scale: float,
    padder_class: type,
) -> tuple[torch.Tensor, torch.Tensor, Any, tuple[int, int]]:
    """Resize, tensorize, and pad one rectified image pair."""
    scaled_left = cv2.resize(
        left,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    scaled_right = cv2.resize(
        right,
        (scaled_left.shape[1], scaled_left.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    height, width = scaled_left.shape[:2]
    left_tensor = (
        torch.as_tensor(scaled_left)
        .cuda()
        .float()[None]
        .permute(0, 3, 1, 2)
    )
    right_tensor = (
        torch.as_tensor(scaled_right)
        .cuda()
        .float()[None]
        .permute(0, 3, 1, 2)
    )
    padder = padder_class(
        left_tensor.shape,
        divis_by=32,
        force_square=False,
    )
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    return left_tensor, right_tensor, padder, (width, height)


def diagnostic_image(
    rectified_left: np.ndarray,
    disparity: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Build one compact left/disparity diagnostic image."""
    finite = disparity[np.isfinite(disparity) & (disparity > 0)]
    upper = float(np.percentile(finite, 99)) if finite.size else 1.0
    normalized = np.clip(disparity / max(upper, 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap(
        np.uint8(normalized * 255.0),
        cv2.COLORMAP_TURBO,
    )
    left = rectified_left.copy()
    for point in points:
        if np.isfinite(point).all():
            cv2.circle(
                left,
                tuple(np.round(point).astype(int)),
                4,
                (0, 255, 0),
                -1,
            )
    return np.concatenate([left, color], axis=1)


def read_gpu_metadata() -> dict[str, Any]:
    """Collect compact runtime metadata without credentials."""
    query = (
        "name,uuid,memory.total,driver_version"
    )
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return {
        "nvidia_smi": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version()),
    }


def run(args: argparse.Namespace) -> Path:
    """Run one official dense-stereo variant and write canonical results."""
    left_video = project_path(args.left_video)
    right_video = project_path(args.right_video)
    baseline_path = project_path(args.baseline)
    config_path = project_path(args.config)
    repository = Path(args.repository).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not all(
        path.is_file()
        for path in (left_video, right_video, baseline_path, config_path, model_path)
    ):
        raise FileNotFoundError("One or more required input files are missing")
    if not repository.is_dir():
        raise FileNotFoundError(repository)

    left_capture = cv2.VideoCapture(str(left_video))
    right_capture = cv2.VideoCapture(str(right_video))
    width = int(left_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(left_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    right_size = (
        int(right_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(right_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    left_capture.release()
    right_capture.release()
    if (width, height) != right_size:
        raise ValueError("Left and right videos have different image sizes")
    geometry = load_rectification(config_path, (width, height))
    with np.load(baseline_path, allow_pickle=False) as baseline:
        timestamps = np.asarray(baseline["timestamps"], dtype=np.float64)
        raw_left = np.asarray(
            baseline["keypoints_left_2d_raw"],
            dtype=np.float64,
        )
        confidence_left = np.asarray(
            baseline["conf_left"],
            dtype=np.float64,
        )
        if "yolo_left_time_ms" in baseline.files:
            detector_time = np.asarray(
                baseline["yolo_left_time_ms"],
                dtype=np.float64,
            )
            detector_timing_policy = "measured_left_view"
        else:
            detector_time = (
                np.asarray(baseline["yolo_time_ms"], dtype=np.float64) / 2.0
            )
            detector_timing_policy = "estimated_half_of_stereo_detector"
    with np.load(geometry.calibration_path) as calibration:
        matrix_left = np.asarray(calibration["mtx_l"], dtype=np.float64)
        distortion_left = np.asarray(
            calibration["dist_l"],
            dtype=np.float64,
        )
    rectified_left_points = rectify_points_sequence(
        raw_left,
        matrix_left,
        distortion_left,
        geometry.rectification_left,
        geometry.projection_left,
    )
    maximum = min(
        len(timestamps),
        int(args.max_frames) if args.max_frames is not None else len(timestamps),
    )
    timestamps = timestamps[:maximum]
    rectified_left_points = rectified_left_points[:maximum]
    confidence_left = confidence_left[:maximum]
    detector_time = detector_time[:maximum]

    if args.variant == "foundation":
        model, infer = load_foundation_model(
            repository,
            model_path,
            args.valid_iters,
        )
    else:
        model, infer = load_fast_model(
            repository,
            model_path,
            args.valid_iters,
            args.maximum_disparity,
        )
    del model
    from core.utils.utils import InputPadder

    repeat_count = max(1, int(args.repeats))
    inference_times = np.full(
        (repeat_count, maximum),
        np.nan,
        dtype=np.float64,
    )
    dense_pipeline_times = np.full_like(inference_times, np.nan)
    joint_disparity = np.full((maximum, 17), np.nan, dtype=np.float64)
    local_mad = np.full((maximum, 17), np.nan, dtype=np.float64)
    diagnostic_indices = set(
        np.linspace(0, max(maximum - 1, 0), min(args.diagnostic_frames, maximum))
        .round()
        .astype(int)
        .tolist()
    )
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for repeat_index in range(repeat_count):
        left_capture = cv2.VideoCapture(str(left_video))
        right_capture = cv2.VideoCapture(str(right_video))
        for frame_index in range(maximum):
            frame_start = time.perf_counter()
            ok_left, frame_left = left_capture.read()
            ok_right, frame_right = right_capture.read()
            if not ok_left or not ok_right:
                raise RuntimeError(
                    f"Video decode stopped at frame {frame_index}"
                )
            rectified_left = cv2.remap(
                frame_left,
                geometry.left_map_x,
                geometry.left_map_y,
                cv2.INTER_LINEAR,
            )
            rectified_right = cv2.remap(
                frame_right,
                geometry.right_map_x,
                geometry.right_map_y,
                cv2.INTER_LINEAR,
            )
            left_tensor, right_tensor, padder, _ = prepare_tensor_pair(
                rectified_left,
                rectified_right,
                args.scale,
                InputPadder,
            )
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                disparity_tensor = infer(left_tensor, right_tensor)
            torch.cuda.synchronize()
            inference_times[repeat_index, frame_index] = (
                time.perf_counter() - inference_start
            ) * 1000.0
            disparity_scaled = (
                padder.unpad(disparity_tensor.float())
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            if disparity_scaled.ndim != 2:
                raise RuntimeError(
                    f"Unexpected disparity shape: {disparity_scaled.shape}"
                )
            scaled_height = int(round(height * args.scale))
            scaled_width = int(round(width * args.scale))
            disparity_scaled = disparity_scaled[
                :scaled_height,
                :scaled_width,
            ]
            disparity_full = restore_full_resolution_disparity(
                disparity_scaled,
                (width, height),
                args.scale,
            )
            if repeat_index == 0:
                values, mad = sample_joint_disparity(
                    disparity_full,
                    rectified_left_points[frame_index],
                    patch_size=args.patch_size,
                    maximum_mad_px=args.maximum_mad_px,
                )
                joint_disparity[frame_index] = values
                local_mad[frame_index] = mad
                if frame_index in diagnostic_indices:
                    image = diagnostic_image(
                        rectified_left,
                        disparity_full,
                        rectified_left_points[frame_index],
                    )
                    cv2.imwrite(
                        str(
                            diagnostics_dir
                            / f"frame_{frame_index:06d}.jpg"
                        ),
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )
            dense_pipeline_times[repeat_index, frame_index] = (
                time.perf_counter() - frame_start
            ) * 1000.0
        left_capture.release()
        right_capture.release()

    full_pipeline_times = dense_pipeline_times + detector_time[None, :]
    warmup = min(max(int(args.warmup_frames), 0), max(maximum - 1, 0))
    metadata = {
        "schema_version": "foundation_stereo_joint_depth_run_v1",
        "variant": args.variant,
        "candidate_name": args.candidate_name,
        "repository": str(repository),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "config_sha256": sha256_file(config_path),
        "calibration_sha256": sha256_file(geometry.calibration_path),
        "baseline_sha256": sha256_file(baseline_path),
        "left_video_sha256": sha256_file(left_video),
        "right_video_sha256": sha256_file(right_video),
        "image_size": [width, height],
        "scale": float(args.scale),
        "valid_iters": int(args.valid_iters),
        "maximum_disparity": int(args.maximum_disparity),
        "frames": int(maximum),
        "warmup_frames": int(warmup),
        "repeats": int(repeat_count),
        "detector_timing_policy": detector_timing_policy,
        "right_yolo_usage": "diagnostic_only",
        "reference_policy": (
            "Xsens-derived reference is an external comparison system."
        ),
        "gpu_runtime": read_gpu_metadata(),
    }
    timing = {
        "dense_inference": summarize(inference_times, warmup),
        "dense_decode_rectify_inference": summarize(
            dense_pipeline_times,
            warmup,
        ),
        "complete_left_yolo_plus_dense": summarize(
            full_pipeline_times,
            warmup,
        ),
    }
    complete_mean = timing["complete_left_yolo_plus_dense"]["mean_ms"]
    timing["complete_pipeline_fps"] = (
        float(1000.0 / complete_mean)
        if isinstance(complete_mean, float) and complete_mean > 0
        else None
    )
    timing["meets_12_5_fps"] = bool(
        isinstance(complete_mean, float) and complete_mean <= 80.0
    )
    metadata["timing"] = timing

    disparity_path = output_dir / "joint_disparity.npz"
    np.savez_compressed(
        disparity_path,
        joint_disparity_px=joint_disparity,
        local_disparity_mad_px=local_mad,
        inference_time_ms=inference_times[0],
        dense_pipeline_time_ms=dense_pipeline_times[0],
        full_pipeline_time_ms=full_pipeline_times[0],
        rectified_left_points=rectified_left_points,
        left_confidence=confidence_left,
        metadata_json=np.asarray(json.dumps(jsonable(metadata), sort_keys=True)),
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(jsonable(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "timing_internal.json").write_text(
        json.dumps(jsonable(timing), indent=2) + "\n",
        encoding="utf-8",
    )
    candidate_path = output_dir / "candidate_v2.npz"
    adapt(
        disparity_path,
        baseline_path,
        config_path,
        candidate_path,
        args.candidate_name,
        args.maximum_mad_px,
        args.minimum_confidence,
    )
    return candidate_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["foundation", "fast"], required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--valid-iters", type=int, required=True)
    parser.add_argument("--maximum-disparity", type=int, default=192)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--maximum-mad-px", type=float, default=2.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.2)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--diagnostic-frames", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected model variant."""
    args = parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
