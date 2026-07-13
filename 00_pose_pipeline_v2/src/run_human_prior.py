#!/opt/anaconda3/envs/pose/bin/python
"""Run calibrated-stereo human-prior candidates on an existing SKT result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.angles import SEMANTIC_ANGLE_NAMES, compute_angle_sequence  # noqa: E402
from common.config import load_config, resolve_path, section  # noqa: E402
from common.human_prior import (  # noqa: E402
    KinematicFitConfig,
    compute_reprojection_errors,
    fit_kinematic_sequence,
    select_gate_indices,
)
from common.metrics import jsonable  # noqa: E402
from common.research_candidate import CandidateResult  # noqa: E402


SKELETON_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", choices=["kinematic", "smpl"], required=True)
    parser.add_argument("--gate", choices=["feasibility", "short", "full"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-npz", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_metadata() -> dict[str, Any]:
    """Collect best-effort GPU and Torch runtime metadata."""
    metadata: dict[str, Any] = {}
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
        name, uuid, memory_mb, driver = [value.strip() for value in line.split(",", 3)]
        metadata.update({"gpu_name": name, "gpu_uuid": uuid, "gpu_memory_mb": int(memory_mb), "driver": driver})
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        metadata["gpu_name"] = "unavailable"
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
            }
        )
    except ImportError:
        metadata["torch"] = "unavailable"
    return metadata


def build_projection_matrices(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build rectified projection matrices from the frozen tracked calibration."""
    calibration_path = resolve_path(section(config, "calibration").get("camera_params"), must_exist=True)
    left_video = resolve_path(section(config, "dataset").get("left_video"), must_exist=True)
    assert calibration_path is not None and left_video is not None
    capture = cv2.VideoCapture(str(left_video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"could not read video dimensions: {left_video}")
    with np.load(calibration_path, allow_pickle=False) as payload:
        _, _, projection_left, projection_right, _, _, _ = cv2.stereoRectify(
            payload["mtx_l"],
            payload["dist_l"],
            payload["mtx_r"],
            payload["dist_r"],
            (width, height),
            payload["R"],
            payload["T"],
            alpha=0,
        )
    return projection_left, projection_right, {
        "camera_params": str(calibration_path),
        "camera_params_sha256": file_sha256(calibration_path),
        "image_width": width,
        "image_height": height,
    }


def project_points(points: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Project one pose into a rectified image."""
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = homogeneous @ projection.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[:, :2] / projected[:, 2:3]


def _draw_2d(ax: plt.Axes, observed: np.ndarray, projected: np.ndarray, title: str, width: int, height: int) -> None:
    """Draw observed and fitted keypoints in one rectified view."""
    for joint_a, joint_b in SKELETON_EDGES:
        if np.isfinite(projected[[joint_a, joint_b]]).all():
            ax.plot(projected[[joint_a, joint_b], 0], projected[[joint_a, joint_b], 1], color="#d1495b", linewidth=1.5)
    valid_observed = np.isfinite(observed).all(axis=1)
    valid_projected = np.isfinite(projected).all(axis=1)
    ax.scatter(observed[valid_observed, 0], observed[valid_observed, 1], s=16, color="#2878b5", label="2D observation")
    ax.scatter(projected[valid_projected, 0], projected[valid_projected, 1], s=18, marker="x", color="#d1495b", label="fitted projection")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(alpha=0.15)


def _draw_3d(ax: plt.Axes, raw: np.ndarray, fitted: np.ndarray) -> None:
    """Draw raw and human-prior skeletons in 3D."""
    for pose, color, label in ((raw, "#9aa5b1", "raw SKT"), (fitted, "#087f5b", "human prior")):
        first = True
        for joint_a, joint_b in SKELETON_EDGES:
            segment = pose[[joint_a, joint_b]]
            if np.isfinite(segment).all():
                ax.plot(segment[:, 0], segment[:, 2], -segment[:, 1], color=color, linewidth=2.0, label=label if first else None)
                first = False
    finite = fitted[np.isfinite(fitted).all(axis=1)]
    if len(finite):
        center = np.mean(finite, axis=0)
        radius = max(float(np.ptp(finite, axis=0).max()) * 0.65, 20.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(-center[1] - radius, -center[1] + radius)
    ax.set_title("3D reconstruction")
    ax.set_xlabel("X cm")
    ax.set_ylabel("Z cm")
    ax.set_zlabel("-Y cm")


def select_visual_frames(raw: np.ndarray, fitted: np.ndarray, quality: np.ndarray, reprojection: np.ndarray) -> list[int]:
    """Select up to twelve deterministic diagnostic frames."""
    count = len(raw)
    correction = np.nanmedian(np.linalg.norm(fitted - raw, axis=2), axis=1)
    reprojection_frame = np.nanmedian(reprojection[:, 5:17], axis=1)
    quality_frame = np.nanmedian(quality[:, 5:17], axis=1)
    selected: list[int] = []

    def append_ranked(values: np.ndarray, reverse: bool, amount: int) -> None:
        finite = np.where(np.isfinite(values))[0]
        order = finite[np.argsort(values[finite])]
        if reverse:
            order = order[::-1]
        for index in order[:amount]:
            if int(index) not in selected:
                selected.append(int(index))

    append_ranked(reprojection_frame, True, 2)
    append_ranked(correction, True, 2)
    append_ranked(quality_frame, False, 2)
    for index in np.linspace(0, max(count - 1, 0), min(12, count), dtype=int):
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) >= min(12, count):
            break
    return sorted(selected[:12])


def render_artifacts(
    run_dir: Path,
    raw: np.ndarray,
    fitted: np.ndarray,
    observations_left: np.ndarray,
    observations_right: np.ndarray,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    quality: np.ndarray,
    reprojection: np.ndarray,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    """Render standard sampled reconstruction figures and a compact preview."""
    visual_dir = run_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    selected = select_visual_frames(raw, fitted, quality, reprojection)
    rendered_images: list[Path] = []
    for frame_index in selected:
        projected_left = project_points(fitted[frame_index], projection_left)
        projected_right = project_points(fitted[frame_index], projection_right)
        figure = plt.figure(figsize=(13.5, 4.2))
        left_axis = figure.add_subplot(1, 3, 1)
        right_axis = figure.add_subplot(1, 3, 2)
        pose_axis = figure.add_subplot(1, 3, 3, projection="3d")
        _draw_2d(left_axis, observations_left[frame_index], projected_left, "Left rectified view", width, height)
        _draw_2d(right_axis, observations_right[frame_index], projected_right, "Right rectified view", width, height)
        _draw_3d(pose_axis, raw[frame_index], fitted[frame_index])
        handles, labels = left_axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        figure.suptitle(f"Frame {frame_index} | median reprojection {np.nanmedian(reprojection[frame_index]):.2f} px")
        figure.tight_layout(rect=(0, 0.08, 1, 0.94))
        output = visual_dir / f"frame_{frame_index:04d}.png"
        figure.savefig(output, dpi=130)
        plt.close(figure)
        rendered_images.append(output)

    preview_path = visual_dir / "reconstruction_preview.mp4"
    preview_written = False
    canvas_size = (960, 540)
    writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), canvas_size)
    if writer.isOpened():
        x_values = fitted[:, 5:17, 0]
        y_values = fitted[:, 5:17, 1]
        finite_x = x_values[np.isfinite(x_values)]
        finite_y = y_values[np.isfinite(y_values)]
        x_range = (float(np.min(finite_x)), float(np.max(finite_x))) if len(finite_x) else (-100.0, 100.0)
        y_range = (float(np.min(finite_y)), float(np.max(finite_y))) if len(finite_y) else (-100.0, 100.0)
        for frame_index, pose in enumerate(fitted):
            canvas = np.full((canvas_size[1], canvas_size[0], 3), 248, dtype=np.uint8)
            for joint_a, joint_b in SKELETON_EDGES:
                segment = pose[[joint_a, joint_b]]
                if not np.isfinite(segment).all():
                    continue
                points = []
                for point in segment:
                    x = int(120 + 720 * (point[0] - x_range[0]) / max(x_range[1] - x_range[0], 1e-6))
                    y = int(70 + 400 * (point[1] - y_range[0]) / max(y_range[1] - y_range[0], 1e-6))
                    points.append((x, y))
                cv2.line(canvas, points[0], points[1], (75, 127, 8), 4, cv2.LINE_AA)
            cv2.putText(canvas, f"Human-prior reconstruction | frame {frame_index}", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (23, 32, 51), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"median reprojection: {np.nanmedian(reprojection[frame_index]):.2f} px", (25, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (23, 32, 51), 2, cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
        preview_written = preview_path.is_file() and preview_path.stat().st_size > 0
    return {
        "sampled_frames": selected,
        "image_count": len(rendered_images),
        "preview": str(preview_path) if preview_written else None,
        "preview_codec": "mp4v",
    }


def save_resolved_config(config: dict[str, Any], run_dir: Path) -> None:
    """Save the exact resolved experiment configuration."""
    serializable = {key: value for key, value in config.items() if key != "_config_path"}
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")


def run_smpl_asset_gate(config: dict[str, Any], run_dir: Path, gate: str) -> None:
    """Record the licensed SMPL asset gate without using unofficial weights."""
    raw_asset = section(config, "human_prior").get("smpl_asset", "/workspace/model_assets/smpl/SMPL_NEUTRAL.pkl")
    asset = Path(str(raw_asset)).expanduser()
    status = "ready" if asset.is_file() else "asset_blocked"
    metrics = {
        "candidate": "smpl",
        "gate": gate,
        "scientific_status": status,
        "asset_path": str(asset),
        "reference_policy": "Xsens-derived reference is external comparison only",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if status == "ready":
        raise RuntimeError("SMPL asset is present; the verified EasyMocap fitting adapter must run before reporting results")


def run_kinematic(config: dict[str, Any], run_dir: Path, gate: str, source_override: Path | None) -> None:
    """Run the lightweight calibrated stereo kinematic candidate."""
    prior_cfg = section(config, "human_prior")
    source = source_override or resolve_path(prior_cfg.get("source_npz"), must_exist=True)
    if source is None:
        raise ValueError("human_prior.source_npz is required")
    projection_left, projection_right, calibration_metadata = build_projection_matrices(config)
    with np.load(source, allow_pickle=False) as payload:
        indices = select_gate_indices(len(payload["timestamps"]), gate)
        timestamps = np.asarray(payload["timestamps"][indices], dtype=np.float64)
        raw = np.asarray(payload["keypoints"][indices], dtype=np.float64)
        observations_left = np.asarray(payload["keypoints_left_rect"][indices], dtype=np.float64)
        observations_right = np.asarray(payload["keypoints_right_rect"][indices], dtype=np.float64)
        confidence_left = np.asarray(payload["triang_conf_left"][indices], dtype=np.float64)
        confidence_right = np.asarray(payload["triang_conf_right"][indices], dtype=np.float64)
        epipolar = np.asarray(payload["epipolar_error"][indices], dtype=np.float64)
        baseline_reprojection = np.asarray(payload["reprojection_error"][indices], dtype=np.float64)
        frame_time = np.asarray(payload["frame_time_ms"][indices], dtype=np.float64) if "frame_time_ms" in payload.files else None
        model_name = str(np.asarray(payload["model_name"]).item()) if "model_name" in payload.files else "unknown"

    fit_config = KinematicFitConfig.from_dict(prior_cfg.get("kinematic"))
    fit = fit_kinematic_sequence(
        raw,
        observations_left,
        observations_right,
        confidence_left,
        confidence_right,
        epipolar,
        baseline_reprojection,
        projection_left,
        projection_right,
        fit_config,
    )
    gpu = gpu_metadata()
    evaluation_names = tuple(section(config, "evaluation").get("angle_names", SEMANTIC_ANGLE_NAMES))
    total_online_ms = None
    if frame_time is not None and np.isfinite(frame_time).any():
        total_online_ms = float(np.nanmean(frame_time) + fit.stage_time_ms["human_prior_per_frame"])
    metadata = {
        "candidate": "kinematic",
        "gate": gate,
        "source_npz": str(source),
        "source_npz_sha256": file_sha256(source),
        "source_model": model_name,
        "source_frame_indices": [int(indices[0]), int(indices[-1])],
        "coordinate_unit": "cm",
        "coordinate_frame": "left_rectified_camera",
        "joint_convention": "COCO-17",
        "selected_weights": fit.selected_weights,
        "bone_lengths_cm": fit.bone_lengths_cm,
        "scientific_status": fit.metrics["scientific_status"],
        "reference_policy": "Xsens-derived reference is external comparison only",
        "gpu": gpu,
        **calibration_metadata,
    }
    result = CandidateResult(
        candidate_name="stereo_kinematic_reprojection",
        timestamps=timestamps,
        keypoints_3d=fit.keypoints_3d,
        keypoints_3d_raw=raw,
        angle_names=evaluation_names,
        confidence_2d=np.stack([confidence_left, confidence_right], axis=2),
        epipolar_error_px=epipolar,
        reprojection_error_px=fit.reprojection_error_px,
        joint_quality=fit.joint_quality,
        prior_weight=fit.prior_weight,
        stage_time_ms=fit.stage_time_ms,
        metadata=metadata,
        extra_arrays={
            "source_frame_indices": indices,
            "projection_left": projection_left,
            "projection_right": projection_right,
        },
    )
    result.save(run_dir / "candidate_result.npz")
    metrics = {
        "candidate": "kinematic",
        "gate": gate,
        "frames": len(indices),
        "selected_weights": fit.selected_weights,
        "geometry": fit.metrics,
        "grid_results": fit.grid_results,
        "stage_time_ms": fit.stage_time_ms,
        "source_online_mean_ms": float(np.nanmean(frame_time)) if frame_time is not None else None,
        "estimated_end_to_end_ms": total_online_ms,
        "estimated_end_to_end_fps": 1000.0 / total_online_ms if total_online_ms else None,
        "gpu": gpu,
        "reference_policy": "Xsens-derived reference is external comparison only",
    }
    (run_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    (run_dir / "timing.json").write_text(
        json.dumps(jsonable({"stage_time_ms": fit.stage_time_ms, "estimated_end_to_end_ms": total_online_ms}), indent=2),
        encoding="utf-8",
    )
    visual_metadata = render_artifacts(
        run_dir,
        raw,
        fit.keypoints_3d,
        observations_left,
        observations_right,
        projection_left,
        projection_right,
        fit.joint_quality,
        fit.reprojection_error_px,
        calibration_metadata["image_width"],
        calibration_metadata["image_height"],
        fps=12.5,
    )
    (run_dir / "visuals.json").write_text(json.dumps(visual_metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Run one configured candidate and save a local-sync-ready result directory."""
    args = parse_args()
    config = load_config(args.config)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir)
    (run_dir / "gpu_metadata.json").write_text(json.dumps(gpu_metadata(), indent=2), encoding="utf-8")
    if args.candidate == "kinematic":
        run_kinematic(config, run_dir, args.gate, args.source_npz)
    else:
        run_smpl_asset_gate(config, run_dir, args.gate)
    print(run_dir)


if __name__ == "__main__":
    main()
