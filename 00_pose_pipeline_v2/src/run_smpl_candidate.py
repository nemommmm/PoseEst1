#!/opt/anaconda3/envs/pose/bin/python
"""Run the calibrated-stereo EasyMocap/SMPL feasibility candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
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
from common.research_candidate import CandidateResult, compute_bone_statistics  # noqa: E402
from common.smpl_prior import (  # noqa: E402
    SmplFitConfig,
    fit_smpl_sequence,
    project_points_numpy,
    select_gate_indices,
)


EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def parse_args() -> argparse.Namespace:
    """Parse one SMPL candidate run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", choices=["feasibility", "short", "full"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return one reproducibility digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (SCRIPT_DIR.parents[1] / candidate).resolve()


def gpu_metadata() -> dict[str, Any]:
    """Collect the GPU and Torch versions used by the fitting run."""
    import torch

    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[0]
    name, uuid, memory_mb, driver = [value.strip() for value in line.split(",", 3)]
    return {
        "gpu_name": name,
        "gpu_uuid": uuid,
        "gpu_memory_mb": int(memory_mb),
        "driver": driver,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def projection_matrices(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build rectified projection matrices from the frozen tracked calibration."""
    calibration = resolve(config["calibration"]["camera_params"])
    video = resolve(config["dataset"]["left_video"])
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"could not read video dimensions: {video}")
    with np.load(calibration, allow_pickle=False) as payload:
        _, _, left, right, _, _, _ = cv2.stereoRectify(
            payload["mtx_l"],
            payload["dist_l"],
            payload["mtx_r"],
            payload["dist_r"],
            (width, height),
            payload["R"],
            payload["T"],
            alpha=0,
        )
    return left, right, {
        "calibration_path": str(calibration),
        "calibration_sha256": sha256(calibration),
        "image_width": width,
        "image_height": height,
    }


def angle_jump_count(points: np.ndarray, threshold: float = 10.0) -> int:
    """Count large consecutive changes across core ergonomic angles."""
    angles = compute_angle_sequence(points, list(SEMANTIC_ANGLE_NAMES))
    return sum(
        int(np.count_nonzero(np.isfinite(diff) & (np.abs(diff) > threshold)))
        for values in angles.values()
        for diff in [np.diff(values)]
    )


def mean_bone_cv(points: np.ndarray) -> float:
    """Return the mean temporal coefficient of variation over supported bones."""
    statistics = compute_bone_statistics(points)
    values = []
    for item in statistics.values():
        median = item["median_cm"]
        std = item["std_cm"]
        if median is not None and std is not None and median > 1e-6:
            values.append(std / median)
    return float(np.mean(values)) if values else math.inf


def metric_summary(
    fitted: np.ndarray,
    raw: np.ndarray,
    quality: np.ndarray,
    reprojection: np.ndarray,
    cfg: SmplFitConfig,
) -> dict[str, Any]:
    """Compute geometry-only admission diagnostics."""
    core = slice(5, 17)
    correction = np.linalg.norm(fitted - raw, axis=2)[:, core]
    high_quality = (quality[:, core] >= 0.7) & np.isfinite(correction)
    finite_reprojection = reprojection[:, core][np.isfinite(reprojection[:, core])]
    high_correction = correction[high_quality]
    metrics = {
        "finite_ratio": float(np.isfinite(fitted[:, core]).all(axis=2).mean()),
        "reprojection_p50_px": float(np.median(finite_reprojection)),
        "reprojection_p95_px": float(np.percentile(finite_reprojection, 95.0)),
        "bone_cv_mean": mean_bone_cv(fitted),
        "angle_jump_count": angle_jump_count(fitted),
        "high_quality_correction_median_cm": float(np.median(high_correction)),
        "high_quality_correction_p95_cm": float(np.percentile(high_correction, 95.0)),
    }
    metrics["geometry_gate_pass"] = bool(
        metrics["reprojection_p95_px"] <= cfg.max_reprojection_p95_px
        and metrics["high_quality_correction_median_cm"]
        <= cfg.max_high_quality_correction_cm
    )
    metrics["scientific_status"] = (
        "admit" if metrics["geometry_gate_pass"] else "reject"
    )
    return metrics


def render_diagnostics(
    run_dir: Path,
    raw: np.ndarray,
    fitted: np.ndarray,
    observed_left: np.ndarray,
    observed_right: np.ndarray,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    reprojection: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Save twelve diagnostic figures and a compact H.264 preview."""
    visual_dir = run_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    frame_score = np.nanmedian(reprojection[:, 5:17], axis=1)
    ranked = list(np.argsort(np.nan_to_num(frame_score, nan=-1.0))[::-1][:4])
    uniform = list(np.linspace(0, len(fitted) - 1, min(12, len(fitted)), dtype=int))
    selected = []
    for index in ranked + uniform:
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) == min(12, len(fitted)):
            break

    def draw_frame(index: int, output: Path) -> None:
        projected_left = project_points_numpy(fitted[index:index + 1], projection_left)[0]
        projected_right = project_points_numpy(fitted[index:index + 1], projection_right)[0]
        figure = plt.figure(figsize=(13.5, 4.2))
        for panel, observed, projected, title in (
            (1, observed_left[index], projected_left, "Left rectified view"),
            (2, observed_right[index], projected_right, "Right rectified view"),
        ):
            axis = figure.add_subplot(1, 3, panel)
            for joint_a, joint_b in EDGES:
                if np.isfinite(projected[[joint_a, joint_b]]).all():
                    axis.plot(projected[[joint_a, joint_b], 0], projected[[joint_a, joint_b], 1], color="#d1495b")
            axis.scatter(observed[:, 0], observed[:, 1], s=13, color="#2878b5")
            axis.scatter(projected[:, 0], projected[:, 1], s=15, marker="x", color="#d1495b")
            axis.set(xlim=(0, width), ylim=(height, 0), title=title)
            axis.set_aspect("equal", adjustable="box")
        axis_3d = figure.add_subplot(1, 3, 3, projection="3d")
        for pose, color in ((raw[index], "#9aa5b1"), (fitted[index], "#087f5b")):
            for joint_a, joint_b in EDGES:
                segment = pose[[joint_a, joint_b]]
                if np.isfinite(segment).all():
                    axis_3d.plot(segment[:, 0], segment[:, 2], -segment[:, 1], color=color, linewidth=2)
        axis_3d.set_title("Raw SKT (gray) / SMPL (green)")
        figure.suptitle(f"Frame {index} | median reprojection {frame_score[index]:.2f} px")
        figure.tight_layout()
        figure.savefig(output, dpi=130)
        plt.close(figure)

    for index in selected:
        draw_frame(index, visual_dir / f"frame_{index:04d}.png")

    temporary = visual_dir / "preview_mp4v.mp4"
    preview = visual_dir / "reconstruction_preview_h264.mp4"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (960, 540)
    )
    for index, pose in enumerate(fitted):
        canvas = np.full((540, 960, 3), 248, dtype=np.uint8)
        finite = pose[np.isfinite(pose).all(axis=1)]
        center = np.mean(finite[:, :2], axis=0) if len(finite) else np.zeros(2)
        for joint_a, joint_b in EDGES:
            segment = pose[[joint_a, joint_b]]
            if not np.isfinite(segment).all():
                continue
            pixels = [
                (int(480 + 5 * (point[0] - center[0])), int(270 + 5 * (point[1] - center[1])))
                for point in segment
            ]
            cv2.line(canvas, pixels[0], pixels[1], (75, 127, 8), 4, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"SMPL stereo fit | frame {index}",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (23, 32, 51),
            2,
        )
        writer.write(canvas)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    codec = "mp4v"
    if ffmpeg:
        completed = subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(preview),
            ],
            check=False,
        )
        if completed.returncode == 0:
            temporary.unlink(missing_ok=True)
            codec = "h264"
        else:
            temporary.replace(preview)
    else:
        temporary.replace(preview)
    return {"sampled_frames": selected, "image_count": len(selected), "preview": str(preview), "preview_codec": codec}


def main() -> None:
    """Fit SMPL, write canonical output, diagnostics, metrics, and metadata."""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, args.run_dir / "resolved_config.yaml")
    prior_config = config.get("human_prior", {})
    asset = Path(prior_config.get("smpl_asset", "/workspace/model_assets/smpl/SMPL_NEUTRAL.pkl"))
    regressor = Path(
        prior_config.get(
            "smpl_regressor",
            "/workspace/external/EasyMocap/data/smplx/J_regressor_body25.npy",
        )
    )
    if not asset.is_file():
        (args.run_dir / "metrics.json").write_text(
            json.dumps({"candidate": "smpl", "scientific_status": "asset_blocked"}, indent=2),
            encoding="utf-8",
        )
        return
    source = resolve(prior_config["source_npz"])
    projection_left, projection_right, calibration_meta = projection_matrices(config)
    with np.load(source, allow_pickle=False) as payload:
        indices = select_gate_indices(len(payload["timestamps"]), args.gate)
        timestamps = np.asarray(payload["timestamps"][indices], dtype=np.float64)
        raw = np.asarray(payload["keypoints"][indices], dtype=np.float64)
        observed_left = np.asarray(payload["keypoints_left_rect"][indices], dtype=np.float64)
        observed_right = np.asarray(payload["keypoints_right_rect"][indices], dtype=np.float64)
        confidence_left = np.asarray(payload["triang_conf_left"][indices], dtype=np.float64)
        confidence_right = np.asarray(payload["triang_conf_right"][indices], dtype=np.float64)
        epipolar = np.asarray(payload["epipolar_error"][indices], dtype=np.float64)
        baseline_reprojection = np.asarray(payload["reprojection_error"][indices], dtype=np.float64)
        frame_time = np.asarray(payload["frame_time_ms"][indices], dtype=np.float64)
    fit_config = SmplFitConfig.from_dict(prior_config.get("smpl"))
    fit = fit_smpl_sequence(
        raw,
        observed_left,
        observed_right,
        confidence_left,
        confidence_right,
        epipolar,
        baseline_reprojection,
        projection_left,
        projection_right,
        asset,
        regressor,
        fit_config,
    )
    gpu = gpu_metadata()
    metrics = metric_summary(
        fit.keypoints_3d_cm,
        raw,
        fit.joint_quality,
        fit.reprojection_error_px,
        fit_config,
    )
    end_to_end_ms = float(np.nanmean(frame_time) + fit.stage_time_ms["smpl_per_frame"])
    metrics.update(
        {
            "candidate": "smpl",
            "gate": args.gate,
            "frames": len(timestamps),
            "stage_time_ms": fit.stage_time_ms,
            "estimated_end_to_end_ms": end_to_end_ms,
            "estimated_end_to_end_fps": 1000.0 / end_to_end_ms,
            "gpu": gpu,
            "reference_policy": "Xsens-derived reference is an external comparison system only",
        }
    )
    metadata = {
        "candidate": "easymocap_smpl_stereo",
        "coordinate_unit": "cm",
        "coordinate_frame": "left_rectified_camera",
        "joint_convention": "COCO-17 mapped from BODY-25",
        "source_npz": str(source),
        "source_npz_sha256": sha256(source),
        "smpl_asset_sha256": sha256(asset),
        "smpl_asset_path_private": str(asset),
        "regressor_sha256": sha256(regressor),
        "source_frame_indices": [int(indices[0]), int(indices[-1])],
        "gpu": gpu,
        "reference_policy": "Xsens-derived reference is an external comparison system only",
        **calibration_meta,
    }
    result = CandidateResult(
        candidate_name="easymocap_smpl_stereo",
        timestamps=timestamps,
        keypoints_3d=fit.keypoints_3d_cm,
        keypoints_3d_raw=raw,
        confidence_2d=np.stack([confidence_left, confidence_right], axis=2),
        epipolar_error_px=epipolar,
        reprojection_error_px=fit.reprojection_error_px,
        joint_quality=fit.joint_quality,
        prior_weight=fit.prior_weight,
        stage_time_ms=fit.stage_time_ms,
        metadata=metadata,
        extra_arrays={
            "smpl_pose": fit.poses,
            "smpl_betas": fit.betas,
            "smpl_translation": fit.translation_m,
            "source_frame_indices": indices,
            "projection_left": projection_left,
            "projection_right": projection_right,
        },
    )
    result.save(args.run_dir / "candidate_result.npz")
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.run_dir / "timing.json").write_text(json.dumps(fit.stage_time_ms, indent=2), encoding="utf-8")
    (args.run_dir / "gpu_metadata.json").write_text(json.dumps(gpu, indent=2), encoding="utf-8")
    visuals = render_diagnostics(
        args.run_dir,
        raw,
        fit.keypoints_3d_cm,
        observed_left,
        observed_right,
        projection_left,
        projection_right,
        fit.reprojection_error_px,
        calibration_meta["image_width"],
        calibration_meta["image_height"],
    )
    (args.run_dir / "visuals.json").write_text(json.dumps(visuals, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
