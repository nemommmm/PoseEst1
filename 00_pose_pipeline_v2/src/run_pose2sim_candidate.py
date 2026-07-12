"""Run Pose2Sim/OpenSim inverse kinematics from an existing SKT result."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from common.angles import COCO17_NAMES, SEMANTIC_ANGLE_NAMES
from common.research_candidate import CandidateResult


def fill_trajectories(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate missing marker coordinates for OpenSim while retaining a mask."""
    values = np.asarray(points, dtype=np.float64).copy()
    interpolated = ~np.isfinite(values).all(axis=2)
    frame_index = np.arange(len(values), dtype=np.float64)
    for joint_idx in range(values.shape[1]):
        for axis_idx in range(3):
            series = values[:, joint_idx, axis_idx]
            finite = np.isfinite(series)
            if finite.sum() >= 2:
                series[~finite] = np.interp(frame_index[~finite], frame_index[finite], series[finite])
            elif finite.sum() == 1:
                series[~finite] = series[finite][0]
            values[:, joint_idx, axis_idx] = series
    return values, interpolated


def camera_cm_to_opensim_m(points: np.ndarray) -> np.ndarray:
    """Rotate x-right/y-down/z-forward camera coordinates into OpenSim Y-up meters."""
    values = np.asarray(points, dtype=np.float64)
    result = np.empty_like(values)
    result[..., 0] = values[..., 2]
    result[..., 1] = -values[..., 1]
    result[..., 2] = values[..., 0]
    return result / 100.0


def write_trc(path: Path, timestamps: np.ndarray, points_m: np.ndarray) -> None:
    """Write a COCO-17 marker trajectory in OpenSim TRC format."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    median_dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.08
    rate = 1.0 / median_dt
    marker_line = "Frame#\tTime\t" + "\t\t\t".join(COCO17_NAMES) + "\t\t\t\n"
    axis_line = "\t\t" + "\t".join(
        value for marker_idx in range(1, 18) for value in (f"X{marker_idx}", f"Y{marker_idx}", f"Z{marker_idx}")
    ) + "\n"
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        f"{rate:.8f}\t{rate:.8f}\t{len(points_m)}\t17\tm\t{rate:.8f}\t1\t{len(points_m)}\n",
        marker_line,
        axis_line,
        "\n",
    ]
    origin = float(timestamps[0]) if len(timestamps) else 0.0
    for frame_idx, (timestamp, pose) in enumerate(zip(timestamps, points_m, strict=True), start=1):
        coords = "\t".join(f"{value:.8f}" for value in pose.reshape(-1))
        lines.append(f"{frame_idx}\t{timestamp - origin:.8f}\t{coords}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def read_mot(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read an OpenSim MOT coordinate table."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_idx = next(idx for idx, line in enumerate(lines) if line.strip().lower() == "endheader") + 1
    names = lines[header_idx].split()
    values = np.loadtxt(lines[header_idx + 1 :], dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    return values[:, 0], {name: values[:, idx] for idx, name in enumerate(names[1:], start=1)}


def opensim_semantic_angles(coordinates: dict[str, np.ndarray]) -> np.ndarray:
    """Map OpenSim generalized coordinates to the project's eight semantic angles."""
    length = len(next(iter(coordinates.values())))

    def absolute(name: str) -> np.ndarray:
        return np.abs(np.asarray(coordinates.get(name, np.full(length, np.nan)), dtype=np.float64))

    def shoulder(side: str) -> np.ndarray:
        flexion = absolute(f"arm_flex_{side}")
        adduction = absolute(f"arm_add_{side}")
        return np.sqrt(flexion**2 + adduction**2)

    semantic = {
        "LeftShoulder": shoulder("l"),
        "RightShoulder": shoulder("r"),
        "LeftElbow": absolute("elbow_flex_l"),
        "RightElbow": absolute("elbow_flex_r"),
        "LeftHip": absolute("hip_flexion_l"),
        "RightHip": absolute("hip_flexion_r"),
        "LeftKnee": absolute("knee_angle_l"),
        "RightKnee": absolute("knee_angle_r"),
    }
    return np.column_stack([semantic[name] for name in SEMANTIC_ANGLE_NAMES])


def run_pose2sim(project_dir: Path, simple_model: bool) -> Path:
    """Invoke Pose2Sim scaling and inverse kinematics using a COCO-17 TRC."""
    try:
        import Pose2Sim
    except ImportError as exc:
        raise RuntimeError("run this script with the dedicated Pose2Sim environment") from exc
    config = {
        "project": {
            "project_dir": str(project_dir),
            "participant_height": "auto",
            "participant_mass": 70.0,
            "multi_person": False,
            "frame_range": "all",
        },
        "pose": {"pose_model": "BODY"},
        "kinematics": {
            "use_augmentation": False,
            "use_simple_model": simple_model,
            "right_left_symmetry": True,
            "filter_ik": False,
            "remove_individual_scaling_setup": False,
            "remove_individual_ik_setup": False,
            "default_height": 1.75,
            "large_hip_knee_angles": 90,
            "trimmed_extrema_percent": 50,
        },
    }
    (project_dir / "Config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    Pose2Sim.kinematics(config)
    mot_files = sorted((project_dir / "kinematics").glob("*.mot"))
    if not mot_files:
        raise RuntimeError("Pose2Sim completed without producing an MOT file")
    return mot_files[-1]


def main() -> None:
    """Export SKT markers, run IK, and write a canonical candidate NPZ."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skt", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--simple-model", action="store_true")
    args = parser.parse_args()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    pose_dir = args.work_dir / "pose-3d"
    with np.load(args.skt, allow_pickle=False) as payload:
        stop = args.max_frames or len(payload["timestamps"])
        timestamps = np.asarray(payload["timestamps"][:stop], dtype=np.float64)
        raw_keypoints = np.asarray(payload["keypoints"][:stop], dtype=np.float64)
        confidence = np.minimum(payload["conf_left"][:stop], payload["conf_right"][:stop])
        epipolar = payload["epipolar_error"][:stop]
        reprojection = payload["reprojection_error"][:stop]
    filled, interpolation_mask = fill_trajectories(raw_keypoints)
    write_trc(pose_dir / "skt_coco17.trc", timestamps, camera_cm_to_opensim_m(filled))
    start = time.perf_counter()
    mot_path = run_pose2sim(args.work_dir, args.simple_model)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    mot_time, coordinates = read_mot(mot_path)
    angle_matrix = opensim_semantic_angles(coordinates)
    result = CandidateResult(
        candidate_name=f"Pose2Sim-OpenSim-{'simple' if args.simple_model else 'full'}",
        timestamps=timestamps[: len(mot_time)],
        keypoints_3d=raw_keypoints[: len(mot_time)],
        angles_override=angle_matrix,
        confidence_2d=confidence[: len(mot_time)],
        epipolar_error_px=epipolar[: len(mot_time)],
        reprojection_error_px=reprojection[: len(mot_time)],
        stage_time_ms={"opensim_total": elapsed_ms, "opensim_per_frame": elapsed_ms / len(mot_time)},
        metadata={
            "coordinate_unit": "cm",
            "joint_convention": "COCO-17 plus OpenSim generalized coordinates",
            "trc_transform": "[camera_z, -camera_y, camera_x] / 100",
            "interpolated_marker_ratio": float(interpolation_mask.mean()),
            "mot_path": str(mot_path),
            "reference_policy": "Xsens-derived reference is external comparison only",
        },
    )
    print(result.save(args.output))


if __name__ == "__main__":
    main()
