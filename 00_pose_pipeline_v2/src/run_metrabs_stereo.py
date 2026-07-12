"""Run calibrated MeTRAbs inference on synchronized stereo video pairs."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from common.config import load_config, resolve_path, section
from common.research_candidate import CandidateResult, map_to_coco17
from stereo_loader import StereoFrameReader, build_synced_timeline


def install_posepile_inference_stub() -> None:
    """Avoid importing PosePile dataset-only dependencies during inference."""
    if "posepile.datasets3d" not in sys.modules:
        module = types.ModuleType("posepile.datasets3d")
        module.get_joint_info = lambda name: None
        sys.modules["posepile.datasets3d"] = module


def load_model(model_dir: Path):
    """Load the official experimental PyTorch MeTRAbs checkpoint."""
    install_posepile_inference_stub()
    import posepile.joint_info
    import metrabs_pytorch.backbones.efficientnet as efficientnet
    import metrabs_pytorch.models.metrabs as metrabs_model
    from metrabs_pytorch.multiperson import multiperson_model
    from metrabs_pytorch.util import get_config

    cfg = get_config(str((model_dir / "config.yaml").resolve()))
    joint_payload = np.load(model_dir / "joint_info.npz")
    joint_info = posepile.joint_info.JointInfo(
        joint_payload["joint_names"], joint_payload["joint_edges"]
    )
    backbone_raw = getattr(efficientnet, f"efficientnet_v2_{cfg.efficientnet_size}")()
    crop_model = metrabs_model.Metrabs(
        torch.nn.Sequential(efficientnet.PreprocLayer(), backbone_raw.features), joint_info
    ).cuda().eval()
    with torch.inference_mode():
        crop_model(
            (
                torch.zeros((1, 3, cfg.proc_side, cfg.proc_side), device="cuda"),
                torch.eye(3, device="cuda")[None],
            )
        )
    crop_model.load_state_dict(
        torch.load(model_dir / "ckpt.pt", map_location="cuda", weights_only=True)
    )
    skeleton_infos = pickle.loads((model_dir / "skeleton_infos.pkl").read_bytes())
    transform = np.load(model_dir / "joint_transform_matrix.npy")
    # The detector is not used because this experiment reuses deterministic
    # SKT/YOLO boxes. Avoid constructing Ultralytics inside the wrapper: its
    # public ``train`` method is incompatible with recursive nn.Module.eval().
    multiperson_model.person_detector.PersonDetector = torch.nn.Identity
    with torch.device("cuda"):
        estimator = multiperson_model.Pose3dEstimator(crop_model, skeleton_infos, transform)
    estimator = estimator.cuda().eval()
    return estimator, cfg, skeleton_infos


def rotated_camera_parameters(
    calibration: np.lib.npyio.NpzFile,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return undistortion maps, rotated intrinsics, and world-to-camera extrinsics."""
    intrinsics = []
    maps = []
    for side in ("l", "r"):
        matrix = np.asarray(calibration[f"mtx_{side}"], dtype=np.float64)
        distortion = np.asarray(calibration[f"dist_{side}"], dtype=np.float64)
        map_x, map_y = cv2.initUndistortRectifyMap(
            matrix, distortion, np.eye(3), matrix, (width, height), cv2.CV_32FC1
        )
        rotated = matrix.copy()
        rotated[0, 2] = width - 1 - matrix[0, 2]
        rotated[1, 2] = height - 1 - matrix[1, 2]
        intrinsics.append(rotated)
        maps.append((map_x, map_y))
    sensor_rotation = np.diag([-1.0, -1.0, 1.0])
    extrinsic_left = np.eye(4, dtype=np.float64)
    extrinsic_left[:3, :3] = sensor_rotation
    extrinsic_right = np.eye(4, dtype=np.float64)
    extrinsic_right[:3, :3] = sensor_rotation @ calibration["R"]
    extrinsic_right[:3, 3] = (sensor_rotation @ calibration["T"].reshape(3)) * 10.0
    return maps[0][0], maps[0][1], maps[1][0], maps[1][1], np.stack(intrinsics), np.stack(
        [extrinsic_left, extrinsic_right]
    )


def undistort_rotated_points(
    points: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Undistort 2D points and apply the dataset's 180-degree display rotation."""
    values = np.asarray(points, dtype=np.float64)
    output = np.full_like(values, np.nan)
    valid = np.isfinite(values).all(axis=1)
    if valid.any():
        undistorted = cv2.undistortPoints(
            values[valid, None, :], matrix, distortion, P=matrix
        )[:, 0]
        undistorted[:, 0] = width - 1 - undistorted[:, 0]
        undistorted[:, 1] = height - 1 - undistorted[:, 1]
        output[valid] = undistorted
    return output


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert one xyxy box to xywh."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return np.asarray([x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32)


def transform_box(
    box: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Transform a distorted rotated xyxy box to the undistorted rotated image."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    corners_rotated = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    corners_sensor = np.column_stack(
        [width - 1 - corners_rotated[:, 0], height - 1 - corners_rotated[:, 1]]
    )
    undistorted_sensor = cv2.undistortPoints(
        corners_sensor[:, None, :], matrix, distortion, P=matrix
    )[:, 0]
    corners = np.column_stack(
        [width - 1 - undistorted_sensor[:, 0], height - 1 - undistorted_sensor[:, 1]]
    )
    return xyxy_to_xywh(
        [corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()]
    )


def fuse_views(
    poses3d_cm: np.ndarray,
    poses2d: np.ndarray,
    observed_2d: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse two world-space poses using agreement with observed YOLO keypoints."""
    errors = np.linalg.norm(poses2d - observed_2d, axis=2)
    errors[~np.isfinite(observed_2d).all(axis=2)] = np.nan
    weights = confidence * np.exp(-np.nan_to_num(errors, nan=1000.0) / 25.0)
    weights[~np.isfinite(poses3d_cm).all(axis=2)] = 0.0
    denominator = weights.sum(axis=0)
    fused = np.full((17, 3), np.nan, dtype=np.float64)
    valid = denominator > 1e-8
    fused[valid] = (
        poses3d_cm[:, valid] * weights[:, valid, None]
    ).sum(axis=0) / denominator[valid, None]
    stereo_consistency = np.linalg.norm(poses3d_cm[0] - poses3d_cm[1], axis=1)
    reprojection = np.nanmean(errors, axis=0)
    return fused, stereo_consistency, reprojection


def main() -> None:
    """Run the configured calibrated stereo MeTRAbs evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--num-aug", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = section(config, "dataset")
    calib_cfg = section(config, "calibration")
    left_video = resolve_path(dataset["left_video"], must_exist=True)
    right_video = resolve_path(dataset["right_video"], must_exist=True)
    left_meta = resolve_path(dataset["left_metadata"], must_exist=True)
    right_meta = resolve_path(dataset["right_metadata"], must_exist=True)
    calibration_path = resolve_path(calib_cfg["camera_params"], must_exist=True)
    assert left_video and right_video and left_meta and right_meta and calibration_path
    timestamps, synced, _, _ = build_synced_timeline(
        left_meta, right_meta, dataset.get("timestamp_format", "seconds_microseconds_columns")
    )
    with np.load(args.baseline_npz, allow_pickle=False) as baseline:
        limit = min(len(synced), len(baseline["timestamps"]), args.max_frames or len(synced))
        boxes_left = np.asarray(baseline["bbox_left"][:limit])
        boxes_right = np.asarray(baseline["bbox_right"][:limit])
        observed_left = np.asarray(baseline["keypoints_left_2d"][:limit])
        observed_right = np.asarray(baseline["keypoints_right_2d"][:limit])
        conf_left = np.asarray(baseline["conf_left"][:limit])
        conf_right = np.asarray(baseline["conf_right"][:limit])
        epipolar = np.asarray(baseline["epipolar_error"][:limit])
    synced = synced[:limit]
    timestamps = timestamps[:limit]
    reader = StereoFrameReader(left_video, right_video, synced, rotate_180=False)
    ok, first_left, first_right = reader.read_synced(0)
    if not ok or first_left is None or first_right is None:
        raise RuntimeError("could not read the first stereo pair")
    height, width = first_left.shape[:2]
    calibration = np.load(calibration_path)
    map_lx, map_ly, map_rx, map_ry, intrinsics, extrinsics = rotated_camera_parameters(
        calibration, width, height
    )
    model, model_cfg, skeleton_infos = load_model(args.model_dir)
    source_names = skeleton_infos["coco_19"]["names"]
    outputs = []
    consistencies = []
    reprojections = []
    decode_times = []
    preprocess_times = []
    inference_times = []
    fusion_times = []
    try:
        for frame_idx in range(limit):
            decode_start = time.perf_counter()
            ok, frame_left, frame_right = reader.read_synced_sequential(frame_idx)
            decode_times.append((time.perf_counter() - decode_start) * 1000.0)
            if not ok or frame_left is None or frame_right is None:
                break
            preprocess_start = time.perf_counter()
            frame_left = cv2.rotate(
                cv2.remap(frame_left, map_lx, map_ly, cv2.INTER_LINEAR), cv2.ROTATE_180
            )
            frame_right = cv2.rotate(
                cv2.remap(frame_right, map_rx, map_ry, cv2.INTER_LINEAR), cv2.ROTATE_180
            )
            frames = torch.from_numpy(
                np.stack([frame_left, frame_right])[:, :, :, ::-1].copy()
            ).permute(0, 3, 1, 2).cuda()
            boxes = torch.stack([
                torch.tensor(
                    transform_box(boxes_left[frame_idx], calibration["mtx_l"], calibration["dist_l"], width, height)[None],
                    device="cuda",
                ),
                torch.tensor(
                    transform_box(boxes_right[frame_idx], calibration["mtx_r"], calibration["dist_r"], width, height)[None],
                    device="cuda",
                ),
            ])
            observed = np.stack(
                [
                    undistort_rotated_points(observed_left[frame_idx], calibration["mtx_l"], calibration["dist_l"], width, height),
                    undistort_rotated_points(observed_right[frame_idx], calibration["mtx_r"], calibration["dist_r"], width, height),
                ]
            )
            preprocess_times.append((time.perf_counter() - preprocess_start) * 1000.0)
            inference_start = time.perf_counter()
            with torch.inference_mode(), torch.device("cuda"):
                boxes_with_confidence = torch.cat(
                    [boxes, torch.ones_like(boxes[..., :1])], dim=-1
                )
                prediction = model._estimate_poses_batched(
                    images=frames,
                    boxes=boxes_with_confidence,
                    intrinsic_matrix=torch.as_tensor(intrinsics, dtype=torch.float32, device="cuda"),
                    distortion_coeffs=torch.zeros((2, 5), dtype=torch.float32, device="cuda"),
                    extrinsic_matrix=torch.as_tensor(extrinsics, dtype=torch.float32, device="cuda"),
                    world_up_vector=torch.tensor([0.0, -1.0, 0.0], device="cuda"),
                    default_fov_degrees=55,
                    skeleton="coco_19",
                    num_aug=args.num_aug,
                    average_aug=True,
                    internal_batch_size=max(2 * args.num_aug, 2),
                    antialias_factor=1,
                    suppress_implausible_poses=False,
                )
            torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - inference_start) * 1000.0)
            fusion_start = time.perf_counter()
            poses3d_source = np.stack([value[0].detach().cpu().numpy() for value in prediction["poses3d"]])
            poses2d_source = np.stack([value[0].detach().cpu().numpy() for value in prediction["poses2d"]])
            poses3d = np.stack([map_to_coco17(pose[None], source_names)[0] for pose in poses3d_source]) / 10.0
            poses2d = np.stack([map_to_coco17(np.column_stack([pose, np.zeros(len(pose))])[None], source_names)[0, :, :2] for pose in poses2d_source])
            fused, consistency, reprojection = fuse_views(
                poses3d, poses2d, observed, np.stack([conf_left[frame_idx], conf_right[frame_idx]])
            )
            outputs.append(fused)
            consistencies.append(consistency)
            reprojections.append(reprojection)
            fusion_times.append((time.perf_counter() - fusion_start) * 1000.0)
    finally:
        reader.release()
    frame_count = len(outputs)
    result = CandidateResult(
        candidate_name=f"MeTRAbs-EfficientNetV2-{model_cfg.efficientnet_size}-stereo",
        timestamps=timestamps[:frame_count],
        keypoints_3d=np.asarray(outputs),
        confidence_2d=np.minimum(conf_left[:frame_count], conf_right[:frame_count]),
        epipolar_error_px=epipolar[:frame_count],
        reprojection_error_px=np.asarray(reprojections),
        stereo_consistency_cm=np.asarray(consistencies),
        stage_time_ms={
            "decode": np.asarray(decode_times),
            "preprocess": np.asarray(preprocess_times),
            "inference": np.asarray(inference_times),
            "fusion": np.asarray(fusion_times),
        },
        metadata={
            "model_dir": str(args.model_dir),
            "model_repository_commit": "8b2b116dd27372e7dbd8207809f868df4e3f852e",
            "coordinate_unit": "cm",
            "joint_convention": "MeTRAbs coco_19 mapped to COCO-17",
            "num_aug": args.num_aug,
            "rotation_calibration": "180-degree sensor rotation included in extrinsics",
            "fusion": "YOLO-confidence times exp(-2D disagreement / 25px)",
            "license_note": "official pretrained weights are non-commercial research only",
            "reference_policy": "Xsens-derived reference is external comparison only",
        },
    )
    print(result.save(args.output))


if __name__ == "__main__":
    main()
