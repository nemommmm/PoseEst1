"""Independent sparse keypoint triangulation (SKT) stage for 00_pose_pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from common.config import resolve_path, section
from stereo_loader import StereoFrameReader, build_synced_timeline


def choose_person(result) -> tuple[np.ndarray, np.ndarray] | None:
    """Choose the largest/highest-confidence detected person from a YOLO result."""
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    scores = result.boxes.conf.cpu().numpy().astype(np.float64)
    keypoints = result.keypoints.xy.cpu().numpy().astype(np.float64)
    conf = result.keypoints.conf.cpu().numpy().astype(np.float64)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    mean_conf = np.nanmean(conf, axis=1)
    idx = int(np.argmax(scores * 0.5 + mean_conf * 0.3 + areas / max(np.nanmax(areas), 1.0) * 0.2))
    return keypoints[idx], conf[idx]


def rectify_points(points: np.ndarray, mtx: np.ndarray, dist: np.ndarray, rect_r: np.ndarray, proj_p: np.ndarray) -> np.ndarray:
    """Rectify 2D keypoints into stereo-rectified pixel coordinates."""
    out = np.full_like(points, np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return out
    pts = points[valid].reshape(-1, 1, 2).astype(np.float64)
    rect = cv2.undistortPoints(pts, mtx, dist, R=rect_r, P=proj_p).reshape(-1, 2)
    out[valid] = rect
    return out


def triangulate_pose(
    p1: np.ndarray,
    p2: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    min_pair_conf: float,
    max_reproj_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate one COCO-17 pose with basic reprojection quality metrics."""
    pose = np.full((17, 3), np.nan, dtype=np.float64)
    reproj = np.full(17, np.nan, dtype=np.float64)
    pair_conf = np.full(17, np.nan, dtype=np.float64)
    epi = np.full(17, np.nan, dtype=np.float64)
    disparity = np.full(17, np.nan, dtype=np.float64)
    for idx in range(17):
        if not (np.isfinite(left[idx]).all() and np.isfinite(right[idx]).all()):
            continue
        c = min(float(conf_l[idx]) if np.isfinite(conf_l[idx]) else 0.0, float(conf_r[idx]) if np.isfinite(conf_r[idx]) else 0.0)
        pair_conf[idx] = c
        epi[idx] = abs(float(left[idx, 1] - right[idx, 1]))
        disparity[idx] = float(left[idx, 0] - right[idx, 0])
        if c < min_pair_conf:
            continue
        homog = cv2.triangulatePoints(p1, p2, left[idx].reshape(2, 1), right[idx].reshape(2, 1)).ravel()
        if abs(float(homog[3])) < 1e-9:
            continue
        point = homog[:3] / homog[3]
        if not np.isfinite(point).all():
            continue
        proj_l = p1 @ np.r_[point, 1.0]
        proj_r = p2 @ np.r_[point, 1.0]
        if abs(proj_l[2]) < 1e-9 or abs(proj_r[2]) < 1e-9:
            continue
        pl = proj_l[:2] / proj_l[2]
        pr = proj_r[:2] / proj_r[2]
        err = 0.5 * (np.linalg.norm(pl - left[idx]) + np.linalg.norm(pr - right[idx]))
        reproj[idx] = float(err)
        if err <= max_reproj_px:
            pose[idx] = point
    stereo_quality = pair_conf.copy()
    finite_reproj = np.isfinite(reproj)
    stereo_quality[finite_reproj] = pair_conf[finite_reproj] / (1.0 + reproj[finite_reproj] / max(max_reproj_px, 1e-6))
    return pose, reproj, pair_conf, epi, disparity, stereo_quality


def run_skt(config: dict, run_dir: Path) -> Path:
    """Run independent SKT inference or reuse an existing configured NPZ."""
    skt = section(config, "skt")
    if skt.get("use_existing_npz", False):
        path = resolve_path(skt.get("existing_npz"), must_exist=True)
        assert path is not None
        print(f"[skt] using existing NPZ: {path}")
        return path

    dataset = section(config, "dataset")
    calib = section(config, "calibration")
    left_video = resolve_path(dataset.get("left_video"), must_exist=True)
    right_video = resolve_path(dataset.get("right_video"), must_exist=True)
    left_meta = resolve_path(dataset.get("left_metadata"), must_exist=True)
    right_meta = resolve_path(dataset.get("right_metadata"), must_exist=True)
    camera_params = resolve_path(calib.get("camera_params"), must_exist=True)
    model_path = resolve_path(skt.get("model_path"), must_exist=True)
    assert left_video and right_video and left_meta and right_meta and camera_params and model_path

    time_s, synced, _, _ = build_synced_timeline(left_meta, right_meta, dataset.get("timestamp_format", "seconds_microseconds_columns"))
    max_frames = skt.get("max_frames")
    if max_frames:
        synced = synced[: int(max_frames)]
        time_s = time_s[: int(max_frames)]

    params = np.load(camera_params)
    mtx_l, dist_l = params["mtx_l"], params["dist_l"]
    mtx_r, dist_r = params["mtx_r"], params["dist_r"]
    r, t = params["R"], params["T"]
    reader = StereoFrameReader(left_video, right_video, synced, rotate_180=bool(dataset.get("rotate_180", False)))
    ok, frame_l, _ = reader.read_synced(0)
    if not ok or frame_l is None:
        raise RuntimeError("Could not read first stereo frame.")
    height, width = frame_l.shape[:2]
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (width, height), r, t, alpha=0)

    model = YOLO(str(model_path))
    keypoints = []
    reproj_all = []
    pair_conf_all = []
    epi_all = []
    disp_all = []
    quality_all = []
    conf_l_all = []
    conf_r_all = []
    left_2d_all = []
    right_2d_all = []
    min_pair_conf = float(skt.get("min_pair_confidence", 0.25))
    max_reproj_px = float(skt.get("max_reprojection_px", 80.0))
    conf_threshold = float(skt.get("confidence_threshold", 0.35))

    for idx in tqdm(range(len(synced)), desc="SKT", unit="frame"):
        ok, frame_l, frame_r = reader.read_synced(idx)
        if not ok or frame_l is None or frame_r is None:
            break
        det_l = choose_person(model(frame_l, conf=conf_threshold, verbose=False)[0])
        det_r = choose_person(model(frame_r, conf=conf_threshold, verbose=False)[0])
        pts_l = np.full((17, 2), np.nan, dtype=np.float64)
        pts_r = np.full((17, 2), np.nan, dtype=np.float64)
        conf_l = np.full(17, np.nan, dtype=np.float64)
        conf_r = np.full(17, np.nan, dtype=np.float64)
        if det_l is not None:
            pts_l, conf_l = det_l
        if det_r is not None:
            pts_r, conf_r = det_r
        rect_l = rectify_points(pts_l, mtx_l, dist_l, r1, p1)
        rect_r = rectify_points(pts_r, mtx_r, dist_r, r2, p2)
        pose, reproj, pair_conf, epi, disparity, quality = triangulate_pose(
            p1, p2, rect_l, rect_r, conf_l, conf_r, min_pair_conf, max_reproj_px
        )
        keypoints.append(pose)
        reproj_all.append(reproj)
        pair_conf_all.append(pair_conf)
        epi_all.append(epi)
        disp_all.append(disparity)
        quality_all.append(quality)
        conf_l_all.append(conf_l)
        conf_r_all.append(conf_r)
        left_2d_all.append(pts_l)
        right_2d_all.append(pts_r)
    reader.release()

    n = len(keypoints)
    out_path = run_dir / skt.get("output_npz", "skt_pose_optimized.npz")
    np.savez(
        out_path,
        timestamps=time_s[:n],
        keypoints=np.asarray(keypoints, dtype=np.float64),
        reprojection_error=np.asarray(reproj_all, dtype=np.float64),
        epipolar_error=np.asarray(epi_all, dtype=np.float64),
        disparity_px=np.asarray(disp_all, dtype=np.float64),
        stereo_quality=np.asarray(quality_all, dtype=np.float64),
        pair_confidence=np.asarray(pair_conf_all, dtype=np.float64),
        conf_left=np.asarray(conf_l_all, dtype=np.float64),
        conf_right=np.asarray(conf_r_all, dtype=np.float64),
        keypoints_left_2d=np.asarray(left_2d_all, dtype=np.float64),
        keypoints_right_2d=np.asarray(right_2d_all, dtype=np.float64),
        model_name=np.asarray(str(model_path.name)),
        postprocess_variant=np.asarray("00_pose_pipeline_simple_skt"),
        reprojection_threshold_px=np.asarray(max_reproj_px, dtype=np.float64),
    )
    print(f"[skt] saved {out_path}")
    return out_path
