"""Sparse keypoint triangulation (SKT) inference for 00_pose_pipeline_v2."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from common.config import resolve_path, section
from common.person_tracking import (
    DetectionCandidate,
    StereoSanityConfig,
    TrackState,
    TrackingConfig,
    extract_candidates,
    select_candidate,
    stereo_sanity_check,
    infer_tracked_pose,
)
from common.triangulation import (
    TemporalWindowConfig,
    TriangulationConfig,
    rectify_points,
    retriangulate_sequence,
    temporal_window_rescue_rectified,
)
from stereo_loader import StereoFrameReader, build_synced_timeline

NUM_COCO_JOINTS = 17


def choose_person(
    result,
    img_width: int = 0,
    center_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Backward-compatible single-frame person selector for diagnostics."""
    cfg = TrackingConfig(enabled=False, center_person_weight=center_weight)
    candidates = extract_candidates(result)
    frame_shape = (1, max(int(img_width), 1), 3)
    candidate, _ = select_candidate(candidates, None, frame_shape, cfg)
    if candidate is None:
        return None
    return candidate.keypoints, candidate.conf


def _empty_keypoints() -> np.ndarray:
    """Return an empty COCO-17 2D keypoint array."""
    return np.full((NUM_COCO_JOINTS, 2), np.nan, dtype=np.float64)


def _empty_conf() -> np.ndarray:
    """Return an empty COCO-17 confidence array."""
    return np.full(NUM_COCO_JOINTS, np.nan, dtype=np.float64)


def _candidate_payload(candidate: DetectionCandidate | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return keypoints, confidence, and bbox arrays for a candidate."""
    if candidate is None:
        return _empty_keypoints(), _empty_conf(), np.full(4, np.nan, dtype=np.float64)
    return (
        np.asarray(candidate.keypoints, dtype=np.float64),
        np.asarray(candidate.conf, dtype=np.float64),
        np.asarray(candidate.bbox, dtype=np.float64),
    )


def _variant_name(tri_cfg: TriangulationConfig, tw_cfg: TemporalWindowConfig, tracking_cfg: TrackingConfig) -> str:
    """Build a compact SKT variant label for saved NPZ metadata."""
    parts = ["v2_tracked" if tracking_cfg.enabled else "v2_untracked", "weighted_dlt"]
    if tri_cfg.enforce_epipolar_constraint:
        parts.append("soft_epipolar")
    if tw_cfg.enabled:
        parts.append("window_rescue")
    return "_".join(parts)


def run_skt(config: dict, run_dir: Path) -> Path:
    """Run SKT inference or reuse an existing configured NPZ."""
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

    time_s, synced, _, _ = build_synced_timeline(
        left_meta,
        right_meta,
        dataset.get("timestamp_format", "seconds_microseconds_columns"),
    )
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

    tracking_cfg = TrackingConfig.from_skt_config(skt)
    sanity_cfg = StereoSanityConfig.from_skt_config(skt)
    tri_cfg = TriangulationConfig.from_skt_config(skt)
    tw_cfg = TemporalWindowConfig.from_skt_config(skt)

    model = YOLO(str(model_path))
    track_l = TrackState()
    track_r = TrackState()

    left_2d_raw: list[np.ndarray] = []
    right_2d_raw: list[np.ndarray] = []
    left_rect: list[np.ndarray] = []
    right_rect: list[np.ndarray] = []
    conf_l_raw: list[np.ndarray] = []
    conf_r_raw: list[np.ndarray] = []
    bbox_l_all: list[np.ndarray] = []
    bbox_r_all: list[np.ndarray] = []
    track_score_l: list[float] = []
    track_score_r: list[float] = []
    track_source_l: list[str] = []
    track_source_r: list[str] = []
    sanity_ok_all: list[bool] = []
    sanity_reason_all: list[str] = []

    for idx in tqdm(range(len(synced)), desc="SKT", unit="frame"):
        ok, frame_l, frame_r = reader.read_synced_sequential(idx)
        if not ok or frame_l is None or frame_r is None:
            break

        cand_l, track_l = infer_tracked_pose(model, frame_l, track_l, idx, tracking_cfg)
        cand_r, track_r = infer_tracked_pose(model, frame_r, track_r, idx, tracking_cfg)
        pts_l, conf_l, bbox_l = _candidate_payload(cand_l)
        pts_r, conf_r, bbox_r = _candidate_payload(cand_r)

        rect_l = rectify_points(pts_l, mtx_l, dist_l, r1, p1)
        rect_r = rectify_points(pts_r, mtx_r, dist_r, r2, p2)
        sanity_ok, sanity_reason = stereo_sanity_check(cand_l, cand_r, rect_l, rect_r, sanity_cfg)
        if not sanity_ok:
            rect_l = _empty_keypoints()
            rect_r = _empty_keypoints()

        left_2d_raw.append(pts_l)
        right_2d_raw.append(pts_r)
        left_rect.append(rect_l)
        right_rect.append(rect_r)
        conf_l_raw.append(conf_l)
        conf_r_raw.append(conf_r)
        bbox_l_all.append(bbox_l)
        bbox_r_all.append(bbox_r)
        track_score_l.append(float(track_l.last_score))
        track_score_r.append(float(track_r.last_score))
        track_source_l.append(str(track_l.last_source))
        track_source_r.append(str(track_r.last_source))
        sanity_ok_all.append(bool(sanity_ok))
        sanity_reason_all.append(str(sanity_reason))

    reader.release()

    n_frames = len(left_rect)
    if n_frames == 0:
        raise RuntimeError("SKT inference produced no frames.")

    timestamps = time_s[:n_frames]
    left_rect_arr = np.asarray(left_rect, dtype=np.float64)
    right_rect_arr = np.asarray(right_rect, dtype=np.float64)
    conf_l_arr = np.asarray(conf_l_raw, dtype=np.float64)
    conf_r_arr = np.asarray(conf_r_raw, dtype=np.float64)

    pass1 = retriangulate_sequence(p1, p2, left_rect_arr, right_rect_arr, conf_l_arr, conf_r_arr, tri_cfg)
    final = pass1
    rescue_mask_left = np.zeros(left_rect_arr.shape[:2], dtype=bool)
    rescue_mask_right = np.zeros(right_rect_arr.shape[:2], dtype=bool)
    if tw_cfg.enabled:
        (
            rescued_left,
            rescued_right,
            rescued_conf_left,
            rescued_conf_right,
            rescue_mask_left,
            rescue_mask_right,
        ) = temporal_window_rescue_rectified(
            left_rect_arr,
            right_rect_arr,
            conf_l_arr,
            conf_r_arr,
            timestamps,
            pass1["keypoints"],
            pass1["stereo_quality"],
            tw_cfg,
        )
        final = retriangulate_sequence(p1, p2, rescued_left, rescued_right, rescued_conf_left, rescued_conf_right, tri_cfg)
        final["keypoints_left_rect_raw"] = pass1["keypoints_left_rect"]
        final["keypoints_right_rect_raw"] = pass1["keypoints_right_rect"]
        final["reprojection_error_pass1"] = pass1["reprojection_error"]
        final["stereo_quality_pass1"] = pass1["stereo_quality"]
        final["pair_confidence_pass1"] = pass1["pair_confidence"]

    out_path = run_dir / skt.get("output_npz", "skt_pose_optimized.npz")
    np.savez(
        out_path,
        timestamps=timestamps,
        keypoints=final["keypoints"],
        reprojection_error=final["reprojection_error"],
        epipolar_error=final["epipolar_error"],
        epipolar_error_pre=final["epipolar_error_pre"],
        disparity_px=final["disparity_px"],
        stereo_quality=final["stereo_quality"],
        pair_confidence=final["pair_confidence"],
        keypoints_left_2d_raw=np.asarray(left_2d_raw, dtype=np.float64),
        keypoints_right_2d_raw=np.asarray(right_2d_raw, dtype=np.float64),
        keypoints_left_2d=np.asarray(left_2d_raw, dtype=np.float64),
        keypoints_right_2d=np.asarray(right_2d_raw, dtype=np.float64),
        keypoints_left_rect=final["keypoints_left_rect"],
        keypoints_right_rect=final["keypoints_right_rect"],
        keypoints_left_rect_raw=final.get("keypoints_left_rect_raw", pass1["keypoints_left_rect"]),
        keypoints_right_rect_raw=final.get("keypoints_right_rect_raw", pass1["keypoints_right_rect"]),
        conf_left=conf_l_arr,
        conf_right=conf_r_arr,
        triang_conf_left=final["triang_conf_left"],
        triang_conf_right=final["triang_conf_right"],
        bbox_left=np.asarray(bbox_l_all, dtype=np.float64),
        bbox_right=np.asarray(bbox_r_all, dtype=np.float64),
        track_score_left=np.asarray(track_score_l, dtype=np.float64),
        track_score_right=np.asarray(track_score_r, dtype=np.float64),
        track_source_left=np.asarray(track_source_l),
        track_source_right=np.asarray(track_source_r),
        stereo_sanity_ok=np.asarray(sanity_ok_all, dtype=bool),
        stereo_sanity_reason=np.asarray(sanity_reason_all),
        epipolar_shift_left_px=final["epipolar_shift_left_px"],
        epipolar_shift_right_px=final["epipolar_shift_right_px"],
        temporal_rescue_left=rescue_mask_left,
        temporal_rescue_right=rescue_mask_right,
        reprojection_error_pass1=final.get("reprojection_error_pass1", pass1["reprojection_error"]),
        stereo_quality_pass1=final.get("stereo_quality_pass1", pass1["stereo_quality"]),
        pair_confidence_pass1=final.get("pair_confidence_pass1", pass1["pair_confidence"]),
        model_name=np.asarray(str(model_path.name)),
        postprocess_variant=np.asarray(_variant_name(tri_cfg, tw_cfg, tracking_cfg)),
        reprojection_threshold_px=np.asarray(tri_cfg.reprojection_max_px, dtype=np.float64),
    )
    print(f"[skt] saved {out_path}")
    return out_path
