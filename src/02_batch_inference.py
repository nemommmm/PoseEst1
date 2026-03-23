import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pose_postprocess import estimate_bone_priors, postprocess_sequence
from utils import StereoDataLoader


# ================= Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")
MODEL_NAME = os.environ.get("POSE_MODEL_NAME", "yolov8m-pose.pt")
MODEL_PATH = os.path.join(SRC_DIR, MODEL_NAME)
MODEL_SLUG = os.path.splitext(MODEL_NAME)[0].replace("-", "_")

FULL_FRAME_CONF = float(os.environ.get("POSE_FULL_FRAME_CONF", "0.35"))
CROP_CONF = float(os.environ.get("POSE_CROP_CONF", "0.20"))
CROP_EXPAND = float(os.environ.get("POSE_TRACK_CROP_EXPAND", "1.55"))
TRACK_MAX_MISSES = int(os.environ.get("POSE_TRACK_MAX_MISSES", "8"))
REFULL_INTERVAL = int(os.environ.get("POSE_TRACK_REFULL_INTERVAL", "30"))
MIN_CROP_ACCEPT_SCORE = float(os.environ.get("POSE_TRACK_MIN_CROP_SCORE", "0.62"))

MIN_PAIR_CONF = float(os.environ.get("POSE_MIN_PAIR_CONF", "0.25"))
MIN_DISPARITY_PX = float(os.environ.get("POSE_MIN_DISPARITY_PX", "1.5"))
EPIPOLAR_BASE_PX = float(os.environ.get("POSE_EPIPOLAR_BASE_PX", "6.0"))
EPIPOLAR_CONF_GAIN_PX = float(os.environ.get("POSE_EPIPOLAR_CONF_GAIN_PX", "18.0"))
REPROJECTION_BASE_PX = float(os.environ.get("POSE_REPROJECTION_BASE_PX", "18.0"))
REPROJECTION_CONF_GAIN_PX = float(os.environ.get("POSE_REPROJECTION_CONF_GAIN_PX", "42.0"))
REPROJECTION_MAX_PX = float(os.environ.get("POSE_REPROJECTION_MAX_PX", "80.0"))

ENABLE_BONE_CONSTRAINT = os.environ.get("POSE_ENABLE_BONE_CONSTRAINT", "1") != "0"
ENABLE_QUALITY_AWARE_BLEND = os.environ.get("POSE_ENABLE_QUALITY_AWARE_BLEND", "0") == "1"
ENABLE_ONE_EURO = os.environ.get("POSE_ENABLE_ONE_EURO", "1") == "1"
FLOOR_AXIS = None
FLOOR_VALUE = None
DISABLE_PROGRESS = os.environ.get("POSE_DISABLE_PROGRESS", "0") == "1"

TORSO_JOINTS = np.array([5, 6, 11, 12], dtype=np.int64)
UPPER_BODY_JOINTS = np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
# ===============================================


@dataclass
class DetectionCandidate:
    bbox: np.ndarray
    keypoints: np.ndarray
    conf: np.ndarray
    det_conf: float
    mean_conf: float
    torso_conf: float
    upper_conf: float
    area: float
    source: str


@dataclass
class TrackState:
    bbox: np.ndarray | None = None
    misses: int = 0
    last_score: float = float("nan")
    last_source: str = "none"


def resolve_postprocess_variant():
    if ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO and not ENABLE_QUALITY_AWARE_BLEND:
        return "tracked_crop_weighted_stereo_plus_rigid_one_euro"
    if ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO and ENABLE_QUALITY_AWARE_BLEND:
        return "tracked_crop_weighted_stereo_plus_rigid_quality_one_euro"
    if ENABLE_BONE_CONSTRAINT and not ENABLE_ONE_EURO:
        return "tracked_crop_weighted_stereo_plus_rigid_only"
    if not ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO:
        return "tracked_crop_weighted_stereo_plus_one_euro_only"
    return "tracked_crop_weighted_stereo_raw"


def nanmean_subset(values, indices):
    subset = values[indices]
    finite = subset[np.isfinite(subset)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def bbox_area(bbox):
    w = max(float(bbox[2] - bbox[0]), 0.0)
    h = max(float(bbox[3] - bbox[1]), 0.0)
    return w * h


def bbox_center(bbox):
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)


def compute_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(x2 - x1, 0.0)
    inter_h = max(y2 - y1, 0.0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    denom = bbox_area(box_a) + bbox_area(box_b) - inter
    if denom <= 1e-6:
        return 0.0
    return float(inter / denom)


def expand_bbox_to_crop(bbox, image_shape, expand_factor):
    h, w = image_shape[:2]
    cx, cy = bbox_center(bbox)
    bw = max(float(bbox[2] - bbox[0]), 32.0) * expand_factor
    bh = max(float(bbox[3] - bbox[1]), 32.0) * expand_factor
    x1 = int(np.clip(math.floor(cx - 0.5 * bw), 0, w - 1))
    y1 = int(np.clip(math.floor(cy - 0.5 * bh), 0, h - 1))
    x2 = int(np.clip(math.ceil(cx + 0.5 * bw), x1 + 1, w))
    y2 = int(np.clip(math.ceil(cy + 0.5 * bh), y1 + 1, h))
    return x1, y1, x2, y2


def extract_candidates(result, offset_xy=(0.0, 0.0), source="full"):
    if result.boxes is None or result.keypoints is None:
        return []
    if len(result.boxes) == 0 or len(result.keypoints) == 0:
        return []

    off_x, off_y = offset_xy
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    det_conf = result.boxes.conf.cpu().numpy().astype(np.float64)
    keypoints_xy = result.keypoints.xy.cpu().numpy().astype(np.float64)
    keypoints_conf = result.keypoints.conf.cpu().numpy().astype(np.float64)

    candidates = []
    for bbox, score, kpts_xy, kpts_conf in zip(boxes, det_conf, keypoints_xy, keypoints_conf):
        bbox = bbox.copy()
        bbox[[0, 2]] += off_x
        bbox[[1, 3]] += off_y
        kpts_xy = kpts_xy.copy()
        kpts_xy[:, 0] += off_x
        kpts_xy[:, 1] += off_y
        candidates.append(
            DetectionCandidate(
                bbox=bbox,
                keypoints=kpts_xy,
                conf=kpts_conf.astype(np.float64),
                det_conf=float(score),
                mean_conf=nanmean_subset(kpts_conf, np.arange(len(kpts_conf))),
                torso_conf=nanmean_subset(kpts_conf, TORSO_JOINTS),
                upper_conf=nanmean_subset(kpts_conf, UPPER_BODY_JOINTS),
                area=float(bbox_area(bbox)),
                source=source,
            )
        )
    return candidates


def score_candidate(candidate, prev_bbox, frame_shape):
    h, w = frame_shape[:2]
    frame_area = max(float(h * w), 1.0)
    area_score = min(candidate.area / frame_area, 1.0)
    base_score = (
        0.38 * candidate.torso_conf
        + 0.24 * candidate.upper_conf
        + 0.20 * candidate.mean_conf
        + 0.12 * candidate.det_conf
        + 0.06 * math.sqrt(area_score)
    )
    if prev_bbox is None:
        return float(base_score)

    diag = max(math.hypot(w, h), 1.0)
    prev_center = bbox_center(prev_bbox)
    curr_center = bbox_center(candidate.bbox)
    center_penalty = np.linalg.norm(curr_center - prev_center) / diag
    iou_score = compute_iou(candidate.bbox, prev_bbox)
    size_ratio = min(candidate.area, bbox_area(prev_bbox)) / max(candidate.area, bbox_area(prev_bbox), 1e-6)
    return float(base_score + 0.42 * iou_score + 0.12 * size_ratio - 0.15 * center_penalty)


def select_candidate(candidates, prev_bbox, frame_shape):
    if not candidates:
        return None, -np.inf
    scored = [(score_candidate(candidate, prev_bbox, frame_shape), candidate) for candidate in candidates]
    best_score, best_candidate = max(scored, key=lambda item: item[0])
    return best_candidate, float(best_score)


def infer_tracked_pose(model, frame, track_state, frame_idx):
    frame_shape = frame.shape
    attempts = []
    if (
        track_state.bbox is not None
        and track_state.misses <= TRACK_MAX_MISSES
        and frame_idx % REFULL_INTERVAL != 0
    ):
        x1, y1, x2, y2 = expand_bbox_to_crop(track_state.bbox, frame_shape, CROP_EXPAND)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            attempts.append(("crop", crop, (float(x1), float(y1)), CROP_CONF))
    attempts.append(("full", frame, (0.0, 0.0), FULL_FRAME_CONF))

    chosen = None
    chosen_score = -np.inf
    chosen_source = "none"
    for source, image, offset_xy, conf_th in attempts:
        result = model(image, verbose=False, conf=conf_th)[0]
        candidates = extract_candidates(result, offset_xy=offset_xy, source=source)
        candidate, candidate_score = select_candidate(candidates, track_state.bbox, frame_shape)
        if candidate is None:
            continue
        if source == "crop" and candidate_score >= MIN_CROP_ACCEPT_SCORE:
            chosen = candidate
            chosen_score = candidate_score
            chosen_source = source
            break
        if candidate_score > chosen_score:
            chosen = candidate
            chosen_score = candidate_score
            chosen_source = source

    if chosen is not None:
        track_state.bbox = chosen.bbox.copy()
        track_state.misses = 0
        track_state.last_score = chosen_score
        track_state.last_source = chosen_source
        return chosen, track_state

    track_state.misses += 1
    if track_state.misses > TRACK_MAX_MISSES:
        track_state.bbox = None
    track_state.last_score = float("nan")
    track_state.last_source = "none"
    return None, track_state


def rectify_points(points_xy, camera_matrix, dist_coeffs, rect_r, rect_p):
    rectified = np.full((len(points_xy), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points_xy).all(axis=1)
    if not np.any(valid):
        return rectified
    rectified_valid = cv2.undistortPoints(
        points_xy[valid].reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        R=rect_r,
        P=rect_p,
    )[:, 0, :]
    rectified[valid] = rectified_valid
    return rectified


def weighted_dlt_triangulate(P1, P2, pt1, pt2, w1, w2):
    w1 = math.sqrt(max(float(w1), 1e-4))
    w2 = math.sqrt(max(float(w2), 1e-4))
    A = np.vstack(
        [
            w1 * (pt1[0] * P1[2] - P1[0]),
            w1 * (pt1[1] * P1[2] - P1[1]),
            w2 * (pt2[0] * P2[2] - P2[0]),
            w2 * (pt2[1] * P2[2] - P2[1]),
        ]
    )
    _, _, vt = np.linalg.svd(A)
    homog = vt[-1]
    if abs(homog[3]) < 1e-8:
        return np.full(3, np.nan, dtype=np.float64)
    return homog[:3] / homog[3]


def project_point(P, pt3d):
    homog = np.append(pt3d, 1.0)
    proj = P @ homog
    if abs(proj[2]) < 1e-8:
        return np.full(2, np.nan, dtype=np.float64), float("nan")
    return proj[:2] / proj[2], float(proj[2])


def compute_joint_quality(rect_l, rect_r, conf_l, conf_r):
    pair_conf = np.sqrt(np.clip(conf_l, 0.0, 1.0) * np.clip(conf_r, 0.0, 1.0))
    epipolar_error = np.abs(rect_l[:, 1] - rect_r[:, 1])
    disparity = np.abs(rect_l[:, 0] - rect_r[:, 0])
    epipolar_scale = EPIPOLAR_BASE_PX + EPIPOLAR_CONF_GAIN_PX * (1.0 - pair_conf)
    stereo_quality = pair_conf * np.exp(-epipolar_error / np.maximum(epipolar_scale, 1e-6))
    valid = (
        np.isfinite(rect_l).all(axis=1)
        & np.isfinite(rect_r).all(axis=1)
        & np.isfinite(pair_conf)
        & (pair_conf >= MIN_PAIR_CONF)
        & np.isfinite(disparity)
        & (disparity >= MIN_DISPARITY_PX)
    )
    stereo_quality[~np.isfinite(stereo_quality)] = np.nan
    return pair_conf, epipolar_error, disparity, stereo_quality, valid


def triangulate_pose(P1, P2, rect_l, rect_r, conf_l, conf_r):
    num_joints = rect_l.shape[0]
    pose_3d = np.full((num_joints, 3), np.nan, dtype=np.float64)
    reprojection_error = np.full(num_joints, np.nan, dtype=np.float64)
    pair_conf, epipolar_error, disparity, stereo_quality, valid = compute_joint_quality(
        rect_l, rect_r, conf_l, conf_r
    )

    for joint_idx in np.where(valid)[0]:
        pt3d = weighted_dlt_triangulate(
            P1,
            P2,
            rect_l[joint_idx],
            rect_r[joint_idx],
            conf_l[joint_idx],
            conf_r[joint_idx],
        )
        if not np.isfinite(pt3d).all():
            continue

        proj_l, depth_l = project_point(P1, pt3d)
        proj_r, depth_r = project_point(P2, pt3d)
        if not np.isfinite(proj_l).all() or not np.isfinite(proj_r).all():
            continue
        if depth_l <= 0.0 or depth_r <= 0.0:
            continue

        err_l = np.linalg.norm(proj_l - rect_l[joint_idx])
        err_r = np.linalg.norm(proj_r - rect_r[joint_idx])
        mean_err = 0.5 * (err_l + err_r)
        reproj_threshold = min(
            REPROJECTION_MAX_PX,
            REPROJECTION_BASE_PX
            + REPROJECTION_CONF_GAIN_PX * (1.0 - pair_conf[joint_idx])
            + 0.35 * min(epipolar_error[joint_idx], 120.0),
        )
        if mean_err > reproj_threshold:
            continue

        pose_3d[joint_idx] = pt3d
        reprojection_error[joint_idx] = mean_err

    return pose_3d, reprojection_error, pair_conf, epipolar_error, disparity, stereo_quality


def estimate_synchronized_pair_count(left_data, right_data):
    ptr_l = 0
    ptr_r = 0
    matches = 0
    while ptr_l < len(left_data) and ptr_r < len(right_data):
        id_l = left_data[ptr_l]["id"]
        id_r = right_data[ptr_r]["id"]
        if id_l == id_r:
            matches += 1
            ptr_l += 1
            ptr_r += 1
        elif id_l < id_r:
            ptr_l += 1
        else:
            ptr_r += 1
    return matches


def empty_bbox():
    return np.full(4, np.nan, dtype=np.float64)


def main():
    print("[Info] Starting batch 3D inference with tracked crops and weighted stereo triangulation...")

    if not os.path.exists(PARAM_PATH):
        print(f"[Error] Calibration parameters not found: {PARAM_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] YOLO model not found: {MODEL_PATH}")
        return

    data = np.load(PARAM_PATH)
    mtx_l, dist_l = data["mtx_l"], data["dist_l"]
    mtx_r, dist_r = data["mtx_r"], data["dist_r"]
    R, T = data["R"], data["T"]

    print("[Info] Computing rectification matrices...")
    loader_test = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"),
        os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"),
        os.path.join(DATA_DIR, "1_video_right.txt"),
    )
    estimated_total_pairs = estimate_synchronized_pair_count(loader_test.left_data, loader_test.right_data)
    print(f"[Info] Estimated synchronized frame pairs: {estimated_total_pairs}")
    frame_l, _, _, _ = loader_test.get_next_pair()
    if frame_l is None:
        print("[Error] Failed to read video. Please check the paths.")
        return
    h, w = frame_l.shape[:2]
    loader_test.release()

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0)

    print("[Info] Loading YOLO model...")
    print(f"[Info] Using pose model: {os.path.basename(MODEL_PATH)}")
    model = YOLO(MODEL_PATH)

    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"),
        os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"),
        os.path.join(DATA_DIR, "1_video_right.txt"),
    )

    track_left = TrackState()
    track_right = TrackState()

    all_timestamps = []
    all_keypoints_3d = []
    all_reprojection_errors = []
    all_epipolar_errors = []
    all_disparity = []
    all_stereo_quality = []
    all_keypoints_left_2d = []
    all_keypoints_right_2d = []
    all_keypoints_left_rect = []
    all_keypoints_right_rect = []
    all_conf_left = []
    all_conf_right = []
    all_pair_conf = []
    all_bbox_left = []
    all_bbox_right = []
    all_source_left = []
    all_source_right = []

    pbar = tqdm(
        total=estimated_total_pairs if estimated_total_pairs > 0 else None,
        desc="Processing Frames",
        unit="frame",
        dynamic_ncols=True,
        mininterval=1.0,
        disable=DISABLE_PROGRESS,
    )
    start_time = time.perf_counter()
    frame_idx = 0

    while True:
        frame_l, frame_r, _, ts = loader.get_next_pair()
        if frame_l is None:
            break

        candidate_l, track_left = infer_tracked_pose(model, frame_l, track_left, frame_idx)
        candidate_r, track_right = infer_tracked_pose(model, frame_r, track_right, frame_idx)

        pts_l = np.full((17, 2), np.nan, dtype=np.float64)
        pts_r = np.full((17, 2), np.nan, dtype=np.float64)
        conf_l = np.full(17, np.nan, dtype=np.float64)
        conf_r = np.full(17, np.nan, dtype=np.float64)
        bbox_l = empty_bbox()
        bbox_r = empty_bbox()
        source_l = "none"
        source_r = "none"

        if candidate_l is not None:
            pts_l = candidate_l.keypoints
            conf_l = candidate_l.conf
            bbox_l = candidate_l.bbox
            source_l = candidate_l.source
        if candidate_r is not None:
            pts_r = candidate_r.keypoints
            conf_r = candidate_r.conf
            bbox_r = candidate_r.bbox
            source_r = candidate_r.source

        rect_l = rectify_points(pts_l, mtx_l, dist_l, R1, P1)
        rect_r = rectify_points(pts_r, mtx_r, dist_r, R2, P2)
        pose_3d, reproj_error, pair_conf, epipolar_error, disparity, stereo_quality = triangulate_pose(
            P1,
            P2,
            rect_l,
            rect_r,
            conf_l,
            conf_r,
        )

        all_timestamps.append(ts)
        all_keypoints_3d.append(pose_3d)
        all_reprojection_errors.append(reproj_error)
        all_epipolar_errors.append(epipolar_error)
        all_disparity.append(disparity)
        all_stereo_quality.append(stereo_quality)
        all_keypoints_left_2d.append(pts_l)
        all_keypoints_right_2d.append(pts_r)
        all_keypoints_left_rect.append(rect_l)
        all_keypoints_right_rect.append(rect_r)
        all_conf_left.append(conf_l)
        all_conf_right.append(conf_r)
        all_pair_conf.append(pair_conf)
        all_bbox_left.append(bbox_l)
        all_bbox_right.append(bbox_r)
        all_source_left.append(source_l)
        all_source_right.append(source_r)

        frame_idx += 1
        pbar.update(1)
        processed_frames = frame_idx
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        fps = processed_frames / elapsed
        valid_joints = int(np.isfinite(pose_3d).all(axis=1).sum())
        crop_hits = int(source_l == "crop") + int(source_r == "crop")
        eta_seconds = None
        if estimated_total_pairs > 0 and fps > 1e-6:
            eta_seconds = max(estimated_total_pairs - processed_frames, 0) / fps
        eta_text = "--:--"
        if eta_seconds is not None:
            eta_text = time.strftime("%M:%S", time.gmtime(int(eta_seconds)))
        pbar.set_postfix(valid_joints=valid_joints, crop_hits=crop_hits, fps=f"{fps:.2f}", eta=eta_text)

    loader.release()
    pbar.close()

    all_timestamps = np.asarray(all_timestamps, dtype=np.float64)
    all_keypoints_3d = np.asarray(all_keypoints_3d, dtype=np.float64)
    all_reprojection_errors = np.asarray(all_reprojection_errors, dtype=np.float64)
    all_epipolar_errors = np.asarray(all_epipolar_errors, dtype=np.float64)
    all_disparity = np.asarray(all_disparity, dtype=np.float64)
    all_stereo_quality = np.asarray(all_stereo_quality, dtype=np.float64)
    all_keypoints_left_2d = np.asarray(all_keypoints_left_2d, dtype=np.float64)
    all_keypoints_right_2d = np.asarray(all_keypoints_right_2d, dtype=np.float64)
    all_keypoints_left_rect = np.asarray(all_keypoints_left_rect, dtype=np.float64)
    all_keypoints_right_rect = np.asarray(all_keypoints_right_rect, dtype=np.float64)
    all_conf_left = np.asarray(all_conf_left, dtype=np.float64)
    all_conf_right = np.asarray(all_conf_right, dtype=np.float64)
    all_pair_conf = np.asarray(all_pair_conf, dtype=np.float64)
    all_bbox_left = np.asarray(all_bbox_left, dtype=np.float64)
    all_bbox_right = np.asarray(all_bbox_right, dtype=np.float64)
    all_source_left = np.asarray(all_source_left, dtype="<U8")
    all_source_right = np.asarray(all_source_right, dtype="<U8")

    with np.errstate(invalid="ignore"):
        pair_confidence = np.nanmean(np.stack([all_conf_left, all_conf_right], axis=0), axis=0)

    finite_reproj = all_reprojection_errors[np.isfinite(all_reprojection_errors)]
    finite_epi = all_epipolar_errors[np.isfinite(all_epipolar_errors)]
    finite_quality = all_stereo_quality[np.isfinite(all_stereo_quality)]
    if finite_epi.size > 0:
        print("[Info] Epipolar error summary (px):")
        print(f"       p50={np.percentile(finite_epi, 50):.2f}, p90={np.percentile(finite_epi, 90):.2f}, p95={np.percentile(finite_epi, 95):.2f}")
    if finite_reproj.size > 0:
        keep_ratio = np.mean(finite_reproj <= REPROJECTION_MAX_PX)
        print("[Info] Reprojection error summary (px):")
        print(f"       p50={np.percentile(finite_reproj, 50):.2f}, p90={np.percentile(finite_reproj, 90):.2f}, p95={np.percentile(finite_reproj, 95):.2f}")
        print(f"       Keep ratio @ {REPROJECTION_MAX_PX:.1f}px: {keep_ratio:.3f}")
    if finite_quality.size > 0:
        print(f"[Info] Mean stereo quality: {np.mean(finite_quality):.3f}")
    crop_ratio_l = float(np.mean(all_source_left == "crop")) if len(all_source_left) else 0.0
    crop_ratio_r = float(np.mean(all_source_right == "crop")) if len(all_source_right) else 0.0
    print(f"[Info] Crop usage ratio (L/R): {crop_ratio_l:.3f} / {crop_ratio_r:.3f}")

    optimized_keypoints = all_keypoints_3d.copy()
    priors = estimate_bone_priors(all_keypoints_3d, all_timestamps)
    print("[Info] Estimated bone priors (cm):")
    for bone_name, value in priors.items():
        print(f"       {bone_name}: {value:.2f}")

    if ENABLE_BONE_CONSTRAINT or ENABLE_ONE_EURO:
        optimized_keypoints = postprocess_sequence(
            all_keypoints_3d,
            all_timestamps,
            priors=priors,
            reprojection_errors=all_reprojection_errors,
            pair_confidence=pair_confidence,
            floor_axis=FLOOR_AXIS,
            floor_value=FLOOR_VALUE,
            enable_bone_constraint=ENABLE_BONE_CONSTRAINT,
            enable_quality_blend=ENABLE_QUALITY_AWARE_BLEND,
            enable_one_euro=ENABLE_ONE_EURO,
        )

    postprocess_variant = resolve_postprocess_variant()
    raw_save_file = os.path.join(OUTPUT_DIR, "yolo_3d_raw.npz")
    raw_model_save_file = os.path.join(OUTPUT_DIR, f"yolo_3d_raw_{MODEL_SLUG}.npz")
    np.savez(
        raw_save_file,
        timestamps=all_timestamps,
        keypoints=all_keypoints_3d,
        reprojection_error=all_reprojection_errors,
        epipolar_error=all_epipolar_errors,
        disparity_px=all_disparity,
        stereo_quality=all_stereo_quality,
        pair_confidence=all_pair_conf,
        keypoints_left_2d=all_keypoints_left_2d,
        keypoints_right_2d=all_keypoints_right_2d,
        keypoints_left_rect=all_keypoints_left_rect,
        keypoints_right_rect=all_keypoints_right_rect,
        conf_left=all_conf_left,
        conf_right=all_conf_right,
        bbox_left=all_bbox_left,
        bbox_right=all_bbox_right,
        source_left=all_source_left,
        source_right=all_source_right,
        model_name=np.array(MODEL_NAME),
        postprocess_variant=np.array("tracked_crop_weighted_stereo_raw"),
        reprojection_threshold_px=np.array(REPROJECTION_MAX_PX, dtype=np.float64),
    )
    np.savez(
        raw_model_save_file,
        timestamps=all_timestamps,
        keypoints=all_keypoints_3d,
        reprojection_error=all_reprojection_errors,
        epipolar_error=all_epipolar_errors,
        disparity_px=all_disparity,
        stereo_quality=all_stereo_quality,
        pair_confidence=all_pair_conf,
        keypoints_left_2d=all_keypoints_left_2d,
        keypoints_right_2d=all_keypoints_right_2d,
        keypoints_left_rect=all_keypoints_left_rect,
        keypoints_right_rect=all_keypoints_right_rect,
        conf_left=all_conf_left,
        conf_right=all_conf_right,
        bbox_left=all_bbox_left,
        bbox_right=all_bbox_right,
        source_left=all_source_left,
        source_right=all_source_right,
        model_name=np.array(MODEL_NAME),
        postprocess_variant=np.array("tracked_crop_weighted_stereo_raw"),
        reprojection_threshold_px=np.array(REPROJECTION_MAX_PX, dtype=np.float64),
    )

    optimized_save_file = os.path.join(OUTPUT_DIR, "yolo_3d_optimized.npz")
    optimized_model_save_file = os.path.join(OUTPUT_DIR, f"yolo_3d_optimized_{MODEL_SLUG}.npz")
    np.savez(
        optimized_save_file,
        timestamps=all_timestamps,
        keypoints=optimized_keypoints,
        reprojection_error=all_reprojection_errors,
        epipolar_error=all_epipolar_errors,
        disparity_px=all_disparity,
        stereo_quality=all_stereo_quality,
        pair_confidence=all_pair_conf,
        keypoints_left_2d=all_keypoints_left_2d,
        keypoints_right_2d=all_keypoints_right_2d,
        keypoints_left_rect=all_keypoints_left_rect,
        keypoints_right_rect=all_keypoints_right_rect,
        conf_left=all_conf_left,
        conf_right=all_conf_right,
        bbox_left=all_bbox_left,
        bbox_right=all_bbox_right,
        source_left=all_source_left,
        source_right=all_source_right,
        prior_names=np.array(list(priors.keys())),
        prior_values=np.array(list(priors.values()), dtype=np.float64),
        model_name=np.array(MODEL_NAME),
        postprocess_variant=np.array(postprocess_variant),
        reprojection_threshold_px=np.array(REPROJECTION_MAX_PX, dtype=np.float64),
    )
    np.savez(
        optimized_model_save_file,
        timestamps=all_timestamps,
        keypoints=optimized_keypoints,
        reprojection_error=all_reprojection_errors,
        epipolar_error=all_epipolar_errors,
        disparity_px=all_disparity,
        stereo_quality=all_stereo_quality,
        pair_confidence=all_pair_conf,
        keypoints_left_2d=all_keypoints_left_2d,
        keypoints_right_2d=all_keypoints_right_2d,
        keypoints_left_rect=all_keypoints_left_rect,
        keypoints_right_rect=all_keypoints_right_rect,
        conf_left=all_conf_left,
        conf_right=all_conf_right,
        bbox_left=all_bbox_left,
        bbox_right=all_bbox_right,
        source_left=all_source_left,
        source_right=all_source_right,
        prior_names=np.array(list(priors.keys())),
        prior_values=np.array(list(priors.values()), dtype=np.float64),
        model_name=np.array(MODEL_NAME),
        postprocess_variant=np.array(postprocess_variant),
        reprojection_threshold_px=np.array(REPROJECTION_MAX_PX, dtype=np.float64),
    )

    print(f"\n[Info] Raw data saved to {raw_save_file}")
    print(f"[Info] Model-specific raw data saved to {raw_model_save_file}")
    print(f"[Info] Optimized data saved to {optimized_save_file}")
    print(f"[Info] Model-specific optimized data saved to {optimized_model_save_file}")
    print(f"[Info] Total frames processed: {len(all_timestamps)}")


if __name__ == "__main__":
    main()
