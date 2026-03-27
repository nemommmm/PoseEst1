import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

USE_RTMPOSE = os.environ.get("POSE_USE_RTMPOSE", "0") == "1"
RTMPOSE_MODE = os.environ.get("POSE_RTMPOSE_MODE", "balanced")  # lightweight / balanced / performance

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pose_postprocess import OneEuroFilter, estimate_bone_priors, postprocess_sequence
from utils import StereoDataLoader


# ================= Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
OUTPUT_DIR = os.environ.get("POSE_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
if not os.path.isabs(OUTPUT_DIR):
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")
OUTPUT_TAG = os.environ.get("POSE_OUTPUT_TAG", "").strip()
MODEL_NAME = os.environ.get("POSE_MODEL_NAME", "yolov8n-pose.pt")
MODEL_PATH = os.path.join(SRC_DIR, MODEL_NAME)
MODEL_SLUG = f"rtmpose_{RTMPOSE_MODE}" if USE_RTMPOSE else os.path.splitext(MODEL_NAME)[0].replace("-", "_")

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
ENFORCE_EPIPOLAR_CONSTRAINT = os.environ.get("POSE_ENFORCE_EPIPOLAR_CONSTRAINT", "0") == "1"
EPIPOLAR_SOFT_THRESHOLD_PX = float(os.environ.get("POSE_EPIPOLAR_SOFT_THRESHOLD_PX", "10.0"))
EPIPOLAR_SOFT_MAX_STRENGTH = float(os.environ.get("POSE_EPIPOLAR_SOFT_MAX_STRENGTH", "0.80"))
EPIPOLAR_CORRECTION_DECAY_PX = float(os.environ.get("POSE_EPIPOLAR_CORRECTION_DECAY_PX", "3.0"))

ENABLE_TEMPORAL_WINDOW_TRIANGULATION = os.environ.get("POSE_ENABLE_TEMPORAL_WINDOW_TRIANGULATION", "0") == "1"
TEMPORAL_WINDOW_RADIUS = int(os.environ.get("POSE_TRI_WINDOW_RADIUS", "3"))
TEMPORAL_WINDOW_MIN_SUPPORT = int(os.environ.get("POSE_TRI_WINDOW_MIN_SUPPORT", "3"))
TEMPORAL_WINDOW_MIN_STEREO_QUALITY = float(os.environ.get("POSE_TRI_WINDOW_MIN_STEREO_QUALITY", "0.35"))
TEMPORAL_WINDOW_DECAY_SEC = float(os.environ.get("POSE_TRI_WINDOW_DECAY_SEC", "0.06"))
TEMPORAL_WINDOW_CONF_FLOOR = float(os.environ.get("POSE_TRI_WINDOW_CONF_FLOOR", "0.12"))
TEMPORAL_WINDOW_SUPPORT_CONF_SCALE = float(os.environ.get("POSE_TRI_WINDOW_SUPPORT_CONF_SCALE", "0.75"))

ENABLE_2D_TEMPORAL_SMOOTHING = os.environ.get("POSE_ENABLE_2D_TEMPORAL_SMOOTHING", "1") == "1"
TEMPORAL_2D_MIN_CUTOFF = float(os.environ.get("POSE_2D_MIN_CUTOFF", "1.4"))
TEMPORAL_2D_BETA = float(os.environ.get("POSE_2D_BETA", "0.05"))
TEMPORAL_2D_MOTION_SCALE_PX = float(os.environ.get("POSE_2D_MOTION_SCALE_PX", "12.0"))

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


def tagged_name(filename):
    if not OUTPUT_TAG:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{OUTPUT_TAG}{ext}"


def resolve_output_path(filename):
    return os.path.join(OUTPUT_DIR, tagged_name(filename))


def resolve_postprocess_variant():
    base = "tracked_crop_weighted_stereo"
    if ENABLE_2D_TEMPORAL_SMOOTHING:
        base += "_2d_temporal"
    if ENFORCE_EPIPOLAR_CONSTRAINT:
        base += "_soft_epipolar"
    if ENABLE_TEMPORAL_WINDOW_TRIANGULATION:
        base += "_window_retri"
    if ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO and not ENABLE_QUALITY_AWARE_BLEND:
        return f"{base}_plus_rigid_one_euro"
    if ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO and ENABLE_QUALITY_AWARE_BLEND:
        return f"{base}_plus_rigid_quality_one_euro"
    if ENABLE_BONE_CONSTRAINT and not ENABLE_ONE_EURO:
        return f"{base}_plus_rigid_only"
    if not ENABLE_BONE_CONSTRAINT and ENABLE_ONE_EURO:
        return f"{base}_plus_one_euro_only"
    return f"{base}_raw"


def resolve_raw_variant():
    base = "tracked_crop_weighted_stereo"
    if ENABLE_2D_TEMPORAL_SMOOTHING:
        base += "_2d_temporal"
    if ENFORCE_EPIPOLAR_CONSTRAINT:
        base += "_soft_epipolar"
    if ENABLE_TEMPORAL_WINDOW_TRIANGULATION:
        base += "_window_retri"
    return f"{base}_raw"


class TemporalKeypointSmoother:
    def __init__(self):
        self.filter = OneEuroFilter(
            shape=(17, 2),
            min_cutoff=TEMPORAL_2D_MIN_CUTOFF,
            beta=TEMPORAL_2D_BETA,
            d_cutoff=1.0,
        )

    def update(self, timestamp, points_xy, confidence):
        points_xy = np.asarray(points_xy, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        filtered = self.filter(timestamp, points_xy)
        smoothed = points_xy.copy()

        valid = (
            np.isfinite(points_xy).all(axis=1)
            & np.isfinite(filtered).all(axis=1)
            & np.isfinite(confidence)
        )
        if not np.any(valid):
            return smoothed

        conf = np.clip(confidence[valid], 0.0, 1.0)
        lag = np.linalg.norm(points_xy[valid] - filtered[valid], axis=1)
        motion = np.clip(lag / max(TEMPORAL_2D_MOTION_SCALE_PX, 1e-6), 0.0, 1.0)

        # Low-confidence joints lean more on the filtered trajectory, while
        # fast motion still trusts the current observation to avoid lag.
        raw_weight = 0.18 + 0.52 * conf + 0.22 * motion
        raw_weight = np.clip(raw_weight, 0.15, 0.95)
        smoothed[valid] = raw_weight[:, None] * points_xy[valid] + (1.0 - raw_weight)[:, None] * filtered[valid]
        return smoothed


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


def extract_candidates_rtmpose(keypoints_all, scores_all, offset_xy=(0.0, 0.0), source="full"):
    """Convert rtmlib Body output to DetectionCandidate list (same format as extract_candidates)."""
    off_x, off_y = offset_xy
    candidates = []
    for kpts_xy, kpts_conf in zip(keypoints_all, scores_all):
        kpts_xy = kpts_xy.copy().astype(np.float64)
        kpts_conf = kpts_conf.copy().astype(np.float64)
        kpts_xy[:, 0] += off_x
        kpts_xy[:, 1] += off_y
        valid_mask = np.isfinite(kpts_xy).all(axis=1) & (kpts_conf > 0)
        if not valid_mask.any():
            continue
        x1 = kpts_xy[valid_mask, 0].min() - 20.0
        y1 = kpts_xy[valid_mask, 1].min() - 20.0
        x2 = kpts_xy[valid_mask, 0].max() + 20.0
        y2 = kpts_xy[valid_mask, 1].max() + 20.0
        bbox = np.array([x1, y1, x2, y2], dtype=np.float64)
        candidates.append(
            DetectionCandidate(
                bbox=bbox,
                keypoints=kpts_xy,
                conf=kpts_conf,
                det_conf=float(np.nanmean(kpts_conf)),
                mean_conf=nanmean_subset(kpts_conf, np.arange(len(kpts_conf))),
                torso_conf=nanmean_subset(kpts_conf, TORSO_JOINTS),
                upper_conf=nanmean_subset(kpts_conf, UPPER_BODY_JOINTS),
                area=float(bbox_area(bbox)),
                source=source,
            )
        )
    return candidates


def infer_tracked_pose_rtmpose(rtm_body, frame, track_state, frame_idx):
    """RTMPose variant of infer_tracked_pose — uses rtmlib Body instead of YOLO."""
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
            attempts.append(("crop", crop, (float(x1), float(y1))))
    attempts.append(("full", frame, (0.0, 0.0)))

    chosen = None
    chosen_score = -np.inf
    chosen_source = "none"
    for source, image, offset_xy in attempts:
        keypoints_all, scores_all = rtm_body(image)
        if keypoints_all is None or len(keypoints_all) == 0:
            continue
        candidates = extract_candidates_rtmpose(keypoints_all, scores_all, offset_xy=offset_xy, source=source)
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


def enforce_epipolar_constraint(rect_l, rect_r, conf_l, conf_r):
    corrected_l = rect_l.copy()
    corrected_r = rect_r.copy()
    pre_error = np.full(rect_l.shape[0], np.nan, dtype=np.float64)
    post_error = np.full(rect_l.shape[0], np.nan, dtype=np.float64)
    shift_l = np.full(rect_l.shape[0], np.nan, dtype=np.float64)
    shift_r = np.full(rect_l.shape[0], np.nan, dtype=np.float64)
    eff_conf_l = np.clip(np.asarray(conf_l, dtype=np.float64), 0.0, 1.0)
    eff_conf_r = np.clip(np.asarray(conf_r, dtype=np.float64), 0.0, 1.0)

    for joint_idx in range(rect_l.shape[0]):
        left_pt = rect_l[joint_idx]
        right_pt = rect_r[joint_idx]
        if not (np.isfinite(left_pt).all() and np.isfinite(right_pt).all()):
            continue

        pre_error[joint_idx] = abs(float(left_pt[1] - right_pt[1]))
        post_error[joint_idx] = pre_error[joint_idx]
        shift_l[joint_idx] = 0.0
        shift_r[joint_idx] = 0.0
        if pre_error[joint_idx] > EPIPOLAR_SOFT_THRESHOLD_PX:
            continue

        wl = max(float(conf_l[joint_idx]) if np.isfinite(conf_l[joint_idx]) else 0.01, 0.01)
        wr = max(float(conf_r[joint_idx]) if np.isfinite(conf_r[joint_idx]) else 0.01, 0.01)
        merged_y = (wl * left_pt[1] + wr * right_pt[1]) / (wl + wr)
        correction_ratio = 1.0 - (pre_error[joint_idx] / max(EPIPOLAR_SOFT_THRESHOLD_PX, 1e-6))
        alpha = EPIPOLAR_SOFT_MAX_STRENGTH * max(correction_ratio, 0.0)
        corrected_l[joint_idx, 1] = left_pt[1] + alpha * (merged_y - left_pt[1])
        corrected_r[joint_idx, 1] = right_pt[1] + alpha * (merged_y - right_pt[1])
        shift_l[joint_idx] = abs(float(corrected_l[joint_idx, 1] - left_pt[1]))
        shift_r[joint_idx] = abs(float(corrected_r[joint_idx, 1] - right_pt[1]))
        post_error[joint_idx] = abs(float(corrected_l[joint_idx, 1] - corrected_r[joint_idx, 1]))

        if EPIPOLAR_CORRECTION_DECAY_PX > 0.0:
            eff_conf_l[joint_idx] *= math.exp(-shift_l[joint_idx] / EPIPOLAR_CORRECTION_DECAY_PX)
            eff_conf_r[joint_idx] *= math.exp(-shift_r[joint_idx] / EPIPOLAR_CORRECTION_DECAY_PX)

    return corrected_l, corrected_r, pre_error, post_error, shift_l, shift_r, eff_conf_l, eff_conf_r


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
    conf_l = np.clip(np.asarray(conf_l, dtype=np.float64), 0.0, 1.0)
    conf_r = np.clip(np.asarray(conf_r, dtype=np.float64), 0.0, 1.0)
    pair_conf = np.sqrt(conf_l * conf_r)
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


def temporal_window_point_estimate(points_seq, conf_seq, timestamps, frame_idx, joint_idx):
    start = max(0, frame_idx - TEMPORAL_WINDOW_RADIUS)
    end = min(len(points_seq), frame_idx + TEMPORAL_WINDOW_RADIUS + 1)
    support_idx = np.arange(start, end, dtype=np.int64)
    support_idx = support_idx[support_idx != frame_idx]
    if support_idx.size == 0:
        return np.full(2, np.nan, dtype=np.float64), float("nan")

    support_points = points_seq[support_idx, joint_idx]
    support_conf = conf_seq[support_idx, joint_idx]
    valid = np.isfinite(support_points).all(axis=1) & np.isfinite(support_conf) & (support_conf >= TEMPORAL_WINDOW_CONF_FLOOR)
    if np.count_nonzero(valid) < TEMPORAL_WINDOW_MIN_SUPPORT:
        return np.full(2, np.nan, dtype=np.float64), float("nan")

    support_idx = support_idx[valid]
    support_points = support_points[valid]
    support_conf = support_conf[valid]
    dt = np.abs(timestamps[support_idx] - timestamps[frame_idx])
    temporal_weight = np.exp(-dt / max(TEMPORAL_WINDOW_DECAY_SEC, 1e-6))
    weights = support_conf * temporal_weight
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 1e-6:
        return np.full(2, np.nan, dtype=np.float64), float("nan")

    estimate = np.sum(weights[:, None] * support_points, axis=0) / weight_sum
    support_strength = float(np.sum(weights * support_conf) / weight_sum)
    return estimate, support_strength


def temporal_window_rescue_rectified(
    rect_left,
    rect_right,
    conf_left,
    conf_right,
    timestamps,
    pose_3d,
    stereo_quality,
):
    rescued_left = rect_left.copy()
    rescued_right = rect_right.copy()
    rescued_conf_left = np.clip(conf_left.copy(), 0.0, 1.0)
    rescued_conf_right = np.clip(conf_right.copy(), 0.0, 1.0)
    rescue_mask_left = np.zeros(rect_left.shape[:2], dtype=bool)
    rescue_mask_right = np.zeros(rect_right.shape[:2], dtype=bool)

    num_frames, num_joints, _ = rect_left.shape
    for frame_idx in range(num_frames):
        for joint_idx in range(num_joints):
            joint_valid = np.isfinite(pose_3d[frame_idx, joint_idx]).all()
            joint_quality = stereo_quality[frame_idx, joint_idx] if np.isfinite(stereo_quality[frame_idx, joint_idx]) else 0.0
            if joint_valid and joint_quality >= TEMPORAL_WINDOW_MIN_STEREO_QUALITY:
                continue

            left_estimate, left_support = temporal_window_point_estimate(
                rect_left,
                conf_left,
                timestamps,
                frame_idx,
                joint_idx,
            )
            right_estimate, right_support = temporal_window_point_estimate(
                rect_right,
                conf_right,
                timestamps,
                frame_idx,
                joint_idx,
            )
            if not (np.isfinite(left_estimate).all() and np.isfinite(right_estimate).all()):
                continue

            blend = np.clip(joint_quality / max(TEMPORAL_WINDOW_MIN_STEREO_QUALITY, 1e-6), 0.0, 1.0)

            current_left = rect_left[frame_idx, joint_idx]
            if np.isfinite(current_left).all():
                rescued_left[frame_idx, joint_idx] = blend * current_left + (1.0 - blend) * left_estimate
            else:
                rescued_left[frame_idx, joint_idx] = left_estimate
            rescue_mask_left[frame_idx, joint_idx] = True

            current_right = rect_right[frame_idx, joint_idx]
            if np.isfinite(current_right).all():
                rescued_right[frame_idx, joint_idx] = blend * current_right + (1.0 - blend) * right_estimate
            else:
                rescued_right[frame_idx, joint_idx] = right_estimate
            rescue_mask_right[frame_idx, joint_idx] = True

            current_conf_left = conf_left[frame_idx, joint_idx] if np.isfinite(conf_left[frame_idx, joint_idx]) else 0.0
            current_conf_right = conf_right[frame_idx, joint_idx] if np.isfinite(conf_right[frame_idx, joint_idx]) else 0.0
            rescued_conf_left[frame_idx, joint_idx] = max(
                blend * current_conf_left,
                (1.0 - blend) * left_support * TEMPORAL_WINDOW_SUPPORT_CONF_SCALE,
            )
            rescued_conf_right[frame_idx, joint_idx] = max(
                blend * current_conf_right,
                (1.0 - blend) * right_support * TEMPORAL_WINDOW_SUPPORT_CONF_SCALE,
            )

    return rescued_left, rescued_right, rescued_conf_left, rescued_conf_right, rescue_mask_left, rescue_mask_right


def retriangulate_sequence(P1, P2, rect_left_seq, rect_right_seq, conf_left_seq, conf_right_seq):
    num_frames, num_joints, _ = rect_left_seq.shape
    keypoints_3d = np.full((num_frames, num_joints, 3), np.nan, dtype=np.float64)
    reprojection_error = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    pair_confidence = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    epipolar_error_pre = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    epipolar_error_post = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    disparity_px = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    stereo_quality = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    rect_left_final = rect_left_seq.copy()
    rect_right_final = rect_right_seq.copy()
    epipolar_shift_left = np.full((num_frames, num_joints), np.nan, dtype=np.float64)
    epipolar_shift_right = np.full((num_frames, num_joints), np.nan, dtype=np.float64)

    for frame_idx in range(num_frames):
        rect_l = rect_left_seq[frame_idx].copy()
        rect_r = rect_right_seq[frame_idx].copy()
        triang_conf_l = np.clip(conf_left_seq[frame_idx], 0.0, 1.0)
        triang_conf_r = np.clip(conf_right_seq[frame_idx], 0.0, 1.0)

        if ENFORCE_EPIPOLAR_CONSTRAINT:
            (
                rect_l,
                rect_r,
                epi_pre,
                epi_post,
                shift_l,
                shift_r,
                triang_conf_l,
                triang_conf_r,
            ) = enforce_epipolar_constraint(rect_l, rect_r, triang_conf_l, triang_conf_r)
        else:
            epi_pre = np.abs(rect_l[:, 1] - rect_r[:, 1])
            epi_post = epi_pre.copy()
            shift_l = np.zeros(num_joints, dtype=np.float64)
            shift_r = np.zeros(num_joints, dtype=np.float64)

        pose_3d, reproj_error, pair_conf, _, disparity, quality = triangulate_pose(
            P1,
            P2,
            rect_l,
            rect_r,
            triang_conf_l,
            triang_conf_r,
        )
        keypoints_3d[frame_idx] = pose_3d
        reprojection_error[frame_idx] = reproj_error
        pair_confidence[frame_idx] = pair_conf
        epipolar_error_pre[frame_idx] = epi_pre
        epipolar_error_post[frame_idx] = epi_post
        disparity_px[frame_idx] = disparity
        stereo_quality[frame_idx] = quality
        rect_left_final[frame_idx] = rect_l
        rect_right_final[frame_idx] = rect_r
        epipolar_shift_left[frame_idx] = shift_l
        epipolar_shift_right[frame_idx] = shift_r

    return {
        "keypoints": keypoints_3d,
        "reprojection_error": reprojection_error,
        "pair_confidence": pair_confidence,
        "epipolar_error_pre": epipolar_error_pre,
        "epipolar_error": epipolar_error_post,
        "disparity_px": disparity_px,
        "stereo_quality": stereo_quality,
        "keypoints_left_rect": rect_left_final,
        "keypoints_right_rect": rect_right_final,
        "epipolar_shift_left_px": epipolar_shift_left,
        "epipolar_shift_right_px": epipolar_shift_right,
    }


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
    print(
        "[Info] 2D temporal smoothing: "
        + (
            f"enabled (OneEuro, min_cutoff={TEMPORAL_2D_MIN_CUTOFF:.2f}, beta={TEMPORAL_2D_BETA:.2f})"
            if ENABLE_2D_TEMPORAL_SMOOTHING
            else "disabled"
        )
    )
    if ENFORCE_EPIPOLAR_CONSTRAINT:
        print(
            "[Info] Epipolar soft correction: "
            f"enabled (threshold={EPIPOLAR_SOFT_THRESHOLD_PX:.1f}px, max_strength={EPIPOLAR_SOFT_MAX_STRENGTH:.2f})"
        )
    if ENABLE_TEMPORAL_WINDOW_TRIANGULATION:
        print(
            "[Info] Temporal window triangulation rescue: "
            f"enabled (radius={TEMPORAL_WINDOW_RADIUS}, min_support={TEMPORAL_WINDOW_MIN_SUPPORT})"
        )
    if USE_RTMPOSE:
        from rtmlib import Body
        model = Body(mode=RTMPOSE_MODE, backend="onnxruntime", device="cpu")
        print(f"[Info] Using RTMPose (mode={RTMPOSE_MODE})")
    else:
        model = YOLO(MODEL_PATH)

    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"),
        os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"),
        os.path.join(DATA_DIR, "1_video_right.txt"),
    )

    track_left = TrackState()
    track_right = TrackState()
    smoother_left = TemporalKeypointSmoother() if ENABLE_2D_TEMPORAL_SMOOTHING else None
    smoother_right = TemporalKeypointSmoother() if ENABLE_2D_TEMPORAL_SMOOTHING else None

    all_timestamps = []
    all_keypoints_3d = []
    all_epipolar_errors_pre = []
    all_reprojection_errors = []
    all_epipolar_errors = []
    all_disparity = []
    all_stereo_quality = []
    all_keypoints_left_2d_raw = []
    all_keypoints_right_2d_raw = []
    all_keypoints_left_2d = []
    all_keypoints_right_2d = []
    all_keypoints_left_rect_raw = []
    all_keypoints_right_rect_raw = []
    all_keypoints_left_rect = []
    all_keypoints_right_rect = []
    all_conf_left = []
    all_conf_right = []
    all_pair_conf = []
    all_triang_conf_left = []
    all_triang_conf_right = []
    all_epipolar_shift_left = []
    all_epipolar_shift_right = []
    temporal_rescue_left = []
    temporal_rescue_right = []
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

        if USE_RTMPOSE:
            candidate_l, track_left = infer_tracked_pose_rtmpose(model, frame_l, track_left, frame_idx)
            candidate_r, track_right = infer_tracked_pose_rtmpose(model, frame_r, track_right, frame_idx)
        else:
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

        pts_l_raw = pts_l.copy()
        pts_r_raw = pts_r.copy()
        if ENABLE_2D_TEMPORAL_SMOOTHING:
            pts_l = smoother_left.update(ts, pts_l, conf_l)
            pts_r = smoother_right.update(ts, pts_r, conf_r)

        rect_l = rectify_points(pts_l, mtx_l, dist_l, R1, P1)
        rect_r = rectify_points(pts_r, mtx_r, dist_r, R2, P2)
        rect_l_raw = rect_l.copy()
        rect_r_raw = rect_r.copy()
        if ENFORCE_EPIPOLAR_CONSTRAINT:
            (
                rect_l,
                rect_r,
                epipolar_error_pre,
                epipolar_error_post,
                epipolar_shift_left,
                epipolar_shift_right,
                triang_conf_l,
                triang_conf_r,
            ) = enforce_epipolar_constraint(
                rect_l,
                rect_r,
                conf_l,
                conf_r,
            )
        else:
            epipolar_error_pre = np.abs(rect_l[:, 1] - rect_r[:, 1])
            epipolar_error_post = epipolar_error_pre.copy()
            epipolar_shift_left = np.zeros(17, dtype=np.float64)
            epipolar_shift_right = np.zeros(17, dtype=np.float64)
            triang_conf_l = np.clip(conf_l.copy(), 0.0, 1.0)
            triang_conf_r = np.clip(conf_r.copy(), 0.0, 1.0)
        pose_3d, reproj_error, pair_conf, epipolar_error, disparity, stereo_quality = triangulate_pose(
            P1,
            P2,
            rect_l,
            rect_r,
            triang_conf_l,
            triang_conf_r,
        )

        all_timestamps.append(ts)
        all_keypoints_3d.append(pose_3d)
        all_epipolar_errors_pre.append(epipolar_error_pre)
        all_reprojection_errors.append(reproj_error)
        all_epipolar_errors.append(epipolar_error_post if ENFORCE_EPIPOLAR_CONSTRAINT else epipolar_error)
        all_disparity.append(disparity)
        all_stereo_quality.append(stereo_quality)
        all_keypoints_left_2d_raw.append(pts_l_raw)
        all_keypoints_right_2d_raw.append(pts_r_raw)
        all_keypoints_left_2d.append(pts_l)
        all_keypoints_right_2d.append(pts_r)
        all_keypoints_left_rect_raw.append(rect_l_raw)
        all_keypoints_right_rect_raw.append(rect_r_raw)
        all_keypoints_left_rect.append(rect_l)
        all_keypoints_right_rect.append(rect_r)
        all_conf_left.append(conf_l)
        all_conf_right.append(conf_r)
        all_pair_conf.append(pair_conf)
        all_triang_conf_left.append(triang_conf_l)
        all_triang_conf_right.append(triang_conf_r)
        all_epipolar_shift_left.append(epipolar_shift_left)
        all_epipolar_shift_right.append(epipolar_shift_right)
        temporal_rescue_left.append(np.zeros(17, dtype=bool))
        temporal_rescue_right.append(np.zeros(17, dtype=bool))
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
    all_epipolar_errors_pre = np.asarray(all_epipolar_errors_pre, dtype=np.float64)
    all_reprojection_errors = np.asarray(all_reprojection_errors, dtype=np.float64)
    all_epipolar_errors = np.asarray(all_epipolar_errors, dtype=np.float64)
    all_disparity = np.asarray(all_disparity, dtype=np.float64)
    all_stereo_quality = np.asarray(all_stereo_quality, dtype=np.float64)
    all_keypoints_left_2d_raw = np.asarray(all_keypoints_left_2d_raw, dtype=np.float64)
    all_keypoints_right_2d_raw = np.asarray(all_keypoints_right_2d_raw, dtype=np.float64)
    all_keypoints_left_2d = np.asarray(all_keypoints_left_2d, dtype=np.float64)
    all_keypoints_right_2d = np.asarray(all_keypoints_right_2d, dtype=np.float64)
    all_keypoints_left_rect_raw = np.asarray(all_keypoints_left_rect_raw, dtype=np.float64)
    all_keypoints_right_rect_raw = np.asarray(all_keypoints_right_rect_raw, dtype=np.float64)
    all_keypoints_left_rect = np.asarray(all_keypoints_left_rect, dtype=np.float64)
    all_keypoints_right_rect = np.asarray(all_keypoints_right_rect, dtype=np.float64)
    all_conf_left = np.asarray(all_conf_left, dtype=np.float64)
    all_conf_right = np.asarray(all_conf_right, dtype=np.float64)
    all_pair_conf = np.asarray(all_pair_conf, dtype=np.float64)
    all_triang_conf_left = np.asarray(all_triang_conf_left, dtype=np.float64)
    all_triang_conf_right = np.asarray(all_triang_conf_right, dtype=np.float64)
    all_epipolar_shift_left = np.asarray(all_epipolar_shift_left, dtype=np.float64)
    all_epipolar_shift_right = np.asarray(all_epipolar_shift_right, dtype=np.float64)
    temporal_rescue_left = np.asarray(temporal_rescue_left, dtype=bool)
    temporal_rescue_right = np.asarray(temporal_rescue_right, dtype=bool)
    all_bbox_left = np.asarray(all_bbox_left, dtype=np.float64)
    all_bbox_right = np.asarray(all_bbox_right, dtype=np.float64)
    all_source_left = np.asarray(all_source_left, dtype="<U8")
    all_source_right = np.asarray(all_source_right, dtype="<U8")

    rect_left_for_triangulation = all_keypoints_left_rect_raw.copy()
    rect_right_for_triangulation = all_keypoints_right_rect_raw.copy()
    triang_conf_left = np.clip(all_conf_left.copy(), 0.0, 1.0)
    triang_conf_right = np.clip(all_conf_right.copy(), 0.0, 1.0)

    if ENABLE_TEMPORAL_WINDOW_TRIANGULATION:
        (
            rect_left_for_triangulation,
            rect_right_for_triangulation,
            triang_conf_left,
            triang_conf_right,
            temporal_rescue_left,
            temporal_rescue_right,
        ) = temporal_window_rescue_rectified(
            rect_left_for_triangulation,
            rect_right_for_triangulation,
            triang_conf_left,
            triang_conf_right,
            all_timestamps,
            all_keypoints_3d,
            all_stereo_quality,
        )

    retriangulated = retriangulate_sequence(
        P1,
        P2,
        rect_left_for_triangulation,
        rect_right_for_triangulation,
        triang_conf_left,
        triang_conf_right,
    )
    all_keypoints_3d = retriangulated["keypoints"]
    all_reprojection_errors = retriangulated["reprojection_error"]
    all_pair_conf = retriangulated["pair_confidence"]
    all_epipolar_errors_pre = retriangulated["epipolar_error_pre"]
    all_epipolar_errors = retriangulated["epipolar_error"]
    all_disparity = retriangulated["disparity_px"]
    all_stereo_quality = retriangulated["stereo_quality"]
    all_keypoints_left_rect = retriangulated["keypoints_left_rect"]
    all_keypoints_right_rect = retriangulated["keypoints_right_rect"]
    all_epipolar_shift_left = retriangulated["epipolar_shift_left_px"]
    all_epipolar_shift_right = retriangulated["epipolar_shift_right_px"]
    pair_confidence = all_pair_conf

    finite_reproj = all_reprojection_errors[np.isfinite(all_reprojection_errors)]
    finite_epi_pre = all_epipolar_errors_pre[np.isfinite(all_epipolar_errors_pre)]
    finite_epi_post = all_epipolar_errors[np.isfinite(all_epipolar_errors)]
    finite_quality = all_stereo_quality[np.isfinite(all_stereo_quality)]
    if finite_epi_pre.size > 0:
        print("[Info] Epipolar error before correction (px):")
        print(f"       p50={np.percentile(finite_epi_pre, 50):.2f}, p90={np.percentile(finite_epi_pre, 90):.2f}, p95={np.percentile(finite_epi_pre, 95):.2f}")
    if finite_epi_post.size > 0:
        print("[Info] Epipolar error after correction (px):")
        print(f"       p50={np.percentile(finite_epi_post, 50):.2f}, p90={np.percentile(finite_epi_post, 90):.2f}, p95={np.percentile(finite_epi_post, 95):.2f}")
    if finite_reproj.size > 0:
        keep_ratio = np.mean(finite_reproj <= REPROJECTION_MAX_PX)
        print("[Info] Reprojection error summary (px):")
        print(f"       p50={np.percentile(finite_reproj, 50):.2f}, p90={np.percentile(finite_reproj, 90):.2f}, p95={np.percentile(finite_reproj, 95):.2f}")
        print(f"       Keep ratio @ {REPROJECTION_MAX_PX:.1f}px: {keep_ratio:.3f}")
    if finite_quality.size > 0:
        print(f"[Info] Mean stereo quality: {np.mean(finite_quality):.3f}")
    if ENFORCE_EPIPOLAR_CONSTRAINT:
        finite_shift = np.concatenate(
            [
                all_epipolar_shift_left[np.isfinite(all_epipolar_shift_left)],
                all_epipolar_shift_right[np.isfinite(all_epipolar_shift_right)],
            ]
        )
        if finite_shift.size > 0:
            print(f"[Info] Mean epipolar correction shift (px): {np.mean(finite_shift):.2f}")
    if ENABLE_TEMPORAL_WINDOW_TRIANGULATION:
        temporal_rescue_ratio_l = float(np.mean(temporal_rescue_left)) if temporal_rescue_left.size else 0.0
        temporal_rescue_ratio_r = float(np.mean(temporal_rescue_right)) if temporal_rescue_right.size else 0.0
        print(f"[Info] Temporal window rescue ratio (L/R): {temporal_rescue_ratio_l:.3f} / {temporal_rescue_ratio_r:.3f}")
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
    raw_payload = dict(
        timestamps=all_timestamps,
        keypoints=all_keypoints_3d,
        reprojection_error=all_reprojection_errors,
        epipolar_error=all_epipolar_errors,
        epipolar_error_pre=all_epipolar_errors_pre,
        disparity_px=all_disparity,
        stereo_quality=all_stereo_quality,
        pair_confidence=all_pair_conf,
        keypoints_left_2d_raw=all_keypoints_left_2d_raw,
        keypoints_right_2d_raw=all_keypoints_right_2d_raw,
        keypoints_left_2d=all_keypoints_left_2d,
        keypoints_right_2d=all_keypoints_right_2d,
        keypoints_left_rect_raw=all_keypoints_left_rect_raw,
        keypoints_right_rect_raw=all_keypoints_right_rect_raw,
        keypoints_left_rect=all_keypoints_left_rect,
        keypoints_right_rect=all_keypoints_right_rect,
        conf_left=all_conf_left,
        conf_right=all_conf_right,
        triang_conf_left=triang_conf_left,
        triang_conf_right=triang_conf_right,
        epipolar_shift_left_px=all_epipolar_shift_left,
        epipolar_shift_right_px=all_epipolar_shift_right,
        temporal_rescue_left=temporal_rescue_left,
        temporal_rescue_right=temporal_rescue_right,
        bbox_left=all_bbox_left,
        bbox_right=all_bbox_right,
        source_left=all_source_left,
        source_right=all_source_right,
        model_name=np.array(MODEL_NAME),
        postprocess_variant=np.array(resolve_raw_variant()),
        reprojection_threshold_px=np.array(REPROJECTION_MAX_PX, dtype=np.float64),
    )
    raw_save_file = os.path.join(OUTPUT_DIR, "yolo_3d_raw.npz")
    raw_model_save_file = os.path.join(OUTPUT_DIR, f"yolo_3d_raw_{MODEL_SLUG}.npz")
    raw_tagged_save_file = resolve_output_path("yolo_3d_raw.npz")
    np.savez(raw_save_file, **raw_payload)
    np.savez(raw_model_save_file, **raw_payload)
    if raw_tagged_save_file not in {raw_save_file, raw_model_save_file}:
        np.savez(raw_tagged_save_file, **raw_payload)

    optimized_save_file = os.path.join(OUTPUT_DIR, "yolo_3d_optimized.npz")
    optimized_model_save_file = os.path.join(OUTPUT_DIR, f"yolo_3d_optimized_{MODEL_SLUG}.npz")
    optimized_tagged_save_file = resolve_output_path("yolo_3d_optimized.npz")
    optimized_payload = dict(
        timestamps=all_timestamps,
        keypoints=optimized_keypoints,
        reprojection_error=all_reprojection_errors,
        epipolar_error=all_epipolar_errors,
        epipolar_error_pre=all_epipolar_errors_pre,
        disparity_px=all_disparity,
        stereo_quality=all_stereo_quality,
        pair_confidence=all_pair_conf,
        keypoints_left_2d_raw=all_keypoints_left_2d_raw,
        keypoints_right_2d_raw=all_keypoints_right_2d_raw,
        keypoints_left_2d=all_keypoints_left_2d,
        keypoints_right_2d=all_keypoints_right_2d,
        keypoints_left_rect_raw=all_keypoints_left_rect_raw,
        keypoints_right_rect_raw=all_keypoints_right_rect_raw,
        keypoints_left_rect=all_keypoints_left_rect,
        keypoints_right_rect=all_keypoints_right_rect,
        conf_left=all_conf_left,
        conf_right=all_conf_right,
        triang_conf_left=triang_conf_left,
        triang_conf_right=triang_conf_right,
        epipolar_shift_left_px=all_epipolar_shift_left,
        epipolar_shift_right_px=all_epipolar_shift_right,
        temporal_rescue_left=temporal_rescue_left,
        temporal_rescue_right=temporal_rescue_right,
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

    np.savez(optimized_save_file, **optimized_payload)
    np.savez(optimized_model_save_file, **optimized_payload)
    if optimized_tagged_save_file not in {optimized_save_file, optimized_model_save_file}:
        np.savez(optimized_tagged_save_file, **optimized_payload)

    print(f"\n[Info] Raw data saved to {raw_save_file}")
    print(f"[Info] Model-specific raw data saved to {raw_model_save_file}")
    if raw_tagged_save_file not in {raw_save_file, raw_model_save_file}:
        print(f"[Info] Tagged raw data saved to {raw_tagged_save_file}")
    print(f"[Info] Optimized data saved to {optimized_save_file}")
    print(f"[Info] Model-specific optimized data saved to {optimized_model_save_file}")
    if optimized_tagged_save_file not in {optimized_save_file, optimized_model_save_file}:
        print(f"[Info] Tagged optimized data saved to {optimized_tagged_save_file}")
    print(f"[Info] Total frames processed: {len(all_timestamps)}")


if __name__ == "__main__":
    main()
