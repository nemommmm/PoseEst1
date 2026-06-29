"""Person tracking helpers for stereo pose inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TORSO_JOINTS = np.array([5, 6, 11, 12], dtype=np.int64)
UPPER_BODY_JOINTS = np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)


@dataclass
class DetectionCandidate:
    """One detected person candidate from a 2D pose model."""

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
    """Lightweight single-person tracking state."""

    bbox: np.ndarray | None = None
    misses: int = 0
    last_score: float = float("nan")
    last_source: str = "none"


@dataclass
class TrackingConfig:
    """Configuration for crop-based single-person tracking."""

    enabled: bool = True
    full_frame_conf: float = 0.35
    crop_conf: float = 0.20
    crop_expand: float = 1.55
    max_misses: int = 8
    refull_interval: int = 30
    min_crop_accept_score: float = 0.62
    center_person_weight: float = 0.0
    target_x_fraction: float = 0.5

    @classmethod
    def from_skt_config(cls, skt_cfg: dict) -> "TrackingConfig":
        """Build tracking config from a pipeline ``skt`` section."""
        tracking = skt_cfg.get("tracking", {}) or {}
        return cls(
            enabled=bool(tracking.get("enabled", True)),
            full_frame_conf=float(tracking.get("full_frame_conf", skt_cfg.get("confidence_threshold", 0.35))),
            crop_conf=float(tracking.get("crop_conf", 0.20)),
            crop_expand=float(tracking.get("crop_expand", 1.55)),
            max_misses=int(tracking.get("max_misses", 8)),
            refull_interval=int(tracking.get("refull_interval", 30)),
            min_crop_accept_score=float(tracking.get("min_crop_accept_score", 0.62)),
            center_person_weight=float(tracking.get("center_person_weight", skt_cfg.get("center_person_weight", 0.0))),
            target_x_fraction=float(tracking.get("target_x_fraction", skt_cfg.get("target_x_fraction", 0.5))),
        )


@dataclass
class StereoSanityConfig:
    """Defensive checks that left/right detections likely describe one person."""

    enabled: bool = True
    max_bbox_top_y_diff_px: float = 30.0
    bbox_height_ratio_range: tuple[float, float] = (0.6, 1.67)
    max_bbox_center_y_diff_ratio: float = 0.20
    max_rectified_joint_y_median_px: float = 30.0

    @classmethod
    def from_skt_config(cls, skt_cfg: dict) -> "StereoSanityConfig":
        """Build stereo sanity config from a pipeline ``skt`` section."""
        raw = skt_cfg.get("stereo_sanity", {}) or {}
        ratio = raw.get("bbox_height_ratio_range", [0.6, 1.67])
        return cls(
            enabled=bool(raw.get("enabled", True)),
            max_bbox_top_y_diff_px=float(raw.get("max_bbox_top_y_diff_px", 30.0)),
            bbox_height_ratio_range=(float(ratio[0]), float(ratio[1])),
            max_bbox_center_y_diff_ratio=float(raw.get("max_bbox_center_y_diff_ratio", 0.20)),
            max_rectified_joint_y_median_px=float(raw.get("max_rectified_joint_y_median_px", 30.0)),
        )


def nanmean_subset(values: np.ndarray, indices: np.ndarray) -> float:
    """Return finite mean over selected indices, or 0 if none are finite."""
    subset = np.asarray(values, dtype=np.float64)[indices]
    finite = subset[np.isfinite(subset)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def bbox_area(bbox: np.ndarray) -> float:
    """Return area of an xyxy bounding box."""
    w = max(float(bbox[2] - bbox[0]), 0.0)
    h = max(float(bbox[3] - bbox[1]), 0.0)
    return w * h


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Return center point of an xyxy bounding box."""
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)


def compute_iou(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
    """Compute IoU between two xyxy boxes."""
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(x2 - x1, 0.0)
    inter_h = max(y2 - y1, 0.0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    denom = bbox_area(box_a) + bbox_area(box_b) - inter
    if denom <= 1e-6:
        return 0.0
    return float(inter / denom)


def expand_bbox_to_crop(bbox: np.ndarray, image_shape: tuple[int, ...], expand_factor: float) -> tuple[int, int, int, int]:
    """Expand a bbox into an image-clipped crop rectangle."""
    h, w = image_shape[:2]
    cx, cy = bbox_center(bbox)
    bw = max(float(bbox[2] - bbox[0]), 32.0) * float(expand_factor)
    bh = max(float(bbox[3] - bbox[1]), 32.0) * float(expand_factor)
    x1 = int(np.clip(math.floor(cx - 0.5 * bw), 0, w - 1))
    y1 = int(np.clip(math.floor(cy - 0.5 * bh), 0, h - 1))
    x2 = int(np.clip(math.ceil(cx + 0.5 * bw), x1 + 1, w))
    y2 = int(np.clip(math.ceil(cy + 0.5 * bh), y1 + 1, h))
    return x1, y1, x2, y2


def extract_candidates(result, offset_xy: tuple[float, float] = (0.0, 0.0), source: str = "full") -> list[DetectionCandidate]:
    """Extract detected people from one Ultralytics result."""
    if result.boxes is None or result.keypoints is None:
        return []
    if len(result.boxes) == 0 or len(result.keypoints) == 0:
        return []

    off_x, off_y = offset_xy
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    det_conf = result.boxes.conf.cpu().numpy().astype(np.float64)
    keypoints_xy = result.keypoints.xy.cpu().numpy().astype(np.float64)
    if result.keypoints.conf is None:
        keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float64)
    else:
        keypoints_conf = result.keypoints.conf.cpu().numpy().astype(np.float64)

    candidates: list[DetectionCandidate] = []
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


def score_candidate(candidate: DetectionCandidate, prev_bbox: np.ndarray | None, frame_shape: tuple[int, ...], cfg: TrackingConfig) -> float:
    """Score one person candidate using confidence, size, and track continuity."""
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
    if cfg.center_person_weight > 0.0 and w > 0:
        center_x = bbox_center(candidate.bbox)[0]
        target_x = w * float(np.clip(cfg.target_x_fraction, 0.0, 1.0))
        center_bonus = 1.0 - abs(center_x - target_x) / max(w * 0.5, 1.0)
        base_score += cfg.center_person_weight * max(center_bonus, 0.0)

    if prev_bbox is None:
        return float(base_score)

    diag = max(math.hypot(w, h), 1.0)
    prev_center = bbox_center(prev_bbox)
    curr_center = bbox_center(candidate.bbox)
    center_penalty = np.linalg.norm(curr_center - prev_center) / diag
    iou_score = compute_iou(candidate.bbox, prev_bbox)
    size_ratio = min(candidate.area, bbox_area(prev_bbox)) / max(candidate.area, bbox_area(prev_bbox), 1e-6)
    return float(base_score + 0.42 * iou_score + 0.12 * size_ratio - 0.15 * center_penalty)


def select_candidate(
    candidates: list[DetectionCandidate],
    prev_bbox: np.ndarray | None,
    frame_shape: tuple[int, ...],
    cfg: TrackingConfig,
) -> tuple[DetectionCandidate | None, float]:
    """Select the best tracked candidate from model detections."""
    if not candidates:
        return None, -np.inf
    scored = [(score_candidate(candidate, prev_bbox, frame_shape, cfg), candidate) for candidate in candidates]
    best_score, best_candidate = max(scored, key=lambda item: item[0])
    return best_candidate, float(best_score)


def infer_tracked_pose(model, frame: np.ndarray, track_state: TrackState, frame_idx: int, cfg: TrackingConfig) -> tuple[DetectionCandidate | None, TrackState]:
    """Infer a tracked single-person pose using crop-first fallback."""
    frame_shape = frame.shape
    attempts: list[tuple[str, np.ndarray, tuple[float, float], float]] = []
    if (
        cfg.enabled
        and track_state.bbox is not None
        and track_state.misses <= cfg.max_misses
        and frame_idx % max(cfg.refull_interval, 1) != 0
    ):
        x1, y1, x2, y2 = expand_bbox_to_crop(track_state.bbox, frame_shape, cfg.crop_expand)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            attempts.append(("crop", crop, (float(x1), float(y1)), cfg.crop_conf))
    attempts.append(("full", frame, (0.0, 0.0), cfg.full_frame_conf))

    chosen = None
    chosen_score = -np.inf
    chosen_source = "none"
    for source, image, offset_xy, conf_th in attempts:
        result = model(image, verbose=False, conf=conf_th)[0]
        candidates = extract_candidates(result, offset_xy=offset_xy, source=source)
        candidate, candidate_score = select_candidate(candidates, track_state.bbox, frame_shape, cfg)
        if candidate is None:
            continue
        if source == "crop" and candidate_score >= cfg.min_crop_accept_score:
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
    if track_state.misses > cfg.max_misses:
        track_state.bbox = None
    track_state.last_score = float("nan")
    track_state.last_source = "none"
    return None, track_state


def stereo_sanity_check(
    left_candidate: DetectionCandidate | None,
    right_candidate: DetectionCandidate | None,
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    cfg: StereoSanityConfig,
) -> tuple[bool, str]:
    """Check whether left/right detections are plausible stereo counterparts."""
    if not cfg.enabled:
        return True, "disabled"
    if left_candidate is None or right_candidate is None:
        return False, "missing_candidate"

    left_bbox = left_candidate.bbox
    right_bbox = right_candidate.bbox
    left_h = max(float(left_bbox[3] - left_bbox[1]), 1.0)
    right_h = max(float(right_bbox[3] - right_bbox[1]), 1.0)
    height_ratio = min(left_h, right_h) / max(left_h, right_h)
    min_ratio, max_ratio = cfg.bbox_height_ratio_range
    if height_ratio < min_ratio or height_ratio > max_ratio:
        return False, "bbox_height_ratio"

    if abs(float(left_bbox[1] - right_bbox[1])) > cfg.max_bbox_top_y_diff_px:
        return False, "bbox_top_y"

    left_cy = bbox_center(left_bbox)[1]
    right_cy = bbox_center(right_bbox)[1]
    max_center_diff = cfg.max_bbox_center_y_diff_ratio * max(left_h, right_h)
    if abs(float(left_cy - right_cy)) > max_center_diff:
        return False, "bbox_center_y"

    valid_rect = np.isfinite(rect_left).all(axis=1) & np.isfinite(rect_right).all(axis=1)
    if np.count_nonzero(valid_rect) >= 4:
        median_y_diff = float(np.nanmedian(np.abs(rect_left[valid_rect, 1] - rect_right[valid_rect, 1])))
        if median_y_diff > cfg.max_rectified_joint_y_median_px:
            return False, "rectified_joint_y"
    return True, "ok"
