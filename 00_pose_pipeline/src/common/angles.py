"""Geometric angle and Xsens-derived reference helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.interpolate import interp1d

from common.mvnx import MvnxParser

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

COCO17_NAMES = (
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
)

SEMANTIC_ANGLE_NAMES = (
    "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
    "LeftHip", "RightHip", "LeftKnee", "RightKnee",
)

GT_ANGLE_SPECS = {
    "LeftShoulder": {"source": "ergo", "label": "T8_LeftUpperArm", "mode": "xz_mag"},
    "RightShoulder": {"source": "ergo", "label": "T8_RightUpperArm", "mode": "xz_mag"},
    "LeftElbow": {"source": "joint", "label": "jLeftElbow", "axis": 2, "sign": 1.0},
    "RightElbow": {"source": "joint", "label": "jRightElbow", "axis": 2, "sign": 1.0},
    "LeftHip": {"source": "joint", "label": "jLeftHip", "axis": 2, "sign": 1.0},
    "RightHip": {"source": "joint", "label": "jRightHip", "axis": 2, "sign": 1.0},
    "LeftKnee": {"source": "joint", "label": "jLeftKnee", "axis": 2, "sign": 1.0},
    "RightKnee": {"source": "joint", "label": "jRightKnee", "axis": 2, "sign": 1.0},
}

XSENS_TO_COCO = {
    "LeftUpperArm": LEFT_SHOULDER,
    "RightUpperArm": RIGHT_SHOULDER,
    "LeftForeArm": LEFT_ELBOW,
    "RightForeArm": RIGHT_ELBOW,
    "LeftHand": LEFT_WRIST,
    "RightHand": RIGHT_WRIST,
    "LeftUpperLeg": LEFT_HIP,
    "RightUpperLeg": RIGHT_HIP,
    "LeftLowerLeg": LEFT_KNEE,
    "RightLowerLeg": RIGHT_KNEE,
    "LeftFoot": LEFT_ANKLE,
    "RightFoot": RIGHT_ANKLE,
}


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    if not np.isfinite(vec).all():
        return None
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def angle_between_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Angle between two vectors in degrees."""
    unit_a = _normalize(vec_a)
    unit_b = _normalize(vec_b)
    if unit_a is None or unit_b is None:
        return math.nan
    return float(math.degrees(math.acos(np.clip(float(np.dot(unit_a, unit_b)), -1.0, 1.0))))


def interior_angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Interior angle at p2."""
    if not (np.isfinite(p1).all() and np.isfinite(p2).all() and np.isfinite(p3).all()):
        return math.nan
    return angle_between_deg(p1 - p2, p3 - p2)


def compute_semantic_joint_angles(pose: np.ndarray) -> dict[str, float]:
    """Compute ergonomic-style geometric angles from one COCO-17 3D pose."""
    out = {name: math.nan for name in SEMANTIC_ANGLE_NAMES}
    hip_mid = 0.5 * (pose[LEFT_HIP] + pose[RIGHT_HIP])
    shoulder_mid = 0.5 * (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER])
    torso_down = hip_mid - shoulder_mid
    out["LeftShoulder"] = angle_between_deg(pose[LEFT_ELBOW] - pose[LEFT_SHOULDER], torso_down)
    out["RightShoulder"] = angle_between_deg(pose[RIGHT_ELBOW] - pose[RIGHT_SHOULDER], torso_down)

    elbow_l = interior_angle_deg(pose[LEFT_SHOULDER], pose[LEFT_ELBOW], pose[LEFT_WRIST])
    elbow_r = interior_angle_deg(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW], pose[RIGHT_WRIST])
    knee_l = interior_angle_deg(pose[LEFT_HIP], pose[LEFT_KNEE], pose[LEFT_ANKLE])
    knee_r = interior_angle_deg(pose[RIGHT_HIP], pose[RIGHT_KNEE], pose[RIGHT_ANKLE])
    hip_l = interior_angle_deg(pose[LEFT_SHOULDER], pose[LEFT_HIP], pose[LEFT_KNEE])
    hip_r = interior_angle_deg(pose[RIGHT_SHOULDER], pose[RIGHT_HIP], pose[RIGHT_KNEE])
    if np.isfinite(elbow_l):
        out["LeftElbow"] = 180.0 - elbow_l
    if np.isfinite(elbow_r):
        out["RightElbow"] = 180.0 - elbow_r
    if np.isfinite(knee_l):
        out["LeftKnee"] = 180.0 - knee_l
    if np.isfinite(knee_r):
        out["RightKnee"] = 180.0 - knee_r
    if np.isfinite(hip_l):
        out["LeftHip"] = 180.0 - hip_l
    if np.isfinite(hip_r):
        out["RightHip"] = 180.0 - hip_r
    return out


def compute_angle_sequence(keypoints: np.ndarray, angle_names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Compute selected angle time series from a COCO-17 keypoint sequence."""
    names = angle_names or list(SEMANTIC_ANGLE_NAMES)
    result = {name: np.full(len(keypoints), np.nan, dtype=np.float64) for name in names}
    for frame_idx, pose in enumerate(np.asarray(keypoints, dtype=np.float64)):
        angles = compute_semantic_joint_angles(pose)
        for name in names:
            result[name][frame_idx] = angles.get(name, math.nan)
    return result


def build_native_angle_interpolators(mvnx_path: Path) -> dict[str, interp1d]:
    """Build Xsens native angle interpolators."""
    mvnx = MvnxParser(mvnx_path)
    mvnx.parse()
    xsens_ts, unique_idx = np.unique(np.asarray(mvnx.timestamps, dtype=np.float64), return_index=True)
    xsens_ts = xsens_ts - xsens_ts[0]
    interps = {}
    for angle_name, spec in GT_ANGLE_SPECS.items():
        raw = mvnx.get_joint_angle_data(spec["label"]) if spec["source"] == "joint" else mvnx.get_ergo_angle_data(spec["label"])
        if raw is None:
            continue
        if spec.get("mode") == "xz_mag":
            values = np.sqrt(raw[unique_idx, 0] ** 2 + raw[unique_idx, 2] ** 2)
        else:
            values = float(spec.get("sign", 1.0)) * raw[unique_idx, int(spec["axis"])]
        interps[angle_name] = interp1d(xsens_ts, values, kind="linear", bounds_error=False, fill_value=np.nan)
    return interps


def build_fair_angle_interpolators(fair_npz_path: Path | None) -> dict[str, interp1d]:
    """Load precomputed Xsens-derived geometric angle reference interpolators."""
    if fair_npz_path is None or not fair_npz_path.exists():
        return {}
    data = np.load(fair_npz_path)
    ts = np.asarray(data["timestamps"], dtype=np.float64)
    interps = {}
    for name in SEMANTIC_ANGLE_NAMES:
        if name in data:
            interps[name] = interp1d(ts, np.asarray(data[name], dtype=np.float64), kind="linear", bounds_error=False, fill_value=np.nan)
    return interps


def build_xsens_coco_keypoints(mvnx_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Map Xsens segment origins into pseudo-COCO keypoints in centimeters."""
    mvnx = MvnxParser(mvnx_path)
    mvnx.parse()
    xsens_ts, unique_idx = np.unique(np.asarray(mvnx.timestamps, dtype=np.float64), return_index=True)
    xsens_ts = xsens_ts - xsens_ts[0]
    poses = np.full((len(mvnx.timestamps), len(COCO17_NAMES), 3), np.nan, dtype=np.float64)
    for segment_name, coco_idx in XSENS_TO_COCO.items():
        segment = mvnx.get_segment_data(segment_name)
        if segment is not None:
            poses[:, coco_idx, :] = segment
    return xsens_ts, poses[unique_idx]


def sample_interpolators(interps: Dict[str, interp1d], query_time: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Sample named interpolators onto a query timeline."""
    return {
        name: np.asarray(interps[name](query_time), dtype=np.float64) if name in interps else np.full(len(query_time), np.nan)
        for name in names
    }


def moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    """Centered moving average with NaN support."""
    values = np.asarray(values, dtype=np.float64)
    if radius <= 0 or values.size == 0:
        return values.copy()
    window = 2 * int(radius) + 1
    kernel = np.ones(window, dtype=np.float64)
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(np.float64), kernel, mode="same")
    out = np.full_like(values, np.nan)
    mask = denominator > 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def fill_short_gaps(values: np.ndarray, time_s: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill finite-bounded NaN gaps up to max_gap_frames."""
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    flags = np.zeros(len(values), dtype=bool)
    if max_gap_frames <= 0 or len(values) == 0:
        return out, flags
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return out, flags
    idx = 0
    while idx < len(values):
        if finite[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(values) and not finite[idx]:
            idx += 1
        end = idx
        if start > 0 and end < len(values) and (end - start) <= max_gap_frames:
            out[start:end] = np.interp(time_s[start:end], [time_s[start - 1], time_s[end]], [values[start - 1], values[end]])
            flags[start:end] = True
    return out, flags


def odd_window_from_ms(time_s: np.ndarray, window_ms: float) -> tuple[int, int, float]:
    """Convert a millisecond window to odd frame count and radius."""
    diffs = np.diff(np.asarray(time_s, dtype=np.float64))
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    if finite.size == 0 or window_ms <= 0:
        return 1, 0, 0.0
    median_dt_ms = float(np.nanmedian(finite) * 1000.0)
    frames = max(1, int(round(window_ms / median_dt_ms)))
    if frames % 2 == 0:
        frames = frames - 1 if frames > 1 else 1
    return frames, (frames - 1) // 2, frames * median_dt_ms
