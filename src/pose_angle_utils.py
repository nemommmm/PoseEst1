import math

import numpy as np
from scipy.interpolate import interp1d

from pose_postprocess import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
)


SEMANTIC_ANGLE_VERSION = "semantic_pose_angles_v1"
DEFAULT_ANGLE_SMOOTH_RADIUS = 2

SEMANTIC_ANGLE_NAMES = (
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftHip",
    "RightHip",
    "LeftKnee",
    "RightKnee",
)

# These specs are dataset-validated against the current MVNX export.
# Xsens jointAngle axes are not uniformly interchangeable across joints, and the
# shoulder quantity that best matches ergonomic scoring is the upper-arm ergo
# angle rather than the anatomical shoulder joint angle.
GT_ANGLE_SPECS = {
    "LeftShoulder": {"source": "ergo", "label": "T8_LeftUpperArm", "axis": 2, "sign": 1.0},
    "RightShoulder": {"source": "ergo", "label": "T8_RightUpperArm", "axis": 2, "sign": 1.0},
    "LeftElbow": {"source": "joint", "label": "jLeftElbow", "axis": 2, "sign": 1.0},
    "RightElbow": {"source": "joint", "label": "jRightElbow", "axis": 2, "sign": 1.0},
    "LeftHip": {"source": "joint", "label": "jLeftHip", "axis": 2, "sign": 1.0},
    "RightHip": {"source": "joint", "label": "jRightHip", "axis": 2, "sign": 1.0},
    "LeftKnee": {"source": "joint", "label": "jLeftKnee", "axis": 2, "sign": 1.0},
    "RightKnee": {"source": "joint", "label": "jRightKnee", "axis": 2, "sign": 1.0},
}


def _normalize(vec, eps=1e-8):
    if vec is None or not np.isfinite(vec).all():
        return None
    norm = np.linalg.norm(vec)
    if norm < eps:
        return None
    return vec / norm


def angle_between_deg(vec_a, vec_b):
    unit_a = _normalize(vec_a)
    unit_b = _normalize(vec_b)
    if unit_a is None or unit_b is None:
        return np.nan
    cos_angle = np.clip(np.dot(unit_a, unit_b), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def interior_angle_deg(p1, p2, p3):
    if not (np.isfinite(p1).all() and np.isfinite(p2).all() and np.isfinite(p3).all()):
        return np.nan
    return angle_between_deg(p1 - p2, p3 - p2)


def compute_semantic_joint_angles(pose):
    """
    Compute ergonomically meaningful angles from a 3D pose.

    Angle semantics:
    - Shoulder: upper-arm elevation relative to the torso-down vector.
    - Elbow/Knee: hinge flexion (180 - interior angle).
    - Hip: flexion proxy using the ipsilateral shoulder-hip-knee chain. This is
      not a full anatomical hip decomposition, but it matches the current MVNX
      comparison target substantially better than a torso-midline definition.
    """
    angles = {name: np.nan for name in SEMANTIC_ANGLE_NAMES}

    hip_mid = 0.5 * (pose[LEFT_HIP] + pose[RIGHT_HIP])
    shoulder_mid = 0.5 * (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER])
    torso_down = hip_mid - shoulder_mid

    angles["LeftShoulder"] = angle_between_deg(
        pose[LEFT_ELBOW] - pose[LEFT_SHOULDER],
        torso_down,
    )
    angles["RightShoulder"] = angle_between_deg(
        pose[RIGHT_ELBOW] - pose[RIGHT_SHOULDER],
        torso_down,
    )

    left_elbow_interior = interior_angle_deg(pose[LEFT_SHOULDER], pose[LEFT_ELBOW], pose[9])
    right_elbow_interior = interior_angle_deg(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW], pose[10])
    left_knee_interior = interior_angle_deg(pose[LEFT_HIP], pose[LEFT_KNEE], pose[LEFT_ANKLE])
    right_knee_interior = interior_angle_deg(pose[RIGHT_HIP], pose[RIGHT_KNEE], pose[RIGHT_ANKLE])
    left_hip_interior = interior_angle_deg(pose[LEFT_SHOULDER], pose[LEFT_HIP], pose[LEFT_KNEE])
    right_hip_interior = interior_angle_deg(pose[RIGHT_SHOULDER], pose[RIGHT_HIP], pose[RIGHT_KNEE])

    if np.isfinite(left_elbow_interior):
        angles["LeftElbow"] = 180.0 - left_elbow_interior
    if np.isfinite(right_elbow_interior):
        angles["RightElbow"] = 180.0 - right_elbow_interior
    if np.isfinite(left_knee_interior):
        angles["LeftKnee"] = 180.0 - left_knee_interior
    if np.isfinite(right_knee_interior):
        angles["RightKnee"] = 180.0 - right_knee_interior
    if np.isfinite(left_hip_interior):
        angles["LeftHip"] = 180.0 - left_hip_interior
    if np.isfinite(right_hip_interior):
        angles["RightHip"] = 180.0 - right_hip_interior

    return angles


def compute_semantic_angle_sequence(keypoints_3d):
    angle_values = np.full((len(keypoints_3d), len(SEMANTIC_ANGLE_NAMES)), np.nan, dtype=np.float64)
    for frame_idx, pose in enumerate(keypoints_3d):
        pose_angles = compute_semantic_joint_angles(pose)
        for angle_idx, angle_name in enumerate(SEMANTIC_ANGLE_NAMES):
            angle_values[frame_idx, angle_idx] = pose_angles[angle_name]
    return list(SEMANTIC_ANGLE_NAMES), angle_values


def median_filter_angle_sequence(angle_values, radius=DEFAULT_ANGLE_SMOOTH_RADIUS):
    angle_values = np.asarray(angle_values, dtype=np.float64)
    if radius <= 0 or angle_values.ndim != 2 or len(angle_values) == 0:
        return angle_values.copy()

    filtered = np.full_like(angle_values, np.nan)
    num_frames, num_angles = angle_values.shape
    for frame_idx in range(num_frames):
        lo = max(0, frame_idx - radius)
        hi = min(num_frames, frame_idx + radius + 1)
        window = angle_values[lo:hi]
        for angle_idx in range(num_angles):
            finite_vals = window[np.isfinite(window[:, angle_idx]), angle_idx]
            if finite_vals.size > 0:
                filtered[frame_idx, angle_idx] = float(np.median(finite_vals))
    return filtered


def compute_aligned_trunk_flexion(pose, vertical_axis=None):
    if vertical_axis is None:
        vertical_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    hip_mid = 0.5 * (pose[LEFT_HIP] + pose[RIGHT_HIP])
    shoulder_mid = 0.5 * (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER])
    return angle_between_deg(shoulder_mid - hip_mid, vertical_axis)


def build_gt_angle_interpolators(mvnx, xsens_ts, xidx, specs=None):
    specs = GT_ANGLE_SPECS if specs is None else specs
    interpolators = {}
    for angle_name, spec in specs.items():
        source = spec["source"]
        if source == "joint":
            raw = mvnx.get_joint_angle_data(spec["label"])
        elif source == "ergo":
            raw = mvnx.get_ergo_angle_data(spec["label"])
        else:
            raise ValueError(f"Unsupported Xsens angle source: {source}")

        if raw is None:
            continue

        sign = float(spec.get("sign", 1.0))
        series = sign * raw[xidx, int(spec["axis"])]
        interpolators[angle_name] = interp1d(
            xsens_ts,
            series,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    return interpolators


def reduce_max_finite(values):
    finite_vals = [float(v) for v in values if np.isfinite(v)]
    if not finite_vals:
        return np.nan
    return float(max(finite_vals))
