import math
import os

import numpy as np
from scipy.interpolate import interp1d

from pose_postprocess import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


SHOULDER_REFERENCE_MODE = os.environ.get("POSE_SHOULDER_REFERENCE_MODE", "midline").strip().lower()
SEMANTIC_ANGLE_VERSION = f"semantic_pose_angles_v2_{SHOULDER_REFERENCE_MODE}"
DEFAULT_ANGLE_SMOOTH_RADIUS = max(0, int(os.environ.get("POSE_ANGLE_SMOOTH_RADIUS", "2")))
DEFAULT_WRIST_SMOOTH_RADIUS = max(0, int(os.environ.get("POSE_WRIST_SMOOTH_RADIUS", "0")))

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
    # Shoulders: use xz_mag (sqrt(x^2+z^2)) = total elevation in coronal+sagittal plane,
    # which matches our geometric angle_between(upper_arm, torso_down) better than axis=2 alone.
    "LeftShoulder": {"source": "ergo", "label": "T8_LeftUpperArm", "mode": "xz_mag"},
    "RightShoulder": {"source": "ergo", "label": "T8_RightUpperArm", "mode": "xz_mag"},
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
    - Shoulder: upper-arm elevation relative to the torso-down vector. The default
      reference is the shared torso midline; an ipsilateral shoulder-hip vector
      can be enabled for ablation via POSE_SHOULDER_REFERENCE_MODE=ipsilateral.
    - Elbow/Knee: hinge flexion (180 - interior angle).
    - Hip: flexion proxy using the ipsilateral shoulder-hip-knee chain. This is
      not a full anatomical hip decomposition, but it matches the current MVNX
      comparison target substantially better than a torso-midline definition.
    """
    angles = {name: np.nan for name in SEMANTIC_ANGLE_NAMES}

    if SHOULDER_REFERENCE_MODE == "ipsilateral":
        left_shoulder_ref = pose[LEFT_HIP] - pose[LEFT_SHOULDER]
        right_shoulder_ref = pose[RIGHT_HIP] - pose[RIGHT_SHOULDER]
    else:
        hip_mid = 0.5 * (pose[LEFT_HIP] + pose[RIGHT_HIP])
        shoulder_mid = 0.5 * (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER])
        torso_down = hip_mid - shoulder_mid
        left_shoulder_ref = torso_down
        right_shoulder_ref = torso_down

    angles["LeftShoulder"] = angle_between_deg(
        pose[LEFT_ELBOW] - pose[LEFT_SHOULDER],
        left_shoulder_ref,
    )
    angles["RightShoulder"] = angle_between_deg(
        pose[RIGHT_ELBOW] - pose[RIGHT_SHOULDER],
        right_shoulder_ref,
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


def median_smooth_keypoints(keypoints_3d, joint_indices, radius):
    """Apply per-axis median filter to specific joints across frames.

    keypoints_3d: (N, J, 3) array.
    joint_indices: list of joint indices to smooth.
    radius: temporal radius (window = 2*radius+1 frames).
    Returns a copy with smoothed joints.
    """
    if radius <= 0:
        return keypoints_3d
    kpts = np.array(keypoints_3d, dtype=np.float64)
    N = len(kpts)
    for j in joint_indices:
        for axis in range(3):
            col = kpts[:, j, axis]
            smoothed = col.copy()
            for i in range(N):
                lo = max(0, i - radius)
                hi = min(N, i + radius + 1)
                finite = col[lo:hi][np.isfinite(col[lo:hi])]
                if finite.size > 0:
                    smoothed[i] = float(np.median(finite))
            kpts[:, j, axis] = smoothed
    return kpts


def compute_semantic_angle_sequence(keypoints_3d, wrist_smooth_radius=DEFAULT_WRIST_SMOOTH_RADIUS):
    kpts = np.asarray(keypoints_3d, dtype=np.float64)
    if wrist_smooth_radius > 0:
        kpts = median_smooth_keypoints(kpts, [LEFT_WRIST, RIGHT_WRIST], wrist_smooth_radius)
    angle_values = np.full((len(kpts), len(SEMANTIC_ANGLE_NAMES)), np.nan, dtype=np.float64)
    for frame_idx, pose in enumerate(kpts):
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

        mode = spec.get("mode", "axis")
        if mode == "xz_mag":
            # Total elevation magnitude in coronal+sagittal plane: sqrt(x^2 + z^2).
            # Better match for absolute elevation angle (e.g. shoulder raise).
            series = np.sqrt(raw[xidx, 0] ** 2 + raw[xidx, 2] ** 2)
        else:
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


def fit_piecewise_calibration(est_values, gt_values, n_bins=10):
    """Fit a piecewise-linear calibration mapping est -> corrected.

    Returns (bin_centers, corrections) that can be used with np.interp.
    Only uses finite pairs for fitting.
    """
    mask = np.isfinite(est_values) & np.isfinite(gt_values)
    est_f = np.asarray(est_values)[mask]
    gt_f = np.asarray(gt_values)[mask]
    if len(est_f) < 20:
        return None

    bin_edges = np.percentile(est_f, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    centers = []
    corrections = []
    for b in range(n_bins):
        m = (est_f >= bin_edges[b]) & (est_f < bin_edges[b + 1])
        if m.sum() >= 5:
            centers.append(float(np.mean(est_f[m])))
            corrections.append(float(np.mean(gt_f[m] - est_f[m])))

    if len(centers) < 2:
        return None
    return np.array(centers), np.array(corrections)


def apply_piecewise_calibration(est_values, calibration):
    """Apply piecewise-linear calibration to estimated angle values.

    calibration: output of fit_piecewise_calibration, or None (no-op).
    Returns corrected values (same shape).
    """
    if calibration is None:
        return np.asarray(est_values, dtype=np.float64).copy()
    centers, corrections = calibration
    est = np.asarray(est_values, dtype=np.float64)
    out = est.copy()
    finite = np.isfinite(est)
    out[finite] = est[finite] + np.interp(est[finite], centers, corrections)
    return out


def save_calibration(path, calibrations):
    """Save per-joint calibration curves to an npz file.

    calibrations: dict of angle_name -> (centers, corrections) or None.
    """
    data = {}
    for name, cal in calibrations.items():
        if cal is not None:
            data[f"{name}__centers"] = cal[0]
            data[f"{name}__corrections"] = cal[1]
    np.savez(path, **data)


def load_calibration(path, angle_names):
    """Load per-joint calibration from npz file.

    Returns dict of angle_name -> (centers, corrections) or None.
    """
    d = np.load(path)
    result = {}
    for name in angle_names:
        c_key = f"{name}__centers"
        r_key = f"{name}__corrections"
        if c_key in d and r_key in d:
            result[name] = (d[c_key], d[r_key])
        else:
            result[name] = None
    return result


def reduce_max_finite(values):
    finite_vals = [float(v) for v in values if np.isfinite(v)]
    if not finite_vals:
        return np.nan
    return float(max(finite_vals))
