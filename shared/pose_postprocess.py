import math
import numpy as np

# COCO keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

DEFAULT_PRIOR_INIT_SECONDS = 10.0
MIN_PRIOR_FRAMES = 30
STABLE_SPEED_PERCENTILE = 40.0


DEFAULT_BONE_PRIORS_CM = {
    "hip_width": 23.0,
    "shoulder_width": 33.0,
    "torso_height": 54.0,
    "head_height": 26.0,
    "left_upper_arm": 28.0,
    "right_upper_arm": 28.0,
    "left_lower_arm": 26.0,
    "right_lower_arm": 26.0,
    "left_thigh": 39.5,
    "right_thigh": 39.5,
    "left_shank": 40.5,
    "right_shank": 40.5,
}


BONE_DEFS = {
    "hip_width": (LEFT_HIP, RIGHT_HIP),
    "shoulder_width": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "left_upper_arm": (LEFT_SHOULDER, LEFT_ELBOW),
    "right_upper_arm": (RIGHT_SHOULDER, RIGHT_ELBOW),
    "left_lower_arm": (LEFT_ELBOW, LEFT_WRIST),
    "right_lower_arm": (RIGHT_ELBOW, RIGHT_WRIST),
    "left_thigh": (LEFT_HIP, LEFT_KNEE),
    "right_thigh": (RIGHT_HIP, RIGHT_KNEE),
    "left_shank": (LEFT_KNEE, LEFT_ANKLE),
    "right_shank": (RIGHT_KNEE, RIGHT_ANKLE),
}

JOINT_REPROJECTION_THRESHOLDS_PX = {
    0: (25.0, 70.0),
    1: (20.0, 60.0),
    2: (20.0, 60.0),
    3: (25.0, 75.0),
    4: (25.0, 75.0),
    5: (25.0, 70.0),
    6: (25.0, 70.0),
    7: (35.0, 90.0),
    8: (35.0, 90.0),
    9: (40.0, 110.0),
    10: (40.0, 110.0),
    11: (25.0, 70.0),
    12: (25.0, 70.0),
    13: (30.0, 80.0),
    14: (30.0, 80.0),
    15: (35.0, 90.0),
    16: (35.0, 90.0),
}

CONFIDENCE_MIN = 0.35
CONFIDENCE_MAX = 0.85


def _is_valid_point(pt):
    return np.isfinite(pt).all()


def _normalize(vec, eps=1e-6):
    if not np.isfinite(vec).all():
        return None
    norm = np.linalg.norm(vec)
    if norm < eps:
        return None
    return vec / norm


def _pick_direction(curr_pose, idx_a, idx_b, prev_pose, default_dir=None):
    candidates = []
    if curr_pose is not None and _is_valid_point(curr_pose[idx_a]) and _is_valid_point(curr_pose[idx_b]):
        candidates.append(curr_pose[idx_b] - curr_pose[idx_a])
    if prev_pose is not None and _is_valid_point(prev_pose[idx_a]) and _is_valid_point(prev_pose[idx_b]):
        candidates.append(prev_pose[idx_b] - prev_pose[idx_a])
    for vec in candidates:
        direction = _normalize(vec)
        if direction is not None:
            return direction
    if default_dir is None:
        return None
    return np.array(default_dir, dtype=np.float64)


def _midpoint(pose, idx_a, idx_b):
    if pose is None:
        return None
    if not (_is_valid_point(pose[idx_a]) and _is_valid_point(pose[idx_b])):
        return None
    return 0.5 * (pose[idx_a] + pose[idx_b])


def _pick_midline_direction(curr_pose, curr_a, curr_b, prev_pose, prev_a, prev_b, default_dir=None):
    candidates = []
    curr_mid_a = _midpoint(curr_pose, *curr_a)
    curr_mid_b = _midpoint(curr_pose, *curr_b)
    prev_mid_a = _midpoint(prev_pose, *prev_a)
    prev_mid_b = _midpoint(prev_pose, *prev_b)

    if curr_mid_a is not None and curr_mid_b is not None:
        candidates.append(curr_mid_b - curr_mid_a)
    if prev_mid_a is not None and prev_mid_b is not None:
        candidates.append(prev_mid_b - prev_mid_a)

    for vec in candidates:
        direction = _normalize(vec)
        if direction is not None:
            return direction
    if default_dir is None:
        return None
    return np.array(default_dir, dtype=np.float64)


def _compute_distance_series(keypoints_3d, idx_a, idx_b):
    return np.linalg.norm(keypoints_3d[:, idx_a] - keypoints_3d[:, idx_b], axis=1)


def _compute_midpoint_distance_series(keypoints_3d, pair_a, pair_b):
    mid_a = 0.5 * (keypoints_3d[:, pair_a[0]] + keypoints_3d[:, pair_a[1]])
    mid_b = 0.5 * (keypoints_3d[:, pair_b[0]] + keypoints_3d[:, pair_b[1]])
    return np.linalg.norm(mid_a - mid_b, axis=1)


def _trimmed_median(values, trim_percentile):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    lo = np.percentile(values, trim_percentile)
    hi = np.percentile(values, 100.0 - trim_percentile)
    trimmed = values[(values >= lo) & (values <= hi)]
    if len(trimmed) == 0:
        return np.nan
    return float(np.median(trimmed))


def _select_prior_frame_mask(keypoints_3d, timestamps, init_seconds):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    rel_ts = timestamps - timestamps[0]

    pelvis_center = 0.5 * (keypoints_3d[:, LEFT_HIP] + keypoints_3d[:, RIGHT_HIP])
    valid_center = np.isfinite(pelvis_center).all(axis=1)
    candidate_mask = valid_center & (rel_ts <= init_seconds)
    if candidate_mask.sum() < MIN_PRIOR_FRAMES:
        candidate_mask = valid_center
    if candidate_mask.sum() < MIN_PRIOR_FRAMES:
        return np.ones(len(keypoints_3d), dtype=bool)

    candidate_indices = np.where(candidate_mask)[0]
    candidate_ts = rel_ts[candidate_indices]
    candidate_center = pelvis_center[candidate_indices]
    speed = np.linalg.norm(
        np.gradient(candidate_center, candidate_ts, axis=0, edge_order=1),
        axis=1,
    )
    finite_speed = speed[np.isfinite(speed)]
    if len(finite_speed) < MIN_PRIOR_FRAMES:
        return candidate_mask

    threshold = np.percentile(finite_speed, STABLE_SPEED_PERCENTILE)
    stable_local = np.isfinite(speed) & (speed <= threshold)
    stable_mask = np.zeros(len(keypoints_3d), dtype=bool)
    stable_mask[candidate_indices[stable_local]] = True

    if stable_mask.sum() < MIN_PRIOR_FRAMES:
        return candidate_mask
    return stable_mask


def estimate_bone_priors(keypoints_3d, timestamps=None, trim_percentile=10.0, init_seconds=DEFAULT_PRIOR_INIT_SECONDS):
    priors = DEFAULT_BONE_PRIORS_CM.copy()
    prior_mask = np.ones(len(keypoints_3d), dtype=bool)
    if timestamps is not None and len(keypoints_3d) >= MIN_PRIOR_FRAMES:
        prior_mask = _select_prior_frame_mask(keypoints_3d, timestamps, init_seconds)

    for bone_name, (idx_a, idx_b) in BONE_DEFS.items():
        dists = _compute_distance_series(keypoints_3d[prior_mask], idx_a, idx_b)
        prior_val = _trimmed_median(dists, trim_percentile)
        if not np.isfinite(prior_val):
            dists = _compute_distance_series(keypoints_3d, idx_a, idx_b)
            prior_val = _trimmed_median(dists, trim_percentile)
        if not np.isfinite(prior_val):
            continue
        priors[bone_name] = prior_val

    torso_series = _compute_midpoint_distance_series(
        keypoints_3d[prior_mask],
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_SHOULDER, RIGHT_SHOULDER),
    )
    torso_val = _trimmed_median(torso_series, trim_percentile)
    if not np.isfinite(torso_val):
        torso_series = _compute_midpoint_distance_series(
            keypoints_3d,
            (LEFT_HIP, RIGHT_HIP),
            (LEFT_SHOULDER, RIGHT_SHOULDER),
        )
        torso_val = _trimmed_median(torso_series, trim_percentile)
    if np.isfinite(torso_val):
        priors["torso_height"] = torso_val

    head_series = _compute_midpoint_distance_series(
        keypoints_3d[prior_mask],
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (0, 0),
    )
    head_val = _trimmed_median(head_series, trim_percentile)
    if not np.isfinite(head_val):
        head_series = _compute_midpoint_distance_series(
            keypoints_3d,
            (LEFT_SHOULDER, RIGHT_SHOULDER),
            (0, 0),
        )
        head_val = _trimmed_median(head_series, trim_percentile)
    if np.isfinite(head_val):
        priors["head_height"] = head_val

    symmetric_pairs = [
        ("left_upper_arm", "right_upper_arm"),
        ("left_lower_arm", "right_lower_arm"),
        ("left_thigh", "right_thigh"),
        ("left_shank", "right_shank"),
    ]
    for left_name, right_name in symmetric_pairs:
        mean_val = 0.5 * (priors[left_name] + priors[right_name])
        priors[left_name] = mean_val
        priors[right_name] = mean_val

    return priors


def apply_bone_length_constraints(pose, priors, prev_pose=None):
    corrected = pose.copy()

    # Keep the pelvis width stable, but preserve the observed pelvis center.
    hip_dir = _pick_direction(corrected, LEFT_HIP, RIGHT_HIP, prev_pose, (1.0, 0.0, 0.0))
    if hip_dir is not None:
        if _is_valid_point(corrected[LEFT_HIP]) and _is_valid_point(corrected[RIGHT_HIP]):
            hip_mid = 0.5 * (corrected[LEFT_HIP] + corrected[RIGHT_HIP])
            half_width = 0.5 * priors["hip_width"]
            corrected[LEFT_HIP] = hip_mid - half_width * hip_dir
            corrected[RIGHT_HIP] = hip_mid + half_width * hip_dir
        elif _is_valid_point(corrected[LEFT_HIP]):
            corrected[RIGHT_HIP] = corrected[LEFT_HIP] + priors["hip_width"] * hip_dir
        elif _is_valid_point(corrected[RIGHT_HIP]):
            corrected[LEFT_HIP] = corrected[RIGHT_HIP] - priors["hip_width"] * hip_dir

    # Keep the shoulder width stable, but do not pull the torso to a fixed height.
    shoulder_dir = _pick_direction(corrected, LEFT_SHOULDER, RIGHT_SHOULDER, prev_pose, (1.0, 0.0, 0.0))
    if shoulder_dir is not None:
        if _is_valid_point(corrected[LEFT_SHOULDER]) and _is_valid_point(corrected[RIGHT_SHOULDER]):
            shoulder_mid = 0.5 * (corrected[LEFT_SHOULDER] + corrected[RIGHT_SHOULDER])
            half_width = 0.5 * priors["shoulder_width"]
            corrected[LEFT_SHOULDER] = shoulder_mid - half_width * shoulder_dir
            corrected[RIGHT_SHOULDER] = shoulder_mid + half_width * shoulder_dir
        elif _is_valid_point(corrected[LEFT_SHOULDER]):
            corrected[RIGHT_SHOULDER] = corrected[LEFT_SHOULDER] + priors["shoulder_width"] * shoulder_dir
        elif _is_valid_point(corrected[RIGHT_SHOULDER]):
            corrected[LEFT_SHOULDER] = corrected[RIGHT_SHOULDER] - priors["shoulder_width"] * shoulder_dir

    chain_constraints = [
        (LEFT_HIP, LEFT_KNEE, "left_thigh", None),
        (RIGHT_HIP, RIGHT_KNEE, "right_thigh", None),
        (LEFT_KNEE, LEFT_ANKLE, "left_shank", None),
        (RIGHT_KNEE, RIGHT_ANKLE, "right_shank", None),
        (LEFT_SHOULDER, LEFT_ELBOW, "left_upper_arm", None),
        (RIGHT_SHOULDER, RIGHT_ELBOW, "right_upper_arm", None),
        (LEFT_ELBOW, LEFT_WRIST, "left_lower_arm", None),
        (RIGHT_ELBOW, RIGHT_WRIST, "right_lower_arm", None),
    ]
    for parent_idx, child_idx, bone_name, default_dir in chain_constraints:
        if not _is_valid_point(corrected[parent_idx]):
            continue
        direction = _pick_direction(corrected, parent_idx, child_idx, prev_pose, default_dir)
        if direction is None:
            continue
        corrected[child_idx] = corrected[parent_idx] + priors[bone_name] * direction

    return corrected


def _joint_trust_weight(joint_idx, reproj_err=None, confidence=None):
    trust = np.nan
    if reproj_err is not None and np.isfinite(reproj_err):
        low, high = JOINT_REPROJECTION_THRESHOLDS_PX.get(joint_idx, (30.0, 80.0))
        if reproj_err <= low:
            trust = 1.0
        elif reproj_err >= high:
            trust = 0.0
        else:
            trust = 1.0 - (reproj_err - low) / (high - low)
    if confidence is not None and np.isfinite(confidence):
        conf_trust = (confidence - CONFIDENCE_MIN) / max(CONFIDENCE_MAX - CONFIDENCE_MIN, 1e-6)
        conf_trust = float(np.clip(conf_trust, 0.0, 1.0))
        trust = conf_trust if not np.isfinite(trust) else min(trust, conf_trust)
    if not np.isfinite(trust):
        return 0.5
    return float(np.clip(trust, 0.0, 1.0))


def _blend_pose(raw_pose, constrained_pose, reproj_error=None, confidence=None):
    blended = raw_pose.copy()
    num_joints = raw_pose.shape[0]
    for joint_idx in range(num_joints):
        raw_valid = _is_valid_point(raw_pose[joint_idx])
        constrained_valid = _is_valid_point(constrained_pose[joint_idx])
        if not raw_valid and constrained_valid:
            blended[joint_idx] = constrained_pose[joint_idx]
            continue
        if raw_valid and not constrained_valid:
            blended[joint_idx] = raw_pose[joint_idx]
            continue
        if not raw_valid and not constrained_valid:
            continue

        joint_reproj = None if reproj_error is None else reproj_error[joint_idx]
        joint_conf = None if confidence is None else confidence[joint_idx]
        trust = _joint_trust_weight(joint_idx, joint_reproj, joint_conf)
        blended[joint_idx] = trust * raw_pose[joint_idx] + (1.0 - trust) * constrained_pose[joint_idx]
    return blended


def clamp_axis_to_floor(pose, axis, floor_value, joint_indices):
    corrected = pose.copy()
    if axis is None or floor_value is None:
        return corrected
    for joint_idx in joint_indices:
        if _is_valid_point(corrected[joint_idx]) and corrected[joint_idx, axis] < floor_value:
            corrected[joint_idx, axis] = floor_value
    return corrected


def _smoothing_factor(dt, cutoff):
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


def _exp_smoothing(alpha, value, prev_value):
    return alpha * value + (1.0 - alpha) * prev_value


class OneEuroFilter:
    def __init__(self, shape, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.shape = tuple(shape)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.prev_t = None
        self.prev_x = np.full(self.shape, np.nan, dtype=np.float64)
        self.prev_dx = np.zeros(self.shape, dtype=np.float64)

    def __call__(self, t, x):
        x = np.asarray(x, dtype=np.float64)
        if self.prev_t is None:
            self.prev_t = float(t)
            self.prev_x = x.copy()
            return x.copy()

        dt = max(float(t) - self.prev_t, 1e-3)
        self.prev_t = float(t)

        result = x.copy()
        valid_mask = np.isfinite(x) & np.isfinite(self.prev_x)
        new_mask = np.isfinite(x) & ~np.isfinite(self.prev_x)

        if np.any(valid_mask):
            alpha_d = _smoothing_factor(dt, self.d_cutoff)
            dx = (x[valid_mask] - self.prev_x[valid_mask]) / dt
            edx = _exp_smoothing(alpha_d, dx, self.prev_dx[valid_mask])
            cutoff = self.min_cutoff + self.beta * np.abs(edx)
            alpha = _smoothing_factor(dt, cutoff)
            result[valid_mask] = _exp_smoothing(alpha, x[valid_mask], self.prev_x[valid_mask])
            self.prev_dx[valid_mask] = edx

        if np.any(new_mask):
            self.prev_dx[new_mask] = 0.0

        self.prev_x[np.isfinite(result)] = result[np.isfinite(result)]
        return result


def postprocess_sequence(
    keypoints_3d,
    timestamps,
    priors,
    reprojection_errors=None,
    pair_confidence=None,
    floor_axis=None,
    floor_value=None,
    enable_bone_constraint=True,
    enable_quality_blend=False,
    enable_one_euro=True,
):
    filtered_keypoints = np.full_like(keypoints_3d, np.nan, dtype=np.float64)
    one_euro = OneEuroFilter(shape=keypoints_3d.shape[1:])
    prev_pose = None

    for idx, (pose, ts) in enumerate(zip(keypoints_3d, timestamps)):
        raw_pose = pose.copy()
        constrained = raw_pose.copy()
        if enable_bone_constraint:
            constrained = apply_bone_length_constraints(constrained, priors, prev_pose=prev_pose)
        blended = constrained
        if enable_quality_blend:
            blended = _blend_pose(
                raw_pose,
                constrained,
                reproj_error=None if reprojection_errors is None else reprojection_errors[idx],
                confidence=None if pair_confidence is None else pair_confidence[idx],
            )
        constrained = clamp_axis_to_floor(
            blended,
            axis=floor_axis,
            floor_value=floor_value,
            joint_indices=(LEFT_ANKLE, RIGHT_ANKLE),
        )
        filtered = constrained
        if enable_one_euro:
            filtered = one_euro(ts, constrained)
        filtered_keypoints[idx] = filtered
        prev_pose = filtered

    return filtered_keypoints
