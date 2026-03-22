import math
import numpy as np

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


HINGE_ROM_DEG = {
    "left_elbow": (20.0, 175.0),
    "right_elbow": (20.0, 175.0),
    "left_knee": (25.0, 175.0),
    "right_knee": (25.0, 175.0),
}

IK_ANGLE_DEFINITIONS = {
    "LeftElbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "RightElbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "LeftKnee": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "RightKnee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    "LeftShoulder": (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
    "RightShoulder": (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
    "LeftHip": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "RightHip": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
}


def _is_valid_point(point):
    return point is not None and np.isfinite(point).all()


def _normalize(vec, eps=1e-8):
    if vec is None or not np.isfinite(vec).all():
        return None
    norm = np.linalg.norm(vec)
    if norm < eps:
        return None
    return vec / norm


def _arbitrary_orthogonal(unit_vec):
    basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(unit_vec, basis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ortho = basis - np.dot(basis, unit_vec) * unit_vec
    return _normalize(ortho)


def _project_orthogonal(vec, axis):
    axis = _normalize(axis)
    if axis is None:
        return None
    return vec - np.dot(vec, axis) * axis


def _interior_angle_deg(p1, p2, p3):
    if not (_is_valid_point(p1) and _is_valid_point(p2) and _is_valid_point(p3)):
        return np.nan
    v1 = _normalize(p1 - p2)
    v2 = _normalize(p3 - p2)
    if v1 is None or v2 is None:
        return np.nan
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _distance_for_angle(l1, l2, angle_deg):
    theta = math.radians(angle_deg)
    dist_sq = l1 * l1 + l2 * l2 - 2.0 * l1 * l2 * math.cos(theta)
    return math.sqrt(max(dist_sq, 0.0))


def _pole_from_pose(pose, root_idx, mid_idx, end_idx):
    if pose is None:
        return None
    root = pose[root_idx]
    mid = pose[mid_idx]
    end = pose[end_idx]
    if not (_is_valid_point(root) and _is_valid_point(mid) and _is_valid_point(end)):
        return None
    root_to_end = _normalize(end - root)
    if root_to_end is None:
        return None
    pole = _project_orthogonal(mid - root, root_to_end)
    return _normalize(pole)


def solve_two_bone_ik(root, target, pole_dir, upper_len, lower_len, min_angle_deg, max_angle_deg):
    if not (_is_valid_point(root) and _is_valid_point(target)):
        return None, None

    root_to_target = target - root
    direction = _normalize(root_to_target)
    if direction is None:
        return None, None

    min_reach = _distance_for_angle(upper_len, lower_len, min_angle_deg)
    max_reach = _distance_for_angle(upper_len, lower_len, max_angle_deg)
    target_dist = np.linalg.norm(root_to_target)
    clamped_dist = float(np.clip(target_dist, min_reach, max_reach))

    pole_dir = _project_orthogonal(
        pole_dir if pole_dir is not None else _arbitrary_orthogonal(direction),
        direction,
    )
    pole_dir = _normalize(pole_dir)
    if pole_dir is None:
        pole_dir = _arbitrary_orthogonal(direction)
    if pole_dir is None:
        return None, None

    x = (upper_len * upper_len - lower_len * lower_len + clamped_dist * clamped_dist) / max(2.0 * clamped_dist, 1e-8)
    height_sq = max(upper_len * upper_len - x * x, 0.0)
    height = math.sqrt(height_sq)

    mid = root + x * direction + height * pole_dir
    end = root + clamped_dist * direction
    return mid, end


def refine_pose_with_hinge_ik(pose, priors, prev_pose=None):
    refined = pose.copy()
    chain_defs = [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, priors["left_upper_arm"], priors["left_lower_arm"], HINGE_ROM_DEG["left_elbow"]),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, priors["right_upper_arm"], priors["right_lower_arm"], HINGE_ROM_DEG["right_elbow"]),
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, priors["left_thigh"], priors["left_shank"], HINGE_ROM_DEG["left_knee"]),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, priors["right_thigh"], priors["right_shank"], HINGE_ROM_DEG["right_knee"]),
    ]

    for root_idx, mid_idx, end_idx, upper_len, lower_len, rom_deg in chain_defs:
        root = refined[root_idx]
        end = refined[end_idx]
        if not (_is_valid_point(root) and _is_valid_point(end)):
            continue

        pole_dir = _pole_from_pose(refined, root_idx, mid_idx, end_idx)
        if pole_dir is None:
            pole_dir = _pole_from_pose(prev_pose, root_idx, mid_idx, end_idx)
        if pole_dir is None:
            current_mid = refined[mid_idx] if _is_valid_point(refined[mid_idx]) else None
            prev_mid = prev_pose[mid_idx] if prev_pose is not None and _is_valid_point(prev_pose[mid_idx]) else None
            candidate = current_mid if current_mid is not None else prev_mid
            if candidate is not None:
                pole_dir = _project_orthogonal(candidate - root, end - root)
                pole_dir = _normalize(pole_dir)
        mid, end_adjusted = solve_two_bone_ik(
            root,
            end,
            pole_dir,
            upper_len,
            lower_len,
            min_angle_deg=rom_deg[0],
            max_angle_deg=rom_deg[1],
        )
        if mid is None or end_adjusted is None:
            continue
        refined[mid_idx] = mid
        refined[end_idx] = end_adjusted

    return refined


def refine_sequence_with_hinge_ik(keypoints_3d, priors):
    refined = np.full_like(keypoints_3d, np.nan, dtype=np.float64)
    prev_pose = None
    for idx, pose in enumerate(keypoints_3d):
        refined_pose = refine_pose_with_hinge_ik(pose, priors, prev_pose=prev_pose)
        refined[idx] = refined_pose
        prev_pose = refined_pose
    return refined


def compute_joint_angles_for_pose(pose):
    angles = {}
    for name, (idx_a, idx_b, idx_c) in IK_ANGLE_DEFINITIONS.items():
        angles[name] = _interior_angle_deg(pose[idx_a], pose[idx_b], pose[idx_c])
    return angles


def compute_joint_angles_sequence(keypoints_3d):
    angle_names = list(IK_ANGLE_DEFINITIONS.keys())
    angle_values = np.full((len(keypoints_3d), len(angle_names)), np.nan, dtype=np.float64)
    for frame_idx, pose in enumerate(keypoints_3d):
        pose_angles = compute_joint_angles_for_pose(pose)
        for angle_idx, angle_name in enumerate(angle_names):
            angle_values[frame_idx, angle_idx] = pose_angles[angle_name]
    return angle_names, angle_values
