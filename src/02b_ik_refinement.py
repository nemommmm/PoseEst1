"""
02b_ik_refinement.py

Post-processing step that refines the 3D keypoints from batch inference
using Inverse Kinematics constraints and extracts joint angle time-series
for downstream evaluation and ergonomic scoring.

Input:  results/yolo_3d_optimized.npz  (from 02_batch_inference.py)
Output: results/yolo_3d_ik_refined.npz (refined 3D + angle sequences)
"""
import math
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pose_postprocess import estimate_bone_priors, LEFT_SHOULDER, RIGHT_SHOULDER, \
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, \
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE

# ================= Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Hinge joint Range of Motion (degrees)
HINGE_ROM_DEG = {
    "left_elbow":  (20.0, 175.0),
    "right_elbow": (20.0, 175.0),
    "left_knee":   (5.0, 175.0),
    "right_knee":  (5.0, 175.0),
}

# Joint angle definitions: name -> (proximal_idx, joint_idx, distal_idx)
# These compute the interior angle at the joint_idx vertex.
ANGLE_DEFINITIONS = {
    "LeftElbow":     (LEFT_SHOULDER,  LEFT_ELBOW,   LEFT_WRIST),
    "RightElbow":    (RIGHT_SHOULDER, RIGHT_ELBOW,  RIGHT_WRIST),
    "LeftKnee":      (LEFT_HIP,      LEFT_KNEE,    LEFT_ANKLE),
    "RightKnee":     (RIGHT_HIP,     RIGHT_KNEE,   RIGHT_ANKLE),
    "LeftShoulder":  (LEFT_HIP,      LEFT_SHOULDER, LEFT_ELBOW),
    "RightShoulder": (RIGHT_HIP,     RIGHT_SHOULDER, RIGHT_ELBOW),
    "LeftHip":       (LEFT_SHOULDER,  LEFT_HIP,     LEFT_KNEE),
    "RightHip":      (RIGHT_SHOULDER, RIGHT_HIP,    RIGHT_KNEE),
}

# Trunk flexion: angle between shoulder-midpoint, hip-midpoint, and a vertical vector
# This is computed separately since it doesn't map to a single COCO triplet.
# ================================================


def _normalize(vec, eps=1e-8):
    norm = np.linalg.norm(vec)
    if norm < eps or not np.isfinite(norm):
        return None
    return vec / norm


def _interior_angle_deg(p1, p2, p3):
    """Compute interior angle at p2 in degrees."""
    if not (np.isfinite(p1).all() and np.isfinite(p2).all() and np.isfinite(p3).all()):
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


def _project_orthogonal(vec, axis):
    axis_n = _normalize(axis)
    if axis_n is None:
        return None
    return vec - np.dot(vec, axis_n) * axis_n


def _arbitrary_orthogonal(unit_vec):
    basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(unit_vec, basis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ortho = basis - np.dot(basis, unit_vec) * unit_vec
    return _normalize(ortho)


def solve_two_bone_ik(root, target, pole_dir, upper_len, lower_len, min_angle_deg, max_angle_deg):
    """
    Solve a two-bone IK chain. Returns (mid_position, end_position) or (None, None).
    The chain goes: root → mid → end, targeting end at `target`.
    The plane of the chain is defined by `pole_dir`.
    """
    if not (np.isfinite(root).all() and np.isfinite(target).all()):
        return None, None

    root_to_target = target - root
    direction = _normalize(root_to_target)
    if direction is None:
        return None, None

    min_reach = _distance_for_angle(upper_len, lower_len, min_angle_deg)
    max_reach = _distance_for_angle(upper_len, lower_len, max_angle_deg)
    target_dist = np.linalg.norm(root_to_target)
    clamped_dist = float(np.clip(target_dist, min_reach, max_reach))

    # Resolve pole direction
    if pole_dir is not None:
        pole_dir = _project_orthogonal(pole_dir, direction)
        pole_dir = _normalize(pole_dir)
    if pole_dir is None:
        pole_dir = _arbitrary_orthogonal(direction)
    if pole_dir is None:
        return None, None

    x = (upper_len**2 - lower_len**2 + clamped_dist**2) / max(2.0 * clamped_dist, 1e-8)
    height_sq = max(upper_len**2 - x**2, 0.0)
    height = math.sqrt(height_sq)

    mid = root + x * direction + height * pole_dir
    end = root + clamped_dist * direction
    return mid, end


def _pole_from_pose(pose, root_idx, mid_idx, end_idx):
    """Extract a pole vector from the current pose configuration."""
    root, mid, end = pose[root_idx], pose[mid_idx], pose[end_idx]
    if not (np.isfinite(root).all() and np.isfinite(mid).all() and np.isfinite(end).all()):
        return None
    root_to_end = _normalize(end - root)
    if root_to_end is None:
        return None
    pole = _project_orthogonal(mid - root, root_to_end)
    return _normalize(pole)


def refine_pose_with_ik(pose, priors, prev_pose=None):
    """Apply IK constraints to a single frame."""
    refined = pose.copy()
    chain_defs = [
        (LEFT_SHOULDER, LEFT_ELBOW,  LEFT_WRIST,  priors["left_upper_arm"],  priors["left_lower_arm"],  HINGE_ROM_DEG["left_elbow"]),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, priors["right_upper_arm"], priors["right_lower_arm"], HINGE_ROM_DEG["right_elbow"]),
        (LEFT_HIP,     LEFT_KNEE,   LEFT_ANKLE,  priors["left_thigh"],      priors["left_shank"],      HINGE_ROM_DEG["left_knee"]),
        (RIGHT_HIP,    RIGHT_KNEE,  RIGHT_ANKLE, priors["right_thigh"],     priors["right_shank"],     HINGE_ROM_DEG["right_knee"]),
    ]

    for root_idx, mid_idx, end_idx, upper_len, lower_len, rom_deg in chain_defs:
        root = refined[root_idx]
        end = refined[end_idx]
        if not (np.isfinite(root).all() and np.isfinite(end).all()):
            continue

        # Try current pose, then previous pose for pole direction
        pole_dir = _pole_from_pose(refined, root_idx, mid_idx, end_idx)
        if pole_dir is None and prev_pose is not None:
            pole_dir = _pole_from_pose(prev_pose, root_idx, mid_idx, end_idx)

        mid, end_adjusted = solve_two_bone_ik(
            root, end, pole_dir, upper_len, lower_len,
            min_angle_deg=rom_deg[0], max_angle_deg=rom_deg[1],
        )
        if mid is not None and end_adjusted is not None:
            refined[mid_idx] = mid
            refined[end_idx] = end_adjusted

    return refined


def compute_joint_angles(pose):
    """Compute all defined joint angles for a single pose. Returns dict."""
    angles = {}
    for name, (idx_a, idx_b, idx_c) in ANGLE_DEFINITIONS.items():
        angles[name] = _interior_angle_deg(pose[idx_a], pose[idx_b], pose[idx_c])
    return angles


def compute_trunk_flexion(pose):
    """
    Estimate trunk flexion angle from COCO keypoints.
    Uses the angle between:
      - Hip midpoint to Shoulder midpoint vector (torso direction)
      - A vertical reference vector (assumed to be [0, 0, 1] in the world frame)
    Returns the flexion angle in degrees (0° = upright, 90° = fully bent).
    """
    hip_mid = 0.5 * (pose[LEFT_HIP] + pose[RIGHT_HIP])
    shoulder_mid = 0.5 * (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER])
    if not (np.isfinite(hip_mid).all() and np.isfinite(shoulder_mid).all()):
        return np.nan

    torso_dir = _normalize(shoulder_mid - hip_mid)
    if torso_dir is None:
        return np.nan

    # Vertical direction (assuming Z-up after alignment)
    vertical = np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(np.dot(torso_dir, vertical), -1.0, 1.0)
    deviation_deg = math.degrees(math.acos(cos_angle))
    return deviation_deg  # 0° = perfectly upright


def main():
    print("[Info] Starting IK refinement and joint angle extraction...")

    # Find input data
    input_candidates = [
        os.path.join(RESULTS_DIR, "yolo_3d_optimized.npz"),
        os.path.join(RESULTS_DIR, "yolo_3d_raw.npz"),
    ]
    input_path = None
    for p in input_candidates:
        if os.path.exists(p):
            input_path = p
            break
    if input_path is None:
        print("[Error] No input data found. Run 02_batch_inference.py first.")
        return

    print(f"[Info] Loading: {os.path.basename(input_path)}")
    data = np.load(input_path)
    keypoints = data['keypoints']      # (N, 17, 3)
    timestamps = data['timestamps']    # (N,)

    # Estimate bone priors from the data
    priors = estimate_bone_priors(keypoints, timestamps)
    print("[Info] Estimated bone priors (cm):")
    for name, val in priors.items():
        print(f"       {name}: {val:.2f}")

    # IK refinement pass
    print("[Info] Applying IK constraints...")
    refined_keypoints = np.full_like(keypoints, np.nan, dtype=np.float64)
    prev_pose = None
    for idx in range(len(keypoints)):
        refined_keypoints[idx] = refine_pose_with_ik(keypoints[idx], priors, prev_pose=prev_pose)
        prev_pose = refined_keypoints[idx]

    # Compute joint angle sequences
    print("[Info] Computing joint angle sequences...")
    angle_names = list(ANGLE_DEFINITIONS.keys())
    angle_values = np.full((len(refined_keypoints), len(angle_names)), np.nan, dtype=np.float64)
    trunk_flexion = np.full(len(refined_keypoints), np.nan, dtype=np.float64)

    for frame_idx, pose in enumerate(refined_keypoints):
        angles = compute_joint_angles(pose)
        for angle_idx, angle_name in enumerate(angle_names):
            angle_values[frame_idx, angle_idx] = angles[angle_name]
        trunk_flexion[frame_idx] = compute_trunk_flexion(pose)

    # Summary statistics
    n_valid = np.isfinite(angle_values).all(axis=1).sum()
    n_trunk_valid = np.isfinite(trunk_flexion).sum()
    print(f"[Info] Joint angle coverage: {n_valid}/{len(angle_values)} frames with all angles valid")
    print(f"[Info] Trunk flexion coverage: {n_trunk_valid}/{len(trunk_flexion)} frames valid")

    for i, name in enumerate(angle_names):
        med = np.nanmedian(angle_values[:, i])
        print(f"       {name}: median = {med:.1f}°")
    print(f"       Trunk Flexion: median = {np.nanmedian(trunk_flexion):.1f}°")

    # Save
    output_path = os.path.join(RESULTS_DIR, "yolo_3d_ik_refined.npz")
    np.savez(
        output_path,
        timestamps=timestamps,
        keypoints=refined_keypoints,
        angle_names=np.array(angle_names),
        angle_values=angle_values,
        trunk_flexion=trunk_flexion,
        bone_prior_names=np.array(list(priors.keys())),
        bone_prior_values=np.array(list(priors.values()), dtype=np.float64),
    )
    print(f"[Info] IK-refined data saved to: {output_path}")
    print("[Info] Done.")


if __name__ == "__main__":
    main()
