"""Prototype anatomically constrained stereo triangulation with limb-length priors."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))


CAMERA_PARAMS_PATH = PROJECT_ROOT / "shared" / "camera_params.npz"
DATA_DIR = PROJECT_ROOT / "2025_Ergonomics_Data"
INPUT_POSE_PATH = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "raw_pose.npz"
)
RESULTS_DIR = METHOD_DIR / "results" / "constrained_triangulation_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_VALUES = tuple(
    float(x.strip())
    for x in os.environ.get("POSE_CONSTRAINED_LAMBDAS", "0.001,0.01,0.1,1.0").split(",")
    if x.strip()
)
MAX_NFEV = int(os.environ.get("POSE_CONSTRAINED_MAX_NFEV", "40"))
MIN_CONF = float(os.environ.get("POSE_CONSTRAINED_MIN_CONF", "0.05"))
LEFT_VIDEO = DATA_DIR / "0_video_left.avi"

LIMB_BONES = (
    ("upper_arm", 5, 7),
    ("upper_arm", 6, 8),
    ("forearm", 7, 9),
    ("forearm", 8, 10),
    ("thigh", 11, 13),
    ("thigh", 12, 14),
    ("shank", 13, 15),
    ("shank", 14, 16),
)

TARGET_LENGTHS_CM = {
    "upper_arm": 30.167881800268958,
    "forearm": 24.658789644972423,
    "thigh": 41.114084566423244,
    "shank": 40.11084295399438,
}


def load_image_size(video_path: Path) -> tuple[int, int]:
    """Read one frame to recover the dataset image size."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read video frame from {video_path}")
    height, width = frame.shape[:2]
    return width, height


def compute_projection_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Rebuild rectified stereo projection matrices P1/P2."""
    params = np.load(CAMERA_PARAMS_PATH)
    mtx_l, dist_l = params["mtx_l"], params["dist_l"]
    mtx_r, dist_r = params["mtx_r"], params["dist_r"]
    rotation, translation = params["R"], params["T"]
    image_size = load_image_size(LEFT_VIDEO)
    _, _, p1, p2, _, _, _ = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        rotation,
        translation,
        alpha=0,
    )
    return p1.astype(np.float64), p2.astype(np.float64)


def project_point(proj_matrix: np.ndarray, point_3d: np.ndarray) -> np.ndarray:
    """Project one 3D point into one rectified camera."""
    homog = np.append(point_3d, 1.0)
    proj = proj_matrix @ homog
    if abs(proj[2]) < 1e-8:
        return np.full(2, np.nan, dtype=np.float64)
    return proj[:2] / proj[2]


def weighted_dlt_triangulate(
    p1: np.ndarray,
    p2: np.ndarray,
    pt_left: np.ndarray,
    pt_right: np.ndarray,
    conf_left: float,
    conf_right: float,
) -> np.ndarray:
    """Triangulate one point using the same weighted DLT structure as the baseline."""
    w1 = math.sqrt(max(float(conf_left), 1e-4))
    w2 = math.sqrt(max(float(conf_right), 1e-4))
    mat = np.vstack(
        [
            w1 * (pt_left[0] * p1[2] - p1[0]),
            w1 * (pt_left[1] * p1[2] - p1[1]),
            w2 * (pt_right[0] * p2[2] - p2[0]),
            w2 * (pt_right[1] * p2[2] - p2[1]),
        ]
    )
    _, _, vt = np.linalg.svd(mat)
    homog = vt[-1]
    if abs(homog[3]) < 1e-8:
        return np.full(3, np.nan, dtype=np.float64)
    return homog[:3] / homog[3]


def load_target_lengths() -> dict[str, float]:
    """Return fixed Xsens limb targets used by constrained triangulation."""
    return dict(TARGET_LENGTHS_CM)


def ensure_initial_pose(
    pose_frame: np.ndarray,
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    conf_left: np.ndarray,
    conf_right: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Fill easy missing joints using direct DLT when rectified 2D observations exist."""
    initial = np.asarray(pose_frame, dtype=np.float64).copy()
    for joint_idx in range(initial.shape[0]):
        if np.isfinite(initial[joint_idx]).all():
            continue
        if not (
            np.isfinite(rect_left[joint_idx]).all()
            and np.isfinite(rect_right[joint_idx]).all()
            and conf_left[joint_idx] >= MIN_CONF
            and conf_right[joint_idx] >= MIN_CONF
        ):
            continue
        initial[joint_idx] = weighted_dlt_triangulate(
            p1,
            p2,
            rect_left[joint_idx],
            rect_right[joint_idx],
            conf_left[joint_idx],
            conf_right[joint_idx],
        )
    return initial


def collect_frame_reprojection_stats(
    pose_frame: np.ndarray,
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    conf_left: np.ndarray,
    conf_right: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> tuple[float, int]:
    """Measure mean reprojection error across observed joints."""
    errors = []
    for joint_idx in range(pose_frame.shape[0]):
        if not np.isfinite(pose_frame[joint_idx]).all():
            continue
        if not (
            np.isfinite(rect_left[joint_idx]).all()
            and np.isfinite(rect_right[joint_idx]).all()
            and conf_left[joint_idx] >= MIN_CONF
            and conf_right[joint_idx] >= MIN_CONF
        ):
            continue
        proj_l = project_point(p1, pose_frame[joint_idx])
        proj_r = project_point(p2, pose_frame[joint_idx])
        if not np.isfinite(proj_l).all() or not np.isfinite(proj_r).all():
            continue
        err_l = float(np.linalg.norm(proj_l - rect_left[joint_idx]))
        err_r = float(np.linalg.norm(proj_r - rect_right[joint_idx]))
        errors.append(0.5 * (err_l + err_r))
    if not errors:
        return float("nan"), 0
    return float(np.mean(errors)), len(errors)


def collect_frame_bone_stats(
    pose_frame: np.ndarray,
    target_lengths: dict[str, float],
) -> dict[str, float]:
    """Measure mean absolute bone-length error across constrained bones."""
    out: dict[str, list[float]] = {segment: [] for segment in target_lengths}
    for segment, idx_a, idx_b in LIMB_BONES:
        if not (np.isfinite(pose_frame[idx_a]).all() and np.isfinite(pose_frame[idx_b]).all()):
            continue
        length = float(np.linalg.norm(pose_frame[idx_a] - pose_frame[idx_b]))
        out[segment].append(abs(length - target_lengths[segment]))
    return {
        segment: (float(np.mean(values)) if values else float("nan"))
        for segment, values in out.items()
    }


def optimize_frame(
    pose_frame: np.ndarray,
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    conf_left: np.ndarray,
    conf_right: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    target_lengths: dict[str, float],
    lambda_bone: float,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Optimize one frame with reprojection + limb-length residuals."""
    initial = ensure_initial_pose(pose_frame, rect_left, rect_right, conf_left, conf_right, p1, p2)
    observed = (
        np.isfinite(initial).all(axis=1)
        & np.isfinite(rect_left).all(axis=1)
        & np.isfinite(rect_right).all(axis=1)
        & (conf_left >= MIN_CONF)
        & (conf_right >= MIN_CONF)
    )

    variable_mask = np.isfinite(initial).all(axis=1) & observed
    if int(variable_mask.sum()) < 4:
        return initial, {
            "optimized": False,
            "num_variable_joints": int(variable_mask.sum()),
            "num_active_bones": 0,
            "nfev": 0,
        }

    joint_indices = np.where(variable_mask)[0]
    index_lookup = {joint_idx: local_idx for local_idx, joint_idx in enumerate(joint_indices)}
    active_bones = []
    for segment, idx_a, idx_b in LIMB_BONES:
        if idx_a in index_lookup and idx_b in index_lookup:
            active_bones.append((segment, idx_a, idx_b))

    if not active_bones:
        return initial, {
            "optimized": False,
            "num_variable_joints": int(variable_mask.sum()),
            "num_active_bones": 0,
            "nfev": 0,
        }

    x0 = initial[variable_mask].reshape(-1)

    def residuals(flat_state: np.ndarray) -> np.ndarray:
        pose = initial.copy()
        pose[variable_mask] = flat_state.reshape(-1, 3)
        values = []

        for joint_idx in joint_indices:
            if not observed[joint_idx]:
                continue
            point = pose[joint_idx]
            proj_l = project_point(p1, point)
            proj_r = project_point(p2, point)
            if not np.isfinite(proj_l).all() or not np.isfinite(proj_r).all():
                continue
            weight_l = math.sqrt(max(float(conf_left[joint_idx]), 1e-4))
            weight_r = math.sqrt(max(float(conf_right[joint_idx]), 1e-4))
            values.extend((weight_l * (proj_l - rect_left[joint_idx])).tolist())
            values.extend((weight_r * (proj_r - rect_right[joint_idx])).tolist())

        bone_weight = math.sqrt(lambda_bone)
        for segment, idx_a, idx_b in active_bones:
            dist = float(np.linalg.norm(pose[idx_a] - pose[idx_b]))
            values.append(bone_weight * (dist - target_lengths[segment]))

        return np.asarray(values, dtype=np.float64)

    result = least_squares(
        residuals,
        x0,
        method="trf",
        max_nfev=MAX_NFEV,
        xtol=1e-4,
        ftol=1e-4,
        gtol=1e-4,
    )

    optimized_pose = initial.copy()
    optimized_pose[variable_mask] = result.x.reshape(-1, 3)
    return optimized_pose, {
        "optimized": bool(result.success),
        "num_variable_joints": int(variable_mask.sum()),
        "num_active_bones": int(len(active_bones)),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
    }


def compute_segment_length_medians(keypoints: np.ndarray) -> dict[str, float]:
    """Compute per-segment median lengths over both sides."""
    values: dict[str, list[float]] = {
        "upper_arm": [],
        "forearm": [],
        "thigh": [],
        "shank": [],
    }
    for segment, idx_a, idx_b in LIMB_BONES:
        diff = keypoints[:, idx_a, :] - keypoints[:, idx_b, :]
        lengths = np.linalg.norm(diff, axis=1)
        finite = lengths[np.isfinite(lengths)]
        if finite.size:
            values[segment].extend(finite.tolist())
    return {
        segment: float(np.median(segment_vals)) if segment_vals else float("nan")
        for segment, segment_vals in values.items()
    }


def write_markdown_summary(summary_rows: list[dict[str, Any]], path: Path) -> None:
    """Write lambda-sweep results in Markdown."""
    header = (
        "| lambda | optimized_frames | reproj_before_px | reproj_after_px | "
        "upper_arm_ratio | forearm_ratio | thigh_ratio | shank_ratio |\n"
    )
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    lines = [
        "# Constrained Triangulation Summary",
        "",
        "- Constraint set: `upper_arm`, `forearm`, `thigh`, `shank` only",
        "- Target source: fixed Xsens limb-length constants embedded in this script",
        "",
        header.strip(),
        divider.strip(),
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['lambda']:.3f} | {row['optimized_frames']} | "
            f"{row['mean_reproj_before_px']:.3f} | {row['mean_reproj_after_px']:.3f} | "
            f"{row['upper_arm_ratio']:.3f} | {row['forearm_ratio']:.3f} | "
            f"{row['thigh_ratio']:.3f} | {row['shank_ratio']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run lambda-sweep constrained triangulation."""
    pose_data = np.load(INPUT_POSE_PATH, allow_pickle=True)
    keypoints = np.asarray(pose_data["keypoints"], dtype=np.float64)
    rect_left = np.asarray(pose_data["keypoints_left_rect"], dtype=np.float64)
    rect_right = np.asarray(pose_data["keypoints_right_rect"], dtype=np.float64)
    conf_left = np.asarray(pose_data["triang_conf_left"], dtype=np.float64)
    conf_right = np.asarray(pose_data["triang_conf_right"], dtype=np.float64)

    p1, p2 = compute_projection_matrices()
    target_lengths = load_target_lengths()
    summary_rows: list[dict[str, Any]] = []
    full_summary: dict[str, Any] = {
        "input_pose_path": str(INPUT_POSE_PATH),
        "camera_params_path": str(CAMERA_PARAMS_PATH),
        "target_lengths_cm": target_lengths,
        "lambda_results": [],
    }

    for lambda_bone in LAMBDA_VALUES:
        print(f"[run] lambda={lambda_bone:.3f}")
        constrained_keypoints = np.array(keypoints, copy=True)
        frame_logs = []
        reproj_before = []
        reproj_after = []
        bone_before = {segment: [] for segment in target_lengths}
        bone_after = {segment: [] for segment in target_lengths}
        optimized_frames = 0

        for frame_idx in range(len(keypoints)):
            before_pose = keypoints[frame_idx]
            after_pose, log = optimize_frame(
                before_pose,
                rect_left[frame_idx],
                rect_right[frame_idx],
                conf_left[frame_idx],
                conf_right[frame_idx],
                p1,
                p2,
                target_lengths,
                lambda_bone,
            )
            constrained_keypoints[frame_idx] = after_pose
            frame_logs.append(log)
            optimized_frames += int(log["optimized"])

            before_reproj, _ = collect_frame_reprojection_stats(
                before_pose,
                rect_left[frame_idx],
                rect_right[frame_idx],
                conf_left[frame_idx],
                conf_right[frame_idx],
                p1,
                p2,
            )
            after_reproj, _ = collect_frame_reprojection_stats(
                after_pose,
                rect_left[frame_idx],
                rect_right[frame_idx],
                conf_left[frame_idx],
                conf_right[frame_idx],
                p1,
                p2,
            )
            if np.isfinite(before_reproj):
                reproj_before.append(before_reproj)
            if np.isfinite(after_reproj):
                reproj_after.append(after_reproj)

            before_bones = collect_frame_bone_stats(before_pose, target_lengths)
            after_bones = collect_frame_bone_stats(after_pose, target_lengths)
            for segment in target_lengths:
                if np.isfinite(before_bones[segment]):
                    bone_before[segment].append(before_bones[segment])
                if np.isfinite(after_bones[segment]):
                    bone_after[segment].append(after_bones[segment])

            if (frame_idx + 1) % 400 == 0 or frame_idx + 1 == len(keypoints):
                print(
                    f"  frame {frame_idx + 1}/{len(keypoints)} "
                    f"(optimized so far: {optimized_frames})"
                )

        segment_medians = compute_segment_length_medians(constrained_keypoints)
        ratios = {
            f"{segment}_ratio": float(segment_medians[segment] / target_lengths[segment])
            for segment in target_lengths
        }
        output_payload = {key: pose_data[key] for key in pose_data.files}
        output_payload["keypoints"] = constrained_keypoints
        output_payload["source_method"] = np.array("constrained_triangulation_v1")
        output_payload["constraint_segments"] = np.array(list(target_lengths.keys()))
        output_payload["constraint_target_lengths_cm"] = np.array(
            [target_lengths[name] for name in target_lengths], dtype=np.float64
        )
        output_payload["constraint_lambda"] = np.array(lambda_bone, dtype=np.float64)

        lambda_slug = str(lambda_bone).replace(".", "p")
        output_npz = RESULTS_DIR / f"constrained_triangulation_pose_lambda_{lambda_slug}.npz"
        np.savez(output_npz, **output_payload)

        lambda_summary = {
            "lambda": lambda_bone,
            "output_npz": str(output_npz),
            "optimized_frames": int(optimized_frames),
            "mean_reproj_before_px": float(np.mean(reproj_before)) if reproj_before else float("nan"),
            "mean_reproj_after_px": float(np.mean(reproj_after)) if reproj_after else float("nan"),
            "mean_upper_arm_abs_error_cm_after": float(np.mean(bone_after["upper_arm"])) if bone_after["upper_arm"] else float("nan"),
            "mean_forearm_abs_error_cm_after": float(np.mean(bone_after["forearm"])) if bone_after["forearm"] else float("nan"),
            "mean_thigh_abs_error_cm_after": float(np.mean(bone_after["thigh"])) if bone_after["thigh"] else float("nan"),
            "mean_shank_abs_error_cm_after": float(np.mean(bone_after["shank"])) if bone_after["shank"] else float("nan"),
            "segment_medians_cm": segment_medians,
            "frame_log_sample": frame_logs[:5],
            **ratios,
        }
        full_summary["lambda_results"].append(lambda_summary)
        summary_rows.append(lambda_summary)
        print(
            f"  optimized_frames={optimized_frames}, reproj {lambda_summary['mean_reproj_before_px']:.3f}"
            f" -> {lambda_summary['mean_reproj_after_px']:.3f}, thigh_ratio={ratios['thigh_ratio']:.3f}"
        )

    summary_json = RESULTS_DIR / "constrained_triangulation_summary.json"
    summary_md = RESULTS_DIR / "constrained_triangulation_summary.md"
    summary_json.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary_rows, summary_md)
    print(f"[saved] {summary_json}")
    print(f"[saved] {summary_md}")


if __name__ == "__main__":
    main()
