"""Sanity-check stereo calibration scale using the calibration board and pose height proxy."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from calibration_utils import (  # noqa: E402
    build_grid_edges,
    detect_circle_grid_pairs,
    evaluate_calibration,
)


DATA_DIR = PROJECT_ROOT / "Calibration_video"
VIDEO_PAIRS = [
    ("cap_0_left.avi", "cap_0_right.avi", "cap_0_left.txt", "cap_0_right.txt"),
    ("cap_1_left.avi", "cap_1_right.avi", "cap_1_left.txt", "cap_1_right.txt"),
]
PATTERN_SIZE = (5, 9)
GRID_SPACING_CM = 15.0
CAMERA_PARAMS_PATH = PROJECT_ROOT / "shared" / "camera_params.npz"
POSE_INPUT_PATH = (
    PROJECT_ROOT
    / "01_stereo_triangulation"
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
RESULTS_DIR = METHOD_DIR / "results"
SUMMARY_JSON_PATH = RESULTS_DIR / "calibration_sanity_check.json"
SUMMARY_MD_PATH = RESULTS_DIR / "calibration_sanity_check.md"
POSE_TRUE_HEIGHT_CM = 169.0


def summarize_values(values: np.ndarray) -> dict[str, float]:
    """Return compact summary statistics for finite values."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def load_camera_params(path: Path) -> dict[str, np.ndarray]:
    """Load active stereo camera parameters."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def compute_camera_geometry_summary(params: dict[str, np.ndarray]) -> dict[str, float]:
    """Extract baseline and focal geometry from camera parameters."""
    mtx_l = np.asarray(params["mtx_l"], dtype=np.float64)
    mtx_r = np.asarray(params["mtx_r"], dtype=np.float64)
    translation = np.asarray(params["T"], dtype=np.float64).reshape(3)
    return {
        "baseline_cm": float(np.linalg.norm(translation)),
        "tx_cm": float(translation[0]),
        "ty_cm": float(translation[1]),
        "tz_cm": float(translation[2]),
        "fx_left_px": float(mtx_l[0, 0]),
        "fy_left_px": float(mtx_l[1, 1]),
        "cx_left_px": float(mtx_l[0, 2]),
        "cy_left_px": float(mtx_l[1, 2]),
        "fx_right_px": float(mtx_r[0, 0]),
        "fy_right_px": float(mtx_r[1, 1]),
        "cx_right_px": float(mtx_r[0, 2]),
        "cy_right_px": float(mtx_r[1, 2]),
    }


def compute_board_frame_metrics(
    entries: list[dict[str, Any]],
    image_size: tuple[int, int],
    params: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    """Triangulate each detected board frame and collect scale metrics."""
    edges = build_grid_edges(PATTERN_SIZE)
    mtx_l, dist_l = params["mtx_l"], params["dist_l"]
    mtx_r, dist_r = params["mtx_r"], params["dist_r"]
    rotation, translation = params["R"], params["T"]

    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        rotation,
        translation,
        alpha=0,
    )

    frame_metrics: list[dict[str, float | int | str]] = []
    for entry in entries:
        img_l = entry["img_l"].astype(np.float64)
        img_r = entry["img_r"].astype(np.float64)
        objp = entry["obj"].astype(np.float64)

        rect_l = cv2.undistortPoints(img_l, mtx_l, dist_l, R=r1, P=p1)[:, 0, :]
        rect_r = cv2.undistortPoints(img_r, mtx_r, dist_r, R=r2, P=p2)[:, 0, :]
        homog = cv2.triangulatePoints(p1, p2, rect_l.T, rect_r.T).T
        valid = np.abs(homog[:, 3]) > 1e-8
        points_3d = np.full((len(homog), 3), np.nan, dtype=np.float64)
        points_3d[valid] = homog[valid, :3] / homog[valid, 3:4]

        edge_errors = []
        edge_scales = []
        for idx_a, idx_b in edges:
            if not (np.isfinite(points_3d[idx_a]).all() and np.isfinite(points_3d[idx_b]).all()):
                continue
            tri_dist = float(np.linalg.norm(points_3d[idx_a] - points_3d[idx_b]))
            obj_dist = float(np.linalg.norm(objp[idx_a] - objp[idx_b]))
            if obj_dist <= 1e-8:
                continue
            edge_errors.append(abs(tri_dist - obj_dist))
            edge_scales.append(tri_dist / obj_dist)

        frame_metrics.append(
            {
                "pair_name": str(entry["pair_name"]),
                "frame_id": int(entry["frame_id"]),
                "edge_abs_error_cm_mean": float(np.mean(edge_errors)) if edge_errors else float("nan"),
                "edge_scale_mean": float(np.mean(edge_scales)) if edge_scales else float("nan"),
            }
        )
    return frame_metrics


def compute_height_proxy_summary(pose_path: Path) -> dict[str, Any]:
    """Estimate body height proxy from the stereo pose skeleton."""
    pose = np.load(pose_path, allow_pickle=True)
    keypoints = np.asarray(pose["keypoints"], dtype=np.float64)

    head = keypoints[:, 0, :]
    shoulder_mid = 0.5 * (keypoints[:, 5, :] + keypoints[:, 6, :])
    hip_mid = 0.5 * (keypoints[:, 11, :] + keypoints[:, 12, :])

    trunk = (
        np.linalg.norm(head - shoulder_mid, axis=1)
        + np.linalg.norm(shoulder_mid - hip_mid, axis=1)
    )
    left_leg = (
        np.linalg.norm(keypoints[:, 11, :] - keypoints[:, 13, :], axis=1)
        + np.linalg.norm(keypoints[:, 13, :] - keypoints[:, 15, :], axis=1)
    )
    right_leg = (
        np.linalg.norm(keypoints[:, 12, :] - keypoints[:, 14, :], axis=1)
        + np.linalg.norm(keypoints[:, 14, :] - keypoints[:, 16, :], axis=1)
    )

    left_height = trunk + left_leg
    right_height = trunk + right_leg
    mean_height = trunk + 0.5 * (left_leg + right_leg)
    best_side_height = np.maximum(left_height, right_height)

    summary = {
        "pose_input_path": str(pose_path),
        "left_path_cm": summarize_values(left_height),
        "right_path_cm": summarize_values(right_height),
        "mean_legs_path_cm": summarize_values(mean_height),
        "best_side_path_cm": summarize_values(best_side_height),
    }

    reference_cm = summary["best_side_path_cm"]["median"]
    summary["reference_true_height_cm"] = POSE_TRUE_HEIGHT_CM
    summary["height_bias_cm"] = float(reference_cm - POSE_TRUE_HEIGHT_CM)
    summary["height_scale_ratio"] = float(reference_cm / POSE_TRUE_HEIGHT_CM)
    summary["height_abs_error_percent"] = float(abs(reference_cm / POSE_TRUE_HEIGHT_CM - 1.0) * 100.0)
    return summary


def to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-safe builtins."""
    if isinstance(obj, dict):
        return {key: to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_builtin(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    """Render a compact Markdown report."""
    geom = payload["camera_geometry"]
    board = payload["board_sanity"]
    height = payload["height_proxy"]
    rep = board["representative_frame"]

    lines = [
        "# Calibration Sanity Check",
        "",
        "## Conclusion",
        "",
        f"- Decision: **{payload['decision']}**",
        f"- Rationale: {payload['decision_reason']}",
        "",
        "## Camera Geometry",
        "",
        f"- Baseline: `{geom['baseline_cm']:.3f} cm`",
        f"- Left intrinsics: `fx={geom['fx_left_px']:.2f}px`, `fy={geom['fy_left_px']:.2f}px`, `cx={geom['cx_left_px']:.2f}px`, `cy={geom['cy_left_px']:.2f}px`",
        f"- Right intrinsics: `fx={geom['fx_right_px']:.2f}px`, `fy={geom['fy_right_px']:.2f}px`, `cx={geom['cx_right_px']:.2f}px`, `cy={geom['cy_right_px']:.2f}px`",
        "",
        "## Board Sanity",
        "",
        f"- Target type: `asymmetric circle grid`",
        f"- Pattern size: `{PATTERN_SIZE[0]} x {PATTERN_SIZE[1]}`",
        f"- Grid spacing: `{GRID_SPACING_CM:.2f} cm`",
        f"- Detected stereo frames: `{board['detected_frames']}`",
        f"- Mean board edge scale: `{board['edge_scale_mean']:.4f}`",
        f"- Board scale error: `{board['edge_scale_abs_error_percent']:.2f}%`",
        f"- Mean edge absolute error: `{board['edge_abs_error_cm_mean']:.3f} cm`",
        f"- Mean rigid alignment RMSE: `{board['rigid_alignment_rmse_cm_mean']:.3f} cm`",
        f"- Mean plane RMS: `{board['plane_rms_cm_mean']:.3f} cm`",
        "",
        "Representative frame:",
        f"- Pair / frame: `{rep['pair_name']}` / `{rep['frame_id']}`",
        f"- Mean edge scale: `{rep['edge_scale_mean']:.4f}`",
        f"- Mean edge absolute error: `{rep['edge_abs_error_cm_mean']:.3f} cm`",
        "",
        "## Height Proxy",
        "",
        f"- Pose input: `{height['pose_input_path']}`",
        f"- Reference true height: `{height['reference_true_height_cm']:.1f} cm`",
        f"- Stereo best-side path median: `{height['best_side_path_cm']['median']:.2f} cm`",
        f"- Height scale ratio: `{height['height_scale_ratio']:.4f}`",
        f"- Height absolute error: `{height['height_abs_error_percent']:.2f}%`",
        f"- Height bias: `{height['height_bias_cm']:+.2f} cm`",
        "",
        "Auxiliary path statistics:",
        f"- Left path median / p95: `{height['left_path_cm']['median']:.2f} / {height['left_path_cm']['p95']:.2f} cm`",
        f"- Right path median / p95: `{height['right_path_cm']['median']:.2f} / {height['right_path_cm']['p95']:.2f} cm`",
        f"- Mean-legs path median / p95: `{height['mean_legs_path_cm']['median']:.2f} / {height['mean_legs_path_cm']['p95']:.2f} cm`",
        "",
        "## Notes",
        "",
        "- The calibration board sanity check is the primary decision signal.",
        "- The body-height proxy is only a secondary scale check because COCO keypoints do not directly encode anatomical body height.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the calibration sanity check."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    params = load_camera_params(CAMERA_PARAMS_PATH)
    geometry = compute_camera_geometry_summary(params)

    entries, image_size, _ = detect_circle_grid_pairs(
        str(DATA_DIR),
        VIDEO_PAIRS,
        PATTERN_SIZE,
        GRID_SPACING_CM,
        use_clustering=True,
    )
    if image_size is None or not entries:
        raise RuntimeError("No valid stereo circle-grid detections found in Calibration_video.")

    board_summary = evaluate_calibration(entries, image_size, params, PATTERN_SIZE, GRID_SPACING_CM)
    frame_metrics = compute_board_frame_metrics(entries, image_size, params)
    finite_frames = [item for item in frame_metrics if np.isfinite(item["edge_scale_mean"])]
    representative_frame = min(
        finite_frames,
        key=lambda item: abs(float(item["edge_scale_mean"]) - 1.0),
    )

    board_metrics = board_summary["aggregate_mean_of_frame_metrics"]
    edge_scale_mean = float(board_metrics["edge_scale_mean"]["mean"])
    board_scale_abs_error_percent = abs(edge_scale_mean - 1.0) * 100.0

    height_summary = compute_height_proxy_summary(POSE_INPUT_PATH)

    if board_scale_abs_error_percent > 5.0:
        decision = "Calibration suspicious"
        reason = (
            f"Board scale error is {board_scale_abs_error_percent:.2f}%, above the 5% threshold."
        )
    elif board_scale_abs_error_percent < 2.0:
        decision = "Calibration scale looks OK"
        reason = (
            f"Board scale error is {board_scale_abs_error_percent:.2f}%, below the 2% threshold."
        )
    else:
        decision = "Calibration scale is inconclusive"
        reason = (
            f"Board scale error is {board_scale_abs_error_percent:.2f}%, between the 2% and 5% thresholds."
        )

    payload = {
        "camera_params_path": str(CAMERA_PARAMS_PATH),
        "camera_geometry": geometry,
        "board_sanity": {
            "target_type": "asymmetric_circle_grid",
            "pattern_size": list(PATTERN_SIZE),
            "grid_spacing_cm": GRID_SPACING_CM,
            "detected_frames": len(entries),
            "baseline_cm": board_summary["baseline_cm"],
            "edge_scale_mean": edge_scale_mean,
            "edge_scale_abs_error_percent": float(board_scale_abs_error_percent),
            "edge_abs_error_cm_mean": float(board_metrics["edge_abs_error_cm_mean"]["mean"]),
            "edge_abs_error_cm_p95_mean": float(board_metrics["edge_abs_error_cm_p95"]["mean"]),
            "rigid_alignment_rmse_cm_mean": float(board_metrics["rigid_alignment_rmse_cm"]["mean"]),
            "plane_rms_cm_mean": float(board_metrics["plane_rms_cm"]["mean"]),
            "left_reprojection_px_mean": float(board_metrics["left_reprojection_px"]["mean"]),
            "right_reprojection_px_mean": float(board_metrics["right_reprojection_px"]["mean"]),
            "vertical_disparity_px_mean": float(board_metrics["vertical_disparity_px_mean"]["mean"]),
            "vertical_disparity_px_p95_mean": float(board_metrics["vertical_disparity_px_p95"]["mean"]),
            "representative_frame": representative_frame,
        },
        "height_proxy": height_summary,
        "decision": decision,
        "decision_reason": reason,
    }

    SUMMARY_JSON_PATH.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")
    write_markdown_report(payload, SUMMARY_MD_PATH)

    print(f"[saved] {SUMMARY_JSON_PATH}")
    print(f"[saved] {SUMMARY_MD_PATH}")
    print(f"[result] {decision}: {reason}")


if __name__ == "__main__":
    main()
