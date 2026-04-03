import math
import os

import cv2
import numpy as np

from utils import StereoDataLoader


DEFAULT_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)


def build_asymmetric_grid_object_points(pattern_size, square_size_cm):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    for row_idx in range(pattern_size[1]):
        for col_idx in range(pattern_size[0]):
            x = (col_idx + 0.5 * (row_idx % 2)) * square_size_cm
            y = row_idx * (square_size_cm / 2.0)
            objp[row_idx * pattern_size[0] + col_idx] = [x, y, 0.0]
    return objp


def build_grid_edges(pattern_size):
    edges = []
    cols, rows = pattern_size
    for row_idx in range(rows):
        for col_idx in range(cols):
            idx = row_idx * cols + col_idx
            if col_idx + 1 < cols:
                edges.append((idx, idx + 1))
            if row_idx + 1 < rows:
                next_idx = (row_idx + 1) * cols + col_idx
                edges.append((idx, next_idx))
    return edges


def detect_circle_grid_pairs(data_dir, video_pairs, pattern_size, square_size_cm, use_clustering=True):
    object_points = build_asymmetric_grid_object_points(pattern_size, square_size_cm)
    all_entries = []
    image_size = None
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID
    if use_clustering and hasattr(cv2, "CALIB_CB_CLUSTERING"):
        flags |= cv2.CALIB_CB_CLUSTERING

    per_pair_counts = {}
    for left_vid, right_vid, left_txt, right_txt in video_pairs:
        pair_name = left_vid.replace("_left.avi", "")
        loader = StereoDataLoader(
            os.path.join(data_dir, left_vid),
            os.path.join(data_dir, right_vid),
            os.path.join(data_dir, left_txt),
            os.path.join(data_dir, right_txt),
        )
        detected = 0
        while True:
            frame_l, frame_r, frame_id, _ = loader.get_next_pair()
            if frame_l is None:
                break

            if image_size is None:
                image_size = frame_l.shape[:2][::-1]

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            found_l, corners_l = cv2.findCirclesGrid(gray_l, pattern_size, flags=flags)
            found_r, corners_r = cv2.findCirclesGrid(gray_r, pattern_size, flags=flags)
            if not (found_l and found_r):
                continue

            detected += 1
            all_entries.append(
                {
                    "pair_name": pair_name,
                    "frame_id": int(frame_id),
                    "obj": object_points.copy(),
                    "img_l": corners_l.astype(np.float32),
                    "img_r": corners_r.astype(np.float32),
                }
            )
        loader.release()
        per_pair_counts[pair_name] = detected

    return all_entries, image_size, per_pair_counts


def group_entries_by_pair(entries):
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry["pair_name"], []).append(entry)
    return grouped


def _calibrate_single_camera(objpoints, imgpoints, image_size, use_rational_model=False):
    flags = cv2.CALIB_RATIONAL_MODEL if use_rational_model else 0
    return cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
        criteria=DEFAULT_CRITERIA,
        flags=flags,
    )


def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        reproj, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, reproj, cv2.NORM_L2) / len(reproj)
        errors.append(float(error))
    return np.asarray(errors, dtype=np.float64)


def calibrate_stereo_from_entries(entries, image_size, reprojection_threshold_px, use_rational_model=False, fix_intrinsic=True):
    if len(entries) < 8:
        return None

    objpoints = [entry["obj"] for entry in entries]
    imgpoints_l = [entry["img_l"] for entry in entries]
    imgpoints_r = [entry["img_r"] for entry in entries]

    try:
        _, mtx_l, dist_l, rvecs_l, tvecs_l = _calibrate_single_camera(
            objpoints, imgpoints_l, image_size, use_rational_model=use_rational_model
        )
        _, mtx_r, dist_r, rvecs_r, tvecs_r = _calibrate_single_camera(
            objpoints, imgpoints_r, image_size, use_rational_model=use_rational_model
        )
    except cv2.error:
        return None

    errors_l = compute_reprojection_errors(objpoints, imgpoints_l, rvecs_l, tvecs_l, mtx_l, dist_l)
    errors_r = compute_reprojection_errors(objpoints, imgpoints_r, rvecs_r, tvecs_r, mtx_r, dist_r)
    keep_mask = (errors_l <= reprojection_threshold_px) & (errors_r <= reprojection_threshold_px)
    kept_entries = [entry for entry, keep in zip(entries, keep_mask.tolist()) if keep]
    if len(kept_entries) < 8:
        return None

    kept_obj = [entry["obj"] for entry in kept_entries]
    kept_l = [entry["img_l"] for entry in kept_entries]
    kept_r = [entry["img_r"] for entry in kept_entries]

    try:
        _, mtx_l, dist_l, _, _ = _calibrate_single_camera(
            kept_obj, kept_l, image_size, use_rational_model=use_rational_model
        )
        _, mtx_r, dist_r, _, _ = _calibrate_single_camera(
            kept_obj, kept_r, image_size, use_rational_model=use_rational_model
        )

        stereo_flags = cv2.CALIB_RATIONAL_MODEL if use_rational_model else 0
        if fix_intrinsic:
            stereo_flags |= cv2.CALIB_FIX_INTRINSIC
        else:
            stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            kept_obj,
            kept_l,
            kept_r,
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            image_size,
            criteria=DEFAULT_CRITERIA,
            flags=stereo_flags,
        )
    except cv2.error:
        return None

    return {
        "rms": float(ret),
        "mtx_l": M1,
        "dist_l": d1,
        "mtx_r": M2,
        "dist_r": d2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "kept_entries": kept_entries,
        "kept_count": len(kept_entries),
        "total_count": len(entries),
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "use_rational_model": bool(use_rational_model),
        "fix_intrinsic": bool(fix_intrinsic),
    }


def rigid_alignment_rmse(points_a, points_b):
    if len(points_a) < 4:
        return float("nan")

    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)
    aa = points_a - centroid_a
    bb = points_b - centroid_b
    h = aa.T @ bb
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = centroid_b - rotation @ centroid_a
    aligned = (rotation @ points_a.T).T + translation
    return float(np.sqrt(np.mean(np.sum((aligned - points_b) ** 2, axis=1))))


def fit_plane_rms(points_3d):
    centered = points_3d - np.mean(points_3d, axis=0)
    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1]
    distances = centered @ normal
    return float(np.sqrt(np.mean(distances**2)))


def summarize_values(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def solve_frame_pose_reprojection(objp, imgp, camera_matrix, dist_coeffs):
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return float("nan")
    reproj, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    return float(cv2.norm(imgp, reproj, cv2.NORM_L2) / len(reproj))


def evaluate_calibration(entries, image_size, params, pattern_size, square_size_cm):
    edges = build_grid_edges(pattern_size)
    mtx_l, dist_l = params["mtx_l"], params["dist_l"]
    mtx_r, dist_r = params["mtx_r"], params["dist_r"]
    rotation, translation = params["R"], params["T"]

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        rotation,
        translation,
        alpha=0,
    )

    grouped = group_entries_by_pair(entries)
    pair_summaries = []
    aggregate_frame_metrics = {
        "left_reprojection_px": [],
        "right_reprojection_px": [],
        "vertical_disparity_px_mean": [],
        "vertical_disparity_px_p95": [],
        "edge_abs_error_cm_mean": [],
        "edge_abs_error_cm_p95": [],
        "edge_scale_mean": [],
        "plane_rms_cm": [],
        "rigid_alignment_rmse_cm": [],
    }

    for pair_name, pair_entries in grouped.items():
        per_frame = {key: [] for key in aggregate_frame_metrics}
        for entry in pair_entries:
            objp = entry["obj"].astype(np.float64)
            img_l = entry["img_l"].astype(np.float64)
            img_r = entry["img_r"].astype(np.float64)

            left_reproj = solve_frame_pose_reprojection(objp, img_l, mtx_l, dist_l)
            right_reproj = solve_frame_pose_reprojection(objp, img_r, mtx_r, dist_r)

            rect_l = cv2.undistortPoints(img_l, mtx_l, dist_l, R=R1, P=P1)[:, 0, :]
            rect_r = cv2.undistortPoints(img_r, mtx_r, dist_r, R=R2, P=P2)[:, 0, :]
            vertical_disparity = np.abs(rect_l[:, 1] - rect_r[:, 1])

            homog = cv2.triangulatePoints(P1, P2, rect_l.T, rect_r.T).T
            valid = np.abs(homog[:, 3]) > 1e-8
            points_3d = np.full((len(homog), 3), np.nan, dtype=np.float64)
            points_3d[valid] = homog[valid, :3] / homog[valid, 3:4]
            finite_points = points_3d[np.isfinite(points_3d).all(axis=1)]
            finite_obj = objp[np.isfinite(points_3d).all(axis=1)]

            rigid_rmse = rigid_alignment_rmse(finite_points, finite_obj) if len(finite_points) >= 4 else float("nan")
            plane_rms = fit_plane_rms(finite_points) if len(finite_points) >= 4 else float("nan")

            edge_errors = []
            edge_scales = []
            for idx_a, idx_b in edges:
                if not (np.isfinite(points_3d[idx_a]).all() and np.isfinite(points_3d[idx_b]).all()):
                    continue
                tri_dist = np.linalg.norm(points_3d[idx_a] - points_3d[idx_b])
                obj_dist = np.linalg.norm(objp[idx_a] - objp[idx_b])
                if obj_dist <= 1e-6:
                    continue
                edge_errors.append(abs(tri_dist - obj_dist))
                edge_scales.append(tri_dist / obj_dist)

            frame_metrics = {
                "left_reprojection_px": left_reproj,
                "right_reprojection_px": right_reproj,
                "vertical_disparity_px_mean": float(np.mean(vertical_disparity)),
                "vertical_disparity_px_p95": float(np.percentile(vertical_disparity, 95)),
                "edge_abs_error_cm_mean": float(np.mean(edge_errors)) if edge_errors else float("nan"),
                "edge_abs_error_cm_p95": float(np.percentile(edge_errors, 95)) if edge_errors else float("nan"),
                "edge_scale_mean": float(np.mean(edge_scales)) if edge_scales else float("nan"),
                "plane_rms_cm": plane_rms,
                "rigid_alignment_rmse_cm": rigid_rmse,
            }

            for key, value in frame_metrics.items():
                per_frame[key].append(value)
                aggregate_frame_metrics[key].append(value)

        pair_summaries.append(
            {
                "pair_name": pair_name,
                "total_pairs": len(pair_entries),
                "detected_frames": len(pair_entries),
                "used_frames": len(pair_entries),
                "metrics": {key: summarize_values(values) for key, values in per_frame.items()},
            }
        )

    return {
        "baseline_cm": float(np.linalg.norm(translation)),
        "square_size_cm": float(square_size_cm),
        "pairs": pair_summaries,
        "aggregate_mean_of_frame_metrics": {
            key: summarize_values(values) for key, values in aggregate_frame_metrics.items()
        },
    }


def score_validation_summary(summary):
    metrics = summary["aggregate_mean_of_frame_metrics"]
    disp_mean = metrics["vertical_disparity_px_mean"]["mean"]
    disp_p95 = metrics["vertical_disparity_px_p95"]["mean"]
    rigid_rmse = metrics["rigid_alignment_rmse_cm"]["mean"]
    plane_rms = metrics["plane_rms_cm"]["mean"]
    return float(disp_mean + 0.25 * disp_p95 + 10.0 * rigid_rmse + 5.0 * plane_rms)


def search_calibration_config(entries_by_pair, image_size, pattern_size, square_size_cm, config_grid):
    pair_names = sorted(entries_by_pair)
    search_results = []
    for config in config_grid:
        fold_results = []
        for holdout_pair in pair_names:
            train_entries = []
            for pair_name, pair_entries in entries_by_pair.items():
                if pair_name == holdout_pair:
                    continue
                train_entries.extend(pair_entries)
            validation_entries = list(entries_by_pair[holdout_pair])

            params = calibrate_stereo_from_entries(
                train_entries,
                image_size,
                reprojection_threshold_px=config["reprojection_threshold_px"],
                use_rational_model=config["use_rational_model"],
                fix_intrinsic=config["fix_intrinsic"],
            )
            if params is None:
                fold_results = []
                break

            validation_summary = evaluate_calibration(
                validation_entries,
                image_size,
                params,
                pattern_size,
                square_size_cm,
            )
            fold_results.append(
                {
                    "holdout_pair": holdout_pair,
                    "train_frames": int(params["kept_count"]),
                    "validation_summary": validation_summary,
                    "validation_score": score_validation_summary(validation_summary),
                }
            )

        if not fold_results:
            continue

        search_results.append(
            {
                "config": config,
                "folds": fold_results,
                "mean_validation_score": float(np.mean([fold["validation_score"] for fold in fold_results])),
                "mean_holdout_vertical_disparity_px": float(
                    np.mean(
                        [
                            fold["validation_summary"]["aggregate_mean_of_frame_metrics"]["vertical_disparity_px_mean"]["mean"]
                            for fold in fold_results
                        ]
                    )
                ),
                "mean_holdout_vertical_disparity_p95_px": float(
                    np.mean(
                        [
                            fold["validation_summary"]["aggregate_mean_of_frame_metrics"]["vertical_disparity_px_p95"]["mean"]
                            for fold in fold_results
                        ]
                    )
                ),
                "mean_holdout_rigid_rmse_cm": float(
                    np.mean(
                        [
                            fold["validation_summary"]["aggregate_mean_of_frame_metrics"]["rigid_alignment_rmse_cm"]["mean"]
                            for fold in fold_results
                        ]
                    )
                ),
            }
        )

    if not search_results:
        return None, []

    best_result = min(search_results, key=lambda item: item["mean_validation_score"])
    return best_result, search_results

