import json
import os

import numpy as np

try:
    from calibration_utils import (
        calibrate_stereo_from_entries,
        detect_circle_grid_pairs,
        evaluate_calibration,
        group_entries_by_pair,
        score_validation_summary,
        search_calibration_config,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "01_calibration.py requires calibration_utils.py, but that module is missing from the current workspace. "
        "The calibration pipeline is not reproducible until calibration_utils.py is restored."
    ) from exc


CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Calibration_video")

VIDEO_PAIRS = [
    ("cap_0_left.avi", "cap_0_right.avi", "cap_0_left.txt", "cap_0_right.txt"),
    ("cap_1_left.avi", "cap_1_right.avi", "cap_1_left.txt", "cap_1_right.txt"),
]

PATTERN_SIZE = (5, 9)
SQUARE_SIZE_CM = 15.0
USE_CLUSTERING = os.environ.get("CALIB_USE_CLUSTERING", "1") == "1"
PROMOTE_IF_IMPROVED = os.environ.get("CALIB_PROMOTE_IF_IMPROVED", "1") == "1"

OUTPUT_PATH = os.environ.get("CALIB_OUTPUT_PATH", os.path.join(SRC_DIR, "camera_params.npz"))
CANDIDATE_OUTPUT_PATH = os.environ.get("CALIB_CANDIDATE_OUTPUT_PATH", os.path.join(SRC_DIR, "camera_params_candidate.npz"))
SUMMARY_PATH = os.environ.get("CALIB_SUMMARY_PATH", os.path.join(SRC_DIR, "calibration_search_summary.json"))

CONFIG_GRID = [
    {"reprojection_threshold_px": threshold, "use_rational_model": use_rational, "fix_intrinsic": fix_intrinsic}
    for threshold in (0.35, 0.50, 0.75, 1.00)
    for use_rational in (False, True)
    for fix_intrinsic in (True, False)
]


def save_camera_params(path, params):
    np.savez(
        path,
        mtx_l=params["mtx_l"],
        dist_l=params["dist_l"],
        mtx_r=params["mtx_r"],
        dist_r=params["dist_r"],
        R=params["R"],
        T=params["T"],
        E=params["E"],
        F=params["F"],
    )


def load_camera_params(path):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "mtx_l": data["mtx_l"],
        "dist_l": data["dist_l"],
        "mtx_r": data["mtx_r"],
        "dist_r": data["dist_r"],
        "R": data["R"],
        "T": data["T"],
        "E": data["E"] if "E" in data.files else None,
        "F": data["F"] if "F" in data.files else None,
    }


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def print_validation(title, summary):
    metrics = summary["aggregate_mean_of_frame_metrics"]
    print(f"\n{title}")
    print("-" * len(title))
    print(
        "Vertical disparity mean / p95 (px): "
        f"{metrics['vertical_disparity_px_mean']['mean']:.2f} / {metrics['vertical_disparity_px_p95']['mean']:.2f}"
    )
    print(
        "Rigid alignment RMSE / plane RMS (cm): "
        f"{metrics['rigid_alignment_rmse_cm']['mean']:.3f} / {metrics['plane_rms_cm']['mean']:.3f}"
    )
    print(
        "Left / Right reprojection (px): "
        f"{metrics['left_reprojection_px']['mean']:.3f} / {metrics['right_reprojection_px']['mean']:.3f}"
    )
    print(f"Composite score: {score_validation_summary(summary):.2f}")


def main():
    print("🚀 Calibration optimization started")
    print(f"[Info] Using circle-grid clustering: {USE_CLUSTERING}")

    entries, image_size, per_pair_counts = detect_circle_grid_pairs(
        DATA_DIR,
        VIDEO_PAIRS,
        PATTERN_SIZE,
        SQUARE_SIZE_CM,
        use_clustering=USE_CLUSTERING,
    )
    if image_size is None or len(entries) < 12:
        raise RuntimeError("Not enough valid stereo circle-grid detections to calibrate")

    entries_by_pair = group_entries_by_pair(entries)
    print(f"[Info] Image size: {image_size[0]} x {image_size[1]}")
    print(f"[Info] Total detected stereo calibration frames: {len(entries)}")
    for pair_name, count in per_pair_counts.items():
        print(f"       {pair_name}: {count} frames")

    current_params = load_camera_params(OUTPUT_PATH)
    current_validation = None
    if current_params is not None:
        current_validation = evaluate_calibration(entries, image_size, current_params, PATTERN_SIZE, SQUARE_SIZE_CM)
        print_validation("Current camera_params.npz", current_validation)

    best_result, search_results = search_calibration_config(
        entries_by_pair,
        image_size,
        PATTERN_SIZE,
        SQUARE_SIZE_CM,
        CONFIG_GRID,
    )
    if best_result is None:
        raise RuntimeError("Calibration config search failed for all candidates")

    best_config = best_result["config"]
    print("\n[Info] Best cross-validated config:")
    print(
        f"       threshold={best_config['reprojection_threshold_px']:.2f}px, "
        f"rational={best_config['use_rational_model']}, "
        f"fix_intrinsic={best_config['fix_intrinsic']}"
    )
    print(f"       mean holdout score={best_result['mean_validation_score']:.2f}")
    print(
        f"       mean holdout disparity={best_result['mean_holdout_vertical_disparity_px']:.2f}px, "
        f"holdout rigid rmse={best_result['mean_holdout_rigid_rmse_cm']:.3f}cm"
    )

    final_params = calibrate_stereo_from_entries(
        entries,
        image_size,
        reprojection_threshold_px=best_config["reprojection_threshold_px"],
        use_rational_model=best_config["use_rational_model"],
        fix_intrinsic=best_config["fix_intrinsic"],
    )
    if final_params is None:
        raise RuntimeError("Final calibration failed with the selected config")

    final_validation = evaluate_calibration(entries, image_size, final_params, PATTERN_SIZE, SQUARE_SIZE_CM)
    print_validation("Candidate calibration (all pairs)", final_validation)

    save_camera_params(CANDIDATE_OUTPUT_PATH, final_params)
    print(f"[Info] Candidate parameters saved to: {CANDIDATE_OUTPUT_PATH}")

    current_score = score_validation_summary(current_validation) if current_validation is not None else float("inf")
    candidate_score = score_validation_summary(final_validation)
    promoted = False
    if PROMOTE_IF_IMPROVED and candidate_score < current_score:
        save_camera_params(OUTPUT_PATH, final_params)
        promoted = True
        print(f"[Info] Candidate promoted to active camera params: {OUTPUT_PATH}")
    elif current_validation is None and PROMOTE_IF_IMPROVED:
        save_camera_params(OUTPUT_PATH, final_params)
        promoted = True
        print(f"[Info] Candidate promoted to active camera params: {OUTPUT_PATH}")
    else:
        print("[Info] Active camera params left unchanged.")

    summary_payload = {
        "image_size": list(image_size),
        "use_clustering": USE_CLUSTERING,
        "detected_frames_per_pair": per_pair_counts,
        "total_detected_frames": len(entries),
        "config_grid_size": len(CONFIG_GRID),
        "best_crossval_result": best_result,
        "top_search_results": sorted(search_results, key=lambda item: item["mean_validation_score"])[:5],
        "current_validation": current_validation,
        "candidate_validation": final_validation,
        "current_score": current_score,
        "candidate_score": candidate_score,
        "candidate_kept_frames": int(final_params["kept_count"]),
        "candidate_total_frames": int(final_params["total_count"]),
        "candidate_output_path": CANDIDATE_OUTPUT_PATH,
        "active_output_path": OUTPUT_PATH,
        "promoted_to_active": promoted,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary_payload), f, indent=2)
    print(f"[Info] Search summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
