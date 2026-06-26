"""Stereo calibration for the 2026 Assar dataset (A255 and A257 sensor pairs).

Uses the same circle-grid + cv2.stereoCalibrate pipeline as 01_calibration.py, but:
- Discovers calibration video pairs automatically from
  ``2026_Assar_Data/<location>/SensorCalibration`` and ``SiteCalibration``.
- Treats ``cap_X_0.avi`` as the "left" image and ``cap_X_1.avi`` as the "right"
  image — this is a naming convention only; the resulting T tells us the actual
  baseline direction.
- Writes ``camera_params.npz`` (and ``camera_params_candidate.npz`` +
  ``calibration_search_summary.json``) into each location folder so each pair
  has its own intrinsics/extrinsics.

Run::

    /opt/anaconda3/envs/pose/bin/python 01_stereo_triangulation/src/01_calibration_2026.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from calibration_utils import (  # noqa: E402
    calibrate_stereo_from_entries,
    detect_circle_grid_pairs,
    evaluate_calibration,
    group_entries_by_pair,
    score_validation_summary,
    search_calibration_config,
)

PATTERN_SIZE = (5, 9)
SQUARE_SIZE_CM = 15.0
# A255 prefers clustering ON (the SensorCalibration captures benefit from de-duping).
# A257 prefers clustering OFF (SiteCalibration has many distinct poses that clustering drops).
# The current saved A257 result was produced with CALIB_USE_CLUSTERING=0 — set that env var when re-running A257.
USE_CLUSTERING = os.environ.get("CALIB_USE_CLUSTERING", "1") == "1"

CONFIG_GRID = [
    {"reprojection_threshold_px": threshold, "use_rational_model": use_rational, "fix_intrinsic": fix_intrinsic}
    for threshold in (0.35, 0.50, 0.75, 1.00)
    for use_rational in (False, True)
    for fix_intrinsic in (True, False)
]


_MIN_SENSOR_STEREO_PAIRS = 12


def _collect_video_pairs(sub_dir: Path, location_dir: Path) -> list[tuple[str, str, str, str]]:
    """List valid (left, right, left_txt, right_txt) relative paths under sub_dir."""
    pairs = []
    for left_avi in sorted(sub_dir.glob("cap_*_0.avi")):
        stem = left_avi.stem
        right_avi = sub_dir / f"{stem[:-2]}_1.avi"
        left_txt = sub_dir / f"{stem}.txt"
        right_txt = sub_dir / f"{stem[:-2]}_1.txt"
        if not (right_avi.exists() and left_txt.exists() and right_txt.exists()):
            continue
        pairs.append((
            str(left_avi.relative_to(location_dir)),
            str(right_avi.relative_to(location_dir)),
            str(left_txt.relative_to(location_dir)),
            str(right_txt.relative_to(location_dir)),
        ))
    return pairs


def discover_pairs(location_dir: Path, pattern_size: tuple, use_clustering: bool) -> list[tuple[str, str, str, str]]:
    """Find calibration video pairs under a location.

    Runs the actual stereo detection on SensorCalibration first.  If the
    simultaneous-detection count reaches _MIN_SENSOR_STEREO_PAIRS, SensorCalibration
    is used alone (SiteCalibration data can add noise when Sensor data is rich).
    Otherwise SiteCalibration is included as supplemental data to reach a stable count.

    Returns paths relative to ``location_dir``.
    """
    sensor_dir = location_dir / "SensorCalibration"
    sensor_pairs = _collect_video_pairs(sensor_dir, location_dir) if sensor_dir.exists() else []

    # Detect actual simultaneous stereo pairs to decide whether we need supplemental data.
    if sensor_pairs:
        sensor_entries, _, _ = detect_circle_grid_pairs(
            str(location_dir), sensor_pairs, pattern_size, SQUARE_SIZE_CM,
            use_clustering=use_clustering,
        )
        sensor_stereo_count = len(sensor_entries)
        print(f"[Info] SensorCalibration: {sensor_stereo_count} simultaneous stereo pairs detected.")
    else:
        sensor_stereo_count = 0

    if sensor_stereo_count >= _MIN_SENSOR_STEREO_PAIRS:
        print(f"[Info] Sufficient sensor data — using SensorCalibration only.")
        return sensor_pairs

    print(
        f"[Warn] SensorCalibration only yielded {sensor_stereo_count} stereo pairs "
        f"(< {_MIN_SENSOR_STEREO_PAIRS}); adding SiteCalibration as supplemental data."
    )
    site_dir = location_dir / "SiteCalibration"
    site_pairs = _collect_video_pairs(site_dir, location_dir) if site_dir.exists() else []
    return sensor_pairs + site_pairs


def save_camera_params(path: Path, params: dict) -> None:
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


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def print_validation(title: str, summary: dict) -> None:
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


def calibrate_location(location_dir: Path) -> dict | None:
    """Calibrate one stereo location; returns the search summary payload."""
    print(f"\n{'=' * 70}\n📍 Calibrating {location_dir.name}\n{'=' * 70}")

    pairs = discover_pairs(location_dir, PATTERN_SIZE, USE_CLUSTERING)
    if not pairs:
        print(f"[Error] No calibration video pairs found under {location_dir}")
        return None
    print(f"[Info] Found {len(pairs)} calibration video pair(s):")
    for left, right, _, _ in pairs:
        print(f"       {left}  <->  {right}")

    entries, image_size, per_pair_counts = detect_circle_grid_pairs(
        str(location_dir),
        pairs,
        PATTERN_SIZE,
        SQUARE_SIZE_CM,
        use_clustering=USE_CLUSTERING,
    )
    if image_size is None or len(entries) < 8:
        print(f"[Error] Not enough valid stereo detections (got {len(entries)}); need >= 8")
        return None
    if len(entries) < 12:
        print(f"[Warn] Only {len(entries)} valid stereo pairs — calibration may be less robust (ideally >= 12)")

    entries_by_pair = group_entries_by_pair(entries)
    print(f"[Info] Image size: {image_size[0]} x {image_size[1]}")
    print(f"[Info] Total detected stereo calibration frames: {len(entries)}")
    for pair_name, count in per_pair_counts.items():
        print(f"       {pair_name}: {count} frames")

    # Cross-validated config search needs >= 2 pairs with enough entries each.
    # Drop "thin" pairs (< 8 entries) from CV since they break the leave-one-out training.
    cv_eligible = {name: entries_list for name, entries_list in entries_by_pair.items() if len(entries_list) >= 8}
    best_result = None
    search_results: list[dict] = []
    best_config: dict | None = None
    if len(cv_eligible) >= 2:
        best_result, search_results = search_calibration_config(
            cv_eligible, image_size, PATTERN_SIZE, SQUARE_SIZE_CM, CONFIG_GRID,
        )
        if best_result is not None:
            best_config = best_result["config"]
            print("\n[Info] Best cross-validated config:")
            print(
                f"       threshold={best_config['reprojection_threshold_px']:.2f}px, "
                f"rational={best_config['use_rational_model']}, "
                f"fix_intrinsic={best_config['fix_intrinsic']}"
            )
            print(f"       mean holdout score={best_result['mean_validation_score']:.2f}")
    if best_config is None:
        # Fallback when CV is impossible (single pair, or all pairs too thin).
        # Pick a reasonable default; the final calibration below still uses ALL entries.
        thin = {name: len(rows) for name, rows in entries_by_pair.items() if len(rows) < 8}
        if thin:
            print(f"[Warn] Pair(s) too thin for cross-validation: {thin}; using default config.")
        else:
            print("[Warn] Cross-validation unavailable; using default config.")
        best_config = {"reprojection_threshold_px": 0.75, "use_rational_model": True, "fix_intrinsic": False}

    final_params = calibrate_stereo_from_entries(
        entries,
        image_size,
        reprojection_threshold_px=best_config["reprojection_threshold_px"],
        use_rational_model=best_config["use_rational_model"],
        fix_intrinsic=best_config["fix_intrinsic"],
    )
    if final_params is None:
        print("[Error] Final calibration failed")
        return None

    final_validation = evaluate_calibration(entries, image_size, final_params, PATTERN_SIZE, SQUARE_SIZE_CM)
    print_validation(f"Calibration result for {location_dir.name}", final_validation)

    print("\n[Info] Stereo geometry:")
    print(f"       fx (left, right):    {final_params['mtx_l'][0,0]:.1f}, {final_params['mtx_r'][0,0]:.1f} px")
    print(f"       fy (left, right):    {final_params['mtx_l'][1,1]:.1f}, {final_params['mtx_r'][1,1]:.1f} px")
    print(f"       cx (left, right):    {final_params['mtx_l'][0,2]:.1f}, {final_params['mtx_r'][0,2]:.1f} px")
    print(f"       cy (left, right):    {final_params['mtx_l'][1,2]:.1f}, {final_params['mtx_r'][1,2]:.1f} px")
    print(f"       Baseline T (cm):     {final_params['T'].ravel().tolist()}")
    print(f"       |T| (cm):            {float(np.linalg.norm(final_params['T'])):.3f}")
    rot = final_params["R"]
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
    print(f"       Rotation angle (°):  {angle_deg:.3f}")

    shared_dir = PROJECT_ROOT / "shared"
    loc = location_dir.name  # e.g. "A255"
    output_path = shared_dir / f"camera_params_{loc}.npz"
    candidate_path = shared_dir / f"camera_params_{loc}_candidate.npz"
    summary_path = shared_dir / f"calibration_search_summary_{loc}.json"

    save_camera_params(candidate_path, final_params)
    save_camera_params(output_path, final_params)
    print(f"\n[Info] Saved: {output_path}")
    print(f"[Info] Saved: {candidate_path}")

    summary_payload = {
        "location": location_dir.name,
        "image_size": list(image_size),
        "use_clustering": USE_CLUSTERING,
        "detected_frames_per_pair": per_pair_counts,
        "total_detected_frames": len(entries),
        "config_grid_size": len(CONFIG_GRID),
        "best_crossval_result": best_result,
        "top_search_results": sorted(search_results, key=lambda item: item["mean_validation_score"])[:5] if search_results else [],
        "candidate_validation": final_validation,
        "candidate_score": score_validation_summary(final_validation),
        "candidate_kept_frames": int(final_params["kept_count"]),
        "candidate_total_frames": int(final_params["total_count"]),
        "output_path": str(output_path),
        "selected_config": best_config,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary_payload), f, indent=2)
    print(f"[Info] Summary saved: {summary_path}")
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "2026_Assar_Data",
        help="Root folder containing A255 / A257 subdirectories.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=["A255", "A257"],
        help="Location subfolders to calibrate.",
    )
    args = parser.parse_args()

    for loc in args.locations:
        location_dir = args.dataset_root / loc
        if not location_dir.exists():
            print(f"[Skip] {location_dir} not found")
            continue
        calibrate_location(location_dir)


if __name__ == "__main__":
    main()
