"""Evaluate one SKT postprocess variant against the Xsens-derived reference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.angles import (
    build_fair_angle_interpolators,
    build_native_angle_interpolators,
    sample_interpolators,
)
from common.config import get_run_dir, load_config, resolve_path, section
from common.metrics import jsonable, mae, median_abs_error, rula_bin
from estimate_offset import load_selected_offset
from eval_filter_ablation import _prepare_skt_variant, default_variants
from eval_motion_delta import k_delta, pair_metrics, threshold_for_k


def _variant_by_name(config: dict, variant_name: str) -> dict:
    """Resolve a configured or default filter-ablation variant by name."""
    variants = section(config, "filter_ablation").get("variants") or default_variants()
    for variant in variants:
        if str(variant.get("name")) == variant_name:
            return dict(variant)
    known = ", ".join(str(variant.get("name")) for variant in variants)
    raise ValueError(f"Unknown variant '{variant_name}'. Known variants: {known}")


def _load_xsens_reference(config: dict, time_s: np.ndarray, offset_s: float, angle_names: list[str]) -> dict[str, np.ndarray]:
    """Sample Xsens-derived reference angles on the video timeline."""
    refs = section(config, "references")
    fair_path = resolve_path(refs.get("xsens_fair_angles"), must_exist=False)
    fair_interps = build_fair_angle_interpolators(fair_path)
    if not fair_interps:
        fair_interps = build_native_angle_interpolators(resolve_path(refs.get("xsens_mvnx"), must_exist=True))
    return sample_interpolators(fair_interps, time_s - float(offset_s), angle_names)


def _write_angle_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """Write angle summary rows."""
    fieldnames = [
        "system",
        "reference",
        "angle",
        "valid_pair_count",
        "valid_ratio",
        "mae_deg",
        "median_abs_error_deg",
        "bias_deg",
        "rula_like_agreement",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_motion_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write flat motion summary rows."""
    fieldnames = [
        "system",
        "reference",
        "angle",
        "k",
        "valid_pair_count",
        "pearson_delta",
        "spearman_delta",
        "slope_target_vs_reference",
        "delta_mae_deg",
        "delta_rmse_deg",
        "active_pair_count",
        "active_delta_mae_deg",
        "active_delta_rmse_deg",
        "target_quiet_delta_std_deg",
        "target_path_deg",
        "reference_path_deg",
        "path_ratio_target_reference",
        "target_high_delta_count",
        "active_delta_threshold_deg",
        "noise_floor_threshold_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _angle_summary_rows(
    candidate_label: str,
    candidate_angles: dict[str, np.ndarray],
    xsens_angles: dict[str, np.ndarray],
    angle_names: list[str],
    rula_bins: dict[str, list[float]],
) -> list[dict[str, object]]:
    """Build angle-agreement rows against XsensFair."""
    rows: list[dict[str, object]] = []
    for angle_name in angle_names:
        target = candidate_angles[angle_name]
        reference = xsens_angles[angle_name]
        valid = np.isfinite(target) & np.isfinite(reference)
        bins = rula_bins.get(angle_name)
        agreement = None
        if bins and np.any(valid):
            agreement = float(np.mean(rula_bin(target[valid], bins) == rula_bin(reference[valid], bins)))
        rows.append({
            "system": candidate_label,
            "reference": "XsensFair",
            "angle": angle_name,
            "valid_pair_count": int(np.sum(valid)),
            "valid_ratio": float(np.mean(valid)) if len(valid) else 0.0,
            "mae_deg": mae(target, reference),
            "median_abs_error_deg": median_abs_error(target, reference),
            "bias_deg": float(np.nanmean(target[valid] - reference[valid])) if np.any(valid) else None,
            "rula_like_agreement": agreement,
        })
    return rows


def _motion_summary_rows(
    candidate_label: str,
    candidate_angles: dict[str, np.ndarray],
    xsens_angles: dict[str, np.ndarray],
    angle_names: list[str],
    k_list: list[int],
    eval_cfg: dict,
) -> list[dict[str, object]]:
    """Build K-frame motion-delta rows against XsensFair."""
    rows: list[dict[str, object]] = []
    for angle_name in angle_names:
        target = candidate_angles[angle_name]
        reference = xsens_angles[angle_name]
        for k in k_list:
            target_delta, _ = k_delta(target, k)
            ref_delta, _ = k_delta(reference, k)
            active = threshold_for_k(float(eval_cfg.get("active_delta_threshold_deg", 1.0)), k, "active")
            noise = threshold_for_k(float(eval_cfg.get("noise_floor_threshold_deg", 0.5)), k, "noise")
            metrics = pair_metrics(target_delta, ref_delta, active, noise)
            anomaly_threshold = threshold_for_k(float(eval_cfg.get("anomaly_delta_deg", 30.0)), k, "anomaly")
            anomaly = np.isfinite(target_delta) & (np.abs(target_delta) > anomaly_threshold)
            rows.append({
                "system": candidate_label,
                "reference": "XsensFair",
                "angle": angle_name,
                "k": k,
                **metrics,
                "target_high_delta_count": int(np.sum(anomaly)),
                "active_delta_threshold_deg": active,
                "noise_floor_threshold_deg": noise,
            })
    return rows


def evaluate_variant_regression(
    config: dict,
    run_dir: Path,
    variant_name: str,
    candidate_label: str,
) -> Path:
    """Run variant-aware angle and motion regression evaluation."""
    offset_s = load_selected_offset(run_dir)
    variant = _variant_by_name(config, variant_name)
    time_s, candidate_angles, meta = _prepare_skt_variant(config, run_dir, variant)
    eval_cfg = section(config, "evaluation")
    angle_names = [str(name) for name in eval_cfg.get("angle_names", list(candidate_angles.keys())) if name in candidate_angles]
    xsens_angles = _load_xsens_reference(config, time_s, offset_s, angle_names)
    k_list = [int(k) for k in eval_cfg.get("k_frame_list", [1, 6, 12, 25])]

    angle_rows = _angle_summary_rows(
        candidate_label,
        candidate_angles,
        xsens_angles,
        angle_names,
        eval_cfg.get("rula_bins", {}),
    )
    motion_rows = _motion_summary_rows(
        candidate_label,
        candidate_angles,
        xsens_angles,
        angle_names,
        k_list,
        eval_cfg,
    )

    out_dir = run_dir / "variant_regression" / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    angle_csv = out_dir / "angle_summary.csv"
    motion_csv = out_dir / "motion_delta_summary.csv"
    _write_angle_summary(angle_csv, angle_rows)
    _write_motion_summary_csv(motion_csv, motion_rows)

    summary = {
        "config": {
            "candidate_label": candidate_label,
            "variant_name": variant_name,
            "variant": variant,
            "selected_offset_seconds": offset_s,
            "reference": "XsensFair (Xsens-derived comparison/reference system)",
            "angle_names": angle_names,
            "k_frame_list": k_list,
            "postprocess_meta": meta,
        },
        "angle_rows": angle_rows,
        "motion_rows": motion_rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[variant_regression] saved {summary_path}")
    return summary_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--variant-name", default="quality_aware_repair_keypoint_savgol")
    parser.add_argument("--candidate-label", default="SKT_quality_aware_repair")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = args.run_dir if args.run_dir is not None else get_run_dir(config)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    evaluate_variant_regression(config, run_dir, args.variant_name, args.candidate_label)


if __name__ == "__main__":
    main()
