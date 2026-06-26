"""Export a selected SKT v2 candidate as a reviewable angle time series."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.config import load_config, section
from common.dataset import build_pose_timeline, load_skt_keypoints
from common.metrics import jsonable
from eval_filter_ablation import _load_fastsam_angles, _prepare_skt_variant, default_variants
from eval_vs_fastsam import build_rows


def _force_run_npz(config: dict) -> dict:
    """Return a config copy that loads SKT output from the supplied run directory."""
    out = deepcopy(config)
    out.setdefault("skt", {})["use_existing_npz"] = False
    return out


def _variant_by_name(config: dict, variant_name: str) -> dict:
    """Resolve one filter-ablation variant by name."""
    variants = section(config, "filter_ablation").get("variants") or default_variants()
    for variant in variants:
        if str(variant.get("name")) == variant_name:
            return dict(variant)
    known = ", ".join(str(v.get("name")) for v in variants)
    raise ValueError(f"Unknown variant '{variant_name}'. Known variants: {known}")


def _write_timeseries_csv(
    path: Path,
    time_s: np.ndarray,
    candidate: np.ndarray,
    fastsam: np.ndarray,
    candidate_label: str,
    angle_name: str,
) -> None:
    """Write candidate and FastSAM3D angle traces to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Frame",
                "Time_s",
                f"{candidate_label}_{angle_name}_deg",
                f"FastSAM3D_{angle_name}_deg",
                "AbsError_deg",
            ],
        )
        writer.writeheader()
        for idx, (time_value, cand_value, ref_value) in enumerate(zip(time_s, candidate, fastsam, strict=False)):
            finite_pair = np.isfinite(cand_value) and np.isfinite(ref_value)
            writer.writerow({
                "Frame": idx,
                "Time_s": f"{float(time_value):.6f}",
                f"{candidate_label}_{angle_name}_deg": f"{float(cand_value):.6f}" if np.isfinite(cand_value) else "",
                f"FastSAM3D_{angle_name}_deg": f"{float(ref_value):.6f}" if np.isfinite(ref_value) else "",
                "AbsError_deg": f"{abs(float(cand_value - ref_value)):.6f}" if finite_pair else "",
            })


def _plot_timeseries(
    path: Path,
    time_s: np.ndarray,
    candidate: np.ndarray,
    fastsam: np.ndarray,
    summary_row: dict,
    candidate_label: str,
    angle_name: str,
) -> None:
    """Plot candidate and FastSAM3D right-elbow angle traces."""
    valid = np.isfinite(candidate) & np.isfinite(fastsam)
    if np.any(valid):
        valid_idx = np.where(valid)[0]
        start_idx = int(valid_idx[0])
        end_idx = int(valid_idx[-1]) + 1
    else:
        start_idx = 0
        end_idx = len(time_s)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(
        time_s[start_idx:end_idx],
        fastsam[start_idx:end_idx],
        color="#3274b9",
        linewidth=2.0,
        label="FastSAM3D reference",
    )
    ax.plot(
        time_s[start_idx:end_idx],
        candidate[start_idx:end_idx],
        color="#d44f4f",
        linewidth=1.7,
        label=candidate_label,
    )
    if start_idx < end_idx:
        window_valid = valid[start_idx:end_idx]
        if not np.all(window_valid):
            y_min, y_max = ax.get_ylim()
            ax.fill_between(
                time_s[start_idx:end_idx],
                y_min,
                y_max,
                where=~window_valid,
                color="#b0b0b0",
                alpha=0.12,
                label="non-overlap / invalid",
            )

    mae = summary_row.get("mae_deg")
    rms = summary_row.get("target_angular_acc_rms_deg_s2")
    jumps = summary_row.get("target_jump_count_full_timeline")
    valid_ratio = summary_row.get("valid_ratio")
    subtitle = (
        f"MAE={mae:.2f} deg | valid={100.0 * valid_ratio:.1f}% | "
        f"acc. RMS={rms:.1f} deg/s^2 | jumps>{summary_row.get('jump_threshold_deg'):.0f} deg: {jumps}"
        if all(value is not None for value in [mae, rms, jumps, valid_ratio])
        else "Candidate angle trace"
    )
    ax.set_title(f"{angle_name} angle over time: {candidate_label} vs FastSAM3D\n{subtitle}", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{angle_name} angle (deg)")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def export_candidate_timeseries(
    config: dict,
    run_dir: Path,
    variant_name: str,
    candidate_label: str,
) -> Path:
    """Export summary, CSV, and plot for one candidate variant."""
    run_config = _force_run_npz(config)
    variant = _variant_by_name(run_config, variant_name)
    skt_path, raw_keypoints, _ = load_skt_keypoints(run_config, run_dir)
    time_s, synced, left_rows, _ = build_pose_timeline(run_config, len(raw_keypoints))
    fast_angles = _load_fastsam_angles(run_config, time_s, synced, left_rows)
    _, candidate_angles, meta = _prepare_skt_variant(run_config, run_dir, variant)

    angle_names = [str(name) for name in section(run_config, "evaluation").get("angle_names", ["RightElbow"])]
    if len(angle_names) != 1:
        raise ValueError("Candidate export currently expects one angle name.")
    angle_name = angle_names[0]

    rows = build_rows(
        time_s=time_s,
        all_angles={"FastSAM3D": fast_angles, candidate_label: candidate_angles},
        angle_names=angle_names,
        targets=[candidate_label],
        rula_bins=section(run_config, "evaluation").get("rula_bins", {}),
        jump_threshold_deg=float(section(run_config, "filter_ablation").get("jump_threshold_deg", 10.0)),
    )
    summary_row = rows[0] if rows else {}

    out_dir = run_dir / "candidate_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_variant = variant_name.replace("/", "_")
    csv_path = out_dir / f"{safe_variant}_timeseries.csv"
    plot_path = out_dir / f"{safe_variant}_right_elbow_timeseries.png"
    summary_path = out_dir / f"{safe_variant}_summary.json"

    _write_timeseries_csv(
        csv_path,
        time_s,
        candidate_angles[angle_name],
        fast_angles[angle_name],
        candidate_label,
        angle_name,
    )
    _plot_timeseries(
        plot_path,
        time_s,
        candidate_angles[angle_name],
        fast_angles[angle_name],
        summary_row,
        candidate_label,
        angle_name,
    )

    summary = {
        "candidate_label": candidate_label,
        "variant_name": variant_name,
        "variant": variant,
        "skt_npz": str(skt_path),
        "angle_name": angle_name,
        "postprocess_meta": meta,
        "summary_row": summary_row,
        "outputs": {
            "timeseries_csv": str(csv_path),
            "plot_png": str(plot_path),
        },
    }
    summary_path.write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[candidate_export] saved {summary_path}")
    print(f"[candidate_export] saved {plot_path}")
    return summary_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant-name", default="hard_filter_keypoint_savgol")
    parser.add_argument("--candidate-label", default="YOLO11l_SKT_keypoint_savgol")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    export_candidate_timeseries(config, run_dir, args.variant_name, args.candidate_label)


if __name__ == "__main__":
    main()
