#!/opt/anaconda3/envs/pose/bin/python
"""Run Phase 4 sensitivity checks for filtering, segmentation, and DTW."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = METHOD_DIR / "results" / "phase4_ablation"
SIDES = ("LeftElbow", "RightElbow")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--windows-ms", default="100,200,300")
    parser.add_argument("--activity-thresholds", default="8,10,12")
    parser.add_argument("--dtw-preprocesses", default="mean_l2,mean,none")
    parser.add_argument("--wrist-smooth-radius", type=int, default=0)
    parser.add_argument("--skip-afh", action="store_true",
                        help="Exclude the legacy AFH NPZ from all ablation runs.")
    parser.add_argument("--afh-filter-status",
                        choices=("unknown_butterworth", "unfiltered", "not_included"),
                        default="unknown_butterworth")
    parser.add_argument("--fastsam-trc", default=None,
                        help="Optional unfiltered FastSAM3D TRC file to include in all runs.")
    parser.add_argument("--merge-trc", default=None,
                        help="Optional Merge TRC file to include in all runs.")
    parser.add_argument("--extra-trc", action="append", default=[],
                        help="Additional TRC source as NAME=PATH. May be supplied multiple times.")
    parser.add_argument("--enable-quality-filter", action="store_true",
                        help="Apply the same SKT quality filter as the main evaluation.")
    return parser.parse_args()


def parse_list(raw: str, cast=float) -> List:
    """Parse comma-separated CLI lists."""
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def run_command(cmd: List[str]) -> None:
    """Run one subprocess with a concise progress line."""
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> Dict:
    """Load JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(summary: Dict, side: str, pair: str, metric: str):
    """Extract one headline metric from a segment summary."""
    if metric == "rom_mae":
        return summary["rom_agreement"][side][pair]["rom_mae_deg"]
    if metric == "dtw_median":
        return summary["dtw_shape_agreement"][side][pair]["median"]
    if metric == "rula":
        return summary["rula_bin_agreement"][side][pair]["agreement_rate"]
    raise KeyError(metric)


def fmt(value) -> str:
    """Format optional numeric Markdown table values."""
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def append_metric_rows(lines: List[str], label: str, summary: Dict, extra: str = "") -> None:
    """Append pair/side rows from one segment summary."""
    for side in SIDES:
        for pair in summary["rom_agreement"][side]:
            lines.append(
                "| "
                + " | ".join([
                    label,
                    extra,
                    side,
                    pair,
                    fmt(metric_value(summary, side, pair, "rom_mae")),
                    fmt(metric_value(summary, side, pair, "dtw_median")),
                    fmt(metric_value(summary, side, pair, "rula")),
                ])
                + " |"
            )


def write_markdown(
    out_dir: Path,
    window_summaries: Dict[str, Dict],
    activity_summaries: Dict[str, Dict],
    dtw_summaries: Dict[str, Dict],
    base_combined_csv: Path,
) -> None:
    """Write compact Phase 4 sensitivity tables."""
    lines: List[str] = [
        "# Phase 4 Sensitivity Summary",
        "",
        "Rows are generated dynamically from the systems present in the combined CSV.",
        "",
        f"- Base combined CSV: `{base_combined_csv}`",
        "",
        "## Filter Window Sweep",
        "",
        "| Window | Extra | Side | Pair | ROM MAE (deg) | DTW median | RULA agreement |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for window, summary in window_summaries.items():
        append_metric_rows(lines, f"{window} ms", summary)

    lines.extend([
        "",
        "## Activity Threshold Sweep",
        "",
        "| Threshold | Extra | Side | Pair | ROM MAE (deg) | DTW median | RULA agreement |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for threshold, summary in activity_summaries.items():
        count_extra = (
            f"L={summary['segments_count']['LeftElbow']}, "
            f"R={summary['segments_count']['RightElbow']}"
        )
        append_metric_rows(lines, f"{threshold} deg", summary, count_extra)

    lines.extend([
        "",
        "## DTW Preprocess Sweep",
        "",
        "| Preprocess | Extra | Side | Pair | ROM MAE (deg) | DTW median | RULA agreement |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for preprocess, summary in dtw_summaries.items():
        append_metric_rows(lines, preprocess, summary)

    (out_dir / "headline_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run Phase 4 ablation package."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_ms = parse_list(args.windows_ms, float)
    activity_thresholds = parse_list(args.activity_thresholds, float)
    preprocesses = parse_list(args.dtw_preprocesses, str)

    compute_script = SCRIPT_DIR / "01_compute_elbow_deltas.py"
    segment_script = SCRIPT_DIR / "03_segment_rom_eval.py"
    compute_common_args: List[str] = [
        "--wrist-smooth-radius",
        str(args.wrist_smooth_radius),
        "--afh-filter-status",
        "not_included" if args.skip_afh else args.afh_filter_status,
    ]
    if args.skip_afh:
        compute_common_args.append("--skip-afh")
    if args.enable_quality_filter:
        compute_common_args.append("--enable-quality-filter")
    if args.fastsam_trc:
        compute_common_args.extend(["--fastsam-trc", args.fastsam_trc])
    if args.merge_trc:
        compute_common_args.extend(["--merge-trc", args.merge_trc])
    for extra_trc in args.extra_trc:
        compute_common_args.extend(["--extra-trc", extra_trc])

    window_summaries: Dict[str, Dict] = {}
    base_combined_csv: Path | None = None
    for window_ms in windows_ms:
        label = f"ma{int(window_ms)}"
        delta_dir = out_dir / f"delta_{label}"
        segment_dir = out_dir / f"segment_{label}"
        run_command([
            sys.executable,
            str(compute_script),
            "--smooth-method",
            "moving_average",
            "--smooth-window-ms",
            str(window_ms),
            *compute_common_args,
            "--out-dir",
            str(delta_dir),
            "--skip-plots",
        ])
        run_command([
            sys.executable,
            str(segment_script),
            "--combined-csv",
            str(delta_dir / "elbow_delta_combined.csv"),
            "--out-dir",
            str(segment_dir),
            "--skip-plots",
        ])
        window_summaries[str(int(window_ms))] = load_json(segment_dir / "segment_rom_summary.json")
        if int(window_ms) == 200:
            base_combined_csv = delta_dir / "elbow_delta_combined.csv"

    if base_combined_csv is None:
        base_combined_csv = out_dir / f"delta_ma{int(windows_ms[0])}" / "elbow_delta_combined.csv"

    activity_summaries: Dict[str, Dict] = {}
    for threshold in activity_thresholds:
        segment_dir = out_dir / f"activity_{int(threshold)}"
        run_command([
            sys.executable,
            str(segment_script),
            "--combined-csv",
            str(base_combined_csv),
            "--activity-threshold-deg",
            str(threshold),
            "--out-dir",
            str(segment_dir),
            "--skip-plots",
        ])
        activity_summaries[str(int(threshold))] = load_json(segment_dir / "segment_rom_summary.json")

    dtw_summaries: Dict[str, Dict] = {}
    for preprocess in preprocesses:
        segment_dir = out_dir / f"dtw_{preprocess}"
        run_command([
            sys.executable,
            str(segment_script),
            "--combined-csv",
            str(base_combined_csv),
            "--dtw-preprocess",
            preprocess,
            "--out-dir",
            str(segment_dir),
            "--skip-plots",
        ])
        dtw_summaries[preprocess] = load_json(segment_dir / "segment_rom_summary.json")

    write_markdown(
        out_dir=out_dir,
        window_summaries=window_summaries,
        activity_summaries=activity_summaries,
        dtw_summaries=dtw_summaries,
        base_combined_csv=base_combined_csv,
    )
    print("[saved]", out_dir / "headline_table.md")


if __name__ == "__main__":
    main()
