#!/opt/anaconda3/envs/pose/bin/python
"""One-Euro Filter parameter sweep on SKT 3D keypoints.

For each (min_cutoff, beta) combination, applies One-Euro to SKT, runs the
Phase 4 frame-delta + segment-ROM evaluation, and extracts headline numbers.
Writes a single summary Markdown for quick comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SIDES = ("LeftElbow", "RightElbow")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-npz",
        default=str(
            PROJECT_ROOT
            / "01_stereo_triangulation"
            / "results"
            / "historical_best_20260324"
            / "recovered_baseline"
            / "optimized_pose.npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "04_frame_delta_eval" / "results" / "phase4_one_euro_sweep"),
    )
    parser.add_argument(
        "--min-cutoffs",
        default="0.5,1.0,2.0,3.0",
        help="Comma-separated One-Euro min_cutoff values (Hz).",
    )
    parser.add_argument(
        "--betas",
        default="0.02,0.05,0.1,0.5",
        help="Comma-separated One-Euro beta values.",
    )
    parser.add_argument("--d-cutoff", type=float, default=1.0)
    parser.add_argument(
        "--fastsam-trc",
        default=str(PROJECT_ROOT.parent / "10 Aitor" / "fastsam3d_2.trc"),
    )
    parser.add_argument(
        "--merge-trc",
        default=str(PROJECT_ROOT.parent / "10 Aitor" / "merged_output_2.trc"),
    )
    parser.add_argument(
        "--motionbert-trc",
        default=str(PROJECT_ROOT / "shared" / "recovered_methods" / "motionbert_markers_results_mono.trc"),
    )
    return parser.parse_args()


def parse_list(raw: str, cast=float) -> List:
    """Parse comma-separated list."""
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def run(cmd: List[str]) -> None:
    """Run subprocess and stream output to stdout."""
    print("[run]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def extract_metrics(summary_json: Path, k: int) -> Dict[str, Dict]:
    """Pull SKT vs XsensFair K-delta metrics from elbow_delta_summary.json."""
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    out: Dict[str, Dict] = {}
    for side in SIDES:
        pair = data["motion_agreement"][side]["SKT_vs_XsensFair"][f"k{k}"]
        out[side] = {
            "pearson": pair["pearson_delta"],
            "active_pearson": pair["active_pearson_delta"],
            "path_ratio": pair["path_ratio_target_reference"],
            "slope": pair["slope_target_vs_reference"],
            "quiet_std": pair["target_quiet_delta_std_deg"],
        }
    return out


def extract_segment_metrics(seg_summary_json: Path) -> Dict[str, Dict]:
    """Pull SKT ROM + DTW + RULA from segment_rom_summary.json."""
    data = json.loads(seg_summary_json.read_text(encoding="utf-8"))
    out: Dict[str, Dict] = {}
    for side in SIDES:
        rom = data["rom_agreement"][side]["SKT_vs_XsensFair"]
        dtw = data["dtw_shape_agreement"][side]["SKT_vs_XsensFair"]
        rula = data["rula_bin_agreement"][side]["SKT_vs_XsensFair"]
        out[side] = {
            "rom_mae": rom["rom_mae_deg"],
            "rom_pearson": rom["pearson_rom"],
            "dtw_median": dtw["median"],
            "rula": rula["agreement_rate"],
        }
    return out


def main() -> None:
    """Run parameter sweep."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_cutoffs = parse_list(args.min_cutoffs, float)
    betas = parse_list(args.betas, float)

    one_euro_script = SCRIPT_DIR / "06_apply_one_euro.py"
    compute_script = SCRIPT_DIR / "01_compute_elbow_deltas.py"
    segment_script = SCRIPT_DIR / "03_segment_rom_eval.py"

    rows: List[Dict] = []
    for min_cutoff in min_cutoffs:
        for beta in betas:
            tag = f"mc{min_cutoff:g}_b{beta:g}".replace(".", "p")
            print(f"\n=== Sweep: min_cutoff={min_cutoff}, beta={beta} ===")
            npz_path = out_dir / f"skt_{tag}.npz"
            delta_dir = out_dir / f"eval_{tag}"
            segment_dir = out_dir / f"segment_{tag}"

            run([
                sys.executable,
                str(one_euro_script),
                "--input-npz", args.input_npz,
                "--output-npz", str(npz_path),
                "--min-cutoff", str(min_cutoff),
                "--beta", str(beta),
                "--d-cutoff", str(args.d_cutoff),
            ])

            run([
                sys.executable,
                str(compute_script),
                "--skt-npz", str(npz_path),
                "--skip-afh",
                "--fastsam-trc", args.fastsam_trc,
                "--merge-trc", args.merge_trc,
                "--extra-trc", f"MotionBert={args.motionbert_trc}",
                "--enable-quality-filter",
                "--smooth-method", "moving_average",
                "--smooth-window-ms", "200",
                "--wrist-smooth-radius", "0",
                "--out-dir", str(delta_dir),
                "--skip-plots",
            ])

            run([
                sys.executable,
                str(segment_script),
                "--combined-csv", str(delta_dir / "elbow_delta_combined.csv"),
                "--out-dir", str(segment_dir),
                "--skip-plots",
            ])

            k6 = extract_metrics(delta_dir / "elbow_delta_summary.json", 6)
            seg = extract_segment_metrics(segment_dir / "segment_rom_summary.json")
            for side in SIDES:
                rows.append({
                    "min_cutoff": min_cutoff,
                    "beta": beta,
                    "side": side,
                    **k6[side],
                    **seg[side],
                })

    rows.sort(key=lambda r: (r["side"], r["min_cutoff"], r["beta"]))

    lines = [
        "# One-Euro Filter sweep — SKT vs XsensFair (K=6, segment metrics)",
        "",
        f"Input: `{args.input_npz}`",
        "",
        "| min_cutoff | beta | Side | K6 Pearson | active Pearson | path_ratio | quiet std (°/fr) | ROM MAE (°) | DTW median | RULA |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value):
            if value is None:
                return "NA"
            return f"{float(value):.3f}"
        lines.append(
            "| "
            + " | ".join([
                f"{row['min_cutoff']:g}",
                f"{row['beta']:g}",
                row["side"],
                fmt(row["pearson"]),
                fmt(row["active_pearson"]),
                fmt(row["path_ratio"]),
                fmt(row["quiet_std"]),
                fmt(row["rom_mae"]),
                fmt(row["dtw_median"]),
                fmt(row["rula"]),
            ])
            + " |"
        )
    (out_dir / "headline_one_euro_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[saved] {out_dir / 'headline_one_euro_sweep.md'}")


if __name__ == "__main__":
    main()
