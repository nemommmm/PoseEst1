#!/opt/anaconda3/envs/pose/bin/python
"""Evaluate activity-segment elbow ROM agreement on an existing combined CSV.

This script uses XsensFair elbow motion to detect activity segments, then
compares per-segment ROM, peak angle, and RULA-like elbow bins across systems.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

SIDES = ("LeftElbow", "RightElbow")
SYSTEMS = ("SKT", "AFH", "XsensFair", "XsensNative")
TARGET_SYSTEMS = ("SKT", "AFH", "XsensNative")
COLORS = {
    "SKT": "#ff7a18",
    "AFH": "#2196F3",
    "XsensFair": "#43a047",
    "XsensNative": "#70757f",
}
LABELS = {
    "SKT": "01 SKT stereo",
    "AFH": "04 AFH hybrid",
    "XsensFair": "Xsens-derived reference",
    "XsensNative": "Xsens native",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--activity-lag-frames", type=int, default=12,
                        help="K-frame spacing used to detect XsensFair activity.")
    parser.add_argument("--activity-threshold-deg", type=float, default=10.0)
    parser.add_argument("--min-duration-s", type=float, default=1.5)
    parser.add_argument("--min-xsens-rom-deg", type=float, default=15.0)
    parser.add_argument("--merge-gap-s", type=float, default=2.0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--rula-bins", default="60,100",
                        help="Comma-separated elbow-angle bin thresholds.")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def as_float(value: str) -> float:
    """Convert CSV string to float, preserving blanks as NaN."""
    if value is None or value == "":
        return np.nan
    return float(value)


def load_combined_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read the combined elbow CSV into arrays."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    columns: Dict[str, list] = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            columns[key].append(value)

    data: Dict[str, np.ndarray] = {}
    for key, values in columns.items():
        if (
            key.endswith("_valid")
            or "_valid_k" in key
            or key.endswith("_interpolated")
            or key.endswith("_delta_anomaly_flag")
            or "_delta_anomaly_flag_k" in key
        ):
            data[key] = np.array([v == "True" for v in values], dtype=bool)
        elif key in {"Frame", "StereoFrameId", "LeftVideoFrame", "RightVideoFrame"}:
            data[key] = np.array([int(v) for v in values], dtype=np.int64)
        else:
            data[key] = np.array([as_float(v) for v in values], dtype=np.float64)
    return data


def parse_bins(raw: str) -> List[float]:
    """Parse RULA-like bin thresholds."""
    bins = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if sorted(bins) != bins:
        raise ValueError("--rula-bins must be sorted ascending.")
    return bins


def finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite paired arrays."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Pearson correlation with small-sample guardrails."""
    x_f, y_f = finite_pair(x, y)
    if len(x_f) < 3 or np.nanstd(x_f) < 1e-9 or np.nanstd(y_f) < 1e-9:
        return None
    return float(np.corrcoef(x_f, y_f)[0, 1])


def mean_abs_diff(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Return finite paired MAE."""
    x_f, y_f = finite_pair(x, y)
    if len(x_f) == 0:
        return None
    return float(np.mean(np.abs(x_f - y_f)))


def rounded(value):
    """Round JSON floats recursively."""
    if isinstance(value, dict):
        return {key: rounded(val) for key, val in value.items()}
    if isinstance(value, list):
        return [rounded(val) for val in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    return value


def find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive index runs where mask is true."""
    runs: List[Tuple[int, int]] = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        runs.append((start, idx - 1))
    return runs


def merge_segments(segments: List[Tuple[int, int]], time_s: np.ndarray, merge_gap_s: float) -> List[Tuple[int, int]]:
    """Merge adjacent segments separated by short gaps."""
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        gap_s = float(time_s[start] - time_s[prev_end])
        if gap_s <= merge_gap_s:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def detect_segments(
    time_s: np.ndarray,
    ref_angle: np.ndarray,
    activity_lag_frames: int,
    activity_threshold_deg: float,
    min_duration_s: float,
    min_xsens_rom_deg: float,
    merge_gap_s: float,
) -> List[Tuple[int, int]]:
    """Detect activity segments from XsensFair K-frame elbow motion."""
    k = max(1, int(activity_lag_frames))
    activity = np.full(len(ref_angle), np.nan, dtype=np.float64)
    if len(ref_angle) > k:
        valid = np.isfinite(ref_angle[k:]) & np.isfinite(ref_angle[:-k])
        activity[k:][valid] = np.abs(ref_angle[k:][valid] - ref_angle[:-k][valid])
    active = np.isfinite(activity) & (activity > activity_threshold_deg)
    raw_segments = [(max(0, start - k), end) for start, end in find_runs(active)]
    merged = merge_segments(raw_segments, time_s, merge_gap_s)

    filtered: List[Tuple[int, int]] = []
    for start, end in merged:
        segment_angle = ref_angle[start:end + 1]
        finite = segment_angle[np.isfinite(segment_angle)]
        duration_s = float(time_s[end] - time_s[start])
        if duration_s < min_duration_s or finite.size < 2:
            continue
        rom = float(np.nanmax(finite) - np.nanmin(finite))
        if rom < min_xsens_rom_deg:
            continue
        filtered.append((start, end))
    return filtered


def segment_stats(
    time_s: np.ndarray,
    angle: np.ndarray,
    start: int,
    end: int,
    min_valid_ratio: float,
) -> Dict[str, float]:
    """Compute angle summary statistics for one segment."""
    values = angle[start:end + 1]
    finite = np.isfinite(values)
    valid_ratio = float(np.mean(finite)) if len(values) else 0.0
    if valid_ratio < min_valid_ratio or np.count_nonzero(finite) < 2:
        return {
            "rom_deg": math.nan,
            "peak_deg": math.nan,
            "trough_deg": math.nan,
            "peak_time_s": math.nan,
            "trough_time_s": math.nan,
            "valid_frame_ratio": valid_ratio,
        }

    local_idx = np.arange(start, end + 1)
    finite_values = values[finite]
    finite_indices = local_idx[finite]
    peak_pos = int(finite_indices[int(np.argmax(finite_values))])
    trough_pos = int(finite_indices[int(np.argmin(finite_values))])
    return {
        "rom_deg": float(np.nanmax(finite_values) - np.nanmin(finite_values)),
        "peak_deg": float(np.nanmax(finite_values)),
        "trough_deg": float(np.nanmin(finite_values)),
        "peak_time_s": float(time_s[peak_pos]),
        "trough_time_s": float(time_s[trough_pos]),
        "valid_frame_ratio": valid_ratio,
    }


def build_segment_rows(data: Dict[str, np.ndarray], args: argparse.Namespace) -> List[Dict[str, object]]:
    """Detect segments and compute per-system ROM rows."""
    time_s = data["Time_s"]
    rows: List[Dict[str, object]] = []
    for side in SIDES:
        segments = detect_segments(
            time_s=time_s,
            ref_angle=data[f"XsensFair_{side}_deg"],
            activity_lag_frames=args.activity_lag_frames,
            activity_threshold_deg=args.activity_threshold_deg,
            min_duration_s=args.min_duration_s,
            min_xsens_rom_deg=args.min_xsens_rom_deg,
            merge_gap_s=args.merge_gap_s,
        )
        prefix = "L" if side == "LeftElbow" else "R"
        for idx, (start, end) in enumerate(segments, start=1):
            row: Dict[str, object] = {
                "SegmentID": f"{prefix}{idx:02d}",
                "Side": side,
                "StartFrame": int(data["Frame"][start]),
                "EndFrame": int(data["Frame"][end]),
                "Start_s": float(time_s[start]),
                "End_s": float(time_s[end]),
                "Duration_s": float(time_s[end] - time_s[start]),
            }
            for system in SYSTEMS:
                stats = segment_stats(
                    time_s=time_s,
                    angle=data[f"{system}_{side}_deg"],
                    start=start,
                    end=end,
                    min_valid_ratio=args.min_valid_ratio,
                )
                for key, value in stats.items():
                    row[f"{system}_{key}"] = value
            rows.append(row)
    return rows


def write_segment_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write segment rows to CSV."""
    base = ["SegmentID", "Side", "StartFrame", "EndFrame", "Start_s", "End_s", "Duration_s"]
    metrics = ("rom_deg", "peak_deg", "trough_deg", "peak_time_s", "trough_time_s", "valid_frame_ratio")
    fieldnames = base + [f"{system}_{metric}" for system in SYSTEMS for metric in metrics]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    out[key] = "" if not math.isfinite(value) else f"{value:.6f}"
                else:
                    out[key] = value
            writer.writerow(out)


def row_values(rows: List[Dict[str, object]], side: str, system: str, metric: str) -> np.ndarray:
    """Extract one metric vector for side/system from segment rows."""
    return np.asarray(
        [float(row.get(f"{system}_{metric}", math.nan)) for row in rows if row["Side"] == side],
        dtype=np.float64,
    )


def rula_categories(values: np.ndarray, bins: List[float]) -> np.ndarray:
    """Convert peak elbow angles into simple RULA-like bins."""
    out = np.full(len(values), -1, dtype=np.int64)
    finite = np.isfinite(values)
    out[finite] = np.digitize(values[finite], bins, right=False)
    return out


def confusion_summary(target: np.ndarray, reference: np.ndarray, bins: List[float]) -> Dict[str, object]:
    """Build RULA-bin agreement metrics using reference rows and target columns."""
    target_cat = rula_categories(target, bins)
    ref_cat = rula_categories(reference, bins)
    n_bins = len(bins) + 1
    matrix = np.zeros((n_bins, n_bins), dtype=int)
    valid = (target_cat >= 0) & (ref_cat >= 0)
    for r_cat, t_cat in zip(ref_cat[valid], target_cat[valid]):
        matrix[int(r_cat), int(t_cat)] += 1
    total = int(matrix.sum())
    agreement = float(np.trace(matrix) / total) if total else None
    off_by_one = None
    if total:
        off_by_one_count = 0
        for ref_idx in range(n_bins):
            for target_idx in range(n_bins):
                if abs(ref_idx - target_idx) <= 1:
                    off_by_one_count += int(matrix[ref_idx, target_idx])
        off_by_one = float(off_by_one_count / total)
    return {
        "agreement_rate": agreement,
        "off_by_one_rate": off_by_one,
        "valid_segment_count": total,
        "confusion_matrix": matrix.tolist(),
    }


def agreement_summary(rows: List[Dict[str, object]], args: argparse.Namespace, bins: List[float]) -> Dict[str, object]:
    """Summarize ROM, peak, and RULA-bin agreement by side and pair."""
    summary: Dict[str, object] = {
        "config": {
            "combined_csv": str(Path(args.combined_csv)),
            "activity_lag_frames": int(args.activity_lag_frames),
            "activity_threshold_deg": float(args.activity_threshold_deg),
            "min_duration_s": float(args.min_duration_s),
            "min_xsens_rom_deg": float(args.min_xsens_rom_deg),
            "merge_gap_s": float(args.merge_gap_s),
            "min_valid_ratio": float(args.min_valid_ratio),
            "rula_bins_deg": bins,
        },
        "segments_count": {},
        "rom_agreement": {},
        "rula_bin_agreement": {},
    }
    for side in SIDES:
        side_rows = [row for row in rows if row["Side"] == side]
        summary["segments_count"][side] = len(side_rows)
        ref_rom = row_values(rows, side, "XsensFair", "rom_deg")
        ref_peak = row_values(rows, side, "XsensFair", "peak_deg")
        summary["rom_agreement"][side] = {}
        summary["rula_bin_agreement"][side] = {}
        for system in TARGET_SYSTEMS:
            target_rom = row_values(rows, side, system, "rom_deg")
            target_peak = row_values(rows, side, system, "peak_deg")
            rom_f, ref_rom_f = finite_pair(target_rom, ref_rom)
            ratio_mask = np.isfinite(target_rom) & np.isfinite(ref_rom) & (np.abs(ref_rom) > 1e-9)
            ratios = target_rom[ratio_mask] / ref_rom[ratio_mask]
            pair_name = f"{system}_vs_XsensFair"
            summary["rom_agreement"][side][pair_name] = {
                "valid_segment_count": int(len(rom_f)),
                "pearson_rom": pearson(ref_rom, target_rom),
                "rom_ratio_median": float(np.nanmedian(ratios)) if len(ratios) else None,
                "rom_ratio_iqr": [
                    float(np.nanpercentile(ratios, 25)),
                    float(np.nanpercentile(ratios, 75)),
                ] if len(ratios) else None,
                "rom_mae_deg": mean_abs_diff(target_rom, ref_rom),
                "peak_pearson": pearson(ref_peak, target_peak),
                "peak_mae_deg": mean_abs_diff(target_peak, ref_peak),
            }
            summary["rula_bin_agreement"][side][pair_name] = confusion_summary(target_peak, ref_peak, bins)
    return rounded(summary)


def plot_segments_timeline(data: Dict[str, np.ndarray], rows: List[Dict[str, object]], side: str, out_dir: Path) -> None:
    """Plot elbow angle timeline with detected activity spans."""
    time_s = data["Time_s"]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for row in rows:
        if row["Side"] != side:
            continue
        ax.axvspan(float(row["Start_s"]), float(row["End_s"]), color="#9e9e9e", alpha=0.18)
        ax.text(float(row["Start_s"]), 176, str(row["SegmentID"]), fontsize=8, color="#555555", rotation=90)
    for system in SYSTEMS:
        ax.plot(time_s, data[f"{system}_{side}_deg"], color=COLORS[system], linewidth=1.2, label=LABELS[system])
    ax.set_title(f"{side} activity segments and elbow angles", fontsize=13, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Elbow angle (deg)")
    ax.set_ylim(0, 180)
    ax.grid(True, alpha=0.24)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_segments_timeline_{side.lower()}.png", dpi=170)
    plt.close(fig)


def plot_rom_scatter(rows: List[Dict[str, object]], side: str, out_dir: Path) -> None:
    """Plot per-segment ROM scatter against XsensFair."""
    side_rows = [row for row in rows if row["Side"] == side]
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    max_val = 20.0
    ref = np.asarray([float(row["XsensFair_rom_deg"]) for row in side_rows], dtype=np.float64)
    for system in ("SKT", "AFH", "XsensNative"):
        target = np.asarray([float(row[f"{system}_rom_deg"]) for row in side_rows], dtype=np.float64)
        x, y = finite_pair(ref, target)
        if len(x):
            max_val = max(max_val, float(np.nanmax(np.r_[x, y])))
        ax.scatter(x, y, s=45, alpha=0.75, color=COLORS[system], label=LABELS[system])
    for row in side_rows:
        if math.isfinite(float(row["XsensFair_rom_deg"])):
            ax.text(float(row["XsensFair_rom_deg"]), float(row["XsensFair_rom_deg"]), str(row["SegmentID"]), fontsize=7)
    ax.plot([0, max_val], [0, max_val], color="#333333", linewidth=1.0, alpha=0.55, label="ideal y=x")
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{side} segment ROM agreement", fontsize=13, weight="bold")
    ax.set_xlabel("XsensFair ROM (deg)")
    ax.set_ylabel("Target system ROM (deg)")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_rom_scatter_{side.lower()}.png", dpi=170)
    plt.close(fig)


def plot_rom_bars(rows: List[Dict[str, object]], side: str, out_dir: Path) -> None:
    """Plot grouped ROM bars per activity segment."""
    side_rows = [row for row in rows if row["Side"] == side]
    if not side_rows:
        return
    x = np.arange(len(side_rows))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(9, len(side_rows) * 0.75), 5.2))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(SYSTEMS))
    for offset, system in zip(offsets, SYSTEMS):
        values = [float(row[f"{system}_rom_deg"]) for row in side_rows]
        ax.bar(x + offset, values, width=width, color=COLORS[system], label=LABELS[system], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["SegmentID"]) for row in side_rows], rotation=45)
    ax.set_title(f"{side} segment ROM by system", fontsize=13, weight="bold")
    ax.set_xlabel("Segment")
    ax.set_ylabel("ROM (deg)")
    ax.grid(True, axis="y", alpha=0.24)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_rom_bars_{side.lower()}.png", dpi=170)
    plt.close(fig)


def plot_rula_confusion(summary: Dict[str, object], side: str, out_dir: Path) -> None:
    """Plot RULA-like confusion matrices for SKT and AFH."""
    pairs = ("SKT_vs_XsensFair", "AFH_vs_XsensFair")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, pair in zip(axes, pairs):
        matrix = np.asarray(summary["rula_bin_agreement"][side][pair]["confusion_matrix"], dtype=int)
        im = ax.imshow(matrix, cmap="YlGnBu")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(int(value)), ha="center", va="center", color="#111111", fontsize=10)
        ax.set_title(pair, fontsize=11, weight="bold")
        ax.set_xlabel("Target bin")
        ax.set_ylabel("XsensFair bin")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{side} RULA-like peak-angle bin agreement", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_rula_confusion_{side.lower()}.png", dpi=170)
    plt.close(fig)


def write_plot_index(out_dir: Path, rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    """Write a Markdown index for segment ROM artifacts."""
    lines = ["# Activity-Segment ROM Evaluation", ""]
    lines.append("## Segment Counts")
    for side in SIDES:
        lines.append(f"- {side}: {summary['segments_count'][side]} segments")
    lines.append("")
    lines.append("## Headline Metrics")
    for side in SIDES:
        lines.append(f"### {side}")
        for pair, metrics in summary["rom_agreement"][side].items():
            rula = summary["rula_bin_agreement"][side][pair]
            lines.append(
                f"- {pair}: rom_pearson={metrics['pearson_rom']}, rom_mae={metrics['rom_mae_deg']}, "
                f"peak_mae={metrics['peak_mae_deg']}, rula_agreement={rula['agreement_rate']}, "
                f"off_by_one={rula['off_by_one_rate']}"
            )
        lines.append("")
    lines.append("## Figures")
    for side in SIDES:
        side_l = side.lower()
        lines.extend([
            f"- `plot_segments_timeline_{side_l}.png`",
            f"- `plot_rom_scatter_{side_l}.png`",
            f"- `plot_rom_bars_{side_l}.png`",
            f"- `plot_rula_confusion_{side_l}.png`",
        ])
    lines.append("")
    lines.append("## Segments")
    for row in rows:
        lines.append(
            f"- {row['SegmentID']} {row['Side']}: {float(row['Start_s']):.2f}s-"
            f"{float(row['End_s']):.2f}s, XsensFair ROM={float(row['XsensFair_rom_deg']):.2f} deg"
        )
    (out_dir / "plot_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run segment ROM evaluation."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = parse_bins(args.rula_bins)
    data = load_combined_csv(Path(args.combined_csv))
    rows = build_segment_rows(data, args)
    summary = agreement_summary(rows, args, bins)
    write_segment_csv(out_dir / "segment_rom.csv", rows)
    (out_dir / "segment_rom_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    for side in SIDES:
        plot_segments_timeline(data, rows, side, out_dir)
        plot_rom_scatter(rows, side, out_dir)
        plot_rom_bars(rows, side, out_dir)
        plot_rula_confusion(summary, side, out_dir)
    write_plot_index(out_dir, rows, summary)

    print("[saved]", out_dir / "segment_rom.csv")
    print("[saved]", out_dir / "segment_rom_summary.json")
    for side in SIDES:
        print(f"[{side}] segments={summary['segments_count'][side]}")
        for pair, metrics in summary["rom_agreement"][side].items():
            rula = summary["rula_bin_agreement"][side][pair]
            print(
                f"  {pair}: ROM Pearson={metrics['pearson_rom']} "
                f"ROM MAE={metrics['rom_mae_deg']} RULA agreement={rula['agreement_rate']}"
            )


if __name__ == "__main__":
    main()
