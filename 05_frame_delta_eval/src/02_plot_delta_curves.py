#!/opt/anaconda3/envs/pose/bin/python
"""Plot elbow motion-delta comparison curves from combined CSV output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SIDES = ("LeftElbow", "RightElbow")
SYSTEMS = ("SKT", "AFH", "XsensFair", "XsensNative")
COLORS = {
    "SKT": "#ff7a18",
    "AFH": "#2196F3",
    "XsensFair": "#43a047",
    "XsensNative": "#70757f",
}
LABELS = {
    "SKT": "SKT stereo",
    "AFH": "AFH hybrid",
    "XsensFair": "Xsens-derived geometric",
    "XsensNative": "Xsens native",
}
LINESTYLES = {
    "SKT": "-",
    "AFH": "-",
    "XsensFair": "-",
    "XsensNative": "--",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k-frame-list", default=None,
                        help="Optional comma-separated K list; defaults to summary config.")
    return parser.parse_args()


def as_float(value: str) -> float:
    """Convert CSV string to float with blanks as NaN."""
    if value is None or value == "":
        return np.nan
    return float(value)


def load_combined_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read combined CSV into column arrays."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    columns: Dict[str, list] = {key: [] for key in rows[0].keys()}
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


def finite_plot(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mask non-finite plot points."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def title_side(side: str) -> str:
    """Human-readable side title."""
    return "Left elbow" if side == "LeftElbow" else "Right elbow"


def setup_axes(ax, title: str, ylabel: str) -> None:
    """Apply shared plot styling."""
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def parse_k_list(raw: str | None, summary: Dict) -> List[int]:
    """Resolve K-frame list from CLI or summary JSON."""
    if raw:
        return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    values = summary.get("config", {}).get("k_frame_list")
    if values:
        return [int(v) for v in values]
    return [1]


def k_key(k: int) -> str:
    """Summary key for one K-frame spacing."""
    return f"k{int(k)}"


def delta_col(system: str, side: str, k: int, data: Dict[str, np.ndarray]) -> str:
    """Return the delta column name, with legacy K=1 fallback."""
    preferred = f"{system}_{side}_delta_k{k}_deg"
    if preferred in data:
        return preferred
    legacy = f"{system}_{side}_delta_deg"
    if int(k) == 1 and legacy in data:
        return legacy
    raise KeyError(f"Missing delta column for {system} {side} K={k}")


def metrics_for(summary: Dict, side: str, pair_name: str, k: int) -> Dict:
    """Return pair metrics for nested K summaries or legacy flat summaries."""
    metrics = summary["motion_agreement"][side][pair_name]
    nested = metrics.get(k_key(k))
    if isinstance(nested, dict):
        return nested
    if int(k) == 1:
        return metrics
    return {}


def active_threshold_for(summary: Dict, k: int) -> float:
    """Return active-motion threshold for one K-frame spacing."""
    thresholds = summary.get("config", {}).get("thresholds_by_k", {})
    item = thresholds.get(k_key(k), {})
    if "active_delta_threshold_deg" in item:
        return float(item["active_delta_threshold_deg"])
    return float(summary.get("config", {}).get("active_delta_threshold_deg", 1.0))


def plot_delta(data: Dict[str, np.ndarray], side: str, k: int, out_dir: Path) -> None:
    """Plot signed K-frame elbow deltas."""
    time = data["Time_s"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for system in SYSTEMS:
        y = data[delta_col(system, side, k, data)]
        x_f, y_f = finite_plot(time, y)
        ax.plot(
            x_f,
            y_f,
            color=COLORS[system],
            linestyle=LINESTYLES[system],
            linewidth=1.2 if system != "XsensNative" else 1.0,
            alpha=0.95 if system != "XsensNative" else 0.75,
            label=LABELS[system],
        )
    setup_axes(ax, f"{title_side(side)} K={k} frame angle delta", f"Delta angle over {k} frame(s) (deg)")
    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.45)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_delta_k{k}_{side.lower()}.png", dpi=160)
    plt.close(fig)


def cumulative_abs_path(delta: np.ndarray) -> np.ndarray:
    """Cumulative angular path, holding value steady over invalid samples."""
    out = np.zeros_like(delta, dtype=np.float64)
    total = 0.0
    for idx, value in enumerate(delta):
        if np.isfinite(value):
            total += abs(float(value))
        out[idx] = total
    return out


def plot_cumulative(data: Dict[str, np.ndarray], side: str, k: int, out_dir: Path) -> None:
    """Plot cumulative angular path."""
    time = data["Time_s"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for system in SYSTEMS:
        path = cumulative_abs_path(data[delta_col(system, side, k, data)])
        ax.plot(
            time,
            path,
            color=COLORS[system],
            linestyle=LINESTYLES[system],
            linewidth=1.5 if system != "XsensNative" else 1.0,
            alpha=0.95 if system != "XsensNative" else 0.75,
            label=LABELS[system],
        )
    setup_axes(ax, f"{title_side(side)} cumulative K={k} angular path", f"Cumulative |K={k} delta| (deg)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_cumdelta_k{k}_{side.lower()}.png", dpi=160)
    plt.close(fig)


def zeroed_angle(angle: np.ndarray) -> np.ndarray:
    """Subtract first finite angle to remove absolute offset."""
    out = angle.copy()
    finite = np.where(np.isfinite(out))[0]
    if finite.size == 0:
        return out
    return out - out[finite[0]]


def plot_zeroed(data: Dict[str, np.ndarray], side: str, out_dir: Path) -> None:
    """Plot angles after removing each system's initial offset."""
    time = data["Time_s"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for system in SYSTEMS:
        y = zeroed_angle(data[f"{system}_{side}_deg"])
        x_f, y_f = finite_plot(time, y)
        ax.plot(
            x_f,
            y_f,
            color=COLORS[system],
            linestyle=LINESTYLES[system],
            linewidth=1.25 if system != "XsensNative" else 1.0,
            alpha=0.95 if system != "XsensNative" else 0.75,
            label=LABELS[system],
        )
    setup_axes(ax, f"{title_side(side)} zeroed elbow angle", "Angle change from first finite frame (deg)")
    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.45)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_zeroed_{side.lower()}.png", dpi=160)
    plt.close(fig)


def plot_scatter(
    data: Dict[str, np.ndarray],
    side: str,
    system: str,
    k: int,
    out_dir: Path,
    active_threshold: float | None = None,
) -> None:
    """Plot target delta vs XsensFair reference delta."""
    ref = data[delta_col("XsensFair", side, k, data)]
    target = data[delta_col(system, side, k, data)]
    if active_threshold is not None:
        active = np.abs(ref) > float(active_threshold)
        x, y = finite_plot(ref[active], target[active])
    else:
        x, y = finite_plot(ref, target)
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.scatter(x, y, s=10, alpha=0.35, color=COLORS[system], edgecolors="none")
    if len(x) and len(y):
        lim = float(np.nanmax(np.abs(np.concatenate([x, y]))))
        lim = max(lim, 5.0)
        ax.plot([-lim, lim], [-lim, lim], color="#333333", linewidth=1.0, alpha=0.55, label="ideal y=x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    suffix = "" if active_threshold is None else f" active |ref|>{active_threshold:g}"
    ax.set_title(
        f"{title_side(side)} K={k} delta scatter{suffix}: {LABELS[system]} vs Xsens-derived",
        fontsize=11,
        weight="bold",
    )
    ax.set_xlabel(f"Xsens-derived geometric K={k} delta (deg)")
    ax.set_ylabel(f"{LABELS[system]} K={k} delta (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    name_prefix = "plot_scatter" if active_threshold is None else "plot_scatter_active"
    fig.savefig(out_dir / f"{name_prefix}_{system.lower()}_vs_xsensfair_k{k}_{side.lower()}.png", dpi=160)
    plt.close(fig)


def plot_pearson_vs_k(summary: Dict, side: str, k_list: List[int], out_dir: Path) -> None:
    """Plot motion-agreement Pearson as a function of K-frame spacing."""
    pair_names = ("SKT_vs_XsensFair", "AFH_vs_XsensFair", "SKT_vs_AFH", "XsensNative_vs_XsensFair")
    colors = {
        "SKT_vs_XsensFair": COLORS["SKT"],
        "AFH_vs_XsensFair": COLORS["AFH"],
        "SKT_vs_AFH": "#8e44ad",
        "XsensNative_vs_XsensFair": COLORS["XsensNative"],
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for pair_name in pair_names:
        y = []
        x = []
        for k in k_list:
            metrics = metrics_for(summary, side, pair_name, k)
            pearson = metrics.get("pearson_delta")
            if pearson is not None:
                x.append(k)
                y.append(float(pearson))
        if x:
            ax.plot(x, y, marker="o", linewidth=1.7, color=colors[pair_name], label=pair_name)
    ax.set_xscale("log")
    ax.set_xticks(k_list)
    ax.set_xticklabels([str(k) for k in k_list])
    ax.set_ylim(-0.25, 1.05)
    setup_axes(ax, f"{title_side(side)} Pearson vs K-frame delta spacing", "Delta Pearson")
    ax.set_xlabel("K-frame spacing")
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.45)
    ax.axhline(0.7, color="#2e7d32", linewidth=0.9, linestyle="--", alpha=0.6, label="r=0.7 reference")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_pearson_vs_k_{side.lower()}.png", dpi=170)
    plt.close(fig)


def write_plot_index(out_dir: Path, summary: Dict, k_list: List[int]) -> None:
    """Write a compact Markdown index of generated figures and headline metrics."""
    lines = ["# K-Frame Elbow Motion Evaluation", ""]
    lines.append("## Headline Metrics")
    for side in SIDES:
        lines.append(f"### {title_side(side)}")
        for pair_name, metrics in summary["motion_agreement"][side].items():
            for k in k_list:
                k_metrics = metrics.get(k_key(k), metrics if int(k) == 1 else {})
                pearson = k_metrics.get("pearson_delta")
                active = k_metrics.get("active_pearson_delta")
                slope = k_metrics.get("slope_target_vs_reference")
                ratio = k_metrics.get("path_ratio_target_reference")
                lag = k_metrics.get("lag_sweep", {}).get("best_lag_frames")
                lag_r = k_metrics.get("lag_sweep", {}).get("best_pearson_delta")
                lines.append(
                    f"- {pair_name} {k_key(k)}: pearson={pearson}, active_pearson={active}, "
                    f"slope={slope}, path_ratio={ratio}, best_lag={lag}, lag_pearson={lag_r}"
                )
        lines.append("")
    lines.append("## Figures")
    for side in SIDES:
        side_l = side.lower()
        lines.append(f"- `plot_zeroed_{side_l}.png`")
        lines.append(f"- `plot_pearson_vs_k_{side_l}.png`")
        for k in k_list:
            lines.extend([
                f"- `plot_delta_k{k}_{side_l}.png`",
                f"- `plot_cumdelta_k{k}_{side_l}.png`",
                f"- `plot_scatter_skt_vs_xsensfair_k{k}_{side_l}.png`",
                f"- `plot_scatter_afh_vs_xsensfair_k{k}_{side_l}.png`",
                f"- `plot_scatter_active_skt_vs_xsensfair_k{k}_{side_l}.png`",
                f"- `plot_scatter_active_afh_vs_xsensfair_k{k}_{side_l}.png`",
            ])
    (out_dir / "plot_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_combined_csv(Path(args.combined_csv))
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    k_list = parse_k_list(args.k_frame_list, summary)
    plt.style.use("seaborn-v0_8-whitegrid")
    for side in SIDES:
        plot_zeroed(data, side, out_dir)
        plot_pearson_vs_k(summary, side, k_list, out_dir)
        for k in k_list:
            active_threshold = active_threshold_for(summary, k)
            plot_delta(data, side, k, out_dir)
            plot_cumulative(data, side, k, out_dir)
            plot_scatter(data, side, "SKT", k, out_dir)
            plot_scatter(data, side, "AFH", k, out_dir)
            plot_scatter(data, side, "SKT", k, out_dir, active_threshold=active_threshold)
            plot_scatter(data, side, "AFH", k, out_dir, active_threshold=active_threshold)
    write_plot_index(out_dir, summary, k_list)
    print("[saved plots]", out_dir)


if __name__ == "__main__":
    main()
