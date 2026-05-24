#!/opt/anaconda3/envs/pose/bin/python
"""Generate K-frame delta scatter plots with Pearson + Spearman annotations.

Consumes the combined CSV produced by 01_compute_elbow_deltas.py and emits
per-system / per-side scatter plots against XsensFair, plus a grid overview
and a small Markdown headline table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SIDES = ("LeftElbow", "RightElbow")
REFERENCE = "XsensFair"
DEFAULT_TARGETS = ("SKT", "FastSAM3D", "Merge", "XsensNative")
COLORS = {
    "SKT": "#ff7a18",
    "FastSAM3D": "#2196F3",
    "Merge": "#8e44ad",
    "XsensNative": "#43a047",
}
LABELS = {
    "SKT": "SKT (stereo + RTMPose)",
    "FastSAM3D": "FastSAM3D (Aitor unfiltered)",
    "Merge": "Merge (Viscando × FastSAM3D)",
    "XsensNative": "XsensNative (anchor)",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-threshold-deg", type=float, default=None,
                        help="Optional |reference delta| threshold; produces active-only versions.")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                        help="Comma-separated system names to plot against XsensFair.")
    return parser.parse_args()


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    """Load the combined elbow CSV into typed arrays."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    data: Dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if (
            key.endswith(("_valid", "_interpolated", "_delta_anomaly_flag"))
            or "_valid_k" in key
            or "_delta_anomaly_flag_k" in key
        ):
            data[key] = np.array([value == "True" for value in values], dtype=bool)
        elif key in {"Frame", "StereoFrameId", "LeftVideoFrame", "RightVideoFrame"}:
            data[key] = np.array([int(value) for value in values], dtype=np.int64)
        else:
            data[key] = np.array(
                [float(value) if value != "" else np.nan for value in values],
                dtype=np.float64,
            )
    return data


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite paired samples."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def title_side(side: str) -> str:
    """Human-readable side label."""
    return "Left elbow" if side == "LeftElbow" else "Right elbow"


def plot_scatter(
    ref: np.ndarray,
    target: np.ndarray,
    system: str,
    side: str,
    k: int,
    out_path: Path,
    active_threshold: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    """Render one scatter plot of target delta vs reference delta."""
    if active_threshold is not None:
        mask = np.abs(ref) > float(active_threshold)
        ref_p, tgt_p = finite_pair(ref[mask], target[mask])
    else:
        ref_p, tgt_p = finite_pair(ref, target)
    if len(ref_p) < 3:
        return None

    pearson = float(np.corrcoef(ref_p, tgt_p)[0, 1])
    spearman_result = spearmanr(ref_p, tgt_p)
    spearman = float(spearman_result.correlation if hasattr(spearman_result, "correlation") else spearman_result[0])

    lim = float(np.nanmax(np.abs(np.r_[ref_p, tgt_p])))
    lim = max(lim, 5.0)

    color = COLORS.get(system, "#555555")
    label = LABELS.get(system, system)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(ref_p, tgt_p, s=14, alpha=0.40, color=color, edgecolors="none")
    ax.plot([-lim, lim], [-lim, lim], color="#333333", linewidth=1.0, alpha=0.55, label="ideal y=x")
    ax.axhline(0.0, color="#888888", linewidth=0.5, alpha=0.4)
    ax.axvline(0.0, color="#888888", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    suffix = "" if active_threshold is None else f"  (active |ref|>{active_threshold:g}°)"
    ax.set_title(
        f"{title_side(side)} K={k} delta{suffix}\n{label} vs {REFERENCE}",
        fontsize=11,
        weight="bold",
    )
    ax.set_xlabel(f"{REFERENCE} K={k} delta (°)")
    ax.set_ylabel(f"{label} K={k} delta (°)")
    ax.grid(True, alpha=0.25)

    info = (
        f"Pearson r = {pearson:.3f}\n"
        f"Spearman ρ = {spearman:.3f}\n"
        f"n = {len(ref_p)}"
    )
    ax.text(
        0.03,
        0.97,
        info,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    return {"system": system, "side": side, "pearson": pearson, "spearman": spearman, "n": len(ref_p)}


def plot_grid(
    data: Dict[str, np.ndarray],
    systems: List[str],
    k: int,
    out_path: Path,
) -> None:
    """Render a side x system grid overview."""
    fig, axes = plt.subplots(len(SIDES), len(systems), figsize=(4.3 * len(systems), 4.3 * len(SIDES)))
    if len(SIDES) == 1:
        axes = np.array([axes])
    if len(systems) == 1:
        axes = axes.reshape(-1, 1)
    for row_idx, side in enumerate(SIDES):
        ref_col = f"{REFERENCE}_{side}_delta_k{k}_deg"
        if ref_col not in data:
            continue
        ref = data[ref_col]
        for col_idx, system in enumerate(systems):
            ax = axes[row_idx, col_idx]
            tgt_col = f"{system}_{side}_delta_k{k}_deg"
            if tgt_col not in data:
                ax.set_visible(False)
                continue
            ref_p, tgt_p = finite_pair(ref, data[tgt_col])
            if len(ref_p) < 3:
                ax.set_visible(False)
                continue
            pearson = float(np.corrcoef(ref_p, tgt_p)[0, 1])
            spearman_result = spearmanr(ref_p, tgt_p)
            spearman = float(spearman_result.correlation if hasattr(spearman_result, "correlation") else spearman_result[0])
            lim = max(float(np.nanmax(np.abs(np.r_[ref_p, tgt_p]))), 5.0)
            ax.scatter(ref_p, tgt_p, s=9, alpha=0.4, color=COLORS.get(system, "#555555"), edgecolors="none")
            ax.plot([-lim, lim], [-lim, lim], color="#333333", linewidth=0.7, alpha=0.55)
            ax.axhline(0.0, color="#888888", linewidth=0.4, alpha=0.4)
            ax.axvline(0.0, color="#888888", linewidth=0.4, alpha=0.4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")
            side_short = "L" if side == "LeftElbow" else "R"
            ax.set_title(
                f"{side_short} · {system}\nr={pearson:.3f}  ρ={spearman:.3f}  n={len(ref_p)}",
                fontsize=10,
                weight="bold",
            )
            if row_idx == len(SIDES) - 1:
                ax.set_xlabel(f"{REFERENCE} (°)")
            if col_idx == 0:
                ax.set_ylabel(f"{system} (°)")
            ax.grid(True, alpha=0.22)
    fig.suptitle(
        f"K={k} frame-delta scatter (Phase 4: MA 200ms, Xsens unsmoothed, SKT quality filtered)",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    """CLI entry."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_csv(Path(args.combined_csv))
    k = int(args.k)
    targets = [name.strip() for name in args.targets.split(",") if name.strip()]

    plt.style.use("seaborn-v0_8-whitegrid")
    summary_rows: List[Dict[str, object]] = []
    for side in SIDES:
        ref_col = f"{REFERENCE}_{side}_delta_k{k}_deg"
        if ref_col not in data:
            print(f"[skip] missing {ref_col}")
            continue
        ref = data[ref_col]
        for system in targets:
            tgt_col = f"{system}_{side}_delta_k{k}_deg"
            if tgt_col not in data:
                print(f"[skip] missing {tgt_col}")
                continue
            out_path = out_dir / f"scatter_{system.lower()}_vs_xsensfair_k{k}_{side.lower()}.png"
            row = plot_scatter(ref, data[tgt_col], system, side, k, out_path)
            if row is not None:
                summary_rows.append(row)
            if args.active_threshold_deg is not None:
                active_path = out_dir / f"scatter_active_{system.lower()}_vs_xsensfair_k{k}_{side.lower()}.png"
                plot_scatter(
                    ref,
                    data[tgt_col],
                    system,
                    side,
                    k,
                    active_path,
                    active_threshold=args.active_threshold_deg,
                )

    plot_grid(data, targets, k, out_dir / f"scatter_grid_k{k}.png")

    lines = [
        f"# K={k} frame-delta motion agreement (Phase 4)",
        "",
        f"Source CSV: `{args.combined_csv}`",
        "",
        "| System | Side | Pearson r | Spearman ρ | n |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['side']} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['n']} |"
        )
    (out_dir / "headline_k_delta.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
