"""Scatter plots for K-frame motion deltas."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.config import section
from common.metrics import pearson, spearman


COLORS = {
    "SKT": "#ff7a18",
    "FastSAM3D": "#2196F3",
    "Merge": "#8e44ad",
    "XsensNative": "#43a047",
}


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Load CSV columns."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    data = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key.endswith("_valid") or "_valid_k" in key:
            data[key] = np.asarray([value == "True" for value in values], dtype=bool)
        elif key == "Frame":
            data[key] = np.asarray([int(value) for value in values], dtype=np.int64)
        else:
            data[key] = np.asarray([float(value) if value else np.nan for value in values], dtype=np.float64)
    return data


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def plot_one(ref: np.ndarray, target: np.ndarray, system: str, angle: str, k: int, out_path: Path, active_threshold: float | None = None) -> dict | None:
    """Render one scatter plot."""
    if active_threshold is not None:
        active = np.abs(ref) > active_threshold
        ref, target = ref[active], target[active]
    ref_p, target_p = finite_pair(ref, target)
    if len(ref_p) < 3:
        return None
    p = pearson(ref_p, target_p)
    s = spearman(ref_p, target_p)
    lim = max(float(np.nanmax(np.abs(np.r_[ref_p, target_p]))), 5.0)
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(ref_p, target_p, s=13, alpha=0.42, color=COLORS.get(system, "#555555"), edgecolors="none")
    ax.plot([-lim, lim], [-lim, lim], color="#333333", linewidth=1.0, alpha=0.6)
    ax.axhline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="#888888", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    suffix = "" if active_threshold is None else f" active |ref|>{active_threshold:g}°"
    ax.set_title(f"{angle} K={k} delta{suffix}\n{system} vs XsensFair", fontsize=11, weight="bold")
    ax.set_xlabel("XsensFair delta (deg)")
    ax.set_ylabel(f"{system} delta (deg)")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {p:.3f}\nSpearman rho = {s:.3f}\nn = {len(ref_p)}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"system": system, "angle": angle, "k": k, "pearson": p, "spearman": s, "n": len(ref_p), "active_threshold": active_threshold}


def render_scatter(config: dict, run_dir: Path) -> Path:
    """Generate scatter plots from motion output."""
    scatter_cfg = section(config, "scatter")
    eval_cfg = section(config, "evaluation")
    k = int(scatter_cfg.get("k", 6))
    targets = scatter_cfg.get("targets", ["SKT", "FastSAM3D", "Merge", "XsensNative"])
    active_threshold = scatter_cfg.get("active_threshold_deg")
    motion_path = run_dir / "motion_delta" / "motion_delta_combined.csv"
    if not motion_path.exists():
        raise FileNotFoundError(f"Run motion stage first; missing {motion_path}")
    data = load_csv(motion_path)
    out_dir = run_dir / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for angle in eval_cfg.get("angle_names", ["LeftElbow", "RightElbow"]):
        ref_col = f"XsensFair_{angle}_delta_k{k}_deg"
        if ref_col not in data:
            continue
        for system in targets:
            target_col = f"{system}_{angle}_delta_k{k}_deg"
            if target_col not in data:
                continue
            result = plot_one(data[ref_col], data[target_col], system, angle, k, out_dir / f"scatter_{system.lower()}_vs_xsensfair_{angle.lower()}_k{k}.png")
            if result:
                rows.append(result)
            if active_threshold is not None:
                active_result = plot_one(
                    data[ref_col],
                    data[target_col],
                    system,
                    angle,
                    k,
                    out_dir / f"scatter_active_{system.lower()}_vs_xsensfair_{angle.lower()}_k{k}.png",
                    float(active_threshold),
                )
                if active_result:
                    rows.append(active_result)
    csv_path = out_dir / f"scatter_summary_k{k}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "angle", "k", "pearson", "spearman", "n", "active_threshold"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[scatter] saved {csv_path}")
    return csv_path
