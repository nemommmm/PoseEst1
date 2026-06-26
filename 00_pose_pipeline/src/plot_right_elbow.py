"""Plot right-elbow angle time series for SKT / FastSAM3D / XsensFair.

Reads ``angle_eval/angle_timeseries.csv`` produced by the ``angle`` pipeline
stage and generates a PNG focused on the triple-overlap window — the contiguous
segment where all three systems simultaneously have valid data.

Usage::

    /opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/plot_right_elbow.py \
        --run-dir 00_pose_pipeline/runs/<run_tag>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_SYSTEMS = ["SKT", "FastSAM3D", "XsensFair"]
_COLORS = {"SKT": "#e05c5c", "FastSAM3D": "#4a90d9", "XsensFair": "#2ca02c"}
_JOINT = "RightElbow"


def _load_timeseries(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (time_s, {system: angle_array}) from angle_timeseries.csv."""
    time_list: list[float] = []
    cols: dict[str, list[float]] = {s: [] for s in _SYSTEMS}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["Time_s"])
            except (KeyError, ValueError):
                continue
            time_list.append(t)
            for sys in _SYSTEMS:
                key = f"{sys}_{_JOINT}_deg"
                raw = row.get(key, "")
                cols[sys].append(float(raw) if raw.strip() != "" else math.nan)

    time_s = np.asarray(time_list, dtype=np.float64)
    angles = {s: np.asarray(cols[s], dtype=np.float64) for s in _SYSTEMS}
    return time_s, angles


def _triple_overlap(angles: dict[str, np.ndarray]) -> np.ndarray:
    """Boolean mask where SKT, FastSAM3D, and XsensFair are all finite."""
    mask = np.ones(len(next(iter(angles.values()))), dtype=bool)
    for s in _SYSTEMS:
        mask &= np.isfinite(angles[s])
    return mask


def _window_range(time_s: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """First and last index (inclusive) of the triple-overlap window."""
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return 0, len(time_s) - 1
    return int(idxs[0]), int(idxs[-1])


def _compute_stats(
    time_s: np.ndarray,
    angles: dict[str, np.ndarray],
    triple_mask: np.ndarray,
    i_start: int,
    i_end: int,
) -> dict:
    """Compute MAE and valid_ratio in the triple-overlap window."""
    window_slice = slice(i_start, i_end + 1)
    n_window = i_end - i_start + 1
    ref = angles["XsensFair"]
    stats: dict = {
        "t_start_s": float(time_s[i_start]),
        "t_end_s": float(time_s[i_end]),
        "duration_s": float(time_s[i_end] - time_s[i_start]),
        "n_window_frames": n_window,
        "n_overlap_frames": int(triple_mask[window_slice].sum()),
    }
    for sys in ["SKT", "FastSAM3D"]:
        arr = angles[sys]
        # valid = finite in the triple mask region (same denominator for all)
        valid_in_window = np.isfinite(arr[window_slice])
        n_valid = int(valid_in_window.sum())
        stats[f"valid_ratio_{sys}"] = round(n_valid / n_window, 4) if n_window else 0.0
        # MAE strictly on triple-overlap frames
        triple_in_window = triple_mask[window_slice]
        if triple_in_window.any():
            diff = np.abs(arr[window_slice][triple_in_window] - ref[window_slice][triple_in_window])
            stats[f"mae_deg_{sys}"] = round(float(np.mean(diff)), 3)
        else:
            stats[f"mae_deg_{sys}"] = None

    # XsensFair valid ratio in window
    xf_valid = int(np.isfinite(ref[window_slice]).sum())
    stats["valid_ratio_XsensFair"] = round(xf_valid / n_window, 4) if n_window else 0.0
    return stats


def plot_right_elbow(run_dir: Path) -> None:
    """Generate right-elbow time-series PNG for the given pipeline run directory."""
    angle_dir = run_dir / "angle_eval"
    csv_path = angle_dir / "angle_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"angle_timeseries.csv not found at {csv_path}. "
            "Run the 'angle' pipeline stage first."
        )

    time_s, angles = _load_timeseries(csv_path)
    triple_mask = _triple_overlap(angles)
    i_start, i_end = _window_range(time_s, triple_mask)
    stats = _compute_stats(time_s, angles, triple_mask, i_start, i_end)

    # Save summary JSON
    summary_path = angle_dir / "right_elbow_triple_overlap_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[plot_right_elbow] Summary: {summary_path}")
    print(f"  window: {stats['t_start_s']:.1f}s – {stats['t_end_s']:.1f}s "
          f"({stats['duration_s']:.1f}s, {stats['n_overlap_frames']} overlap frames)")
    for sys in ["SKT", "FastSAM3D"]:
        mae = stats.get(f"mae_deg_{sys}")
        vr = stats.get(f"valid_ratio_{sys}", 0.0)
        print(f"  {sys}: valid={vr*100:.1f}%  MAE={mae:.1f}°" if mae is not None else f"  {sys}: no overlap")

    # Build legend labels
    def label(sys: str) -> str:
        if sys == "XsensFair":
            n = stats["n_overlap_frames"]
            vr = stats["valid_ratio_XsensFair"]
            return f"XsensFair  (reference, valid={vr*100:.0f}%, n={n})"
        mae = stats.get(f"mae_deg_{sys}")
        vr = stats.get(f"valid_ratio_{sys}", 0.0)
        n = stats["n_overlap_frames"]
        if mae is not None:
            return f"{sys}  (valid={vr*100:.0f}%, MAE={mae:.1f}°, n={n})"
        return f"{sys}  (no overlap)"

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    t_window = time_s[i_start: i_end + 1]
    for sys in _SYSTEMS:
        arr = angles[sys][i_start: i_end + 1]
        zorder = 3 if sys == "XsensFair" else 2
        lw = 1.5 if sys == "SKT" else 1.8
        ax.plot(t_window, arr, label=label(sys), color=_COLORS[sys],
                linewidth=lw, zorder=zorder)

    # Shade regions where triple overlap is False inside the window
    triple_window = triple_mask[i_start: i_end + 1]
    non_overlap = ~triple_window
    if non_overlap.any():
        ax.fill_between(t_window, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=non_overlap, alpha=0.08, color="gray",
                        label="non-overlap region")

    run_tag = run_dir.name
    ax.set_title(
        f"Right Elbow Flexion — {run_tag}\n"
        f"Triple-overlap window: {stats['t_start_s']:.1f}–{stats['t_end_s']:.1f} s "
        f"({stats['duration_s']:.1f} s)",
        fontsize=11,
    )
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Right Elbow Angle (°)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = angle_dir / "right_elbow_timeseries.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_right_elbow] Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to a pipeline run directory (contains angle_eval/).")
    args = parser.parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    plot_right_elbow(run_dir)


if __name__ == "__main__":
    main()
