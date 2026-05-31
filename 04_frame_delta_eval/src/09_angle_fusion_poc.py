#!/opt/anaconda3/envs/pose/bin/python
"""Angle-level fusion PoC.

Reads the combined CSV that already contains SKT, MotionBert, FastSAM3D and
XsensFair elbow angle series, fuses SKT + MotionBert angle traces with a sweep
of weights, and reports K=1/K=6/K=12 Pearson + Spearman vs XsensFair.

Coordinate-frame issues vanish because angles are coordinate invariant.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

SIDES = ("LeftElbow", "RightElbow")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0",
                        help="Comma-separated SKT weights to sweep (MB weight = 1 - w_skt).")
    parser.add_argument("--ks", default="1,6,12",
                        help="Comma-separated K values for the delta correlation analysis.")
    return parser.parse_args()


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    """Load CSV columns into typed arrays."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: Dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        if key in {"Frame", "StereoFrameId", "LeftVideoFrame", "RightVideoFrame"}:
            out[key] = np.array([int(v) for v in vals], dtype=np.int64)
        elif key.endswith(("_valid", "_interpolated", "_delta_anomaly_flag")) or "_valid_k" in key or "_delta_anomaly_flag_k" in key:
            out[key] = np.array([v == "True" for v in vals], dtype=bool)
        else:
            out[key] = np.array([float(v) if v != "" else np.nan for v in vals], dtype=np.float64)
    return out


def fuse_angles(skt_angle: np.ndarray, mb_angle: np.ndarray, w_skt: float) -> np.ndarray:
    """Weighted average of two angle series, NaN-aware."""
    w_skt = float(w_skt)
    w_mb = 1.0 - w_skt
    skt_finite = np.isfinite(skt_angle)
    mb_finite = np.isfinite(mb_angle)
    fused = np.full_like(skt_angle, np.nan, dtype=np.float64)
    both = skt_finite & mb_finite
    only_skt = skt_finite & ~mb_finite
    only_mb = ~skt_finite & mb_finite
    fused[both] = w_skt * skt_angle[both] + w_mb * mb_angle[both]
    fused[only_skt] = skt_angle[only_skt]
    fused[only_mb] = mb_angle[only_mb]
    return fused


def compute_k_delta(angle: np.ndarray, k: int) -> np.ndarray:
    """K-frame delta with NaN propagation."""
    delta = np.full_like(angle, np.nan, dtype=np.float64)
    if len(angle) > k:
        delta[k:] = angle[k:] - angle[:-k]
    return delta


def finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite paired samples."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Pearson with guardrails."""
    xf, yf = finite_pair(x, y)
    if len(xf) < 3 or np.nanstd(xf) < 1e-9 or np.nanstd(yf) < 1e-9:
        return None
    return float(np.corrcoef(xf, yf)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Spearman with guardrails."""
    xf, yf = finite_pair(x, y)
    if len(xf) < 3:
        return None
    result = spearmanr(xf, yf)
    rho = result.correlation if hasattr(result, "correlation") else result[0]
    return float(rho)


def path_ratio(target: np.ndarray, reference: np.ndarray) -> Optional[float]:
    """Cumulative |delta| ratio target / reference."""
    tf, rf = finite_pair(target, reference)
    if len(tf) < 2:
        return None
    t_path = float(np.sum(np.abs(tf)))
    r_path = float(np.sum(np.abs(rf)))
    if r_path < 1e-9:
        return None
    return t_path / r_path


def main() -> None:
    """Run angle-level fusion sweep."""
    args = parse_args()
    data = load_csv(Path(args.combined_csv))
    weights = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]

    lines: List[str] = [
        "# Angle-level SKT + MotionBERT fusion sweep",
        "",
        f"Source CSV: `{args.combined_csv}`",
        "",
        "| w_skt | w_mb | Side | K | Pearson r | Spearman ρ | path_ratio | n |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    for w in weights:
        for side in SIDES:
            skt_angle = data[f"SKT_{side}_deg"]
            mb_angle = data[f"MotionBert_{side}_deg"]
            ref_angle = data[f"XsensFair_{side}_deg"]
            fused_angle = fuse_angles(skt_angle, mb_angle, w_skt=w)
            for k in ks:
                fused_delta = compute_k_delta(fused_angle, k)
                ref_delta = compute_k_delta(ref_angle, k)
                p = pearson(ref_delta, fused_delta)
                s = spearman(ref_delta, fused_delta)
                pr = path_ratio(fused_delta, ref_delta)
                xf, yf = finite_pair(ref_delta, fused_delta)
                lines.append(
                    "| "
                    + " | ".join([
                        f"{w:.2f}",
                        f"{1.0 - w:.2f}",
                        side,
                        f"{k}",
                        f"{p:.3f}" if p is not None else "NA",
                        f"{s:.3f}" if s is not None else "NA",
                        f"{pr:.3f}" if pr is not None else "NA",
                        f"{len(xf)}",
                    ])
                    + " |"
                )

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
