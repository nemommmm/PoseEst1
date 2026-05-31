#!/opt/anaconda3/envs/pose/bin/python
"""Inject angle-level SKT × MotionBERT fusion as a new system into combined CSV.

Reads an existing elbow_delta_combined.csv, computes fused angle traces using
the SKT + MotionBert columns, and writes a new CSV with all the same content
plus AngleFusion_<side>_deg / _valid / _interpolated and the per-K delta /
valid / anomaly columns expected by 03_segment_rom_eval.py.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

SIDES = ("LeftElbow", "RightElbow")
DEFAULT_W_SKT = 0.25
ANOMALY_THRESHOLDS = {1: 30.0, 6: 60.0, 12: 90.0, 25: 120.0}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--w-skt", type=float, default=DEFAULT_W_SKT)
    parser.add_argument("--name", default="AngleFusion",
                        help="System name to use in the new CSV columns.")
    return parser.parse_args()


def load_csv_rows(path: Path):
    """Return list of dict rows + fieldnames list."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def detect_k_list(fieldnames: List[str]) -> List[int]:
    """Detect the K values present in the input CSV column names."""
    ks = set()
    pat = re.compile(r"_delta_k(\d+)_deg$")
    for name in fieldnames:
        match = pat.search(name)
        if match:
            ks.add(int(match.group(1)))
    return sorted(ks)


def fuse_angles(skt_vals, mb_vals, w_skt: float):
    """Weighted angle fusion, NaN-aware."""
    skt = np.asarray(skt_vals, dtype=np.float64)
    mb = np.asarray(mb_vals, dtype=np.float64)
    w_mb = 1.0 - w_skt
    skt_finite = np.isfinite(skt)
    mb_finite = np.isfinite(mb)
    fused = np.full_like(skt, np.nan, dtype=np.float64)
    both = skt_finite & mb_finite
    only_skt = skt_finite & ~mb_finite
    only_mb = ~skt_finite & mb_finite
    fused[both] = w_skt * skt[both] + w_mb * mb[both]
    fused[only_skt] = skt[only_skt]
    fused[only_mb] = mb[only_mb]
    return fused


def compute_delta(angle: np.ndarray, k: int):
    """K-frame delta with NaN propagation."""
    delta = np.full_like(angle, np.nan, dtype=np.float64)
    if len(angle) > k:
        delta[k:] = angle[k:] - angle[:-k]
    return delta


def main() -> None:
    """Inject fused angle columns and write new CSV."""
    args = parse_args()
    rows, fieldnames = load_csv_rows(Path(args.input_csv))

    def col_floats(name: str) -> np.ndarray:
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows], dtype=np.float64)

    def col_bools(name: str) -> np.ndarray:
        return np.array([r[name] == "True" for r in rows], dtype=bool)

    ks = detect_k_list(fieldnames)
    print(f"[detect] K values: {ks}")
    print(f"[fuse] w_skt={args.w_skt}, w_mb={1.0 - args.w_skt}")

    new_columns: Dict[str, np.ndarray] = {}
    for side in SIDES:
        skt_angle = col_floats(f"SKT_{side}_deg")
        mb_angle = col_floats(f"MotionBert_{side}_deg")
        fused_angle = fuse_angles(skt_angle, mb_angle, args.w_skt)
        valid = np.isfinite(fused_angle)
        new_columns[f"{args.name}_{side}_deg"] = fused_angle
        new_columns[f"{args.name}_{side}_valid"] = valid
        new_columns[f"{args.name}_{side}_interpolated"] = np.zeros(len(fused_angle), dtype=bool)
        for k in ks:
            delta = compute_delta(fused_angle, k)
            valid_delta = np.zeros(len(fused_angle), dtype=bool)
            if len(fused_angle) > k:
                valid_delta[k:] = valid[k:] & valid[:-k]
            threshold = float(ANOMALY_THRESHOLDS.get(k, 60.0))
            anomaly = np.zeros(len(fused_angle), dtype=bool)
            anomaly[valid_delta] = np.abs(delta[valid_delta]) > threshold
            new_columns[f"{args.name}_{side}_delta_k{k}_deg"] = delta
            new_columns[f"{args.name}_{side}_delta_valid_k{k}"] = valid_delta
            new_columns[f"{args.name}_{side}_delta_anomaly_flag_k{k}"] = anomaly

    out_fields = list(fieldnames)
    for key in new_columns:
        if key not in out_fields:
            out_fields.append(key)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            new_row = dict(row)
            for key, arr in new_columns.items():
                value = arr[idx]
                if isinstance(value, (bool, np.bool_)):
                    new_row[key] = bool(value)
                elif isinstance(value, (float, np.floating)):
                    new_row[key] = "" if not np.isfinite(value) else f"{float(value):.6f}"
                else:
                    new_row[key] = value
            writer.writerow(new_row)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
