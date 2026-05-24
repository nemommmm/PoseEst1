"""Repair short low-quality SKT elbow-chain gaps.

This experiment tests a conservative alternative to trusting every triangulated
elbow frame. It uses existing SKT quality signals to flag suspicious frames in
the shoulder-elbow-wrist chain, then linearly interpolates only short bad
segments. Long bad segments are left untouched to avoid inventing motion.

The script is reference-free: Xsens is not used. The output is a repaired NPZ
plus before/after high-delta summaries for K=1 and K=6.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_METHOD_DIR = _SRC_DIR.parent
PROJECT_ROOT = _METHOD_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from pose_angle_utils import compute_semantic_angle_sequence  # noqa: E402


DEFAULT_INPUT = _METHOD_DIR / "results" / "historical_best_20260324" / "recovered_baseline" / "optimized_pose.npz"
DEFAULT_OUTPUT_DIR = _METHOD_DIR / "results" / "skt_model_fusion" / "elbow_quality_repair"
ELBOW_CHAINS = {
    "LeftElbow": np.array([5, 7, 9], dtype=np.int64),
    "RightElbow": np.array([6, 8, 10], dtype=np.int64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input SKT NPZ file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--max-gap-frames", type=int, default=5, help="Repair only bad segments up to this length.")
    parser.add_argument("--min-pair-conf", type=float, default=0.60)
    parser.add_argument("--min-stereo-quality", type=float, default=0.40)
    parser.add_argument("--max-epipolar-px", type=float, default=6.0)
    parser.add_argument("--max-reprojection-px", type=float, default=30.0)
    parser.add_argument("--high-delta-deg", type=float, default=35.0)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 6])
    return parser.parse_args()


def optional_array(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int]) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=np.float64)
    return np.full(shape, np.nan, dtype=np.float64)


def nanmin_chain(values: np.ndarray, joints: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    subset = values[:, joints]
    for idx, row in enumerate(subset):
        finite = row[np.isfinite(row)]
        if finite.size:
            result[idx] = float(np.min(finite))
    return result


def nanmax_chain(values: np.ndarray, joints: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    subset = values[:, joints]
    for idx, row in enumerate(subset):
        finite = row[np.isfinite(row)]
        if finite.size:
            result[idx] = float(np.max(finite))
    return result


def build_bad_mask(
    data: np.lib.npyio.NpzFile,
    keypoints: np.ndarray,
    joints: np.ndarray,
    min_pair_conf: float,
    min_stereo_quality: float,
    max_epipolar_px: float,
    max_reprojection_px: float,
) -> np.ndarray:
    shape = keypoints.shape[:2]
    conf_left = optional_array(data, "conf_left", shape)
    conf_right = optional_array(data, "conf_right", shape)
    pair_conf = optional_array(data, "pair_confidence", shape)
    if not np.isfinite(pair_conf).any():
        pair_conf = np.sqrt(np.clip(conf_left, 0.0, 1.0) * np.clip(conf_right, 0.0, 1.0))
    detect_conf = np.minimum(conf_left, conf_right)
    stereo_quality = optional_array(data, "stereo_quality", shape)
    epipolar_error = optional_array(data, "epipolar_error", shape)
    reprojection_error = optional_array(data, "reprojection_error", shape)

    chain_valid = np.isfinite(keypoints[:, joints]).all(axis=(1, 2))
    detect_min = nanmin_chain(detect_conf, joints)
    pair_min = nanmin_chain(pair_conf, joints)
    quality_min = nanmin_chain(stereo_quality, joints)
    epipolar_max = nanmax_chain(epipolar_error, joints)
    reproj_max = nanmax_chain(reprojection_error, joints)

    bad = ~chain_valid
    bad |= np.isfinite(detect_min) & (detect_min < min_pair_conf)
    bad |= np.isfinite(pair_min) & (pair_min < min_pair_conf)
    bad |= np.isfinite(quality_min) & (quality_min < min_stereo_quality)
    bad |= np.isfinite(epipolar_max) & (epipolar_max > max_epipolar_px)
    bad |= np.isfinite(reproj_max) & (reproj_max > max_reprojection_px)
    return bad


def iter_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        segments.append((start, idx))
    return segments


def repair_short_segments(
    keypoints: np.ndarray,
    timestamps: np.ndarray,
    joints: np.ndarray,
    bad_mask: np.ndarray,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    repaired = keypoints.copy()
    repaired_frame_mask = np.zeros(len(keypoints), dtype=bool)

    for start, end in iter_true_segments(bad_mask):
        if end - start > max_gap_frames:
            continue
        prev_idx = start - 1
        next_idx = end
        if prev_idx < 0 or next_idx >= len(keypoints):
            continue
        if bad_mask[prev_idx] or bad_mask[next_idx]:
            continue

        t0 = float(timestamps[prev_idx])
        t1 = float(timestamps[next_idx])
        if not np.isfinite(t0) or not np.isfinite(t1) or abs(t1 - t0) < 1e-9:
            continue

        segment_any_repaired = False
        for frame_idx in range(start, end):
            alpha = (float(timestamps[frame_idx]) - t0) / (t1 - t0)
            if not np.isfinite(alpha):
                continue
            for joint_idx in joints:
                before = keypoints[prev_idx, joint_idx]
                after = keypoints[next_idx, joint_idx]
                if not (np.isfinite(before).all() and np.isfinite(after).all()):
                    continue
                repaired[frame_idx, joint_idx] = (1.0 - alpha) * before + alpha * after
                segment_any_repaired = True
        if segment_any_repaired:
            repaired_frame_mask[start:end] = True

    return repaired, repaired_frame_mask


def angle_summary(keypoints: np.ndarray, k_values: list[int], high_delta_deg: float) -> list[dict[str, object]]:
    angle_names, angles = compute_semantic_angle_sequence(keypoints, wrist_smooth_radius=0)
    name_to_idx = {name: idx for idx, name in enumerate(angle_names)}
    rows: list[dict[str, object]] = []
    for angle_name in ("LeftElbow", "RightElbow"):
        angle_idx = name_to_idx[angle_name]
        values = angles[:, angle_idx]
        valid_angle_ratio = float(np.mean(np.isfinite(values)))
        for k in k_values:
            if len(values) <= k:
                continue
            delta = np.abs(values[k:] - values[:-k])
            valid = np.isfinite(delta)
            if not np.any(valid):
                continue
            finite_delta = delta[valid]
            rows.append(
                {
                    "angle": angle_name,
                    "k": int(k),
                    "valid_angle_ratio": valid_angle_ratio,
                    "n_delta_pairs": int(finite_delta.size),
                    "mean_delta_deg": float(np.mean(finite_delta)),
                    "p90_delta_deg": float(np.percentile(finite_delta, 90)),
                    "p95_delta_deg": float(np.percentile(finite_delta, 95)),
                    "high_delta_count": int(np.count_nonzero(finite_delta >= high_delta_deg)),
                    "high_delta_rate": float(np.mean(finite_delta >= high_delta_deg)),
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def write_markdown(
    before_rows: list[dict[str, object]],
    after_rows: list[dict[str, object]],
    output_path: Path,
    input_path: Path,
    left_bad: np.ndarray,
    right_bad: np.ndarray,
    left_repaired: np.ndarray,
    right_repaired: np.ndarray,
    args: argparse.Namespace,
) -> None:
    before_lookup = {(row["angle"], row["k"]): row for row in before_rows}
    after_lookup = {(row["angle"], row["k"]): row for row in after_rows}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# SKT Elbow Quality Repair Experiment\n\n")
        handle.write(f"- Input: `{input_path}`\n")
        handle.write("- Scope: reference-free internal SKT stability; Xsens is not used.\n")
        handle.write(
            f"- Gate: pair/detect conf < `{args.min_pair_conf}`, stereo quality < `{args.min_stereo_quality}`, "
            f"epipolar > `{args.max_epipolar_px}` px, reprojection > `{args.max_reprojection_px}` px.\n"
        )
        handle.write(f"- Repair: linear interpolation for bad segments up to `{args.max_gap_frames}` frames.\n\n")
        handle.write("## Repair coverage\n\n")
        handle.write("| Side | Bad frames | Bad ratio | Repaired frames | Repaired/bad |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for side, bad, repaired in (
            ("Left", left_bad, left_repaired),
            ("Right", right_bad, right_repaired),
        ):
            bad_count = int(np.count_nonzero(bad))
            repaired_count = int(np.count_nonzero(repaired))
            handle.write(
                f"| {side} | {bad_count} | {fmt(np.mean(bad))} | {repaired_count} | "
                f"{fmt(repaired_count / bad_count if bad_count else np.nan)} |\n"
            )

        handle.write("\n## Before vs after elbow deltas\n\n")
        handle.write(
            "| Angle | K | Before high-rate | After high-rate | Before p95 | After p95 | "
            "Before valid | After valid |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for key in sorted(before_lookup):
            before = before_lookup[key]
            after = after_lookup[key]
            handle.write(
                f"| {key[0]} | {key[1]} | {fmt(before['high_delta_rate'])} | "
                f"{fmt(after['high_delta_rate'])} | {fmt(before['p95_delta_deg'])} | "
                f"{fmt(after['p95_delta_deg'])} | {fmt(before['valid_angle_ratio'])} | "
                f"{fmt(after['valid_angle_ratio'])} |\n"
            )
        handle.write("\n## Interpretation guardrails\n\n")
        handle.write("- A lower high-delta rate is useful only if valid-angle coverage does not collapse.\n")
        handle.write("- This is not yet an accuracy result against Xsens-derived reference or FastSAM3D.\n")
        handle.write("- If this improves internal stability, the repaired NPZ should be rerun through frame-delta evaluation.\n")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = np.load(args.input, allow_pickle=True)
    keypoints = np.asarray(data["keypoints"], dtype=np.float64)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)

    left_bad = build_bad_mask(
        data,
        keypoints,
        ELBOW_CHAINS["LeftElbow"],
        args.min_pair_conf,
        args.min_stereo_quality,
        args.max_epipolar_px,
        args.max_reprojection_px,
    )
    right_bad = build_bad_mask(
        data,
        keypoints,
        ELBOW_CHAINS["RightElbow"],
        args.min_pair_conf,
        args.min_stereo_quality,
        args.max_epipolar_px,
        args.max_reprojection_px,
    )

    repaired_keypoints, left_repaired = repair_short_segments(
        keypoints,
        timestamps,
        ELBOW_CHAINS["LeftElbow"],
        left_bad,
        args.max_gap_frames,
    )
    repaired_keypoints, right_repaired = repair_short_segments(
        repaired_keypoints,
        timestamps,
        ELBOW_CHAINS["RightElbow"],
        right_bad,
        args.max_gap_frames,
    )

    payload = {key: data[key] for key in data.files}
    payload["keypoints"] = repaired_keypoints
    payload["keypoints_before_elbow_quality_repair"] = keypoints
    payload["elbow_quality_bad_left"] = left_bad
    payload["elbow_quality_bad_right"] = right_bad
    payload["elbow_quality_repaired_left"] = left_repaired
    payload["elbow_quality_repaired_right"] = right_repaired
    payload["elbow_quality_repair_config"] = np.array(
        [
            args.max_gap_frames,
            args.min_pair_conf,
            args.min_stereo_quality,
            args.max_epipolar_px,
            args.max_reprojection_px,
        ],
        dtype=np.float64,
    )
    payload["postprocess_variant"] = np.array("skt_elbow_quality_short_gap_repair")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repaired_path = args.output_dir / "skt_elbow_quality_repaired.npz"
    np.savez(repaired_path, **payload)

    before_rows = angle_summary(keypoints, args.k_values, args.high_delta_deg)
    after_rows = angle_summary(repaired_keypoints, args.k_values, args.high_delta_deg)
    comparison_rows = []
    for stage, rows in (("before", before_rows), ("after", after_rows)):
        for row in rows:
            comparison_rows.append({"stage": stage, **row})
    write_csv(comparison_rows, args.output_dir / "elbow_quality_repair_summary.csv")
    write_markdown(
        before_rows,
        after_rows,
        args.output_dir / "elbow_quality_repair_summary.md",
        args.input,
        left_bad,
        right_bad,
        left_repaired,
        right_repaired,
        args,
    )

    print(f"[Info] Wrote repaired pose: {repaired_path}")
    print(f"[Info] Wrote summary: {args.output_dir / 'elbow_quality_repair_summary.md'}")


if __name__ == "__main__":
    main()
