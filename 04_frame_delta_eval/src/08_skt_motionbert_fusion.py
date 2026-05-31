#!/opt/anaconda3/envs/pose/bin/python
"""Confidence-weighted fusion of SKT and MotionBERT 3D keypoints.

For each frame and each COCO-17 keypoint, blend the SKT 3D position with the
MotionBERT 3D position using a per-keypoint weight derived from SKT stereo
quality (triangulation confidence + epipolar / reprojection error). Where SKT
fails (NaN or low quality), the fused output falls back to MotionBERT.

The output NPZ is compatible with the existing Phase 4 evaluation framework.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))


COCO17_NAMES = (
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
)
TRC_TO_COCO17 = {name: idx for idx, name in enumerate(COCO17_NAMES)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skt-npz",
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
        "--motionbert-trc",
        default=str(
            PROJECT_ROOT / "shared" / "recovered_methods" / "motionbert_markers_results_mono.trc"
        ),
    )
    parser.add_argument(
        "--left-meta",
        default=str(PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.txt"),
    )
    parser.add_argument(
        "--right-meta",
        default=str(PROJECT_ROOT / "2025_Ergonomics_Data" / "1_video_right.txt"),
    )
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--mb-baseline-conf", type=float, default=0.5,
                        help="MotionBERT baseline confidence; SKT exceeds this when quality is good.")
    parser.add_argument("--epi-tau", type=float, default=10.0,
                        help="Epipolar error scale (px) for exp(-epi/tau) decay.")
    parser.add_argument("--rep-tau", type=float, default=10.0,
                        help="Reprojection error scale (cm) for exp(-rep/tau) decay.")
    parser.add_argument("--min-skt-conf-floor", type=float, default=0.05,
                        help="Below this SKT quality, force fusion to use MotionBERT only.")
    return parser.parse_args()


def parse_meta(path: Path):
    """Parse stereo metadata txt with microsecond precision."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append({
                    "id": int(parts[0]),
                    "left_idx": None,
                    "ts": int(parts[1]) + int(parts[2]) * 1e-6,
                })
            except ValueError:
                continue
    return rows


def build_synced_pairs(left_rows, right_rows):
    """Match left and right rows on hardware frame id."""
    synced = []
    li, ri = 0, 0
    while li < len(left_rows) and ri < len(right_rows):
        l, r = left_rows[li], right_rows[ri]
        if l["id"] == r["id"]:
            synced.append({"frame_id": l["id"], "left_idx": li, "right_idx": ri, "ts": l["ts"]})
            li += 1
            ri += 1
        elif l["id"] < r["id"]:
            li += 1
        else:
            ri += 1
    return synced


def load_trc(trc_path: Path):
    """Parse a TRC file into timestamps, marker names, and positions."""
    with trc_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) < 7:
        raise ValueError(f"TRC too short: {trc_path}")
    header_values = lines[2].strip().split("\t")
    if len(header_values) < 5:
        header_values = lines[2].strip().split()
    num_markers = int(header_values[3])
    units = header_values[4]
    raw_names = lines[3].rstrip("\n").split("\t")[2:]
    marker_names = [name.strip() for name in raw_names if name.strip()]
    if len(marker_names) != num_markers:
        fallback = lines[3].strip().split()[2:]
        marker_names = [name.strip() for name in fallback if name.strip()]
    timestamps = []
    frames = []
    expected = num_markers * 3
    for line in lines[6:]:
        if not line.strip():
            continue
        values = line.rstrip("\n").split("\t")
        if len(values) < 2:
            values = line.strip().split()
        timestamps.append(float(values[1]))
        coords_raw = values[2:]
        if len(coords_raw) < expected:
            coords_raw = coords_raw + [""] * (expected - len(coords_raw))
        coords = [float(v) if v != "" else np.nan for v in coords_raw[:expected]]
        frames.append(coords)
    positions = np.asarray(frames, dtype=np.float64).reshape(-1, num_markers, 3)
    return np.asarray(timestamps, dtype=np.float64), marker_names, positions, units


def unit_scale_to_cm(units: str) -> float:
    """Convert TRC unit string to centimeter scale factor."""
    u = units.strip().lower()
    if u == "cm":
        return 1.0
    if u == "mm":
        return 0.1
    if u in {"m", "meter", "meters", "metre", "metres"}:
        return 100.0
    raise ValueError(f"Unsupported TRC unit: {units}")


def trc_to_coco17(marker_names, positions_cm):
    """Project TRC markers into COCO-17 keypoint order."""
    name_to_idx = {name: idx for idx, name in enumerate(marker_names)}
    out = np.full((positions_cm.shape[0], 17, 3), np.nan, dtype=np.float64)
    for name, coco_idx in TRC_TO_COCO17.items():
        if name in name_to_idx:
            out[:, coco_idx, :] = positions_cm[:, name_to_idx[name], :]
    return out


def align_mb_to_synced(mb_keypoints, n_synced, synced_meta):
    """Align MotionBERT (left-camera 3015 frames) to synced 2801 frames by left_idx."""
    if len(mb_keypoints) == n_synced:
        return mb_keypoints.copy()
    left_indices = np.asarray([row["left_idx"] for row in synced_meta], dtype=np.int64)
    if np.any(left_indices >= len(mb_keypoints)):
        raise RuntimeError("MotionBERT TRC has fewer frames than required left_idx range.")
    return mb_keypoints[left_indices]


def compute_skt_quality(payload, n_synced):
    """Compute per-frame per-keypoint SKT quality score in [0, 1]."""
    triang_l = np.asarray(payload["triang_conf_left"], dtype=np.float64)
    triang_r = np.asarray(payload["triang_conf_right"], dtype=np.float64)
    epi = np.asarray(payload["epipolar_error"], dtype=np.float64)
    rep = np.asarray(payload["reprojection_error"], dtype=np.float64)
    if len(triang_l) != n_synced:
        n = min(len(triang_l), n_synced)
        triang_l = triang_l[:n]
        triang_r = triang_r[:n]
        epi = epi[:n]
        rep = rep[:n]
    conf = np.minimum(triang_l, triang_r)
    conf = np.where(np.isfinite(conf), conf, 0.0)
    epi = np.where(np.isfinite(epi), epi, 1e6)
    rep = np.where(np.isfinite(rep), rep, 1e6)
    return conf, epi, rep


ANCHOR_KPT_INDICES = (5, 6, 11, 12)  # LShoulder, RShoulder, LHip, RHip


def align_mb_to_skt_per_frame(skt, mb):
    """Translate MotionBERT keypoints into SKT camera frame using torso anchors.

    For each frame, compute the median translation between SKT and MB anchor
    keypoints (shoulders + hips) where both are finite, and apply that
    translation to all MB keypoints. Returns the aligned MB array (or NaN
    where no anchor pair is available).
    """
    n = len(skt)
    aligned = np.full_like(mb, np.nan, dtype=np.float64)
    for i in range(n):
        offsets = []
        for j in ANCHOR_KPT_INDICES:
            if np.isfinite(skt[i, j]).all() and np.isfinite(mb[i, j]).all():
                offsets.append(skt[i, j] - mb[i, j])
        if not offsets:
            continue
        translation = np.median(np.asarray(offsets), axis=0)
        aligned[i] = mb[i] + translation
    return aligned


def fuse_keypoints(skt_kp, mb_kp, skt_conf, epi, rep, args):
    """Combine SKT and MotionBERT keypoints with per-keypoint weighted fusion.

    MotionBERT keypoints are first translated to the SKT camera frame using
    torso anchor points (shoulders + hips), then weighted by SKT quality.
    """
    n = min(len(skt_kp), len(mb_kp))
    skt = skt_kp[:n]
    mb_raw = mb_kp[:n]
    skt_conf = skt_conf[:n]
    epi = epi[:n]
    rep = rep[:n]

    mb = align_mb_to_skt_per_frame(skt, mb_raw)

    epi_w = np.exp(-epi / float(args.epi_tau))
    rep_w = np.exp(-rep / float(args.rep_tau))
    skt_quality = skt_conf * epi_w * rep_w
    skt_quality = np.clip(skt_quality, 0.0, 1.0)

    skt_finite = np.isfinite(skt).all(axis=2)
    mb_finite = np.isfinite(mb).all(axis=2)

    skt_quality = np.where(skt_finite, skt_quality, 0.0)
    floor = float(args.min_skt_conf_floor)
    skt_quality = np.where(skt_quality < floor, 0.0, skt_quality)

    mb_quality = np.full_like(skt_quality, float(args.mb_baseline_conf))
    mb_quality = np.where(mb_finite, mb_quality, 0.0)

    denom = skt_quality + mb_quality
    safe = denom > 1e-9
    w_skt = np.zeros_like(skt_quality)
    w_mb = np.zeros_like(skt_quality)
    w_skt[safe] = skt_quality[safe] / denom[safe]
    w_mb[safe] = mb_quality[safe] / denom[safe]

    fused = np.full_like(skt, np.nan, dtype=np.float64)
    skt_only = ~mb_finite & skt_finite
    mb_only = ~skt_finite & mb_finite
    both = skt_finite & mb_finite

    fused[skt_only] = skt[skt_only]
    fused[mb_only] = mb[mb_only]
    fused[both] = (
        w_skt[both, None] * skt[both]
        + w_mb[both, None] * mb[both]
    )
    return fused, skt_quality, w_skt, w_mb


def main() -> None:
    """Run fusion."""
    args = parse_args()
    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] SKT: {args.skt_npz}")
    skt_payload = np.load(args.skt_npz, allow_pickle=True)
    skt_kp = np.asarray(skt_payload["keypoints"], dtype=np.float64)
    timestamps = np.asarray(skt_payload["timestamps"], dtype=np.float64)
    n_skt = len(skt_kp)
    print(f"[load] SKT keypoints: {skt_kp.shape}, timestamps: {timestamps.shape}")

    print(f"[load] MotionBERT TRC: {args.motionbert_trc}")
    mb_ts, mb_markers, mb_positions, mb_units = load_trc(Path(args.motionbert_trc))
    print(f"[load] MotionBERT frames: {len(mb_ts)}, markers: {len(mb_markers)}, units: {mb_units}")
    mb_positions_cm = mb_positions * unit_scale_to_cm(mb_units)
    mb_coco17 = trc_to_coco17(mb_markers, mb_positions_cm)

    print("[align] aligning MotionBERT to SKT synced timeline")
    left_rows = parse_meta(Path(args.left_meta))
    right_rows = parse_meta(Path(args.right_meta))
    synced = build_synced_pairs(left_rows, right_rows)[:n_skt]
    if len(mb_coco17) == n_skt:
        mb_aligned = mb_coco17.copy()
        align_mode = "synced_index"
    elif len(mb_coco17) == len(left_rows):
        mb_aligned = align_mb_to_synced(mb_coco17, n_skt, synced)
        align_mode = "left_metadata_index"
    else:
        raise RuntimeError(
            f"MotionBERT frame count {len(mb_coco17)} matches neither SKT timeline "
            f"({n_skt}) nor left metadata ({len(left_rows)})."
        )
    print(f"[align] mode={align_mode}, MotionBERT aligned shape={mb_aligned.shape}")

    print("[quality] computing SKT per-keypoint quality")
    skt_conf, epi, rep = compute_skt_quality(skt_payload, n_skt)

    print("[fuse] running fusion")
    fused, skt_quality, w_skt, w_mb = fuse_keypoints(skt_kp, mb_aligned, skt_conf, epi, rep, args)

    valid_skt = float(np.mean(np.isfinite(skt_kp).all(axis=2)))
    valid_mb = float(np.mean(np.isfinite(mb_aligned).all(axis=2)))
    valid_fused = float(np.mean(np.isfinite(fused).all(axis=2)))
    avg_w_skt = float(np.nanmean(w_skt[np.isfinite(skt_kp).all(axis=2) & np.isfinite(mb_aligned).all(axis=2)]))
    print(f"[stats] valid fraction: SKT={valid_skt:.4f}, MB={valid_mb:.4f}, FUSED={valid_fused:.4f}")
    print(f"[stats] mean w_skt where both finite: {avg_w_skt:.4f}")

    out_payload = {key: skt_payload[key] for key in skt_payload.files}
    out_payload["keypoints"] = fused
    out_payload["fusion_w_skt"] = w_skt.astype(np.float32)
    out_payload["fusion_w_mb"] = w_mb.astype(np.float32)
    out_payload["fusion_skt_quality"] = skt_quality.astype(np.float32)
    out_payload["fusion_mb_baseline_conf"] = np.array(args.mb_baseline_conf, dtype=np.float64)
    out_payload["fusion_epi_tau"] = np.array(args.epi_tau, dtype=np.float64)
    out_payload["fusion_rep_tau"] = np.array(args.rep_tau, dtype=np.float64)
    out_payload["fusion_min_skt_floor"] = np.array(args.min_skt_conf_floor, dtype=np.float64)
    np.savez_compressed(out_path, **out_payload)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
