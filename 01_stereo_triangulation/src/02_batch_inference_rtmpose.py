#!/opt/anaconda3/envs/pose/bin/python
"""Minimal SKT stereo pipeline with RTMPose 2D detector.

For A2 ablation: replace YOLOv8m-pose with rtmlib RTMPose Body (17-kpt),
run left+right inference, triangulate, and emit an NPZ compatible with
the existing Phase 4 evaluation framework.

Notes
- Single-person assumption: keeps the highest-scoring detection per frame.
- No tracking, no crop refinement, no temporal rescue (intentional for
  fair detector-only ablation).
- Uses the same camera_params.npz as the existing YOLO pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from rtmlib import Body

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent
SHARED_DIR = PROJECT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

DEFAULT_CAMERA_PARAMS = SHARED_DIR / "camera_params_2025.npz"
DEFAULT_LEFT_VIDEO = PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.avi"
DEFAULT_RIGHT_VIDEO = PROJECT_ROOT / "2025_Ergonomics_Data" / "1_video_right.avi"
DEFAULT_LEFT_META = PROJECT_ROOT / "2025_Ergonomics_Data" / "0_video_left.txt"
DEFAULT_RIGHT_META = PROJECT_ROOT / "2025_Ergonomics_Data" / "1_video_right.txt"

COCO_NUM_KPTS = 17


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-params", default=str(DEFAULT_CAMERA_PARAMS))
    parser.add_argument("--left-video", default=str(DEFAULT_LEFT_VIDEO))
    parser.add_argument("--right-video", default=str(DEFAULT_RIGHT_VIDEO))
    parser.add_argument("--left-meta", default=str(DEFAULT_LEFT_META))
    parser.add_argument("--right-meta", default=str(DEFAULT_RIGHT_META))
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--mode", default="balanced", choices=("performance", "balanced", "lightweight"))
    parser.add_argument("--min-detection-conf", type=float, default=0.20)
    parser.add_argument("--limit-frames", type=int, default=None,
                        help="Optional cap on number of synced frame pairs to process.")
    return parser.parse_args()


def parse_meta(path: Path) -> list:
    """Parse stereo metadata txt with corrected microsecond handling."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append({
                    "id": int(parts[0]),
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
    if not synced:
        raise RuntimeError("No synchronized stereo metadata pairs found.")
    return synced


def select_best_person(keypoints: np.ndarray, scores: np.ndarray, min_conf: float):
    """Return the single best detection per frame, or NaN if none qualify."""
    if keypoints is None or len(keypoints) == 0:
        return None, None
    person_scores = scores.mean(axis=1)
    best = int(np.argmax(person_scores))
    if float(person_scores[best]) < min_conf:
        return None, None
    return keypoints[best].astype(np.float64), scores[best].astype(np.float64)


def undistort_points_to_normalized(points_2d, camera_matrix, dist_coeffs):
    """Convert raw pixel points to normalized image plane coordinates."""
    if points_2d.size == 0:
        return points_2d
    points_2d_reshaped = points_2d.reshape(-1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(points_2d_reshaped, camera_matrix, dist_coeffs)
    return undistorted.reshape(-1, 2)


def triangulate_points(points_left_norm, points_right_norm, R, T):
    """Triangulate from normalized image plane points."""
    P_l = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_r = np.hstack([R, T.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(
        P_l,
        P_r,
        points_left_norm.T.astype(np.float64),
        points_right_norm.T.astype(np.float64),
    )
    pts3 = (pts4[:3] / pts4[3]).T
    return pts3 * 100.0  # meters -> cm


def compute_reprojection_errors_cm(points_left_raw, points_right_raw, points_3d_cm,
                                    mtx_l, dist_l, mtx_r, dist_r, R, T):
    """Reproject the triangulated 3D points back to both views and measure pixel error."""
    pts3d_m = (points_3d_cm / 100.0).astype(np.float64)
    rvec_l = np.zeros(3, dtype=np.float64)
    tvec_l = np.zeros(3, dtype=np.float64)
    proj_l, _ = cv2.projectPoints(pts3d_m, rvec_l, tvec_l, mtx_l, dist_l)
    proj_l = proj_l.reshape(-1, 2)
    err_l = np.linalg.norm(proj_l - points_left_raw, axis=1)
    rvec_r, _ = cv2.Rodrigues(R)
    proj_r, _ = cv2.projectPoints(pts3d_m, rvec_r, T.reshape(3, 1).astype(np.float64), mtx_r, dist_r)
    proj_r = proj_r.reshape(-1, 2)
    err_r = np.linalg.norm(proj_r - points_right_raw, axis=1)
    return err_l, err_r


def compute_epipolar_errors(points_left_raw, points_right_raw, F):
    """Symmetric Sampson-like epipolar distance per keypoint."""
    ones = np.ones((len(points_left_raw), 1))
    pl = np.hstack([points_left_raw, ones])
    pr = np.hstack([points_right_raw, ones])
    lines_r = (F @ pl.T).T
    lines_l = (F.T @ pr.T).T
    norm_r = np.sqrt(lines_r[:, 0] ** 2 + lines_r[:, 1] ** 2) + 1e-9
    norm_l = np.sqrt(lines_l[:, 0] ** 2 + lines_l[:, 1] ** 2) + 1e-9
    dist_r = np.abs(np.sum(pr * lines_r, axis=1)) / norm_r
    dist_l = np.abs(np.sum(pl * lines_l, axis=1)) / norm_l
    return 0.5 * (dist_l + dist_r)


def main() -> None:
    """Run minimal RTMPose stereo pipeline."""
    args = parse_args()
    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] camera params: {args.camera_params}")
    cp = np.load(args.camera_params, allow_pickle=True)
    mtx_l = cp["mtx_l"]
    dist_l = cp["dist_l"]
    mtx_r = cp["mtx_r"]
    dist_r = cp["dist_r"]
    R = cp["R"]
    T = cp["T"].reshape(3, 1)
    F = cp["F"]

    print(f"[meta] parsing metadata")
    left_rows = parse_meta(Path(args.left_meta))
    right_rows = parse_meta(Path(args.right_meta))
    synced = build_synced_pairs(left_rows, right_rows)
    if args.limit_frames is not None:
        synced = synced[: int(args.limit_frames)]
    n = len(synced)
    print(f"[meta] {n} synced frame pairs")

    print(f"[rtmpose] loading Body model (mode={args.mode}, device={args.device})")
    t0 = time.time()
    body = Body(mode=args.mode, backend="onnxruntime", device=args.device)
    print(f"[rtmpose] loaded in {time.time()-t0:.1f}s")

    cap_l = cv2.VideoCapture(args.left_video)
    cap_r = cv2.VideoCapture(args.right_video)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError("Could not open one of the stereo videos.")

    timestamps = np.array([row["ts"] - synced[0]["ts"] for row in synced], dtype=np.float64)
    keypoints_3d = np.full((n, COCO_NUM_KPTS, 3), np.nan, dtype=np.float64)
    keypoints_left_2d = np.full((n, COCO_NUM_KPTS, 2), np.nan, dtype=np.float64)
    keypoints_right_2d = np.full((n, COCO_NUM_KPTS, 2), np.nan, dtype=np.float64)
    conf_left = np.full((n, COCO_NUM_KPTS), np.nan, dtype=np.float64)
    conf_right = np.full((n, COCO_NUM_KPTS), np.nan, dtype=np.float64)
    epi_err = np.full((n, COCO_NUM_KPTS), np.nan, dtype=np.float64)
    rep_err = np.full((n, COCO_NUM_KPTS), np.nan, dtype=np.float64)

    last_left_idx = -1
    last_right_idx = -1

    for i, row in enumerate(tqdm(synced, desc="rtmpose stereo")):
        # Seek to the correct frame if needed
        if row["left_idx"] != last_left_idx + 1:
            cap_l.set(cv2.CAP_PROP_POS_FRAMES, row["left_idx"])
        if row["right_idx"] != last_right_idx + 1:
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, row["right_idx"])
        last_left_idx = row["left_idx"]
        last_right_idx = row["right_idx"]

        ok_l, img_l = cap_l.read()
        ok_r, img_r = cap_r.read()
        if not ok_l or not ok_r:
            continue

        kpts_l_all, sc_l_all = body(img_l)
        kpts_r_all, sc_r_all = body(img_r)
        kpts_l, sc_l = select_best_person(kpts_l_all, sc_l_all, args.min_detection_conf)
        kpts_r, sc_r = select_best_person(kpts_r_all, sc_r_all, args.min_detection_conf)
        if kpts_l is None or kpts_r is None:
            continue

        keypoints_left_2d[i] = kpts_l
        keypoints_right_2d[i] = kpts_r
        conf_left[i] = sc_l
        conf_right[i] = sc_r

        l_norm = undistort_points_to_normalized(kpts_l, mtx_l, dist_l)
        r_norm = undistort_points_to_normalized(kpts_r, mtx_r, dist_r)
        pts3d = triangulate_points(l_norm, r_norm, R, T)
        keypoints_3d[i] = pts3d

        err_l, err_r = compute_reprojection_errors_cm(
            kpts_l, kpts_r, pts3d, mtx_l, dist_l, mtx_r, dist_r, R, T,
        )
        rep_err[i] = 0.5 * (err_l + err_r)
        epi_err[i] = compute_epipolar_errors(kpts_l, kpts_r, F)

    cap_l.release()
    cap_r.release()

    triang_conf_left = conf_left.copy()
    triang_conf_right = conf_right.copy()

    np.savez_compressed(
        out_path,
        timestamps=timestamps,
        keypoints=keypoints_3d,
        keypoints_left_2d=keypoints_left_2d,
        keypoints_right_2d=keypoints_right_2d,
        conf_left=conf_left,
        conf_right=conf_right,
        triang_conf_left=triang_conf_left,
        triang_conf_right=triang_conf_right,
        epipolar_error=epi_err,
        reprojection_error=rep_err,
        model_name=np.array("rtmpose-m_body17"),
    )

    valid_3d = np.mean(np.isfinite(keypoints_3d).all(axis=2))
    print(f"[saved] {out_path}")
    print(f"[stats] valid 3D keypoint fraction: {valid_3d:.4f}")


if __name__ == "__main__":
    main()
