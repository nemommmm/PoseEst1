import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pose_postprocess import estimate_bone_priors, postprocess_sequence
from utils import StereoDataLoader
from utils_mvnx import MvnxParser


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")
ALIGNMENT_SUMMARY_PATH = os.path.join(RESULTS_DIR, "alignment_summary.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEGMENT_LABEL = os.environ.get("POSE_COMPARE_SEGMENT", "Walking (Normal)")
ACTIVITY_SEGMENTS = {
    "Walking (Normal)": (17.0, 32.0),
}

MODEL_SPECS = os.environ.get(
    "POSE_COMPARE_MODELS",
    "ultralytics:yolov8m-pose.pt,ultralytics:yolov8l-pose.pt",
)

MIN_KEYPOINT_CONF = 0.35
REPROJECTION_ERROR_THRESHOLD = 80.0
HIGH_CONF_THRESHOLD = 0.70
OFFSET_SEARCH_RADIUS = 1.0
OFFSET_STEP_SECONDS = 0.05
TOP_K = 150
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]

JOINT_MAPPING = {
    0: "Head",
    11: "Pelvis",
    12: "Pelvis",
    5: "LeftShoulder",
    6: "RightShoulder",
    7: "LeftUpperArm",
    8: "RightUpperArm",
    9: "LeftForeArm",
    10: "RightForeArm",
    13: "LeftUpperLeg",
    14: "RightUpperLeg",
    15: "LeftLowerLeg",
    16: "RightLowerLeg",
}

JOINT_NAMES = {
    0: "Nose",
    1: "LeftEye",
    2: "RightEye",
    3: "LeftEar",
    4: "RightEar",
    5: "LeftShoulder",
    6: "RightShoulder",
    7: "LeftElbow",
    8: "RightElbow",
    9: "LeftWrist",
    10: "RightWrist",
    11: "LeftHip",
    12: "RightHip",
    13: "LeftKnee",
    14: "RightKnee",
    15: "LeftAnkle",
    16: "RightAnkle",
}

MEDIAPIPE_TO_COCO17 = {
    0: 0,   # nose
    2: 1,   # left eye
    5: 2,   # right eye
    7: 3,   # left ear
    8: 4,   # right ear
    11: 5,  # left shoulder
    12: 6,  # right shoulder
    13: 7,  # left elbow
    14: 8,  # right elbow
    15: 9,  # left wrist
    16: 10, # right wrist
    23: 11, # left hip
    24: 12, # right hip
    25: 13, # left knee
    26: 14, # right knee
    27: 15, # left ankle
    28: 16, # right ankle
}


def parse_model_specs():
    specs = []
    for raw_item in MODEL_SPECS.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            backend, model_name = item.split(":", 1)
        else:
            backend, model_name = "ultralytics", item
        model_path = model_name
        if not os.path.isabs(model_path):
            model_path = os.path.join(SRC_DIR, model_name)
        specs.append(
            {
                "backend": backend.strip(),
                "model_name": os.path.basename(model_name.strip()),
                "model_path": model_path,
            }
        )
    return specs


def find_first_synced_timestamp(left_data, right_data):
    ptr_l = 0
    ptr_r = 0
    while ptr_l < len(left_data) and ptr_r < len(right_data):
        meta_l = left_data[ptr_l]
        meta_r = right_data[ptr_r]
        if meta_l["id"] == meta_r["id"]:
            return float(meta_l["ts"])
        if meta_l["id"] < meta_r["id"]:
            ptr_l += 1
        else:
            ptr_r += 1
    raise RuntimeError("Failed to locate the first synchronized stereo frame.")


def count_segment_pairs(left_data, right_data, first_ts, start_s, end_s):
    ptr_l = 0
    ptr_r = 0
    matches = 0
    while ptr_l < len(left_data) and ptr_r < len(right_data):
        meta_l = left_data[ptr_l]
        meta_r = right_data[ptr_r]
        if meta_l["id"] == meta_r["id"]:
            rel_ts = float(meta_l["ts"] - first_ts)
            if start_s <= rel_ts <= end_s:
                matches += 1
            if rel_ts > end_s:
                break
            ptr_l += 1
            ptr_r += 1
        elif meta_l["id"] < meta_r["id"]:
            ptr_l += 1
        else:
            ptr_r += 1
    return matches


def compute_rectified_reprojection_error(P, pts_3d, pts_2d_rect):
    reproj_error = np.full(len(pts_3d), np.nan, dtype=np.float64)
    valid_idx = np.where(np.isfinite(pts_3d).all(axis=1))[0]
    if len(valid_idx) == 0:
        return reproj_error

    pts_h = np.hstack([pts_3d[valid_idx], np.ones((len(valid_idx), 1), dtype=np.float64)])
    proj = (P @ pts_h.T).T
    valid_depth = np.abs(proj[:, 2]) > 1e-8
    if not np.any(valid_depth):
        return reproj_error

    proj = proj[valid_depth]
    valid_idx = valid_idx[valid_depth]
    proj_xy = proj[:, :2] / proj[:, 2:3]
    gt_xy = pts_2d_rect[valid_idx, 0, :]
    reproj_error[valid_idx] = np.linalg.norm(proj_xy - gt_xy, axis=1)
    return reproj_error


def calculate_limb_error(kpts, gt_lengths):
    lengths = np.vstack(
        [
            np.linalg.norm(kpts[:, 11] - kpts[:, 13], axis=1),
            np.linalg.norm(kpts[:, 12] - kpts[:, 14], axis=1),
            np.linalg.norm(kpts[:, 13] - kpts[:, 15], axis=1),
            np.linalg.norm(kpts[:, 14] - kpts[:, 16], axis=1),
        ]
    ).T
    return np.sum(np.abs(lengths - gt_lengths), axis=1)


def kabsch_transform(P, Q):
    mask = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P = P[mask]
    Q = Q[mask]
    if len(P) < 10:
        return np.eye(3), np.zeros(3)

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    AA = P - centroid_P
    BB = Q - centroid_Q
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    rot = Vt.T @ U.T
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
    t = centroid_Q - rot @ centroid_P
    return rot, t


def resolve_best_offset():
    if os.path.exists(ALIGNMENT_SUMMARY_PATH):
        with open(ALIGNMENT_SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("best_offset_seconds", 17.20))
    return 17.20


def build_rectification():
    data = np.load(PARAM_PATH)
    mtx_l, dist_l = data["mtx_l"], data["dist_l"]
    mtx_r, dist_r = data["mtx_r"], data["dist_r"]
    R, T = data["R"], data["T"]

    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"),
        os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"),
        os.path.join(DATA_DIR, "1_video_right.txt"),
    )
    first_ts = find_first_synced_timestamp(loader.left_data, loader.right_data)
    frame_l, _, _, _ = loader.get_next_pair()
    if frame_l is None:
        raise RuntimeError("Failed to read the stereo videos.")
    h, w = frame_l.shape[:2]
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0)
    loader.release()
    return {
        "mtx_l": mtx_l,
        "dist_l": dist_l,
        "mtx_r": mtx_r,
        "dist_r": dist_r,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "first_ts": first_ts,
    }


def load_xsens():
    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    xsens_ts = mvnx.timestamps
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts = xsens_ts - xsens_ts[0]
    xsens_interp = {}
    for seg_name in sorted(set(JOINT_MAPPING.values())):
        seg_data = mvnx.get_segment_data(seg_name)[xidx]
        xsens_interp[seg_name] = interp1d(
            xsens_ts,
            seg_data,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    return xsens_interp


def create_runner(spec):
    backend = spec["backend"]
    model_path = spec["model_path"]
    if backend == "ultralytics":
        from ultralytics import YOLO

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return YOLO(model_path)
    if backend == "mediapipe":
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe backend requested but mediapipe is not installed.") from exc
        complexity_alias = {
            "lite": 0,
            "full": 1,
            "heavy": 2,
            "pose_lite": 0,
            "pose_full": 1,
            "pose_heavy": 2,
        }
        complexity = complexity_alias.get(spec["model_name"].lower(), 2)
        return mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )
    if backend == "rtmlib":
        from rtmlib import Body

        mode_alias = {
            "lightweight": "lightweight",
            "balanced": "balanced",
            "performance": "performance",
            "s": "lightweight",
            "m": "balanced",
            "x": "performance",
        }
        mode = mode_alias.get(spec["model_name"].lower(), "balanced")
        return Body(
            mode=mode,
            to_openpose=False,
            backend="onnxruntime",
            device="cpu",
        )
    raise RuntimeError(
        f"Unsupported backend '{backend}'. Add a runner in create_runner()/run_pose_model() after installing the backend."
    )


def run_pose_model(runner, backend, frame):
    if backend == "ultralytics":
        result = runner(frame, verbose=False, conf=0.5)[0]
        if len(result.keypoints) == 0:
            return None, None
        pts = result.keypoints.xy[0].cpu().numpy().reshape(-1, 1, 2)
        conf = result.keypoints.conf[0].cpu().numpy()
        return pts, conf

    if backend == "mediapipe":
        result = runner.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None, None
        pts = np.full((17, 1, 2), np.nan, dtype=np.float64)
        conf = np.full(17, np.nan, dtype=np.float64)
        frame_h, frame_w = frame.shape[:2]
        landmarks = result.pose_landmarks.landmark
        for mp_idx, coco_idx in MEDIAPIPE_TO_COCO17.items():
            lm = landmarks[mp_idx]
            pts[coco_idx, 0, 0] = lm.x * frame_w
            pts[coco_idx, 0, 1] = lm.y * frame_h
            conf[coco_idx] = float(lm.visibility)
        return pts, conf

    if backend == "rtmlib":
        keypoints, scores = runner(frame)
        if keypoints is None or scores is None or len(keypoints) == 0:
            return None, None
        if keypoints.ndim != 3 or scores.ndim != 2:
            return None, None
        best_idx = int(np.nanargmax(np.nanmean(scores, axis=1)))
        pts = keypoints[best_idx].reshape(-1, 1, 2).astype(np.float64)
        conf = scores[best_idx].astype(np.float64)
        return pts, conf

    raise RuntimeError(f"Unsupported backend '{backend}'.")


def process_segment_for_model(spec, rectification):
    runner = create_runner(spec)
    start_s, end_s = ACTIVITY_SEGMENTS[SEGMENT_LABEL]

    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"),
        os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"),
        os.path.join(DATA_DIR, "1_video_right.txt"),
    )
    total_pairs = count_segment_pairs(loader.left_data, loader.right_data, rectification["first_ts"], start_s, end_s)

    timestamps = []
    keypoints_3d = []
    reprojection_errors = []
    keypoints_left_2d = []
    keypoints_right_2d = []
    conf_left = []
    conf_right = []

    print(f"[Info] Comparing {spec['model_name']} on '{SEGMENT_LABEL}' ({start_s:.1f}s-{end_s:.1f}s)...")
    start_wall = time.perf_counter()
    pbar = tqdm(
        total=total_pairs if total_pairs > 0 else None,
        desc=spec["model_name"],
        unit="frame",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    while True:
        frame_l, frame_r, _, ts = loader.get_next_pair()
        if frame_l is None:
            break

        rel_ts = float(ts - rectification["first_ts"])
        if rel_ts < start_s:
            continue
        if rel_ts > end_s:
            break

        pts_l_xy = np.full((17, 2), np.nan, dtype=np.float64)
        pts_r_xy = np.full((17, 2), np.nan, dtype=np.float64)
        conf_l_out = np.full(17, np.nan, dtype=np.float64)
        conf_r_out = np.full(17, np.nan, dtype=np.float64)
        kpts_3d = np.full((17, 3), np.nan, dtype=np.float64)
        reproj_error = np.full(17, np.nan, dtype=np.float64)

        pts_l, conf_l = run_pose_model(runner, spec["backend"], frame_l)
        pts_r, conf_r = run_pose_model(runner, spec["backend"], frame_r)

        if pts_l is not None and pts_r is not None:
            pts_l_xy = pts_l[:, 0, :].astype(np.float64)
            pts_r_xy = pts_r[:, 0, :].astype(np.float64)
            conf_l_out = conf_l.astype(np.float64)
            conf_r_out = conf_r.astype(np.float64)
            valid_mask = (conf_l >= MIN_KEYPOINT_CONF) & (conf_r >= MIN_KEYPOINT_CONF)

            pts_l_rect = cv2.undistortPoints(
                pts_l,
                rectification["mtx_l"],
                rectification["dist_l"],
                R=rectification["R1"],
                P=rectification["P1"],
            )
            pts_r_rect = cv2.undistortPoints(
                pts_r,
                rectification["mtx_r"],
                rectification["dist_r"],
                R=rectification["R2"],
                P=rectification["P2"],
            )
            pts4d = cv2.triangulatePoints(rectification["P1"], rectification["P2"], pts_l_rect, pts_r_rect)
            w_vec = pts4d[3, :]
            valid_w = w_vec != 0
            pts3d_raw = np.full((3, 17), np.nan, dtype=np.float64)
            pts3d_raw[:, valid_w] = pts4d[:3, valid_w] / w_vec[valid_w]
            kpts_3d = pts3d_raw.T
            kpts_3d[~valid_mask] = np.nan

            reproj_l = compute_rectified_reprojection_error(rectification["P1"], kpts_3d, pts_l_rect)
            reproj_r = compute_rectified_reprojection_error(rectification["P2"], kpts_3d, pts_r_rect)
            reproj_error = np.nanmean(np.vstack([reproj_l, reproj_r]), axis=0)
            valid_reproj = np.isfinite(reproj_error) & (reproj_error <= REPROJECTION_ERROR_THRESHOLD)
            kpts_3d[~valid_reproj] = np.nan

        timestamps.append(rel_ts)
        keypoints_3d.append(kpts_3d)
        reprojection_errors.append(reproj_error)
        keypoints_left_2d.append(pts_l_xy)
        keypoints_right_2d.append(pts_r_xy)
        conf_left.append(conf_l_out)
        conf_right.append(conf_r_out)
        pbar.update(1)
        processed = len(timestamps)
        elapsed = max(time.perf_counter() - start_wall, 1e-6)
        fps = processed / elapsed
        eta_text = "--:--"
        if total_pairs > 0 and fps > 1e-6:
            eta_seconds = max(total_pairs - processed, 0) / fps
            eta_text = time.strftime("%M:%S", time.gmtime(int(eta_seconds)))
        pbar.set_postfix(fps=f"{fps:.2f}", eta=eta_text)

    loader.release()
    pbar.close()
    if hasattr(runner, "close"):
        runner.close()

    timestamps = np.asarray(timestamps, dtype=np.float64)
    keypoints_3d = np.asarray(keypoints_3d, dtype=np.float64)
    reprojection_errors = np.asarray(reprojection_errors, dtype=np.float64)
    keypoints_left_2d = np.asarray(keypoints_left_2d, dtype=np.float64)
    keypoints_right_2d = np.asarray(keypoints_right_2d, dtype=np.float64)
    conf_left = np.asarray(conf_left, dtype=np.float64)
    conf_right = np.asarray(conf_right, dtype=np.float64)
    pair_confidence = np.nanmean(np.stack([conf_left, conf_right], axis=0), axis=0)

    priors = estimate_bone_priors(keypoints_3d, timestamps)
    optimized = postprocess_sequence(
        keypoints_3d,
        timestamps,
        priors=priors,
        reprojection_errors=reprojection_errors,
        pair_confidence=pair_confidence,
        floor_axis=None,
        floor_value=None,
        enable_bone_constraint=True,
        enable_quality_blend=False,
        enable_one_euro=True,
    )

    safe_model_name = spec["model_name"].replace("-", "_").replace(".", "_")
    npz_path = os.path.join(RESULTS_DIR, f"model_compare_{safe_model_name}_{SEGMENT_LABEL.lower().replace(' ', '_').replace('(', '').replace(')', '')}.npz")
    np.savez(
        npz_path,
        timestamps=timestamps,
        keypoints_raw=keypoints_3d,
        keypoints_optimized=optimized,
        reprojection_error=reprojection_errors,
        keypoints_left_2d=keypoints_left_2d,
        keypoints_right_2d=keypoints_right_2d,
        conf_left=conf_left,
        conf_right=conf_right,
    )

    return {
        "timestamps": timestamps,
        "keypoints_raw": keypoints_3d,
        "keypoints_optimized": optimized,
        "reprojection_error": reprojection_errors,
        "pair_confidence": pair_confidence,
        "output_npz": npz_path,
    }


def evaluate_model_result(spec, result_data, xsens_interp, global_best_offset):
    y_ts = result_data["timestamps"]
    y_kpts = result_data["keypoints_optimized"]
    reprojection_error = result_data["reprojection_error"]
    pair_confidence = result_data["pair_confidence"]
    y_center = 0.5 * (y_kpts[:, 11] + y_kpts[:, 12])

    valid_root_mask = (y_center[:, 2] > 10) & (y_center[:, 2] < 1000) & np.isfinite(y_center).all(axis=1)
    y_ts_eval = y_ts[valid_root_mask]
    y_kpts_eval = y_kpts[valid_root_mask]
    y_center_eval = y_center[valid_root_mask]

    search_offsets = np.arange(
        global_best_offset - OFFSET_SEARCH_RADIUS,
        global_best_offset + OFFSET_SEARCH_RADIUS + 0.5 * OFFSET_STEP_SECONDS,
        OFFSET_STEP_SECONDS,
    )
    errors = calculate_limb_error(y_kpts_eval, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    local_top_k = min(TOP_K, len(valid_err_idx))
    if local_top_k == 0:
        raise RuntimeError(f"No valid elite frames for {spec['model_name']}.")
    elite_indices = valid_err_idx[np.argsort(errors[valid_err_idx])[:local_top_k]]
    p_elite = y_center_eval[elite_indices]

    best_shift = np.nan
    best_root_error = np.inf
    best_R = np.eye(3)
    best_t = np.zeros(3)
    for shift in search_offsets:
        q_elite = xsens_interp["Pelvis"](y_ts_eval[elite_indices] - shift)
        R_mat, t_vec = kabsch_transform(p_elite, q_elite)
        y_aligned = (R_mat @ y_center_eval.T).T + t_vec
        x_gt = xsens_interp["Pelvis"](y_ts_eval - shift)
        dist = np.linalg.norm(y_aligned - x_gt, axis=1)
        valid_dist = dist[np.isfinite(dist)]
        if len(valid_dist) < 30:
            continue
        mean_dist = float(np.mean(valid_dist))
        if mean_dist < best_root_error:
            best_root_error = mean_dist
            best_shift = float(shift)
            best_R = R_mat
            best_t = t_vec

    y_kpts_flat = y_kpts.reshape(-1, 3)
    y_kpts_aligned = ((best_R @ y_kpts_flat.T).T + best_t).reshape(y_kpts.shape)

    joint_errors = []
    root_errors = []
    for i, curr_t in enumerate(y_ts):
        target_t = curr_t - best_shift
        gt_pelvis = xsens_interp["Pelvis"](target_t)
        y_root = 0.5 * (y_kpts_aligned[i, 11] + y_kpts_aligned[i, 12])
        if np.isfinite(gt_pelvis).all() and np.isfinite(y_root).all():
            root_errors.append(np.linalg.norm(y_root - gt_pelvis))

        for y_idx, x_name in JOINT_MAPPING.items():
            gt_pos = xsens_interp[x_name](target_t)
            est_pos = y_kpts_aligned[i, y_idx]
            if np.isfinite(gt_pos).all() and np.isfinite(est_pos).all():
                joint_errors.append(
                    {
                        "model": spec["model_name"],
                        "joint": JOINT_NAMES[y_idx],
                        "proxy": x_name,
                        "error_cm": float(np.linalg.norm(est_pos - gt_pos)),
                    }
                )

    df_joint = pd.DataFrame(joint_errors)
    per_joint_quality = []
    for joint_idx in range(17):
        joint_reproj = reprojection_error[:, joint_idx]
        joint_conf = pair_confidence[:, joint_idx]
        valid_mask = np.isfinite(result_data["keypoints_raw"][:, joint_idx]).all(axis=1)
        confident_bad = np.isfinite(joint_reproj) & np.isfinite(joint_conf) & (joint_reproj > REPROJECTION_ERROR_THRESHOLD) & (joint_conf >= HIGH_CONF_THRESHOLD)
        per_joint_quality.append(
            {
                "model": spec["model_name"],
                "joint": JOINT_NAMES[joint_idx],
                "valid_ratio": float(np.mean(valid_mask)),
                "pair_confidence_mean": float(np.nanmean(joint_conf)) if np.isfinite(joint_conf).any() else np.nan,
                "reprojection_mean_px": float(np.nanmean(joint_reproj)) if np.isfinite(joint_reproj).any() else np.nan,
                "reprojection_p95_px": float(np.nanpercentile(joint_reproj, 95)) if np.isfinite(joint_reproj).any() else np.nan,
                "high_conf_bad_ratio": float(np.mean(confident_bad)),
            }
        )

    raw_valid_ratio = np.mean(np.isfinite(result_data["keypoints_raw"]).all(axis=2))
    opt_valid_ratio = np.mean(np.isfinite(result_data["keypoints_optimized"]).all(axis=2))
    finite_reproj = reprojection_error[np.isfinite(reprojection_error)]
    finite_conf = pair_confidence[np.isfinite(pair_confidence)]
    confident_bad_all = (
        np.isfinite(reprojection_error)
        & np.isfinite(pair_confidence)
        & (reprojection_error > REPROJECTION_ERROR_THRESHOLD)
        & (pair_confidence >= HIGH_CONF_THRESHOLD)
    )

    metrics = {
        "model": spec["model_name"],
        "backend": spec["backend"],
        "segment": SEGMENT_LABEL,
        "frames": int(len(y_ts)),
        "best_offset_s": best_shift,
        "root_mean_error_cm": float(np.mean(root_errors)) if root_errors else np.nan,
        "mpjpe_cm": float(df_joint["error_cm"].mean()) if not df_joint.empty else np.nan,
        "median_joint_error_cm": float(df_joint["error_cm"].median()) if not df_joint.empty else np.nan,
        "raw_valid_joint_ratio": float(raw_valid_ratio),
        "optimized_valid_joint_ratio": float(opt_valid_ratio),
        "pair_confidence_mean": float(np.mean(finite_conf)) if finite_conf.size > 0 else np.nan,
        "reprojection_mean_px": float(np.mean(finite_reproj)) if finite_reproj.size > 0 else np.nan,
        "reprojection_p90_px": float(np.percentile(finite_reproj, 90)) if finite_reproj.size > 0 else np.nan,
        "reprojection_p95_px": float(np.percentile(finite_reproj, 95)) if finite_reproj.size > 0 else np.nan,
        "high_conf_bad_ratio": float(np.mean(confident_bad_all)) if np.size(confident_bad_all) > 0 else np.nan,
        "valid_joint_samples": int(len(df_joint)),
        "valid_root_samples": int(len(root_errors)),
    }
    return metrics, pd.DataFrame(per_joint_quality)


def main():
    if SEGMENT_LABEL not in ACTIVITY_SEGMENTS:
        raise KeyError(f"Unknown segment label: {SEGMENT_LABEL}")

    print("[Info] Building short-segment pose model comparison...")
    print(f"[Info] Segment: {SEGMENT_LABEL}")
    print(f"[Info] Models: {MODEL_SPECS}")
    best_offset = resolve_best_offset()
    rectification = build_rectification()
    xsens_interp = load_xsens()

    model_specs = parse_model_specs()
    summary_rows = []
    joint_quality_frames = []
    for spec in model_specs:
        result_data = process_segment_for_model(spec, rectification)
        metrics, df_joint_quality = evaluate_model_result(spec, result_data, xsens_interp, best_offset)
        summary_rows.append(metrics)
        joint_quality_frames.append(df_joint_quality)

    summary_df = pd.DataFrame(summary_rows).sort_values("mpjpe_cm")
    joint_quality_df = pd.concat(joint_quality_frames, ignore_index=True)

    segment_slug = SEGMENT_LABEL.lower().replace(" ", "_").replace("(", "").replace(")", "")
    summary_csv_path = os.path.join(RESULTS_DIR, f"model_compare_{segment_slug}_summary.csv")
    joint_csv_path = os.path.join(RESULTS_DIR, f"model_compare_{segment_slug}_joint_quality.csv")
    summary_json_path = os.path.join(RESULTS_DIR, f"model_compare_{segment_slug}_summary.json")

    summary_df.to_csv(summary_csv_path, index=False)
    joint_quality_df.to_csv(joint_csv_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\n[Result] Model comparison summary")
    print("-" * 72)
    print(summary_df.to_string(index=False))
    print(f"\n[Info] Summary saved to {summary_csv_path}")
    print(f"[Info] Joint quality saved to {joint_csv_path}")
    print(f"[Info] JSON saved to {summary_json_path}")


if __name__ == "__main__":
    main()
