import os
import json

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pose_angle_utils import (
    DEFAULT_ANGLE_SMOOTH_RADIUS,
    SEMANTIC_ANGLE_VERSION,
    build_gt_angle_interpolators,
    compute_semantic_angle_sequence,
    median_filter_angle_sequence,
)
from utils_mvnx import MvnxParser
from utils import StereoDataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_DIR = os.environ.get("POSE_RESULTS_DIR", DEFAULT_RESULTS_DIR)
if not os.path.isabs(RESULTS_DIR):
    RESULTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_DIR)
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")
RESULT_PATH = os.environ.get(
    "POSE_INPUT_PATH",
    os.path.join(DEFAULT_RESULTS_DIR, os.environ.get("POSE_INPUT_FILENAME", "yolo_3d_optimized.npz")),
)
TIMELINE_PATH = os.environ.get(
    "POSE_TIMELINE_PATH",
    os.path.join(DEFAULT_RESULTS_DIR, "eval_error_timeline.csv"),
)
OUTPUT_FIG_PATH = os.environ.get(
    "POSE_OUTPUT_FIG_PATH",
    os.path.join(SRC_DIR, "stereo_keypoint_mismatch_examples.png"),
)
OUTPUT_SUMMARY_PATH = os.environ.get(
    "POSE_OUTPUT_SUMMARY_PATH",
    os.path.join(DEFAULT_RESULTS_DIR, "stereo_keypoint_mismatch_examples.csv"),
)
ALIGNMENT_SUMMARY_PATH = os.environ.get(
    "POSE_ALIGNMENT_SUMMARY_NAME",
    os.path.join(DEFAULT_RESULTS_DIR, "alignment_summary.json"),
)
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")

BEST_OFFSET = 17.20
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]
TOP_K = 150

ACTIVITY_SEGMENTS = {
    "Walking (Normal)": [17, 32],
    "Walking (Late)": [220, 240],
    "Sitting (Lower Occluded)": [32, 62],
    "Walking (Upper Occluded)": [87, 97],
    "Walking (Lower Occluded 1)": [130, 140],
    "Walking (Lower Occluded 2)": [164, 170],
    "Chair Interaction (Complex)": [140, 160],
    "Lifting Box (Near Chair)": [214, 218],
    "Squatting": [66, 69],
    "Squatting (Check)": [156, 160],
}

SCENARIO_MAPPING = {
    "Walking (Normal)": "Baseline",
    "Walking (Late)": "Baseline",
    "Sitting (Lower Occluded)": "Occlusion",
    "Walking (Upper Occluded)": "Occlusion",
    "Walking (Lower Occluded 1)": "Occlusion",
    "Walking (Lower Occluded 2)": "Occlusion",
    "Chair Interaction (Complex)": "Environmental Interference",
    "Lifting Box (Near Chair)": "Environmental Interference",
    "Squatting": "Dynamic Action",
    "Squatting (Check)": "Dynamic Action",
}

JOINT_POSITION_MAPPING = {
    0: "Head",
    5: "LeftShoulder",
    6: "RightShoulder",
    7: "LeftUpperArm",
    8: "RightUpperArm",
    9: "LeftForeArm",
    10: "RightForeArm",
    11: "Pelvis",
    12: "Pelvis",
    13: "LeftUpperLeg",
    14: "RightUpperLeg",
    15: "LeftLowerLeg",
    16: "RightLowerLeg",
}

JOINT_NAMES = [
    "Nose",
    "LeftEye",
    "RightEye",
    "LeftEar",
    "RightEar",
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftWrist",
    "RightWrist",
    "LeftHip",
    "RightHip",
    "LeftKnee",
    "RightKnee",
    "LeftAnkle",
    "RightAnkle",
]

SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

RIGHT_ARM_CHAIN = [6, 8, 10]
HIGHLIGHT_JOINT = 10  # RightWrist


def load_pose_arrays():
    pose = np.load(RESULT_PATH)
    if os.path.exists(TIMELINE_PATH):
        timeline = pd.read_csv(TIMELINE_PATH)
        print(f"[Info] Loaded timeline: {TIMELINE_PATH}")
    else:
        print("[Info] eval_error_timeline.csv not found. Building frame timeline from current evaluation logic...")
        timeline = build_frame_timeline(pose)
    return pose, timeline


def resolve_best_offset():
    if os.path.exists(ALIGNMENT_SUMMARY_PATH):
        with open(ALIGNMENT_SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("best_offset_seconds", BEST_OFFSET))
    return BEST_OFFSET


def kabsch_transform(P, Q):
    mask = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P, Q = P[mask], Q[mask]
    if len(P) < 10:
        return np.eye(3), np.zeros(3)
    cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    rot = Vt.T @ U.T
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
    t = cQ - rot @ cP
    return rot, t


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


def get_activity_and_scenario(t):
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t <= end:
            return label, SCENARIO_MAPPING.get(label, "Other")
    return "Unclassified", "Unclassified"


def build_frame_timeline(pose):
    best_offset = resolve_best_offset()
    est_kpts = pose["keypoints"]
    est_ts = pose["timestamps"].astype(np.float64)
    est_ts_norm = est_ts - est_ts[0]

    saved_angle_definition = ""
    if "angle_value_definition" in pose.files:
        saved_angle_definition = str(pose["angle_value_definition"])
    if {
        "angle_names",
        "angle_values",
    }.issubset(pose.files) and saved_angle_definition == SEMANTIC_ANGLE_VERSION:
        est_angle_names = [str(x) for x in pose["angle_names"]]
        est_angle_values = pose["angle_values"]
    else:
        est_angle_names, est_angle_values = compute_semantic_angle_sequence(est_kpts)
        est_angle_values = median_filter_angle_sequence(
            est_angle_values,
            radius=DEFAULT_ANGLE_SMOOTH_RADIUS,
        )

    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]
    gt_angle_interp = build_gt_angle_interpolators(mvnx, xsens_ts, xidx)

    xsens_pos_interp = {}
    for seg_name in set(JOINT_POSITION_MAPPING.values()):
        seg_data = mvnx.get_segment_data(seg_name)
        if seg_data is not None:
            from scipy.interpolate import interp1d

            xsens_pos_interp[seg_name] = interp1d(
                xsens_ts,
                seg_data[xidx],
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

    est_center = 0.5 * (est_kpts[:, 11] + est_kpts[:, 12])
    valid_mask = (
        (est_center[:, 2] > 10)
        & (est_center[:, 2] < 1000)
        & np.isfinite(est_center).all(axis=1)
    )
    est_kpts_v = est_kpts[valid_mask]
    est_ts_v = est_ts_norm[valid_mask]
    est_ts_v, uidx = np.unique(est_ts_v, return_index=True)
    est_kpts_v = est_kpts_v[uidx]

    y_pelvis = 0.5 * (est_kpts_v[:, 11] + est_kpts_v[:, 12])
    errors = calculate_limb_error(est_kpts_v, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    elite_indices = valid_err_idx[np.argsort(errors[valid_err_idx])[:TOP_K]]
    q_elite = xsens_pos_interp["Pelvis"](est_ts_v[elite_indices] - best_offset)
    R, t_vec = kabsch_transform(y_pelvis[elite_indices], q_elite)
    est_kpts_aligned = (R @ est_kpts.reshape(-1, 3).T).T.reshape(len(est_kpts), -1, 3) + t_vec

    rows = []
    for i, curr_t in enumerate(est_ts_norm):
        target_t = curr_t - best_offset
        activity, scenario = get_activity_and_scenario(curr_t)

        angle_errors = []
        for angle_idx, angle_name in enumerate(est_angle_names):
            est_val = est_angle_values[i, angle_idx]
            if not np.isfinite(est_val) or angle_name not in gt_angle_interp:
                continue
            gt_val = float(gt_angle_interp[angle_name](target_t))
            if np.isfinite(gt_val):
                angle_errors.append(abs(float(est_val) - gt_val))

        joint_errors = []
        for y_idx, seg_name in JOINT_POSITION_MAPPING.items():
            if seg_name not in xsens_pos_interp:
                continue
            gt_pos = xsens_pos_interp[seg_name](target_t)
            est_pos = est_kpts_aligned[i, y_idx]
            if np.isfinite(gt_pos).all() and np.isfinite(est_pos).all():
                joint_errors.append(float(np.linalg.norm(est_pos - gt_pos)))

        root_error = np.nan
        gt_pelvis = xsens_pos_interp["Pelvis"](target_t) if "Pelvis" in xsens_pos_interp else np.full(3, np.nan)
        est_pelvis = 0.5 * (est_kpts_aligned[i, 11] + est_kpts_aligned[i, 12])
        if np.isfinite(gt_pelvis).all() and np.isfinite(est_pelvis).all():
            root_error = float(np.linalg.norm(est_pelvis - gt_pelvis))

        rows.append(
            {
                "Time": float(curr_t),
                "Activity": activity,
                "Scenario": scenario,
                "Mean_Joint_Error_cm": float(np.mean(joint_errors)) if joint_errors else np.nan,
                "Root_Error_cm": root_error,
                "Mean_Angle_Error_deg": float(np.mean(angle_errors)) if angle_errors else np.nan,
            }
        )
    return pd.DataFrame(rows)


def map_scenarios_to_frames(timestamps_norm, timeline):
    frame_times = timeline["Time"].to_numpy()
    rows = []
    for i, t in enumerate(timestamps_norm):
        j = int(np.argmin(np.abs(frame_times - t)))
        row = timeline.iloc[j]
        rows.append(
            {
                "index": i,
                "time_s": float(t),
                "activity": row["Activity"],
                "scenario": row["Scenario"],
                "mean_joint_error_cm": row["Mean_Joint_Error_cm"],
                "root_error_cm": row["Root_Error_cm"],
                "mean_angle_error_deg": row["Mean_Angle_Error_deg"],
            }
        )
    return pd.DataFrame(rows)


def select_examples(pose, timeline):
    timestamps = pose["timestamps"].astype(np.float64)
    timestamps_norm = timestamps - timestamps[0]
    conf_l = pose["conf_left"]
    conf_r = pose["conf_right"]
    reproj = pose["reprojection_error"]
    frame_meta = map_scenarios_to_frames(timestamps_norm, timeline)

    df = frame_meta.copy()
    pair_conf = 0.5 * (conf_l[:, HIGHLIGHT_JOINT] + conf_r[:, HIGHLIGHT_JOINT])
    both_nan = ~np.isfinite(conf_l[:, HIGHLIGHT_JOINT]) & ~np.isfinite(conf_r[:, HIGHLIGHT_JOINT])
    pair_conf[both_nan] = np.nan
    df["pair_conf"] = pair_conf
    df["joint_reproj"] = reproj[:, HIGHLIGHT_JOINT]
    df["joint_name"] = JOINT_NAMES[HIGHLIGHT_JOINT]

    baseline_walk = df[
        (df["scenario"] == "Baseline")
        & (df["activity"] == "Walking (Normal)")
        & np.isfinite(df["joint_reproj"])
    ].copy()
    env_frames = df[
        (df["scenario"] == "Environmental Interference")
        & np.isfinite(df["joint_reproj"])
    ].copy()

    if baseline_walk.empty or env_frames.empty:
        raise RuntimeError("Failed to locate baseline/environmental frames for comparison.")

    baseline_reproj_hi = float(np.nanpercentile(baseline_walk["joint_reproj"], 90))
    env_reproj_hi = float(np.nanpercentile(env_frames["joint_reproj"], 90))
    baseline_joint_ok = float(np.nanpercentile(baseline_walk["mean_joint_error_cm"], 90))
    baseline_root_ok = float(np.nanpercentile(baseline_walk["root_error_cm"], 90))
    env_joint_bad = float(np.nanpercentile(env_frames["mean_joint_error_cm"], 75))

    good = baseline_walk[
        (baseline_walk["pair_conf"] > 0.90)
    ].sort_values(["joint_reproj", "mean_joint_error_cm", "root_error_cm"])
    if good.empty:
        good = baseline_walk.sort_values(["joint_reproj", "mean_joint_error_cm", "root_error_cm"])

    baseline_bad = baseline_walk[
        (baseline_walk["pair_conf"] > 0.95)
        & (baseline_walk["joint_reproj"] >= baseline_reproj_hi)
        & (baseline_walk["mean_joint_error_cm"] <= baseline_joint_ok)
        & (baseline_walk["root_error_cm"] <= baseline_root_ok)
    ].sort_values(["joint_reproj", "mean_joint_error_cm"], ascending=[False, True])
    if baseline_bad.empty:
        baseline_bad = baseline_walk[
            baseline_walk["joint_reproj"] >= baseline_reproj_hi
        ].sort_values(["joint_reproj", "mean_joint_error_cm"], ascending=[False, True])

    bad = env_frames[
        (env_frames["pair_conf"] > 0.90)
        & (env_frames["joint_reproj"] >= env_reproj_hi)
        & (env_frames["mean_joint_error_cm"] >= env_joint_bad)
    ].sort_values(["joint_reproj", "mean_joint_error_cm"], ascending=[False, False])
    if bad.empty:
        bad = env_frames.sort_values(["joint_reproj", "mean_joint_error_cm"], ascending=[False, False])

    return good.iloc[0].to_dict(), baseline_bad.iloc[0].to_dict(), bad.iloc[0].to_dict()


def fetch_frames(indices):
    targets = set(indices)
    frames = {}
    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"),
        os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"),
        os.path.join(DATA_DIR, "1_video_right.txt"),
    )
    curr_idx = 0
    while True:
        frame_l, frame_r, _, _ = loader.get_next_pair()
        if frame_l is None:
            break
        if curr_idx in targets:
            frames[curr_idx] = (frame_l.copy(), frame_r.copy())
            if len(frames) == len(targets):
                break
        curr_idx += 1
    loader.release()
    if len(frames) != len(targets):
        missing = sorted(targets - set(frames))
        raise RuntimeError(f"Failed to extract target frames: {missing}")
    return frames


def compute_rectified_points(pose):
    params = np.load(PARAM_PATH)
    mtx_l, dist_l = params["mtx_l"], params["dist_l"]
    mtx_r, dist_r = params["mtx_r"], params["dist_r"]
    rot, trans = params["R"], params["T"]

    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"),
        os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"),
        os.path.join(DATA_DIR, "1_video_right.txt"),
    )
    frame_l, _, _, _ = loader.get_next_pair()
    loader.release()
    if frame_l is None:
        raise RuntimeError("Failed to read the stereo videos.")
    h, w = frame_l.shape[:2]
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, (w, h), rot, trans, alpha=0
    )

    pts_l = pose["keypoints_left_2d"].astype(np.float64).reshape(-1, 17, 1, 2)
    pts_r = pose["keypoints_right_2d"].astype(np.float64).reshape(-1, 17, 1, 2)
    rect_l = np.full((len(pts_l), 17, 2), np.nan, dtype=np.float64)
    rect_r = np.full((len(pts_r), 17, 2), np.nan, dtype=np.float64)
    for i in range(len(pts_l)):
        valid_l = np.isfinite(pts_l[i, :, 0, :]).all(axis=1)
        valid_r = np.isfinite(pts_r[i, :, 0, :]).all(axis=1)
        if np.any(valid_l):
            rect_l[i, valid_l] = cv2.undistortPoints(
                pts_l[i, valid_l], mtx_l, dist_l, R=R1, P=P1
            )[:, 0, :]
        if np.any(valid_r):
            rect_r[i, valid_r] = cv2.undistortPoints(
                pts_r[i, valid_r], mtx_r, dist_r, R=R2, P=P2
            )[:, 0, :]
    return rect_l, rect_r


def draw_skeleton(ax, image, keypoints, title, highlight_joint, confs=None):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    for a, b in SKELETON_EDGES:
        if np.isfinite(keypoints[[a, b]]).all():
            ax.plot(
                [keypoints[a, 0], keypoints[b, 0]],
                [keypoints[a, 1], keypoints[b, 1]],
                color="#3BA3EC",
                linewidth=2.0,
                alpha=0.8,
            )

    chain_points = keypoints[RIGHT_ARM_CHAIN]
    if np.isfinite(chain_points).all():
        ax.plot(
            chain_points[:, 0],
            chain_points[:, 1],
            color="#FF7F0E",
            linewidth=3.0,
            alpha=0.9,
        )

    for idx, point in enumerate(keypoints):
        if not np.isfinite(point).all():
            continue
        size = 28 if idx == highlight_joint else 18
        color = "#D62728" if idx == highlight_joint else "#FFFFFF"
        edge = "#000000"
        ax.scatter(
            point[0],
            point[1],
            s=size,
            c=color,
            edgecolors=edge,
            linewidths=0.8,
            zorder=3,
        )
        if idx == highlight_joint:
            label = JOINT_NAMES[idx]
            if confs is not None and np.isfinite(confs[idx]):
                label += f" ({confs[idx]:.2f})"
            ax.text(
                point[0] + 10,
                point[1] - 10,
                label,
                color="#D62728",
                fontsize=10,
                weight="bold",
            )


def draw_rectified_overlay(ax, rect_l, rect_r, example_name, reproj_error, conf_l, conf_r):
    ax.set_title(f"{example_name}: rectified right-arm alignment", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel("Relative rectified x (px)")
    ax.set_ylabel("Relative rectified y (px)")

    left_chain = rect_l[RIGHT_ARM_CHAIN]
    right_chain = rect_r[RIGHT_ARM_CHAIN]
    left_labels = [JOINT_NAMES[i] for i in RIGHT_ARM_CHAIN]
    finite_shoulders = []
    if np.isfinite(rect_l[RIGHT_ARM_CHAIN[0]]).all():
        finite_shoulders.append(rect_l[RIGHT_ARM_CHAIN[0]])
    if np.isfinite(rect_r[RIGHT_ARM_CHAIN[0]]).all():
        finite_shoulders.append(rect_r[RIGHT_ARM_CHAIN[0]])
    origin = (
        np.mean(finite_shoulders, axis=0)
        if finite_shoulders
        else np.zeros(2, dtype=np.float64)
    )
    left_chain_rel = left_chain - origin
    right_chain_rel = right_chain - origin

    if np.isfinite(left_chain_rel).all():
        ax.plot(
            left_chain_rel[:, 0],
            left_chain_rel[:, 1],
            "-o",
            color="#1F77B4",
            linewidth=2.5,
            label="Left view",
        )
    if np.isfinite(right_chain_rel).all():
        ax.plot(
            right_chain_rel[:, 0],
            right_chain_rel[:, 1],
            "-o",
            color="#FF7F0E",
            linewidth=2.5,
            label="Right view",
        )

    for idx, label in zip(RIGHT_ARM_CHAIN, left_labels):
        p_l = rect_l[idx] - origin
        p_r = rect_r[idx] - origin
        if np.isfinite(p_l).all():
            ax.text(p_l[0] + 8, p_l[1], f"L {label}", color="#1F77B4", fontsize=9)
        if np.isfinite(p_r).all():
            ax.text(p_r[0] + 8, p_r[1], f"R {label}", color="#FF7F0E", fontsize=9)
        if np.isfinite(p_l).all() and np.isfinite(p_r).all():
            ax.plot(
                [p_l[0], p_r[0]],
                [p_l[1], p_r[1]],
                linestyle="--",
                color="#7F7F7F",
                linewidth=1.0,
            )

    wrist_dy = np.nan
    wrist_dx = np.nan
    if np.isfinite(rect_l[HIGHLIGHT_JOINT]).all() and np.isfinite(rect_r[HIGHLIGHT_JOINT]).all():
        wrist_dy = abs(rect_l[HIGHLIGHT_JOINT, 1] - rect_r[HIGHLIGHT_JOINT, 1])
        wrist_dx = abs(rect_l[HIGHLIGHT_JOINT, 0] - rect_r[HIGHLIGHT_JOINT, 0])

    text = (
        f"RightWrist conf L/R: {conf_l[HIGHLIGHT_JOINT]:.2f} / {conf_r[HIGHLIGHT_JOINT]:.2f}\n"
        f"RightWrist reproj: {reproj_error[HIGHLIGHT_JOINT]:.1f} px\n"
        f"Rectified |Δy|: {wrist_dy:.1f} px\n"
        f"Rectified disparity |Δx|: {wrist_dx:.1f} px"
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )
    ax.legend(loc="upper right")


def main():
    pose, timeline = load_pose_arrays()
    good, baseline_bad, bad = select_examples(pose, timeline)
    examples = [
        ("Good baseline example", good),
        ("Baseline mismatch example", baseline_bad),
        ("Bad environmental example", bad),
    ]
    frames = fetch_frames([int(good["index"]), int(baseline_bad["index"]), int(bad["index"])])
    rect_l_all, rect_r_all = compute_rectified_points(pose)

    summary_rows = []
    fig, axes = plt.subplots(
        len(examples), 3, figsize=(18, 5.2 * len(examples)), constrained_layout=True
    )

    for row_idx, (label, meta) in enumerate(examples):
        idx = int(meta["index"])
        frame_l, frame_r = frames[idx]
        kpts_l = pose["keypoints_left_2d"][idx]
        kpts_r = pose["keypoints_right_2d"][idx]
        conf_l = pose["conf_left"][idx]
        conf_r = pose["conf_right"][idx]
        reproj = pose["reprojection_error"][idx]
        rect_l = rect_l_all[idx]
        rect_r = rect_r_all[idx]

        left_title = (
            f"{label}\nLeft | t={meta['time_s']:.2f}s | {meta['activity']}\n"
            f"frame MPJPE={meta['mean_joint_error_cm']:.1f} cm"
        )
        right_title = (
            f"Right | root={meta['root_error_cm']:.1f} cm | "
            f"angle={meta['mean_angle_error_deg']:.1f} deg"
        )
        draw_skeleton(axes[row_idx, 0], frame_l, kpts_l, left_title, HIGHLIGHT_JOINT, conf_l)
        draw_skeleton(axes[row_idx, 1], frame_r, kpts_r, right_title, HIGHLIGHT_JOINT, conf_r)
        draw_rectified_overlay(axes[row_idx, 2], rect_l, rect_r, label, reproj, conf_l, conf_r)

        wrist_dy = np.nan
        if np.isfinite(rect_l[HIGHLIGHT_JOINT]).all() and np.isfinite(rect_r[HIGHLIGHT_JOINT]).all():
            wrist_dy = abs(rect_l[HIGHLIGHT_JOINT, 1] - rect_r[HIGHLIGHT_JOINT, 1])

        summary_rows.append(
            {
                "example": label,
                "frame_index": idx,
                "time_s": meta["time_s"],
                "activity": meta["activity"],
                "scenario": meta["scenario"],
                "frame_mean_joint_error_cm": meta["mean_joint_error_cm"],
                "frame_root_error_cm": meta["root_error_cm"],
                "frame_mean_angle_error_deg": meta["mean_angle_error_deg"],
                "joint": JOINT_NAMES[HIGHLIGHT_JOINT],
                "conf_left": float(conf_l[HIGHLIGHT_JOINT]),
                "conf_right": float(conf_r[HIGHLIGHT_JOINT]),
                "reprojection_error_px": float(reproj[HIGHLIGHT_JOINT]),
                "rectified_delta_y_px": float(wrist_dy),
            }
        )

    fig.suptitle(
        "Stereo keypoint mismatch examples: individually plausible 2D detections can diverge across views",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(OUTPUT_FIG_PATH, dpi=300, bbox_inches="tight")
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY_PATH, index=False)
    print(f"[Info] Figure saved to: {OUTPUT_FIG_PATH}")
    print(f"[Info] Summary saved to: {OUTPUT_SUMMARY_PATH}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
