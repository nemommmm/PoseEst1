import os

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import StereoDataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")
RESULT_PATH = os.path.join(RESULTS_DIR, "yolo_3d_optimized.npz")
TIMELINE_PATH = os.path.join(RESULTS_DIR, "eval_error_timeline.csv")
OUTPUT_FIG_PATH = os.path.join(SRC_DIR, "stereo_keypoint_mismatch_examples.png")
OUTPUT_SUMMARY_PATH = os.path.join(RESULTS_DIR, "stereo_keypoint_mismatch_examples.csv")

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
    timeline = pd.read_csv(TIMELINE_PATH)
    return pose, timeline


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
    df["pair_conf"] = np.nanmean(
        np.stack([conf_l[:, HIGHLIGHT_JOINT], conf_r[:, HIGHLIGHT_JOINT]], axis=0),
        axis=0,
    )
    df["joint_reproj"] = reproj[:, HIGHLIGHT_JOINT]
    df["joint_name"] = JOINT_NAMES[HIGHLIGHT_JOINT]

    good = df[
        (df["scenario"] == "Baseline")
        & (df["activity"] == "Walking (Normal)")
        & (df["pair_conf"] > 0.90)
        & np.isfinite(df["joint_reproj"])
    ].sort_values(["joint_reproj", "mean_joint_error_cm", "root_error_cm"])

    baseline_bad = df[
        (df["scenario"] == "Baseline")
        & (df["activity"] == "Walking (Normal)")
        & (df["pair_conf"] > 0.95)
        & (df["joint_reproj"] > 100.0)
        & (df["mean_joint_error_cm"] < 35.0)
        & (df["root_error_cm"] < 15.0)
    ].sort_values(["joint_reproj", "mean_joint_error_cm"], ascending=[False, True])

    bad = df[
        (df["scenario"] == "Environmental Interference")
        & (df["pair_conf"] > 0.85)
        & np.isfinite(df["joint_reproj"])
    ].sort_values(["joint_reproj", "mean_joint_error_cm"], ascending=[False, False])

    if good.empty or baseline_bad.empty or bad.empty:
        raise RuntimeError("Failed to locate suitable comparison frames.")

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
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w, h), rot, trans, alpha=0)

    pts_l = pose["keypoints_left_2d"].astype(np.float64).reshape(-1, 17, 1, 2)
    pts_r = pose["keypoints_right_2d"].astype(np.float64).reshape(-1, 17, 1, 2)
    rect_l = np.full((len(pts_l), 17, 2), np.nan, dtype=np.float64)
    rect_r = np.full((len(pts_r), 17, 2), np.nan, dtype=np.float64)
    for i in range(len(pts_l)):
        valid_l = np.isfinite(pts_l[i, :, 0, :]).all(axis=1)
        valid_r = np.isfinite(pts_r[i, :, 0, :]).all(axis=1)
        if np.any(valid_l):
            rect_l[i, valid_l] = cv2.undistortPoints(pts_l[i, valid_l], mtx_l, dist_l, R=R1, P=P1)[:, 0, :]
        if np.any(valid_r):
            rect_r[i, valid_r] = cv2.undistortPoints(pts_r[i, valid_r], mtx_r, dist_r, R=R2, P=P2)[:, 0, :]
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
        ax.plot(chain_points[:, 0], chain_points[:, 1], color="#FF7F0E", linewidth=3.0, alpha=0.9)

    for idx, point in enumerate(keypoints):
        if not np.isfinite(point).all():
            continue
        size = 28 if idx == highlight_joint else 18
        color = "#D62728" if idx == highlight_joint else "#FFFFFF"
        edge = "#000000"
        ax.scatter(point[0], point[1], s=size, c=color, edgecolors=edge, linewidths=0.8, zorder=3)
        if idx == highlight_joint:
            label = JOINT_NAMES[idx]
            if confs is not None and np.isfinite(confs[idx]):
                label += f" ({confs[idx]:.2f})"
            ax.text(point[0] + 10, point[1] - 10, label, color="#D62728", fontsize=10, weight="bold")


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
    origin = np.mean(finite_shoulders, axis=0) if finite_shoulders else np.zeros(2, dtype=np.float64)
    left_chain_rel = left_chain - origin
    right_chain_rel = right_chain - origin

    if np.isfinite(left_chain_rel).all():
        ax.plot(left_chain_rel[:, 0], left_chain_rel[:, 1], "-o", color="#1F77B4", linewidth=2.5, label="Left view")
    if np.isfinite(right_chain_rel).all():
        ax.plot(right_chain_rel[:, 0], right_chain_rel[:, 1], "-o", color="#FF7F0E", linewidth=2.5, label="Right view")

    for idx, label in zip(RIGHT_ARM_CHAIN, left_labels):
        p_l = rect_l[idx] - origin
        p_r = rect_r[idx] - origin
        if np.isfinite(p_l).all():
            ax.text(p_l[0] + 8, p_l[1], f"L {label}", color="#1F77B4", fontsize=9)
        if np.isfinite(p_r).all():
            ax.text(p_r[0] + 8, p_r[1], f"R {label}", color="#FF7F0E", fontsize=9)
        if np.isfinite(p_l).all() and np.isfinite(p_r).all():
            ax.plot([p_l[0], p_r[0]], [p_l[1], p_r[1]], linestyle="--", color="#7F7F7F", linewidth=1.0)

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
    fig, axes = plt.subplots(len(examples), 3, figsize=(18, 5.2 * len(examples)), constrained_layout=True)

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
