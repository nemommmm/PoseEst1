"""
08_ergonomic_scoring.py

RULA (Rapid Upper Limb Assessment) scoring module.
Computes per-frame RULA posture scores from 3D keypoints and compares
them against Xsens ground-truth RULA scores derived from Xsens joint angles.

This module computes anatomically-correct angles directly from 3D geometry
rather than using the IK interior angles (which map differently to RULA).

Input:
  - results/yolo_3d_optimized.npz   (preferred canonical pose output)
  - Xsens_ground_truth/Aitor-001.mvnx
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import json
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_mvnx import MvnxParser
from pose_postprocess import (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW,
    RIGHT_ELBOW, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE,
    RIGHT_ANKLE)
from pose_angle_utils import (
    build_gt_angle_interpolators,
    compute_aligned_trunk_flexion,
    compute_semantic_joint_angles,
    reduce_max_finite,
)

# ================= Configuration =================
BEST_OFFSET = 17.20
GT_LIMB_LENGTHS = [38.6, 39.8, 40.3, 39.5]
TOP_K = 150

ACTIVITY_SEGMENTS = {
    "Walking (Normal)":          [17, 32],
    "Walking (Late)":            [220, 240],
    "Sitting (Lower Occluded)":  [32, 62],
    "Walking (Upper Occluded)":  [87, 97],
    "Walking (Lower Occluded 1)":[130, 140],
    "Walking (Lower Occluded 2)":[164, 170],
    "Chair Interaction (Complex)":[140, 160],
    "Lifting Box (Near Chair)":  [214, 218],
    "Squatting":                 [66, 69],
    "Squatting (Check)":         [156, 160],
}

SCENARIO_MAPPING = {
    "Walking (Normal)":            "Baseline",
    "Walking (Late)":              "Baseline",
    "Sitting (Lower Occluded)":    "Occlusion",
    "Walking (Upper Occluded)":    "Occlusion",
    "Walking (Lower Occluded 1)":  "Occlusion",
    "Walking (Lower Occluded 2)":  "Occlusion",
    "Chair Interaction (Complex)": "Environmental Interference",
    "Lifting Box (Near Chair)":    "Environmental Interference",
    "Squatting":                   "Dynamic Action",
    "Squatting (Check)":           "Dynamic Action",
}
# ================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MVNX_PATH = os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx")
RESULTS_DIR = os.environ.get("POSE_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
if not os.path.isabs(RESULTS_DIR):
    RESULTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_DIR)
OUTPUT_TAG = os.environ.get("POSE_OUTPUT_TAG", "").strip()
ALIGNMENT_SUMMARY_PATH = os.environ.get(
    "POSE_ALIGNMENT_SUMMARY_NAME",
    os.path.join(RESULTS_DIR, "alignment_summary.json"),
)
if not os.path.isabs(ALIGNMENT_SUMMARY_PATH):
    ALIGNMENT_SUMMARY_PATH = os.path.join(RESULTS_DIR, ALIGNMENT_SUMMARY_PATH)
SAVE_PLOTS = os.environ.get("POSE_SAVE_PLOTS", "1").lower() not in {"0", "false", "no"}
os.makedirs(RESULTS_DIR, exist_ok=True)


def tagged_name(filename):
    if not OUTPUT_TAG:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{OUTPUT_TAG}{ext}"


def resolve_results_path(filename):
    return os.path.join(RESULTS_DIR, tagged_name(filename))


def resolve_src_plot_path(filename):
    return os.path.join(SRC_DIR, tagged_name(filename))


def resolve_best_offset():
    if os.path.exists(ALIGNMENT_SUMMARY_PATH):
        with open(ALIGNMENT_SUMMARY_PATH, "r") as f:
            data = json.load(f)
        return float(data.get("best_offset_seconds", BEST_OFFSET))
    return BEST_OFFSET


def resolve_pose_input_path():
    preferred = os.environ.get("POSE_INPUT_FILENAME", "yolo_3d_optimized.npz")
    candidates = [preferred, "yolo_3d_raw.npz"]
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        path = name if os.path.isabs(name) else os.path.join(RESULTS_DIR, name)
        if os.path.exists(path):
            return path
    return None


def kabsch_transform(P, Q):
    mask = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P, Q = P[mask], Q[mask]
    if len(P) < 10:
        return np.eye(3), np.zeros(3)
    cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    rot = Vt.T @ U.T
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = Vt.T @ U.T
    return rot, cQ - rot @ cP


def calculate_limb_error(kpts, gt_lengths):
    lengths = np.vstack([
        np.linalg.norm(kpts[:, 11] - kpts[:, 13], axis=1),
        np.linalg.norm(kpts[:, 12] - kpts[:, 14], axis=1),
        np.linalg.norm(kpts[:, 13] - kpts[:, 15], axis=1),
        np.linalg.norm(kpts[:, 14] - kpts[:, 16], axis=1),
    ]).T
    return np.sum(np.abs(lengths - gt_lengths), axis=1)


# ======================== Angle Extraction from 3D Poses ========================

def _safe_angle_deg(vec_a, vec_b):
    """Angle between two vectors in degrees."""
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na < 1e-6 or nb < 1e-6:
        return np.nan
    cos = np.clip(np.dot(vec_a, vec_b) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def compute_rula_angles_from_pose(pose):
    """
    Compute RULA-relevant angles directly from COCO 3D keypoints.
    Returns dict with: shoulder_elev, elbow_flex, trunk_flex, knee_flex.
    All angles in degrees.
    """
    joint_angles = compute_semantic_joint_angles(pose)
    result = {
        "trunk_flex": compute_aligned_trunk_flexion(pose),
        "left_shoulder_elev": joint_angles["LeftShoulder"],
        "right_shoulder_elev": joint_angles["RightShoulder"],
        "left_elbow_flex": joint_angles["LeftElbow"],
        "right_elbow_flex": joint_angles["RightElbow"],
        "left_knee_flex": joint_angles["LeftKnee"],
        "right_knee_flex": joint_angles["RightKnee"],
    }
    return result


# ======================== RULA Scoring Functions ========================

def score_upper_arm(shoulder_elevation_deg):
    angle = abs(shoulder_elevation_deg)
    if angle <= 20:
        return 1
    elif angle <= 45:
        return 2
    elif angle <= 90:
        return 3
    else:
        return 4


def score_lower_arm(elbow_flexion_deg):
    angle = abs(elbow_flexion_deg)
    if 60 <= angle <= 100:
        return 1
    else:
        return 2


def score_trunk(trunk_flexion_deg):
    angle = abs(trunk_flexion_deg)
    if angle <= 10:
        return 1
    elif angle <= 20:
        return 2
    elif angle <= 60:
        return 3
    else:
        return 4


def score_legs(knee_flexion_max_deg):
    if abs(knee_flexion_max_deg) > 30:
        return 2
    return 1


RULA_TABLE_A = {
    (1, 1): 1, (1, 2): 2,
    (2, 1): 2, (2, 2): 3,
    (3, 1): 3, (3, 2): 4,
    (4, 1): 4, (4, 2): 5,
}

RULA_TABLE_B = {
    (1, 1): 1, (1, 2): 2,
    (2, 1): 2, (2, 2): 3,
    (3, 1): 3, (3, 2): 4,
    (4, 1): 4, (4, 2): 5,
}

RULA_TABLE_C = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 3, (1, 5): 4,
    (2, 1): 2, (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 4,
    (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 4, (3, 5): 5,
    (4, 1): 3, (4, 2): 4, (4, 3): 4, (4, 4): 5, (4, 5): 6,
    (5, 1): 4, (5, 2): 4, (5, 3): 5, (5, 4): 6, (5, 5): 7,
}


def compute_rula_score(shoulder_deg, elbow_deg, trunk_deg, knee_deg):
    if any(not np.isfinite(v) for v in [shoulder_deg, elbow_deg, trunk_deg, knee_deg]):
        return None
    upper_arm = score_upper_arm(shoulder_deg)
    lower_arm = score_lower_arm(elbow_deg)
    trunk = score_trunk(trunk_deg)
    legs = score_legs(knee_deg)
    score_a = RULA_TABLE_A.get((upper_arm, lower_arm), max(upper_arm, lower_arm))
    score_b = RULA_TABLE_B.get((trunk, legs), max(trunk, legs))
    grand = RULA_TABLE_C.get((score_a, score_b), max(score_a, score_b))
    return {
        "grand_score": grand, "score_a": score_a, "score_b": score_b,
        "upper_arm": upper_arm, "lower_arm": lower_arm,
        "trunk": trunk, "legs": legs,
    }


def get_activity_and_scenario(t):
    for label, (start, end) in ACTIVITY_SEGMENTS.items():
        if start <= t <= end:
            return label, SCENARIO_MAPPING.get(label, "Other")
    return "Unclassified", "Unclassified"


# ======================== Main ========================

def main():
    print("=" * 60)
    print("[RULA] Ergonomic Risk Scoring — Estimated vs. Ground Truth")
    print("=" * 60)

    best_offset = resolve_best_offset()
    print(f"[Info] Temporal offset: {best_offset:.2f} s")

    # --- 1. Load canonical 3D keypoints ---
    pose_path = resolve_pose_input_path()
    if pose_path is None:
        print("[Error] No pose result file found. Run 02_batch_inference.py first.")
        return
    print(f"[Info] Pose input: {os.path.basename(pose_path)}")
    pose_data = np.load(pose_path)
    est_kpts = pose_data['keypoints']
    est_ts = pose_data['timestamps']

    # Filter valid frames
    est_center = (est_kpts[:, 11] + est_kpts[:, 12]) / 2.0
    valid_mask = (est_center[:, 2] > 10) & (est_center[:, 2] < 1000) & np.isfinite(est_center).all(axis=1)
    est_kpts = est_kpts[valid_mask]
    est_ts = est_ts[valid_mask]
    est_ts, uidx = np.unique(est_ts, return_index=True)
    est_kpts = est_kpts[uidx]
    est_ts -= est_ts[0]
    print(f"[Info] Valid frames: {len(est_ts)}")

    # --- 2. Load Xsens GT ---
    mvnx = MvnxParser(MVNX_PATH)
    mvnx.parse()
    xsens_ts = mvnx.timestamps.copy()
    xsens_ts, xidx = np.unique(xsens_ts, return_index=True)
    xsens_ts -= xsens_ts[0]

    # GT joint angle interpolators (for RULA sub-scores)
    gt_interp = build_gt_angle_interpolators(mvnx, xsens_ts, xidx)

    # GT trunk from ergo angles
    trunk_ergo = mvnx.get_ergo_angle_data("Pelvis_T8")
    gt_trunk_interp = None
    if trunk_ergo is not None:
        gt_trunk_interp = interp1d(
            xsens_ts, trunk_ergo[xidx, 0], kind='linear',
            bounds_error=False, fill_value=np.nan
        )

    # Position interpolator for Kabsch alignment
    xsens_pos_interp = {}
    for seg_name in ['Pelvis']:
        seg_data = mvnx.get_segment_data(seg_name)
        if seg_data is not None:
            xsens_pos_interp[seg_name] = interp1d(
                xsens_ts, seg_data[xidx], axis=0, kind='linear',
                bounds_error=False, fill_value=np.nan
            )

    # --- 3. Kabsch alignment (needed for trunk angle computation) ---
    y_pelvis = (est_kpts[:, 11] + est_kpts[:, 12]) / 2.0
    errors = calculate_limb_error(est_kpts, GT_LIMB_LENGTHS)
    valid_err_idx = np.where(np.isfinite(errors))[0]
    elite_indices = valid_err_idx[np.argsort(errors[valid_err_idx])[:TOP_K]]
    p_elite = y_pelvis[elite_indices]
    q_elite = xsens_pos_interp['Pelvis'](est_ts[elite_indices] - best_offset)
    R, t_vec = kabsch_transform(p_elite, q_elite)

    N_frames = len(est_ts)
    est_kpts_aligned = (R @ est_kpts.reshape(-1, 3).T).T.reshape(N_frames, -1, 3) + t_vec

    # --- 4. Compute RULA scores ---
    print("[Info] Computing per-frame RULA scores...")
    records = []

    for i, curr_t in enumerate(est_ts):
        target_t = curr_t - best_offset
        activity, scenario = get_activity_and_scenario(curr_t)

        # --- Estimated RULA (from aligned keypoints) ---
        angles = compute_rula_angles_from_pose(est_kpts_aligned[i])
        est_shoulder = reduce_max_finite([
            angles.get("left_shoulder_elev", np.nan),
            angles.get("right_shoulder_elev", np.nan),
        ])
        est_elbow = reduce_max_finite([
            abs(angles.get("left_elbow_flex", np.nan)),
            abs(angles.get("right_elbow_flex", np.nan)),
        ])
        est_trunk = angles.get("trunk_flex", np.nan)
        est_knee = reduce_max_finite([
            abs(angles.get("left_knee_flex", np.nan)),
            abs(angles.get("right_knee_flex", np.nan)),
        ])

        est_rula = compute_rula_score(est_shoulder, est_elbow, est_trunk, est_knee)

        # --- GT RULA (from Xsens joint angles) ---
        gt_l_shoulder = gt_interp.get("LeftShoulder", lambda t: np.nan)(target_t)
        gt_r_shoulder = gt_interp.get("RightShoulder", lambda t: np.nan)(target_t)
        gt_l_elbow = gt_interp.get("LeftElbow", lambda t: np.nan)(target_t)
        gt_r_elbow = gt_interp.get("RightElbow", lambda t: np.nan)(target_t)
        gt_l_knee = gt_interp.get("LeftKnee", lambda t: np.nan)(target_t)
        gt_r_knee = gt_interp.get("RightKnee", lambda t: np.nan)(target_t)
        gt_trunk_val = gt_trunk_interp(target_t) if gt_trunk_interp is not None else np.nan

        gt_shoulder = reduce_max_finite([abs(gt_l_shoulder), abs(gt_r_shoulder)])
        gt_elbow = reduce_max_finite([abs(gt_l_elbow), abs(gt_r_elbow)])
        gt_knee = reduce_max_finite([abs(gt_l_knee), abs(gt_r_knee)])

        gt_rula = compute_rula_score(gt_shoulder, gt_elbow, abs(gt_trunk_val), gt_knee)

        if est_rula is not None and gt_rula is not None:
            records.append({
                "Time": curr_t, "Activity": activity, "Scenario": scenario,
                "Est_Grand": est_rula["grand_score"],
                "GT_Grand": gt_rula["grand_score"],
                "Est_ScoreA": est_rula["score_a"],
                "GT_ScoreA": gt_rula["score_a"],
                "Est_ScoreB": est_rula["score_b"],
                "GT_ScoreB": gt_rula["score_b"],
                "Est_UpperArm": est_rula["upper_arm"],
                "GT_UpperArm": gt_rula["upper_arm"],
                "Est_LowerArm": est_rula["lower_arm"],
                "GT_LowerArm": gt_rula["lower_arm"],
                "Est_Trunk": est_rula["trunk"],
                "GT_Trunk": gt_rula["trunk"],
                "Est_Legs": est_rula["legs"],
                "GT_Legs": gt_rula["legs"],
                "Est_Shoulder": est_shoulder,
                "GT_Shoulder": gt_shoulder,
                "Est_Elbow": est_elbow,
                "GT_Elbow": gt_elbow,
                "Est_TrunkAngle": est_trunk,
                "GT_TrunkAngle": abs(gt_trunk_val),
            })

    df = pd.DataFrame(records)
    print(f"[Info] Valid RULA score pairs: {len(df)}")

    if df.empty:
        print("[Warning] No valid RULA score pairs could be computed.")
        return

    # --- 5. Results ---
    print("\n" + "=" * 60)
    print("🎯 RULA Score Comparison Results")
    print("=" * 60)

    exact_match = (df["Est_Grand"] == df["GT_Grand"]).mean()
    within_1 = (abs(df["Est_Grand"] - df["GT_Grand"]) <= 1).mean()
    print(f"\n[Result] Grand Score Exact Match:  {exact_match:.1%}")
    print(f"[Result] Grand Score Within ±1:    {within_1:.1%}")

    for sub in ["UpperArm", "LowerArm", "Trunk", "Legs"]:
        match = (df[f"Est_{sub}"] == df[f"GT_{sub}"]).mean()
        print(f"[Result] {sub} Sub-Score Match:  {match:.1%}")

    score_a_match = (df["Est_ScoreA"] == df["GT_ScoreA"]).mean()
    score_b_match = (df["Est_ScoreB"] == df["GT_ScoreB"]).mean()
    print(f"\n[Result] Score A Match (Upper Limb): {score_a_match:.1%}")
    print(f"[Result] Score B Match (Trunk/Legs):  {score_b_match:.1%}")

    # Angle diagnostic
    print("\n[Diagnostic] Angle statistics (Estimated vs GT):")
    for col, label in [("Shoulder", "Shoulder Elevation"),
                        ("Elbow", "Elbow Flexion"),
                        ("TrunkAngle", "Trunk Flexion")]:
        est_med = df[f"Est_{col}"].median()
        gt_med = df[f"GT_{col}"].median()
        print(f"  {label}: Est median={est_med:.1f}°  GT median={gt_med:.1f}°")

    # Scenario breakdown
    print("\n[Result] Grand Score Exact Match by Scenario:")
    print("-" * 50)
    scenario_agg = df.groupby("Scenario")[["Est_Grand", "GT_Grand"]].apply(
        lambda g: pd.Series({
            "ExactMatch": (g["Est_Grand"] == g["GT_Grand"]).mean(),
            "Within1": (abs(g["Est_Grand"] - g["GT_Grand"]) <= 1).mean(),
            "Samples": len(g),
        }), include_groups=False
    ).sort_values("ExactMatch", ascending=False)
    print(scenario_agg.to_string())

    # Score distribution
    print("\n[Result] Grand Score Distribution:")
    print("-" * 50)
    est_dist = df["Est_Grand"].value_counts().sort_index()
    gt_dist = df["GT_Grand"].value_counts().sort_index()
    all_scores = sorted(set(est_dist.index) | set(gt_dist.index))
    for s in all_scores:
        print(f"  Score {s}: Est={est_dist.get(s, 0):<6d} GT={gt_dist.get(s, 0):<6d}")

    # Risk level
    risk_map = {1: "Acceptable", 2: "Acceptable", 3: "Investigate", 4: "Investigate",
                5: "Change Soon", 6: "Change Soon", 7: "Change Now"}
    df["Est_Risk"] = df["Est_Grand"].map(risk_map)
    df["GT_Risk"] = df["GT_Grand"].map(risk_map)
    risk_match = (df["Est_Risk"] == df["GT_Risk"]).mean()
    print(f"\n[Result] Risk Level Agreement: {risk_match:.1%}")

    # --- 6. Save ---
    df.to_csv(resolve_results_path("eval_rula_comparison.csv"), index=False)
    scenario_agg.to_csv(resolve_results_path("eval_rula_by_scenario.csv"))

    summary = {
        "grand_score_exact_match": float(exact_match),
        "grand_score_within_1": float(within_1),
        "risk_level_agreement": float(risk_match),
        "score_a_match": float(score_a_match),
        "score_b_match": float(score_b_match),
        "total_valid_frames": int(len(df)),
    }
    with open(resolve_results_path("eval_rula_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- 7. Visualization ---
    if not SAVE_PLOTS:
        print("\n[Info] Plot generation disabled by POSE_SAVE_PLOTS=0")
        print("\n✅ RULA evaluation complete!")
        return

    print("\n[Info] Generating plots...")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_labels = sorted(set(df["Est_Grand"]) | set(df["GT_Grand"]))
    cm = pd.crosstab(df["GT_Grand"], df["Est_Grand"],
                     rownames=["Ground Truth"], colnames=["Estimated"],
                     dropna=False).reindex(index=all_labels, columns=all_labels, fill_value=0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("RULA Grand Score: Confusion Matrix", fontsize=13)

    risk_labels = ["Acceptable", "Investigate", "Change Soon", "Change Now"]
    cm_risk = pd.crosstab(df["GT_Risk"], df["Est_Risk"],
                          rownames=["Ground Truth"], colnames=["Estimated"],
                          dropna=False).reindex(index=risk_labels, columns=risk_labels, fill_value=0)
    sns.heatmap(cm_risk, annot=True, fmt="d", cmap="Oranges", ax=axes[1], cbar=False)
    axes[1].set_title("RULA Risk Level: Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(resolve_src_plot_path("eval_rula_confusion.png"), dpi=300, bbox_inches='tight')

    plt.figure(figsize=(14, 4))
    plt.plot(df["Time"], df["GT_Grand"], 'k-', alpha=0.5, linewidth=0.8, label="GT (Xsens)")
    plt.plot(df["Time"], df["Est_Grand"], 'r-', alpha=0.5, linewidth=0.8, label="Estimated")
    plt.title("RULA Grand Score Over Time", fontsize=14)
    plt.xlabel("Time (s)"); plt.ylabel("Grand Score (1–7)")
    plt.yticks(range(1, 8)); plt.legend()
    plt.tight_layout()
    plt.savefig(resolve_src_plot_path("eval_rula_timeline.png"), dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    scenarios = scenario_agg.index.tolist()
    x = range(len(scenarios))
    plt.bar(x, scenario_agg["ExactMatch"] * 100, color='steelblue', label="Exact Match")
    plt.bar(x, scenario_agg["Within1"] * 100, alpha=0.4, color='orange', label="Within ±1")
    plt.xticks(x, scenarios, rotation=15, ha='right')
    plt.ylabel("Agreement (%)"); plt.title("RULA Grand Score Agreement by Scenario")
    plt.legend(); plt.tight_layout()
    plt.savefig(resolve_src_plot_path("eval_rula_by_scenario.png"), dpi=300, bbox_inches='tight')

    print("[Info] RULA evaluation complete.")


if __name__ == "__main__":
    main()
