"""Build a three-way bone-length comparison using Xsens as reference.

This script compares:
1. Xsens segment-origin geometry (reference proxy)
2. Stereo SKT keypoint geometry
3. EasyErgo normalized skeleton geometry

It is designed for P1a in the revised AFH1 follow-up plan. The goal is not to
prove a winner in advance, but to determine which system shows the stronger
evidence of systematic segment-length bias under one shared definition.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))

from utils_mvnx import MvnxParser  # noqa: E402


STEREO_NPZ = os.environ.get(
    "POSE_INPUT_FILENAME",
    os.path.join(
        PROJECT_ROOT,
        "01_stereo_triangulation",
        "results",
        "historical_best_20260324",
        "recovered_baseline",
        "optimized_pose.npz",
    ),
)
EASYERGO_NPZ = os.environ.get(
    "POSE_EASYERGO_NPZ",
    os.path.join(PROJECT_ROOT, "04_hybrid_afh1", "results", "easyergo_normalized.npz"),
)
GT_MVNX = os.environ.get(
    "POSE_GT_MVNX",
    os.path.join(PROJECT_ROOT, "..", "Xsens_ground_truth", "Aitor-001.mvnx"),
)
RESULTS_DIR = os.environ.get(
    "POSE_RESULTS_DIR",
    os.path.join(PROJECT_ROOT, "05_bone_correction", "results"),
)


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def median_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Median Euclidean distance across frames."""
    dist = np.linalg.norm(a - b, axis=1)
    dist = dist[np.isfinite(dist)]
    return float(np.median(dist)) if dist.size else np.nan


def median_concat(values: list[np.ndarray]) -> float:
    """Median of concatenated 1D arrays after dropping NaN/inf."""
    merged = np.concatenate(values).astype(np.float64)
    merged = merged[np.isfinite(merged)]
    return float(np.median(merged)) if merged.size else np.nan


def load_stereo_lengths(npz_path: str) -> Dict[str, float]:
    """Compute COCO-17 segment medians from the stereo pose."""
    data = np.load(npz_path)
    kpts = data["keypoints"].astype(np.float64)
    hip_mid = 0.5 * (kpts[:, LEFT_HIP] + kpts[:, RIGHT_HIP])
    shoulder_mid = 0.5 * (kpts[:, LEFT_SHOULDER] + kpts[:, RIGHT_SHOULDER])

    return {
        "shoulder_width": median_distance(kpts[:, LEFT_SHOULDER], kpts[:, RIGHT_SHOULDER]),
        "hip_width": median_distance(kpts[:, LEFT_HIP], kpts[:, RIGHT_HIP]),
        "torso": median_distance(hip_mid, shoulder_mid),
        "upper_arm": median_concat(
            [
                np.linalg.norm(kpts[:, LEFT_SHOULDER] - kpts[:, LEFT_ELBOW], axis=1),
                np.linalg.norm(kpts[:, RIGHT_SHOULDER] - kpts[:, RIGHT_ELBOW], axis=1),
            ]
        ),
        "forearm": median_concat(
            [
                np.linalg.norm(kpts[:, LEFT_ELBOW] - kpts[:, LEFT_WRIST], axis=1),
                np.linalg.norm(kpts[:, RIGHT_ELBOW] - kpts[:, RIGHT_WRIST], axis=1),
            ]
        ),
        "thigh": median_concat(
            [
                np.linalg.norm(kpts[:, LEFT_HIP] - kpts[:, LEFT_KNEE], axis=1),
                np.linalg.norm(kpts[:, RIGHT_HIP] - kpts[:, RIGHT_KNEE], axis=1),
            ]
        ),
        "shank": median_concat(
            [
                np.linalg.norm(kpts[:, LEFT_KNEE] - kpts[:, LEFT_ANKLE], axis=1),
                np.linalg.norm(kpts[:, RIGHT_KNEE] - kpts[:, RIGHT_ANKLE], axis=1),
            ]
        ),
    }


def load_easyergo_lengths(npz_path: str) -> Dict[str, float]:
    """Compute COCO-17 segment medians from the EasyErgo-normalized skeleton."""
    data = np.load(npz_path)
    kpts = data["keypoints_3d"].astype(np.float64)
    hip_mid = 0.5 * (kpts[:, LEFT_HIP] + kpts[:, RIGHT_HIP])
    shoulder_mid = 0.5 * (kpts[:, LEFT_SHOULDER] + kpts[:, RIGHT_SHOULDER])

    return {
        "shoulder_width": median_distance(kpts[:, LEFT_SHOULDER], kpts[:, RIGHT_SHOULDER]),
        "hip_width": median_distance(kpts[:, LEFT_HIP], kpts[:, RIGHT_HIP]),
        "torso": median_distance(hip_mid, shoulder_mid),
        "upper_arm": median_concat(
            [
                np.linalg.norm(kpts[:, LEFT_SHOULDER] - kpts[:, LEFT_ELBOW], axis=1),
                np.linalg.norm(kpts[:, RIGHT_SHOULDER] - kpts[:, RIGHT_ELBOW], axis=1),
            ]
        ),
        "forearm": median_concat(
            [
                np.linalg.norm(kpts[:, LEFT_ELBOW] - kpts[:, LEFT_WRIST], axis=1),
                np.linalg.norm(kpts[:, RIGHT_ELBOW] - kpts[:, RIGHT_WRIST], axis=1),
            ]
        ),
        "thigh": median_concat(
            [
                np.linalg.norm(kpts[:, LEFT_HIP] - kpts[:, LEFT_KNEE], axis=1),
                np.linalg.norm(kpts[:, RIGHT_HIP] - kpts[:, RIGHT_KNEE], axis=1),
            ]
        ),
        "shank": median_concat(
            [
                np.linalg.norm(kpts[:, LEFT_KNEE] - kpts[:, LEFT_ANKLE], axis=1),
                np.linalg.norm(kpts[:, RIGHT_KNEE] - kpts[:, RIGHT_ANKLE], axis=1),
            ]
        ),
    }


def load_xsens_lengths(mvnx_path: str) -> Dict[str, float]:
    """Compute Xsens segment-origin medians using a COCO-compatible proxy."""
    mvnx = MvnxParser(mvnx_path)
    mvnx.parse()

    seg = {
        name: mvnx.get_segment_data(name)
        for name in [
            "Pelvis",
            "LeftUpperArm",
            "RightUpperArm",
            "LeftForeArm",
            "RightForeArm",
            "LeftHand",
            "RightHand",
            "LeftUpperLeg",
            "RightUpperLeg",
            "LeftLowerLeg",
            "RightLowerLeg",
            "LeftFoot",
            "RightFoot",
        ]
    }
    shoulder_mid = 0.5 * (seg["LeftUpperArm"] + seg["RightUpperArm"])
    hip_mid = 0.5 * (seg["LeftUpperLeg"] + seg["RightUpperLeg"])

    return {
        "shoulder_width": median_distance(seg["LeftUpperArm"], seg["RightUpperArm"]),
        "hip_width": median_distance(seg["LeftUpperLeg"], seg["RightUpperLeg"]),
        "torso": median_distance(shoulder_mid, hip_mid),
        "upper_arm": median_concat(
            [
                np.linalg.norm(seg["LeftUpperArm"] - seg["LeftForeArm"], axis=1),
                np.linalg.norm(seg["RightUpperArm"] - seg["RightForeArm"], axis=1),
            ]
        ),
        "forearm": median_concat(
            [
                np.linalg.norm(seg["LeftForeArm"] - seg["LeftHand"], axis=1),
                np.linalg.norm(seg["RightForeArm"] - seg["RightHand"], axis=1),
            ]
        ),
        "thigh": median_concat(
            [
                np.linalg.norm(seg["LeftUpperLeg"] - seg["LeftLowerLeg"], axis=1),
                np.linalg.norm(seg["RightUpperLeg"] - seg["RightLowerLeg"], axis=1),
            ]
        ),
        "shank": median_concat(
            [
                np.linalg.norm(seg["LeftLowerLeg"] - seg["LeftFoot"], axis=1),
                np.linalg.norm(seg["RightLowerLeg"] - seg["RightFoot"], axis=1),
            ]
        ),
    }


def classify_bias(stereo: Dict[str, float], easy: Dict[str, float], xsens: Dict[str, float]) -> str:
    """Produce a conservative high-level verdict from per-segment ratios."""
    segments = list(xsens.keys())
    stereo_ratios = np.array([stereo[s] / xsens[s] for s in segments], dtype=np.float64)
    easy_ratios = np.array([easy[s] / xsens[s] for s in segments], dtype=np.float64)

    stereo_close = np.mean((stereo_ratios >= 0.85) & (stereo_ratios <= 1.15))
    easy_close = np.mean((easy_ratios >= 0.85) & (easy_ratios <= 1.15))
    stereo_low = np.mean(stereo_ratios < 0.85)
    easy_low = np.mean(easy_ratios < 0.85)

    if stereo_low >= 0.6 and easy_close >= 0.6:
        return "stereo_compressed_vs_xsens_easyergo_closer"
    if stereo_close >= 0.6 and easy_low >= 0.6:
        return "easyergo_mismatch_vs_xsens_stereo_closer"
    if stereo_close >= 0.6 and easy_close >= 0.6:
        return "both_reasonably_close_to_xsens"
    return "mixed_or_inconclusive"


def main() -> None:
    """Run the three-way comparison and save compact reports."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stereo = load_stereo_lengths(STEREO_NPZ)
    easy = load_easyergo_lengths(EASYERGO_NPZ)
    xsens = load_xsens_lengths(GT_MVNX)
    verdict = classify_bias(stereo, easy, xsens)

    rows = []
    for segment in xsens:
        x = xsens[segment]
        s = stereo[segment]
        e = easy[segment]
        rows.append(
            {
                "segment": segment,
                "xsens_cm": x,
                "stereo_cm": s,
                "easyergo_cm": e,
                "stereo_over_xsens": s / x if np.isfinite(x) and x != 0 else np.nan,
                "easyergo_over_xsens": e / x if np.isfinite(x) and x != 0 else np.nan,
                "stereo_minus_xsens_cm": s - x,
                "easyergo_minus_xsens_cm": e - x,
            }
        )

    csv_path = os.path.join(RESULTS_DIR, "bone_length_three_way_comparison.csv")
    md_path = os.path.join(RESULTS_DIR, "bone_length_three_way_comparison.md")
    json_path = os.path.join(RESULTS_DIR, "bone_length_three_way_comparison.json")

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Bone Length Three-Way Comparison\n\n")
        handle.write("## Definition\n\n")
        handle.write("- `Xsens` uses segment-origin proxies closest to the COCO-17 skeleton semantics:\n")
        handle.write("  `shoulder_width = dist(LeftUpperArm, RightUpperArm)`\n")
        handle.write("  `hip_width = dist(LeftUpperLeg, RightUpperLeg)`\n")
        handle.write("  `torso = dist(mid(UpperArms), mid(UpperLegs))`\n")
        handle.write("  limb lengths use adjacent segment origins.\n\n")
        handle.write("## Comparison Table\n\n")
        handle.write("| Segment | Xsens (cm) | Stereo (cm) | EasyErgo (cm) | Stereo/Xsens | EasyErgo/Xsens |\n")
        handle.write("|---------|------------|-------------|---------------|--------------|----------------|\n")
        for row in rows:
            handle.write(
                f"| {row['segment']} | {row['xsens_cm']:.2f} | {row['stereo_cm']:.2f} | "
                f"{row['easyergo_cm']:.2f} | {row['stereo_over_xsens']:.3f} | {row['easyergo_over_xsens']:.3f} |\n"
            )
        handle.write("\n## Verdict\n\n")
        handle.write(f"- Classification: `{verdict}`\n")

    payload = {
        "verdict": verdict,
        "definition": {
            "shoulder_width": "dist(LeftUpperArm, RightUpperArm)",
            "hip_width": "dist(LeftUpperLeg, RightUpperLeg)",
            "torso": "dist(mid(UpperArms), mid(UpperLegs))",
            "upper_arm": "dist(UpperArm, ForeArm)",
            "forearm": "dist(ForeArm, Hand)",
            "thigh": "dist(UpperLeg, LowerLeg)",
            "shank": "dist(LowerLeg, Foot)",
        },
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[info] saved: {csv_path}")
    print(f"[info] saved: {md_path}")
    print(f"[info] saved: {json_path}")


if __name__ == "__main__":
    main()
