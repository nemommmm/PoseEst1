"""Tests for distance-stratified detector analysis helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_error_vs_distance import (  # noqa: E402
    _distribution,
    _torso_distance,
    add_distance_bins,
    extract_joint_median_rows,
)


class DistanceErrorAnalysisTest(unittest.TestCase):
    """Cover depth coordinates, robust statistics, and bin boundaries."""

    def test_torso_distance_uses_centimetres_and_requires_two_joints(self) -> None:
        keypoints = np.full((2, 17, 3), np.nan, dtype=np.float64)
        for joint in (5, 6, 11, 12):
            keypoints[0, joint] = [30.0, 40.0, 200.0]
        keypoints[1, 5] = [0.0, 0.0, 300.0]

        depth_m, range_m = _torso_distance(keypoints)

        self.assertAlmostEqual(float(depth_m[0]), 2.0)
        self.assertAlmostEqual(float(range_m[0]), np.sqrt(4.25))
        self.assertTrue(np.isnan(depth_m[1]))
        self.assertTrue(np.isnan(range_m[1]))

    def test_distance_bins_are_left_closed_half_metre_intervals(self) -> None:
        data = pd.DataFrame({"optical_depth_m": [2.0, 2.49, 2.5, 3.01]})
        binned = add_distance_bins(data, 0.5)
        labels = binned["distance_bin"].astype(str).tolist()
        self.assertEqual(labels, ["2.0–2.5", "2.0–2.5", "2.5–3.0", "3.0–3.5"])

    def test_distribution_reports_median_and_tail_separately(self) -> None:
        stats = _distribution(pd.Series([1.0, 2.0, 100.0, np.nan]))
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(float(stats["median_deg"]), 2.0)
        self.assertAlmostEqual(float(stats["mean_deg"]), 103.0 / 3.0)
        self.assertGreater(float(stats["p95_deg"]), 90.0)

    @patch("analyze_error_vs_distance.extract_rows")
    def test_bilateral_target_requires_both_sides_and_uses_median(
        self, extract_rows_mock
    ) -> None:
        identity = {
            "session_id": ["s", "s"],
            "session": ["Session", "Session"],
            "frame": [0, 1],
            "time_s": [0.0, 0.1],
            "model": ["YOLOv8m", "YOLOv8m"],
            "fixed_offset_s": [0.0, 0.0],
            "optical_depth_m": [2.0, 2.0],
            "radial_range_m": [2.1, 2.1],
        }
        left = pd.DataFrame(
            {
                **identity,
                "vision_angle_deg": [20.0, 30.0],
                "reference_angle_deg": [10.0, 10.0],
                "signed_difference_deg": [10.0, 20.0],
                "abs_difference_deg": [10.0, 20.0],
                "model_valid": [True, True],
                "common_valid": [True, True],
            }
        )
        right = pd.DataFrame(
            {
                **identity,
                "vision_angle_deg": [16.0, np.nan],
                "reference_angle_deg": [10.0, np.nan],
                "signed_difference_deg": [6.0, np.nan],
                "abs_difference_deg": [6.0, np.nan],
                "model_valid": [True, False],
                "common_valid": [True, False],
            }
        )
        extract_rows_mock.side_effect = [(left, [{"source": "left"}]), (right, [])]

        result, sources = extract_joint_median_rows(
            [], ("LeftHip", "RightHip"), "BilateralHipMedian"
        )

        self.assertAlmostEqual(float(result.loc[0, "abs_difference_deg"]), 8.0)
        self.assertTrue(bool(result.loc[0, "common_valid"]))
        self.assertFalse(bool(result.loc[1, "common_valid"]))
        self.assertTrue(np.isnan(result.loc[1, "abs_difference_deg"]))
        self.assertEqual(sources, [{"source": "left"}])


if __name__ == "__main__":
    unittest.main()
