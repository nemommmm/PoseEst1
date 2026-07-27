"""Tests for explainable YOLO model-switch analysis helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_dynamic_model_switch import (  # noqa: E402
    apply_threshold,
    build_window_table,
    leave_group_out_switch,
    skeleton_pixel_height,
)


class DynamicModelSwitchAnalysisTest(unittest.TestCase):
    """Cover shared routing features, windows, and held-out switching."""

    def test_skeleton_height_uses_robust_vertical_span(self) -> None:
        points = np.zeros((2, 17, 2), dtype=np.float64)
        points[0, :, 1] = np.arange(17, dtype=np.float64) * 10.0
        points[0, 16, 1] = 1000.0
        points[1, :, 1] = np.arange(17, dtype=np.float64)
        confidence = np.ones((2, 17), dtype=np.float64)
        confidence[1, :12] = 0.1

        height = skeleton_pixel_height(points, confidence)

        expected = np.percentile(points[0, :, 1], 95) - np.percentile(
            points[0, :, 1], 5
        )
        self.assertAlmostEqual(float(height[0]), float(expected))
        self.assertTrue(np.isnan(height[1]))

    def test_threshold_directions_select_expected_model(self) -> None:
        data = pd.DataFrame(
            {
                "feature": [1.0, 3.0],
                "error_yolov8m_deg": [10.0, 20.0],
                "error_yolo11l_deg": [1.0, 2.0],
            }
        )

        high = apply_threshold(data, "feature", 2.0, "high_uses_11l")
        low = apply_threshold(data, "feature", 2.0, "low_uses_11l")

        np.testing.assert_allclose(high, [10.0, 2.0])
        np.testing.assert_allclose(low, [1.0, 20.0])

    def test_window_table_uses_non_overlapping_medians(self) -> None:
        frame_data = pd.DataFrame(
            {
                "session": ["Session"] * 6,
                "session_id": ["session"] * 6,
                "validation_group": ["group"] * 6,
                "time_s": [0.0, 0.2, 0.4, 1.0, 1.2, 1.4],
                "error_yolov8m_deg": [1, 2, 3, 10, 11, 12],
                "error_yolo11l_deg": [3, 2, 1, 12, 11, 10],
                "common_angle_count": [8] * 6,
                "optical_depth_m": [2.0] * 6,
                "skeleton_height_px": [400.0] * 6,
                "reference_motion_deg_s": [1.0] * 3 + [25.0] * 3,
            }
        )

        windows = build_window_table(frame_data, 1.0, 3, (5.0, 20.0))

        self.assertEqual(len(windows), 2)
        np.testing.assert_allclose(
            windows["error_yolov8m_deg"].to_numpy(), [2.0, 11.0]
        )
        self.assertEqual(windows["motion_class"].astype(str).tolist(), [
            "static",
            "fast",
        ])

    def test_leave_group_out_returns_one_prediction_per_input_window(self) -> None:
        rows = []
        for group_index, group in enumerate(("a", "b", "c")):
            for index, feature in enumerate(np.linspace(100.0, 500.0, 10)):
                rows.append(
                    {
                        "session_id": group,
                        "validation_group": group,
                        "skeleton_height_px": feature,
                        "error_yolov8m_deg": 4.0,
                        "error_yolo11l_deg": (
                            6.0 if feature < 300.0 else 2.0
                        ),
                        "oracle_error_deg": (
                            4.0 if feature < 300.0 else 2.0
                        ),
                        "group_index": group_index,
                        "row_index": index,
                    }
                )
        windows = pd.DataFrame(rows)

        predictions, folds = leave_group_out_switch(
            windows, "skeleton_height_px", "high_uses_11l"
        )

        self.assertEqual(set(folds["held_out_group"]), {"a", "b", "c"})
        self.assertEqual(len(predictions), len(windows))
        self.assertTrue((folds["selected_minus_yolov8m_deg"] < 0).all())
        self.assertEqual(folds["learned_threshold"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
