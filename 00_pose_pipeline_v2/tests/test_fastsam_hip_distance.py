"""Tests for FastSAM3D-referenced hip-distance analysis helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_fastsam_hip_distance import (  # noqa: E402
    add_distance_bins,
    summarize_bins,
)


class FastSamHipDistanceTest(unittest.TestCase):
    """Cover fixed distance bins and common-frame eligibility."""

    def test_distance_bins_are_left_closed(self) -> None:
        data = pd.DataFrame(
            {
                "optical_depth_m": [2.0, 2.49, 2.5, 3.0],
            }
        )

        result = add_distance_bins(data, 0.5)

        self.assertEqual(
            result["distance_bin"].astype(str).tolist(),
            ["2.0–2.5", "2.0–2.5", "2.5–3.0", "3.0–3.5"],
        )

    def test_bin_eligibility_requires_both_models(self) -> None:
        rows = []
        for model, count in (("YOLOv8m", 5), ("YOLO11L", 3)):
            for index in range(count):
                rows.append(
                    {
                        "session": "Session A",
                        "session_id": "a",
                        "model": model,
                        "distance_bin": "2.0–2.5",
                        "distance_bin_left_m": 2.0,
                        "optical_depth_m": 2.2,
                        "bilateral_hip_abs_disagreement_deg": index + 1.0,
                    }
                )
        for model in ("YOLOv8m", "YOLO11L"):
            for index in range(5):
                rows.append(
                    {
                        "session": "Session B",
                        "session_id": "b",
                        "model": model,
                        "distance_bin": "2.5–3.0",
                        "distance_bin_left_m": 2.5,
                        "optical_depth_m": 2.7,
                        "bilateral_hip_abs_disagreement_deg": index + 2.0,
                    }
                )

        _, session_summary, labels = summarize_bins(
            pd.DataFrame(rows),
            minimum_common_frames=5,
        )

        self.assertEqual(labels, ["2.5–3.0"])
        self.assertEqual(len(session_summary), 4)


if __name__ == "__main__":
    unittest.main()
