"""Unit tests for SMPL/COCO semantics and calibrated geometry helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common.smpl_prior import (  # noqa: E402
    body25_to_coco17,
    compute_geometry_quality,
    project_points_numpy,
    select_gate_indices,
)


class SmplPriorTest(unittest.TestCase):
    """Cover joint semantics, quality, projection, and gate intervals."""

    def test_body25_mapping_preserves_sides(self) -> None:
        body25 = np.zeros((1, 25, 3), dtype=np.float64)
        body25[0, :, 0] = np.arange(25)
        coco = body25_to_coco17(body25)
        self.assertEqual(coco[0, 5, 0], 5)
        self.assertEqual(coco[0, 6, 0], 2)
        self.assertEqual(coco[0, 11, 0], 12)
        self.assertEqual(coco[0, 12, 0], 9)

    def test_quality_matches_fixed_formula(self) -> None:
        confidence = np.full((1, 17), 0.81)
        epipolar = np.full((1, 17), 6.0)
        reprojection = np.full((1, 17), 10.0)
        quality = compute_geometry_quality(
            confidence, confidence, epipolar, reprojection
        )
        self.assertAlmostEqual(float(quality[0, 0]), 0.81 * np.exp(-2.0))

    def test_projection_uses_centimeter_coordinates(self) -> None:
        projection = np.asarray(
            [[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        points = np.asarray([[[10.0, 20.0, 100.0]]])
        np.testing.assert_allclose(
            project_points_numpy(points, projection), [[[10.0, 20.0]]]
        )

    def test_gate_is_centered_and_continuous(self) -> None:
        indices = select_gate_indices(100, "feasibility")
        np.testing.assert_array_equal(indices, np.arange(30, 70))


if __name__ == "__main__":
    unittest.main()
