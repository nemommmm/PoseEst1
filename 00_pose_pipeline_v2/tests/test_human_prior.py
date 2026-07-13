"""Synthetic tests for geometry-conditioned stereo human-prior fitting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common.human_prior import (  # noqa: E402
    KinematicFitConfig,
    compute_geometry_quality,
    compute_reprojection_errors,
    fit_kinematic_sequence,
    select_gate_indices,
)


def synthetic_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one simple moving skeleton and its calibrated stereo projections."""
    frames = 9
    pose = np.full((frames, 17, 3), np.nan, dtype=np.float64)
    joints = {
        5: (-20.0, 0.0, 300.0), 6: (20.0, 0.0, 300.0),
        7: (-35.0, 20.0, 300.0), 8: (35.0, 20.0, 300.0),
        9: (-45.0, 40.0, 300.0), 10: (45.0, 40.0, 300.0),
        11: (-15.0, 55.0, 300.0), 12: (15.0, 55.0, 300.0),
        13: (-15.0, 95.0, 300.0), 14: (15.0, 95.0, 300.0),
        15: (-15.0, 135.0, 300.0), 16: (15.0, 135.0, 300.0),
    }
    for joint_idx, point in joints.items():
        pose[:, joint_idx] = point
    pose[:, :, 0] += np.arange(frames)[:, None] * 0.2
    projection_left = np.array(
        [[1000.0, 0.0, 1024.0, 0.0], [0.0, 1000.0, 768.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    projection_right = np.array(
        [[1000.0, 0.0, 1024.0, -41000.0], [0.0, 1000.0, 768.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    def project(points: np.ndarray, projection: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=2)
        projected = homogeneous @ projection.T
        return projected[..., :2] / projected[..., 2:3]

    return pose, project(pose, projection_left), project(pose, projection_right), projection_left, projection_right


class HumanPriorTest(unittest.TestCase):
    """Check quality gates, interval selection, and calibrated optimization."""

    def test_geometry_quality_penalizes_bad_observations(self) -> None:
        confidence = np.array([[0.9, 0.9]], dtype=np.float64)
        quality = compute_geometry_quality(
            confidence,
            confidence,
            np.array([[0.0, 12.0]]),
            np.array([[0.0, 20.0]]),
        )
        self.assertGreater(quality[0, 0], quality[0, 1])
        self.assertGreater(quality[0, 0], 0.8)

    def test_gate_indices_are_centered_and_continuous(self) -> None:
        feasibility = select_gate_indices(100, "feasibility")
        np.testing.assert_array_equal(feasibility, np.arange(30, 70))
        np.testing.assert_array_equal(select_gate_indices(20, "short"), np.arange(20))
        with self.assertRaises(ValueError):
            select_gate_indices(10, "unknown")

    def test_fit_preserves_high_quality_stereo_geometry(self) -> None:
        pose, left, right, projection_left, projection_right = synthetic_sequence()
        raw = pose.copy()
        raw[:, 10, 0] += 0.2
        confidence = np.full((len(raw), 17), 0.9, dtype=np.float64)
        epipolar = np.abs(left[..., 1] - right[..., 1])
        reprojection = compute_reprojection_errors(raw, left, right, projection_left, projection_right)
        config = KinematicFitConfig(
            iterations=5,
            anchor_weights=(1.0,),
            bone_weights=(1.0,),
            temporal_weights=(0.05,),
            max_reprojection_p95_px=20.0,
            device="cpu",
        )
        result = fit_kinematic_sequence(
            raw,
            left,
            right,
            confidence,
            confidence,
            epipolar,
            reprojection,
            projection_left,
            projection_right,
            config,
        )
        self.assertTrue(result.metrics["geometry_gate_pass"])
        self.assertLess(result.metrics["high_quality_correction_median_cm"], 1.0)
        self.assertLess(result.metrics["reprojection_p95_px"], 1.0)
        self.assertEqual(result.keypoints_3d.shape, raw.shape)


if __name__ == "__main__":
    unittest.main()
