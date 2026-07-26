"""Unit tests for NVIDIA mono/stereo matrix helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "00_pose_pipeline_v2" / "src"))

from adapt_nvidia_bodypose3d_monocular import (  # noqa: E402
    right_camera_to_left,
)
from common.candidate_evaluation import (  # noqa: E402
    angle_agreement,
    finite_distribution,
)
from evaluate_nvidia_bodypose3d_stereo import (  # noqa: E402
    COCO17_FROM_BODYPOSE34,
)
from adapt_foundation_stereo_joint_depth import (  # noqa: E402
    disparity_to_left_camera_cm,
    rectify_points_sequence,
    restore_full_resolution_disparity,
    sample_joint_disparity,
)


class NvidiaPoseMatrixTest(unittest.TestCase):
    """Validate mapping, transforms, and common-frame comparisons."""

    def test_bodypose34_mapping_covers_coco17_once(self) -> None:
        self.assertEqual(len(COCO17_FROM_BODYPOSE34), 17)
        self.assertEqual(len(set(COCO17_FROM_BODYPOSE34.tolist())), 17)
        self.assertTrue(
            np.all((COCO17_FROM_BODYPOSE34 >= 0) & (COCO17_FROM_BODYPOSE34 < 34))
        )

    def test_right_to_left_transform_inverts_stereo_extrinsics(self) -> None:
        angle = np.deg2rad(3.0)
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        translation = np.asarray([-41.0, 0.2, 0.4])
        points_left = np.asarray([[[10.0, 20.0, 300.0]]])
        points_right = (
            rotation @ points_left.reshape(-1, 3).T
        ).T + translation
        restored = right_camera_to_left(
            points_right.reshape(1, 1, 3),
            rotation,
            translation,
        )
        np.testing.assert_allclose(restored, points_left, atol=1e-9)

    def test_right_to_left_preserves_nan(self) -> None:
        result = right_camera_to_left(
            np.asarray([[[np.nan, 0.0, 1.0]]]),
            np.eye(3),
            np.zeros(3),
        )
        self.assertTrue(np.isnan(result).all())

    def test_angle_agreement_uses_identical_common_mask(self) -> None:
        candidate = np.asarray([10.0, 20.0, np.nan, 40.0])
        baseline = np.asarray([11.0, np.nan, 31.0, 41.0])
        reference = np.asarray([12.0, 22.0, 32.0, 42.0])
        result = angle_agreement(
            candidate,
            baseline,
            reference,
            [20.0, 45.0],
        )
        np.testing.assert_array_equal(
            result["common_finite_mask"],
            [True, False, False, True],
        )
        self.assertEqual(
            result["candidate"]["valid_pair_count"],
            result["baseline"]["valid_pair_count"],
        )

    def test_distribution_reports_tail(self) -> None:
        result = finite_distribution(
            np.asarray([0.0, 1.0, 2.0, np.nan, 100.0])
        )
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["max"], 100.0)
        self.assertGreater(float(result["p95"]), 80.0)

    def test_disparity_backprojection_uses_metric_baseline(self) -> None:
        points = np.zeros((1, 17, 2), dtype=np.float64)
        points[..., 0] = 1128.0
        points[..., 1] = 564.0
        disparity = np.full((1, 17), 112.8)
        projection = np.asarray(
            [
                [1128.0, 0.0, 1024.0, 0.0],
                [0.0, 1128.0, 768.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        result = disparity_to_left_camera_cm(
            points,
            disparity,
            projection,
            baseline_cm=41.0,
        )
        np.testing.assert_allclose(result[..., 2], 410.0)
        np.testing.assert_allclose(result[..., 0], 37.80141844)
        np.testing.assert_allclose(result[..., 1], -74.14893617)

    def test_joint_disparity_rejects_depth_discontinuity(self) -> None:
        disparity = np.full((30, 30), 20.0)
        points = np.full((17, 2), [15.0, 15.0])
        values, mad = sample_joint_disparity(
            disparity,
            points,
            patch_size=7,
            maximum_mad_px=0.1,
        )
        np.testing.assert_allclose(values, 20.0)
        np.testing.assert_allclose(mad, 0.0)
        disparity[12:19, 12:19] = np.arange(49).reshape(7, 7) + 2.0
        values, mad = sample_joint_disparity(
            disparity,
            points,
            patch_size=7,
            maximum_mad_px=0.1,
        )
        self.assertTrue(np.isnan(values).all())
        self.assertTrue(np.all(mad > 0.1))

    def test_disparity_resize_restores_full_resolution_units(self) -> None:
        scaled = np.full((4, 5), 20.0, dtype=np.float32)
        restored = restore_full_resolution_disparity(
            scaled,
            (10, 8),
            scale=0.5,
        )
        self.assertEqual(restored.shape, (8, 10))
        np.testing.assert_allclose(restored, 40.0)

    def test_left_point_rectification_is_view_independent(self) -> None:
        points = np.full((2, 17, 2), [120.0, 80.0])
        points[1, 3] = np.nan
        camera = np.asarray(
            [
                [500.0, 0.0, 100.0],
                [0.0, 500.0, 70.0],
                [0.0, 0.0, 1.0],
            ]
        )
        output = rectify_points_sequence(
            points,
            camera,
            np.zeros(5),
            np.eye(3),
            np.column_stack([camera, np.zeros(3)]),
        )
        np.testing.assert_allclose(output[0], points[0], atol=1e-9)
        self.assertTrue(np.isnan(output[1, 3]).all())


if __name__ == "__main__":
    unittest.main()
