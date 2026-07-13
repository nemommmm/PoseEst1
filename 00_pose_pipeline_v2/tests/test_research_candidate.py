"""Tests for the unified external research candidate schema."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common.research_candidate import (  # noqa: E402
    CandidateResult,
    convert_to_centimeters,
    load_candidate_npz,
    map_to_coco17,
    transform_points,
)


class ResearchCandidateTest(unittest.TestCase):
    """Cover joint semantics, units, coordinates, and serialization."""

    def test_named_mapping_preserves_left_and_right(self) -> None:
        source = np.zeros((1, 4, 3), dtype=np.float64)
        source[0, :, 0] = [1.0, 2.0, 3.0, 4.0]
        mapped = map_to_coco17(
            source,
            ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
        )
        self.assertEqual(mapped[0, 5, 0], 1.0)
        self.assertEqual(mapped[0, 6, 0], 2.0)
        self.assertEqual(mapped[0, 7, 0], 3.0)
        self.assertEqual(mapped[0, 8, 0], 4.0)

    def test_unit_conversion(self) -> None:
        points = np.ones((1, 17, 3), dtype=np.float64)
        np.testing.assert_allclose(convert_to_centimeters(points, "m"), 100.0)
        np.testing.assert_allclose(convert_to_centimeters(points, "mm"), 0.1)
        np.testing.assert_allclose(convert_to_centimeters(points, "cm"), 1.0)
        with self.assertRaises(ValueError):
            convert_to_centimeters(points, "inch")

    def test_rigid_transform_keeps_nan_and_applies_translation(self) -> None:
        points = np.zeros((1, 17, 3), dtype=np.float64)
        points[0, 0] = np.nan
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        output = transform_points(points, transform)
        self.assertTrue(np.isnan(output[0, 0]).all())
        np.testing.assert_allclose(output[0, 1], [1.0, 2.0, 3.0])

    def test_angle_direction_and_npz_schema(self) -> None:
        pose = np.full((1, 17, 3), np.nan, dtype=np.float64)
        pose[0, 5] = [-1.0, 0.0, 1.0]
        pose[0, 6] = [1.0, 0.0, 1.0]
        pose[0, 7] = [-1.0, 0.0, 0.0]
        pose[0, 8] = [1.0, 0.0, 0.0]
        pose[0, 9] = [-2.0, 0.0, 0.0]
        pose[0, 10] = [2.0, 0.0, 0.0]
        pose[0, 11] = [-0.5, 0.0, -1.0]
        pose[0, 12] = [0.5, 0.0, -1.0]
        pose[0, 13] = [-0.5, 0.0, -2.0]
        pose[0, 14] = [0.5, 0.0, -2.0]
        pose[0, 15] = [-0.5, 0.0, -3.0]
        pose[0, 16] = [0.5, 0.0, -3.0]
        quality = np.full((1, 17), 0.8, dtype=np.float64)
        result = CandidateResult(
            "synthetic",
            np.array([0.0]),
            pose,
            keypoints_3d_raw=pose.copy(),
            joint_quality=quality,
            prior_weight=1.0 - quality,
            extra_arrays={"smpl_betas": np.zeros(10, dtype=np.float64)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = result.save(Path(tmp) / "candidate.npz")
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(str(payload["schema_version"]), "research_candidate_v2")
                np.testing.assert_allclose(payload["keypoints_3d_raw"], pose, equal_nan=True)
                np.testing.assert_allclose(payload["joint_quality"], quality)
                self.assertEqual(payload["smpl_betas"].shape, (10,))
                angle_names = list(payload["angle_names"])
                right_elbow = payload["angles"][0, angle_names.index("RightElbow")]
                self.assertAlmostEqual(float(right_elbow), 90.0)

    def test_reserved_extra_array_is_rejected(self) -> None:
        pose = np.zeros((1, 17, 3), dtype=np.float64)
        result = CandidateResult(
            "invalid",
            np.array([0.0]),
            pose,
            extra_arrays={"timestamps": np.array([1.0])},
        )
        with self.assertRaisesRegex(ValueError, "reserved names"):
            result.validate()

    def test_v1_and_legacy_candidate_loading(self) -> None:
        pose = np.zeros((2, 17, 3), dtype=np.float64)
        timestamps = np.array([0.0, 0.08], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            v1_path = Path(tmp) / "v1.npz"
            np.savez_compressed(
                v1_path,
                schema_version=np.asarray("research_candidate_v1"),
                candidate_name=np.asarray("v1-test"),
                timestamps=timestamps,
                keypoints_3d=pose,
                metadata_json=np.asarray("{}"),
            )
            loaded_v1 = load_candidate_npz(v1_path)
            self.assertEqual(loaded_v1["schema_version"], "research_candidate_v1")
            np.testing.assert_array_equal(loaded_v1["keypoints_3d_raw"], pose)

            legacy_path = Path(tmp) / "legacy.npz"
            np.savez_compressed(legacy_path, timestamps=timestamps, keypoints=pose)
            loaded_legacy = load_candidate_npz(legacy_path)
            self.assertEqual(loaded_legacy["schema_version"], "legacy_skt")
            np.testing.assert_array_equal(loaded_legacy["keypoints_3d"], pose)


if __name__ == "__main__":
    unittest.main()
