"""Tests for synchronized FastSAM3D TRC pose-space fusion."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "stereo_trc_pose_fusion.py"
SPEC = importlib.util.spec_from_file_location("stereo_trc_pose_fusion", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StereoTrcPoseFusionTest(unittest.TestCase):
    """Validate the geometry primitives used by the experiment."""

    def test_fit_rigid_transform_recovers_known_mapping(self) -> None:
        """The fitted row-vector transform should recover a rigid mapping."""
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        angle = np.deg2rad(17.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        translation = np.asarray([4.0, -3.0, 2.0])
        target = source @ rotation + translation

        fitted_rotation, fitted_translation = MODULE.fit_rigid_transform(source, target)

        np.testing.assert_allclose(
            source @ fitted_rotation + fitted_translation, target, atol=1e-10
        )

    def test_equal_fusion_preserves_identical_and_one_sided_values(self) -> None:
        """Equal fusion should not perturb identical or valid one-sided coordinates."""
        left = np.asarray([[[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]]])
        right = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

        fused = MODULE.fuse_equal(left, right)

        np.testing.assert_allclose(
            fused, np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        )

    def test_pose_alignment_uses_rigid_transform(self) -> None:
        """Per-frame alignment should recover a transformed synthetic pose."""
        source = np.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        )
        angle = np.deg2rad(-12.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        target = source @ rotation + np.asarray([2.0, 3.0, -1.0])

        aligned, _, _, residual = MODULE.align_pose_sequence(
            source, target, [0, 1, 2, 3]
        )

        np.testing.assert_allclose(aligned, target, atol=1e-10)
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
