"""Tests for explicit CPU benchmark device handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common.person_tracking import (  # noqa: E402
    TrackState,
    TrackingConfig,
    infer_tracked_pose,
)


class _EmptyResult:
    boxes = None
    keypoints = None


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, _image: np.ndarray, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        return [_EmptyResult()]


class CpuBenchmarkDeviceTest(unittest.TestCase):
    """Ensure benchmark device overrides do not change normal inference."""

    def test_explicit_cpu_is_forwarded_to_ultralytics(self) -> None:
        model = _RecordingModel()
        infer_tracked_pose(
            model,
            np.zeros((32, 32, 3), dtype=np.uint8),
            TrackState(),
            0,
            TrackingConfig(enabled=False),
            device="cpu",
        )
        self.assertEqual(model.calls[0]["device"], "cpu")

    def test_default_inference_does_not_override_device(self) -> None:
        model = _RecordingModel()
        infer_tracked_pose(
            model,
            np.zeros((32, 32, 3), dtype=np.uint8),
            TrackState(),
            0,
            TrackingConfig(enabled=False),
        )
        self.assertNotIn("device", model.calls[0])


if __name__ == "__main__":
    unittest.main()
