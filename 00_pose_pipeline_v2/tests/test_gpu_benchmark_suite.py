"""Tests for repeated GPU benchmark orchestration and comparisons."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_gpu_benchmark_suite import (  # noqa: E402
    compare_npz,
    pooled_timing_summary,
    reusable_repeat_outputs,
)


def synthetic_keypoints(frames: int = 4) -> np.ndarray:
    """Return a finite COCO-17 skeleton with a 90-degree right elbow."""
    keypoints = np.zeros((frames, 17, 3), dtype=np.float64)
    keypoints[:, 6] = [0.0, 0.0, 0.0]
    keypoints[:, 8] = [1.0, 0.0, 0.0]
    keypoints[:, 10] = [1.0, 1.0, 0.0]
    return keypoints


def write_comparison_npz(path: Path, keypoints: np.ndarray) -> None:
    """Write the minimal numeric and categorical comparison payload."""
    np.savez_compressed(
        path,
        timestamps=np.arange(len(keypoints), dtype=np.float64),
        keypoints=keypoints,
        track_source_left=np.asarray(["full"] * len(keypoints)),
        stereo_sanity_ok=np.ones(len(keypoints), dtype=bool),
    )


class GpuBenchmarkSuiteTest(unittest.TestCase):
    """Cover resume validation, timing aggregation, and scientific equality."""

    def test_reusable_repeat_requires_readable_json_and_npz_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repeat_dir = Path(temporary)
            (repeat_dir / "benchmark.json").write_text(
                json.dumps({"stages": {}, "online_fps": 29.0}),
                encoding="utf-8",
            )
            np.savez_compressed(
                repeat_dir / "skt_pose_optimized.npz",
                timestamps=np.arange(3),
                keypoints=synthetic_keypoints(3),
                frame_time_ms=np.ones(3),
                yolo_time_ms=np.ones(3),
            )
            self.assertTrue(reusable_repeat_outputs(repeat_dir))
            (repeat_dir / "benchmark.json").write_text(
                "not json", encoding="utf-8"
            )
            self.assertFalse(reusable_repeat_outputs(repeat_dir))

    def test_identical_npz_passes_exact_repeat_gate(self) -> None:
        thresholds = {
            "require_key_arrays_exact": True,
            "min_finite_mask_agreement": 1.0,
            "max_keypoint_p95_distance_cm": 1e-6,
            "max_right_elbow_mae_deg": 1e-6,
            "min_rula_bin_agreement": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.npz"
            second = root / "second.npz"
            keypoints = synthetic_keypoints()
            write_comparison_npz(first, keypoints)
            write_comparison_npz(second, keypoints.copy())
            result = compare_npz(
                second,
                first,
                [
                    "timestamps",
                    "keypoints",
                    "track_source_left",
                    "stereo_sanity_ok",
                ],
                "RightElbow",
                [60.0, 100.0],
                thresholds,
            )
            self.assertTrue(result["key_arrays_all_exact"])
            self.assertTrue(result["passes_thresholds"])
            self.assertEqual(result["angle"]["rula_bin_agreement"], 1.0)

    def test_historical_gate_can_accept_small_rigid_translation(self) -> None:
        thresholds = {
            "require_key_arrays_exact": False,
            "min_finite_mask_agreement": 1.0,
            "max_keypoint_p95_distance_cm": 0.02,
            "max_right_elbow_mae_deg": 0.1,
            "min_rula_bin_agreement": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            keypoints = synthetic_keypoints()
            write_comparison_npz(reference, keypoints)
            write_comparison_npz(candidate, keypoints + 0.01)
            result = compare_npz(
                candidate,
                reference,
                ["timestamps", "keypoints"],
                "RightElbow",
                [60.0, 100.0],
                thresholds,
            )
            self.assertFalse(result["key_arrays_all_exact"])
            self.assertTrue(result["passes_thresholds"])
            self.assertAlmostEqual(
                result["keypoints"]["distance_cm"]["p95"],
                np.sqrt(3.0) * 0.01,
            )
            self.assertAlmostEqual(
                result["angle"]["absolute_difference_deg"]["mean"], 0.0
            )

    def test_pooled_timing_excludes_each_repeat_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, values in enumerate(([999.0, 20.0, 40.0], [999.0, 30.0, 50.0])):
                path = root / f"repeat_{index}.npz"
                np.savez_compressed(
                    path,
                    decode_time_ms=np.asarray(values) / 4.0,
                    yolo_time_ms=np.asarray(values) / 2.0,
                    geometry_time_ms=np.asarray(values) / 20.0,
                    frame_time_ms=np.asarray(values),
                )
                paths.append(path)
            result = pooled_timing_summary(
                paths,
                warmup_frames=1,
                repeat_benchmarks=[
                    {"online_fps": 25.0},
                    {"online_fps": 30.0},
                ],
            )
            online = result["stages"]["end_to_end_online"]
            self.assertAlmostEqual(online["pooled_frame_ms"]["mean"], 35.0)
            self.assertAlmostEqual(online["pooled_frame_ms"]["median"], 35.0)
            self.assertAlmostEqual(online["pooled_frame_ms"]["p95"], 48.5)
            self.assertAlmostEqual(online["pooled_fps"], 1000.0 / 35.0)
            self.assertAlmostEqual(
                result["online_fps_across_repeats"]["median"], 27.5
            )


if __name__ == "__main__":
    unittest.main()
