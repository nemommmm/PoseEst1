"""Unit tests for NVIDIA mono/stereo matrix helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "00_pose_pipeline_v2" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

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
from run_nvidia_bodypose3d_matrix import create_warmup_video  # noqa: E402
from remote_gpu_pose_matrix import rsync_remote_destination  # noqa: E402
from generate_nvidia_pose_matrix_report import (  # noqa: E402
    assess_method_gates,
    collect_dataset_fps,
    collect_evaluations,
    reconcile_repaired_evaluation_statuses,
    relative_score_class,
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

    def test_bodypose_warmup_uses_exact_short_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mkv"
            source.write_bytes(b"source")
            output = root / "warmup.mkv"

            def create_mock_output(command: list[str], **_: object) -> str:
                Path(command[-1]).write_bytes(b"warmup")
                self.assertEqual(command[command.index("-frames:v") + 1], "10")
                self.assertEqual(command[command.index("-c:v") + 1], "libx264")
                self.assertEqual(command[command.index("-qp") + 1], "0")
                return ""

            with patch(
                "run_nvidia_bodypose3d_matrix.run_text",
                side_effect=create_mock_output,
            ):
                result = create_warmup_video(source, output, 10)
            self.assertEqual(result, output)

    def test_rsync_remote_path_preserves_spaces(self) -> None:
        self.assertEqual(
            rsync_remote_destination(
                "poseest1-runpod",
                "/workspace/PoseEst1/TRC FastSAM3D/",
            ),
            "poseest1-runpod:'/workspace/PoseEst1/TRC FastSAM3D/'",
        )

    def test_report_collects_nested_bodypose_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = (
                root
                / "bodypose3d/evaluation/fanbo3/accuracy/stereo/metrics.json"
            )
            target.parent.mkdir(parents=True)
            target.write_text(
                json.dumps(
                    {
                        "candidate": "BodyPose3DNet-accuracy-stereo",
                        "dataset": "assar2026_fanbo3_a255",
                        "rows": [
                            {
                                "candidate": {
                                    "absolute_error_deg": {
                                        "median": 1.0,
                                        "p95": 2.0,
                                    },
                                    "valid_ratio": 0.8,
                                    "rula_like_agreement": 0.9,
                                    "jump_count": 2,
                                },
                                "baseline": {
                                    "absolute_error_deg": {
                                        "median": 1.5,
                                        "p95": 3.0,
                                    },
                                    "valid_ratio": 0.85,
                                    "rula_like_agreement": 0.9,
                                },
                            }
                        ],
                        "aggregate_median_improvement_ratio": 1.0 / 3.0,
                    }
                ),
                encoding="utf-8",
            )
            rows = collect_evaluations(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "Fanbo3")

    def test_report_collects_dataset_fps_and_conservative_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline/fanbo3/benchmark.json"
            baseline.parent.mkdir(parents=True)
            baseline.write_text(
                json.dumps({"online_fps": 30.0}),
                encoding="utf-8",
            )
            selection = {
                "accepted": {
                    "fanbo3": {"synchronized_frames": 100},
                }
            }
            (root / "input_selection.json").write_text(
                json.dumps(selection),
                encoding="utf-8",
            )
            body_root = root / "bodypose3d"
            body_root.mkdir()
            summary = {
                "results": [
                    {
                        "dataset": "fanbo3",
                        "mode": "accuracy",
                        "side": "left",
                        "trials": [{"elapsed_seconds": 2.0}],
                    },
                    {
                        "dataset": "fanbo3",
                        "mode": "accuracy",
                        "side": "right",
                        "trials": [{"elapsed_seconds": 4.0}],
                    },
                ]
            }
            (body_root / "bodypose3d_run_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            values = collect_dataset_fps(root)
            self.assertEqual(
                values[("YOLOv8m-PyTorch-SKT", "Fanbo3")],
                30.0,
            )
            self.assertEqual(
                values[
                    (
                        "BodyPose3DNet-accuracy_monocular_left",
                        "Fanbo3",
                    )
                ],
                50.0,
            )
            self.assertAlmostEqual(
                values[("BodyPose3DNet-accuracy-stereo", "Fanbo3")],
                1.0 / (1.0 / 50.0 + 1.0 / 25.0),
            )

    def test_report_score_colors_respect_metric_direction(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(
            relative_score_class(1.0, values, higher_is_better=False),
            "score-good",
        )
        self.assertEqual(
            relative_score_class(4.0, values, higher_is_better=False),
            "score-bad",
        )
        self.assertEqual(
            relative_score_class(4.0, values, higher_is_better=True),
            "score-good",
        )

    def test_report_gate_rejects_missing_dataset_even_if_fast(self) -> None:
        evaluations = [
            {
                "candidate": "Candidate",
                "dataset": dataset,
                "median": 5.0,
                "baseline_median": 10.0,
                "rula": 1.0,
                "baseline_rula": 1.0,
                "valid_ratio": 1.0,
                "baseline_valid_ratio": 1.0,
            }
            for dataset in ("Fanbo3", "Fanbo7")
        ]
        methods = [
            {
                "candidate": "Candidate",
                "fps": 30.0,
                "latency_p95_ms": 40.0,
            }
        ]
        result = assess_method_gates(evaluations, methods)["Candidate"]
        self.assertFalse(result["offline_passed"])
        self.assertFalse(result["realtime_passed"])
        self.assertIn("not all three datasets completed", result["reasons"])

    def test_report_gate_requires_accuracy_rula_validity_and_latency(self) -> None:
        evaluations = [
            {
                "candidate": "Candidate",
                "dataset": dataset,
                "median": 9.0,
                "baseline_median": 10.0,
                "rula": 0.95,
                "baseline_rula": 0.95,
                "valid_ratio": 0.98,
                "baseline_valid_ratio": 1.0,
            }
            for dataset in ("Fanbo3", "Fanbo4", "Fanbo7")
        ]
        methods = [
            {
                "candidate": "Candidate",
                "fps": 15.0,
                "latency_p95_ms": 75.0,
            }
        ]
        result = assess_method_gates(evaluations, methods)["Candidate"]
        self.assertTrue(result["offline_passed"])
        self.assertTrue(result["realtime_passed"])
        methods[0]["latency_p95_ms"] = 81.0
        result = assess_method_gates(evaluations, methods)["Candidate"]
        self.assertTrue(result["offline_passed"])
        self.assertFalse(result["realtime_passed"])
        evaluations[0]["rula"] = None
        result = assess_method_gates(evaluations, methods)["Candidate"]
        self.assertFalse(result["offline_passed"])
        self.assertIn("RULA-like agreement is unavailable", result["reasons"])

    def test_report_reconciles_rerun_bodypose_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "candidate_matrix_status.json"
            status.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "route": "BodyPose3DNet_stereo",
                                "dataset": "fanbo4",
                                "mode": "accuracy",
                                "status": "failed",
                                "return_code": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            metrics = (
                root
                / "bodypose3d/evaluation/fanbo4/accuracy/stereo/metrics.json"
            )
            metrics.parent.mkdir(parents=True)
            metrics.write_text("{}", encoding="utf-8")
            self.assertEqual(reconcile_repaired_evaluation_statuses(root), 1)
            repaired = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(
                repaired["records"][0]["status"],
                "completed_after_reference_upload_repair",
            )
            self.assertEqual(repaired["records"][0]["return_code"], 0)


if __name__ == "__main__":
    unittest.main()
