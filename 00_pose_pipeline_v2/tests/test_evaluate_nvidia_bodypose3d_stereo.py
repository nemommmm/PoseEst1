"""Unit tests for the NVIDIA BodyPose3DNet stereo adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "00_pose_pipeline_v2" / "src"))

from evaluate_nvidia_bodypose3d_stereo import (  # noqa: E402
    COCO17_FROM_BODYPOSE34,
    BodyPoseEvaluationError,
    PoseRecord,
    align_tracks_to_synced_timeline,
    index_primary_person,
    load_deepstream_records,
    load_processed_skt_angles,
    select_primary_track,
)
from stereo_loader import SyncedFrame  # noqa: E402


def pose_vector(offset: float = 0.0) -> list[float]:
    """Return a valid, distinguishable 34-by-4 pose vector."""

    values = np.zeros((34, 4), dtype=np.float64)
    values[:, 0] = np.arange(34) + offset
    values[:, 1] = 2 * np.arange(34) + offset
    values[:, 2] = 3 * np.arange(34) + offset
    values[:, 3] = 0.8
    return values.reshape(-1).tolist()


class NvidiaBodyPoseStereoTest(unittest.TestCase):
    """Exercise parsing, track selection, mapping, and synchronization."""

    def test_parser_validates_and_reshapes_pose_vectors(self) -> None:
        payload = [
            {
                "batches": [
                    {
                        "frame_num": 4,
                        "objects": [
                            {
                                "object_id": 9,
                                "pose25d": pose_vector(),
                                "pose3d": pose_vector(1.0),
                            }
                        ],
                    }
                ]
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = load_deepstream_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].pose25d.shape, (34, 4))
        self.assertEqual(records[0].frame_num, 4)
        self.assertEqual(records[0].object_id, 9)

    def test_parser_rejects_wrong_joint_count(self) -> None:
        payload = [
            {
                "batches": [
                    {
                        "frame_num": 0,
                        "objects": [
                            {
                                "object_id": 1,
                                "pose25d": [0.0] * 8,
                                "pose3d": pose_vector(),
                            }
                        ],
                    }
                ]
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BodyPoseEvaluationError):
                load_deepstream_records(path)

    def test_primary_track_prefers_persistence_before_size(self) -> None:
        records = []
        for frame in range(5):
            records.append(
                PoseRecord(
                    frame,
                    1,
                    np.asarray(pose_vector()).reshape(34, 4),
                    np.asarray(pose_vector()).reshape(34, 4),
                )
            )
        for frame in range(2):
            large = np.asarray(pose_vector()).reshape(34, 4)
            large[:, :2] *= 10
            records.append(PoseRecord(frame, 2, large, large))
        selected, summaries = select_primary_track(records)
        self.assertEqual(selected, 1)
        self.assertEqual(summaries[0]["frame_count"], 5)

    def test_coco_mapping_and_raw_frame_synchronization(self) -> None:
        left_pose = np.asarray(pose_vector()).reshape(34, 4)
        right_pose = np.asarray(pose_vector(100.0)).reshape(34, 4)
        left = {3: PoseRecord(3, 1, left_pose, left_pose)}
        right = {7: PoseRecord(7, 1, right_pose, right_pose)}
        synced = [
            SyncedFrame(frame_id=10, left_idx=3, right_idx=7, ts=0.0)
        ]
        result = align_tracks_to_synced_timeline(
            left, right, synced, frame_count=1
        )
        np.testing.assert_array_equal(
            result["keypoints_left_2d_raw"][0, :, 0],
            COCO17_FROM_BODYPOSE34,
        )
        np.testing.assert_array_equal(
            result["keypoints_right_2d_raw"][0, :, 0],
            COCO17_FROM_BODYPOSE34 + 100,
        )
        np.testing.assert_array_equal(
            result["source_frame_indices"], [[3, 7]]
        )
        np.testing.assert_array_equal(result["track_present"], [[True, True]])
        np.testing.assert_array_equal(
            result["selected_object_ids"], [[1, 1]]
        )

    def test_primary_index_accepts_large_nonoverlapping_id_continuation(
        self,
    ) -> None:
        records = []
        anchor = np.asarray(pose_vector()).reshape(34, 4)
        continuation = anchor.copy()
        continuation[:, 1] *= 1.2
        false_positive = anchor.copy()
        false_positive[:, :2] *= 0.3
        for frame in range(4):
            records.append(PoseRecord(frame, 1, anchor, anchor))
            records.append(
                PoseRecord(frame, 7, false_positive, false_positive)
            )
        records.append(PoseRecord(5, 9, continuation, continuation))
        records.append(PoseRecord(5, 7, false_positive, false_positive))

        indexed, summary = index_primary_person(records)

        self.assertEqual(indexed[0].object_id, 1)
        self.assertEqual(indexed[5].object_id, 9)
        self.assertEqual(summary["continuation_object_ids"], [9])

    def test_same_input_skt_control_uses_saved_timeline(self) -> None:
        frame_count = 10
        timestamps = np.arange(frame_count, dtype=np.float64) * 0.04
        keypoints = np.zeros((frame_count, 17, 3), dtype=np.float64)
        for joint_index in range(17):
            keypoints[:, joint_index, :] = [
                joint_index,
                joint_index % 4,
                100.0 + joint_index,
            ]
        triang_confidence = np.ones((frame_count, 17), dtype=np.float64)
        epipolar_error = np.zeros((frame_count, 17), dtype=np.float64)
        config = {
            "evaluation": {
                "camera_smooth_window_ms": 200.0,
                "max_gap_frames": 5,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.npz"
            np.savez_compressed(
                path,
                keypoints=keypoints,
                triang_conf_left=triang_confidence,
                triang_conf_right=triang_confidence,
                epipolar_error=epipolar_error,
                timestamps=timestamps,
            )
            angles, metadata = load_processed_skt_angles(
                path, frame_count, timestamps, config
            )
            with self.assertRaises(BodyPoseEvaluationError):
                load_processed_skt_angles(
                    path, frame_count, timestamps + 0.01, config
                )

        self.assertIn("RightElbow", angles)
        self.assertEqual(len(angles["RightElbow"]), frame_count)
        self.assertIn("angle_processing", metadata)


if __name__ == "__main__":
    unittest.main()
