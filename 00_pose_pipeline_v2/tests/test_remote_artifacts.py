"""Tests for remote human-prior result packaging and synchronization policy."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "00_pose_pipeline_v2" / "src"))

from build_artifact_manifest import allowed_artifact  # noqa: E402
from remote_experiment import standard_rsync_command  # noqa: E402


class RemoteArtifactTest(unittest.TestCase):
    """Ensure private assets and large inputs never enter standard bundles."""

    def test_manifest_policy_excludes_models_and_raw_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertTrue(allowed_artifact(run_dir / "candidate_result.npz", run_dir))
            self.assertFalse(allowed_artifact(run_dir / "SMPL_NEUTRAL.pkl", run_dir))
            self.assertFalse(allowed_artifact(run_dir / "raw.avi", run_dir))
            self.assertFalse(allowed_artifact(run_dir / "cache" / "tensor.npy", run_dir))

    def test_rsync_standard_profile_is_non_destructive(self) -> None:
        command = standard_rsync_command("poseest1-runpod", "/workspace/PoseEst1", "run-tag", False)
        joined = " ".join(command)
        self.assertIn("--partial", command)
        self.assertIn("--checksum", command)
        self.assertNotIn("--delete", command)
        self.assertIn("*.pkl", joined)
        self.assertIn("human_prior_fusion/run-tag", joined)


if __name__ == "__main__":
    unittest.main()
