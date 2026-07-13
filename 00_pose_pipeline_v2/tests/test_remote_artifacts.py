"""Tests for standard remote result bundle construction and verification."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from artifact_bundle import build_manifest, verify_manifest  # noqa: E402
from remote_experiment import rsync_command  # noqa: E402


class RemoteArtifactTest(unittest.TestCase):
    """Cover checksums, corruption, sensitive assets, and resumable rsync."""

    def test_manifest_verification_and_corruption_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics.json").write_text(
                json.dumps({"scientific_status": "reject"}), encoding="utf-8"
            )
            (run_dir / "candidate_result.npz").write_bytes(b"result")
            build_manifest(run_dir, "test", "command", "completed")
            self.assertTrue(verify_manifest(run_dir)["ok"])
            (run_dir / "candidate_result.npz").write_bytes(b"changed")
            result = verify_manifest(run_dir)
            self.assertFalse(result["ok"])
            self.assertIn("size:candidate_result.npz", result["failures"])

    def test_sensitive_asset_is_excluded_and_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            (run_dir / "SMPL_NEUTRAL.pkl").write_bytes(b"private")
            manifest = build_manifest(run_dir, "test", "", "blocked")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertNotIn("SMPL_NEUTRAL.pkl", [item["path"] for item in payload["files"]])
            self.assertFalse(verify_manifest(run_dir)["ok"])

    def test_rsync_is_resumable_and_excludes_private_models(self) -> None:
        command = rsync_command("host", "/workspace/PoseEst1", "tag")
        self.assertIn("--partial", command)
        self.assertIn("--checksum", command)
        self.assertIn("*.pkl", command)
        self.assertNotIn("--delete", command)


if __name__ == "__main__":
    unittest.main()
