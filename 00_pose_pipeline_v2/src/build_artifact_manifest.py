#!/opt/anaconda3/envs/pose/bin/python
"""Build a checksummed manifest for one remote human-prior result bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_PARTS = {"model_assets", "model_weights", "cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".avi", ".mkv", ".pt", ".pth", ".engine", ".onnx", ".pkl"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--command", default="")
    parser.add_argument("--execution-status", choices=["completed", "failed", "blocked"], default="completed")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Hash a result artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    """Return one Git value or an explicit unavailable marker."""
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def allowed_artifact(path: Path, run_dir: Path) -> bool:
    """Exclude raw data, model weights, caches, and licensed assets."""
    relative = path.relative_to(run_dir)
    if path.name == "artifact_manifest.json":
        return False
    if EXCLUDED_PARTS.intersection(relative.parts):
        return False
    return path.suffix.lower() not in EXCLUDED_SUFFIXES


def main() -> None:
    """Write a deterministic manifest beside the artifacts."""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and allowed_artifact(path, run_dir):
            files.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    scientific_status = "unknown"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        scientific_status = str(metrics.get("scientific_status", metrics.get("geometry", {}).get("scientific_status", "unknown")))
    manifest = {
        "schema_version": "human_prior_artifacts_v1",
        "run_tag": args.run_tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": args.execution_status,
        "scientific_status": scientific_status,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "command": args.command,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "excluded_policy": "raw videos, model weights, caches, and licensed SMPL assets are never bundled",
    }
    output = run_dir / "artifact_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
