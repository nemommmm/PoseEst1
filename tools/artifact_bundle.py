#!/opt/anaconda3/envs/pose/bin/python
"""Build and verify checksummed research experiment result bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXCLUDED_PARTS = {"model_assets", "model_weights", "cache", "__pycache__"}
EXCLUDED_SUFFIXES = {
    ".avi", ".mkv", ".pt", ".pth", ".engine", ".onnx", ".pkl",
}


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed_artifact(path: Path, run_dir: Path) -> bool:
    """Return whether a file belongs in a standard downloadable bundle."""
    relative = path.relative_to(run_dir)
    if path.name == "artifact_manifest.json":
        return False
    if EXCLUDED_PARTS.intersection(relative.parts):
        return False
    return path.suffix.lower() not in EXCLUDED_SUFFIXES


def git_value(*args: str) -> str:
    """Return a Git value or an explicit unavailable marker."""
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def scientific_status(run_dir: Path) -> str:
    """Read the scientific state without assuming a candidate-specific schema."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return "unknown"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    direct = metrics.get("scientific_status")
    if direct is not None:
        return str(direct)
    geometry = metrics.get("geometry", {})
    return str(geometry.get("scientific_status", "unknown"))


def build_manifest(
    run_dir: Path,
    run_tag: str,
    command: str,
    execution_status: str,
) -> Path:
    """Build a deterministic manifest for an existing experiment directory."""
    resolved = run_dir.resolve()
    files = []
    for path in sorted(resolved.rglob("*")):
        if path.is_file() and allowed_artifact(path, resolved):
            files.append(
                {
                    "path": str(path.relative_to(resolved)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": "human_prior_artifacts_v1",
        "run_tag": run_tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": execution_status,
        "scientific_status": scientific_status(resolved),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "command": command,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "excluded_policy": (
            "raw videos, model weights, caches, and licensed SMPL assets "
            "are never bundled"
        ),
    }
    output = resolved / "artifact_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def verify_manifest(run_dir: Path) -> dict[str, Any]:
    """Verify every listed file and reject unlisted sensitive artifacts."""
    resolved = run_dir.resolve()
    manifest_path = resolved / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in manifest.get("files", []):
        relative = Path(item["path"])
        path = resolved / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        if not allowed_artifact(path, resolved):
            failures.append(f"sensitive:{relative}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            failures.append(f"size:{relative}")
        elif sha256(path) != item["sha256"]:
            failures.append(f"sha256:{relative}")
    actual_sensitive = [
        str(path.relative_to(resolved))
        for path in resolved.rglob("*")
        if path.is_file() and not allowed_artifact(path, resolved)
        and path.name != "artifact_manifest.json"
    ]
    failures.extend(f"sensitive:{path}" for path in actual_sensitive)
    return {
        "run_tag": manifest.get("run_tag"),
        "verified_files": len(manifest.get("files", [])),
        "ok": not failures,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    """Parse manifest build and verification arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument("--run-tag", required=True)
    build.add_argument("--command", default="")
    build.add_argument(
        "--execution-status",
        choices=["completed", "failed", "blocked"],
        default="completed",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build or verify one result bundle."""
    args = parse_args()
    if args.action == "build":
        print(
            build_manifest(
                args.run_dir,
                args.run_tag,
                args.command,
                args.execution_status,
            )
        )
        return
    result = verify_manifest(args.run_dir)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
