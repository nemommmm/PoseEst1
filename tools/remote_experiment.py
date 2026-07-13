#!/opt/anaconda3/envs/pose/bin/python
"""Run and synchronize reproducible human-prior experiments on RunPod."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_ROOT = "/workspace/PoseEst1"
DEFAULT_RESULTS_RELATIVE = Path("00_pose_pipeline_v2/runs/human_prior_fusion")


def parse_args() -> argparse.Namespace:
    """Parse run or synchronization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="Run one remote candidate and synchronize its standard bundle.")
    run.add_argument("--candidate", choices=["kinematic", "smpl"], required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--gate", choices=["feasibility", "short", "full"], required=True)
    run.add_argument("--sync-profile", choices=["standard"], default="standard")
    run.add_argument("--run-tag")
    run.add_argument("--dry-run", action="store_true")

    sync = subparsers.add_parser("sync", help="Resume synchronization for an existing remote run.")
    sync.add_argument("--run-tag", required=True)
    sync.add_argument("--sync-profile", choices=["standard"], default="standard")
    sync.add_argument("--dry-run", action="store_true")

    for child in (run, sync):
        child.add_argument("--host", default="poseest1-runpod")
        child.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    return parser.parse_args()


def checked_output(command: list[str], cwd: Path = PROJECT_ROOT) -> str:
    """Return stripped output from one local command."""
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def dataset_name(config_path: Path) -> str:
    """Read the stable dataset name from one experiment configuration."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return str(config.get("dataset", {}).get("name", config_path.stem))


def build_run_tag(config_path: Path, candidate: str, gate: str) -> str:
    """Build the required descriptive and collision-resistant run tag."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    git_short = checked_output(["git", "rev-parse", "--short=7", "HEAD"])
    return f"{dataset_name(config_path)}__{candidate}__{gate}__{timestamp}__{git_short}"


def standard_rsync_command(host: str, remote_root: str, run_tag: str, dry_run: bool) -> list[str]:
    """Build a resumable standard-profile rsync command."""
    local_dir = PROJECT_ROOT / DEFAULT_RESULTS_RELATIVE / run_tag
    command = [
        "rsync",
        "-avP",
        "--partial",
        "--checksum",
        "--exclude", "*.avi",
        "--exclude", "*.mkv",
        "--exclude", "*.pt",
        "--exclude", "*.pth",
        "--exclude", "*.engine",
        "--exclude", "*.onnx",
        "--exclude", "*.pkl",
        f"{host}:{remote_root}/{DEFAULT_RESULTS_RELATIVE}/{run_tag}/",
        f"{local_dir}/",
    ]
    if dry_run:
        command.insert(1, "--dry-run")
    return command


def synchronize(host: str, remote_root: str, run_tag: str, dry_run: bool) -> int:
    """Synchronize one result bundle without deleting its remote source."""
    command = standard_rsync_command(host, remote_root, run_tag, dry_run)
    print("[sync]", shlex.join(command))
    if dry_run:
        return 0
    (PROJECT_ROOT / DEFAULT_RESULTS_RELATIVE / run_tag).mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=PROJECT_ROOT)


def preflight(host: str, remote_root: str) -> None:
    """Fail before billing work if SSH, GPU, workspace, or Git is unavailable."""
    remote = (
        "set -eu; "
        f"test -d {shlex.quote(remote_root)}/.git; "
        "command -v nvidia-smi >/dev/null; "
        f"test -x {shlex.quote(remote_root)}/../venv-pose/bin/python; "
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
    )
    subprocess.check_call(["ssh", "-o", "BatchMode=yes", host, remote], cwd=PROJECT_ROOT)


def run_remote(args: argparse.Namespace) -> int:
    """Run one remote candidate, always build a manifest, and then synchronize."""
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_relative = config_path.relative_to(PROJECT_ROOT)
    run_tag = args.run_tag or build_run_tag(config_path, args.candidate, args.gate)
    remote_run_relative = DEFAULT_RESULTS_RELATIVE / run_tag
    remote_run_dir = f"{args.remote_root}/{remote_run_relative}"
    python = f"{Path(args.remote_root).parent}/venv-pose/bin/python"
    runner_arguments = [
        python,
        "00_pose_pipeline_v2/src/run_human_prior.py",
        "--config", str(config_relative),
        "--candidate", args.candidate,
        "--gate", args.gate,
        "--run-dir", str(remote_run_relative),
    ]
    runner_command = shlex.join(runner_arguments)
    manifest_base = [
        python,
        "00_pose_pipeline_v2/src/build_artifact_manifest.py",
        "--run-dir", str(remote_run_relative),
        "--run-tag", run_tag,
        "--command", runner_command,
    ]
    remote_script = "\n".join(
        [
            "set +e",
            f"cd {shlex.quote(args.remote_root)} || exit 90",
            "git pull --ff-only || exit 91",
            f"mkdir -p {shlex.quote(str(remote_run_relative))}",
            f"{runner_command} > {shlex.quote(str(remote_run_relative / 'stdout.log'))} 2>&1",
            "run_status=$?",
            "if [ $run_status -eq 0 ]; then execution_status=completed; else execution_status=failed; fi",
            f"{shlex.join(manifest_base)} --execution-status \"$execution_status\" >> {shlex.quote(str(remote_run_relative / 'stdout.log'))} 2>&1",
            "manifest_status=$?",
            "if [ $run_status -ne 0 ]; then exit $run_status; fi",
            "exit $manifest_status",
        ]
    )
    print(f"[run] tag={run_tag}")
    print("[run]", runner_command)
    if args.dry_run:
        print(remote_script)
        return synchronize(args.host, args.remote_root, run_tag, dry_run=True)
    preflight(args.host, args.remote_root)
    remote_status = subprocess.call(["ssh", "-o", "BatchMode=yes", args.host, remote_script], cwd=PROJECT_ROOT)
    sync_status = synchronize(args.host, args.remote_root, run_tag, dry_run=False)
    if sync_status != 0:
        print(f"[sync] failed; retry with --run-tag {run_tag}")
        return sync_status
    return remote_status


def main() -> None:
    """Dispatch run or synchronization actions."""
    args = parse_args()
    if args.action == "run":
        raise SystemExit(run_remote(args))
    raise SystemExit(synchronize(args.host, args.remote_root, args.run_tag, args.dry_run))


if __name__ == "__main__":
    main()
