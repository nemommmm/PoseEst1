#!/opt/anaconda3/envs/pose/bin/python
"""Run admissible remote gates and synchronize standard result bundles."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT_DEFAULT = "/workspace/PoseEst1"
RESULTS_RELATIVE = Path("00_pose_pipeline_v2/runs/human_prior_fusion")
REJECTED_CANDIDATES = {
    "kinematic": (
        "The calibrated kinematic adapter was rejected at the Fanbo7 "
        "feasibility gate and reverted. See experiment_log.md."
    )
}


def parse_args() -> argparse.Namespace:
    """Parse run, synchronization, and verification commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--candidate", choices=["kinematic", "smpl"], required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--gate", choices=["feasibility", "short", "full"], required=True)
    run.add_argument("--sync-profile", choices=["standard"], default="standard")
    run.add_argument("--run-tag")
    sync = subparsers.add_parser("sync")
    sync.add_argument("--run-tag", required=True)
    sync.add_argument("--sync-profile", choices=["standard"], default="standard")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-tag", required=True)
    for child in (run, sync):
        child.add_argument("--host", default="poseest1-runpod")
        child.add_argument("--remote-root", default=REMOTE_ROOT_DEFAULT)
    return parser.parse_args()


def dataset_name(config: Path) -> str:
    """Read the stable dataset name from one YAML configuration."""
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return str(data.get("dataset", {}).get("name", config.stem))


def run_tag(config: Path, candidate: str, gate: str) -> str:
    """Build the standard descriptive run tag."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    return f"{dataset_name(config)}__{candidate}__{gate}__{timestamp}__{commit}"


def preflight(host: str, remote_root: str) -> None:
    """Check non-interactive SSH, the repository, Python, and one CUDA GPU."""
    command = (
        "set -eu; "
        f"test -d {shlex.quote(remote_root)}/.git; "
        f"test -x {shlex.quote(str(Path(remote_root).parent / 'venv-pose/bin/python'))}; "
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
    )
    subprocess.check_call(
        ["ssh", "-o", "BatchMode=yes", host, command], cwd=PROJECT_ROOT
    )


def rsync_command(host: str, remote_root: str, tag: str) -> list[str]:
    """Build a resumable, checksum-based, sensitive-asset-excluding sync."""
    local = PROJECT_ROOT / RESULTS_RELATIVE / tag
    return [
        "rsync", "-avP", "--partial", "--checksum",
        "--exclude", "*.avi", "--exclude", "*.mkv",
        "--exclude", "*.pt", "--exclude", "*.pth",
        "--exclude", "*.engine", "--exclude", "*.onnx",
        "--exclude", "*.pkl",
        f"{host}:{remote_root}/{RESULTS_RELATIVE}/{tag}/",
        f"{local}/",
    ]


def synchronize(host: str, remote_root: str, tag: str) -> None:
    """Retry interrupted synchronization and verify the resulting local bundle."""
    local = PROJECT_ROOT / RESULTS_RELATIVE / tag
    local.mkdir(parents=True, exist_ok=True)
    command = rsync_command(host, remote_root, tag)
    for attempt in range(1, 4):
        print(f"[sync {attempt}/3]", shlex.join(command))
        if subprocess.call(command, cwd=PROJECT_ROOT) == 0:
            break
        if attempt == 3:
            raise RuntimeError(f"rsync failed; retry with sync --run-tag {tag}")
        time.sleep(2 ** (attempt - 1))
    verify = subprocess.run(
        [
            "/opt/anaconda3/envs/pose/bin/python",
            "tools/artifact_bundle.py",
            "verify",
            "--run-dir",
            str(local),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    print(verify.stdout.strip())
    if verify.returncode != 0:
        if verify.stderr:
            print(verify.stderr.strip())
        raise RuntimeError("downloaded artifact bundle failed checksum verification")


def run_remote(args: argparse.Namespace) -> None:
    """Run one non-rejected gate, always manifest it, and synchronize it."""
    if args.candidate in REJECTED_CANDIDATES:
        raise RuntimeError(REJECTED_CANDIDATES[args.candidate])
    config = args.config.resolve() if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    relative_config = config.relative_to(PROJECT_ROOT)
    tag = args.run_tag or run_tag(config, args.candidate, args.gate)
    relative_run = RESULTS_RELATIVE / tag
    python = str(Path(args.remote_root).parent / "venv-pose/bin/python")
    runner = [
        python,
        "00_pose_pipeline_v2/src/run_smpl_candidate.py",
        "--config", str(relative_config),
        "--gate", args.gate,
        "--run-dir", str(relative_run),
    ]
    runner_text = shlex.join(runner)
    manifest = [
        python,
        "tools/artifact_bundle.py",
        "build",
        "--run-dir", str(relative_run),
        "--run-tag", tag,
        "--command", runner_text,
    ]
    script = "\n".join(
        [
            "set +e",
            f"cd {shlex.quote(args.remote_root)} || exit 90",
            "git pull --ff-only || exit 91",
            f"mkdir -p {shlex.quote(str(relative_run))}",
            f"{runner_text} > {shlex.quote(str(relative_run / 'stdout.log'))} 2>&1",
            "run_status=$?",
            "if [ $run_status -eq 0 ]; then state=completed; else state=failed; fi",
            f"{shlex.join(manifest)} --execution-status \"$state\" >/dev/null 2>&1",
            "manifest_status=$?",
            "if [ $run_status -ne 0 ]; then exit $run_status; fi",
            "exit $manifest_status",
        ]
    )
    print(f"[run] {tag}")
    preflight(args.host, args.remote_root)
    status = subprocess.call(
        ["ssh", "-o", "BatchMode=yes", args.host, script], cwd=PROJECT_ROOT
    )
    synchronize(args.host, args.remote_root, tag)
    if status != 0:
        raise SystemExit(status)


def main() -> None:
    """Dispatch remote experiment operations."""
    args = parse_args()
    if args.action == "run":
        run_remote(args)
    elif args.action == "sync":
        synchronize(args.host, args.remote_root, args.run_tag)
    else:
        local = PROJECT_ROOT / RESULTS_RELATIVE / args.run_tag
        completed = subprocess.run(
            [
                "/opt/anaconda3/envs/pose/bin/python",
                "tools/artifact_bundle.py",
                "verify",
                "--run-dir",
                str(local),
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        print(completed.stdout)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
