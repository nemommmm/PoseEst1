#!/opt/anaconda3/envs/pose/bin/python
"""Prepare, bootstrap, run, and synchronize the NVIDIA GPU pose matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = Path(
    "00_pose_pipeline_v2/configs/nvidia_pose_matrix.yaml"
)
REMOTE_ROOT = "/workspace/PoseEst1"


def project_path(value: str | Path) -> Path:
    """Resolve a path against the project root."""
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping: {path}")
    return payload


def command(
    args: Sequence[str],
    *,
    cwd: Path = PROJECT_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run and echo one reproducible local command."""
    print("+", shlex.join([str(value) for value in args]))
    return subprocess.run(
        [str(value) for value in args],
        cwd=cwd,
        check=check,
        text=True,
    )


def ssh(host: str, script: str, *, check: bool = True) -> int:
    """Run a non-interactive remote shell script."""
    print(f"+ ssh {host} {shlex.quote(script)}")
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, script],
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
    ).returncode


def current_commit() -> str:
    """Return the full current Git commit."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def prepare_inputs(args: argparse.Namespace) -> None:
    """Create requested local compressed-video candidates."""
    script = PROJECT_ROOT / "tools" / "prepare_pose_proxy_videos.py"
    call = [
        "/opt/anaconda3/envs/pose/bin/python",
        str(script),
        "--matrix",
        str(project_path(args.matrix)),
    ]
    if args.datasets:
        call.extend(["--datasets", *args.datasets])
    if args.crfs is not None:
        call.extend(["--crfs", *[str(value) for value in args.crfs]])
    if args.include_lossless:
        call.append("--include-lossless")
    command(call)


def build_bundle(destination: Path) -> None:
    """Create a self-contained bundle for the current main branch."""
    command(
        [
            "git",
            "bundle",
            "create",
            str(destination),
            "main",
        ]
    )


def bootstrap(args: argparse.Namespace) -> None:
    """Update the remote repository from a bundle and build the Python env."""
    commit = current_commit()
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "PoseEst1.bundle"
        build_bundle(bundle)
        command(
            [
                "rsync",
                "-rtvP",
                "--partial",
                "--no-owner",
                "--no-group",
                str(bundle),
                f"{args.host}:/workspace/PoseEst1.bundle",
            ]
        )
    script = "\n".join(
        [
            "set -eu",
            "if [ ! -d /workspace/PoseEst1/.git ]; then",
            "  git clone /workspace/PoseEst1.bundle /workspace/PoseEst1",
            "fi",
            f"cd {shlex.quote(args.remote_root)}",
            "git fetch /workspace/PoseEst1.bundle main",
            "git checkout main",
            "git merge --ff-only FETCH_HEAD",
            f"test \"$(git rev-parse HEAD)\" = {shlex.quote(commit)}",
            (
                "VENV_PATH=/workspace/venv-pose BASE_PYTHON=python "
                "bash tools/setup_remote_env.sh"
            ),
            (
                "/workspace/venv-pose/bin/python -m unittest "
                "00_pose_pipeline_v2.tests.test_nvidia_pose_matrix"
            ),
        ]
    )
    ssh(args.host, script)


def upload_inputs(args: argparse.Namespace) -> None:
    """Upload selected proxy inputs and small comparison assets."""
    selection_path = project_path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    paths = [project_path(value) for value in selection["files"]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(PROJECT_ROOT)
        remote_parent = str(
            Path(args.remote_root) / relative.parent
        )
        ssh(args.host, f"mkdir -p {shlex.quote(remote_parent)}")
        command(
            [
                "rsync",
                "-rtvP",
                "--partial",
                "--checksum",
                "--no-owner",
                "--no-group",
                str(path),
                f"{args.host}:{remote_parent}/",
            ]
        )


def run_baselines(args: argparse.Namespace) -> None:
    """Run the deterministic PyTorch control on every accepted proxy."""
    matrix_path = project_path(args.matrix)
    matrix = load_yaml(matrix_path)
    selection = json.loads(
        project_path(args.selection).read_text(encoding="utf-8")
    )
    accepted = selection["accepted"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit7 = current_commit()[:7]
    tag = args.run_tag or f"nvidia_pose_matrix__{timestamp}__{commit7}"
    commands: list[str] = []
    for dataset_name, spec in matrix["datasets"].items():
        proxy = accepted[dataset_name]
        output = (
            Path(matrix["output_root"])
            / tag
            / "baseline"
            / dataset_name
        )
        config = spec["config"]
        model = "01_stereo_triangulation/src/yolov8m-pose.pt"
        commands.append(
            shlex.join(
                [
                    "/workspace/venv-pose/bin/python",
                    "00_pose_pipeline_v2/src/benchmark_realtime.py",
                    "--config",
                    config,
                    "--model",
                    model,
                    "--left-video",
                    proxy["left"],
                    "--right-video",
                    proxy["right"],
                    "--input-upright",
                    "--max-frames",
                    str(proxy["synchronized_frames"]),
                    "--warmup-frames",
                    "10",
                    "--run-dir",
                    str(output),
                    "--output-json",
                    str(output / "benchmark.json"),
                ]
            )
        )
    remote_script = "\n".join(
        [
            "set -eu",
            f"cd {shlex.quote(args.remote_root)}",
            *commands,
        ]
    )
    try:
        ssh(args.host, remote_script)
    finally:
        synchronize(args.host, args.remote_root, tag)


def synchronize(host: str, remote_root: str, tag: str) -> None:
    """Download one result tree while excluding licensed or bulky assets."""
    relative = Path("00_pose_pipeline_v2/runs/nvidia_pose_matrix") / tag
    local = PROJECT_ROOT / relative
    local.mkdir(parents=True, exist_ok=True)
    command(
        [
            "rsync",
            "-rtvP",
            "--partial",
            "--checksum",
            "--exclude",
            "*.avi",
            "--exclude",
            "*.mp4",
            "--exclude",
            "*.mkv",
            "--exclude",
            "*.pt",
            "--exclude",
            "*.pth",
            "--exclude",
            "*.onnx",
            "--exclude",
            "*.engine",
            "--exclude",
            "*.pkl",
            "--exclude",
            "*ngc*key*",
            f"{host}:{remote_root}/{relative}/",
            f"{local}/",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare-inputs")
    prepare.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    prepare.add_argument("--datasets", nargs="*", default=[])
    prepare.add_argument("--crfs", nargs="*", type=int)
    prepare.add_argument("--include-lossless", action="store_true")

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--host", default="poseest1-runpod")
    bootstrap_parser.add_argument("--remote-root", default=REMOTE_ROOT)

    upload = subparsers.add_parser("upload-inputs")
    upload.add_argument("--host", default="poseest1-runpod")
    upload.add_argument("--remote-root", default=REMOTE_ROOT)
    upload.add_argument("--selection", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    run.add_argument("--host", default="poseest1-runpod")
    run.add_argument("--remote-root", default=REMOTE_ROOT)
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--run-tag")
    run.add_argument("--matrix-name", choices=["formal"], default="formal")

    sync = subparsers.add_parser("sync")
    sync.add_argument("--host", default="poseest1-runpod")
    sync.add_argument("--remote-root", default=REMOTE_ROOT)
    sync.add_argument("--run-tag", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the selected workflow."""
    args = parse_args(argv)
    if args.action == "prepare-inputs":
        prepare_inputs(args)
    elif args.action == "bootstrap":
        bootstrap(args)
    elif args.action == "upload-inputs":
        upload_inputs(args)
    elif args.action == "run":
        run_baselines(args)
    else:
        synchronize(args.host, args.remote_root, args.run_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
