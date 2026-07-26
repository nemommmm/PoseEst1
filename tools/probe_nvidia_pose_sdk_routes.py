#!/opt/anaconda3/envs/pose/bin/python
"""Record reproducible availability checks for NVIDIA pose SDK routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import onnx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BODYPOSENET = Path(
    "/workspace/official_nvidia/bodyposenet/model.onnx"
)
DEFAULT_TAO_APP = Path(
    "/workspace/official_nvidia/deepstream_tao_apps_ds64/"
    "apps/tao_others/deepstream-bodypose2d-app"
)
DEFAULT_MAXINE_ROOT = Path("/usr/local/NVIDIA_AR_SDK")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one command and capture combined text output."""
    return subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def git_commit(repository: Path) -> str | None:
    """Return a repository commit when available."""
    result = run(["git", "-C", repository, "rev-parse", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_onnx(path: Path) -> dict[str, Any]:
    """Describe the official BodyPoseNet ONNX interface."""
    model = onnx.load(str(path))

    def tensor_shape(value: Any) -> list[int | str | None]:
        dimensions: list[int | str | None] = []
        for dimension in value.type.tensor_type.shape.dim:
            dimensions.append(
                int(dimension.dim_value)
                if dimension.dim_value
                else str(dimension.dim_param) or None
            )
        return dimensions

    return {
        "sha256": sha256_file(path),
        "inputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in model.graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in model.graph.output
        ],
    }


def probe_bodyposenet(
    model_path: Path,
    app_root: Path,
) -> dict[str, Any]:
    """Test the last official PAF application against installed DeepStream."""
    deepstream_root = Path("/opt/nvidia/deepstream/deepstream-8.0")
    required_header = (
        deepstream_root
        / "sources/includes/cvcore_headers/cv/bodypose2d/BodyPose2D.h"
    )
    required_library = (
        deepstream_root / "lib/cvcore_libs/libnvcv_bodypose2d.so"
    )
    build_output = ""
    build_return_code: int | None = None
    if app_root.is_dir():
        run(["make", "clean"], cwd=app_root)
        build = run(["make"], cwd=app_root)
        build_output = build.stdout
        build_return_code = build.returncode
    available = bool(
        model_path.is_file()
        and required_header.is_file()
        and required_library.is_file()
        and build_return_code == 0
    )
    return {
        "candidate": "NVIDIA BodyPoseNet 2D + calibrated SKT",
        "status": "available_pending_run" if available else "runtime_blocked",
        "model": (
            inspect_onnx(model_path)
            if model_path.is_file()
            else {"missing": str(model_path)}
        ),
        "model_semantics": {
            "joints": 18,
            "heatmap_channels": 19,
            "part_affinity_field_channels": 38,
            "required_postprocess": (
                "NVIDIA BodyPose2D PAF NMS and bipartite matching"
            ),
        },
        "official_application": {
            "path": str(app_root),
            "repository_commit": git_commit(app_root.parents[2])
            if app_root.is_dir()
            else None,
            "documented_for": "DeepStream 6.2 and later at that commit",
            "build_return_code": build_return_code,
            "build_log": build_output,
        },
        "installed_deepstream": "8.0",
        "missing_runtime_components": [
            str(path)
            for path in (required_header, required_library)
            if not path.is_file()
        ],
        "classification": (
            "Runtime compatibility blocker, not a model-accuracy failure."
        ),
        "official_sources": [
            "https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/"
            "bodyposenet/deployable_onnx_v1.0.1",
            "https://docs.nvidia.com/tao/tao-toolkit/latest/text/"
            "ds_tao/deepstream_tao_integration.html",
        ],
    }


def probe_maxine(maxine_root: Path) -> dict[str, Any]:
    """Check Maxine SDK installation, access, and documented GPU coverage."""
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    ).stdout.strip()
    feature_candidates = list(maxine_root.glob("**/*body*pose*"))
    sdk_installed = maxine_root.is_dir() and bool(feature_candidates)
    qualified_installer_identifiers = {
        "a40",
        "l40",
        "l4",
        "a30",
        "b200",
        "a2",
        "h100",
        "a10",
        "t4",
        "b100",
        "a16",
        "a100",
        "b40",
    }
    gpu_identifier = gpu.split(",", maxsplit=1)[0].lower().replace("nvidia", "")
    installer_match = any(
        identifier in gpu_identifier
        for identifier in qualified_installer_identifiers
    )
    available = sdk_installed and installer_match
    return {
        "candidate": "NVIDIA Maxine 3D Body Pose HQ/HP",
        "status": (
            "available_pending_run"
            if available
            else "access_or_platform_blocked"
        ),
        "gpu": gpu,
        "sdk_root": str(maxine_root),
        "sdk_feature_installed": sdk_installed,
        "documented_installer_gpu_match": installer_match,
        "note": (
            "The RTX A6000 is Ampere-compatible in the broad architecture "
            "statement, but it is absent from the current Linux feature "
            "installer's explicit supported-GPU identifiers. The NGC AR SDK "
            "resource also requires a subscription. No SDK package or NGC "
            "credential was supplied, so the model was not run and this is "
            "not classified as an accuracy failure."
        ),
        "official_sources": [
            "https://docs.nvidia.com/maxine/ar/latest/LinuxARSDK/"
            "InstalltheARSDK.html",
            "https://catalog.ngc.nvidia.com/orgs/nvidia/maxine/resources/"
            "maxine_linux_ar_sdk",
            "https://catalog.ngc.nvidia.com/orgs/nvidia/maxine/collections/"
            "nvarbodyposeestimation",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodyposenet-model", type=Path, default=DEFAULT_BODYPOSENET)
    parser.add_argument("--bodyposenet-app", type=Path, default=DEFAULT_TAO_APP)
    parser.add_argument("--maxine-root", type=Path, default=DEFAULT_MAXINE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run both probes and save one durable evidence bundle."""
    args = parse_args(argv)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir.is_absolute()
        else (PROJECT_ROOT / args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "nvidia_pose_sdk_probe_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(PROJECT_ROOT),
        "platform": platform.platform(),
        "bodyposenet": probe_bodyposenet(
            args.bodyposenet_model.expanduser().resolve(),
            args.bodyposenet_app.expanduser().resolve(),
        ),
        "maxine": probe_maxine(args.maxine_root.expanduser().resolve()),
    }
    output = output_dir / "sdk_route_status.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
