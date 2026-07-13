#!/opt/anaconda3/envs/pose/bin/python
"""Record and, when possible, validate the official SMPL asset gate."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch


DEFAULT_ASSET = Path("/workspace/model_assets/smpl/SMPL_NEUTRAL.pkl")
DEFAULT_REGRESSOR = Path(
    "/workspace/external/EasyMocap/data/smplx/J_regressor_body25.npy"
)


def parse_args() -> argparse.Namespace:
    """Parse the asset gate arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", choices=["feasibility", "short", "full"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--regressor", type=Path, default=DEFAULT_REGRESSOR)
    return parser.parse_args()


def gpu_metadata() -> dict[str, Any]:
    """Collect reproducibility metadata from Torch and nvidia-smi."""
    metadata: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        name, uuid, memory_mb, driver = [value.strip() for value in line.split(",", 3)]
        metadata.update(
            {
                "gpu_name": name,
                "gpu_uuid": uuid,
                "gpu_memory_mb": int(memory_mb),
                "driver": driver,
            }
        )
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        metadata["gpu_name"] = "unavailable"
    return metadata


def main() -> None:
    """Write a blocked bundle or execute the existing CUDA asset validator."""
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, args.run_dir / "resolved_config.yaml")
    gpu = gpu_metadata()
    (args.run_dir / "gpu_metadata.json").write_text(
        json.dumps(gpu, indent=2), encoding="utf-8"
    )
    metrics: dict[str, Any] = {
        "candidate": "smpl",
        "gate": args.gate,
        "asset_path": str(args.asset),
        "reference_policy": (
            "Xsens-derived reference is an external comparison system only"
        ),
    }
    if not args.asset.is_file():
        metrics["scientific_status"] = "asset_blocked"
        metrics["next_action"] = "Follow docs/smpl_asset_setup.md"
    else:
        if not args.regressor.is_file():
            raise FileNotFoundError(args.regressor)
        validator = Path(__file__).with_name("validate_smpl_asset.py")
        completed = subprocess.run(
            [
                str(Path("/workspace/venv-pose/bin/python")),
                str(validator),
                "--model",
                str(args.asset),
                "--regressor",
                str(args.regressor),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        validation = json.loads(completed.stdout)
        (args.run_dir / "smpl_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        metrics["scientific_status"] = "asset_ready"
        metrics["next_action"] = (
            "Implement and snapshot the EasyMocap fitting adapter before the "
            "40-frame geometry gate"
        )
    (args.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
