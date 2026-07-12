"""Validate a private SMPL asset with an EasyMocap CUDA smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    """Return the SHA-256 digest without exposing model contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Load the licensed model and run a zero-pose CUDA forward pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"SMPL asset not found: {args.model}. Follow docs/smpl_asset_setup.md."
        )
    if not args.regressor.is_file():
        raise FileNotFoundError(f"SMPL joint regressor not found: {args.regressor}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the remote SMPL validation")
    from easymocap.bodymodel.smpl import SMPLModel

    model = SMPLModel(
        model_path=str(args.model),
        regressor_path=str(args.regressor),
        device=torch.device("cuda"),
    )
    params = model.init_params(nFrames=1)
    tensor_params = {
        name: torch.as_tensor(value, dtype=torch.float32, device="cuda")
        for name, value in params.items()
        if isinstance(value, np.ndarray)
    }
    with torch.inference_mode():
        output = model(return_verts=False, return_tensor=True, **tensor_params)
    summary = {
        "asset_path": str(args.model),
        "asset_bytes": args.model.stat().st_size,
        "asset_sha256": sha256(args.model),
        "device": str(next(model.parameters()).device),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
        "reference_policy": "No Xsens signal is used for SMPL asset validation",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
