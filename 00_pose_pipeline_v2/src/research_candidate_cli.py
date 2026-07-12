"""Command-line utilities for canonical research candidate results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.research_candidate import adapt_skt_npz, compute_bone_statistics


def inspect_result(path: Path) -> None:
    """Print a compact validation summary without writing extra files."""
    with np.load(path, allow_pickle=False) as payload:
        required = {"schema_version", "candidate_name", "timestamps", "keypoints_3d", "angles"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"missing required arrays: {sorted(missing)}")
        keypoints = np.asarray(payload["keypoints_3d"], dtype=np.float64)
        summary = {
            "schema_version": str(payload["schema_version"]),
            "candidate_name": str(payload["candidate_name"]),
            "frames": len(keypoints),
            "finite_keypoint_ratio": float(np.isfinite(keypoints).all(axis=2).mean()),
            "bone_statistics_cm": compute_bone_statistics(keypoints),
            "metadata": json.loads(str(payload["metadata_json"])),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    """Run the requested candidate result utility."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapt = subparsers.add_parser("adapt-skt")
    adapt.add_argument("source", type=Path)
    adapt.add_argument("destination", type=Path)
    adapt.add_argument("--name", default="YOLOv8m-SKT")
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "adapt-skt":
        print(adapt_skt_npz(args.source, args.destination, args.name))
    else:
        inspect_result(args.path)


if __name__ == "__main__":
    main()
