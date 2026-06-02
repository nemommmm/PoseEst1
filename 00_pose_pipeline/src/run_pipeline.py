#!/opt/anaconda3/envs/pose/bin/python
"""Standalone end-to-end pose pipeline runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.config import get_run_dir, load_config, resolve_path, section
from common.metrics import jsonable
from stereo_loader import validate_stereo_inputs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", default="validate,offset,angle,motion",
                        help="Comma-separated stages: validate,skt,offset,angle,motion,segment,scatter,video or all.")
    return parser.parse_args()


def parse_stages(raw: str) -> list[str]:
    """Parse stage list."""
    stages = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if "all" in stages:
        return ["validate", "skt", "offset", "angle", "motion", "segment", "scatter", "video"]
    return stages


def validate_config(config: dict, run_dir: Path) -> Path:
    """Validate configured dataset and write summary."""
    summary = validate_stereo_inputs(config)
    for label, raw_path in [
        ("camera_params", section(config, "calibration").get("camera_params")),
        ("skt_existing_npz", section(config, "skt").get("existing_npz") if section(config, "skt").get("use_existing_npz") else None),
        ("xsens_mvnx", section(config, "references").get("xsens_mvnx")),
        ("xsens_fair_angles", section(config, "references").get("xsens_fair_angles")),
        ("fastsam_trc", section(config, "references").get("fastsam_trc")),
        ("merge_trc", section(config, "references").get("merge_trc")),
    ]:
        path = resolve_path(raw_path, must_exist=False)
        summary[f"{label}_path"] = None if path is None else str(path)
        summary[f"{label}_exists"] = bool(path and path.exists())
    out_path = run_dir / "validate_summary.json"
    out_path.write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[validate] saved {out_path}")
    return out_path


def main() -> None:
    """Run selected stages."""
    args = parse_args()
    config = load_config(args.config)
    run_dir = get_run_dir(config)
    stages = parse_stages(args.stages)
    print(f"[run] {run_dir}")
    for stage in stages:
        if stage == "validate":
            validate_config(config, run_dir)
        elif stage == "skt":
            from skt_inference import run_skt

            run_skt(config, run_dir)
        elif stage == "offset":
            from estimate_offset import estimate_offset

            estimate_offset(config, run_dir)
        elif stage == "angle":
            from eval_angles import evaluate_angles

            evaluate_angles(config, run_dir)
        elif stage == "motion":
            from eval_motion_delta import evaluate_motion_delta

            evaluate_motion_delta(config, run_dir)
        elif stage == "segment":
            from eval_segments import evaluate_segments

            evaluate_segments(config, run_dir)
        elif stage == "scatter":
            from plot_scatter import render_scatter

            render_scatter(config, run_dir)
        elif stage == "video":
            from render_comparison_video import render_video

            render_video(config, run_dir)
        else:
            raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
