"""Run full SKT replacement ablations for selected YOLO pose models."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.config import get_run_dir, load_config, section
from common.metrics import jsonable
from eval_filter_ablation import evaluate_filter_ablation
from skt_inference import run_skt


DEFAULT_VARIANTS = [
    {
        "name": "yolo11m_full_skt",
        "model_path": "00_pose_pipeline_v2/model_weights/yolo11m-pose.pt",
    },
    {
        "name": "yolo11l_full_skt",
        "model_path": "00_pose_pipeline_v2/model_weights/yolo11l-pose.pt",
    },
]


def _load_best_filter_row(run_dir: Path) -> dict[str, object] | None:
    """Load the best row with guardrails against overly sparse valid pairs."""
    csv_path = run_dir / "filter_ablation" / "summary.csv"
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row["_mae"] = float(row["mae_deg"])
                row["_rms"] = float(row["target_angular_acc_rms_deg_s2"])
                row["_valid_ratio"] = float(row["valid_ratio"])
            except (TypeError, ValueError):
                continue
            rows.append(row)
    if not rows:
        return None
    eligible = [row for row in rows if row["_valid_ratio"] >= 0.30]
    pool = eligible or rows
    return min(pool, key=lambda row: (row["_mae"], row["_rms"]))


def make_variant_config(base_config: dict, variant: dict[str, object]) -> dict:
    """Create an isolated run config for one model replacement variant."""
    config = deepcopy(base_config)
    skt = config.setdefault("skt", {})
    skt["use_existing_npz"] = False
    skt["existing_npz"] = None
    skt["model_path"] = str(variant["model_path"])
    skt["output_npz"] = "skt_pose_optimized.npz"
    outputs = config.setdefault("outputs", {})
    outputs["run_tag"] = f"assar2026_fanbo7_a257_{variant['name']}"
    # Keep offset/TRC settings unchanged so the only intended variable is detector model.
    return config


def run_model_replacement_ablation(config: dict) -> Path:
    """Run SKT + filter ablation for each configured replacement model."""
    variants = section(config, "model_replacement_ablation").get("variants") or DEFAULT_VARIANTS
    summary_rows = []
    for variant in variants:
        name = str(variant["name"])
        variant_config = make_variant_config(config, variant)
        run_dir = get_run_dir(variant_config)
        print(f"[model_replace] {name}: run_dir={run_dir}")
        run_skt(variant_config, run_dir)
        evaluate_filter_ablation(variant_config, run_dir)
        best_row = _load_best_filter_row(run_dir)
        summary_rows.append({
            "variant": name,
            "model_path": variant["model_path"],
            "run_dir": str(run_dir),
            "best_filter_variant": None if best_row is None else best_row["target"],
            "mae_deg": None if best_row is None else float(best_row["mae_deg"]),
            "median_abs_error_deg": None if best_row is None else float(best_row["median_abs_error_deg"]),
            "rmse_deg": None if best_row is None else float(best_row["rmse_deg"]),
            "angular_acc_rms_deg_s2": None if best_row is None else float(best_row["target_angular_acc_rms_deg_s2"]),
            "jump_count": None if best_row is None else int(float(best_row["target_jump_count"])),
            "valid_ratio": None if best_row is None else float(best_row["valid_ratio"]),
        })

    base_run_dir = get_run_dir(config)
    out_dir = base_run_dir / "model_replacement_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "variant", "model_path", "run_dir", "best_filter_variant", "mae_deg",
        "median_abs_error_deg", "rmse_deg", "angular_acc_rms_deg_s2",
        "jump_count", "valid_ratio",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(jsonable({"rows": summary_rows}), indent=2), encoding="utf-8")
    print(f"[model_replace] saved {csv_path}")
    return csv_path


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_model_replacement_ablation(config)


if __name__ == "__main__":
    main()
