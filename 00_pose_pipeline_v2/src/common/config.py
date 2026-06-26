"""Configuration helpers for the standalone pose pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML pipeline config."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    return config


def resolve_path(value: str | Path | None, *, must_exist: bool = False) -> Path | None:
    """Resolve absolute or project-root-relative paths."""
    if value is None or value == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def get_run_dir(config: dict[str, Any]) -> Path:
    """Return and create the configured run directory."""
    outputs = config.get("outputs", {})
    runs_dir = resolve_path(outputs.get("runs_dir", "00_pose_pipeline/runs"))
    dataset_name = config.get("dataset", {}).get("name", "dataset")
    tag = outputs.get("run_tag") or dataset_name
    run_dir = runs_dir / str(tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a config section as a dictionary."""
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}
