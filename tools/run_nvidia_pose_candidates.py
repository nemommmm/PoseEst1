#!/opt/anaconda3/envs/pose/bin/python
"""Run and evaluate the formal NVIDIA pose candidate matrix on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSE_PYTHON = Path("/workspace/venv-pose/bin/python")
FOUNDATION_PYTHON = Path(
    "/workspace/venv-foundation-stereo/bin/python"
)


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping: {path}")
    return payload


def run_logged(
    command: Sequence[str | Path],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one command, persist its output, and return a durable status."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    process = subprocess.run(
        [str(value) for value in command],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    ended = datetime.now(timezone.utc)
    rendered = " ".join(
        subprocess.list2cmdline([str(value)])
        for value in command
    )
    log_path.write_text(
        f"$ {rendered}\n{process.stdout}",
        encoding="utf-8",
    )
    return {
        "command": [str(value) for value in command],
        "log": str(log_path.relative_to(log_path.parents[2])),
        "return_code": int(process.returncode),
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "status": "completed" if process.returncode == 0 else "failed",
    }


def evaluation_command(
    *,
    candidate: Path,
    baseline: Path,
    config: str,
    output_dir: Path,
    dataset_spec: dict[str, Any],
) -> list[str | Path]:
    """Build one fixed-reference evaluation command."""
    reference = dataset_spec["reference"]
    return [
        POSE_PYTHON,
        "00_pose_pipeline_v2/src/evaluate_research_candidate.py",
        "--candidate",
        candidate,
        "--baseline",
        baseline,
        "--config",
        config,
        "--output-dir",
        output_dir,
        "--reference-kind",
        reference["kind"],
        "--reference-label",
        reference["label"],
        "--reference-offset-seconds",
        str(reference["offset_seconds"]),
        "--angle-names",
        *dataset_spec["angle_names"],
    ]


def run_sdk_probes(output_root: Path) -> dict[str, Any]:
    """Record access and runtime blockers without treating them as failures."""
    output_dir = output_root / "sdk_probes"
    record = run_logged(
        [
            POSE_PYTHON,
            "tools/probe_nvidia_pose_sdk_routes.py",
            "--output-dir",
            output_dir,
        ],
        output_dir / "probe.log",
    )
    record["route"] = "BodyPoseNet_and_Maxine_availability"
    return record


def run_bodypose3d(
    matrix: dict[str, Any],
    selection_path: Path,
    output_root: Path,
    repeats: int,
) -> list[dict[str, Any]]:
    """Run official BodyPose3DNet, adapt all routes, and evaluate them."""
    body_root = output_root / "bodypose3d"
    records: list[dict[str, Any]] = []
    runner = run_logged(
        [
            POSE_PYTHON,
            "tools/run_nvidia_bodypose3d_matrix.py",
            "--selection",
            selection_path,
            "--output-dir",
            body_root,
            "--repeats",
            str(repeats),
        ],
        body_root / "matrix_runner.log",
    )
    runner["route"] = "BodyPose3DNet_official_reference_app"
    records.append(runner)
    for mode in ("accuracy", "performance"):
        for dataset_name, dataset_spec in matrix["datasets"].items():
            config = dataset_spec["config"]
            raw_root = body_root / "raw" / dataset_name / mode
            left_json = raw_root / "left" / "pose.json"
            right_json = raw_root / "right" / "pose.json"
            baseline = (
                output_root / "baseline" / dataset_name
                / "skt_pose_optimized.npz"
            )
            adapted = body_root / "adapted" / dataset_name / mode
            mono_record = run_logged(
                [
                    POSE_PYTHON,
                    "00_pose_pipeline_v2/src/"
                    "adapt_nvidia_bodypose3d_monocular.py",
                    "--left-json",
                    left_json,
                    "--right-json",
                    right_json,
                    "--config",
                    config,
                    "--output-dir",
                    adapted,
                    "--candidate-prefix",
                    f"BodyPose3DNet-{mode}",
                    "--minimum-confidence",
                    str(
                        matrix["models"]["bodypose3dnet"][
                            "minimum_joint_confidence"
                        ]
                    ),
                ],
                adapted / "adapt_monocular.log",
            )
            mono_record.update(
                {
                    "route": "BodyPose3DNet_monocular",
                    "dataset": dataset_name,
                    "mode": mode,
                }
            )
            records.append(mono_record)
            stereo_path = adapted / "candidate_stereo.npz"
            stereo_record = run_logged(
                [
                    POSE_PYTHON,
                    "00_pose_pipeline_v2/src/"
                    "adapt_nvidia_bodypose3d_stereo.py",
                    "--left-json",
                    left_json,
                    "--right-json",
                    right_json,
                    "--config",
                    config,
                    "--output",
                    stereo_path,
                    "--candidate-name",
                    f"BodyPose3DNet-{mode}-stereo",
                ],
                adapted / "adapt_stereo.log",
            )
            stereo_record.update(
                {
                    "route": "BodyPose3DNet_stereo",
                    "dataset": dataset_name,
                    "mode": mode,
                }
            )
            records.append(stereo_record)
            candidate_paths = {
                "monocular_left": adapted / "candidate_monocular_left.npz",
                "monocular_right": adapted / "candidate_monocular_right.npz",
                "stereo": stereo_path,
            }
            for route, candidate_path in candidate_paths.items():
                evaluation_dir = (
                    body_root / "evaluation" / dataset_name / mode / route
                )
                evaluation_record = run_logged(
                    evaluation_command(
                        candidate=candidate_path,
                        baseline=baseline,
                        config=config,
                        output_dir=evaluation_dir,
                        dataset_spec=dataset_spec,
                    ),
                    evaluation_dir / "evaluation.log",
                )
                evaluation_record.update(
                    {
                        "route": f"BodyPose3DNet_{route}",
                        "dataset": dataset_name,
                        "mode": mode,
                    }
                )
                records.append(evaluation_record)
    return records


def core_joint_gate(candidate_path: Path) -> dict[str, Any]:
    """Check whether a dense route provides usable ergonomic core joints."""
    core_indices = np.asarray([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    with np.load(candidate_path, allow_pickle=False) as payload:
        points = np.asarray(payload["keypoints_3d"], dtype=np.float64)
    finite = np.isfinite(points[:, core_indices]).all(axis=2)
    finite_per_frame = np.sum(finite, axis=1)
    usable = finite_per_frame >= 6
    return {
        "frames": int(len(points)),
        "median_finite_core_joints": float(np.median(finite_per_frame)),
        "usable_frame_ratio": float(np.mean(usable)) if len(points) else 0.0,
        "passed": bool(len(points) and np.mean(usable) >= 0.5),
    }


def dense_command(
    *,
    variant: str,
    model_path: Path,
    repository: Path,
    left_video: str,
    right_video: str,
    baseline: Path,
    config: str,
    output_dir: Path,
    candidate_name: str,
    frames: int,
    valid_iters: int,
    repeats: int,
) -> list[str | Path]:
    """Build one official FoundationStereo route command."""
    return [
        FOUNDATION_PYTHON,
        "tools/run_foundation_stereo_matrix.py",
        "--variant",
        variant,
        "--repository",
        repository,
        "--model-path",
        model_path,
        "--left-video",
        left_video,
        "--right-video",
        right_video,
        "--baseline",
        baseline,
        "--config",
        config,
        "--output-dir",
        output_dir,
        "--candidate-name",
        candidate_name,
        "--max-frames",
        str(frames),
        "--scale",
        "0.5",
        "--valid-iters",
        str(valid_iters),
        "--warmup-frames",
        "10",
        "--repeats",
        str(repeats),
        "--diagnostic-frames",
        "12",
    ]


def run_dense_models(
    matrix: dict[str, Any],
    selection: dict[str, Any],
    output_root: Path,
    repeats: int,
) -> list[dict[str, Any]]:
    """Run 40-frame gates and full accepted dense-stereo candidates."""
    records: list[dict[str, Any]] = []
    variants = [
        {
            "key": "foundation_vitl_32",
            "variant": "foundation",
            "name": "FoundationStereo-ViT-L-32iter",
            "repository": Path(
                "/workspace/official_nvidia/FoundationStereo"
            ),
            "model": Path(
                "/workspace/official_nvidia/FoundationStereo/"
                "pretrained_models/23-51-11/model_best_bp2.pth"
            ),
            "iters": 32,
        },
        {
            "key": "fast_foundation_8",
            "variant": "fast",
            "name": "Fast-FoundationStereo-8iter",
            "repository": Path(
                "/workspace/official_nvidia/Fast-FoundationStereo"
            ),
            "model": Path(
                "/workspace/official_nvidia/Fast-FoundationStereo/"
                "weights/23-36-37/model_best_bp2_serialize.pth"
            ),
            "iters": 8,
        },
    ]
    environment = dict(os.environ)
    environment["XFORMERS_DISABLED"] = "1"
    feasibility_frames = int(matrix["evaluation"]["feasibility_frames"])
    for spec in variants:
        fast_fallback_required = False
        for dataset_name, dataset_spec in matrix["datasets"].items():
            selected = selection["accepted"][dataset_name]
            baseline = (
                output_root / "baseline" / dataset_name
                / "skt_pose_optimized.npz"
            )
            feasibility_dir = (
                output_root / "dense_stereo" / "feasibility"
                / dataset_name / spec["key"]
            )
            feasibility_record = run_logged(
                dense_command(
                    variant=spec["variant"],
                    model_path=spec["model"],
                    repository=spec["repository"],
                    left_video=selected["left"],
                    right_video=selected["right"],
                    baseline=baseline,
                    config=dataset_spec["config"],
                    output_dir=feasibility_dir,
                    candidate_name=spec["name"],
                    frames=feasibility_frames,
                    valid_iters=spec["iters"],
                    repeats=repeats,
                ),
                feasibility_dir / "runner.log",
                env=environment,
            )
            feasibility_record.update(
                {
                    "route": spec["name"],
                    "dataset": dataset_name,
                    "gate": "feasibility",
                }
            )
            if feasibility_record["return_code"] == 0:
                gate = core_joint_gate(
                    feasibility_dir / "candidate_v2.npz"
                )
                feasibility_record["core_joint_gate"] = gate
                timing_path = feasibility_dir / "timing_internal.json"
                timing = json.loads(timing_path.read_text(encoding="utf-8"))
                if (
                    spec["variant"] == "fast"
                    and not bool(timing["meets_12_5_fps"])
                ):
                    fast_fallback_required = True
            records.append(feasibility_record)
            if (
                feasibility_record["return_code"] != 0
                or not feasibility_record.get("core_joint_gate", {}).get(
                    "passed",
                    False,
                )
            ):
                continue
            full_frames = int(selected["synchronized_frames"])
            full_dir = (
                output_root / "dense_stereo" / "full"
                / dataset_name / spec["key"]
            )
            full_record = run_logged(
                dense_command(
                    variant=spec["variant"],
                    model_path=spec["model"],
                    repository=spec["repository"],
                    left_video=selected["left"],
                    right_video=selected["right"],
                    baseline=baseline,
                    config=dataset_spec["config"],
                    output_dir=full_dir,
                    candidate_name=spec["name"],
                    frames=full_frames,
                    valid_iters=spec["iters"],
                    repeats=repeats,
                ),
                full_dir / "runner.log",
                env=environment,
            )
            full_record.update(
                {
                    "route": spec["name"],
                    "dataset": dataset_name,
                    "gate": "full",
                }
            )
            records.append(full_record)
            if full_record["return_code"] == 0:
                evaluation_dir = full_dir / "evaluation"
                evaluation_record = run_logged(
                    evaluation_command(
                        candidate=full_dir / "candidate_v2.npz",
                        baseline=baseline,
                        config=dataset_spec["config"],
                        output_dir=evaluation_dir,
                        dataset_spec=dataset_spec,
                    ),
                    evaluation_dir / "evaluation.log",
                )
                evaluation_record.update(
                    {
                        "route": spec["name"],
                        "dataset": dataset_name,
                        "gate": "evaluation",
                    }
                )
                records.append(evaluation_record)
        if spec["variant"] == "fast" and fast_fallback_required:
            fallback = dict(spec)
            fallback["key"] = "fast_foundation_4"
            fallback["name"] = "Fast-FoundationStereo-4iter"
            fallback["iters"] = 4
            records.extend(
                run_dense_variant(
                    matrix,
                    selection,
                    output_root,
                    fallback,
                    repeats,
                    environment,
                )
            )
    return records


def run_dense_variant(
    matrix: dict[str, Any],
    selection: dict[str, Any],
    output_root: Path,
    spec: dict[str, Any],
    repeats: int,
    environment: dict[str, str],
) -> list[dict[str, Any]]:
    """Run one fixed Fast-FoundationStereo fallback over all datasets."""
    records: list[dict[str, Any]] = []
    for dataset_name, dataset_spec in matrix["datasets"].items():
        selected = selection["accepted"][dataset_name]
        baseline = (
            output_root / "baseline" / dataset_name
            / "skt_pose_optimized.npz"
        )
        full_dir = (
            output_root / "dense_stereo" / "full"
            / dataset_name / spec["key"]
        )
        record = run_logged(
            dense_command(
                variant=spec["variant"],
                model_path=spec["model"],
                repository=spec["repository"],
                left_video=selected["left"],
                right_video=selected["right"],
                baseline=baseline,
                config=dataset_spec["config"],
                output_dir=full_dir,
                candidate_name=spec["name"],
                frames=int(selected["synchronized_frames"]),
                valid_iters=spec["iters"],
                repeats=repeats,
            ),
            full_dir / "runner.log",
            env=environment,
        )
        record.update(
            {
                "route": spec["name"],
                "dataset": dataset_name,
                "gate": "full_realtime_fallback",
            }
        )
        records.append(record)
        if record["return_code"] == 0:
            evaluation_dir = full_dir / "evaluation"
            evaluation = run_logged(
                evaluation_command(
                    candidate=full_dir / "candidate_v2.npz",
                    baseline=baseline,
                    config=dataset_spec["config"],
                    output_dir=evaluation_dir,
                    dataset_spec=dataset_spec,
                ),
                evaluation_dir / "evaluation.log",
            )
            evaluation.update(
                {
                    "route": spec["name"],
                    "dataset": dataset_name,
                    "gate": "evaluation",
                }
            )
            records.append(evaluation)
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("00_pose_pipeline_v2/configs/nvidia_pose_matrix.yaml"),
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=["probes", "bodypose3d", "dense"],
        default=["probes", "bodypose3d", "dense"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run requested formal routes while retaining cell-level failures."""
    args = parse_args(argv)
    matrix = load_yaml(project_path(args.matrix))
    selection_path = project_path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "accepted":
        raise ValueError("Formal matrix requires an accepted proxy selection")
    output_root = project_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if "probes" in args.routes:
        records.append(run_sdk_probes(output_root))
    if "bodypose3d" in args.routes:
        records.extend(
            run_bodypose3d(
                matrix,
                selection_path,
                output_root,
                args.repeats,
            )
        )
    if "dense" in args.routes:
        records.extend(
            run_dense_models(
                matrix,
                selection,
                output_root,
                args.repeats,
            )
        )
    summary = {
        "schema_version": "nvidia_pose_candidate_matrix_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": str(selection_path),
        "routes": args.routes,
        "records": records,
    }
    output = output_root / "candidate_matrix_status.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
