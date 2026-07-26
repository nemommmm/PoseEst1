#!/opt/anaconda3/envs/pose/bin/python
"""Render a local H.264 left-view plus 3D preview for one candidate."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = PROJECT_ROOT / "00_pose_pipeline_v2" / "src"
sys.path.insert(0, str(PIPELINE_SRC))

from common.angles import compute_angle_sequence  # noqa: E402
from common.config import load_config, resolve_path, section  # noqa: E402
from render_comparison_video import (  # noqa: E402
    canonicalize,
    render_skeleton_panel,
)
from stereo_loader import (  # noqa: E402
    StereoFrameReader,
    build_synced_timeline,
)


def project_path(value: str | Path) -> Path:
    """Resolve one path relative to the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_candidate(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Load candidate name, 3D points, and raw points."""
    with np.load(path, allow_pickle=False) as payload:
        name = str(np.asarray(payload["candidate_name"]).item())
        points = np.asarray(payload["keypoints_3d"], dtype=np.float64)
        raw = np.asarray(
            payload["keypoints_3d_raw"]
            if "keypoints_3d_raw" in payload.files
            else points,
            dtype=np.float64,
        )
    return name, points, raw


def evaluation_errors(path: Path | None, count: int) -> np.ndarray:
    """Return per-frame mean absolute angle difference from an evaluation CSV."""
    errors = np.full(count, np.nan, dtype=np.float64)
    if path is None or not path.is_file():
        return errors
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            differences = []
            for key, candidate in row.items():
                if not key.startswith("Candidate_") or not key.endswith("_deg"):
                    continue
                reference_key = key.replace("Candidate_", "Reference_", 1)
                reference = row.get(reference_key, "")
                try:
                    differences.append(abs(float(candidate) - float(reference)))
                except (TypeError, ValueError):
                    continue
            values.append(float(np.mean(differences)) if differences else np.nan)
    errors[: min(count, len(values))] = values[:count]
    return errors


def representative_indices(
    points: np.ndarray,
    raw: np.ndarray,
    errors: np.ndarray,
    count: int = 12,
) -> list[int]:
    """Choose best, worst, correction, missing, jump, and uniform frames."""
    frame_count = len(points)
    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is not None and 0 <= index < frame_count and index not in selected:
            selected.append(index)

    finite_error = np.flatnonzero(np.isfinite(errors))
    if finite_error.size:
        add(int(finite_error[np.argmin(errors[finite_error])]))
        add(int(finite_error[np.argmax(errors[finite_error])]))
    correction = np.linalg.norm(points - raw, axis=2)
    correction[~np.isfinite(correction)] = np.nan
    per_frame_correction = np.nanmax(correction, axis=1)
    if np.any(np.isfinite(per_frame_correction)):
        add(int(np.nanargmax(per_frame_correction)))
    missing = np.sum(~np.isfinite(points).all(axis=2), axis=1)
    add(int(np.argmax(missing)))
    angles = compute_angle_sequence(
        points,
        ["LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow"],
    )
    jump = np.zeros(frame_count, dtype=np.float64)
    for values in angles.values():
        difference = np.abs(np.diff(values, prepend=np.nan))
        jump = np.fmax(jump, np.nan_to_num(difference, nan=0.0))
    add(int(np.argmax(jump)))
    for index in np.linspace(0, max(frame_count - 1, 0), count).round().astype(int):
        add(int(index))
        if len(selected) >= count:
            break
    return sorted(selected[:count])


def render(
    candidate_path: Path,
    config_path: Path,
    selection_path: Path,
    dataset_name: str,
    output_dir: Path,
    evaluation_csv: Path | None,
    duration_seconds: float,
) -> Path:
    """Render one 20-second H.264 preview and 12 evidence frames."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["accepted"][dataset_name]
    left_video = project_path(selected["left"])
    right_video = project_path(selected["right"])
    config = load_config(config_path)
    dataset = section(config, "dataset")
    left_metadata = resolve_path(dataset.get("left_metadata"), must_exist=True)
    right_metadata = resolve_path(dataset.get("right_metadata"), must_exist=True)
    assert left_metadata is not None and right_metadata is not None
    timestamps, synced, _, _ = build_synced_timeline(
        left_metadata,
        right_metadata,
        dataset.get("timestamp_format", "seconds_microseconds_columns"),
    )
    name, points, raw = load_candidate(candidate_path)
    frame_count = min(len(points), len(synced), len(timestamps))
    points = points[:frame_count]
    raw = raw[:frame_count]
    timestamps = timestamps[:frame_count]
    synced = synced[:frame_count]
    canonical = canonicalize(points)
    angles = compute_angle_sequence(
        points,
        ["LeftElbow", "RightElbow"],
    )
    errors = evaluation_errors(evaluation_csv, frame_count)
    evidence = representative_indices(points, raw, errors)
    evidence_dir = output_dir / "representative_frames"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "best_candidate_preview_mjpg.avi"
    output = output_dir / "best_candidate_preview_h264.mp4"
    fps = 12.5
    maximum_frames = min(
        frame_count,
        int(round(duration_seconds * fps)),
    )
    reader = StereoFrameReader(
        left_video,
        right_video,
        synced,
        rotate_180=False,
    )
    writer: cv2.VideoWriter | None = None
    try:
        for frame_index in range(maximum_frames):
            ok, left, _ = reader.read_synced_sequential(frame_index)
            if not ok or left is None:
                continue
            left = cv2.resize(left, (640, 480))
            panel = render_skeleton_panel(
                {"Candidate": canonical},
                frame_index,
                {"Candidate": angles},
            )
            panel = cv2.resize(panel, (640, 480))
            canvas = np.hstack([left, panel])
            cv2.putText(
                canvas,
                f"{name} | t={timestamps[frame_index]:.2f}s | frame={frame_index}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(temporary),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    fps,
                    (canvas.shape[1], canvas.shape[0]),
                )
            writer.write(canvas)
            if frame_index in evidence:
                cv2.imwrite(
                    str(evidence_dir / f"frame_{frame_index:06d}.jpg"),
                    canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
    finally:
        reader.release()
        if writer is not None:
            writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    temporary.unlink(missing_ok=True)
    (output_dir / "preview_manifest.json").write_text(
        json.dumps(
            {
                "candidate": name,
                "dataset": dataset_name,
                "candidate_path": str(candidate_path),
                "duration_seconds": float(duration_seconds),
                "rendered_frames": int(maximum_frames),
                "representative_frame_indices": evidence,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset", choices=["fanbo3", "fanbo4", "fanbo7"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-csv", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the requested preview."""
    args = parse_args(argv)
    output = render(
        project_path(args.candidate),
        project_path(args.config),
        project_path(args.selection),
        args.dataset,
        project_path(args.output_dir),
        project_path(args.evaluation_csv) if args.evaluation_csv else None,
        args.duration_seconds,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
