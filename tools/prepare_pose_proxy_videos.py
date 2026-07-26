#!/opt/anaconda3/envs/pose/bin/python
"""Create upright, frame-preserving upload proxies for the GPU pose matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = Path(
    "00_pose_pipeline_v2/configs/nvidia_pose_matrix.yaml"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str | Path) -> Path:
    """Resolve a path against the project root."""
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one required YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def probe(path: Path) -> dict[str, Any]:
    """Read compact video metadata with ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_packets",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,"
        "nb_frames,nb_read_packets,duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}")
    return dict(streams[0])


def encode(
    source: Path,
    destination: Path,
    *,
    crf: int | None,
    preset: str,
    codec: str,
    pixel_format: str,
    rotate_180: bool,
    maximum_frames: int | None,
) -> dict[str, Any]:
    """Encode one proxy and verify dimensions, rate, and packet count."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
    ]
    if rotate_180:
        command.extend(["-vf", "hflip,vflip"])
    command.extend(["-c:v", codec, "-preset", preset])
    if crf is None:
        command.extend(["-qp", "0"])
    else:
        command.extend(["-crf", str(crf)])
    if maximum_frames is not None:
        command.extend(["-frames:v", str(maximum_frames)])
    command.extend(["-pix_fmt", pixel_format, str(destination)])
    subprocess.check_call(command)
    source_info = probe(source)
    output_info = probe(destination)
    comparable = ("width", "height", "r_frame_rate")
    mismatches = {
        key: [source_info.get(key), output_info.get(key)]
        for key in comparable
        if str(source_info.get(key)) != str(output_info.get(key))
    }
    source_packets = int(source_info["nb_read_packets"])
    expected_packets = (
        min(source_packets, maximum_frames)
        if maximum_frames is not None
        else source_packets
    )
    output_packets = int(output_info["nb_read_packets"])
    if output_packets != expected_packets:
        mismatches["nb_read_packets"] = [
            expected_packets,
            output_packets,
        ]
    if mismatches:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Proxy contract failed for {source}: {mismatches}"
        )
    return {
        "source": str(source),
        "destination": str(destination),
        "command": shlex.join(command),
        "source_sha256": sha256_file(source),
        "destination_sha256": sha256_file(destination),
        "source_bytes": source.stat().st_size,
        "destination_bytes": destination.stat().st_size,
        "compression_ratio": (
            destination.stat().st_size / source.stat().st_size
        ),
        "source_video": source_info,
        "destination_video": output_info,
        "maximum_frames": maximum_frames,
    }


def prepare(
    matrix_path: Path,
    dataset_names: Sequence[str],
    crf_values: Sequence[int],
    include_lossless: bool,
    maximum_frames: int | None,
) -> Path:
    """Create requested proxy variants and their provenance manifest."""
    matrix = load_yaml(matrix_path)
    datasets = matrix.get("datasets", {})
    proxy = matrix.get("proxy", {})
    output_root = project_path(proxy["output_root"])
    selected = list(dataset_names) or list(datasets)
    records: list[dict[str, Any]] = []
    for dataset_name in selected:
        if dataset_name not in datasets:
            raise KeyError(f"Unknown dataset: {dataset_name}")
        dataset_config_path = project_path(
            datasets[dataset_name]["config"]
        )
        dataset_config = load_yaml(dataset_config_path)
        source_config = dataset_config["dataset"]
        sources = {
            "left": project_path(source_config["left_video"]),
            "right": project_path(source_config["right_video"]),
        }
        variants: list[tuple[str, int | None, str, str]] = [
            (
                f"crf{value}",
                int(value),
                str(proxy.get("codec", "libx264")),
                str(proxy.get("pixel_format", "yuv420p")),
            )
            for value in crf_values
        ]
        if include_lossless:
            fallback = proxy["lossless_fallback"]
            variants.append(
                (
                    "lossless_qp0",
                    None,
                    str(fallback.get("codec", "libx264")),
                    str(fallback.get("pixel_format", "gray")),
                )
            )
        for variant, crf, codec, pixel_format in variants:
            output_variant = (
                f"{variant}_screen{maximum_frames}"
                if maximum_frames is not None
                else variant
            )
            extension = ".mkv" if crf is None else ".mp4"
            for side, source in sources.items():
                destination = (
                    output_root
                    / dataset_name
                    / output_variant
                    / f"{side}{extension}"
                )
                if destination.exists():
                    existing = {
                        "dataset": dataset_name,
                        "side": side,
                        "variant": output_variant,
                        "destination": str(destination),
                        "destination_sha256": sha256_file(destination),
                        "destination_bytes": destination.stat().st_size,
                        "destination_video": probe(destination),
                        "reused": True,
                    }
                    records.append(existing)
                    continue
                record = encode(
                    source,
                    destination,
                    crf=crf,
                    preset=str(proxy.get("preset", "slow")),
                    codec=codec,
                    pixel_format=pixel_format,
                    rotate_180=bool(
                        proxy.get("rotate_180_during_encode", True)
                    ),
                    maximum_frames=maximum_frames,
                )
                record.update(
                    {
                        "dataset": dataset_name,
                        "side": side,
                        "variant": output_variant,
                        "reused": False,
                    }
                )
                records.append(record)
    manifest_path = output_root / "proxy_manifest.json"
    existing_records: list[dict[str, Any]] = []
    if manifest_path.is_file():
        existing_payload = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        existing_records = list(existing_payload.get("records", []))
    record_keys = {
        (record["dataset"], record["side"], record["variant"])
        for record in records
    }
    merged_records = [
        record
        for record in existing_records
        if (record["dataset"], record["side"], record["variant"])
        not in record_keys
    ]
    merged_records.extend(records)
    manifest = {
        "schema_version": "pose_proxy_manifest_v1",
        "matrix_config": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "rotation": "180 degrees applied once during proxy encoding",
        "records": merged_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--crfs", nargs="*", type=int)
    parser.add_argument("--include-lossless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Create proxies."""
    args = parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    matrix_path = project_path(args.matrix)
    matrix = load_yaml(matrix_path)
    crfs = (
        args.crfs
        if args.crfs is not None
        else [int(value) for value in matrix["proxy"]["crf_ladder"]]
    )
    manifest = prepare(
        matrix_path,
        args.datasets,
        crfs,
        args.include_lossless,
        args.max_frames,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
