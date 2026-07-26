"""No-GPU tests for the NVDEC compatibility transcode helper."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from transcode_nvdec_compatible import (  # noqa: E402
    HEVC_LOSSLESS,
    TranscodeError,
    build_transcode_command,
    default_manifest_path,
    parse_gpu_query,
    parse_probe_payload,
    sha256_file,
    validate_paths,
    verify_output_metadata,
)


class TranscodeNvdecCompatibleTest(unittest.TestCase):
    """Exercise helper logic without invoking FFmpeg or a GPU."""

    def test_command_contains_validated_near_lossless_settings(self) -> None:
        command = build_transcode_command(
            "ffmpeg",
            Path("/data/source.mkv"),
            Path("/data/proxy.mkv"),
            max_frames=12,
        )
        expected_pairs = {
            "-vf": "format=nv12",
            "-c:v": "h264_nvenc",
            "-preset": "p4",
            "-tune": "hq",
            "-rc": "constqp",
            "-qp": "0",
            "-profile:v": "high",
            "-bf": "0",
            "-frames:v": "12",
        }
        for option, value in expected_pairs.items():
            index = command.index(option)
            self.assertEqual(command[index + 1], value)
        self.assertIn("-n", command)
        self.assertEqual(command[-1], "/data/proxy.mkv")

    def test_command_omits_frame_limit_when_not_requested(self) -> None:
        command = build_transcode_command(
            "ffmpeg",
            Path("/data/source.mkv"),
            Path("/data/proxy.mkv"),
        )
        self.assertNotIn("-frames:v", command)

    def test_command_can_rotate_upside_down_sensor_frames(self) -> None:
        command = build_transcode_command(
            "ffmpeg",
            Path("/data/source.mkv"),
            Path("/data/proxy.mkv"),
            rotate_180=True,
        )
        filter_index = command.index("-vf")
        self.assertEqual(
            command[filter_index + 1],
            "hflip,vflip,format=nv12",
        )

    def test_hevc_lossless_command_preserves_full_range(self) -> None:
        command = build_transcode_command(
            "ffmpeg",
            Path("/data/source.mkv"),
            Path("/data/proxy.mkv"),
            mode=HEVC_LOSSLESS,
        )
        expected_pairs = {
            "-c:v": "libx265",
            "-preset": "ultrafast",
            "-x265-params": "lossless=1:range=full:pools=24",
            "-color_range": "pc",
        }
        for option, value in expected_pairs.items():
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn(
            "scale=in_range=full:out_range=full",
            command[command.index("-vf") + 1],
        )

    def test_probe_parser_exposes_compatibility_fields(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_name": "h264",
                    "profile": "High",
                    "width": 2048,
                    "height": 1536,
                    "pix_fmt": "yuv420p",
                    "color_range": "tv",
                    "avg_frame_rate": "25/2",
                }
            ],
            "format": {
                "format_name": "matroska,webm",
                "duration": "10.0",
                "size": "12345",
            },
        }
        result = parse_probe_payload(payload, Path("proxy.mkv"))
        self.assertEqual(result["codec_name"], "h264")
        self.assertEqual(result["profile"], "High")
        self.assertEqual(result["pixel_format"], "yuv420p")
        self.assertEqual(result["container_size_bytes"], "12345")

    def test_output_validation_rejects_high_444(self) -> None:
        errors = verify_output_metadata(
            {
                "codec_name": "h264",
                "profile": "High 4:4:4 Predictive",
                "pixel_format": "yuv444p",
            }
        )
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("profile" in error for error in errors))
        self.assertTrue(any("pixel format" in error for error in errors))
        self.assertEqual(
            verify_output_metadata(
                {
                    "codec_name": "h264",
                    "profile": "High",
                    "pixel_format": "yuv420p",
                }
            ),
            [],
        )
        self.assertEqual(
            verify_output_metadata(
                {
                    "codec_name": "hevc",
                    "profile": "Main",
                    "pixel_format": "yuvj420p",
                    "color_range": "pc",
                },
                mode=HEVC_LOSSLESS,
            ),
            [],
        )

    def test_hash_manifest_name_and_no_overwrite_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mkv"
            output = root / "proxy.mkv"
            manifest = default_manifest_path(output)
            source.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(source),
                (
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb"
                    "410ff61f20015ad"
                ),
            )
            self.assertEqual(manifest.name, "proxy.mkv.manifest.json")
            validate_paths(source, output, manifest, sys.executable, sys.executable)
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(TranscodeError, "overwrite"):
                validate_paths(
                    source,
                    output,
                    manifest,
                    sys.executable,
                    sys.executable,
                )

    def test_gpu_parser(self) -> None:
        parsed = parse_gpu_query(
            "0, NVIDIA RTX A6000, GPU-abcd, 580.95.05, 49140\n"
        )
        self.assertEqual(parsed[0]["index"], 0)
        self.assertEqual(parsed[0]["name"], "NVIDIA RTX A6000")
        self.assertEqual(parsed[0]["memory_total_mib"], 49140)


if __name__ == "__main__":
    unittest.main()
