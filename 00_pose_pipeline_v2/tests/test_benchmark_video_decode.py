"""Tests for FFmpeg CPU/NVDEC benchmark parsing and command construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_video_decode import (  # noqa: E402
    VideoInfo,
    build_decode_command,
    build_validation_command,
    compare_luma_bytes,
    frame_budget,
    nvdec_compatibility_issues,
    parse_gpu_query,
    parse_probe_payload,
    parse_ratio,
    percentile,
    summarize,
)


class BenchmarkVideoDecodeTest(unittest.TestCase):
    """Exercise logic that does not require FFmpeg or an NVIDIA GPU."""

    def test_parse_probe_prefers_counted_frames(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 2048,
                    "height": 1536,
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "25/1",
                    "nb_read_frames": "123",
                    "nb_frames": "120",
                    "duration": "4.92",
                }
            ],
            "format": {"duration": "5.0"},
        }
        result = parse_probe_payload(payload, Path("left.avi"))
        self.assertEqual(result.frame_count, 123)
        self.assertEqual(result.frame_count_source, "ffprobe_nb_read_frames")
        self.assertEqual(result.frame_rate, 25.0)
        self.assertEqual(result.width, 2048)

    def test_parse_probe_can_estimate_frames(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_name": "hevc",
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": "30000/1001",
                }
            ],
            "format": {"duration": "10.0"},
        }
        result = parse_probe_payload(payload, Path("right.avi"))
        self.assertEqual(result.frame_count, 300)
        self.assertEqual(
            result.frame_count_source, "duration_times_frame_rate_estimate"
        )
        self.assertAlmostEqual(result.frame_rate or 0, 29.97002997)

    def test_parse_ratio_rejects_invalid_values(self) -> None:
        self.assertIsNone(parse_ratio("0/0"))
        self.assertIsNone(parse_ratio("N/A"))
        self.assertIsNone(parse_ratio("-1"))
        self.assertEqual(parse_ratio("12.5"), 12.5)

    def test_cpu_command_maps_all_inputs(self) -> None:
        videos = [Path("/data/left.avi"), Path("/data/right.avi")]
        command = build_decode_command(
            "ffmpeg", videos, "cpu", frame_limit=40
        )
        self.assertEqual(command.count("-hwaccel"), 2)
        self.assertEqual(command.count("none"), 2)
        self.assertNotIn("cuda", command)
        self.assertIn("0:v:0", command)
        self.assertIn("1:v:0", command)
        self.assertEqual(command.count("-frames:v"), 2)
        self.assertEqual(command.count("40"), 2)

    def test_nvdec_command_is_explicit_and_auditable(self) -> None:
        videos = [Path("/data/left.avi"), Path("/data/right.avi")]
        command = build_decode_command(
            "ffmpeg", videos, "nvdec", frame_limit=None, gpu_id=2
        )
        self.assertEqual(command.count("cuda"), 4)
        self.assertEqual(command.count("-hwaccel_device"), 2)
        self.assertEqual(command.count("2"), 2)
        self.assertEqual(command.count("-hwaccel_output_format"), 2)
        self.assertNotIn("-frames:v", command)

    def test_validation_command_adds_gpu_download_only_for_nvdec(self) -> None:
        video = Path("/data/sample.avi")
        cpu = build_validation_command("ffmpeg", video, "cpu", frames=3)
        nvdec = build_validation_command("ffmpeg", video, "nvdec", frames=3)
        self.assertIn("-vf", cpu)
        self.assertIn("-vf", nvdec)
        self.assertIn("format=gray", cpu)
        self.assertIn("hwdownload,format=nv12,format=gray", nvdec)
        self.assertIn("gray", cpu)
        self.assertIn("gray", nvdec)

    def test_statistics_and_frame_budget(self) -> None:
        info = VideoInfo(
            path="/data/sample.avi",
            codec_name="h264",
            profile="High",
            width=10,
            height=10,
            pixel_format="yuv420p",
            color_range="tv",
            color_space="bt709",
            frame_rate=25.0,
            duration_seconds=4.0,
            frame_count=100,
            frame_count_source="ffprobe_nb_read_frames",
        )
        self.assertEqual(frame_budget(info, None), 100)
        self.assertEqual(frame_budget(info, 40), 40)
        self.assertEqual(frame_budget(info, 120), 100)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        summary = summarize([1.0, 2.0, 9.0])
        self.assertEqual(summary["median"], 2.0)
        self.assertGreater(summary["p95"], summary["median"])

    def test_high_444_h264_is_rejected_for_nvdec_timing(self) -> None:
        info = VideoInfo(
            path="/data/lossless.mkv",
            codec_name="h264",
            profile="High 4:4:4 Predictive",
            width=2048,
            height=1536,
            pixel_format="yuvj420p",
            color_range="pc",
            color_space=None,
            frame_rate=12.5,
            duration_seconds=10.0,
            frame_count=125,
            frame_count_source="ffprobe_nb_read_frames",
        )
        issues = nvdec_compatibility_issues([info])
        self.assertEqual(len(issues), 1)
        self.assertEqual(
            issues[0]["code"], "unsupported_h264_high_444_profile"
        )

    def test_parse_gpu_query(self) -> None:
        output = (
            "NVIDIA RTX A6000, GPU-abcd, 580.159.03, 49140\n"
            "NVIDIA T4, GPU-efgh, 570.1, 15360\n"
        )
        result = parse_gpu_query(output)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "NVIDIA RTX A6000")
        self.assertEqual(result[0]["memory_total_mib"], 49140)

    def test_luma_comparison_reports_checksum_and_tolerance(self) -> None:
        cpu = bytes([0, 10, 20, 30])
        nvdec = bytes([0, 11, 22, 35])
        result = compare_luma_bytes(
            cpu,
            nvdec,
            tolerance=2,
            minimum_within_tolerance=0.75,
        )
        self.assertNotEqual(result["cpu_sha256"], result["nvdec_sha256"])
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["maximum_absolute_pixel_difference"], 5)
        self.assertEqual(result["within_tolerance_fraction"], 0.75)
        self.assertTrue(result["comparison_passed"])


if __name__ == "__main__":
    unittest.main()
