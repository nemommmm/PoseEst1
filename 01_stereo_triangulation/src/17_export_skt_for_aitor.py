#!/opt/anaconda3/envs/pose/bin/python
"""Export the retained SKT skeleton to TRC for Aitor's external workflow.

This utility converts the historical-best SKT output into a plain TRC file so
that Aitor can inspect the coordinate system, floor position, and possible
transformations in his own toolchain. It also writes short English/Chinese
notes summarizing how the SKT coordinate frame is defined and where calibration
is applied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent

DEFAULT_FINAL_POSE_NPZ = (
    METHOD_DIR
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
DEFAULT_CAMERA_PARAMS = PROJECT_ROOT / "shared" / "camera_params.npz"
OUTPUT_DIR = METHOD_DIR / "results" / "skt_for_aitor"
TRC_FINAL_OUT = OUTPUT_DIR / "markers_skt_coco17_mm.trc"
SUMMARY_JSON = OUTPUT_DIR / "skt_coordinate_audit.json"
SUMMARY_EN = OUTPUT_DIR / "README_for_Aitor_EN.md"
SUMMARY_CN = OUTPUT_DIR / "README_for_Aitor_CN.md"
STALE_TRC_FILES = [
    OUTPUT_DIR / "markers_skt_pre_correction_coco17_mm.trc",
    OUTPUT_DIR / "markers_skt_post_correction_coco17_mm.trc",
    OUTPUT_DIR / "markers_skt_raw_coco17_mm.trc",
    OUTPUT_DIR / "markers_skt_optimized_coco17_mm.trc",
]

COCO17_NAMES = [
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
]


def infer_fps(timestamps: np.ndarray) -> float:
    """Infer the effective frame rate from timestamp differences."""
    if len(timestamps) < 2:
        return 0.0
    dt = np.diff(timestamps)
    dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def format_trc_value(value: float) -> str:
    """Format one TRC coordinate; use blank for NaN to keep the file readable."""
    if not np.isfinite(value):
        return ""
    return f"{value:.6f}"


def write_trc(
    marker_positions: np.ndarray,
    marker_names: Iterable[str],
    timestamps_rel: np.ndarray,
    fps: float,
    units: str,
    output_path: Path,
) -> None:
    """Write a TRC file using the provided timestamps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    marker_names = list(marker_names)
    num_frames = int(marker_positions.shape[0])
    num_markers = int(marker_positions.shape[1])

    if len(marker_names) != num_markers:
        raise ValueError(
            f"Marker-name count mismatch: expected {num_markers}, got {len(marker_names)}."
        )

    lines: list[str] = []
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}")
    lines.append(
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    )
    lines.append(
        f"{fps:.6f}\t{fps:.6f}\t{num_frames}\t{num_markers}\t{units}\t{fps:.6f}\t1\t{num_frames}"
    )

    marker_header = "Frame#\tTime"
    for name in marker_names:
        marker_header += f"\t{name}\t\t"
    lines.append(marker_header.rstrip("\t"))

    coord_header = "\t"
    for idx in range(num_markers):
        coord_header += f"\tX{idx + 1}\tY{idx + 1}\tZ{idx + 1}"
    lines.append(coord_header)
    lines.append("")

    for frame_idx in range(num_frames):
        row = [str(frame_idx + 1), f"{float(timestamps_rel[frame_idx]):.6f}"]
        for marker_idx in range(num_markers):
            x, y, z = marker_positions[frame_idx, marker_idx]
            row.extend(
                [
                    format_trc_value(float(x)),
                    format_trc_value(float(y)),
                    format_trc_value(float(z)),
                ]
            )
        lines.append("\t".join(row))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    final_pose_npz_path: Path,
    camera_param_path: Path,
    timestamps_rel: np.ndarray,
    keypoints_cm: np.ndarray,
    postprocess_variant: str,
) -> dict[str, object]:
    """Build a machine-readable audit summary for the exported SKT skeleton."""
    cam = np.load(camera_param_path)
    translation = np.asarray(cam["T"], dtype=np.float64).reshape(-1)
    baseline_cm = float(np.linalg.norm(translation))

    finite_mask = np.isfinite(keypoints_cm).all(axis=2)
    joint_coverage = {name: float(np.mean(finite_mask[:, idx])) for idx, name in enumerate(COCO17_NAMES)}
    pelvis = 0.5 * (keypoints_cm[:, 11] + keypoints_cm[:, 12])
    valid_pelvis = pelvis[np.isfinite(pelvis).all(axis=1)]

    return {
        "source_pose_npz": str(final_pose_npz_path),
        "exported_trc": str(TRC_FINAL_OUT),
        "camera_params_path": str(camera_param_path),
        "source_units": "cm",
        "export_units": "mm",
        "frame_count": int(len(timestamps_rel)),
        "fps_inferred": infer_fps(timestamps_rel),
        "time_start_s": float(timestamps_rel[0]) if len(timestamps_rel) else 0.0,
        "time_end_s": float(timestamps_rel[-1]) if len(timestamps_rel) else 0.0,
        "baseline_cm": baseline_cm,
        "left_camera_intrinsics": {
            "fx": float(cam["mtx_l"][0, 0]),
            "fy": float(cam["mtx_l"][1, 1]),
            "cx": float(cam["mtx_l"][0, 2]),
            "cy": float(cam["mtx_l"][1, 2]),
        },
        "translation_left_to_right_cm": translation.tolist(),
        "coordinate_frame": {
            "name": "rectified_left_camera_frame",
            "origin": "optical center of the rectified left camera (P1 camera center)",
            "x_axis": "positive to image right",
            "y_axis": "positive to image down",
            "z_axis": "positive forward away from the camera",
            "note": (
                "This is a camera-centric stereo reconstruction frame, not a ground/world frame "
                "and not an OpenSim biomechanical frame."
            ),
        },
        "calibration_application": {
            "video_loading": (
                "Raw left/right videos are rotated 180 degrees to upright in shared/utils.py "
                "before pose inference."
            ),
            "rectification_step": (
                "Stereo calibration is applied inside 01_stereo_triangulation/src/02_batch_inference.py "
                "via cv2.stereoRectify + cv2.undistortPoints."
            ),
            "triangulation_step": (
                "3D points are triangulated from rectified 2D keypoints using rectified projection "
                "matrices P1/P2."
            ),
        },
        "pose_extent_cm": {
            "pelvis_min": np.nanmin(valid_pelvis, axis=0).tolist() if len(valid_pelvis) else [np.nan] * 3,
            "pelvis_max": np.nanmax(valid_pelvis, axis=0).tolist() if len(valid_pelvis) else [np.nan] * 3,
        },
        "pose_variant": {
            "postprocess_variant": postprocess_variant,
            "interpretation": (
                "This file represents the retained final SKT output after the final pose-correction "
                "stage (bone prior enforcement + OneEuro filtering)."
            ),
        },
        "joint_coverage_ratio": joint_coverage,
    }


def render_english_summary(summary: dict[str, object]) -> str:
    """Render a concise English note for Aitor."""
    intr = summary["left_camera_intrinsics"]
    coord = summary["coordinate_frame"]
    calib = summary["calibration_application"]
    pose_variant = summary["pose_variant"]
    trc_name = Path(summary["exported_trc"]).name
    return "\n".join(
        [
            "# SKT Export for Aitor",
            "",
            "## Orientation note",
            "",
            "- The raw stereo videos are physically upside down.",
            "- All image-direction references, axis interpretations, and processing steps in this note refer to the 180-degree-rotated upright frames used by our pipeline, not the original upside-down AVI frames.",
            "",
            "## Delivered file",
            "",
            f"- Final TRC: `{trc_name}`",
            f"- Units: `{summary['export_units']}`",
            f"- Frames: `{summary['frame_count']}`",
            f"- FPS: `{summary['fps_inferred']:.4f}`",
            "",
            "## Processing note",
            "",
            f"- Variant tag: `{pose_variant['postprocess_variant']}`",
            f"- Meaning: {pose_variant['interpretation']}",
            "",
            "## SKT coordinate frame",
            "",
            f"- Frame: `{coord['name']}`",
            f"- Origin: {coord['origin']}",
            f"- X: {coord['x_axis']}",
            f"- Y: {coord['y_axis']}",
            f"- Z: {coord['z_axis']}",
            f"- Note: {coord['note']}",
            "",
            "## Calibration / rectification",
            "",
            f"- Baseline: `{summary['baseline_cm']:.3f} cm`",
            f"- Left intrinsics: `fx={intr['fx']:.3f}`, `fy={intr['fy']:.3f}`, `cx={intr['cx']:.3f}`, `cy={intr['cy']:.3f}`",
            f"- Upright handling: {calib['video_loading']}",
            f"- Calibration step: {calib['rectification_step']}",
            f"- Triangulation step: {calib['triangulation_step']}",
            "",
        ]
    )


def render_chinese_summary(summary: dict[str, object]) -> str:
    """Render a concise Chinese note for local review."""
    intr = summary["left_camera_intrinsics"]
    coord = summary["coordinate_frame"]
    calib = summary["calibration_application"]
    pose_variant = summary["pose_variant"]
    trc_name = Path(summary["exported_trc"]).name
    return "\n".join(
        [
            "# SKT 导出给 Aitor 的说明",
            "",
            "## 朝向说明",
            "",
            "- 原始双目视频本身是上下颠倒的。",
            "- 本说明后续提到的图像方向、坐标轴方向和处理步骤，均以先旋转 180° 后的正常朝向视频为准，而不是原始倒置 AVI。",
            "",
            "## 已导出文件",
            "",
            f"- 最终 TRC: `{trc_name}`",
            f"- 导出单位: `{summary['export_units']}`",
            f"- 帧数: `{summary['frame_count']}`",
            f"- 按时间戳推断 FPS: `{summary['fps_inferred']:.4f}`",
            "",
            "## 处理说明",
            "",
            f"- 版本标签: `{pose_variant['postprocess_variant']}`",
            "- 含义: 该文件对应当前保留下来的最终 SKT 3D 输出，已经经过最终 3D 姿态后处理",
            "  （bone prior enforcement + OneEuro filtering）。",
            "",
            "## SKT 坐标系",
            "",
            f"- 坐标系名称: `{coord['name']}`",
            "- 原点: 矫正后左相机的光心（P1 camera center）",
            "- X 轴: 朝图像右侧为正",
            "- Y 轴: 朝图像下方为正",
            "- Z 轴: 朝相机前方、远离相机方向为正",
            "- 备注: 这是以相机为中心的双目重建坐标系，不是地面/世界坐标系，也不是 OpenSim 坐标系。",
            "",
            "## 标定 / 矫正链路",
            "",
            f"- 基线长度: `{summary['baseline_cm']:.3f} cm`",
            f"- 左目内参: `fx={intr['fx']:.3f}`, `fy={intr['fy']:.3f}`, `cx={intr['cx']:.3f}`, `cy={intr['cy']:.3f}`",
            "- 翻正步骤: 原始左右视频会先在 `shared/utils.py` 中旋转 180°，再进入姿态估计。",
            "- 标定生效步骤: `01_stereo_triangulation/src/02_batch_inference.py` 中通过 `cv2.stereoRectify` 和 `cv2.undistortPoints` 应用双目标定。",
            "- 三角化步骤: 使用矫正后的 2D 关键点和矫正后的投影矩阵 `P1/P2` 完成三角化。",
            "",
        ]
    )


def cleanup_stale_trc_files() -> None:
    """Remove legacy multi-version TRC exports to keep one final deliverable."""
    for path in STALE_TRC_FILES:
        if path.exists():
            path.unlink()


def main() -> None:
    """Export the retained SKT skeleton and write audit notes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_trc_files()

    pose_payload = np.load(DEFAULT_FINAL_POSE_NPZ, allow_pickle=True)

    timestamps_abs = np.asarray(pose_payload["timestamps"], dtype=np.float64)
    keypoints_cm = np.asarray(pose_payload["keypoints"], dtype=np.float64)

    if keypoints_cm.ndim != 3 or keypoints_cm.shape[1:] != (17, 3):
        raise ValueError(f"Unexpected keypoint shape in {DEFAULT_FINAL_POSE_NPZ}: {keypoints_cm.shape}")

    timestamps_rel = timestamps_abs - timestamps_abs[0] if len(timestamps_abs) else timestamps_abs
    fps = infer_fps(timestamps_rel)

    keypoints_mm = keypoints_cm * 10.0
    write_trc(
        marker_positions=keypoints_mm,
        marker_names=COCO17_NAMES,
        timestamps_rel=timestamps_rel,
        fps=fps,
        units="mm",
        output_path=TRC_FINAL_OUT,
    )

    summary = build_summary(
        final_pose_npz_path=DEFAULT_FINAL_POSE_NPZ,
        camera_param_path=DEFAULT_CAMERA_PARAMS,
        timestamps_rel=timestamps_rel,
        keypoints_cm=keypoints_cm,
        postprocess_variant=str(pose_payload["postprocess_variant"]),
    )
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_EN.write_text(render_english_summary(summary), encoding="utf-8")
    SUMMARY_CN.write_text(render_chinese_summary(summary), encoding="utf-8")

    print(f"[saved] {TRC_FINAL_OUT}")
    print(f"[saved] {SUMMARY_JSON}")
    print(f"[saved] {SUMMARY_EN}")
    print(f"[saved] {SUMMARY_CN}")
    print(f"[info] inferred FPS={fps:.4f}")


if __name__ == "__main__":
    main()
