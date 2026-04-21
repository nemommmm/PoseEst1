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

DEFAULT_OPTIMIZED_POSE_NPZ = (
    METHOD_DIR
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "optimized_pose.npz"
)
DEFAULT_RAW_POSE_NPZ = (
    METHOD_DIR
    / "results"
    / "historical_best_20260324"
    / "recovered_baseline"
    / "raw_pose.npz"
)
DEFAULT_CAMERA_PARAMS = PROJECT_ROOT / "shared" / "camera_params.npz"
OUTPUT_DIR = METHOD_DIR / "results" / "skt_for_aitor"
TRC_PRE_CORR_OUT = OUTPUT_DIR / "markers_skt_pre_correction_coco17_mm.trc"
TRC_POST_CORR_OUT = OUTPUT_DIR / "markers_skt_post_correction_coco17_mm.trc"
SUMMARY_JSON = OUTPUT_DIR / "skt_coordinate_audit.json"
SUMMARY_EN = OUTPUT_DIR / "README_for_Aitor_EN.md"
SUMMARY_CN = OUTPUT_DIR / "README_for_Aitor_CN.md"

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
    raw_pose_npz_path: Path,
    optimized_pose_npz_path: Path,
    camera_param_path: Path,
    raw_timestamps_rel: np.ndarray,
    raw_keypoints_cm: np.ndarray,
    optimized_timestamps_rel: np.ndarray,
    optimized_keypoints_cm: np.ndarray,
    raw_variant: str,
    optimized_variant: str,
) -> dict[str, object]:
    """Build a machine-readable audit summary for the exported SKT skeleton."""
    cam = np.load(camera_param_path)
    translation = np.asarray(cam["T"], dtype=np.float64).reshape(-1)
    baseline_cm = float(np.linalg.norm(translation))

    raw_finite_mask = np.isfinite(raw_keypoints_cm).all(axis=2)
    optimized_finite_mask = np.isfinite(optimized_keypoints_cm).all(axis=2)
    raw_joint_coverage = {
        name: float(np.mean(raw_finite_mask[:, idx])) for idx, name in enumerate(COCO17_NAMES)
    }
    optimized_joint_coverage = {
        name: float(np.mean(optimized_finite_mask[:, idx])) for idx, name in enumerate(COCO17_NAMES)
    }
    raw_pelvis = 0.5 * (raw_keypoints_cm[:, 11] + raw_keypoints_cm[:, 12])
    valid_raw_pelvis = raw_pelvis[np.isfinite(raw_pelvis).all(axis=1)]

    return {
        "source_pose_npz_raw": str(raw_pose_npz_path),
        "source_pose_npz_optimized": str(optimized_pose_npz_path),
        "exported_trc_pre_correction": str(TRC_PRE_CORR_OUT),
        "exported_trc_post_correction": str(TRC_POST_CORR_OUT),
        "camera_params_path": str(camera_param_path),
        "source_units": "cm",
        "export_units": "mm",
        "frame_count_raw": int(len(raw_timestamps_rel)),
        "frame_count_optimized": int(len(optimized_timestamps_rel)),
        "fps_inferred_raw": infer_fps(raw_timestamps_rel),
        "fps_inferred_optimized": infer_fps(optimized_timestamps_rel),
        "time_start_s_raw": float(raw_timestamps_rel[0]) if len(raw_timestamps_rel) else 0.0,
        "time_end_s_raw": float(raw_timestamps_rel[-1]) if len(raw_timestamps_rel) else 0.0,
        "time_start_s_optimized": float(optimized_timestamps_rel[0]) if len(optimized_timestamps_rel) else 0.0,
        "time_end_s_optimized": float(optimized_timestamps_rel[-1]) if len(optimized_timestamps_rel) else 0.0,
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
            "easyergo_upload_status": (
                "The current EasyErgo upload in our workflow is an upright-only video. "
                "There is no evidence in the current repo workflow that it was stereo-rectified "
                "before upload."
            ),
            "rectified_easyergo_feasible": True,
        },
        "pose_extent_cm": {
            "pelvis_min_raw": np.nanmin(valid_raw_pelvis, axis=0).tolist() if len(valid_raw_pelvis) else [np.nan] * 3,
            "pelvis_max_raw": np.nanmax(valid_raw_pelvis, axis=0).tolist() if len(valid_raw_pelvis) else [np.nan] * 3,
        },
        "pose_variants": {
            "pre_correction": {
                "postprocess_variant": raw_variant,
                "interpretation": (
                    "This is not sensor-raw. It is the retained SKT pose before the final pose-correction "
                    "stage, but it already includes upstream inference-time steps "
                    "such as tracked crop, 2D temporal smoothing, soft epipolar correction, and window retriangulation."
                ),
            },
            "post_correction": {
                "postprocess_variant": optimized_variant,
                "interpretation": (
                    "This is the retained SKT output after the final pose-correction stage "
                    "(bone prior enforcement + OneEuro filtering)."
                ),
            },
        },
        "joint_coverage_ratio": {
            "raw": raw_joint_coverage,
            "optimized": optimized_joint_coverage,
        },
        "manual_pointcloud_overlay": {
            "feasible": True,
            "existing_scripts": [
                "02_dense_stereo_sgbm/src/09_visualize_point_cloud.py",
                "02_dense_stereo_sgbm/src/10_interactive_point_cloud.py",
            ],
            "note": (
                "The repo already contains dense point-cloud visualizers with SKT and Xsens overlays. "
                "To focus on the person only, we can crop the point cloud by the tracked bbox or by a "
                "2D skeleton hull before rendering."
            ),
        },
    }


def render_english_summary(summary: dict[str, object]) -> str:
    """Render a concise English note for Aitor."""
    intr = summary["left_camera_intrinsics"]
    coord = summary["coordinate_frame"]
    calib = summary["calibration_application"]
    pre_info = summary["pose_variants"]["pre_correction"]
    post_info = summary["pose_variants"]["post_correction"]
    pre_name = Path(summary["exported_trc_pre_correction"]).name
    post_name = Path(summary["exported_trc_post_correction"]).name
    return "\n".join(
        [
            "# SKT Export for Aitor",
            "",
            "## Delivered files",
            "",
            f"- Recommended main file (post-correction): `{post_name}`",
            f"- Reference file (pre-correction): `{pre_name}`",
            f"- Units: `{summary['export_units']}`",
            f"- Frames (raw / optimized): `{summary['frame_count_raw']} / {summary['frame_count_optimized']}`",
            f"- FPS (raw / optimized): `{summary['fps_inferred_raw']:.4f} / {summary['fps_inferred_optimized']:.4f}`",
            "",
            "## Historical evaluation note",
            "",
            "- The reported `13.21° calibrated / 18.59° uncalibrated` joint-angle MAE refers to a downstream piecewise angle-calibration step used during evaluation.",
            "- That calibration is applied after angle computation and does not modify the underlying 3D keypoint coordinates stored in TRC.",
            "- Therefore, there is no separate `calibrated TRC` vs `uncalibrated TRC` in a strict geometric sense.",
            f"- For geometry inspection, the recommended file is still `{post_name}`.",
            "",
            "## Important processing note",
            "",
            "- Here, 'correction' means the final 3D pose-space postprocess stage, not the later angle-calibration step used in evaluation.",
            f"- Pre-correction variant tag: `{pre_info['postprocess_variant']}`",
            f"- Pre-correction meaning: {pre_info['interpretation']}",
            f"- Post-correction variant tag: `{post_info['postprocess_variant']}`",
            f"- Post-correction meaning: {post_info['interpretation']}",
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
    pre_info = summary["pose_variants"]["pre_correction"]
    post_info = summary["pose_variants"]["post_correction"]
    pre_name = Path(summary["exported_trc_pre_correction"]).name
    post_name = Path(summary["exported_trc_post_correction"]).name
    return "\n".join(
        [
            "# SKT 导出给 Aitor 的说明",
            "",
            "## 已导出文件",
            "",
            f"- 推荐主文件（post-correction）: `{post_name}`",
            f"- 参考文件（pre-correction）: `{pre_name}`",
            f"- 导出单位: `{summary['export_units']}`",
            f"- 帧数（raw / optimized）: `{summary['frame_count_raw']} / {summary['frame_count_optimized']}`",
            f"- 按时间戳推断 FPS（raw / optimized）: `{summary['fps_inferred_raw']:.4f} / {summary['fps_inferred_optimized']:.4f}`",
            "",
            "## 历史评估说明",
            "",
            "- 你记得的 `13.21° calibrated / 18.59° uncalibrated`，指的是评估阶段的分段 angle calibration。",
            "- 这一步发生在角度计算之后，不会修改 TRC 中保存的 3D 关键点坐标。",
            "- 所以严格来说，并不存在两份几何上不同的“calibrated TRC / uncalibrated TRC”。",
            f"- 如果要给 Aitor 做几何和坐标系检查，仍然应该优先看 `{post_name}`。",
            "",
            "## 重要说明：这里的“矫正”指最终 3D 姿态后处理，不是后面的角度 calibration",
            "",
            "- 这两份都不是“传感器级完全原始数据”。",
            f"- Pre-correction 版本标签: `{pre_info['postprocess_variant']}`",
            f"- Pre-correction 版本含义: {pre_info['interpretation']}",
            f"- Post-correction 版本标签: `{post_info['postprocess_variant']}`",
            f"- Post-correction 版本含义: {post_info['interpretation']}",
            "",
            "## SKT 坐标系",
            "",
            f"- 坐标系名称: `{coord['name']}`",
            f"- 原点: {coord['origin']}",
            f"- X 轴: {coord['x_axis']}",
            f"- Y 轴: {coord['y_axis']}",
            f"- Z 轴: {coord['z_axis']}",
            f"- 备注: {coord['note']}",
            "",
            "## 标定 / 矫正链路",
            "",
            f"- 基线长度: `{summary['baseline_cm']:.3f} cm`",
            f"- 左目内参: `fx={intr['fx']:.3f}`, `fy={intr['fy']:.3f}`, `cx={intr['cx']:.3f}`, `cy={intr['cy']:.3f}`",
            f"- 翻正步骤: {calib['video_loading']}",
            f"- 标定生效步骤: {calib['rectification_step']}",
            f"- 三角化步骤: {calib['triangulation_step']}",
            "",
        ]
    )


def main() -> None:
    """Export the retained SKT skeleton and write audit notes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_pose_payload = np.load(DEFAULT_RAW_POSE_NPZ, allow_pickle=True)
    optimized_pose_payload = np.load(DEFAULT_OPTIMIZED_POSE_NPZ, allow_pickle=True)

    raw_timestamps_abs = np.asarray(raw_pose_payload["timestamps"], dtype=np.float64)
    raw_keypoints_cm = np.asarray(raw_pose_payload["keypoints"], dtype=np.float64)
    optimized_timestamps_abs = np.asarray(optimized_pose_payload["timestamps"], dtype=np.float64)
    optimized_keypoints_cm = np.asarray(optimized_pose_payload["keypoints"], dtype=np.float64)

    for path, keypoints in [
        (DEFAULT_RAW_POSE_NPZ, raw_keypoints_cm),
        (DEFAULT_OPTIMIZED_POSE_NPZ, optimized_keypoints_cm),
    ]:
        if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
            raise ValueError(f"Unexpected keypoint shape in {path}: {keypoints.shape}")

    raw_timestamps_rel = raw_timestamps_abs - raw_timestamps_abs[0] if len(raw_timestamps_abs) else raw_timestamps_abs
    optimized_timestamps_rel = (
        optimized_timestamps_abs - optimized_timestamps_abs[0]
        if len(optimized_timestamps_abs)
        else optimized_timestamps_abs
    )
    raw_fps = infer_fps(raw_timestamps_rel)
    optimized_fps = infer_fps(optimized_timestamps_rel)

    raw_keypoints_mm = raw_keypoints_cm * 10.0
    optimized_keypoints_mm = optimized_keypoints_cm * 10.0
    write_trc(
        marker_positions=raw_keypoints_mm,
        marker_names=COCO17_NAMES,
        timestamps_rel=raw_timestamps_rel,
        fps=raw_fps,
        units="mm",
        output_path=TRC_PRE_CORR_OUT,
    )
    write_trc(
        marker_positions=optimized_keypoints_mm,
        marker_names=COCO17_NAMES,
        timestamps_rel=optimized_timestamps_rel,
        fps=optimized_fps,
        units="mm",
        output_path=TRC_POST_CORR_OUT,
    )

    summary = build_summary(
        raw_pose_npz_path=DEFAULT_RAW_POSE_NPZ,
        optimized_pose_npz_path=DEFAULT_OPTIMIZED_POSE_NPZ,
        camera_param_path=DEFAULT_CAMERA_PARAMS,
        raw_timestamps_rel=raw_timestamps_rel,
        raw_keypoints_cm=raw_keypoints_cm,
        optimized_timestamps_rel=optimized_timestamps_rel,
        optimized_keypoints_cm=optimized_keypoints_cm,
        raw_variant=str(raw_pose_payload["postprocess_variant"]),
        optimized_variant=str(optimized_pose_payload["postprocess_variant"]),
    )
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_EN.write_text(render_english_summary(summary), encoding="utf-8")
    SUMMARY_CN.write_text(render_chinese_summary(summary), encoding="utf-8")

    print(f"[saved] {TRC_PRE_CORR_OUT}")
    print(f"[saved] {TRC_POST_CORR_OUT}")
    print(f"[saved] {SUMMARY_JSON}")
    print(f"[saved] {SUMMARY_EN}")
    print(f"[saved] {SUMMARY_CN}")
    print(f"[info] inferred FPS raw={raw_fps:.4f}, optimized={optimized_fps:.4f}")


if __name__ == "__main__":
    main()
