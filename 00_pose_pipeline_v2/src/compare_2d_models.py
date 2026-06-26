"""Compare stereo 2D pose detectors before changing the SKT pipeline.

This diagnostic intentionally stops at the 2D layer. It measures whether a
candidate detector improves left/right right-arm consistency, which is the
necessary condition before using it as a full SKT replacement.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.config import get_run_dir, load_config, resolve_path, section
from common.metrics import jsonable
from skt_inference import choose_person, rectify_points
from stereo_loader import StereoFrameReader, build_synced_timeline

RIGHT_ARM = [6, 8, 10]
RIGHT_ARM_NAMES = {6: "RShoulder", 8: "RElbow", 10: "RWrist"}


class Detector(Protocol):
    """Detector interface used by all 2D model wrappers."""

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return COCO-17 keypoints and confidence scores for one person."""


@dataclass
class CandidateResult:
    """One candidate detector outcome."""

    name: str
    status: str
    reason: str = ""
    processed_frames: int = 0
    runtime_mean_ms_pair: float | None = None
    right_arm_chain_valid_ratio: float | None = None
    jump_context_chain_valid_ratio: float | None = None
    right_arm_min_conf_median: float | None = None
    right_arm_epipolar_median_px: float | None = None
    right_arm_epipolar_p95_px: float | None = None
    jump_context_epipolar_median_px: float | None = None
    per_joint_valid_ratio: dict[str, float] | None = None
    per_joint_epipolar_median_px: dict[str, float | None] | None = None


class YoloDetector:
    """Ultralytics YOLO pose wrapper."""

    def __init__(self, model_ref: str, conf_threshold: float, center_weight: float, image_width: int) -> None:
        from ultralytics import YOLO

        resolved = resolve_path(model_ref, must_exist=False)
        model_arg = str(resolved) if resolved is not None and resolved.exists() else Path(model_ref).name
        self.model = YOLO(model_arg)
        if resolved is not None and not resolved.exists():
            downloaded = Path.cwd() / Path(model_ref).name
            if downloaded.exists() and downloaded.resolve() != resolved.resolve():
                resolved.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), str(resolved))
        self.conf_threshold = float(conf_threshold)
        self.center_weight = float(center_weight)
        self.image_width = int(image_width)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect the best person in one frame."""
        result = self.model(image, conf=self.conf_threshold, verbose=False)[0]
        return choose_person(result, img_width=self.image_width, center_weight=self.center_weight)


class RtmposeDetector:
    """RTMLib Body wrapper."""

    def __init__(self, mode: str, device: str, min_detection_conf: float) -> None:
        from rtmlib import Body

        self.body = Body(mode=mode, backend="onnxruntime", device=device)
        self.min_detection_conf = float(min_detection_conf)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect the best person in one frame."""
        keypoints, scores = self.body(image)
        if keypoints is None or scores is None or len(keypoints) == 0:
            return None
        scores = np.asarray(scores, dtype=np.float64)
        person_scores = np.nanmean(scores, axis=1)
        best = int(np.nanargmax(person_scores))
        if not np.isfinite(person_scores[best]) or float(person_scores[best]) < self.min_detection_conf:
            return None
        return np.asarray(keypoints[best], dtype=np.float64), scores[best]


def load_jump_frames(run_dir: Path, threshold_deg: float, radius: int) -> set[int]:
    """Load SKT jump-context frame indices from an existing angle_timeseries.csv."""
    csv_path = run_dir / "angle_eval" / "angle_timeseries.csv"
    if not csv_path.exists():
        return set()
    import csv as csv_module

    values: list[float] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv_module.DictReader(handle)
        for row in reader:
            raw = row.get("SKT_RightElbow_deg", "")
            values.append(float(raw) if raw else math.nan)
    arr = np.asarray(values, dtype=np.float64)
    diffs = np.diff(arr)
    jumps = np.where(np.isfinite(diffs) & (np.abs(diffs) > float(threshold_deg)))[0] + 1
    out: set[int] = set()
    for idx in jumps:
        for ctx_idx in range(int(idx) - radius, int(idx) + radius + 1):
            if 0 <= ctx_idx < len(arr):
                out.add(ctx_idx)
    return out


def _finite_median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(np.median(finite))


def _finite_p95(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(np.percentile(finite, 95))


def instantiate_detector(candidate: dict, image_width: int) -> tuple[Detector | None, str]:
    """Create a detector instance, returning a skip/error reason on failure."""
    if not candidate.get("enabled", True):
        return None, "disabled"
    kind = str(candidate.get("kind", "")).lower()
    try:
        if kind == "yolo":
            model_ref = str(candidate.get("model", ""))
            if not model_ref:
                return None, "missing model"
            return YoloDetector(
                model_ref=model_ref,
                conf_threshold=float(candidate.get("confidence_threshold", 0.35)),
                center_weight=float(candidate.get("center_person_weight", 0.0)),
                image_width=image_width,
            ), ""
        if kind == "rtmpose":
            return RtmposeDetector(
                mode=str(candidate.get("mode", "balanced")),
                device=str(candidate.get("device", "cpu")),
                min_detection_conf=float(candidate.get("min_detection_conf", 0.20)),
            ), ""
        if kind == "sapiens":
            return None, "Sapiens wrapper is not implemented in v2 Stage 1"
        return None, f"unsupported kind: {kind}"
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        return None, f"{type(exc).__name__}: {exc}"


def evaluate_candidate(
    candidate: dict,
    frames: list[int],
    jump_context: set[int],
    reader: StereoFrameReader,
    calibration: dict[str, np.ndarray],
    image_size: tuple[int, int],
) -> CandidateResult:
    """Run one candidate detector on sampled stereo frames and summarize 2D consistency."""
    name = str(candidate.get("name", candidate.get("model", "candidate")))
    detector, reason = instantiate_detector(candidate, image_width=image_size[0])
    if detector is None:
        return CandidateResult(name=name, status="skipped", reason=reason)

    mtx_l = calibration["mtx_l"]
    dist_l = calibration["dist_l"]
    mtx_r = calibration["mtx_r"]
    dist_r = calibration["dist_r"]
    r1 = calibration["r1"]
    r2 = calibration["r2"]
    p1 = calibration["p1"]
    p2 = calibration["p2"]

    min_conf_threshold = float(candidate.get("valid_conf_threshold", 0.25))
    chain_valid: list[bool] = []
    chain_valid_jump: list[bool] = []
    runtime_ms: list[float] = []
    min_conf_values: list[float] = []
    epi_values: list[float] = []
    epi_values_jump: list[float] = []
    per_joint_valid = {idx: [] for idx in RIGHT_ARM}
    per_joint_epi = {idx: [] for idx in RIGHT_ARM}

    for frame_idx in frames:
        ok, frame_l, frame_r = reader.read_synced(frame_idx)
        if not ok or frame_l is None or frame_r is None:
            continue
        t0 = time.perf_counter()
        det_l = detector.detect(frame_l)
        det_r = detector.detect(frame_r)
        runtime_ms.append((time.perf_counter() - t0) * 1000.0)

        pts_l = np.full((17, 2), np.nan, dtype=np.float64)
        pts_r = np.full((17, 2), np.nan, dtype=np.float64)
        conf_l = np.full(17, np.nan, dtype=np.float64)
        conf_r = np.full(17, np.nan, dtype=np.float64)
        if det_l is not None:
            pts_l, conf_l = det_l
        if det_r is not None:
            pts_r, conf_r = det_r

        rect_l = rectify_points(pts_l, mtx_l, dist_l, r1, p1)
        rect_r = rectify_points(pts_r, mtx_r, dist_r, r2, p2)
        joint_valid = []
        for joint_idx in RIGHT_ARM:
            conf = min(float(conf_l[joint_idx]) if np.isfinite(conf_l[joint_idx]) else 0.0,
                       float(conf_r[joint_idx]) if np.isfinite(conf_r[joint_idx]) else 0.0)
            has_points = np.isfinite(rect_l[joint_idx]).all() and np.isfinite(rect_r[joint_idx]).all()
            valid = bool(has_points and conf >= min_conf_threshold)
            per_joint_valid[joint_idx].append(valid)
            joint_valid.append(valid)
            if has_points:
                epi = abs(float(rect_l[joint_idx, 1] - rect_r[joint_idx, 1]))
                per_joint_epi[joint_idx].append(epi)
                epi_values.append(epi)
                if frame_idx in jump_context:
                    epi_values_jump.append(epi)
            if np.isfinite(conf_l[joint_idx]) and np.isfinite(conf_r[joint_idx]):
                min_conf_values.append(conf)
        chain_ok = bool(all(joint_valid))
        chain_valid.append(chain_ok)
        if frame_idx in jump_context:
            chain_valid_jump.append(chain_ok)

    return CandidateResult(
        name=name,
        status="ok",
        processed_frames=len(runtime_ms),
        runtime_mean_ms_pair=float(np.mean(runtime_ms)) if runtime_ms else None,
        right_arm_chain_valid_ratio=float(np.mean(chain_valid)) if chain_valid else None,
        jump_context_chain_valid_ratio=float(np.mean(chain_valid_jump)) if chain_valid_jump else None,
        right_arm_min_conf_median=_finite_median(np.asarray(min_conf_values, dtype=np.float64)),
        right_arm_epipolar_median_px=_finite_median(np.asarray(epi_values, dtype=np.float64)),
        right_arm_epipolar_p95_px=_finite_p95(np.asarray(epi_values, dtype=np.float64)),
        jump_context_epipolar_median_px=_finite_median(np.asarray(epi_values_jump, dtype=np.float64)),
        per_joint_valid_ratio={
            RIGHT_ARM_NAMES[idx]: float(np.mean(vals)) if vals else 0.0
            for idx, vals in per_joint_valid.items()
        },
        per_joint_epipolar_median_px={
            RIGHT_ARM_NAMES[idx]: _finite_median(np.asarray(vals, dtype=np.float64))
            for idx, vals in per_joint_epi.items()
        },
    )


def compare_2d_models(config: dict, run_dir: Path) -> Path:
    """Run the configured Stage 1 2D model diagnostic."""
    dataset = section(config, "dataset")
    skt = section(config, "skt")
    diag_cfg = section(config, "model_diagnostic")
    calib = section(config, "calibration")

    left_video = resolve_path(dataset.get("left_video"), must_exist=True)
    right_video = resolve_path(dataset.get("right_video"), must_exist=True)
    left_meta = resolve_path(dataset.get("left_metadata"), must_exist=True)
    right_meta = resolve_path(dataset.get("right_metadata"), must_exist=True)
    camera_params = resolve_path(calib.get("camera_params"), must_exist=True)
    assert left_video and right_video and left_meta and right_meta and camera_params

    time_s, synced, _, _ = build_synced_timeline(left_meta, right_meta, dataset.get("timestamp_format", "seconds_microseconds_columns"))
    frame_stride = max(1, int(diag_cfg.get("frame_stride", 2)))
    max_frames = diag_cfg.get("max_frames", 160)
    all_frames = list(range(0, len(synced), frame_stride))
    if max_frames and len(all_frames) > int(max_frames):
        sample_idx = np.linspace(0, len(all_frames) - 1, int(max_frames))
        frames = [all_frames[int(round(idx))] for idx in sample_idx]
    else:
        frames = all_frames
    frames = sorted(set(frames))

    reader = StereoFrameReader(left_video, right_video, synced, rotate_180=bool(dataset.get("rotate_180", False)))
    ok, first_l, _ = reader.read_synced(frames[0])
    if not ok or first_l is None:
        raise RuntimeError("Could not read first diagnostic stereo frame.")
    height, width = first_l.shape[:2]

    params = np.load(camera_params)
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        params["mtx_l"], params["dist_l"], params["mtx_r"], params["dist_r"],
        (width, height), params["R"], params["T"], alpha=0,
    )
    calibration = {
        "mtx_l": params["mtx_l"],
        "dist_l": params["dist_l"],
        "mtx_r": params["mtx_r"],
        "dist_r": params["dist_r"],
        "r1": r1,
        "r2": r2,
        "p1": p1,
        "p2": p2,
    }

    jump_context = load_jump_frames(
        run_dir,
        threshold_deg=float(diag_cfg.get("jump_threshold_deg", 10.0)),
        radius=int(diag_cfg.get("jump_context_radius", 2)),
    )
    candidates = diag_cfg.get("candidates")
    if not candidates:
        candidates = [
            {
                "name": "YOLOv8m-current",
                "kind": "yolo",
                "model": skt.get("model_path"),
                "confidence_threshold": skt.get("confidence_threshold", 0.35),
                "center_person_weight": skt.get("center_person_weight", 0.0),
            }
        ]

    out_dir = run_dir / "model_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for candidate in candidates:
            print(f"[modeldiag] running {candidate.get('name', candidate.get('model', 'candidate'))} on {len(frames)} sampled frames")
            result = evaluate_candidate(candidate, frames, jump_context, reader, calibration, (width, height))
            rows.append(result)
            print(f"[modeldiag] {result.name}: {result.status} {result.reason}")
    finally:
        reader.release()

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "name", "status", "reason", "processed_frames", "runtime_mean_ms_pair",
        "right_arm_chain_valid_ratio", "jump_context_chain_valid_ratio",
        "right_arm_min_conf_median", "right_arm_epipolar_median_px",
        "right_arm_epipolar_p95_px", "jump_context_epipolar_median_px",
        "per_joint_valid_ratio", "per_joint_epipolar_median_px",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            raw = item.__dict__.copy()
            raw["per_joint_valid_ratio"] = json.dumps(jsonable(raw["per_joint_valid_ratio"]))
            raw["per_joint_epipolar_median_px"] = json.dumps(jsonable(raw["per_joint_epipolar_median_px"]))
            writer.writerow(raw)

    summary = {
        "config": {
            "n_synced_frames": len(synced),
            "sampled_frame_count": len(frames),
            "frame_stride": frame_stride,
            "first_sampled_frame": frames[0] if frames else None,
            "last_sampled_frame": frames[-1] if frames else None,
            "jump_context_frame_count": len(jump_context),
            "note": "2D detector diagnostic only; does not replace the SKT pipeline.",
        },
        "rows": [item.__dict__ for item in rows],
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(f"[modeldiag] saved {csv_path}")
    return csv_path


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = get_run_dir(config)
    compare_2d_models(config, run_dir)


if __name__ == "__main__":
    main()
