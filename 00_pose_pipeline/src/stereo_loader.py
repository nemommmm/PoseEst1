"""Stereo metadata synchronization and optional frame loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class SyncedFrame:
    """One synchronized stereo-frame record."""

    frame_id: int
    left_idx: int
    right_idx: int
    ts: float


def parse_timestamp(parts: list[str], timestamp_format: str) -> float:
    """Parse a metadata timestamp according to the configured format."""
    if timestamp_format == "seconds_microseconds_columns":
        return int(parts[1]) + int(parts[2]) * 1e-6
    if timestamp_format == "seconds_float_column":
        return float(parts[1])
    raise ValueError(f"Unsupported timestamp_format: {timestamp_format}")


def parse_metadata(path: Path, timestamp_format: str = "seconds_microseconds_columns") -> list[dict[str, float | int]]:
    """Parse a stereo metadata text file."""
    rows: list[dict[str, float | int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append({"id": int(parts[0]), "ts": parse_timestamp(parts, timestamp_format)})
            except ValueError:
                continue
    return rows


def build_synced_frames(
    left_rows: list[dict[str, float | int]],
    right_rows: list[dict[str, float | int]],
) -> list[SyncedFrame]:
    """Synchronize left/right metadata by hardware frame id."""
    synced: list[SyncedFrame] = []
    li = 0
    ri = 0
    while li < len(left_rows) and ri < len(right_rows):
        left = left_rows[li]
        right = right_rows[ri]
        left_id = int(left["id"])
        right_id = int(right["id"])
        if left_id == right_id:
            synced.append(SyncedFrame(frame_id=left_id, left_idx=li, right_idx=ri, ts=float(left["ts"])))
            li += 1
            ri += 1
        elif left_id < right_id:
            li += 1
        else:
            ri += 1
    return synced


def build_synced_timeline(
    left_meta: Path,
    right_meta: Path,
    timestamp_format: str = "seconds_microseconds_columns",
) -> Tuple[np.ndarray, list[SyncedFrame], list[dict[str, float | int]], list[dict[str, float | int]]]:
    """Build a relative video timeline from synchronized stereo metadata."""
    left_rows = parse_metadata(left_meta, timestamp_format)
    right_rows = parse_metadata(right_meta, timestamp_format)
    synced = build_synced_frames(left_rows, right_rows)
    if not synced:
        raise RuntimeError("No synchronized stereo frames found.")
    abs_ts = np.asarray([row.ts for row in synced], dtype=np.float64)
    time_s = abs_ts - abs_ts[0]
    if np.any(np.diff(time_s) <= 0):
        raise RuntimeError("Synchronized timeline is not strictly increasing.")
    return time_s, synced, left_rows, right_rows


def truncate_timeline_to_pose(
    time_s: np.ndarray,
    synced: list[SyncedFrame],
    n_frames: int,
) -> tuple[np.ndarray, list[SyncedFrame]]:
    """Match evaluation timeline length to a pose output sequence."""
    if len(time_s) < n_frames:
        raise RuntimeError(f"Timeline has {len(time_s)} frames but pose has {n_frames}.")
    return time_s[:n_frames], synced[:n_frames]


def validate_stereo_inputs(config: dict) -> dict[str, object]:
    """Validate metadata synchronization and video availability."""
    from common.config import resolve_path

    dataset = config["dataset"]
    left_video = resolve_path(dataset["left_video"], must_exist=True)
    right_video = resolve_path(dataset["right_video"], must_exist=True)
    left_meta = resolve_path(dataset["left_metadata"], must_exist=True)
    right_meta = resolve_path(dataset["right_metadata"], must_exist=True)
    timestamp_format = dataset.get("timestamp_format", "seconds_microseconds_columns")
    time_s, synced, left_rows, right_rows = build_synced_timeline(left_meta, right_meta, timestamp_format)

    cap_l = cv2.VideoCapture(str(left_video))
    cap_r = cv2.VideoCapture(str(right_video))
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise IOError("Could not open one or both stereo videos.")
    left_video_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    right_video_frames = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_l.release()
    cap_r.release()

    dt = np.diff(time_s)
    return {
        "left_metadata_rows": len(left_rows),
        "right_metadata_rows": len(right_rows),
        "synced_frame_count": len(synced),
        "dropped_or_unpaired_rows": len(left_rows) + len(right_rows) - 2 * len(synced),
        "duration_s": float(time_s[-1] - time_s[0]),
        "median_dt_s": float(np.nanmedian(dt)),
        "effective_fps": float(1.0 / np.nanmedian(dt)),
        "first_frame_id": synced[0].frame_id,
        "last_frame_id": synced[-1].frame_id,
        "left_video_frames": left_video_frames,
        "right_video_frames": right_video_frames,
        "rotate_180": bool(dataset.get("rotate_180", False)),
        "timestamp_format": timestamp_format,
    }


class StereoFrameReader:
    """Read synchronized stereo frames by metadata index."""

    def __init__(self, left_video: Path, right_video: Path, synced: list[SyncedFrame], rotate_180: bool = False):
        self.left_video = Path(left_video)
        self.right_video = Path(right_video)
        self.synced = synced
        self.rotate_180 = rotate_180
        self.cap_l = cv2.VideoCapture(str(self.left_video))
        self.cap_r = cv2.VideoCapture(str(self.right_video))
        if not self.cap_l.isOpened() or not self.cap_r.isOpened():
            raise IOError("Could not open stereo videos.")

    def read_synced(self, idx: int) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Read one synchronized frame pair by synced-frame index."""
        item = self.synced[idx]
        self.cap_l.set(cv2.CAP_PROP_POS_FRAMES, item.left_idx)
        self.cap_r.set(cv2.CAP_PROP_POS_FRAMES, item.right_idx)
        ok_l, frame_l = self.cap_l.read()
        ok_r, frame_r = self.cap_r.read()
        if not ok_l or not ok_r:
            return False, None, None
        if self.rotate_180:
            frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)
            frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        return True, frame_l, frame_r

    def release(self) -> None:
        """Release video handles."""
        self.cap_l.release()
        self.cap_r.release()
