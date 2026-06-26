"""Minimal MVNX parser used by the standalone pipeline."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class MvnxParser:
    """Parse Xsens MVNX positions, joint angles, and ergonomic angles."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.data: np.ndarray | None = None
        self.timestamps: np.ndarray | None = None
        self.segment_map: dict[int, str] = {}
        self.joint_labels: list[str] = []
        self.joint_angles: np.ndarray | None = None
        self.ergo_labels: list[str] = []
        self.ergo_angles: np.ndarray | None = None
        self.frame_rate = 60.0

    def parse(self) -> None:
        """Parse the MVNX XML file."""
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)
        text = self.file_path.read_text(encoding="utf-8")
        text = re.sub(r'\sxmlns="[^"]+"', "", text, count=1)
        root = ET.fromstring(text)
        subject = root.find("subject")
        if subject is None:
            subject = next((child for child in root if "subject" in child.tag.lower()), None)
        if subject is None:
            raise ValueError("MVNX subject node not found.")

        self.frame_rate = float(subject.attrib.get("frameRate", 60.0))
        segments = subject.find("segments")
        if segments is None:
            raise ValueError("MVNX segments node not found.")
        for seg in segments.findall("segment"):
            self.segment_map[int(seg.attrib["id"])] = seg.attrib["label"]

        joints = subject.find("joints")
        if joints is not None:
            self.joint_labels = [item.attrib["label"] for item in joints.findall("joint")]

        ergo = subject.find("ergonomicJointAngles")
        if ergo is not None:
            self.ergo_labels = [item.attrib["label"] for item in ergo.findall("ergonomicJointAngle")]

        frames = subject.find("frames")
        if frames is None:
            raise ValueError("MVNX frames node not found.")
        n_segments = len(self.segment_map)
        n_joints = len(self.joint_labels)
        n_ergo = len(self.ergo_labels)
        timestamps = []
        positions = []
        joint_angles = []
        ergo_angles = []

        for frame in frames.findall("frame"):
            if frame.attrib.get("type") != "normal":
                continue
            timestamps.append(float(frame.attrib.get("time", 0.0)) / 1000.0)

            pos_node = frame.find("position")
            if pos_node is not None and pos_node.text:
                values = np.fromstring(pos_node.text, sep=" ")
                positions.append(values.reshape(-1, 3) * 100.0)
            else:
                positions.append(np.full((n_segments, 3), np.nan))

            joint_node = frame.find("jointAngle")
            if joint_node is not None and joint_node.text and n_joints:
                values = np.fromstring(joint_node.text, sep=" ")
                joint_angles.append(values.reshape(-1, 3))
            else:
                joint_angles.append(np.full((n_joints, 3), np.nan))

            ergo_node = frame.find("jointAngleErgo")
            if ergo_node is not None and ergo_node.text and n_ergo:
                values = np.fromstring(ergo_node.text, sep=" ")
                ergo_angles.append(values.reshape(-1, 3))
            else:
                ergo_angles.append(np.full((n_ergo, 3), np.nan))

        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.data = np.asarray(positions, dtype=np.float64)
        self.joint_angles = np.asarray(joint_angles, dtype=np.float64)
        self.ergo_angles = np.asarray(ergo_angles, dtype=np.float64)

    def get_segment_data(self, segment_name: str) -> np.ndarray | None:
        """Return one segment trajectory in cm."""
        if self.data is None:
            raise RuntimeError("Call parse() before accessing MVNX data.")
        for segment_id, name in self.segment_map.items():
            if name.lower() == segment_name.lower():
                idx = segment_id - 1
                return self.data[:, idx, :] if idx < self.data.shape[1] else None
        return None

    def get_joint_angle_data(self, label: str) -> np.ndarray | None:
        """Return one native joint-angle trajectory."""
        if self.joint_angles is None:
            return None
        for idx, name in enumerate(self.joint_labels):
            if name.lower() == label.lower():
                return self.joint_angles[:, idx, :]
        return None

    def get_ergo_angle_data(self, label: str) -> np.ndarray | None:
        """Return one ergonomic angle trajectory."""
        if self.ergo_angles is None:
            return None
        for idx, name in enumerate(self.ergo_labels):
            if name.lower() == label.lower():
                return self.ergo_angles[:, idx, :]
        return None
