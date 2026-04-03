import xml.etree.ElementTree as ET
import numpy as np
import re
import os


class MvnxParser:
    """
    Parser for Xsens MVNX (Mvn Open XML) files.
    Extracts segment positions, joint angles, and ergonomic joint angles.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None            # (Frames, Segments, 3) positions in cm
        self.timestamps = None      # (Frames,) in seconds
        self.segment_map = {}       # {id: label}
        self.joint_labels = []      # [label, ...] for jointAngle data
        self.joint_angles = None    # (Frames, Joints, 3) in degrees
        self.ergo_labels = []       # [label, ...] for jointAngleErgo data
        self.ergo_angles = None     # (Frames, ErgoJoints, 3) in degrees
        self.frame_rate = 60.0

    def parse(self):
        print(f"[Info] Parsing MVNX file: {os.path.basename(self.file_path)} ...")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[Error] File not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        xml_content = re.sub(r'\sxmlns="[^"]+"', '', xml_content, count=1)

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ValueError(f"[Error] Malformed XML format: {e}")

        # Locate subject node
        subject = root.find('subject')
        if subject is None:
            for child in root:
                if 'subject' in child.tag.lower():
                    subject = child
                    break
        if subject is None:
            raise ValueError("[Error] <subject> tag not found.")

        self.frame_rate = float(subject.attrib.get('frameRate', 60))
        print(f"       Frame Rate: {self.frame_rate} fps")

        # --- Extract segment mapping ---
        segments_node = subject.find('segments')
        if segments_node is None:
            raise ValueError("[Error] <segments> block not found.")
        for seg in segments_node.findall('segment'):
            seg_id = int(seg.attrib['id'])
            seg_label = seg.attrib['label']
            self.segment_map[seg_id] = seg_label

        # --- Extract joint labels ---
        joints_node = subject.find('joints')
        if joints_node is not None:
            self.joint_labels = [j.attrib['label'] for j in joints_node.findall('joint')]
            print(f"       Joint definitions: {len(self.joint_labels)}")

        # --- Extract ergonomic joint labels ---
        ergo_node = subject.find('ergonomicJointAngles')
        if ergo_node is not None:
            self.ergo_labels = [
                e.attrib['label'] for e in ergo_node.findall('ergonomicJointAngle')
            ]
            print(f"       Ergonomic joint definitions: {len(self.ergo_labels)}")

        # --- Parse frame data ---
        frames_node = subject.find('frames')
        if frames_node is None:
            raise ValueError("[Error] <frames> block not found.")

        all_positions = []
        all_joint_angles = []
        all_ergo_angles = []
        timestamps = []
        n_segs = len(self.segment_map)
        n_joints = len(self.joint_labels)
        n_ergo = len(self.ergo_labels)

        frame_list = frames_node.findall('frame')
        print(f"       Total frames in file: {len(frame_list)}")

        for frame in frame_list:
            if frame.attrib.get('type') != 'normal':
                continue

            # Timestamp (ms → seconds)
            time_ms = float(frame.attrib.get('time', 0))
            timestamps.append(time_ms / 1000.0)

            # Position (meters → cm)
            pos_node = frame.find('position')
            if pos_node is not None and pos_node.text:
                pos_vals = np.fromstring(pos_node.text, sep=' ')
                all_positions.append(pos_vals.reshape(-1, 3) * 100.0)
            else:
                all_positions.append(np.full((n_segs, 3), np.nan))

            # Joint angles (degrees)
            ja_node = frame.find('jointAngle')
            if ja_node is not None and ja_node.text and n_joints > 0:
                ja_vals = np.fromstring(ja_node.text, sep=' ')
                all_joint_angles.append(ja_vals.reshape(-1, 3))
            else:
                all_joint_angles.append(np.full((n_joints, 3), np.nan))

            # Ergonomic joint angles (degrees)
            jae_node = frame.find('jointAngleErgo')
            if jae_node is not None and jae_node.text and n_ergo > 0:
                jae_vals = np.fromstring(jae_node.text, sep=' ')
                all_ergo_angles.append(jae_vals.reshape(-1, 3))
            else:
                all_ergo_angles.append(np.full((n_ergo, 3), np.nan))

        self.data = np.array(all_positions)
        self.timestamps = np.array(timestamps)
        self.joint_angles = np.array(all_joint_angles) if all_joint_angles else None
        self.ergo_angles = np.array(all_ergo_angles) if all_ergo_angles else None

        print(f"[Info] MVNX Parsing complete.")
        print(f"       Position data: {self.data.shape}")
        if self.joint_angles is not None:
            print(f"       Joint angle data: {self.joint_angles.shape}")
        if self.ergo_angles is not None:
            print(f"       Ergo angle data: {self.ergo_angles.shape}")

    def get_segment_data(self, segment_name):
        """Retrieve position trajectory for a segment. Returns (N_frames, 3)."""
        for idx, name in self.segment_map.items():
            if name.lower() == segment_name.lower():
                target_idx = idx - 1  # 1-based → 0-based
                if target_idx >= self.data.shape[1]:
                    print(f"[Error] Index {target_idx} out of bounds for shape {self.data.shape}")
                    return None
                return self.data[:, target_idx, :]
        print(f"[Warning] Segment '{segment_name}' not found.")
        return None

    def get_joint_angle_data(self, joint_label):
        """
        Retrieve angle time-series for a specific joint.
        Returns (N_frames, 3) in degrees, or None if not found.
        """
        if self.joint_angles is None:
            return None
        for idx, label in enumerate(self.joint_labels):
            if label.lower() == joint_label.lower():
                return self.joint_angles[:, idx, :]
        print(f"[Warning] Joint '{joint_label}' not found. Available: {self.joint_labels}")
        return None

    def get_ergo_angle_data(self, ergo_label):
        """
        Retrieve ergonomic angle time-series (e.g. 'Pelvis_T8' for trunk flexion).
        Returns (N_frames, 3) in degrees, or None if not found.
        """
        if self.ergo_angles is None:
            return None
        for idx, label in enumerate(self.ergo_labels):
            if label.lower() == ergo_label.lower():
                return self.ergo_angles[:, idx, :]
        print(f"[Warning] Ergo joint '{ergo_label}' not found. Available: {self.ergo_labels}")
        return None

    def get_all_joint_angles(self):
        """Return (timestamps, joint_labels, joint_angles) tuple."""
        return self.timestamps, self.joint_labels, self.joint_angles

    def get_all_ergo_angles(self):
        """Return (timestamps, ergo_labels, ergo_angles) tuple."""
        return self.timestamps, self.ergo_labels, self.ergo_angles