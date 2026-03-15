import xml.etree.ElementTree as ET
import numpy as np
import re
import os

class MvnxParser:
    """
    Parser for Xsens MVNX (Mvn Open XML) files.
    Extracts kinematic segment positions and their corresponding timestamps.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.segments = {}
        self.data = None
        self.frame_rate = 60.0
        
    def parse(self):
        print(f"[Info] Parsing MVNX file: {os.path.basename(self.file_path)} ...")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[Error] File not found: {self.file_path}")

        # Pre-processing: Strip XML Namespace to simplify ElementTree tag querying
        with open(self.file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
            
        xml_content = re.sub(r'\sxmlns="[^"]+"', '', xml_content, count=1)
        
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ValueError(f"[Error] Malformed XML format: {e}")

        # 1. Locate Subject Node
        if 'subject' in root.tag:
            subject = root
        else:
            subject = root.find('subject')
            
        if subject is None:
            # Fallback heuristic search
            for child in root:
                if 'subject' in child.tag.lower():
                    subject = child
                    break
        
        if subject is None:
            tags = [child.tag for child in root]
            raise ValueError(f"[Error] <subject> tag not found. Root children: {tags}")

        # 2. Extract Frame Rate
        self.frame_rate = float(subject.attrib.get('frameRate', 60))
        print(f"       Frame Rate: {self.frame_rate} fps")
        
        # 3. Extract Kinematic Segment Mapping (ID -> Name)
        segments_node = subject.find('segments')
        if segments_node is None:
             raise ValueError("[Error] <segments> block not found in MVNX.")

        seg_map = {} 
        for seg in segments_node.findall('segment'):
            seg_id = int(seg.attrib['id'])
            seg_label = seg.attrib['label']
            seg_map[seg_id] = seg_label
            
        # 4. Parse 3D Position Data
        frames_node = subject.find('frames')
        if frames_node is None:
             raise ValueError("[Error] <frames> block not found in MVNX.")
        
        all_positions = []
        timestamps = []
        
        frame_list = frames_node.findall('frame')
        print(f"       Reading kinematic data for {len(frame_list)} frames...")

        for frame in frame_list:
            # Parse Timestamp (ms to seconds)
            if 'time' in frame.attrib:
                time_ms = float(frame.attrib['time']) 
                timestamps.append(time_ms / 1000.0)
            else:
                timestamps.append(0.0) # Fallback
            
            # Parse Position Coordinates
            pos_node = frame.find('position')
            if pos_node is not None and pos_node.text:
                pos_vals = np.fromstring(pos_node.text, sep=' ')
                # Reshape to (N_segments, 3) and convert meters to centimeters
                pos_reshaped = pos_vals.reshape(-1, 3) * 100.0 
                all_positions.append(pos_reshaped)
            else:
                # Handle missing frame data explicitly by inserting NaNs to preserve array shape
                n_segs = len(seg_map)
                all_positions.append(np.full((n_segs, 3), np.nan))
            
        self.data = np.array(all_positions) # Shape: (Frames, Segments, 3)
        self.timestamps = np.array(timestamps)
        self.segment_map = seg_map
        
        print(f"[Info] MVNX Parsing complete. Data matrix shape: {self.data.shape}")
        
    def get_segment_data(self, segment_name):
        """
        Retrieve trajectory for a specific anatomical segment.
        Returns: Numpy array of shape (N_frames, 3)
        """
        target_idx = -1
        # Case-insensitive search
        for idx, name in self.segment_map.items():
            if name.lower() == segment_name.lower():
                # Correct for 1-based indexing commonly found in MVNX IDs
                target_idx = idx - 1
                break
        
        if target_idx == -1:
            print(f"[Warning] Segment '{segment_name}' not found. Available segments: {list(self.segment_map.values())[:5]}...")
            return None
        
        # Boundary protection
        if target_idx >= self.data.shape[1]:
             print(f"[Error] Index out of bounds: ID {target_idx} exceeds matrix dimension {self.data.shape[1]}")
             return None

        return self.data[:, target_idx, :]