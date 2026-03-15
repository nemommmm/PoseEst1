import cv2
import os

class StereoDataLoader:
    """
    A robust data loader for stereo vision that ensures strictly synchronized 
    frame pairs based on hardware timestamp/Frame IDs, compensating for potential frame drops.
    """
    def __init__(self, left_video_path, right_video_path, left_txt_path, right_txt_path):
        # 1. Open video streams
        self.cap_left = cv2.VideoCapture(left_video_path)
        self.cap_right = cv2.VideoCapture(right_video_path)
        
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise IOError("[Error] Cannot open video files. Please verify the paths.")

        # 2. Parse metadata files for synchronization (Frame ID is the absolute truth)
        print("[Info] Parsing metadata files for hardware-level synchronization...")
        self.left_data = self._parse_txt_to_list(left_txt_path)
        self.right_data = self._parse_txt_to_list(right_txt_path)
        
        # 3. Initialize dual-pointers for stream alignment
        self.ptr_l = 0 # Pointer for left stream
        self.ptr_r = 0 # Pointer for right stream
        
        # Stream statistics
        self.count_match = 0
        self.count_drop = 0
        
        print("[Info] Initialization complete.")
        print(f"       Left stream frames: {len(self.left_data)}, Right stream frames: {len(self.right_data)}")
        print("[Info] Ready to serve synchronized frame pairs.")

    def _parse_txt_to_list(self, txt_path):
        """
        Parse metadata text file.
        Returns: A list of dictionaries [{'id': 100, 'ts': 1234.56}, ...]
        """
        data_list = []
        if not os.path.exists(txt_path):
            print(f"[Warning] Metadata file not found: {txt_path}")
            return []

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        # Assuming Column 1 is the absolute Frame ID
                        frame_id = int(parts[0])
                        # Column 2 and 3 constitute the timestamp (seconds.microseconds)
                        timestamp = float(f"{parts[1]}.{parts[2]}")
                        data_list.append({'id': frame_id, 'ts': timestamp})
                    except ValueError:
                        continue
        return data_list

    def get_next_pair(self):
        """
        Intelligently synchronize and fetch the next valid stereo frame pair.
        Logic: The stream with the lower Frame ID advances (drops frames) to catch up with the other.
        Returns: (frame_left, frame_right, common_frame_id, timestamp)
        """
        while self.ptr_l < len(self.left_data) and self.ptr_r < len(self.right_data):
            
            meta_l = self.left_data[self.ptr_l]
            meta_r = self.right_data[self.ptr_r]
            
            id_l = meta_l['id']
            id_r = meta_r['id']
            
            # --- Case 1: Frame IDs match perfectly ---
            if id_l == id_r:
                ret_l, frame_l = self.cap_left.read()
                ret_r, frame_r = self.cap_right.read()
                
                self.ptr_l += 1
                self.ptr_r += 1
                
                if not ret_l or not ret_r:
                    return None, None, None, None
                
                # Image Rectification: Rotate 180 degrees to ensure subjects are upright for YOLO
                frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)
                frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)

                self.count_match += 1
                return frame_l, frame_r, id_l, meta_l['ts']

            # --- Case 2: Left ID is smaller (Left stream is lagging or Right stream dropped a frame) ---
            elif id_l < id_r:
                # Use grab() instead of read() for faster bypassing (demuxing without decoding)
                self.cap_left.grab() 
                self.ptr_l += 1
                self.count_drop += 1
                continue 

            # --- Case 3: Right ID is smaller (Right stream is lagging) ---
            else: 
                self.cap_right.grab()
                self.ptr_r += 1
                self.count_drop += 1
                continue

        # End of stream reached
        return None, None, None, None

    def release(self):
        print("\n[Info] Video streams released. Final Statistics:")
        print(f"       Synchronized Pairs: {self.count_match}")
        print(f"       Dropped/Skipped Frames: {self.count_drop}")
        self.cap_left.release()
        self.cap_right.release()