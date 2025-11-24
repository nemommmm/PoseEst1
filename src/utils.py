import cv2
import os

class StereoDataLoader:
    def __init__(self, left_video_path, right_video_path, left_txt_path, right_txt_path):
        # 1. Open video files
        self.cap_left = cv2.VideoCapture(left_video_path)
        self.cap_right = cv2.VideoCapture(right_video_path)
        
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise IOError("❌ Cannot open video files, please check the paths!")

        # 2. Get total frame counts
        self.total_frames_l = int(self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames_r = int(self.cap_right.get(cv2.CAP_PROP_FRAME_COUNT))

        # 3. Parse txt files -> returns a list: [{'id': 13711, 'ts': 123.45}, ...]
        # Note: We use a list where index 'i' corresponds to the i-th frame of the video
        self.left_data = self._parse_txt_to_list(left_txt_path)
        self.right_data = self._parse_txt_to_list(right_txt_path)
        
        # 4. Simple validation: txt lines should roughly match video frame count
        # If the difference is too large, print a warning but do not stop the program
        diff_l = abs(len(self.left_data) - self.total_frames_l)
        if diff_l > 5:
            print(f"⚠️ Warning: Left video frame count ({self.total_frames_l}) does not match txt line count ({len(self.left_data)})!")

        # Current frame index counter (starts from 0)
        self.current_frame_idx = 0
        
        # Calculate maximum readable frames (min of video frames and txt lines to prevent index out of bounds)
        self.max_idx = min(self.total_frames_l, self.total_frames_r, len(self.left_data), len(self.right_data))
        
        print(f"✅ Initialization complete: Ready to read {self.max_idx} frames sequentially.")

    def _parse_txt_to_list(self, txt_path):
        """
        Reads the txt file and returns a list of information for each line sequentially.
        The i-th element of the list corresponds to the i-th frame of the video.
        """
        data_list = []
        if not os.path.exists(txt_path):
            print(f"⚠️ Warning: txt file not found: {txt_path}")
            return []

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        frame_id = int(parts[0])
                        # Parse timestamp: Col2 = Seconds, Col3 = Micro-seconds
                        timestamp = float(f"{parts[1]}.{parts[2]}")
                        data_list.append({'id': frame_id, 'ts': timestamp})
                    except ValueError:
                        continue
        return data_list

    def get_next_pair(self):
        """
        Returns: (frame_l, frame_r, frame_id, timestamp)
        """
        # 1. Check if we have reached the end
        if self.current_frame_idx >= self.max_idx:
            return None, None, None, None
        
        # 2. Get metadata for the current frame
        # !Assumption: The i-th line in txt corresponds to the i-th frame in video
        meta_l = self.left_data[self.current_frame_idx]
        meta_r = self.right_data[self.current_frame_idx]
        
        # 3. Sync check (Double safety)
        # If Frame IDs for left and right do not match, the files are not aligned
        if meta_l['id'] != meta_r['id']:
            print(f"⚠️ Sync Warning: Frame {self.current_frame_idx} ID mismatch (L:{meta_l['id']} != R:{meta_r['id']})")
            # We choose to continue here (logging only), but this indicates data issues.
        
        target_id = meta_l['id']
        timestamp = meta_l['ts']
        
        # 4. Read video frames
        ret_l, frame_l = self.cap_left.read()
        ret_r, frame_r = self.cap_right.read()
        
        if not ret_l or not ret_r:
            return None, None, None, None
            
        # 5. Update counter
        self.current_frame_idx += 1
            
        return frame_l, frame_r, target_id, timestamp

    def release(self):
        self.cap_left.release()
        self.cap_right.release()