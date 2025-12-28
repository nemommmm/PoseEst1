import cv2
import os

class StereoDataLoader:
    def __init__(self, left_video_path, right_video_path, left_txt_path, right_txt_path):
        # 1. 打开视频
        self.cap_left = cv2.VideoCapture(left_video_path)
        self.cap_right = cv2.VideoCapture(right_video_path)
        
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise IOError("❌ Cannot open video files, please check the paths!")

        # 2. 读取所有元数据 (Frame ID 是核心)
        print("📂 Parsing metadata files for synchronization...")
        self.left_data = self._parse_txt_to_list(left_txt_path)
        self.right_data = self._parse_txt_to_list(right_txt_path)
        
        # 3. 初始化双指针
        self.ptr_l = 0 # 左眼数据的指针 (Index)
        self.ptr_r = 0 # 右眼数据的指针 (Index)
        
        # 统计信息
        self.count_match = 0
        self.count_drop = 0
        
        print(f"✅ Initialization complete.")
        print(f"   Left frames: {len(self.left_data)}, Right frames: {len(self.right_data)}")
        print("   Ready to sync stream based on Frame IDs...")

    def _parse_txt_to_list(self, txt_path):
        """
        读取 txt，返回 [{'id': 100, 'ts': ...}, ...]
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
                        # 假设 Column 1 是 Frame ID (绝对事实)
                        frame_id = int(parts[0])
                        # Column 2+3 是时间戳
                        timestamp = float(f"{parts[1]}.{parts[2]}")
                        data_list.append({'id': frame_id, 'ts': timestamp})
                    except ValueError:
                        continue
        return data_list

    def get_next_pair(self):
        """
        智能同步读取。
        逻辑：谁 ID 小，谁就“跑太快了”或者对方“掉队了”，小的那个要丢弃帧以等待对方。
        Returns: (frame_l, frame_r, common_id, timestamp)
        """
        while self.ptr_l < len(self.left_data) and self.ptr_r < len(self.right_data):
            
            # 获取当前的 ID
            meta_l = self.left_data[self.ptr_l]
            meta_r = self.right_data[self.ptr_r]
            
            id_l = meta_l['id']
            id_r = meta_r['id']
            
            # --- 情况 1: ID 完美匹配 ---
            if id_l == id_r:
                # 只有匹配时，才真正解码图像 (read)
                ret_l, frame_l = self.cap_left.read()
                ret_r, frame_r = self.cap_right.read()
                
                # 指针都要往前走
                self.ptr_l += 1
                self.ptr_r += 1
                
                if not ret_l or not ret_r:
                    return None, None, None, None
                
                # =========== 新增：解决倒置问题 ===========
                # 将画面旋转 180 度，确保人是站立的，YOLO 才能工作
                frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)
                frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
                # ========================================

                self.count_match += 1
                return frame_l, frame_r, id_l, meta_l['ts']

            # --- 情况 2: 左边 ID 小 (左边多出了一帧，或者说右边这帧丢了) ---
            elif id_l < id_r:
                # 左相机需要跳过这一帧来“追赶”右相机
                # 使用 grab() 比 read() 快，因为它只解压不解码
                self.cap_left.grab() 
                self.ptr_l += 1
                self.count_drop += 1
                # print(f"⚠️ Sync: Dropping Left Frame ID {id_l} (Right is at {id_r})")
                continue # 继续循环，再次尝试匹配

            # --- 情况 3: 右边 ID 小 ---
            else: # id_r < id_l
                self.cap_right.grab()
                self.ptr_r += 1
                self.count_drop += 1
                # print(f"⚠️ Sync: Dropping Right Frame ID {id_r} (Left is at {id_l})")
                continue

        # 循环结束说明有一个文件读完了
        return None, None, None, None

    def release(self):
        print(f"\n📊 Stream Released. Stats:")
        print(f"   Matched Frames: {self.count_match}")
        print(f"   Dropped/Skipped Frames: {self.count_drop}")
        self.cap_left.release()
        self.cap_right.release()