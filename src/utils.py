import cv2
import os

class StereoDataLoader:
    def __init__(self, left_video_path, right_video_path, left_txt_path, right_txt_path):
        # 1. 打开视频
        self.cap_left = cv2.VideoCapture(left_video_path)
        self.cap_right = cv2.VideoCapture(right_video_path)
        
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise IOError("❌ 无法打开视频文件，请检查路径！")

        # 2. 获取视频总帧数
        self.total_frames_l = int(self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames_r = int(self.cap_right.get(cv2.CAP_PROP_FRAME_COUNT))

        # 3. 解析 txt 文件 -> 得到列表: [{'id': 13711, 'ts': 123.45}, ...]
        # 注意：这里我们用列表，列表的索引 index 就对应视频的第几帧
        self.left_data = self._parse_txt_to_list(left_txt_path)
        self.right_data = self._parse_txt_to_list(right_txt_path)
        
        # 4. 简单验证：txt行数应该和视频帧数差不多
        # 如果差太多，打印个警告，但不中断程序
        diff_l = abs(len(self.left_data) - self.total_frames_l)
        if diff_l > 5:
            print(f"⚠️ 警告: 左侧视频帧数 ({self.total_frames_l}) 与 txt 行数 ({len(self.left_data)}) 不一致！")

        # 当前读取到第几帧 (内部计数器，从0开始)
        self.current_frame_idx = 0
        
        # 计算最大可读取的帧数 (取视频和txt长度的最小值，防止越界)
        self.max_idx = min(self.total_frames_l, self.total_frames_r, len(self.left_data), len(self.right_data))
        
        print(f"✅ 初始化完成: 准备按顺序读取 {self.max_idx} 帧")

    def _parse_txt_to_list(self, txt_path):
        """
        读取txt，按顺序返回每一行的信息列表。
        列表的第 i 个元素对应视频的第 i 帧。
        """
        data_list = []
        if not os.path.exists(txt_path):
            print(f"⚠️ 警告: 找不到 txt 文件: {txt_path}")
            return []

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        frame_id = int(parts[0])
                        # 解析时间戳: Col 2=Seconds, Col 3=Microseconds
                        timestamp = float(f"{parts[1]}.{parts[2]}")
                        data_list.append({'id': frame_id, 'ts': timestamp})
                    except ValueError:
                        continue
        return data_list

    def get_next_pair(self):
        """
        返回: (frame_l, frame_r, frame_id, timestamp)
        """
        # 1. 检查是否读完了
        if self.current_frame_idx >= self.max_idx:
            return None, None, None, None
        
        # 2. 获取当前帧对应的 txt 信息
        # 假设：txt的第 i 行 对应 视频的第 i 帧
        meta_l = self.left_data[self.current_frame_idx]
        meta_r = self.right_data[self.current_frame_idx]
        
        # 3. 检查同步性 (双保险)
        # 如果左右眼的 Frame ID 不一样，说明文件本身没对齐
        if meta_l['id'] != meta_r['id']:
            print(f"⚠️ 同步警告: 第 {self.current_frame_idx} 帧 ID 不匹配 (L:{meta_l['id']} != R:{meta_r['id']})")
            # 这里可以选择跳过，也可以硬着头皮继续。为了严谨，我们继续，但打个日志。
        
        target_id = meta_l['id']
        timestamp = meta_l['ts']
        
        # 4. 读取视频帧
        ret_l, frame_l = self.cap_left.read()
        ret_r, frame_r = self.cap_right.read()
        
        if not ret_l or not ret_r:
            return None, None, None, None
            
        # 5. 更新计数器
        self.current_frame_idx += 1
            
        return frame_l, frame_r, target_id, timestamp

    def release(self):
        self.cap_left.release()
        self.cap_right.release()