import xml.etree.ElementTree as ET
import numpy as np
import re
import os

class MvnxParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.segments = {}
        self.data = None
        self.frame_rate = 60.0
        
    def parse(self):
        print(f"📂 解析 MVNX 文件: {os.path.basename(self.file_path)} ...")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ 文件不存在: {self.file_path}")

        # === 核心修复：预处理去除 XML Namespace ===
        with open(self.file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
            
        # 使用正则去掉 xmlns="xxx" 定义，让 find() 可以直接用标签名查找
        xml_content = re.sub(r'\sxmlns="[^"]+"', '', xml_content, count=1)
        
        # 从字符串加载 XML
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ValueError(f"❌ XML 格式错误，无法解析: {e}")
        # ==========================================

        # 1. 获取 Subject 节点
        # 考虑到有时候 root 可能就是 subject，或者它藏在别的层级
        if 'subject' in root.tag:
            subject = root
        else:
            subject = root.find('subject')
            
        if subject is None:
            # 最后的尝试：遍历第一层子节点找名字里带 subject 的
            for child in root:
                if 'subject' in child.tag.lower():
                    subject = child
                    break
        
        if subject is None:
            # 打印一下 root 下面到底有啥，方便调试
            tags = [child.tag for child in root]
            raise ValueError(f"❌ 找不到 <subject> 标签。根节点下的标签有: {tags}")

        # 2. 获取帧率
        self.frame_rate = float(subject.attrib.get('frameRate', 60))
        print(f"   -> 帧率: {self.frame_rate} fps")
        
        # 3. 获取骨骼片段名称映射 (ID -> Name)
        segments_node = subject.find('segments')
        if segments_node is None:
             raise ValueError("❌ 找不到 <segments> 数据块")

        seg_map = {} # id -> name
        for seg in segments_node.findall('segment'):
            seg_id = int(seg.attrib['id'])
            seg_label = seg.attrib['label']
            seg_map[seg_id] = seg_label
            
        # 4. 解析 Position 数据
        frames_node = subject.find('frames')
        if frames_node is None:
             raise ValueError("❌ 找不到 <frames> 数据块")
        
        all_positions = []
        timestamps = []
        
        # 遍历所有帧
        frame_list = frames_node.findall('frame')
        print(f"   -> 正在读取 {len(frame_list)} 帧数据...")

        for frame in frame_list:
            # 解析时间戳
            if 'time' in frame.attrib:
                time_ms = float(frame.attrib['time']) 
                timestamps.append(time_ms / 1000.0)
            else:
                timestamps.append(0.0) # Fallback
            
            # 解析 Position
            # position 是一长串文本: "x1 y1 z1 x2 y2 z2 ..."
            # 注意：有时候 XML 里 position 可能是空的，或者某些帧丢失
            pos_node = frame.find('position')
            if pos_node is not None and pos_node.text:
                pos_vals = np.fromstring(pos_node.text, sep=' ')
                # 重塑为 (N_segments, 3)
                # MVNX 单位通常是米，我们转成厘米 (cm)
                pos_reshaped = pos_vals.reshape(-1, 3) * 100.0 
                all_positions.append(pos_reshaped)
            else:
                # 如果这一帧没有 position 数据，填 NaN 防止 shape 对不齐
                n_segs = len(seg_map)
                all_positions.append(np.full((n_segs, 3), np.nan))
            
        self.data = np.array(all_positions) # (Frames, Segments, 3)
        self.timestamps = np.array(timestamps)
        self.segment_map = seg_map
        
        print(f"✅ 解析完成: 数据形状 {self.data.shape}")
        
    def get_segment_data(self, segment_name):
        """获取指定部位的所有帧轨迹 (N, 3)"""
        target_idx = -1
        # 不区分大小写查找
        for idx, name in self.segment_map.items():
            if name.lower() == segment_name.lower():
                # MVNX ID 通常从 1 开始，对应数组 index 要减 1
                # 这是一个巨大的坑，必须确认 seg_map 的 ID 是 1-based 还是 0-based
                # 通常 segments 列表顺序就是 ID 顺序
                target_idx = idx - 1
                break
        
        if target_idx == -1:
            print(f"⚠️ 警告: 没找到部位 '{segment_name}'，可用的部位有: {list(self.segment_map.values())[:5]}...")
            return None
        
        # 保护防止 index 越界
        if target_idx >= self.data.shape[1]:
             print(f"⚠️ 索引越界: ID {target_idx} 超出了数据维度 {self.data.shape[1]}")
             return None

        return self.data[:, target_idx, :]