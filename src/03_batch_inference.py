import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
from tqdm import tqdm

# 导入工具
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader

# ================= 配置 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_params.npz")
# =======================================

def main():
    print("🚀 第一步：批量运行 3D 推理并保存原始数据...")
    
    # 1. 加载参数
    if not os.path.exists(PARAM_PATH):
        print(f"❌ 找不到标定参数: {PARAM_PATH}")
        return
        
    data = np.load(PARAM_PATH)
    mtx_l, dist_l = data['mtx_l'], data['dist_l']
    mtx_r, dist_r = data['mtx_r'], data['dist_r']
    R, T = data['R'], data['T']

    # 2. 预计算校正矩阵
    print("📐 计算校正矩阵...")
    # 尝试读取一帧获取尺寸
    loader_test = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"), os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"), os.path.join(DATA_DIR, "1_video_right.txt")
    )
    frame_l, _, _, _ = loader_test.get_next_pair()
    if frame_l is None:
        print("❌ 无法读取视频，请检查路径！")
        return
    h, w = frame_l.shape[:2]
    loader_test.release()
    
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0)

    # 3. 运行模型
    print("🧠 加载 YOLO 模型...")
    model = YOLO('yolov8l-pose.pt')
    
    # 重新初始化 loader
    loader = StereoDataLoader(
        os.path.join(DATA_DIR, "0_video_left.avi"), os.path.join(DATA_DIR, "1_video_right.avi"),
        os.path.join(DATA_DIR, "0_video_left.txt"), os.path.join(DATA_DIR, "1_video_right.txt")
    )

    all_timestamps = []
    all_keypoints_3d = [] 
    
    print("🎬 开始处理视频流...")
    
    # 【修复重点】不再强行读取 loader 的属性，而是用 total=None 让 tqdm 自动适应
    pbar = tqdm(total=None, desc="Processing Frames", unit="frame")
    
    while True:
        frame_l, frame_r, fid, ts = loader.get_next_pair()
        if frame_l is None: break
        
        # YOLO 推理
        res_l = model(frame_l, verbose=False, conf=0.5)[0]
        res_r = model(frame_r, verbose=False, conf=0.5)[0]
        
        kpts_3d = np.full((17, 3), np.nan) # 默认填 NaN

        if len(res_l.keypoints) > 0 and len(res_r.keypoints) > 0:
            # 这里的 .reshape(-1, 1, 2) 是为了满足 undistortPoints 的输入形状要求
            pts_l = res_l.keypoints.xy[0].cpu().numpy().reshape(-1, 1, 2)
            pts_r = res_r.keypoints.xy[0].cpu().numpy().reshape(-1, 1, 2)
            
            # 坐标校正
            pts_l_rect = cv2.undistortPoints(pts_l, mtx_l, dist_l, R=R1, P=P1)
            pts_r_rect = cv2.undistortPoints(pts_r, mtx_r, dist_r, R=R2, P=P2)
            
            # 三角测量
            pts4d = cv2.triangulatePoints(P1, P2, pts_l_rect, pts_r_rect)
            
            # 归一化 (X/W, Y/W, Z/W)
            # 防止除以0
            w_vec = pts4d[3, :]
            valid_w = w_vec != 0
            pts3d_raw = np.full((3, 17), np.nan)
            pts3d_raw[:, valid_w] = pts4d[:3, valid_w] / w_vec[valid_w]
            
            kpts_3d = pts3d_raw.T # 转置回 (17, 3)

        all_timestamps.append(ts)
        all_keypoints_3d.append(kpts_3d)
        pbar.update(1)

    loader.release()
    pbar.close()

    # 保存结果
    save_file = os.path.join(OUTPUT_DIR, "yolo_3d_raw.npz")
    np.savez(save_file, 
             timestamps=np.array(all_timestamps),
             keypoints=np.array(all_keypoints_3d))
             
    print(f"\n✅ 数据已保存至 {save_file}")
    print(f"📊 总共处理帧数: {len(all_timestamps)}")

if __name__ == "__main__":
    main()