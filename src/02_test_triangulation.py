import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# 导入工具
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader

# ================= 配置 =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "2025_Ergonomics_Data")
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")

# 视频对
LEFT_VIDEO = "0_video_left.avi"
RIGHT_VIDEO = "1_video_right.avi"  # 确保文件名对齐
LEFT_TXT = "0_video_left.txt"
RIGHT_TXT = "1_video_right.txt"

CONF_THRESHOLD = 0.5 
# =======================================

def main():
    if not os.path.exists(PARAM_PATH):
        print("❌ 找不到标定参数")
        return

    # 1. 加载参数
    data = np.load(PARAM_PATH)
    mtx_l, dist_l = data['mtx_l'], data['dist_l']
    mtx_r, dist_r = data['mtx_r'], data['dist_r']
    R, T = data['R'], data['T']

    # 2. 预计算校正矩阵 (Rectification Matrix)
    # 我们不需要 map1, map2 了，只需要 R1, R2, P1, P2
    # 假设图片大概是 1080p，这里主要为了获取 P 矩阵
    # 随便给个尺寸计算矩阵，后续 undistortPoints 会用到
    # 注意：这里的 (w, h) 最好是真实的图片尺寸
    test_loader = StereoDataLoader(
        os.path.join(DATA_DIR, LEFT_VIDEO), os.path.join(DATA_DIR, RIGHT_VIDEO),
        os.path.join(DATA_DIR, LEFT_TXT), os.path.join(DATA_DIR, RIGHT_TXT)
    )
    frame_l, _, _, _ = test_loader.get_next_pair()
    if frame_l is None: return
    h, w = frame_l.shape[:2]
    test_loader.release()

    # 计算校正旋转矩阵 (R1, R2) 和 投影矩阵 (P1, P2)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0
    )

    # 3. 加载 YOLO
    model = YOLO('yolov8n-pose.pt') 

    # 4. 开启正式循环
    loader = StereoDataLoader(
        os.path.join(DATA_DIR, LEFT_VIDEO), os.path.join(DATA_DIR, RIGHT_VIDEO),
        os.path.join(DATA_DIR, LEFT_TXT), os.path.join(DATA_DIR, RIGHT_TXT)
    )

    print("🚀 开始 3D 推理 (Method: YOLO First -> Undistort Points)...")
    
    while True:
        # 这一步已经在 utils.py 里做过旋转了，拿到的是正向图
        frame_l, frame_r, fid, _ = loader.get_next_pair()
        if frame_l is None: break

        # A. YOLO 推理 (直接跑在原图上！)
        res_l = model(frame_l, verbose=False, conf=CONF_THRESHOLD)[0]
        res_r = model(frame_r, verbose=False, conf=CONF_THRESHOLD)[0]
        
        # 简单的可视化底图
        vis_l = res_l.plot()
        vis_r = res_r.plot()
        
        status_msg = "No Match"

        # B. 如果两边都检测到人
        if len(res_l.keypoints) > 0 and len(res_r.keypoints) > 0:
            # 取第一个人的关键点 (17, 2)
            # 注意：YOLO 输出的坐标是基于原图的
            pts_l_raw = res_l.keypoints.xy[0].cpu().numpy()
            pts_r_raw = res_r.keypoints.xy[0].cpu().numpy()
            
            # C. 关键步骤：坐标校正 (Undistort Points)
            # 将 原图坐标 -> 校正后的坐标
            # 输入 shape 需要是 (N, 1, 2)
            pts_l_rect = cv2.undistortPoints(pts_l_raw.reshape(-1, 1, 2), mtx_l, dist_l, R=R1, P=P1)
            pts_r_rect = cv2.undistortPoints(pts_r_raw.reshape(-1, 1, 2), mtx_r, dist_r, R=R2, P=P2)
            
            # D. 三角测量
            # 输入: (2, N)
            pts4d = cv2.triangulatePoints(P1, P2, pts_l_rect, pts_r_rect)
            
            # 归一化
            pts3d = pts4d[:3, :] / pts4d[3, :] # (3, 17)
            pts3d = pts3d.T # (17, 3) -> (X, Y, Z)

            # E. 验证与显示
            # 计算髋关节中心距离 (Left Hip=11, Right Hip=12)
            hip_center = (pts3d[11] + pts3d[12]) / 2.0
            dist_cm = hip_center[2] # Z轴
            
            # 过滤掉一些离谱的值 (比如 Z < 0 代表在相机后面，或者 Z > 20米)
            if 0 < dist_cm < 2000:
                status_msg = f"Dist: {dist_cm:.1f} cm"
                
                # 画在图上方便看
                cv2.putText(vis_l, status_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 打印第一个点的 3D 坐标看看是否在跳动
                # Nose = index 0
                print(f"Frame {fid} | Nose 3D: {pts3d[0]} | Hip Dist: {dist_cm:.1f} cm")
            else:
                status_msg = "Invalid Depth"

        # 显示
        display = np.hstack((vis_l, vis_r))
        display = cv2.resize(display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Robust Stereo 3D', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): cv2.waitKey(0) # 空格暂停

    loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()