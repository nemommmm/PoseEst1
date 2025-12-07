import cv2
import numpy as np
import os
import sys

# 导入工具函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader

# ================= 路径配置 =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Calibration_video")
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")

# 随便选一个视频来验证
LEFT_VIDEO = os.path.join(DATA_DIR, "cap_0_left.avi")
RIGHT_VIDEO = os.path.join(DATA_DIR, "cap_0_right.avi")
LEFT_TXT = os.path.join(DATA_DIR, "cap_0_left.txt")
RIGHT_TXT = os.path.join(DATA_DIR, "cap_0_right.txt")
# ==========================================

def main():
    if not os.path.exists(PARAM_PATH):
        print("❌ 找不到标定参数！请先运行 01_calibration.py")
        return

    # 1. 读取标定参数
    print("📂 读取标定参数...")
    data = np.load(PARAM_PATH)
    mtx_l, dist_l = data['mtx_l'], data['dist_l']
    mtx_r, dist_r = data['mtx_r'], data['dist_r']
    R, T = data['R'], data['T']

    # 2. 初始化视频加载器
    loader = StereoDataLoader(LEFT_VIDEO, RIGHT_VIDEO, LEFT_TXT, RIGHT_TXT)
    frame_l, frame_r, *_ = loader.get_next_pair()
    
    if frame_l is None:
        print("❌ 视频读取失败")
        return

    h, w = frame_l.shape[:2]

    # 3. 计算立体校正映射 (Rectification Maps)
    # 这一步是把两个相机“强行掰正”，让它们的成像平面平行
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T
    )

    # 生成查找表 (Mapping Table)
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w, h), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w, h), cv2.CV_16SC2)

    print("\n👀 正在显示校正结果...")
    print("   - 观察图像是否变直？")
    print("   - 观察绿线：左图的一个点，是否刚好落在右图的绿线上？(极线对齐)")
    print("⌨️  按 'n' 下一帧，按 'q' 退出")

    while True:
        if frame_l is None: break

        # 使用查找表进行校正
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # 拼接显示
        display = np.hstack((rect_l, rect_r))

        # 画绿色的水平辅助线 (每隔 30 像素画一条)
        for y in range(0, display.shape[0], 30):
            cv2.line(display, (0, y), (display.shape[1], y), (0, 255, 0), 1)

        # 缩放显示
        display = cv2.resize(display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Rectification Check', display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            # 读取下一对
            frame_l, frame_r, *_ = loader.get_next_pair()

    loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()