import cv2
import numpy as np
import os
import sys

# 确保能导入同目录下的 utils.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader

# ================= 1. 路径与配置 =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Calibration_video")

# 定义要跑的所有视频对
VIDEO_PAIRS = [
    ("cap_0_left.avi", "cap_0_right.avi", "cap_0_left.txt", "cap_0_right.txt"),
    ("cap_1_left.avi", "cap_1_right.avi", "cap_1_left.txt", "cap_1_right.txt")
]

# 核心参数修正：非对称圆点阵列
# 根据你的图片：5列，9行
PATTERN_SIZE = (5, 9)   
SQUARE_SIZE = 15.0      # 15 cm
# ===================================================

def main():
    # ================= 2. 生成真实世界坐标 (核心修正) =================
    # 针对非对称圆点阵列 (Asymmetric Circle Grid) 的坐标生成
    # 这是一个展平的数组，存 (x, y, 0)
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    
    # 填充坐标
    for i in range(PATTERN_SIZE[1]): # 遍历行 (0到8)
        for j in range(PATTERN_SIZE[0]): # 遍历列 (0到4)
            # X轴公式：列号 * 间距 + (如果是奇数行则向右偏移 0.5 * 间距)
            x = (j + ((i % 2) * 0.5)) * SQUARE_SIZE
            
            # Y轴公式：行号 * (间距的一半)
            # Asymmetric Grid 通常垂直方向是紧密排列的
            y = i * (SQUARE_SIZE / 2)
            
            # 计算在数组中的索引
            index = i * PATTERN_SIZE[0] + j
            objp[index] = [x, y, 0]            
    # ================================================================

    # 全局容器：存放所有视频里找到的点
    all_objpoints = []   
    all_imgpoints_l = [] 
    all_imgpoints_r = [] 
    
    total_valid_frames = 0
    img_shape = None

    print(f"🚀 开始处理 {len(VIDEO_PAIRS)} 组视频任务...")

    # --- 第一阶段：遍历所有视频收集数据 ---
    for v_idx, (l_vid, r_vid, l_txt, r_txt) in enumerate(VIDEO_PAIRS):
        l_path = os.path.join(DATA_DIR, l_vid)
        r_path = os.path.join(DATA_DIR, r_vid)
        l_txt_path = os.path.join(DATA_DIR, l_txt)
        r_txt_path = os.path.join(DATA_DIR, r_txt)

        if not os.path.exists(l_path):
            print(f"⚠️ 跳过：找不到 {l_vid}")
            continue

        print(f"\n📂 正在读取第 {v_idx + 1} 组: {l_vid} & {r_vid}")
        loader = StereoDataLoader(l_path, r_path, l_txt_path, r_txt_path)

        while True:
            frame_l, frame_r, frame_id, _ = loader.get_next_pair()
            if frame_l is None:
                break
            
            # 记录图像尺寸 (只记一次)
            if img_shape is None:
                img_shape = frame_l.shape[:2][::-1]

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            
            # ⚠️ 必须使用 ASYMMETRIC 标志
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            found_l, corners_l = cv2.findCirclesGrid(gray_l, PATTERN_SIZE, flags=flags)
            found_r, corners_r = cv2.findCirclesGrid(gray_r, PATTERN_SIZE, flags=flags)
            
            if found_l and found_r:
                total_valid_frames += 1
                all_objpoints.append(objp)
                all_imgpoints_l.append(corners_l)
                all_imgpoints_r.append(corners_r)
                
                # 可视化一下
                cv2.drawChessboardCorners(frame_l, PATTERN_SIZE, corners_l, found_l)
                status = f"Valid: {total_valid_frames} | Pair: {v_idx+1}"
                cv2.putText(frame_l, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 缩小显示，只看左图
                display = cv2.resize(frame_l, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('Collecting...', display) 
                cv2.waitKey(1) # 快速过
            
        loader.release()
    
    cv2.destroyAllWindows()
    
    if total_valid_frames < 10:
        print(f"\n❌ 有效数据太少 ({total_valid_frames})，无法标定！")
        return

    print(f"\n🛑 数据收集完毕！共 {total_valid_frames} 帧有效数据。")
    print("------------------------------------------------")
    
    # --- 第二阶段：单机预标定 ---
    print("running 单目相机标定 (1/3): 左相机...")
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(all_objpoints, all_imgpoints_l, img_shape, None, None)
    print(f"   -> 左相机初始误差: {ret_l:.4f}")

    print("running 单目相机标定 (2/3): 右相机...")
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(all_objpoints, all_imgpoints_r, img_shape, None, None)
    print(f"   -> 右相机初始误差: {ret_r:.4f}")

    # --- 第三阶段：立体标定 ---
    print("running 立体标定 (3/3): 联合优化...")
    # 使用单目标定的结果作为初始值，并固定内参 (FIX_INTRINSIC) 以稳定结果
    flags = cv2.CALIB_FIX_INTRINSIC
    
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        all_objpoints, all_imgpoints_l, all_imgpoints_r,
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_shape,
        flags=flags
    )
    
    print("------------------------------------------------")
    print(f"✅ 最终立体标定结果 RMS: {ret:.4f}")
    
    if ret < 0.5:
        print("🏆 完美！标定精度极高。")
    elif ret < 1.0:
        print("🎉 合格！可以进行下一步。")
    else:
        print("⚠️ 警告：RMS 仍然偏高，请检查圆点提取是否偶尔有错位。")

    # 保存
    save_path = os.path.join(SRC_DIR, "camera_params.npz")
    np.savez(save_path, mtx_l=M1, dist_l=d1, mtx_r=M2, dist_r=d2, R=R, T=T)
    print(f"💾 参数已保存: {save_path}")

if __name__ == "__main__":
    main()