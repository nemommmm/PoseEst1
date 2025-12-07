import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 导入配置
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import StereoDataLoader

# ================= 配置 =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Calibration_video")
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")

# 使用 cap_1
VIDEO_PAIRS = [("cap_0_left.avi", "cap_0_right.avi", "cap_0_left.txt", "cap_0_right.txt")]
PATTERN_SIZE = (5, 9)
# =======================================

def calculate_errors(img_points, obj_points, mtx, dist):
    """
    只负责计算数据，不负责画图
    """
    all_errors = []
    all_x = []
    all_y = []
    
    for i in range(len(img_points)):
        _, rvec, tvec = cv2.solvePnP(obj_points[i], img_points[i], mtx, dist)
        proj_points, _ = cv2.projectPoints(obj_points[i], rvec, tvec, mtx, dist)
        
        for j in range(len(proj_points)):
            real_pt = img_points[i][j][0]
            proj_pt = proj_points[j][0]
            error = np.linalg.norm(real_pt - proj_pt)
            
            all_errors.append(error)
            all_x.append(real_pt[0])
            all_y.append(real_pt[1])
            
    return np.array(all_x), np.array(all_y), np.array(all_errors)

def plot_on_axes(ax, x, y, errors, mtx, shape, title):
    """
    在指定的子图(ax)上绘画
    """
    # 统一颜色范围 0~1.5 px
    sc = ax.scatter(x, y, c=errors, cmap='jet', vmin=0, vmax=1.5, alpha=0.7, s=15)
    
    # 画光心
    ax.plot(mtx[0, 2], mtx[1, 2], 'g+', markersize=15, markeredgewidth=2, label='Optical Center')
    
    # 设置坐标轴 (Y轴反转)
    ax.set_xlim(0, shape[0])
    ax.set_ylim(shape[1], 0) 
    
    ax.set_title(f"{title}\nMean Error: {np.mean(errors):.4f} px")
    ax.set_xlabel("X Pixel")
    ax.set_ylabel("Y Pixel")
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.3)
    return sc

def main():
    if not os.path.exists(PARAM_PATH):
        print("❌ 找不到参数文件")
        return

    # 1. 读取参数
    data = np.load(PARAM_PATH)
    mtx_l, dist_l = data['mtx_l'], data['dist_l']
    mtx_r, dist_r = data['mtx_r'], data['dist_r']

    # 2. 收集数据
    print("🚀 正在扫描数据...")
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    for i in range(PATTERN_SIZE[1]): 
        for j in range(PATTERN_SIZE[0]): 
            objp[i * PATTERN_SIZE[0] + j] = [(j + ((i % 2) * 0.5)), i * 0.5, 0]

    img_pts_l, img_pts_r, obj_pts = [], [], []
    
    loader = StereoDataLoader(os.path.join(DATA_DIR, VIDEO_PAIRS[0][0]), 
                              os.path.join(DATA_DIR, VIDEO_PAIRS[0][1]), 
                              os.path.join(DATA_DIR, VIDEO_PAIRS[0][2]), 
                              os.path.join(DATA_DIR, VIDEO_PAIRS[0][3]))
    
    img_shape = None
    while True:
        frame_l, frame_r, _, _ = loader.get_next_pair()
        if frame_l is None: break
        if img_shape is None: img_shape = frame_l.shape[:2][::-1]

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID
        
        fl, cl = cv2.findCirclesGrid(gray_l, PATTERN_SIZE, flags=flags)
        fr, cr = cv2.findCirclesGrid(gray_r, PATTERN_SIZE, flags=flags)
        
        if fl and fr:
            img_pts_l.append(cl)
            img_pts_r.append(cr)
            obj_pts.append(objp)
    loader.release()

    print(f"✅ 数据准备完毕，正在绘图...")

    # 3. 计算误差
    lx, ly, lerr = calculate_errors(img_pts_l, obj_pts, mtx_l, dist_l)
    rx, ry, rerr = calculate_errors(img_pts_r, obj_pts, mtx_r, dist_r)

    # 4. 创建并排画布 (1行2列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 绘制左图
    plot_on_axes(ax1, lx, ly, lerr, mtx_l, img_shape, "Left Camera")
    
    # 绘制右图
    sc = plot_on_axes(ax2, rx, ry, rerr, mtx_r, img_shape, "Right Camera")

    # 添加公共颜色条
    cbar = fig.colorbar(sc, ax=[ax1, ax2], fraction=0.03, pad=0.03)
    cbar.set_label('Reprojection Error (pixels)')

    plt.suptitle(f"Stereo Reprojection Error Distribution (Total {len(obj_pts)} frames)", fontsize=14)
    print("🖥️  显示图表...")
    plt.show()

if __name__ == "__main__":
    main()