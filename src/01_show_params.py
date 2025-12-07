import numpy as np
import os

# ================= 路径配置 =================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_SCRIPT_PATH)
PARAM_PATH = os.path.join(SRC_DIR, "camera_params.npz")

def main():
    if not os.path.exists(PARAM_PATH):
        print("❌ 找不到参数文件！请先运行标定脚本。")
        return

    print(f"📂 正在读取参数文件: {os.path.basename(PARAM_PATH)}\n")
    data = np.load(PARAM_PATH)

    # 提取数据
    mtx_l = data['mtx_l'] # 左摄内参
    dist_l = data['dist_l'] # 左摄畸变
    mtx_r = data['mtx_r'] # 右摄内参
    dist_r = data['dist_r'] # 右摄畸变
    R = data['R'] # 旋转矩阵
    T = data['T'] # 平移向量 (单位: cm)

    print("="*40)
    print("       🎥 双目相机标定报告        ")
    print("="*40)

    # 1. 汇报内参 (Intrinsics)
    # 内参矩阵格式: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    print(f"\n【1. 左相机内参】")
    print(f"  - 焦距 (fx, fy): ({mtx_l[0,0]:.2f}, {mtx_l[1,1]:.2f})")
    print(f"  - 光心 (cx, cy): ({mtx_l[0,2]:.2f}, {mtx_l[1,2]:.2f})")
    
    print(f"\n【2. 右相机内参】")
    print(f"  - 焦距 (fx, fy): ({mtx_r[0,0]:.2f}, {mtx_r[1,1]:.2f})")
    print(f"  - 光心 (cx, cy): ({mtx_r[0,2]:.2f}, {mtx_r[1,2]:.2f})")

    # 2. 汇报外参 (Extrinsics)
    # T 是平移向量，代表右相机相对于左相机的位置
    print(f"\n【3. 双目外参 (相对位置)】")
    print(f"  - 平移向量 T (cm): \n{T.T}") # 转置一下方便看
    
    # 计算基线距离 (Baseline) - 这是一个非常直观的物理量
    baseline = np.linalg.norm(T)
    print(f"  - 📏 计算出的基线距离 (Baseline): {baseline:.2f} cm")
    
    # 3. 汇报畸变 (Distortion)
    print(f"\n【4. 畸变系数 (k1, k2, p1, p2, k3)】")
    print(f"  - 左相机: {dist_l.ravel()[:5]}")
    print(f"  - 右相机: {dist_r.ravel()[:5]}")
    
    # print("\n" + "="*40)
    # print("💡 汇报提示：")
    # print(f"1. 告诉老师你的 RMS 误差是 1.33 px (可接受范围)。")
    # print(f"2. 重点展示 '基线距离' ({baseline:.2f} cm)，")
    # print("   如果这个值和实际测量值(比如尺子量一下)接近，说明标定非常准！")

if __name__ == "__main__":
    main()