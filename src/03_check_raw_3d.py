import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ================= 配置 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DATA_PATH = os.path.join(PROJECT_ROOT, "results", "yolo_3d_raw.npz")
# =======================================

def calculate_limb_stats(kpts):
    """计算肢体长度统计信息"""
    # YOLO keypoints: 
    # 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee
    # 15: left_ankle, 16: right_ankle
    
    # 欧氏距离
    l_thigh = np.linalg.norm(kpts[:, 11, :] - kpts[:, 13, :], axis=1)
    r_thigh = np.linalg.norm(kpts[:, 12, :] - kpts[:, 14, :], axis=1)
    l_shin  = np.linalg.norm(kpts[:, 13, :] - kpts[:, 15, :], axis=1)
    r_shin  = np.linalg.norm(kpts[:, 14, :] - kpts[:, 16, :], axis=1)
    
    # 肩宽 (5: l_shoulder, 6: r_shoulder)
    shoulder_width = np.linalg.norm(kpts[:, 5, :] - kpts[:, 6, :], axis=1)
    
    return l_thigh, r_thigh, l_shin, r_shin, shoulder_width

def main():
    if not os.path.exists(YOLO_DATA_PATH):
        print(f"❌ 找不到数据: {YOLO_DATA_PATH}，请先运行 03_batch_inference.py")
        return

    print(f"📂 正在体检 YOLO 3D 数据: {os.path.basename(YOLO_DATA_PATH)}...")
    data = np.load(YOLO_DATA_PATH)
    kpts = data['keypoints'] # (N, 17, 3)
    ts = data['timestamps']
    
    total_frames = len(ts)
    valid_frames = np.sum(~np.isnan(kpts[:, 0, 0]))
    print(f"📊 总帧数: {total_frames}, 有效帧数 (检测到人): {valid_frames} ({valid_frames/total_frames*100:.1f}%)")

    # 1. 物理尺寸检查 (Scale Check)
    print("\n📏 [体检项目 1] 物理尺寸合理性 (单位: cm)")
    l_thigh, r_thigh, l_shin, r_shin, shoulder = calculate_limb_stats(kpts)
    
    # 过滤 NaN 计算中位数
    def get_median(arr):
        return np.nanmedian(arr)

    med_l_thigh = get_median(l_thigh)
    med_r_thigh = get_median(r_thigh)
    med_l_shin = get_median(l_shin)
    med_r_shin = get_median(r_shin)
    med_shoulder = get_median(shoulder)
    
    print(f"   - 左大腿平均长度: {med_l_thigh:.2f} cm")
    print(f"   - 右大腿平均长度: {med_r_thigh:.2f} cm")
    print(f"   - 左小腿平均长度: {med_l_shin:.2f} cm")
    print(f"   - 右小腿平均长度: {med_r_shin:.2f} cm")
    print(f"   - 平均肩宽:      {med_shoulder:.2f} cm")
    
    # 判断标准 (成年男性参考值)
    # 大腿 ~40-50cm, 小腿 ~35-45cm, 肩宽 ~35-45cm
    if 35 < med_l_thigh < 55 and 30 < med_l_shin < 50:
        print("✅ 结论: 尺寸看起来非常合理 (Scale Correct)。")
    elif med_l_thigh < 10:
        print("❌ 结论: 尺寸太小！可能单位搞错了 (比如虽然叫cm其实是m?)")
    elif med_l_thigh > 100:
        print("❌ 结论: 尺寸太大！可能是标定参数的单位问题 (mm vs cm?)")
    else:
        print("⚠️ 结论: 尺寸有点偏差，请人工确认。")

    # 2. 深度稳定性检查 (Depth Stability)
    print("\n📉 [体检项目 2] 深度数据稳定性")
    # 取躯干中心
    center = (kpts[:, 11, :] + kpts[:, 12, :]) / 2.0
    z_vals = center[:, 2]
    
    # 统计 Z 轴的跳变情况
    # 过滤 NaN
    z_clean = z_vals[~np.isnan(z_vals)]
    z_diff = np.abs(np.diff(z_clean))
    
    print(f"   - Z轴范围: {np.min(z_clean):.1f} ~ {np.max(z_clean):.1f} cm")
    print(f"   - Z轴平均跳变 (Jitter): {np.mean(z_diff):.2f} cm/frame")
    
    if np.mean(z_diff) > 10:
        print("⚠️ 警告: 深度抖动严重 (>10cm/frame)，这也是为什么之前很难对齐的原因。")
    else:
        print("✅ 深度相对平滑。")

    # 3. 可视化一帧“黄金帧”看看骨架是否畸形
    # 找一帧大腿长度最接近中位数的
    valid_idx = np.where(~np.isnan(l_thigh))[0]
    if len(valid_idx) > 0:
        diff = np.abs(l_thigh[valid_idx] - med_l_thigh)
        best_local_idx = np.argmin(diff)
        frame_idx = valid_idx[best_local_idx]
        
        print(f"\n🖼️ [体检项目 3] 骨架形态检查 (查看 Frame {frame_idx})")
        print("   正在绘制 3D 骨架图...")
        
        pts = kpts[frame_idx]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # COCO 骨架连接
        connections = [
            (0,1), (0,2), (1,3), (2,4), # Face
            (5,6), (5,7), (7,9), (6,8), (8,10), # Arms
            (5,11), (6,12), # Torso
            (11,12), (11,13), (13,15), (12,14), (14,16) # Legs
        ]
        
        # 画点
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='r', marker='o')
        
        # 画线
        for i, j in connections:
            if not (np.isnan(pts[i]).any() or np.isnan(pts[j]).any()):
                ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], [pts[i,2], pts[j,2]], 'b-')
        
        # 设置轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Skeleton (Frame {frame_idx})\nCheck if it looks like a human')
        
        # 强制比例一致，防止被拉伸
        # Create cubic bounding box to force equal aspect ratio
        max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0
        mid_x = (pts[:,0].max()+pts[:,0].min()) * 0.5
        mid_y = (pts[:,1].max()+pts[:,1].min()) * 0.5
        mid_z = (pts[:,2].max()+pts[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()

if __name__ == "__main__":
    main()