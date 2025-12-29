import cv2
import time
import sys
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ================= 配置区域 =================
# 视频路径 (请确保这些文件存在)
VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "2025_Ergonomics_Data", "0_video_left.avi")
OUTPUT_FILE = "2d_model_comparison.mp4"
CONF_THRESHOLD = 0.5
# ===========================================

class YOLOWrapper:
    def __init__(self, model_name):
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        self.name = model_name.split('.')[0]
        self.color = (0, 255, 0) # Green

    def process(self, frame):
        start = time.time()
        # verbose=False 关闭控制台刷屏
        results = self.model(frame, verbose=False, conf=CONF_THRESHOLD)[0]
        dt = time.time() - start
        
        # 绘制结果
        annotated_frame = results.plot()
        return annotated_frame, dt

class MediaPipeWrapper:
    def __init__(self):
        print("Loading MediaPipe...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # 0=Lite, 1=Full, 2=Heavy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.name = "MediaPipe (Google)"
        self.color = (255, 100, 0) # Blueish

    def process(self, frame):
        start = time.time()
        # MediaPipe 需要 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        dt = time.time() - start

        # 绘制
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
            )
        return annotated_frame, dt

def draw_fps(image, name, dt, color):
    fps = 1.0 / (dt + 1e-5)
    text = f"{name} | FPS: {fps:.1f} | {dt*1000:.1f}ms"
    
    # 加上黑色背景条，看得更清楚
    cv2.rectangle(image, (0, 0), (450, 40), (0,0,0), -1)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 找不到视频: {VIDEO_PATH}")
        return

    # 1. 初始化模型
    # 对比 YOLOv8-Nano (极快) vs YOLOv8-Medium (均衡) vs MediaPipe (不同架构)
    models = [
        YOLOWrapper('yolov8n-pose.pt'),
        YOLOWrapper('yolov8m-pose.pt'),
        MediaPipeWrapper()
    ]

    # 2. 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 准备输出视频 (把三个画面横向拼接)
    # 输出宽度 = 原宽 * 3 * 0.5 (我们缩放一下，不然太宽了)
    target_w = int(width * 0.5)
    target_h = int(height * 0.5)
    out_w = target_w * 3
    out_h = target_h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (out_w, out_h))

    print(f"🚀 开始测试! 视频总帧数: {total_frames}")
    print("按 'q' 提前结束...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # ⚠️ 关键修正：旋转 180 度
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # 依次运行模型
        results_imgs = []
        for model in models:
            res_img, dt = model.process(frame)
            # 缩放以节省显存和屏幕空间
            res_img = cv2.resize(res_img, (target_w, target_h))
            draw_fps(res_img, model.name, dt, model.color)
            results_imgs.append(res_img)

        # 拼接画面
        combined = np.hstack(results_imgs)
        
        # 显示 & 保存
        cv2.imshow('2D Model Benchmark', combined)
        out.write(combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_idx += 1
        print(f"\rProcess: {frame_idx}/{total_frames}", end="")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 完成! 结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()