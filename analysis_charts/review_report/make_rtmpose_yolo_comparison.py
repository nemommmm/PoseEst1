"""RTMPose-x vs YOLOv8m per-joint MAE comparison — key decision point chart."""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

joints = ['R.Shoulder', 'L.Shoulder', 'L.Hip', 'R.Hip', 'L.Knee', 'R.Knee', 'L.Elbow', 'R.Elbow']

yolo_uncalib  = [15.2, 20.0, 16.6, 18.6, 21.3, 19.4, 28.3, 30.7]
yolo_calib    = [13.7, 13.3, 10.0,  8.4, 14.3, 13.1, 16.0, 16.2]
rtm_remapped  = [15.3, 17.2, 26.5, 26.6, 49.9, 57.6, 36.3, 30.2]

x = np.arange(len(joints))
w = 0.26

fig, ax = plt.subplots(figsize=(13, 5))
b1 = ax.bar(x - w,   yolo_uncalib, w, label='YOLOv8m (uncalibrated)', color='#4878CF', alpha=0.85)
b2 = ax.bar(x,        yolo_calib,   w, label='YOLOv8m + angle calib ✓', color='#16A34A', alpha=0.90)
b3 = ax.bar(x + w,   rtm_remapped,  w, label='RTMPose-x (remapped) ✗', color='#D65F5F', alpha=0.85)

ax.axhline(13.21, color='#16A34A', linestyle='--', linewidth=1.2, alpha=0.7, label='13.21° calibrated best')

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h:.0f}°', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(joints, fontsize=10)
ax.set_ylabel('Joint Angle MAE (degrees)', fontsize=11)
ax.set_title('2D Detector Comparison: YOLOv8m vs RTMPose-x\n'
             'RTMPose-x knee joints catastrophically fail due to keypoint convention mismatch',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 72)
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Highlight knee failure
for i in [4, 5]:
    ax.annotate('Convention\nmismatch', xy=(i + w, rtm_remapped[i]),
                xytext=(i + w + 0.05, rtm_remapped[i] + 5),
                fontsize=7.5, color='#DC2626',
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.2))

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'rtmpose_yolo_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
