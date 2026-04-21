"""Calibration evolution chart: Stage 1 initial → Stage 1 refined → Stage 2 optimized."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

stages = ['Stage 1\nInitial', 'Stage 1\nRefined', 'Stage 2\nOptimized\n(current)']

baseline_cm  = [47.0,  40.48, 41.27]
reproj_rms   = [1.30,  0.337, 0.024]   # px (in-frame reprojection)
epipolar     = [None,  0.329, 0.255]   # px vertical disparity (estimated combined mean)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Stereo Calibration Evolution Across Stages', fontsize=13, fontweight='bold')

colors = ['#D65F5F', '#F59E0B', '#16A34A']
x = np.arange(len(stages))

# --- Left: Baseline ---
ax = axes[0]
bars = ax.bar(x, baseline_cm, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, baseline_cm):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.2f} cm', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(41.0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Physical ≈ 41 cm')
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel('Stereo Baseline (cm)', fontsize=11)
ax.set_title('Baseline Distance', fontsize=12)
ax.set_ylim(35, 52)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Right: Reprojection error ---
ax2 = axes[1]
# Use log scale because range is huge (1.3 → 0.024)
valid_repr = reproj_rms
bars2 = ax2.bar(x, valid_repr, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars2, valid_repr):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f} px', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_yscale('log')
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='0.5 px threshold (good)')
ax2.set_xticks(x); ax2.set_xticklabels(stages, fontsize=10)
ax2.set_ylabel('Reprojection Error (px, log scale)', fontsize=11)
ax2.set_title('Reprojection Error\n(lower = better calibration)', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.text(0.5, -0.04,
         'Stage 2 optimized: grid search (threshold=0.35px, rational model, cross-validation). '
         'Current camera_params.npz = Stage 2 result.',
         ha='center', fontsize=9, color='#555', style='italic')

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'calibration_evolution.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
