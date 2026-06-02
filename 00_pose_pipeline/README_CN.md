# 00_pose_pipeline

这是一个独立整理出来的 end-to-end 姿态评估流程，用于把新数据集从 **双目 SKT 推理** 一直跑到 **角度评估** 和 **K-frame motion delta 评估**。

## 快速运行

当前数据集复现：

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/run_pipeline.py \
  --config 00_pose_pipeline/configs/current_2025_ergonomics.yaml \
  --stages validate,offset,angle,motion,segment,scatter
```

如果新数据集还没有 SKT NPZ，则运行：

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/run_pipeline.py \
  --config 00_pose_pipeline/configs/template_new_dataset.yaml \
  --stages validate,skt,offset,angle,motion
```

## 阶段说明

| Stage | 作用 |
|---|---|
| `validate` | 检查路径、左右 metadata、同步帧数、fps、duration。 |
| `skt` | 独立运行 sparse keypoint triangulation，输出 SKT NPZ。 |
| `offset` | 自动搜索 video timeline 与 Xsens timeline 的时间 offset。 |
| `angle` | 传统角度评估，输出 MAE、bias、RULA-like agreement。 |
| `motion` | K-frame delta 评估，输出 motion agreement、high-delta、path ratio。 |
| `segment` | 活动片段 ROM、DTW、RULA-like agreement。 |
| `scatter` | 生成 K-frame delta scatter plots。 |
| `video` | 生成原视频 + skeleton 对比视频。 |

## 重要文件

- `configs/current_2025_ergonomics.yaml`：当前数据集配置。
- `configs/template_new_dataset.yaml`：新数据集模板。
- `docs/dataset_parameters_CN.md`：哪些参数是当前 dataset 特例、哪些新数据集必须检查。
- `runs/<dataset>/alignment_summary.json`：自动 offset 结果。

## Xsens 表述原则

Xsens 在这里是 comparison/reference system，不是 absolute ground truth。报告里建议写 `Xsens-derived reference` 或 `agreement with Xsens comparison system`。
