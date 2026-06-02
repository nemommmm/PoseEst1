# 00_pose_pipeline 数据集参数说明

这个文件夹的目标是把 **SKT 推理 + 时间同步 + 角度评估 + K-frame motion 评估** 整理成一个独立单元。拿到新数据集时，优先复制 `configs/template_new_dataset.yaml`，只改里面的路径和少数 dataset-specific 参数。

## 当前数据集保留参数

| 参数 | 当前值 | 为什么保留 |
|---|---:|---|
| `dataset.rotate_180` | `true` | 当前左右视频读入后人物是倒置的，SKT/YOLO 前需要旋转 180°。 |
| `dataset.sync_by` | `hardware_frame_id` | 左右相机用 metadata 中的硬件 frame id 配对，遇到缺帧时跳过较小 id 的一侧。 |
| `dataset.timestamp_format` | `seconds_microseconds_columns` | metadata 第 2/3 列分别表示秒和微秒，应解析为 `seconds + microseconds * 1e-6`。 |
| `offset.initial_reference_seconds` | `17.25` | 这是旧流程中人工粗找 + 算法细搜得到的参考值，只用于 sanity check，不作为强制 offset。 |
| `evaluation.camera_smooth_window_ms` | `200` | 相机/视觉方法先做 200 ms moving average，再算 motion delta。 |
| `evaluation.xsens_extra_smoothing` | `false` | Xsens 本身已有内部滤波，不再叠加项目 smoothing。 |
| `evaluation.skt_quality_filter` | `enabled` | 复用 05-18 评估规范：用 triangulation confidence、epipolar error、reprojection error 过滤 SKT 上肢关键点。 |

## 新数据集必须检查

1. 视频是否需要旋转：先抽一帧看人物方向，决定 `rotate_180`。
2. metadata 列含义是否一致：确认 frame id、秒、微秒分别在哪几列。
3. 左右相机是否能按 frame id 配对：运行 `validate` 看 synchronized pair 数量和 dropped/skipped 数量。
4. camera calibration 是否仍适用：如果相机位置或内参变化，必须换 `camera_params`。
5. Xsens offset 必须重新自动搜索：不要直接沿用当前数据集的 `17.25s`。
6. FastSAM3D / Merge TRC 对齐方式：确认它是 synced frame、left frame，还是 TRC time column 对齐。

## 推荐运行顺序

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/run_pipeline.py \
  --config 00_pose_pipeline/configs/current_2025_ergonomics.yaml \
  --stages validate,offset,angle,motion,segment,scatter
```

如果是新数据集，且还没有 SKT NPZ：

```bash
/opt/anaconda3/envs/pose/bin/python 00_pose_pipeline/src/run_pipeline.py \
  --config 00_pose_pipeline/configs/template_new_dataset.yaml \
  --stages validate,skt,offset,angle,motion
```

## 输出解释

- `alignment_summary.json`：自动 offset 搜索结果，后续 angle/motion evaluation 都读取 `selected_offset_seconds`。
- `angle_summary.csv/json`：传统角度评估，适合说明绝对角度 agreement。
- `motion_delta_summary.json`：K-frame delta 评估，适合说明动作变化趋势 agreement。
- `segment_summary.csv/json`：活动片段 ROM、DTW、RULA-like agreement。
- `scatter/`：K-frame delta scatter plots。

## 表述原则

Xsens 在本项目里是 external comparison/reference system，不是 absolute ground truth。报告中建议写：

- `agreement with Xsens-derived reference`
- `comparison against the Xsens comparison system`
- `Xsens-derived geometric reference`

避免写：

- `ground truth error`
- `validated against ground truth`
