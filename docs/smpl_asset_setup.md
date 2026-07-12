# SMPL Academic Asset Setup / SMPL 学术模型文件配置

The SMPL body-model parameters are license-controlled and are not distributed
by PoseEst1, EasyMocap, or the setup scripts. They must never be committed to
Git or copied into a public report artifact.

SMPL 人体模型参数受学术许可约束，PoseEst1、EasyMocap 和自动安装脚本都不会分发该文件。
模型文件不得提交到 Git，也不得打包进公开报告。

## 1. Download / 下载

1. Register at the official [SMPL website](https://smpl.is.tue.mpg.de/).
2. Accept the research license and download **SMPL for Python**.
3. Extract the neutral model. Depending on the package version, the source file
   is normally named either:
   - `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`, or
   - `SMPL_NEUTRAL.pkl`.

在 SMPL 官方网站注册并接受研究许可，下载 **SMPL for Python**。解压后找到 neutral
模型文件。不要从非官方网盘或未知 GitHub 仓库下载模型参数。

## 2. Local placement / 本地放置

Create the private directory below and copy the neutral model without changing
its contents:

```bash
mkdir -p model_assets/smpl
cp /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
  model_assets/smpl/SMPL_NEUTRAL.pkl
```

The whole `model_assets/` directory and common SMPL filenames are gitignored.

## 3. Remote placement / 上传到远程持久卷

```bash
ssh poseest1-runpod 'mkdir -p /workspace/model_assets/smpl'
rsync -avP model_assets/smpl/SMPL_NEUTRAL.pkl \
  poseest1-runpod:/workspace/model_assets/smpl/SMPL_NEUTRAL.pkl
```

The remote path is on the persistent `/workspace` volume. Do not place the
licensed asset inside `/workspace/PoseEst1`.

## 4. Validate / 验证

```bash
ssh poseest1-runpod \
  'cd /workspace/PoseEst1 && \
   PYTHONPATH=/workspace/external/easymocap_deps:/workspace/external/EasyMocap \
   /workspace/venv-pose/bin/python \
   00_pose_pipeline_v2/src/validate_smpl_asset.py \
   --model /workspace/model_assets/smpl/SMPL_NEUTRAL.pkl \
   --regressor /workspace/external/EasyMocap/data/smplx/J_regressor_body25.npy'
```

Successful validation prints model metadata and a zero-pose CUDA forward-pass
summary. It does not upload, modify, or calibrate the model.

验证成功会输出模型元数据和一次零姿态 CUDA forward 摘要。该步骤不会上传、修改或用
Xsens 校准人体模型。
