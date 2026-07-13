# Remote GPU Workflow / 远程 GPU 工作流

This project uses the stable local SSH alias `poseest1-runpod`. A RunPod
migration normally changes only the Direct TCP host and port; the private key
remains `~/.ssh/id_ed25519_runpod_poseest1`.

本项目固定使用本地 SSH 别名 `poseest1-runpod`。RunPod 迁移后通常只需要更新
Direct TCP 的 IP 和端口，私钥仍使用
`~/.ssh/id_ed25519_runpod_poseest1`。

## 1. Update the alias after migration / 迁移后更新别名

Edit `~/.ssh/config` without committing it to this repository:

```sshconfig
Host poseest1-runpod
    HostName <DIRECT_TCP_IP>
    Port <DIRECT_TCP_PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519_runpod_poseest1
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 4
```

If the new container does not accept the key, use the RunPod proxy command
shown in the Pod page once, then append the existing public key to
`/root/.ssh/authorized_keys`. Do not generate a new key for each migration.

如果新容器尚未接受该公钥，先用 Pod 页面显示的代理 SSH 命令连接一次，把现有
`~/.ssh/id_ed25519_runpod_poseest1.pub` 幂等加入
`/root/.ssh/authorized_keys`。每次迁移不需要生成新密钥。

Verify the alias:

```bash
ssh -o BatchMode=yes poseest1-runpod \
  'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

## 2. Rebuild the persistent environment / 重建持久环境

The repository and virtual environment live on `/workspace`:

```bash
ssh poseest1-runpod \
  'cd /workspace/PoseEst1 && bash tools/setup_remote_env.sh'
```

EasyMocap is prepared separately because its official SMPL assets are
license-controlled:

```bash
ssh poseest1-runpod \
  'cd /workspace/PoseEst1 && bash tools/setup_easymocap_remote.sh'
```

The second command is expected to exit with code 2 until the user places the
official `SMPL_NEUTRAL.pkl` at `/workspace/model_assets/smpl/`. Follow
[`smpl_asset_setup.md`](smpl_asset_setup.md); no project tool downloads this
asset automatically.

## 3. Run and download a standard bundle / 运行并下载标准结果包

```bash
/opt/anaconda3/envs/pose/bin/python tools/remote_experiment.py run \
  --candidate smpl \
  --config 00_pose_pipeline_v2/configs/human_prior/fanbo7_a257.yaml \
  --gate feasibility \
  --sync-profile standard
```

The command performs SSH/GPU preflight, runs the remote gate, builds a SHA256
manifest, downloads the standard bundle with resumable `rsync`, and verifies
the local files. Raw videos, caches, weights, and all `.pkl` assets are
excluded.

该命令会自动完成 SSH/GPU 检查、远程门槛测试、SHA256 清单生成、可断点续传的
结果下载以及本地校验。原始视频、缓存、权重和所有 `.pkl` 模型不会进入结果包。

Resume an interrupted transfer:

```bash
/opt/anaconda3/envs/pose/bin/python tools/remote_experiment.py sync \
  --run-tag <RUN_TAG>
```

Verify an already downloaded bundle without connecting to the GPU:

```bash
/opt/anaconda3/envs/pose/bin/python tools/remote_experiment.py verify \
  --run-tag <RUN_TAG>
```

The rejected `kinematic` candidate is intentionally blocked by the runner so
that a failed adapter is not silently rerun. Its retained evidence is indexed
in `00_pose_pipeline_v2/results/experiment_log.md`.

已拒绝的 `kinematic` 候选会被编排器主动阻止，避免再次运行已经失败并回滚的
适配器；其负结果证据记录在 `00_pose_pipeline_v2/results/experiment_log.md`。
