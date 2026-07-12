#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_ROOT="${EXTERNAL_ROOT:-/workspace/external}"
ASSET_ROOT="${MODEL_ASSET_ROOT:-/workspace/model_assets/metrabs}"
PYTHON_BIN="${POSE_PYTHON:-/workspace/venv-pose/bin/python}"
DEPS="${EXTERNAL_ROOT}/metrabs_deps"

mkdir -p "${EXTERNAL_ROOT}" "${ASSET_ROOT}" "${DEPS}"
if [[ ! -d "${EXTERNAL_ROOT}/metrabs/.git" ]]; then
  git clone https://github.com/isarandi/metrabs.git "${EXTERNAL_ROOT}/metrabs"
fi
git -C "${EXTERNAL_ROOT}/metrabs" checkout 8b2b116dd27372e7dbd8207809f868df4e3f852e

"${PYTHON_BIN}" -m pip install --target "${DEPS}" --no-deps \
  addict einops 'hydra-core==1.3.2' 'omegaconf==2.3.0' 'antlr4-python3-runtime==4.9.3' \
  numba llvmlite more_itertools transforms3d yacs crc32c zstandard msgpack \
  git+https://github.com/isarandi/cameralib.git \
  git+https://github.com/isarandi/boxlib.git \
  git+https://github.com/isarandi/simplepyutils.git \
  git+https://github.com/isarandi/posepile.git \
  git+https://github.com/isarandi/BareCat.git

for archive in \
  metrabs_eff2s_256px_800k_28ds_pytorch.tar.gz \
  metrabs_eff2l_384px_800k_28ds_pytorch.tar.gz; do
  model_dir="${ASSET_ROOT}/${archive%.tar.gz}"
  if [[ ! -s "${model_dir}/ckpt.pt" ]]; then
    curl -L "https://omnomnom.vision.rwth-aachen.de/data/metrabs/${archive}" \
      | tar --no-same-owner -xz -C "${ASSET_ROOT}"
  fi
done

echo "MeTRAbs PyTorch inference assets ready under ${ASSET_ROOT}"
