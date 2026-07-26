#!/usr/bin/env bash
set -euo pipefail

# Rebuild NVIDIA's official DeepStream 8 BodyPose3DNet reference app.

INSTALL_ROOT="${INSTALL_ROOT:-/workspace/official_nvidia}"
REPOSITORY="${INSTALL_ROOT}/deepstream_reference_apps"
APP_ROOT="${REPOSITORY}/deepstream-bodypose-3d"
REPOSITORY_URL="https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps.git"
REPOSITORY_COMMIT="a6488b5dd1752134e473e65b03bdc75b69639b98"
CUDA_VER="${CUDA_VER:-12.8}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/PoseEst1}"
PATCH_PATH="${PROJECT_ROOT}/tools/patches/deepstream_bodypose3d_ds8.patch"

if ! command -v deepstream-app >/dev/null 2>&1; then
  echo "DeepStream 8 is not installed; BodyPose3DNet is runtime_blocked." >&2
  exit 20
fi
if ! deepstream-app --version-all 2>&1 | grep -q "DeepStreamSDK 8"; then
  echo "DeepStream 8 is required by the pinned reference app." >&2
  exit 21
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential libeigen3-dev libjson-glib-dev libmosquitto1
ln -sfn eigen3/Eigen /usr/include/Eigen

mkdir -p "${INSTALL_ROOT}"
if [[ ! -d "${REPOSITORY}/.git" ]]; then
  git clone "${REPOSITORY_URL}" "${REPOSITORY}"
fi
git -C "${REPOSITORY}" fetch --depth 1 origin "${REPOSITORY_COMMIT}"
git -C "${REPOSITORY}" checkout --detach "${REPOSITORY_COMMIT}"
if git -C "${REPOSITORY}" apply --check "${PATCH_PATH}" 2>/dev/null; then
  git -C "${REPOSITORY}" apply "${PATCH_PATH}"
fi

if [[ ! -s "${APP_ROOT}/models/bodypose3dnet/bodypose3dnet_accuracy.onnx" ]] \
  || [[ ! -s "${APP_ROOT}/models/bodypose3dnet/bodypose3dnet_performance.onnx" ]] \
  || [[ ! -s "${APP_ROOT}/models/peoplenet/resnet34_peoplenet_int8.onnx" ]]; then
  (
    cd "${APP_ROOT}"
    bash download_models.sh
  )
fi

echo "0452b785a70fcd6bc5bd4069249bdfd85eb139c9e9216bcf81f89df33945d028  ${APP_ROOT}/models/bodypose3dnet/bodypose3dnet_accuracy.onnx" | sha256sum --check
echo "3baf1f39b522ab0f4fdf3203f4ba70c46c93970988256e565d3837476a692434  ${APP_ROOT}/models/bodypose3dnet/bodypose3dnet_performance.onnx" | sha256sum --check
echo "2f0b7e5e2af5a61150e19a1fc47435c863fe181a75019e6763e0a1169b26936c  ${APP_ROOT}/models/peoplenet/resnet34_peoplenet_int8.onnx" | sha256sum --check

make -C "${APP_ROOT}/sources/nvdsinfer_custom_impl_BodyPose3DNet" \
  clean all CUDA_VER="${CUDA_VER}"
make -C "${APP_ROOT}/sources" clean all CUDA_VER="${CUDA_VER}"

"${APP_ROOT}/sources/deepstream-pose-estimation-app" --help >/dev/null
git -C "${REPOSITORY}" status --short
