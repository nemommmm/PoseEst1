#!/usr/bin/env bash
# Rebuild NVIDIA's official DeepStream BodyPose3DNet reference app.

set -euo pipefail

APP_PARENT="${APP_PARENT:-/workspace/official_nvidia/deepstream_reference_apps}"
APP_ROOT="${APP_PARENT}/deepstream-bodypose-3d"
APP_COMMIT="${APP_COMMIT:-a6488b5dd1752134e473e65b03bdc75b69639b98}"
CUDA_VER="${CUDA_VER:-12.8}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/PoseEst1}"
PATCH_PATH="${PROJECT_ROOT}/tools/patches/deepstream_bodypose3d_ds8.patch"

if [[ ! -d "${APP_PARENT}/.git" ]]; then
  git clone \
    https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps.git \
    "${APP_PARENT}"
fi
git -C "${APP_PARENT}" fetch --all --tags
git -C "${APP_PARENT}" checkout "${APP_COMMIT}"
if git -C "${APP_PARENT}" apply --check "${PATCH_PATH}" 2>/dev/null; then
  git -C "${APP_PARENT}" apply "${PATCH_PATH}"
fi

if [[ ! -f "${APP_ROOT}/models/bodypose3dnet/bodypose3dnet_accuracy.onnx" ]]; then
  (
    cd "${APP_ROOT}"
    bash ./download_models.sh
  )
fi

make \
  -C "${APP_ROOT}/sources/nvdsinfer_custom_impl_BodyPose3DNet" \
  -j2 \
  "CUDA_VER=${CUDA_VER}"
make -C "${APP_ROOT}/sources" -j2 "CUDA_VER=${CUDA_VER}"

sha256sum \
  "${APP_ROOT}/sources/deepstream-pose-estimation-app" \
  "${APP_ROOT}/sources/nvdsinfer_custom_impl_BodyPose3DNet/libnvdsinfer_custom_impl_BodyPose3DNet.so" \
  "${APP_ROOT}/models/peoplenet/resnet34_peoplenet_int8.onnx" \
  "${APP_ROOT}/models/bodypose3dnet/bodypose3dnet_accuracy.onnx" \
  "${APP_ROOT}/models/bodypose3dnet/bodypose3dnet_performance.onnx"
