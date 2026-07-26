#!/usr/bin/env bash
set -euo pipefail

# Rebuild the isolated runtime and official checkpoints used by the
# FoundationStereo comparison route.

INSTALL_ROOT="${INSTALL_ROOT:-/workspace/official_nvidia}"
VENV_PATH="${VENV_PATH:-/workspace/venv-foundation-stereo}"
BASE_PYTHON="${BASE_PYTHON:-python}"

FOUNDATION_REPO="${INSTALL_ROOT}/FoundationStereo"
FAST_REPO="${INSTALL_ROOT}/Fast-FoundationStereo"
FOUNDATION_COMMIT="6e8806816b533e4d13ddbb95ffa907b797060a62"
FAST_COMMIT="a290ba04c1b3ad1ec41a33974a157b2917b624d4"

clone_at_commit() {
  local repository_url="$1"
  local destination="$2"
  local commit="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${repository_url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
}

"${BASE_PYTHON}" -m venv --system-site-packages "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install \
  omegaconf timm einops scikit-image opencv-contrib-python imageio \
  trimesh transformations albumentations joblib scikit-learn \
  ruamel.yaml huggingface-hub imgaug ninja open3d gdown

mkdir -p "${INSTALL_ROOT}"
clone_at_commit \
  "https://github.com/NVlabs/FoundationStereo.git" \
  "${FOUNDATION_REPO}" \
  "${FOUNDATION_COMMIT}"
clone_at_commit \
  "https://github.com/NVlabs/Fast-FoundationStereo.git" \
  "${FAST_REPO}" \
  "${FAST_COMMIT}"

FOUNDATION_WEIGHT_DIR="${FOUNDATION_REPO}/pretrained_models/23-51-11"
FAST_WEIGHT_DIR="${FAST_REPO}/weights/23-36-37"
mkdir -p "${FOUNDATION_WEIGHT_DIR}" "${FAST_WEIGHT_DIR}"

download_if_missing() {
  local file_id="$1"
  local output_path="$2"
  if [[ ! -s "${output_path}" ]]; then
    "${VENV_PATH}/bin/gdown" "${file_id}" --output "${output_path}"
  fi
}

download_if_missing \
  "1tidGICH1_kTUUqi42aboKscuMY4IK_Xr" \
  "${FOUNDATION_WEIGHT_DIR}/cfg.yaml"
download_if_missing \
  "1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe" \
  "${FOUNDATION_WEIGHT_DIR}/model_best_bp2.pth"
download_if_missing \
  "1GDBRYL-ZaLpXEtWfGFRJvkBc_2sywjgj" \
  "${FAST_WEIGHT_DIR}/cfg.yaml"
download_if_missing \
  "1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E" \
  "${FAST_WEIGHT_DIR}/model_best_bp2_serialize.pth"

echo "a9d9dd2137c30edc2236194f62df14d222dad5fd3287a33c7540b543bb93853f  ${FOUNDATION_WEIGHT_DIR}/cfg.yaml" | sha256sum --check
echo "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1  ${FOUNDATION_WEIGHT_DIR}/model_best_bp2.pth" | sha256sum --check
echo "d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc  ${FAST_WEIGHT_DIR}/cfg.yaml" | sha256sum --check
echo "af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990  ${FAST_WEIGHT_DIR}/model_best_bp2_serialize.pth" | sha256sum --check

"${VENV_PATH}/bin/python" - <<'PY'
import cv2
import open3d
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"opencv={cv2.__version__}")
print(f"open3d={open3d.__version__}")
PY
