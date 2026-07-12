#!/usr/bin/env bash
# Build a persistent GPU environment for the Runpod pose pipeline.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venv-pose}"
BASE_PYTHON="${BASE_PYTHON:-python}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_PATH}"
fi

PYTHON="${VENV_PATH}/bin/python"
"${PYTHON}" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"${PYTHON}" -m pip install --no-cache-dir -r "${REPO_ROOT}/requirements.txt"
"${PYTHON}" -m pip install --no-cache-dir --no-deps "rtmlib>=0.0.13"

YOLO_CONFIG_DIR="${VENV_PATH}/.config/Ultralytics" "${PYTHON}" - <<'PY'
import cv2
import onnxruntime as ort
import torch

providers = ort.get_available_providers()
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"opencv={cv2.__version__}")
print(f"onnxruntime_providers={providers}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("ONNX Runtime CUDAExecutionProvider is unavailable")
PY
