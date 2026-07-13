#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_ROOT="${EXTERNAL_ROOT:-/workspace/external}"
DEPS="${EXTERNAL_ROOT}/easymocap_deps"
PYTHON_BIN="${POSE_PYTHON:-/workspace/venv-pose/bin/python}"

mkdir -p "${EXTERNAL_ROOT}" "${DEPS}" /workspace/model_assets/smpl
if [[ ! -d "${EXTERNAL_ROOT}/EasyMocap/.git" ]]; then
  git clone https://github.com/zju3dv/EasyMocap.git "${EXTERNAL_ROOT}/EasyMocap"
fi
git -C "${EXTERNAL_ROOT}/EasyMocap" checkout e6813197809936ca3353693f7c059025d295b4aa

"${PYTHON_BIN}" -m pip install --target "${DEPS}" --no-deps \
  func_timeout ipdb joblib tabulate termcolor yacs
"${PYTHON_BIN}" -m pip install --target "${DEPS}" --no-deps \
  --no-build-isolation \
  git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da

PYTHONPATH="${DEPS}:${EXTERNAL_ROOT}/EasyMocap" "${PYTHON_BIN}" -c \
  'from easymocap.bodymodel.smpl import SMPLModel; print("EasyMocap SMPL module ready")'

if [[ ! -s /workspace/model_assets/smpl/SMPL_NEUTRAL.pkl ]]; then
  echo "Licensed asset missing: /workspace/model_assets/smpl/SMPL_NEUTRAL.pkl"
  echo "Follow docs/smpl_asset_setup.md; the setup script will not download licensed models."
  exit 2
fi
