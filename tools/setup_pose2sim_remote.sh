#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_ROOT="${EXTERNAL_ROOT:-/workspace/external}"
VENV="${POSE2SIM_VENV:-/workspace/venv-pose2sim}"

mkdir -p "${EXTERNAL_ROOT}"
if [[ ! -d "${EXTERNAL_ROOT}/Pose2Sim/.git" ]]; then
  git clone --depth 1 https://github.com/perfanalytics/pose2sim.git "${EXTERNAL_ROOT}/Pose2Sim"
fi
python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/pip" install -e "${EXTERNAL_ROOT}/Pose2Sim"
"${VENV}/bin/python" -c "import opensim, Pose2Sim; print('Pose2Sim and OpenSim ready')"
