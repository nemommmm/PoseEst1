#!/usr/bin/env bash
# setup_envs.sh — One-time setup for Aitor's monocular pipeline
# Reference: https://github.com/AitorIriondo/RTMDet-MotionBert-OpenSim
#
# Two conda environments:
#   mmpose   (Python 3.10) — RTMW inference + MotionBERT + TRC generation
#   Pose2Sim (Python 3.11) — OpenSim 4.5.1 IK (3.12 not yet supported by opensim-org)
#
# Mac-specific: uses CPU-only PyTorch (no NVIDIA GPU on macOS)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITOR_REPO="$SCRIPT_DIR/RTMDet-MotionBert-OpenSim"

echo "[setup] Aitor repo: $AITOR_REPO"

# ---------------------------------------------------------------------------
# 1. mmpose environment
# ---------------------------------------------------------------------------
if conda env list | grep -q "^mmpose "; then
    echo "[setup] conda env 'mmpose' already exists"
else
    echo "[setup] Creating mmpose env (Python 3.10) ..."
    conda create -n mmpose python=3.10 -y

    echo "[setup] Installing CPU-only PyTorch ..."
    conda run -n mmpose pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu -q

    echo "[setup] Installing mmpose ecosystem ..."
    conda run -n mmpose pip install mmengine -q
    conda run -n mmpose pip install mmcv -q
    conda run -n mmpose pip install mmdet -q
    conda run -n mmpose pip install mmpose -q

    echo "[setup] Installing rtmpose3d ..."
    conda run -n mmpose pip install rtmpose3d -q || \
        conda run -n mmpose pip install git+https://github.com/b-arac/rtmpose3d.git -q

    echo "[setup] Installing project requirements ..."
    conda run -n mmpose pip install -r "$AITOR_REPO/requirements.txt" -q || true

    echo "[setup] mmpose env ready."
fi

# ---------------------------------------------------------------------------
# 2. Pose2Sim environment (OpenSim IK)
# Note: opensim-org conda package supports up to Python 3.11 (not 3.12)
# ---------------------------------------------------------------------------
if conda env list | grep -q "^Pose2Sim "; then
    echo "[setup] conda env 'Pose2Sim' already exists"
else
    echo "[setup] Creating Pose2Sim env (Python 3.11) ..."
    conda create -n Pose2Sim python=3.11 -y

    echo "[setup] Installing OpenSim 4.5.1 ..."
    conda run -n Pose2Sim conda install -c opensim-org opensim=4.5.1 -y

    echo "[setup] Installing Pose2Sim ..."
    conda run -n Pose2Sim pip install pose2sim -q

    echo "[setup] Pose2Sim env ready."
fi

# ---------------------------------------------------------------------------
# 3. Fix Pose2Sim marker bug (required for pose2sim >= 0.10.33)
# ---------------------------------------------------------------------------
echo "[setup] Applying Pose2Sim marker fix ..."
conda run -n Pose2Sim python "$AITOR_REPO/fix_pose2sim.py" || \
    echo "[setup] fix_pose2sim.py skipped (check manually if IK output looks wrong)"

# ---------------------------------------------------------------------------
# 4. MotionBERT checkpoint
# ---------------------------------------------------------------------------
CKPT_DIR="$AITOR_REPO/models/MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite"
CKPT_FILE="$CKPT_DIR/best_epoch.bin"
mkdir -p "$CKPT_DIR"

if [ ! -f "$CKPT_FILE" ]; then
    echo "[setup] Downloading MotionBERT-Lite checkpoint ..."
    conda run -n mmpose pip install gdown -q
    conda run -n mmpose python -m gdown \
        "https://drive.google.com/uc?id=1O_1jMeINpb12BPi1NbkqgdYQ1f_9RVVG" \
        -O "$CKPT_FILE"
else
    echo "[setup] MotionBERT checkpoint already present: $CKPT_FILE"
fi

echo ""
echo "[setup] ✓ All done. Run the pipeline:"
echo "   bash run_pipeline.sh"
