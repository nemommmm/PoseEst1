#!/usr/bin/env bash
# run_pipeline.sh
#
# Runs Aitor Iriondo's monocular pipeline on the left-camera video:
#   Stage 1 (mmpose env):  RTMDet + RTMW + MotionBERT → video_outputs.json
#   Stage 2 (mmpose env):  TRC marker generation + OpenSim IK → output.mot
#
# Output:
#   results_mono/kinematics/<video_name>.mot   — 40-DOF joint angles
#   results_mono/<video_name>.glb              — animated skeleton (optional)
#
# Prerequisites: run setup_envs.sh first.
# Usage:
#   bash run_pipeline.sh                        # default subject height 169.5 cm
#   MONO_HEIGHT_M=1.75 bash run_pipeline.sh     # custom height

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITOR_REPO="$SCRIPT_DIR/RTMDet-MotionBert-OpenSim"
VIDEO_PATH="$(dirname "$SCRIPT_DIR")/2025_Ergonomics_Data/0_video_left.avi"
OUT_DIR="$SCRIPT_DIR/results_mono"

# Subject height in metres (estimated 169.5 cm from Xsens data)
HEIGHT_M="${MONO_HEIGHT_M:-1.695}"

# Butterworth filter cutoff (Hz); Aitor's default is 6
SMOOTH="${MONO_SMOOTH:-6.0}"

# Device: cuda or cpu
DEVICE="${MONO_DEVICE:-cpu}"

mkdir -p "$OUT_DIR"

if [ ! -d "$AITOR_REPO" ]; then
    echo "[run] ERROR: Aitor's repo not found. Run setup_envs.sh first."
    exit 1
fi

echo "[run] Video:  $VIDEO_PATH"
echo "[run] Height: ${HEIGHT_M} m   Smooth: ${SMOOTH} Hz   Device: ${DEVICE}"
echo ""

# ---------------------------------------------------------------------------
# Stage 1: Inference (RTMDet → RTMW 133-kpt → MotionBERT 3D)
# Produces: video_outputs.json  (~89 s for ~1100 frames on CPU)
# ---------------------------------------------------------------------------
echo "[run] Stage 1: RTMDet + RTMW + MotionBERT inference ..."
conda run -n mmpose python "$AITOR_REPO/run_inference.py" \
    --input "$VIDEO_PATH" \
    --output-dir "$OUT_DIR" \
    --device "$DEVICE"

echo "[run] Stage 1 complete → video_outputs.json"
echo ""

# ---------------------------------------------------------------------------
# Stage 2: TRC generation + OpenSim IK  (~17 s)
# Produces: results_mono/kinematics/<name>.mot
# ---------------------------------------------------------------------------
echo "[run] Stage 2: TRC markers + OpenSim IK ..."
conda run -n mmpose python "$AITOR_REPO/run_hybrid_pipeline.py" \
    --input "$VIDEO_PATH" \
    --output-dir "$OUT_DIR" \
    --height "$HEIGHT_M" \
    --smooth "$SMOOTH"

echo "[run] Stage 2 complete → results_mono/kinematics/"
echo ""

# ---------------------------------------------------------------------------
# Locate the .mot output for the evaluation step
# ---------------------------------------------------------------------------
MOT_FILE=$(find "$OUT_DIR/kinematics" -name "*.mot" | head -1)
if [ -z "$MOT_FILE" ]; then
    echo "[run] WARNING: No .mot file found in $OUT_DIR/kinematics/"
    echo "       Check Aitor's pipeline output for errors."
else
    echo "[run] Joint angles: $MOT_FILE"
    echo ""
    echo "Next step → evaluate vs Xsens GT:"
    echo "  MONO_MOT_PATH=\"$MOT_FILE\" /opt/anaconda3/envs/pose/bin/python evaluate_vs_gt.py"
fi
