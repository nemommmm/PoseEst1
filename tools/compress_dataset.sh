#!/usr/bin/env bash
# Compress raw stereo videos while preserving frame timing and grayscale data.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/compress_dataset.sh --mode MODE --output-root DIR VIDEO [VIDEO ...]

Modes:
  lossless   H.264 lossless encoding (libx264, CRF 0)
  visual     H.265 visually lossless encoding (libx265, CRF 18)

Input paths must be inside the repository. Output files are written below
DIR with the same relative paths and an .mkv suffix.
EOF
}

MODE=""
OUTPUT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "${MODE}" || -z "${OUTPUT_ROOT}" || $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

case "${MODE}" in
  lossless)
    CODEC_ARGS=(-c:v libx264 -crf 0 -preset medium)
    ;;
  visual)
    CODEC_ARGS=(-c:v libx265 -crf 18 -preset fast -x265-params log-level=error)
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
OUTPUT_ROOT="$(mkdir -p "${OUTPUT_ROOT}" && cd "${OUTPUT_ROOT}" && pwd)"

for input in "$@"; do
  input_abs="$(cd "$(dirname "${input}")" && pwd)/$(basename "${input}")"
  case "${input_abs}" in
    "${REPO_ROOT}"/*) ;;
    *)
      echo "Input is outside repository: ${input}" >&2
      exit 2
      ;;
  esac

  relative="${input_abs#${REPO_ROOT}/}"
  output="${OUTPUT_ROOT}/${relative%.*}.mkv"
  mkdir -p "$(dirname "${output}")"

  echo "[compress] ${MODE}: ${relative}"
  ffmpeg -hide_banner -loglevel warning -stats -y \
    -i "${input_abs}" -map 0:v:0 -an -fps_mode passthrough \
    "${CODEC_ARGS[@]}" -pix_fmt gray "${output}"

  # Packet counting verifies frame preservation without decoding every 3 MP frame.
  input_frames="$(ffprobe -v error -count_packets -select_streams v:0 \
    -show_entries stream=nb_read_packets -of csv=p=0 "${input_abs}")"
  output_frames="$(ffprobe -v error -count_packets -select_streams v:0 \
    -show_entries stream=nb_read_packets -of csv=p=0 "${output}")"
  if [[ "${input_frames}" != "${output_frames}" ]]; then
    echo "Frame-count mismatch: ${input_frames} != ${output_frames}" >&2
    exit 1
  fi
  echo "[verified] ${output_frames} frames -> ${output}"
done
