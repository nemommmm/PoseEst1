#!/usr/bin/env bash
set -euo pipefail

# Losslessly concatenate Fanbo9 split stereo videos and metadata files.
# Output stays under 2026_Assar_Data, which is git-ignored project data.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

concat_pair() {
  local camera="$1"
  local first_cap="$2"
  local second_cap="$3"
  local merged_name="$4"

  local video_dir="2026_Assar_Data/${camera}/Video"
  local out_dir="${video_dir}/merged"
  mkdir -p "${out_dir}"

  for side in 0 1; do
    local list_file
    list_file="$(mktemp "/tmp/fanbo9_${camera}_${side}.XXXXXX.txt")"
    printf "file '%s'\nfile '%s'\n" \
      "${ROOT_DIR}/${video_dir}/cap_${first_cap}_${side}.avi" \
      "${ROOT_DIR}/${video_dir}/cap_${second_cap}_${side}.avi" \
      > "${list_file}"

    ffmpeg -hide_banner -loglevel error -y \
      -f concat -safe 0 -i "${list_file}" \
      -c copy "${out_dir}/${merged_name}_${side}.avi"

    rm -f "${list_file}"

    cat \
      "${video_dir}/cap_${first_cap}_${side}.txt" \
      "${video_dir}/cap_${second_cap}_${side}.txt" \
      > "${out_dir}/${merged_name}_${side}.txt"
  done
}

concat_pair "A255" "9" "10" "cap_9-10"
concat_pair "A257" "7" "8" "cap_7-8"

echo "Fanbo9 merged stereo videos saved under:"
echo "  2026_Assar_Data/A255/Video/merged/"
echo "  2026_Assar_Data/A257/Video/merged/"
