#!/usr/bin/env bash
#
# Step 3 - Synthesise with the trained embedding.
#
# --ref-embed replaces reference audio. It is mutually exclusive with
# --ref-wav / --ref-wavs / --ref-latent / --ref-latents / --no-ref.
#
# IMPORTANT: use the SAME base checkpoint you trained the embedding against.
#
# Extra flags are passed straight through, e.g.:
#   TEXT="おはようございます。" scripts/03_infer.sh --seed 0 --cfg-scale-speaker 6.0

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${HERE}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXAMPLE_DIR}/.." && pwd)"

# ---- EDIT ME -----------------------------------------------------------------
SPEAKER_NAME="${SPEAKER_NAME:-my_speaker}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/path/to/Irodori-TTS-v4.1-Small/model.safetensors}"
TEXT="${TEXT:-こんにちは、これは学習した話者埋め込みを使った推論です。}"
# ------------------------------------------------------------------------------

EMBED="${EMBED:-${EXAMPLE_DIR}/outputs/${SPEAKER_NAME}/checkpoint_final.speaker.safetensors}"
OUTPUT_WAV="${OUTPUT_WAV:-${EXAMPLE_DIR}/outputs/${SPEAKER_NAME}/sample.wav}"

if [[ ! -f "${EMBED}" ]]; then
  echo "ERROR: embedding not found: ${EMBED}" >&2
  echo "       Run scripts/02_train.sh first, or set EMBED to another snapshot." >&2
  exit 1
fi

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "ERROR: base checkpoint not found: ${BASE_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_WAV}")"

cd "${PROJECT_ROOT}"
uv run --no-sync python infer.py \
  --checkpoint "${BASE_CHECKPOINT}" \
  --ref-embed "${EMBED}" \
  --text "${TEXT}" \
  --output-wav "${OUTPUT_WAV}" \
  "$@"

echo
echo "Wrote ${OUTPUT_WAV}"
