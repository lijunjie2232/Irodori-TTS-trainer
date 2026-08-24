#!/usr/bin/env bash
#
# Step 1 - Build the target-speaker manifest.
#
# Encodes the target speaker's audio into DACVAE latents and writes the JSONL
# manifest that train.py consumes.
#
# NOTE: do NOT pass --speaker-column. Speaker Inversion learns a single embedding
# for the whole run, so speaker_id is not needed, and omitting it also skips the
# dataset's same-speaker reference-concatenation work, which the model would
# ignore anyway.
#
# Override any value by exporting it before running, e.g.:
#   DATASET=myorg/voice_001 scripts/01_prepare_manifest.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${HERE}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXAMPLE_DIR}/.." && pwd)"

# ---- EDIT ME -----------------------------------------------------------------
DATASET="${DATASET:-myorg/my_target_speaker_dataset}"   # HF dataset name or local path
SPLIT="${SPLIT:-train}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
CAPTION_COLUMN="${CAPTION_COLUMN:-}"                    # leave empty to omit captions
DEVICE="${DEVICE:-cuda}"
# ------------------------------------------------------------------------------

MANIFEST="${EXAMPLE_DIR}/data/target_speaker_manifest.jsonl"
LATENT_DIR="${EXAMPLE_DIR}/data/latents"

mkdir -p "${LATENT_DIR}"

ARGS=(
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --audio-column "${AUDIO_COLUMN}"
  --text-column "${TEXT_COLUMN}"
  --output-manifest "${MANIFEST}"
  --latent-dir "${LATENT_DIR}"
  --device "${DEVICE}"
)

if [[ -n "${CAPTION_COLUMN}" ]]; then
  ARGS+=(--caption-column "${CAPTION_COLUMN}")
fi

cd "${PROJECT_ROOT}"
echo "[1/1] prepare_manifest.py  dataset=${DATASET} split=${SPLIT}"
uv run --no-sync python prepare_manifest.py "${ARGS[@]}" "$@"

echo
echo "Manifest: ${MANIFEST}"
echo "Latents:  ${LATENT_DIR}"
echo "Rows:     $(wc -l < "${MANIFEST}" | tr -d ' ')"
echo
echo "Next: scripts/02_train.sh"
