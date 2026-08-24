#!/usr/bin/env bash
#
# Step 2 - Train the Speaker Inversion embedding.
#
# Freezes the whole base model and trains only a (tokens x speaker_dim) embedding
# table. Every checkpoint this run writes is embedding-only: a few KB, named
# checkpoint_*.speaker.safetensors.
#
# Override any value by exporting it before running, e.g.:
#   SPEAKER_NAME=alice NPROC=2 BASE_CHECKPOINT=/models/v4.1-Small/model.safetensors \
#     scripts/02_train.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${HERE}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXAMPLE_DIR}/.." && pwd)"

# ---- EDIT ME -----------------------------------------------------------------
SPEAKER_NAME="${SPEAKER_NAME:-my_speaker}"              # one name per target voice
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/path/to/Irodori-TTS-v4.1-Small/model.safetensors}"
NPROC="${NPROC:-1}"                                     # >1 enables DDP via torchrun
# ------------------------------------------------------------------------------

CONFIG="${CONFIG:-${EXAMPLE_DIR}/configs/speaker_inversion.yaml}"
MANIFEST="${MANIFEST:-${EXAMPLE_DIR}/data/target_speaker_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXAMPLE_DIR}/outputs/${SPEAKER_NAME}}"

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "ERROR: base checkpoint not found: ${BASE_CHECKPOINT}" >&2
  echo "       Download Aratako/Irodori-TTS-v4.1-Small and set BASE_CHECKPOINT." >&2
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}" >&2
  echo "       Run scripts/01_prepare_manifest.sh first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

TRAIN_ARGS=(
  --config "${CONFIG}"
  --manifest "${MANIFEST}"
  --init-checkpoint "${BASE_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
)

cd "${PROJECT_ROOT}"

echo "Speaker:  ${SPEAKER_NAME}"
echo "Config:   ${CONFIG}"
echo "Base:     ${BASE_CHECKPOINT}"
echo "Output:   ${OUTPUT_DIR}"
echo

if [[ "${NPROC}" == "1" ]]; then
  echo "Launching single-process training."
  uv run --no-sync python train.py "${TRAIN_ARGS[@]}" "$@"
else
  echo "Launching DDP training on ${NPROC} processes (batch_size is per process)."
  uv run --no-sync torchrun --nproc_per_node "${NPROC}" train.py \
    "${TRAIN_ARGS[@]}" --device cuda "$@"
fi

echo
echo "Embeddings written to ${OUTPUT_DIR}:"
ls -1 "${OUTPUT_DIR}"/*.speaker.safetensors 2>/dev/null || echo "  (none found)"
echo
echo "Next: scripts/03_infer.sh  (or scripts/04_eval_snapshots.sh to compare snapshots)"
