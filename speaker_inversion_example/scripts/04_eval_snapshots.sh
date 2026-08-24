#!/usr/bin/env bash
#
# Optional - Compare training snapshots.
#
# Speaker Inversion writes a tiny checkpoint every save_every steps, so it is
# cheap to generate the same sentence from every snapshot and pick the one that
# sounds best. The RF loss is only a proxy; listening is the real metric.
#
# All samples use the same text and the same seed, so differences come from the
# embedding alone.
#
# Usage:
#   scripts/04_eval_snapshots.sh                  # first eval text
#   LINE=3 scripts/04_eval_snapshots.sh           # 3rd non-comment eval text
#   TEXT="おはようございます。" scripts/04_eval_snapshots.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${HERE}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXAMPLE_DIR}/.." && pwd)"

# ---- EDIT ME -----------------------------------------------------------------
SPEAKER_NAME="${SPEAKER_NAME:-my_speaker}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/path/to/Irodori-TTS-v4.1-Small/model.safetensors}"
SEED="${SEED:-0}"                                       # fixed, so samples are comparable
LINE="${LINE:-1}"                                       # which eval text to use
# ------------------------------------------------------------------------------

EMBED_DIR="${EMBED_DIR:-${EXAMPLE_DIR}/outputs/${SPEAKER_NAME}}"
OUT_DIR="${OUT_DIR:-${EMBED_DIR}/snapshots}"
TEXT_FILE="${TEXT_FILE:-${EXAMPLE_DIR}/texts/eval_texts.txt}"

if [[ -z "${TEXT:-}" ]]; then
  if [[ ! -f "${TEXT_FILE}" ]]; then
    echo "ERROR: eval text file not found: ${TEXT_FILE}" >&2
    exit 1
  fi
  TEXT="$(grep -v '^[[:space:]]*#' "${TEXT_FILE}" | grep -v '^[[:space:]]*$' | sed -n "${LINE}p")"
  if [[ -z "${TEXT}" ]]; then
    echo "ERROR: no eval text at LINE=${LINE} in ${TEXT_FILE}" >&2
    exit 1
  fi
fi

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "ERROR: base checkpoint not found: ${BASE_CHECKPOINT}" >&2
  exit 1
fi

mapfile -t EMBEDS < <(ls -1 "${EMBED_DIR}"/*.speaker.safetensors 2>/dev/null | sort)
if [[ "${#EMBEDS[@]}" -eq 0 ]]; then
  echo "ERROR: no *.speaker.safetensors found in ${EMBED_DIR}" >&2
  echo "       Run scripts/02_train.sh first." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Text:    ${TEXT}"
echo "Seed:    ${SEED}"
echo "Embeds:  ${#EMBEDS[@]}"
echo "Output:  ${OUT_DIR}"
echo

cd "${PROJECT_ROOT}"
for embed in "${EMBEDS[@]}"; do
  name="$(basename "${embed}" .speaker.safetensors)"
  out="${OUT_DIR}/${name}.wav"
  echo "--- ${name}"
  uv run --no-sync python infer.py \
    --checkpoint "${BASE_CHECKPOINT}" \
    --ref-embed "${embed}" \
    --text "${TEXT}" \
    --seed "${SEED}" \
    --output-wav "${out}" \
    "$@"
  echo
done

echo "Done. Listen to the files in ${OUT_DIR} and keep the embedding you like best."
