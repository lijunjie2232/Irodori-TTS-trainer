# Speaker Inversion Example Workspace

Everything you need to train a **Speaker Inversion** embedding for one target voice against an
Irodori-TTS base model, with the base model frozen.

This workspace contains templates only. Point it at your own dataset and base checkpoint, then run
the four scripts in order.

> 中文说明见 [`../SPEAKER_INVERSION_zh.md`](../SPEAKER_INVERSION_zh.md)。
> 完整原理与参数解释见 [`../SPEAKER_INVERSION.md`](../SPEAKER_INVERSION.md)（English）。

---

## What is in here

```text
speaker_inversion_example/
├── README.md                                  # this file
├── .gitignore                                 # keeps latents, manifests, and audio out of git
├── configs/
│   ├── speaker_inversion.yaml                 # v4.1-Small config (default)
│   └── speaker_inversion_v3_500m.yaml         # legacy v3 500M config
├── data/
│   └── manifest.example.jsonl                 # manifest format reference
├── scripts/
│   ├── 01_prepare_manifest.sh                 # dataset -> DACVAE latents -> JSONL manifest
│   ├── 02_train.sh                            # freeze base, train the embedding
│   ├── 03_infer.sh                            # synthesise with --ref-embed
│   └── 04_eval_snapshots.sh                   # optional: compare periodic snapshots
├── texts/
│   └── eval_texts.txt                         # unseen sentences for judging the voice
└── outputs/                                   # created by the scripts
```

`data/latents/`, `data/target_speaker_manifest.jsonl`, and `outputs/` do not exist yet — the scripts
create them.

---

## Prerequisites

1. **Environment.** From the repository root:

   ```bash
   uv sync --extra cu128      # or --extra rocm / --extra xpu / --extra cpu
   ```

   Every script below uses `uv run --no-sync`, so it will not re-sync the environment and drop your
   PyTorch backend extra.

2. **A base checkpoint.** Download `model.safetensors` from
   [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) (or use a
   local fine-tuned checkpoint with the same architecture). You will pass its path to
   `BASE_CHECKPOINT`.

3. **A target-speaker dataset** with audio and transcripts. One speaker, clean, music-free. A few
   minutes of varied read speech is a reasonable start.

4. **Shell.** The scripts are `bash`. On Windows, run them from Git Bash. The underlying
   `uv run --no-sync python ...` commands work from PowerShell or cmd as well if you prefer to copy
   them out.

---

## Quick start

All commands assume the repository root as the working directory.

### 1. Build the target-speaker manifest

Edit the `EDIT ME` block in `scripts/01_prepare_manifest.sh` (or export the values), then:

```bash
DATASET=myorg/my_target_speaker scripts/01_prepare_manifest.sh
```

This writes `data/target_speaker_manifest.jsonl` and `data/latents/*.pt`.

**Do not pass `--speaker-column`.** The run learns a single embedding, so `speaker_id` is
unnecessary — and leaving it out also skips the dataset's same-speaker reference concatenation,
which the model ignores in this mode.

### 2. Train the embedding

```bash
SPEAKER_NAME=alice \
BASE_CHECKPOINT=/models/Irodori-TTS-v4.1-Small/model.safetensors \
  scripts/02_train.sh
```

Multi-GPU (remember `batch_size` is per process, so 4 GPUs multiplies the global batch by 4):

```bash
SPEAKER_NAME=alice \
BASE_CHECKPOINT=/models/Irodori-TTS-v4.1-Small/model.safetensors \
NPROC=4 \
  scripts/02_train.sh
```

Watch the startup log for these two lines:

```text
Speaker Inversion parameters initialized: embedding=(16, 768).
Speaker Inversion freeze applied: trainable=12,288 frozen=...
```

`trainable` must equal `speaker_inversion_tokens * speaker_dim` (16 x 768 = 12,288 for v4-Small). If
it is much larger, stop and check your config.

### 3. Synthesise with the trained voice

```bash
SPEAKER_NAME=alice \
BASE_CHECKPOINT=/models/Irodori-TTS-v4.1-Small/model.safetensors \
  scripts/03_infer.sh
```

Or override the sentence and pass extra inference flags through:

```bash
TEXT="おはようございます。今日もいい一日にしましょう。" \
SPEAKER_NAME=alice \
BASE_CHECKPOINT=/models/Irodori-TTS-v4.1-Small/model.safetensors \
  scripts/03_infer.sh --seed 0 --cfg-scale-speaker 6.0
```

### 4. Compare snapshots (optional but recommended)

Training saves a tiny checkpoint every `save_every` steps. Generate the same sentence from each one
and keep the embedding that sounds best:

```bash
SPEAKER_NAME=alice \
BASE_CHECKPOINT=/models/Irodori-TTS-v4.1-Small/model.safetensors \
  scripts/04_eval_snapshots.sh
```

Samples land in `outputs/<speaker>/snapshots/`, all with the same text and seed so the only
difference is the embedding.

---

## What you get

| Path | What it is |
|---|---|
| `outputs/<speaker>/checkpoint_final.speaker.safetensors` | The final learned embedding. A few KB. |
| `outputs/<speaker>/checkpoint_0000250.speaker.safetensors`, ... | Periodic snapshots. |
| `outputs/<speaker>/train_config.yaml` | The resolved config, for reproducing the run. |
| `outputs/<speaker>/sample.wav` | Output of `03_infer.sh`. |

The base model is never written out — every checkpoint in `outputs/` is embedding-only.

---

## Values worth knowing before you edit

| Setting | Default | Notes |
|---|---|---|
| `speaker_inversion_tokens` | `16` | Learned token count. Raise to 32 to chase similarity with plenty of clean data; very low values sound generic. |
| `learning_rate` | `0.01` | Deliberately high — the trainable parameter count is tiny. |
| `max_steps` | `3000` | Usually enough for one speaker. |
| `save_every` | `250` | Cheap, because snapshots are tiny. |
| All condition dropouts | `0.0` | Keep them at `0.0` so the embedding learns one unconditional identity. |
| `optimizer` | `adamw` | Required. `muon` is rejected for this mode. |
| `batch_size` | `16` | Per-process micro-batch. Lower it first if you hit OOM. |
| `gradient_checkpointing` | `true` | Less VRAM, slower. Set `false` if you have headroom. |

Things the trainer will refuse, so do not try them here: `--resume`, LoRA, `train_mode` other than
`rf`, `muon`, `caption_warmup`, and pretrained projector warmup. To continue from a saved embedding,
put its path in `speaker_inversion_init_embedding` and start a new run.

---

## Common issues

| Symptom | Fix |
|---|---|
| `base checkpoint not found` | Set `BASE_CHECKPOINT` to a real `model.safetensors`. |
| `manifest not found` | Run `01_prepare_manifest.sh` first. |
| `requires a speaker-conditioned model config` | Use the configs in this workspace; they set `use_speaker_condition: true`. |
| `... supports optimizer='adamw'` | Do not switch the optimizer to `muon`. |
| `... --resume full trainer state is not supported` | Use `speaker_inversion_init_embedding` instead. |
| Voice does not match the target | Confirm the manifest really contains one speaker, then add more varied clean speech. |
| Voice identity is weak | Raise `--cfg-scale-speaker`, or try `--speaker-uncond-mode noise`. |
| Unstable on new text | Overfitting: fewer steps, lower LR, more varied transcripts. |
| CUDA out of memory | Lower `batch_size`, keep `gradient_checkpointing: true`. |

The full troubleshooting table, plus the embedding file format and known limitations, are in
[`../SPEAKER_INVERSION.md`](../SPEAKER_INVERSION.md).

---

## Using the embedding elsewhere

CLI:

```bash
uv run --no-sync python infer.py \
  --checkpoint /models/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed speaker_inversion_example/outputs/alice/checkpoint_final.speaker.safetensors \
  --text "こんにちは。" \
  --output-wav outputs/alice.wav
```

Gradio UI — start `python gradio_app.py`, open the **Speaker Embedding** tab, and upload or paste the
path to the `.speaker.safetensors` file. Reference audio and a speaker embedding cannot be used in
the same request.

Two rules apply everywhere:

- Use the **same base checkpoint** the embedding was trained against.
- Keep the `.speaker.safetensors` suffix. The codebase uses it to tell an embedding apart from a full
  model checkpoint.
