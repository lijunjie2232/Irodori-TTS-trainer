# Speaker Inversion Training

Speaker Inversion trains a small table of learned speaker tokens for **one target voice** while
keeping the entire base Irodori-TTS model frozen. The result is a tiny, reusable
`.speaker.safetensors` file that replaces reference audio at inference time.

This guide covers the full workflow: preparing a target-speaker manifest, configuring the run,
training, tuning, and using the resulting embedding.

> 简体中文译本见 [`SPEAKER_INVERSION_zh.md`](SPEAKER_INVERSION_zh.md)。
> A ready-to-run example workspace with template configs and scripts lives in
> [`speaker_inversion_example/`](speaker_inversion_example/).

For the parameter reference of every field mentioned here, see the
[Parameter Guide](docs/parameters.md#speaker-inversion). For the MeanFlow path, see the
[MeanFlow guide](docs/meanflow.md).

---

## 1. What Speaker Inversion Actually Trains

A normal speaker-conditioned forward pass encodes the reference waveform into a speaker
context:

```
ref_latent -> patch -> speaker_encoder -> speaker_norm -> [mean summary token] + tokens
```

With Speaker Inversion enabled, the model skips that entire path. A learnable tensor of shape
`(num_tokens, speaker_dim)` is expanded across the batch and used **directly** as the speaker
context:

```
speaker_inversion.embedding  (num_tokens, speaker_dim)  ->  speaker context
```

Two consequences are worth knowing before you start:

- **The reference audio in your manifest is ignored.** The dataset still loads it, but the model
  never consumes it, because the embedding replaces the whole speaker branch.
- **No masked-mean summary token is prepended.** A normal reference gets `1 + T_patched` context
  tokens (a global mean token followed by the patched frames). The learned embedding supplies
  `num_tokens` tokens on its own, so `speaker_inversion_tokens` is directly comparable to the
  token count a real reference would produce.

### How it compares to the other training modes

| | Full fine-tune | LoRA | Speaker Inversion |
|---|---|---|---|
| Trainable parameters | Whole model | Adapters | `num_tokens x speaker_dim` (e.g. 16 x 768 = 12,288) |
| Output artifact | `checkpoint_final.pt` / `.safetensors` | Adapter directory | `checkpoint_final.speaker.safetensors` |
| Needs reference audio at inference | No (but recommended) | No (but recommended) | No — pass `--ref-embed` |
| Speakers per checkpoint | Many | Many | **Exactly one** |
| Resumable with `--resume` | Yes | Yes | No — use `speaker_inversion_init_embedding` |

Because the artifact is only a few kilobytes and the base model is untouched, Speaker Inversion
is a good fit when you want a fixed, instantly loadable voice identity, when you want to ship one
voice without shipping reference audio, or when you want to compare many candidate voices cheaply.
Use LoRA or a full fine-tune instead when you need the model itself to change — new languages,
new prosody behaviour, or a large multi-speaker corpus.

---

## 2. Hard Requirements

The trainer validates these before doing anything, so know them up front:

| Requirement | Why |
|---|---|
| The model config must have speaker conditioning enabled | `enable_speaker_inversion()` refuses to build the embedding otherwise. `configs/train_v4_small_speaker_inversion.yaml` sets `use_speaker_condition: true`. |
| `--init-checkpoint` is mandatory | The frozen base must be real trained weights. The embedding only makes sense against the model it was trained with. |
| `train_mode` must be `rf` | MeanFlow distillation and `duration_only` are rejected. |
| `optimizer` must be `adamw` | With only one embedding matrix trainable, Muon has no compatible parameter. If the config or CLI says `muon`, training aborts. |
| LoRA must be off | `lora_enabled` + Speaker Inversion is rejected. |
| `--resume` must not be used | Checkpoints are embedding-only, so there is no optimizer state to restore. |
| `caption_warmup` and pretrained projector warmup must be off | Both are incompatible. |
| `speaker_inversion_tokens > 0`, `speaker_inversion_init_std >= 0` | Validated at startup. |

Set `--gradient-checkpointing` (or `gradient_checkpointing: true`) when VRAM is tight. It slows
training but is cheap insurance, and the supplied configs already enable it.

---

## 3. Step 1 — Prepare a Target-Speaker Manifest

The manifest is produced by `prepare_manifest.py`, which decodes audio through DACVAE and writes
one JSONL row per utterance.

```bash
uv run --no-sync python prepare_manifest.py \
  --dataset myorg/my_target_speaker_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/target_speaker_manifest.jsonl \
  --latent-dir data/target_speaker_latents \
  --device cuda
```

### Manifest fields

| Field | Required | Notes |
|---|---|---|
| `text` | Yes | Transcript. Normalized by default (`--text-normalize`). |
| `latent_path` | Yes | Target audio latent, written by the script. |
| `speaker_id` | **No** | Speaker Inversion does not need it. Every sample is treated as speaker-conditioned, so leaving it out is correct and slightly faster. |
| `caption` | Optional | Only include it if you want the embedding to be usable together with a caption. See the note below. |

Example row:

```json
{"text": "こんにちは", "latent_path": "data/target_speaker_latents/00001.pt", "num_frames": 750}
```

> **Why omitting `speaker_id` is better here.** When `speaker_id` is present, the dataset picks
> same-speaker sibling utterances and concatenates them into a reference latent — work that the
> model then throws away because the embedding overrides the speaker branch. Omitting `speaker_id`
> keeps the manifest simpler and avoids that wasted latent loading.

> **If you do include `caption`**, keep `caption_condition_dropout: 0.0`. The embedding then learns
> a speaker identity that is always presented alongside the caption, which is what you want if you
> plan to drive style with captions at inference time. If you plan to run without captions, leave
> the caption column out of the manifest entirely.

### Data recommendations

- **One speaker only.** A single embedding cannot represent a mixture of voices; the optimization
  will converge toward an average. Use one run per speaker and keep separate output directories.
- **Clean, single-speaker, music-free recordings.** The same advice as reference audio for cloning.
- **A few minutes is a reasonable starting point.** More data helps until returns flatten, but the
  bottleneck is usually audio quality, not hours.
- **Keep `--normalize-db -16.0`.** This matches the loudness normalization the codec was trained
  with.
- **Vary the content.** Read speech with varied text gives the embedding a better chance to separate
  identity from prosody than a handful of near-identical clips.

---

## 4. Step 2 — Configure the Run

Start from the provided config and edit it:

```bash
cp configs/train_v4_small_speaker_inversion.yaml configs/my_speaker.yaml
```

The only fields that matter for this mode:

| Field | Supplied value | Notes |
|---|---|---|
| `train_mode` | `rf` | Required. |
| `speaker_inversion_enabled` | `true` | Turns the mode on. Also settable with `--speaker-inversion`. |
| `speaker_inversion_tokens` | `16` | Number of learned tokens. Free to change. |
| `speaker_inversion_init_std` | `0.02` | Standard deviation of the random initialization. |
| `speaker_inversion_init_embedding` | empty | Path to an existing `.speaker.safetensors` for a warm start. |
| `text_condition_dropout` | `0.0` | Leave at `0.0`. |
| `caption_condition_dropout` | `0.0` | Leave at `0.0`. |
| `speaker_condition_dropout` | `0.0` | Leave at `0.0`. |
| `duration_speaker_dropout` | `0.0` | Leave at `0.0`. |
| `duration_caption_dropout` | `0.0` | Leave at `0.0`. |
| `optimizer` | `adamw` | Required. |
| `learning_rate` | `0.01` | Much higher than normal fine-tuning; the parameter count is tiny. |
| `weight_decay` | `0.0` | No decay on the embedding. |
| `lr_scheduler` | `none` | Keep it simple. |
| `max_steps` | `3000` | Usually enough for a single speaker. |
| `batch_size` | `16` | Per-process micro-batch. |
| `gradient_checkpointing` | `true` | Memory/speed trade-off. |
| `rf_loss_mode` | `utterance_mean` | Correct for variable-length targets. |
| `duration_loss_weight` | `0.1` | Duration loss stays active and still uses the embedding as speaker context. |
| `duration_backprop_to_condition` | `false` | Duration gradients do not flow into the frozen condition path. |
| `max_latent_steps` | `750` | ~30 s at 25 fps. Raise it if your clips are longer. |
| `fixed_target_latent_steps` | empty | Variable-length training. |
| `save_every` | `250` | Embedding checkpoints are tiny, so this can be frequent. |

**About the zero dropouts.** Condition dropout exists to train classifier-free-guidance
unconditional branches. Here it is deliberately disabled so the embedding learns one
unconditional identity rather than an identity that must also survive being dropped. This is why
the trainer forces `use_speaker = True` for every sample and ignores `has_speaker` from the
manifest.

**If you raise `batch_size` or `max_latent_steps`,** watch VRAM first. The frozen model still runs
a full forward and backward pass, so activation memory — not the embedding — is what limits you.

---

## 5. Step 3 — Train

Single GPU:

```bash
uv run --no-sync python train.py \
  --config configs/my_speaker.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --output-dir outputs/speaker_inversion/my_speaker
```

Multi-GPU with DDP:

```bash
uv run --no-sync torchrun --nproc_per_node 4 train.py \
  --config configs/my_speaker.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --output-dir outputs/speaker_inversion/my_speaker \
  --device cuda
```

Note that `batch_size` is per process, so the effective global batch is
`batch_size * gradient_accumulation_steps * world_size`. With four GPUs and the supplied config
that is 64, which may be more than you want for a single speaker — consider lowering
`batch_size` or using one GPU.

### What the log should show

```
Speaker Inversion parameters initialized: embedding=(16, 768).
Speaker Inversion freeze applied: trainable=12,288 frozen=...
```

The `trainable` number should be exactly `speaker_inversion_tokens * speaker_dim` (16 x 768 =
12,288 for v4-Small) and `frozen` should be the rest of the model. If `trainable` is far larger,
the freeze did not take effect the way you expect — stop and check the config.

The reported loss is the rectified-flow loss, normalized per utterance
(`rf_loss_mode: utterance_mean`), plus `0.1 x` the duration Huber loss. Expect a fairly fast drop
in the first few hundred steps and then a slow grind. There is no separate "speaker similarity"
metric in the log; judge the result by generating audio.

### Output files

| File | Content |
|---|---|
| `checkpoint_final.speaker.safetensors` | The final learned embedding. |
| `checkpoint_0000250.speaker.safetensors`, ... | Periodic snapshots, every `save_every` steps. |
| `checkpoint_best_val_loss_*.speaker.safetensors` | Only if validation is enabled (`valid_every > 0`). |
| `train_config.yaml` and friends | The resolved config, useful for reproducing the run. |

Every one of these is embedding-only — a few kilobytes. The base model is never written out.

---

## 6. Step 4 — Inference

Pass the embedding with `--ref-embed`, using **the same base checkpoint** you trained against:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed outputs/speaker_inversion/my_speaker/checkpoint_final.speaker.safetensors \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --output-wav outputs/sample_my_speaker.wav
```

`--ref-embed` is mutually exclusive with `--ref-wav`, `--ref-wavs`, `--ref-latent`,
`--ref-latents`, and `--no-ref`. Exactly one speaker-conditioning source per request.

Because the embedding is loaded at request time, it can be combined with a local `--checkpoint`
or with `--hf-checkpoint`. The base model must have speaker conditioning enabled; if it does not,
the runtime logs `speaker conditioning is disabled for this checkpoint; ignoring speaker embedding`
and generates without it.

### In the Gradio UI

```bash
uv run --no-sync python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Open the **Speaker Embedding** tab and either upload the `.speaker.safetensors` file or paste its
path. Uploading it and supplying reference audio at the same time is rejected — they are
mutually exclusive, just like the CLI flags.

### Sampling parameters that matter

| Parameter | Default | Effect |
|---|---|---|
| `--cfg-scale-speaker` | `5.0` | Speaker guidance strength. Since dropout was `0.0` during training, the unconditional branch is synthesized at inference rather than trained — see the next row. |
| `--speaker-uncond-mode` | `mask` | How the unconditional speaker branch is formed for CFG. `mask` zeroes the tokens and masks them out (default, lower VRAM). `noise` uses Gaussian noise scaled to the embedding's standard deviation. |
| `--speaker-kv-scale`, `--speaker-kv-min-t`, `--speaker-kv-max-layers` | off | Extra speaker K/V emphasis. Experimental, but a cheap thing to try if identity drifts. |
| `--num-steps` | `40` | RF sampling steps. |

Start with defaults, listen, then nudge `--cfg-scale-speaker` and `--speaker-uncond-mode`. If the
voice is too weak, try `--speaker-uncond-mode noise` or a moderate `--speaker-kv-scale 1.1` with
the default `--speaker-kv-min-t 0.9` before pushing CFG scales hard.

---

## 7. Continuing or Warm-Starting a Run

`--resume` is intentionally unsupported. To keep optimizing a saved embedding, point the config at
it:

```yaml
train:
  speaker_inversion_init_embedding: outputs/speaker_inversion/my_speaker/checkpoint_final.speaker.safetensors
```

or pass it on the command line:

```bash
uv run --no-sync python train.py \
  --config configs/my_speaker.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --speaker-inversion-init-embedding outputs/speaker_inversion/my_speaker/checkpoint_final.speaker.safetensors \
  --output-dir outputs/speaker_inversion/my_speaker_v2
```

The loaded embedding must have exactly `speaker_inversion_tokens` rows, otherwise startup fails with
a token-count mismatch. The optimizer and schedule start fresh, so this is a warm start rather than
a true resume — reduce `max_steps` accordingly.

---

## 8. Tuning Notes

**`speaker_inversion_tokens`.** More tokens give the embedding more capacity to describe a voice,
at the cost of a slightly larger artifact and more overfitting risk on small datasets. `16` is the
tuned default and a fine starting point; `32` is a reasonable experiment when you have plenty of
clean data and want to chase similarity. Very small values (e.g. `4`) act as a bottleneck and tend
to produce a generic-sounding voice.

**`speaker_inversion_init_std`.** `0.02` keeps the initial embedding small, so training starts from
a near-null speaker context and grows into the target. A larger value injects more initial energy
and can speed up early convergence, but makes the first steps noisier. Leave it alone unless you
are deliberately experimenting.

**Overfitting.** With a single speaker and a tiny parameter count, overfitting shows up as the
voice becoming brittle — good on text similar to the training set, unstable on everything else.
The fixes, in order of preference: add more varied text, lower `max_steps`, lower
`learning_rate`. Reusing the same embedding across many runs is a sign you should stop earlier.

**Learning rate.** `0.01` with AdamW and no scheduler is tuned for this parameter count. If you
change `speaker_inversion_tokens` by an order of magnitude, revisit the LR.

**Loss vs. quality.** The RF loss is a proxy, not a speaker-similarity score. Generate samples
periodically during training (the periodic `.speaker.safetensors` snapshots make this easy) and
pick the snapshot that sounds best, rather than assuming the last one wins.

**Validation.** Set `valid_ratio` above `0.0` and `valid_every` above `0` if you want
`checkpoint_best_val_loss_*` snapshots. Validation for this mode conditions every sample as
speaker-conditioned, matching training.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `speaker_inversion_enabled=True requires a speaker-conditioned model config.` | The model config disables speaker conditioning (e.g. a caption-only config). | Use `configs/train_v4_small_speaker_inversion.yaml` or set `use_speaker_condition: true`. |
| `... requires --init-checkpoint ...` | Started without base weights. | Add `--init-checkpoint path/to/model.safetensors`. |
| `... --resume full trainer state is not supported.` | Used `--resume`. | Use `speaker_inversion_init_embedding` instead. |
| `... supports optimizer='adamw'` | Config or CLI specified `muon`. | Set `optimizer: adamw`. |
| `... does not support LoRA training.` | `lora_enabled` is on. | Turn LoRA off for this run. |
| `... supports train_mode='rf' only.` | Config says `meanflow_distill` or `duration_only`. | Set `train_mode: rf`. |
| `speaker inversion init embedding token mismatch` | The init file has a different token count. | Match `speaker_inversion_tokens` to the file, or start fresh. |
| `Speaker Inversion file is missing 'speaker_embedding'.` | The file is not an embedding checkpoint (e.g. a full model). | Point at a `*.speaker.safetensors` file. |
| `... must use the '.speaker.safetensors' suffix` | File was renamed. | Keep the `.speaker.safetensors` suffix — the loader keys off it. |
| `speaker conditioning is disabled for this checkpoint; ignoring speaker embedding.` | The base model has no speaker branch. | Use a speaker-conditioned base such as v4-Small. |
| `ref_embed/--ref-embed cannot be combined with reference inputs or no_ref.` | Two conditioning sources in one request. | Pass exactly one. |
| Generated voice does not match the target | Too little clean data, or a mixed-speaker manifest. | Verify one speaker only, add varied clean speech, train longer. |
| Voice identity is weak | Speaker guidance too low. | Raise `--cfg-scale-speaker`, try `--speaker-uncond-mode noise`, then `--speaker-kv-scale`. |
| Output unstable on new text | Overfitting. | Fewer steps, lower LR, more varied transcripts. |
| CUDA out of memory | Activation memory from the frozen forward/backward. | Lower `batch_size` or `max_latent_steps`, keep `gradient_checkpointing: true`. |

---

## 10. Embedding File Format

A Speaker Inversion checkpoint is a plain safetensors file with a single tensor:

| Key | Shape | dtype |
|---|---|---|
| `speaker_embedding` | `(num_tokens, speaker_dim)` | `float32` on save |

- The filename **must** end in `.speaker.safetensors`. This suffix is how the codebase tells an
  embedding apart from a full model checkpoint — for example, the Gradio checkpoint auto-discovery
  deliberately skips files with this suffix.
- `speaker_dim` must match the base model (`768` for the v4-Small family).
- A `(1, tokens, dim)` tensor is accepted and squeezed on load; anything else is rejected.
- The same format is used for both input (`speaker_inversion_init_embedding`, `--ref-embed`) and
  output, so a saved embedding can be fed straight back into a new run.

---

## 11. Known Limitations

- **One speaker per checkpoint.** Train separate runs for separate voices. There is no
  multi-speaker embedding in this mode.
- **The embedding is tied to its base model.** An embedding trained against one base checkpoint
  will not behave correctly against a different one, including a fine-tuned or distilled variant.
- **Not compatible with MeanFlow training.** `meanflow_distill` rejects Speaker Inversion, and
  MeanFlow inference ignores runtime CFG, so the `--speaker-uncond-mode` formulation does not apply
  there. Train and infer against the RF base.
- **No full trainer state.** You cannot resume mid-run; a warm start from a saved embedding is the
  supported continuation path.
- **Legacy caption-only checkpoints are out of scope.** v2 VoiceDesign ignores speaker conditioning
  entirely, so `--ref-embed` has no effect on it.
- **Zero condition dropout means untrained unconditional branches.** This is intentional, but it
  makes `--speaker-uncond-mode` a runtime choice rather than something the model learned. Try both
  modes rather than assuming one is right.

---

## 12. Quick Reference

```bash
# 1. Manifest for the target speaker (no speaker_id, no caption)
uv run --no-sync python prepare_manifest.py \
  --dataset myorg/my_target_speaker \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/target_speaker_manifest.jsonl \
  --latent-dir data/target_speaker_latents \
  --device cuda

# 2. Train the embedding (base model stays frozen)
uv run --no-sync python train.py \
  --config configs/train_v4_small_speaker_inversion.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --output-dir outputs/speaker_inversion/my_speaker

# 3. Infer with the learned embedding
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed outputs/speaker_inversion/my_speaker/checkpoint_final.speaker.safetensors \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --output-wav outputs/sample_my_speaker.wav
```

Relevant files in this repository:

| Path | Role |
|---|---|
| `irodori_tts/speaker_inversion.py` | Embedding module, save/load helpers, `.speaker.safetensors` format |
| `irodori_tts/model.py` | `enable_speaker_inversion()` and the speaker-branch bypass in `encode_conditions()` |
| `irodori_tts/inference_runtime.py` | `--ref-embed` loading and speaker uncond handling |
| `irodori_tts/rf.py` | CFG branches and `speaker_uncond_mode` |
| `train.py` | Freeze logic, validation guards, embedding-only checkpointing |
| `configs/train_v4_small_speaker_inversion.yaml` | v4-Small Speaker Inversion config |
| `configs/train_500m_v3_speaker_inversion.yaml` | Legacy v3 Speaker Inversion config |
