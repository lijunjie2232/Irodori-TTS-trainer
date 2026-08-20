# 说话人反演（Speaker Inversion）训练指南

> 本文是 [`SPEAKER_INVERSION.md`](SPEAKER_INVERSION.md) 的简体中文译本。若两者出现歧义，以英文原版为准。
> 配套的可直接运行示例工作区见 [`speaker_inversion_example/`](speaker_inversion_example/)。

说话人反演（Speaker Inversion）会为**某一个目标音色**训练一张很小的可学习说话人 token 表，同时把整个
Irodori-TTS 基础模型完全冻结。产物是一个只有几 KB 的 `.speaker.safetensors` 文件，推理时可以完全取代
参考音频。

本指南覆盖完整流程：准备目标说话人 manifest、配置训练、开始训练、调参，以及使用训练出的嵌入。

本文出现的所有字段的逐项参数说明，见[参数指南](docs/parameters.md#speaker-inversion)。MeanFlow 相关流程见
[MeanFlow 指南](docs/meanflow.md)。

---

## 1. 说话人反演到底训练了什么

普通的、带说话人条件的推理会把参考音频编码成说话人上下文：

```
ref_latent -> patch -> speaker_encoder -> speaker_norm -> [mean 汇总 token] + tokens
```

而启用说话人反演后，模型会**跳过整条路径**。一个形状为 `(num_tokens, speaker_dim)` 的可学习张量会被扩展到整个
batch，并**直接**作为说话人上下文使用：

```
speaker_inversion.embedding  (num_tokens, speaker_dim)  ->  说话人上下文
```

因此有两点在动手之前必须知道：

- **manifest 里的参考音频会被忽略。** 数据集仍然会加载它，但模型在说话人分支被嵌入取代后根本不会消费它。
- **不会拼接 masked-mean 汇总 token。** 普通参考音频会得到 `1 + T_patched` 个上下文 token（一个全局均值
  token 加后面的分帧 token）。而可学习嵌入自带 `num_tokens` 个 token，所以 `speaker_inversion_tokens`
  可以直接与真实参考音频产生的 token 数量做类比。

### 与其他训练方式的对比

| | 全量微调 | LoRA | 说话人反演 |
|---|---|---|---|
| 可训练参数量 | 整个模型 | 适配器 | `num_tokens x speaker_dim`（例如 16 x 768 = 12,288） |
| 产出文件 | `checkpoint_final.pt` / `.safetensors` | 适配器目录 | `checkpoint_final.speaker.safetensors` |
| 推理时需要参考音频 | 否（但推荐提供） | 否（但推荐提供） | 不需要——传 `--ref-embed` 即可 |
| 每个 checkpoint 能代表的说话人数 | 多个 | 多个 | **只能是一个** |
| 可用 `--resume` 续训 | 可以 | 可以 | 不可以——改用 `speaker_inversion_init_embedding` |

由于产物只有几 KB 且基础模型分毫未动，说话人反演适合这些场景：想要一个固定、可瞬时加载的音色身份；想要在
不附带参考音频的前提下发布某个音色；或者想低成本地对比多个候选音色。如果确实需要改变模型本身——新语言、
新的韵律行为、或大规模多说话人语料——请改用 LoRA 或全量微调。

---

## 2. 硬性约束

训练器在真正开始前会校验以下条件，所以请先了解清楚：

| 约束 | 原因 |
|---|---|
| 模型配置必须启用说话人条件 | 否则 `enable_speaker_inversion()` 拒绝构建嵌入。`configs/train_v4_small_speaker_inversion.yaml` 中设置了 `use_speaker_condition: true`。 |
| 必须提供 `--init-checkpoint` | 冻结的基础模型必须是真实训练过的权重。嵌入只有相对于训练它的那个模型才有意义。 |
| `train_mode` 必须是 `rf` | MeanFlow 蒸馏与 `duration_only` 都会被拒绝。 |
| `optimizer` 必须是 `adamw` | 只有一个嵌入矩阵可训练时，Muon 没有兼容的参数。若配置或命令行写了 `muon`，训练直接中止。 |
| LoRA 必须关闭 | `lora_enabled` 与说话人反演同时开启会被拒绝。 |
| 不能使用 `--resume` | checkpoint 只有嵌入，没有可恢复的优化器状态。 |
| `caption_warmup` 与预训练投影层预热必须关闭 | 两者都不兼容。 |
| `speaker_inversion_tokens > 0`、`speaker_inversion_init_std >= 0` | 启动时校验。 |

显存紧张时请加 `--gradient-checkpointing`（或配置里 `gradient_checkpointing: true`）。它会拖慢训练，但代价很小
而收益明确，提供的配置已默认开启。

---

## 3. 第一步——准备目标说话人 manifest

manifest 由 `prepare_manifest.py` 生成：它把音频通过 DACVAE 编码，并为每句话写一行 JSONL。

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

### manifest 字段

| 字段 | 是否必需 | 说明 |
|---|---|---|
| `text` | 必需 | 转写文本。默认会做归一化（`--text-normalize`）。 |
| `latent_path` | 必需 | 目标音频的 latent，由脚本写出。 |
| `speaker_id` | **不需要** | 说话人反演不需要它。每个样本都会被当作有说话人条件，因此不写才是正确做法，而且还略快一些。 |
| `caption` | 可选 | 只有当你希望这个嵌入能配合 caption 一起使用时才写。见下方说明。 |

示例行：

```json
{"text": "こんにちは", "latent_path": "data/target_speaker_latents/00001.pt", "num_frames": 750}
```

> **为什么这里省略 `speaker_id` 更好。** 一旦存在 `speaker_id`，数据集就会去挑选同一说话人的兄弟语句并拼接成
> 参考 latent——而这份计算随后会被模型丢弃，因为嵌入已经接管了说话人分支。省略 `speaker_id` 既让 manifest
> 更简单，也避免了这次无用的 latent 加载。

> **如果确实要写 `caption`**，请把 `caption_condition_dropout` 保持为 `0.0`。这样嵌入学到的说话人身份会始终与
> caption 同时出现——如果你打算在推理时用 caption 驱动风格，这正是你想要的。如果计划完全不带 caption 运行，
> 那就干脆不要在 manifest 里放 caption 列。

### 数据建议

- **只放一个说话人。** 单个嵌入无法表示多个音色的混合，优化会收敛到一个"平均值"。每位说话人跑一次，并使用各自
  独立的输出目录。
- **干净、单人、无音乐的录音。** 与音色克隆对参考音频的要求相同。
- **几分钟素材是合理的起点。** 数据更多会有帮助，但收益会趋于平缓；瓶颈通常是音频质量，而不是小时数。
- **保持 `--normalize-db -16.0`。** 这与编解码器训练时使用的响度归一化一致。
- **内容要有变化。** 用不同文本的朗读语音，比用几句几乎相同的片段更能让嵌入把"身份"和"韵律"分离开。

---

## 4. 第二步——配置本次训练

从仓库提供的配置复制一份再改：

```bash
cp configs/train_v4_small_speaker_inversion.yaml configs/my_speaker.yaml
```

这个模式下真正关键的字段只有这些：

| 字段 | 提供的值 | 说明 |
|---|---|---|
| `train_mode` | `rf` | 必需。 |
| `speaker_inversion_enabled` | `true` | 打开该模式。也可用 `--speaker-inversion` 设置。 |
| `speaker_inversion_tokens` | `16` | 可学习 token 的数量，可自由调整。 |
| `speaker_inversion_init_std` | `0.02` | 随机初始化的标准差。 |
| `speaker_inversion_init_embedding` | 空 | 用于热启动的已有 `.speaker.safetensors` 路径。 |
| `text_condition_dropout` | `0.0` | 保持 `0.0`。 |
| `caption_condition_dropout` | `0.0` | 保持 `0.0`。 |
| `speaker_condition_dropout` | `0.0` | 保持 `0.0`。 |
| `duration_speaker_dropout` | `0.0` | 保持 `0.0`。 |
| `duration_caption_dropout` | `0.0` | 保持 `0.0`。 |
| `optimizer` | `adamw` | 必需。 |
| `learning_rate` | `0.01` | 远高于常规微调；因为可训练参数量极小。 |
| `weight_decay` | `0.0` | 对嵌入不做权重衰减。 |
| `lr_scheduler` | `none` | 保持简单。 |
| `max_steps` | `3000` | 对单个说话人通常足够。 |
| `batch_size` | `16` | 每进程的 micro-batch。 |
| `gradient_checkpointing` | `true` | 显存与速度的权衡。 |
| `rf_loss_mode` | `utterance_mean` | 变长目标下的正确设置。 |
| `duration_loss_weight` | `0.1` | 时长损失仍然生效，并仍以该嵌入作为说话人条件。 |
| `duration_backprop_to_condition` | `false` | 时长损失的梯度不会流入被冻结的条件路径。 |
| `max_latent_steps` | `750` | 25 fps 下约 30 秒。片段更长时请调大。 |
| `fixed_target_latent_steps` | 空 | 变长训练。 |
| `save_every` | `250` | 嵌入 checkpoint 极小，所以可以存得频繁一些。 |

**关于全部为 0 的 dropout。** 条件 dropout 的作用是训练无分类器引导（CFG）的无条件分支。这里刻意关闭它，是为了
让嵌入学到**一个**无条件的身份，而不是一个还必须能在被丢弃时存活的身份。这也是训练器会强制令每个样本
`use_speaker = True`、并忽略 manifest 中 `has_speaker` 的原因。

**如果调大 `batch_size` 或 `max_latent_steps`，** 先盯显存。冻结的模型依然要跑完整的正向与反向传播，所以限制你
的是激活显存，而不是嵌入本身。

---

## 5. 第三步——开始训练

单卡：

```bash
uv run --no-sync python train.py \
  --config configs/my_speaker.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --output-dir outputs/speaker_inversion/my_speaker
```

多卡 DDP：

```bash
uv run --no-sync torchrun --nproc_per_node 4 train.py \
  --config configs/my_speaker.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --output-dir outputs/speaker_inversion/my_speaker \
  --device cuda
```

注意 `batch_size` 是每进程的值，因此等效全局 batch 为
`batch_size * gradient_accumulation_steps * world_size`。四张卡配上提供的配置就是 64，对单个说话人来说可能
超出所需——可以考虑调小 `batch_size`，或者干脆只用一张卡。

### 日志里应该看到什么

```
Speaker Inversion parameters initialized: embedding=(16, 768).
Speaker Inversion freeze applied: trainable=12,288 frozen=...
```

`trainable` 应当恰好等于 `speaker_inversion_tokens * speaker_dim`（v4-Small 为 16 x 768 = 12,288），
`frozen` 应当是模型其余部分。如果 `trainable` 远大于这个数，说明冻结没有按预期生效——请停下来检查配置。

日志里的 loss 是整流流（rectified flow）损失，按语句归一化（`rf_loss_mode: utterance_mean`），再加上
`0.1 x` 时长 Huber 损失。预期前几百步下降较快，之后缓慢推进。日志中没有单独的"说话人相似度"指标；效果要靠
生成音频来判断。

### 输出文件

| 文件 | 内容 |
|---|---|
| `checkpoint_final.speaker.safetensors` | 最终学到的嵌入。 |
| `checkpoint_0000250.speaker.safetensors`、... | 每 `save_every` 步的周期快照。 |
| `checkpoint_best_val_loss_*.speaker.safetensors` | 仅在开启验证（`valid_every > 0`）时产生。 |
| `train_config.yaml` 等 | 实际解析后的配置，便于复现该次训练。 |

以上每一个文件都**只含嵌入**——只有几 KB。基础模型永远不会被写出。

---

## 6. 第四步——推理

用 `--ref-embed` 传入嵌入，并使用**与训练时完全相同的基础 checkpoint**：

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed outputs/speaker_inversion/my_speaker/checkpoint_final.speaker.safetensors \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --output-wav outputs/sample_my_speaker.wav
```

`--ref-embed` 与 `--ref-wav`、`--ref-wavs`、`--ref-latent`、`--ref-latents`、`--no-ref` 互斥。每个请求只能有
一个说话人条件来源。

由于嵌入是在请求时加载的，它既可以配合本地 `--checkpoint`，也可以配合 `--hf-checkpoint`。基础模型必须启用了
说话人条件；若没有，运行时会打印
`speaker conditioning is disabled for this checkpoint; ignoring speaker embedding`，并在不带该嵌入的情况下生成。

### 在 Gradio 界面中

```bash
uv run --no-sync python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

打开 **Speaker Embedding** 标签页，上传 `.speaker.safetensors` 文件或直接粘贴其路径。同时上传该文件并提供参考
音频会被拒绝——两者互斥，与命令行参数的行为一致。

### 需要关注的采样参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--cfg-scale-speaker` | `5.0` | 说话人引导强度。由于训练时 dropout 为 `0.0`，无条件分支是在推理期合成的，而不是训练出来的——见下一行。 |
| `--speaker-uncond-mode` | `mask` | CFG 中无条件说话人分支的构造方式。`mask` 把 token 置零并屏蔽掉（默认，显存占用更低）。`noise` 使用按嵌入标准差缩放的 Gaussian 噪声。 |
| `--speaker-kv-scale`、`--speaker-kv-min-t`、`--speaker-kv-max-layers` | 关闭 | 额外加强说话人 K/V。属于实验性手段，但在身份漂移时是成本很低的尝试。 |
| `--num-steps` | `40` | RF 采样步数。 |

先用默认值，听过效果后再微调 `--cfg-scale-speaker` 与 `--speaker-uncond-mode`。如果音色太弱，可以先试
`--speaker-uncond-mode noise`，或在默认 `--speaker-kv-min-t 0.9` 下试一个温和的 `--speaker-kv-scale 1.1`，
最后才考虑把 CFG 尺度往上推。

---

## 7. 续训与热启动

`--resume` 被刻意设计为不支持。若要继续优化已保存的嵌入，把配置指向它：

```yaml
train:
  speaker_inversion_init_embedding: outputs/speaker_inversion/my_speaker/checkpoint_final.speaker.safetensors
```

或在命令行传入：

```bash
uv run --no-sync python train.py \
  --config configs/my_speaker.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --speaker-inversion-init-embedding outputs/speaker_inversion/my_speaker/checkpoint_final.speaker.safetensors \
  --output-dir outputs/speaker_inversion/my_speaker_v2
```

载入的嵌入必须恰好有 `speaker_inversion_tokens` 行，否则启动时会因 token 数不匹配而失败。优化器与学习率调度会
重新开始，所以这是热启动而非真正的续训——请相应调小 `max_steps`。

---

## 8. 调参要点

**`speaker_inversion_tokens`。** token 越多，嵌入描述音色的容量越大，代价是产物略大、在小数据集上更容易过拟合。
`16` 是调好的默认值，也是不错的起点；数据充足且想追求相似度时可以试 `32`。取值很小（例如 `4`）会形成瓶颈，
容易得到听起来很"通用"的音色。

**`speaker_inversion_init_std`。** `0.02` 让初始嵌入保持很小，因此训练从接近空的说话人上下文出发、逐步长成
目标音色。取更大值会注入更多初始能量，可能加快早期收敛，但也会让最初几步更嘈杂。除非刻意做实验，否则不要动它。

**过拟合。** 单个说话人加上极小的参数量，过拟合表现为音色变得"脆"——在接近训练集的文本上表现良好，在其他内容
上不稳定。修复手段按优先级排序：增加文本多样性、调小 `max_steps`、调小 `learning_rate`。如果你发现同一个嵌入
被反复复用到很多次训练里，说明应该更早停。

**学习率。** `0.01` 配合 AdamW、不使用调度器，是针对这个参数量调过的。如果你把 `speaker_inversion_tokens` 改变
了一个数量级，请重新审视学习率。

**损失与音质的关系。** RF 损失只是代理指标，不是说话人相似度评分。训练期间请周期性生成样本来试听（周期性的
`.speaker.safetensors` 快照让这件事很方便），然后挑选听起来最好的那个快照，而不是想当然地认为最后一个最好。

**验证。** 如果希望得到 `checkpoint_best_val_loss_*` 快照，请把 `valid_ratio` 设为大于 `0.0`，并把 `valid_every`
设为大于 `0`。该模式下的验证会把每个样本都当作有说话人条件，与训练一致。

---

## 9. 故障排查

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `speaker_inversion_enabled=True requires a speaker-conditioned model config.` | 模型配置关闭了说话人条件（例如仅 caption 的配置）。 | 使用 `configs/train_v4_small_speaker_inversion.yaml`，或设置 `use_speaker_condition: true`。 |
| `... requires --init-checkpoint ...` | 没有提供基础权重就启动了。 | 加上 `--init-checkpoint path/to/model.safetensors`。 |
| `... --resume full trainer state is not supported.` | 使用了 `--resume`。 | 改用 `speaker_inversion_init_embedding`。 |
| `... supports optimizer='adamw'` | 配置或命令行指定了 `muon`。 | 设为 `optimizer: adamw`。 |
| `... does not support LoRA training.` | `lora_enabled` 处于开启状态。 | 本次训练关闭 LoRA。 |
| `... supports train_mode='rf' only.` | 配置写的是 `meanflow_distill` 或 `duration_only`。 | 设为 `train_mode: rf`。 |
| `speaker inversion init embedding token mismatch` | 初始嵌入文件的 token 数与配置不一致。 | 让 `speaker_inversion_tokens` 与该文件匹配，或从零开始。 |
| `Speaker Inversion file is missing 'speaker_embedding'.` | 该文件不是嵌入 checkpoint（例如是一个完整模型）。 | 指向 `*.speaker.safetensors` 文件。 |
| `... must use the '.speaker.safetensors' suffix` | 文件被改名了。 | 保留 `.speaker.safetensors` 后缀——加载器依赖它来识别。 |
| `speaker conditioning is disabled for this checkpoint; ignoring speaker embedding.` | 基础模型没有说话人分支。 | 使用带说话人条件的基础模型，例如 v4-Small。 |
| `ref_embed/--ref-embed cannot be combined with reference inputs or no_ref.` | 同一个请求里给了两个条件来源。 | 只传其中一个。 |
| 生成音色与目标不符 | 干净数据太少，或 manifest 混入了多个说话人。 | 确认只有一个说话人，补充多样化的干净语音，延长训练。 |
| 音色身份偏弱 | 说话人引导强度不足。 | 调大 `--cfg-scale-speaker`，试 `--speaker-uncond-mode noise`，再试 `--speaker-kv-scale`。 |
| 换新文本后输出不稳定 | 过拟合。 | 减少步数、降低学习率、增加文本多样性。 |
| CUDA out of memory | 冻结模型正反向传播带来的激活显存。 | 调小 `batch_size` 或 `max_latent_steps`，保持 `gradient_checkpointing: true`。 |

---

## 10. 嵌入文件格式

说话人反演 checkpoint 就是一个普通的 safetensors 文件，内含单个张量：

| 键 | 形状 | 类型 |
|---|---|---|
| `speaker_embedding` | `(num_tokens, speaker_dim)` | 保存时为 `float32` |

- 文件名**必须**以 `.speaker.safetensors` 结尾。代码库靠这个后缀区分"嵌入"与"完整模型 checkpoint"——例如
  Gradio 的 checkpoint 自动发现会刻意跳过带此后缀的文件。
- `speaker_dim` 必须与基础模型一致（v4-Small 系列为 `768`）。
- 形状为 `(1, tokens, dim)` 的张量会被接受并在载入时压缩掉多余维度；其他形状一律拒绝。
- 输入（`speaker_inversion_init_embedding`、`--ref-embed`）与输出使用同一格式，因此保存下来的嵌入可以直接喂给
  一次新的训练。

---

## 11. 已知限制

- **每个 checkpoint 只能有一个说话人。** 不同音色请分别训练。该模式不提供多说话人嵌入。
- **嵌入与它的基础模型绑定。** 针对某个基础 checkpoint 训练出的嵌入，放到另一个 checkpoint（包括微调版或蒸馏
  版）上不会正常工作。
- **与 MeanFlow 训练不兼容。** `meanflow_distill` 会拒绝说话人反演，而 MeanFlow 推理会忽略运行期 CFG，因此
  `--speaker-uncond-mode` 这套构造在那里不适用。请始终针对 RF 基础模型训练与推理。
- **没有完整的训练器状态。** 无法中途续训；从已保存的嵌入热启动是官方支持的接续方式。
- **旧版仅 caption 的 checkpoint 不在支持范围内。** v2 VoiceDesign 完全忽略说话人条件，`--ref-embed` 对它没有
  效果。
- **条件 dropout 为零意味着无条件分支未经训练。** 这是有意为之，但也使得 `--speaker-uncond-mode` 成为一个运行期
  选择，而不是模型学到的行为。请两种模式都试，不要预设哪种一定更好。

---

## 12. 速查

```bash
# 1. 目标说话人的 manifest（不带 speaker_id，不带 caption）
uv run --no-sync python prepare_manifest.py \
  --dataset myorg/my_target_speaker \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/target_speaker_manifest.jsonl \
  --latent-dir data/target_speaker_latents \
  --device cuda

# 2. 训练嵌入（基础模型保持冻结）
uv run --no-sync python train.py \
  --config configs/train_v4_small_speaker_inversion.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --output-dir outputs/speaker_inversion/my_speaker

# 3. 用学到的嵌入做推理
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed outputs/speaker_inversion/my_speaker/checkpoint_final.speaker.safetensors \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --output-wav outputs/sample_my_speaker.wav
```

本仓库中的相关文件：

| 路径 | 作用 |
|---|---|
| `irodori_tts/speaker_inversion.py` | 嵌入模块、保存/载入辅助函数、`.speaker.safetensors` 格式 |
| `irodori_tts/model.py` | `enable_speaker_inversion()`，以及 `encode_conditions()` 中对说话人分支的旁路 |
| `irodori_tts/inference_runtime.py` | `--ref-embed` 的载入与说话人无条件分支处理 |
| `irodori_tts/rf.py` | CFG 分支与 `speaker_uncond_mode` |
| `train.py` | 冻结逻辑、启动校验、仅嵌入的 checkpoint 保存 |
| `configs/train_v4_small_speaker_inversion.yaml` | v4-Small 说话人反演配置 |
| `configs/train_500m_v3_speaker_inversion.yaml` | 旧版 v3 说话人反演配置 |
| `speaker_inversion_example/` | 可直接运行的示例工作区（配置、脚本、示例 manifest、评测文本） |
