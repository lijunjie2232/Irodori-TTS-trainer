# CrispASR ctypes 绑定 —— 扩展接口与 C 符号参考

主流程（VAD + ASR）见 [SKILL.md](SKILL.md)。本文覆盖其余 ctypes 导出、C 符号对照、
单位与返回码约定、环境变量总表。所有函数都在 `crispasr._binding` 中；
带 `lib_path=` 参数的可显式换库（默认走 `_find_lib()`）。

## 直接调用绑定未包的 C 符号

需要调 binding 里没包装的导出时，沿用绑定同样的手法即可（**不要改 `_binding.py`**）：

```python
import ctypes
cb_lib = ctypes.CDLL(os.environ["CRISPASR_LIB_PATH"])   # 或 cb._find_lib()
fn = cb_lib.crispasr_vad_slices_ex                      # 符号名见 include/crispasr.h
fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ...]
fn.restype = ctypes.c_int
out = ctypes.POINTER(ctypes.c_float)()
n = fn(path.encode("utf-8"), pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
       pcm.size, 16000, 0.5, 250, 100, 30, 30.0, 4, 1, 0, ctypes.byref(out))
cb_lib.crispasr_vad_free(out)                           # 由 C 分配的内存必须交回 C 释放
```

规则：入参数组先 `np.ascontiguousarray(x, dtype=np.float32)`；字符串一律 UTF-8 bytes；
C 返回的指针用配套的 `*_free` 释放；`hasattr(lib, "symbol")` 探测新导出，
旧库缺符号时降级而不是崩。

## Python ↔ C 符号对照

| Python（`crispasr._binding`） | C 符号 |
|---|---|
| `vad_slices(...)` | `crispasr_vad_slices_ex` / `crispasr_vad_free` |
| `vad_segments(...)` | `crispasr_vad_segments`（旧签名，无设备/batch 参数） |
| `Session(...)` | `crispasr_session_open` / `_open_explicit` / `_backend` / `_close` |
| `Session.transcribe` | `crispasr_session_transcribe` / `_transcribe_lang` |
| `Session.transcribe_vad` | `crispasr_session_transcribe_vad` / `_vad_lang` |
| `Session.transcribe_chunked` | `crispasr_session_transcribe_chunked_lang` |
| `Session.get_progress` | `crispasr_get_progress` |
| `Session.*setter*` | `crispasr_session_set_*` |
| `Session.available_backends` | `crispasr_session_available_backends` |
| `detect_language_pcm` | `crispasr_detect_language_pcm` |
| `text_detect_language` | `crispasr_text_detect_language` |
| `diarize_segments[_with_turns]` | `crispasr_diarize_segments_abi` |
| `align_words` | `crispasr_align_words_abi` + `crispasr_align_result_*` |
| `PuncModel` | `crispasr_punc_init` / `_punc_process` / `_punc_free_text` / `_punc_free` |
| `TitaNet` | `crispasr_titanet_init` / `_embed` / `_cosine_sim` / `_free` |
| `SpeakerDB` | `crispasr_speaker_db_open` / `_match` / `_enroll2` / `_count` / `_free` |
| `SpeakerEmbedder` | `crispasr_speaker_embedder_make_abi` / `_free_abi` |
| `PyannoteCache` | `crispasr_pyannote_cache_compute_abi` / `_apply_abi` / `_free_abi` |
| `Mic` | `crispasr_mic_open` / `_start` / `_stop` / `_close` |
| `registry_lookup` / `registry_default_bundle` / `list_known_models` | `crispasr_registry_*_abi` |
| `cache_ensure_file` / `cache_dir` | `crispasr_cache_ensure_file_abi` / `crispasr_cache_dir_abi` |
| `detect_backend_from_gguf` | `crispasr_detect_backend_from_gguf` / `_detect_backends_from_gguf` |
| `lcs_dedup_prefix_count` | `crispasr_lcs_dedup_prefix_count` |
| `enhance_audio_rnnoise` | `crispasr_enhance_audio_rnnoise` |
| `watermark_*` | `crispasr_watermark_load_model` / `_embed` / `_detect` |
| `ChatSession` | `crispasr_chat_*` |
| `CrispASR`（whisper 旧类） | `whisper_*` + `crispasr_params_set_*` |

## 语言识别 / 翻译

```python
res = cb.detect_language_pcm(pcm, model_path="ggml-tiny.bin", method=cb.LidMethod.WHISPER,
                             n_threads=4, use_gpu=False, gpu_device=0, flash_attn=True)
res.lang_code, res.confidence          # LidResult；LidMethod.WHISPER=0 / SILERO=1
cb.text_detect_language("こんにちは", "glotlid-lid176.gguf")   # -> (lang, conf)
sess.translate_text(text, "ja", "en", max_tokens=512)          # 需库内已载 m2m100/nllb 类模型
```

`LidMethod.WHISPER` 需要一个 whisper ASR 模型做声学 LID；`SILERO` 用 silero-lid GGUF。
转写侧的自动语言用 `sess.detected_language()`（仅 whisper，其它后端返回 `"unknown"`）。

## 说话人分离（diarization）

```python
segs = [cb.DiarizeSegment(t0=s.start, t1=s.end) for s in my_session_segments]
ok = cb.diarize_segments(segs, pcm, method=cb.DiarizeMethod.VAD_TURNS,
                         n_threads=4, slice_t0=0.0,
                         min_speakers=0, max_speakers=0, num_speakers=0)
# 结果原地写回 segs[i].speaker（int，-1 = 该方法无信息）
```

| 参数 | 说明 |
|---|---|
| `left` / `right` / `is_stereo` | 单声道传 `left=pcm`；`ENERGY=0`、`XCORR=1` **仅立体声** |
| `method` | `VAD_TURNS=2`（单声道、纯时序）/ `PYANNOTE=3`（需 `pyannote_model_path`）/ `FOXNOSE=4`（需 `foxnose_embedder_path`，如 `wespeaker-resnet34-lm.gguf`） |
| `slice_t0` | 本段 pcm 相对整条音频的起点（秒）——切片后做 diarize 必须传，否则时间轴错位 |
| `min/max/num_speakers` | `0` = 不约束 |

`diarize_segments_with_turns(...)` 返回 `(ok, List[DiarizeTurn(t0,t1,speaker)])`，
只有 FOXNOSE 会给 turns，其它方法返回空列表（**不是错误**）。

自建流水线：`SpeakerEmbedder("auto"|"titanet"|"indextts"|".gguf路径", n_threads=4)` →
`embed(pcm[int(s.start*16000):int(s.end*16000)])` → `agglomerative_cluster(embs,
merge_threshold=0.5, max_speakers=32)`（单链接余弦，给全局稳定 ID）。
跨 chunk 一致性用 `PyannoteCache(pcm_full, model_path)` + `.apply(segs, slice_t0=...)`。
`TitaNet(path)` 输出 192 维 L2 归一化向量，`TitaNet.cosine_sim(a,b)` 即点积。

`SpeakerDB(dir_path, expected_names="A,B", consent=True)`：**必须 `consent=True`**，
否则 `ValueError`（生物特征识别合规门槛，非 API 缺陷）；`match(emb, threshold=0.7)` →
`(name|None, score)`，`enroll(name, emb)` 注册。

## 强制对齐与 logits

```python
words = cb.align_words("canary-ctc-aligner.gguf", "hello world", pcm,
                       t_offset=0.0, n_threads=4)     # -> List[AlignedWord(text,start,end)]
segs, logits = sess.transcribe_with_logits(pcm, 16000, language="ja")
vocab = sess.ctc_vocab()                              # 仅 Omni CTC 有值，否则 None
```

- `logits` 为 frame-major `np.ndarray`（Omni/wav2vec2 是 raw 预 softmax，canary-ctc 是 log-prob），
  后端不产 logits 时为 `None`；调用前需 `sess.set_return_logits(True)`，
  且 `transcribe*` 结束时绑定会**自动关回** `False`（避免污染后续调用）。
- 非拉丁文本给拉丁词表对齐器会被自动罗马化（#252）；`CRISPASR_ALIGN_NO_ROMANIZE=1` 关，
  对齐结果本身仍回原始文字（#419 起）。

## 标点 / 后处理

```python
with cb.PuncModel("fireredpunc-q8_0.gguf") as pm:      # FireRedPunc（BERT）
    text = pm.process("こんにちは今日はいい天気ですね")
sess.set_punctuation(True)                          # canary/cohere 等自带标点的后端开关
sess.set_punc_model("fireredpunc.gguf")             # 在 session 上挂标点模型，转写时自动加标点
```

`PuncModel(model_path)` 除路径外还接受别名 `auto` / `firered` / `fullstop` /
`punctuate-all` / `pcs`（首次使用会下载）。对 CTC 类后端（wav2vec2、omniasr、
fastconformer-ctc、firered-asr）输出全小写无标点特别有用；`process()` 失败时
返回原文而不抛异常，因此可以无脑对每段调用。

## 流式与采集

```python
st = sess.stream_open(step_ms=3000, length_ms=10000, keep_ms=200,
                      language="ja", translate=False, live=False)
st.feed(chunk_pcm)          # -> 已消费的样本数
st.get_text()               # -> dict（partial/final）
st.flush(); st.close()

with cb.Mic(sample_rate=16000, channels=1, callback=fn) as mic:
    mic.start()             # 回调在本线程之外触发
cb.mic_default_device_name()
```

分块自驱转写（自己喂窗口）时用 `lcs_dedup_prefix_count(prev_tail_tokens, curr_tokens,
min_lcs_length=1)` 去掉窗口重叠造成的重复前缀，别自己写 LCS。

## TTS / 音频分析（同一 `Session` 类承载）

```python
s = cb.Session("kokoro-v1.0.gguf")
s.set_voice("af_heart.gguf")            # 也可传参考 *.wav（配 ref_text）
pcm_out = s.synthesize("テキスト")       # 单位：s.output_sample_rate()
s.input_sample_rate()                   # 该后端期望的输入采样率
```

常用：`set_speaker_name`（orpheus）/ `set_speaker_id`（piper、fastpitch、melotts）/
`set_instruct`（qwen3-tts、parler、omnivoice 风格描述）/ `set_language` 系
（`set_source_language`、`set_target_language`、`set_tts_reference_language`）/
`set_tts_steps`、`set_length_scale`、`set_exaggeration`、`set_cfg_weight`、
`set_top_p/top_k/min_p/do_sample`、`set_tts_seed`、`set_max_speech_tokens`、
`set_tts_phonemes`（kokoro/piper 逐音素复现发音，其它后端 `rc=-2` no-op）。

合规相关：`synthesize()` 会自动嵌入 AI 水印，未署名需 `accept_marking_responsibility(attestation)`
+ `set_speaker_identity("real_person"|"synthetic"|...)`；无水印版本是 `synthesize_raw()`。
另有 `separate(pcm_stereo)`（htdemucs 分轨）、`pitch(pcm, hop_ms=10.0)`（F0 轨迹）、
`speech_to_speech(pcm, language)`。降噪：`cb.enhance_audio_rnnoise(pcm)` 要求 **48 kHz**，
16k 素材需先升采样再降回。水印：`watermark_load_model(gguf)` / `watermark_embed(pcm, alpha=-1)`
（原地）/ `watermark_detect(pcm) -> float`。

## 文本 Chat（GGUF LLM）

```python
chat = cb.ChatSession("model.gguf", n_ctx=4096, n_gpu_layers=-1, use_mmap=True)
chat.count_tokens(msgs); chat.n_ctx; chat.template_name
reply = chat.generate(msgs, max_tokens=256, temperature=0.7,
                      should_continue=lambda: not aborted)     # 抛 ChatAborted
chat.generate_stream(msgs, on_token=lambda piece: print(piece, end=""))
chat.reset(); chat.close()
cb.ChatSession.memory_estimate("model.gguf")   # 粗估工作集字节数
```

`should_continue` 回调里抛异常会被搬到 `_ChatCallbackState` 并在原生调用返回后重抛；
`stop` 列表、`prefill_only=True`（只算 prompt）均可用。

## 模型获取（注册表 / 缓存）

```python
entry = cb.registry_lookup("parakeet")            # RegistryEntry(filename, url, approx_size)
cb.registry_lookup_by_filename("parakeet-tdt-0.6b-ja")
bundle = cb.registry_default_bundle("omnivoice")  # RegistryBundle(backend, license, requires_acceptance, artifacts)
if bundle.requires_acceptance:
    ...                                           # 先取得授权再下载
path = cb.cache_ensure_file(a.filename, a.url, quiet=False, cache_dir_override=None)
cb.cache_dir()                                    # $CRISPASR_CACHE_DIR → $CRISPASR_MODELS_DIR → ~/.cache/crispasr
```

模型路径解析建议按「绝对路径 → 仓库相对 → 缓存目录同名文件」三级尝试，
找不到就报错并打印缓存目录，别静默联网下载。
`cb.detect_backend_from_gguf(path)` / `detect_backends_from_gguf(path)`（全部可用后端，主后端在前）
可在开 session 前判断该文件能不能被本库打开。

## Whisper 专用类 `CrispASR`

只在需要 whisper 原生细粒度控制（`strategy=CRISPASR_SAMPLING_BEAM_SEARCH`、`tdrz`）时用：

```python
m = cb.CrispASR("ggml-base.en.bin")
for seg in m.transcribe("audio.wav", language="auto"):     # 直接吃文件路径
    print(seg.start, seg.end, seg.text)
m.transcribe_pcm(pcm, 16000, language="ja", vad=True,
                 vad_model_path="firered-vad.gguf", vad_threshold=0.5,
                 vad_min_speech_ms=250, vad_min_silence_ms=100, tdrz=False)
m.detected_language(); m.close()
```

注意它依赖可选的 `libcrispasr_helpers`（按值传 struct 的包装）；缺失时退到
`whisper_init_from_file` 简化路径。`Segment` 是独立于 `SessionSegment` 的 dataclass。

## 单位与约定

| 事项 | 约定 |
|---|---|
| 时间戳 | C 侧一律 **centiseconds**（1/100 s），绑定除以 100 后暴露为 float 秒 |
| PCM | `float32` 单声道 16 kHz（VAD/ASR）；RNNoise 例外要 48 kHz；TTS 看 `output_sample_rate()` |
| 内存所有权 | 返回的 `float*` / result 句柄由 C 分配，必须走 `crispasr_vad_free` / `crispasr_session_result_free`（绑定已在 `finally` 中处理，自写 ctypes 时别漏） |
| 字符串 | UTF-8 bytes；路径含非 ASCII 在 Windows 上建议先复制成 ASCII 临时名 |
| 未知导出 | `hasattr(lib, sym)` 探测 + 降级；不要假定 wheel / 系统库与源码同版本 |
| 并发 | 为每个工作线程/进程各开一个 `Session`，不要共享同一句柄；VAD 的 ggml context 是进程级缓存 |
| 性能测量 | 设备/batch 组合的对比测试必须各起独立子进程，否则被同进程缓存污染 |

## 环境变量（必须在 `import crispasr` 之前设置的用 ★）

| 变量 | 作用 |
|---|---|
| ★ `CRISPASR_LIB_PATH` | 指定共享库绝对路径（绑定 probe 顺序第 1 位） |
| ★ `CRISPASR_VAD_ENCDEC_CPU=1` | 强制 Whisper-encDec VAD 走 CPU（首次初始化时读取） |
| ★ `CRISPASR_VAD_GPU` | `use_gpu<0` 时的默认设备：`0`/未设=CPU，`1`=GPU 自动后端，其它值=GPU 且指定后端（`cuda`/`vulkan`/`metal`） |
| ★ `CUDA_VISIBLE_DEVICES` / `GGML_VK_VISIBLE_DEVICES` | 标准 ggml/CUDA 设备选择，同时影响 ASR 侧 |
| `CRISPASR_FIRERED_VAD_IMPL` | `scalar`（CPU 手写 DFSMN）/ `graph`（ggml 图，GPU 路径） |
| `CRISPASR_FIRERED_VAD_GRAPH_BATCH` | FireRed 图批大小（默认 4096） |
| `CRISPASR_MARBLENET_VAD_SERIAL=1` | MarbleNet 前端单线程 |
| `CRISPASR_VAD_PROGRESS=0` / `_PROGRESS_MS` | 关闭/调整 VAD  stderr 进度行 |
| `CRISPASR_SESSION_AUTOCHUNK` / `_CHUNK_SECONDS` / `_PERBACKEND_CHUNK` | session 长音频自动切块 |
| `CRISPASR_GAP_FILL` / `_GAP_FILL_MIN_CS` | 对首遍空文本的 span 重转（parakeet 默认开） |
| `CRISPASR_SEG_MAX_CHARS` / `_SEG_DROP_NONVERBAL` / `_SEG_LOGPROB_THOLD` / `_SEG_MERGE_REPEATS` | 解码后清洗（截断幻觉长句、丢纯非语音段、合并重复） |
| `CRISPASR_VAD_FAILOVER=0` | 关闭"VAD 覆盖率过低则回退定长分块"的兜底 |
| `CRISPASR_CACHE_DIR` / `CRISPASR_MODELS_DIR` | 模型缓存根目录 |
| `HF_TOKEN` | 下载受限模型 |

## 返回码汇总

| 接口 | 码 |
|---|---|
| `crispasr_vad_slices_ex` | `n>0` 片段数；`0` 无语音；`-1` 参数非法；`-2` OOM；`-3` 模型加载失败 |
| `crispasr_vad_segments` | 同上，但无 `-3` 区分（加载失败也返回负值） |
| `crispasr_session_*` | 句柄返回 `NULL` = 打开/转写失败（异常信息里会附 `available_backends`） |
| `set_*` 类 | `rc=-2` 表示该后端不认这个字段（soft no-op，不抛异常） |
| `crispasr_diarize_segments_abi` | 非 0 = 失败；绑定封装为 `bool` 返回 + 原地写 `speaker` |
