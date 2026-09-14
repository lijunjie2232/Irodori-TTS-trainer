---
name: crispasr-python-binding
description: 通过 CrispASR 自带的 ctypes 绑定在 Python 进程内直接调用其 C 接口（VAD 与 ASR），覆盖 vad_slices / Session.transcribe / transcribe_vad 的参数语义、设备与批处理规则、环境变量加载顺序、Python↔C 符号对照与故障排查。当需要在 Irodori-TTS 中做语音转写、VAD 切分、加载 crispasr 共享库、或判断"是否要写 pybind11 绑定"时使用。
---

# CrispASR Python 调用 C 接口（ctypes）

## 首要事实

**CrispASR 没有 pybind11 / nanobind。** Python 层是纯 `ctypes`，位于
`third_part/CrispASR/python/crispasr/_binding.py`（约 4800 行），已把需要的 C ABI 全部包好。
不要新增编译期绑定，也不要 `subprocess` 调 `crispasr.exe`——直接 `ctypes.CDLL` 加载本地已编译的共享库即可。

`third_part/` 下的源码一律**只读**，需要扩展就在项目根写脚本调用其公共 API。

| 项目 | 值 |
|---|---|
| 绑定目录 | `third_part/CrispASR/python`（未 pip 安装，需插 `sys.path`） |
| Windows 库 | `third_part/CrispASR/build_win/bin/Release/crispasr.dll` |
| Linux 库 | `third_part/CrispASR/build_linux_cu124/src/libcrispasr.so`、`build_linux/src/libcrispasr.so` |
| 运行解释器 | `.venv.win/Scripts/python.exe`（已含 numpy / soundfile / loguru） |
| 库搜索开关 | 环境变量 `CRISPASR_LIB_PATH`（**必须在 import 之前设置**） |

## Step 1：Bootstrap（顺序错误必然失败）

```python
import importlib, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
CRISP = REPO / "third_part" / "CrispASR"

# 1) 先定设备：encdec VAD 在库首次初始化时读 CRISPASR_VAD_ENCDEC_CPU，之后再改无效
os.environ["CRISPASR_VAD_ENCDEC_CPU"] = "1"        # 仅当需要强制该模型走 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # 仅当 ASR 也要强制 CPU

# 2) 再指库
os.environ["CRISPASR_LIB_PATH"] = str(CRISP / "build_win/bin/Release/crispasr.dll")

# 3) 最后导入
sys.path.insert(0, str(CRISP / "python"))
cb = importlib.import_module("crispasr._binding")   # 动态导入，避免静态解析报错

vad_slices = cb.vad_slices          # ← 只在 _binding 里，crispasr/__init__.py 没导出
Session = cb.Session
```

三个高频坑：
1. `module 'crispasr' has no attribute 'vad_slices'` → 必须 `from crispasr._binding import vad_slices`；
   `Session` / `CrispASR` / `detect_language_pcm` / `diarize_segments` 等 `__init__.py` 里有导出，两处皆可。
2. Windows 报 `DLL load failed while importing`（但依赖 DLL 都在同目录）→ 绑定的
   `_register_dll_dir()` 会 `os.add_dll_directory()` 注册库目录与 `CUDA_PATH_V12_1/bin`，
   前提是 `CRISPASR_LIB_PATH` 指向了 `.dll` 而非裸文件名。
3. `crispasr_vad_slices_ex missing from the loaded library` → 共享库版本过旧，需重新 build。

## Step 2：音频输入约定

所有接口要求 **单声道 float32 16 kHz PCM**；`n_threads` 只管 CPU 侧，GPU 路径不受影响。

```python
import numpy as np, soundfile as sf

def load_pcm(path, sr_dst=16000):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    pcm = data.mean(axis=1)                     # 多声道取均值
    if sr != sr_dst:                            # 与绑定内部一致的线性插值，别引入第二套重采样
        idx = np.linspace(0, pcm.size - 1, int(pcm.size * sr_dst / sr))
        pcm = np.interp(idx, np.arange(pcm.size), pcm)
    return np.ascontiguousarray(pcm, dtype=np.float32)   # ctypes 取指针前必须 contiguous
```

`Session.transcribe` 收到 `sample_rate != 16000` 时会自己重采样，但**逐段切片场景不要依赖它**：
先统一重采样再切片，否则每段都会被二次重采样（质量损失 + 时间戳漂移）。

## Step 3：VAD —— `vad_slices`（C: `crispasr_vad_slices_ex`）

```python
spans = cb.vad_slices(
    pcm, str(vad_model_path),
    sample_rate=16000, threshold=0.5,
    min_speech_ms=250, min_silence_ms=100, speech_pad_ms=30,
    max_chunk_duration_s=30.0, n_threads=4,
    use_gpu=1, batch_size=0, lib_path=None,
)   # -> List[VadSpan(start, end)]，单位秒
```

参数语义（来自 `crispasr_c_api.cpp`，与直觉不同处已标注）：

| 参数 | C 参数 | 语义 |
|---|---|---|
| `threshold` | `float` | **`<=0` 表示"用该模型自带默认"**；`>0` 才置 `threshold_explicit` |
| `min_speech_ms` / `min_silence_ms` | `int` | `<=0` 同样落回库默认（250 / 100） |
| `speech_pad_ms` | `int` | padding 由 **dispatcher 统一施加**（`opts.speech_pad_ms` 被强制清 0），但 Whisper-encDec 内部忽略它 |
| `max_chunk_duration_s` | `float` | `>0` 时 `chunk_seconds = ceil(v)`；`=0` 表示不切长句 |
| `use_gpu` | `int` | `>0` 强制 GPU，`0` 强制 CPU，**`<0` 才回退读 `CRISPASR_VAD_GPU`** |
| `batch_size` | `int` | `0` = auto（CPU 1 / GPU 32），`>0` 显式指定 |
| 返回值 | `int` | `n>0` 片段数；`0` = 加载成功但没检出语音；`-1` 参数非法；`-2` 内存分配失败；**-3 模型加载失败** |

### 后端按文件名分派（`crispasr_vad.cpp`）

| 模型文件名匹配 | 实现 | GPU | batch_size | 备注 |
|---|---|---|---|---|
| basename 含 `firered` + `vad` | FireRed DFSMN | ✅ | 图批，默认 `CRISPASR_FIRERED_VAD_GRAPH_BATCH=4096` | 本机默认推荐 |
| 含 `marblenet` + `.gguf` | MarbleNet | ✅ | 按输出帧切分 | **长度敏感**：`batch>0` 会改变 spans 结果 |
| 含 `whisper` + `vad` + `.gguf` | Whisper encDec | ✅ | **忽略**（30s 窗口固定 1500 帧） | `threshold==0.5` 且未显式指定时**自动降到 0.30** |
| 含 `webrtc` | 纯 C GMM | ❌ | 忽略 | 无需模型文件，最快、精度最低 |
| 其它（如 `ggml-silero-*.bin`） | Silero | ✅ | **>64 触发 `GGML_ASSERT(obj_new)` 崩溃** | 仅 `<=64` 稳定 |

- C++ 侧 context 缓存 key 是 `(path, use_gpu, batch_size)`：同一路径切换设备/batch 会**重建 ggml context**（首次调用慢是正常的）。
- 想保持 encdec 的 0.5 阈值必须传非 0.5 的值（如 `0.51`），否则被自动下调。

本机实测（31.5 s 音频，FireRed）：GPU 0.18 s，CPU 3.77 s。

## Step 4：ASR —— `Session`（C: `crispasr_session_*`）

```python
sess = cb.Session(str(model_path), n_threads=4)      # backend 从 GGUF 自动探测
print(sess.backend)                                  # 'parakeet' / 'whisper' / 'sensevoice' ...
segs = sess.transcribe(pcm, 16000, language="ja")    # -> List[SessionSegment]
sess.close()                                         # 或 with cb.Session(...) as sess:
```

- `Session(model_path, lib_path=None, n_threads=4, backend=None)`：`backend` 显式指定可绕过探测
  （C: `crispasr_session_open_explicit`）。
- **Whisper 的 `.bin` 也能用 `Session`**：`crispasr_session_open` 在 GGUF 探测失败后检查 magic
  `lmgg`/`ggjt` 并落到 whisper 后端，无需再用旧的 `CrispASR` 类。
- `language` 只有 whisper / canary / cohere / voxtral 系会采纳，其余后端静默忽略（不是 bug）。
- 编译进来的后端清单用 `cb.Session.available_backends()` 自查（本机 113 个，含
  `whisper`、`parakeet`、`sensevoice`、`paraformer`、`firered-asr`）。

三种转写路径的选择：

| 方法 | C 符号 | 用于 |
|---|---|---|
| `transcribe(pcm, sr, *, language)` | `crispasr_session_transcribe[_lang]` | 已按 VAD 切好的短片段（**默认选它**） |
| `transcribe_vad(pcm, vad_model, *, threshold, min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms, chunk_seconds, n_threads, language)` | `crispasr_session_transcribe_vad[_lang]` | 一次调用搞定 VAD+拼接转写：片段被缝成单 buffer（0.1 s 静音间隔）后解码一次，再映射回原时间轴；保留跨片段上下文，利于 O(T²) 后端；VAD 加载失败会**静默退化为普通 transcribe** |
| `transcribe_chunked(pcm, chunk_seconds=0, overlap_seconds=-1, *, progress=fn)` | `crispasr_session_transcribe_chunked_lang` | 超长音频；仅对 Parakeet 生效，其余等价 `transcribe`；配 `sess.get_progress()` 轮询 0..100 |

`SessionSegment(text, start, end, words, no_speech_prob, speaker)`：
`words` 为 `List[SessionWord(text,start,end,confidence)]`，后端不产词级时间戳时为 `[]`；
`no_speech_prob` 仅 Whisper 有值，其它后端为哨兵 `-1.0`；`speaker` 目前只有 vibevoice 有，且是**chunk 内局部编号**，跨录音一致性问题请用 `diarize_*`。

常用 ASR 侧 setter（不存在的字段对应后端会 soft no-op，`rc=-2`）：
`set_source_language` / `set_target_language` / `set_translate` / `set_hotwords(hotwords, boost=2.0)` /
`set_punctuation(bool)` / `set_punc_model(path)` / `set_temperature(t, seed=0)` / `set_best_of` /
`set_beam_size` / `set_max_new_tokens` / `set_frequency_penalty` / `set_sensitivity(preset)` /
`set_fallback_thresholds(entropy, logprob, no_speech, temperature_inc)`（`temperature_inc=0` 关闭回退）/
`set_grammar_text(gbnf)`（whisper 约束解码）/ `set_return_logits(bool)`。

## Step 5：典型工作流（VAD → 逐段 ASR）

项目内已有可直接运行的完整实现：`transcribe_crispasr.py`（支持 `-i/-o/-m/-v/-l/-d/-w`、
`--stage vad|asr|both`、输出 `.csv/.jsonl/.srt`）。复用它的两段式契约：

- `--stage vad` 只写 `start_s,end_s,text`（text 留空）到 `.csv`，**`.csv` 就是跨阶段契约文件**；
- `--stage asr` 读回 `.csv` 的 spans，逐段 `Session.transcribe`，重写 `.csv/.jsonl` 并生成 `.srt`；
- 输出已存在时不加 `-w` 必须跳过（续跑语义）；注意 `--stage vad -w` 会重新切分并把 text 清空。

逐段调用的最小正确写法：

```python
for start, end in spans:
    a, b = int(start * 16000), min(pcm.size, int(end * 16000))
    if b - a < 800:            # <50ms 不值得解码
        continue
    text = " ".join(s.text.strip() for s in sess.transcribe(pcm[a:b], 16000, language="ja")).strip()
```

## 故障速查

| 现象 | 原因 / 处置 |
|---|---|
| `has no attribute 'vad_slices'` | 从 `crispasr._binding` 导入，不是 `crispasr` |
| `Failed to open ... backend not supported` | 该 GGUF 的后端没编进库；看异常里的 `available_backends`，或换 quant/换模型 |
| `vad_slices` 返回 `RuntimeError(rc=-3)` | 模型文件路径不对或该后端未编译（marblenet/encdec 受 `CA_HAVE_*` 编译开关控制） |
| VAD 返回 0 片段但音频确有语音 | 阈值过高（MarbleNet 常见）→ 调 `threshold`，或换 FireRed |
| Silero 大 batch 直接 abort | 已知库 bug（ggml LSTM buffer 分配），batch 保持 `<=64` 或 `0` |
| 转写文本重复/幻觉长句 | 走 `Session.transcribe` 时靠库侧 hygiene（`CRISPASR_SEG_MAX_CHARS` 等，见 reference.md） |
| 性能测试数字异常波动 | ggml 有进程级缓存，不同设备/batch 的组合必须**各起独立子进程**测 |

## 进一步内容

- 扩展接口（LID、说话人分离、强制对齐、标点恢复、流式/麦克风、TTS、Chat、模型下载、水印）与 C 符号/环境变量全表：
  见 [reference.md](reference.md)
- CrispASR 官方文档：`third_part/CrispASR/docs/bindings.md`、`docs/environment-variables.md`
