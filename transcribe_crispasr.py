#!/usr/bin/env python
"""Transcribe audio with CrispASR's VAD + ASR called directly from Python.

No pybind11 and no `crispasr.exe` subprocess: CrispASR ships a pure-ctypes
binding (third_part/CrispASR/python/crispasr/_binding.py) that already exposes
everything needed — `vad_slices()` for segmentation and `Session.transcribe()`
for any ASR backend. This script loads that binding against the locally built
shared library and drives both stages in one process.

Two-pass contract (kept from the earlier pipeline): the per-audio `.csv`
(`start_s,end_s,text`) is the inter-stage handoff.

    # VAD only -> csv/jsonl with empty text
    python transcribe_crispasr.py -i data/origin/koyori -o data/transcripts --stage vad

    # ASR: read spans from the csv, fill the text, rewrite csv/jsonl, add srt
    python transcribe_crispasr.py -i data/origin/koyori -o data/transcripts --stage asr

    # or both at once (default)
    python transcribe_crispasr.py -i clip.wav -o out -m third_part/models/parakeet-tdt-0.6b-ja.gguf

Existing outputs are never touched without ``-w/--overwrite``; note that
``--stage vad -w`` re-segments and therefore clears the text column again.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent
CRISPASR_ROOT = REPO_ROOT / "third_part" / "CrispASR"
CRISPASR_PY_DIR = CRISPASR_ROOT / "python"
# Local cmake build trees, in the order a fresh checkout is most likely to hit.
LIB_RELATIVES = (
    Path("build_win") / "bin" / "Release" / "crispasr.dll",
    Path("build_linux_cu124") / "src" / "libcrispasr.so",
    Path("build_linux") / "src" / "libcrispasr.so",
    Path("build") / "src" / "libcrispasr.so",
    Path("build") / "lib" / "libcrispasr.so",
)
DEFAULT_ASR_MODEL = REPO_ROOT / "third_part" / "models" / "parakeet-tdt-0.6b-ja.gguf"
DEFAULT_VAD_MODEL = "firered-vad.gguf"
TARGET_SR = 16000
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".aiff")


# --------------------------------------------------------------------------- #
# ctypes binding bootstrap — env must be set before crispasr is imported
# --------------------------------------------------------------------------- #
def resolve_library_path(explicit: str | None) -> Path:
    """Locate the native libcrispasr shared library."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise SystemExit(f"--lib path does not exist: {p}")
        return p.resolve()

    env = os.environ.get("CRISPASR_LIB_PATH")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()

    for rel in LIB_RELATIVES:
        p = CRISPASR_ROOT / rel
        if p.exists():
            return p.resolve()

    raise SystemExit(
        "libcrispasr not found. Build it (see third_part/CrispASR) or pass "
        "--lib /path/to/crispasr.dll. Checked: "
        + ", ".join(str(CRISPASR_ROOT / r) for r in LIB_RELATIVES)
    )


def load_binding(lib_path: Path, device: str) -> Any:
    """Import CrispASR's ctypes binding against `lib_path`.

    The binding reads `CRISPASR_LIB_PATH` and registers the library folder (plus
    the CUDA `bin/`) with `os.add_dll_directory`, so both must be in place before
    the import — a late env change has no effect on an already-loaded library.
    """
    os.environ["CRISPASR_LIB_PATH"] = str(lib_path)
    if device == "cpu":
        # Session picks CUDA automatically when compiled in; hiding the device
        # is the only portable way to force the ASR side onto CPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if str(CRISPASR_PY_DIR) not in sys.path:
        sys.path.insert(0, str(CRISPASR_PY_DIR))
    mod = importlib.import_module("crispasr._binding")
    logger.debug(f"Loaded crispasr binding from {lib_path}")
    return mod


def resolve_model(name_or_path: str, fallbacks: tuple[str, ...] = ()) -> str:
    """Accept a real path, a bare filename in the CrispASR cache, or a repo-relative path."""
    p = Path(name_or_path).expanduser()
    if p.is_absolute() and p.exists():
        return str(p)
    for candidate in (p, REPO_ROOT / p):
        if candidate.exists():
            return str(candidate.resolve())

    cache = Path(os.environ.get("CRISPASR_CACHE_DIR") or Path.home() / ".cache" / "crispasr")
    for name in (p.name, *fallbacks):
        c = cache / name
        if c.exists():
            return str(c)
    raise SystemExit(
        f"Model {name_or_path!r} not found (file, repo-relative, or in cache {cache}). "
        "Download it via the CrispASR CLI once, or pass an explicit path."
    )


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def load_pcm(path: Path) -> np.ndarray:
    """Decode audio to contiguous mono float32 at 16 kHz."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    pcm = data.mean(axis=1)
    if sr != TARGET_SR:
        # Same linear interpolation CrispASR's own binding applies, so both
        # paths agree on sample positions.
        n = int(pcm.size * TARGET_SR / sr)
        idx = np.linspace(0, pcm.size - 1, n)
        pcm = np.interp(idx, np.arange(pcm.size), pcm)
    return np.ascontiguousarray(pcm, dtype=np.float32)


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def paths_for(out_dir: Path, audio: Path) -> tuple[Path, Path, Path]:
    stem = out_dir / audio.stem
    return Path(f"{stem}.csv"), Path(f"{stem}.jsonl"), Path(f"{stem}.srt")


def write_spans_csv(csv_path: Path, rows: list[tuple[float, float, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["start_s", "end_s", "text"])
        for start, end, text in rows:
            writer.writerow([f"{start:.3f}", f"{end:.3f}", text])


def read_spans_csv(csv_path: Path) -> list[tuple[float, float, str]]:
    """Read the inter-stage csv; header row is optional."""
    rows: list[tuple[float, float, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if not row or (i == 0 and row[0].lower().startswith("start")):
                continue
            text = row[2] if len(row) > 2 else ""
            rows.append((float(row[0]), float(row[1]), text))
    return rows


def write_jsonl(jsonl_path: Path, audio: Path, rows: list[tuple[float, float, str]]) -> None:
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for start, end, text in rows:
            rec = {
                "audio_path": str(audio).replace("\\", "/"),
                "start_ms": int(round(start * 1000)),
                "end_ms": int(round(end * 1000)),
                "text": text,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _srt_ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(srt_path: Path, rows: list[tuple[float, float, str]]) -> None:
    with srt_path.open("w", encoding="utf-8") as fh:
        for i, (start, end, text) in enumerate(rows, 1):
            if not text:
                continue
            fh.write(f"{i}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text}\n\n")


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
        if not files:
            raise SystemExit(f"No audio files found in {input_path}")
        return files
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    return [input_path]


def transcribe_span(sess: Any, pcm: np.ndarray, start: float, end: float,
                    language: str | None) -> str:
    a = max(0, int(start * TARGET_SR))
    b = min(pcm.size, int(end * TARGET_SR))
    if b - a < int(0.05 * TARGET_SR):  # < 50 ms is not worth a decode pass
        return ""
    segs = sess.transcribe(pcm[a:b], TARGET_SR, language=language)
    parts = [s.text.strip() for s in segs if s.text and s.text.strip()]
    return " ".join(parts)


def process_file(cb: Any, audio: Path, args: argparse.Namespace, out_dir: Path,
                 sess: Any) -> tuple[int, int]:
    """Run one audio file through the requested stage(s). Returns (spans, filled)."""
    csv_path, jsonl_path, srt_path = paths_for(out_dir, audio)
    existing = read_spans_csv(csv_path) if csv_path.exists() else []
    filled = sum(1 for _, _, t in existing if t)

    if csv_path.exists() and not args.overwrite:
        if args.stage == "vad" or (args.stage == "both" and filled):
            logger.info(f"Skip {audio.name}: {csv_path.name} exists "
                        f"({len(existing)} spans, {filled} transcribed); use -w to redo")
            return len(existing), filled
    if args.stage == "asr":
        if not existing:
            raise SystemExit(f"--stage asr needs {csv_path}; run --stage vad first "
                             "(or use --stage both)")
        if filled and not args.overwrite:
            logger.info(f"Skip {audio.name}: already transcribed ({filled} lines); use -w to redo")
            return len(existing), filled

    t_start = time.perf_counter()
    pcm = load_pcm(audio)
    duration_s = pcm.size / TARGET_SR
    rows = existing

    if args.stage in ("vad", "both") and (args.overwrite or not existing):
        vad_model = resolve_model(args.vad_model)
        use_gpu = {"cpu": 0, "cuda": 1, "auto": -1}[args.device]
        t0 = time.perf_counter()
        spans = cb.vad_slices(
            pcm, vad_model,
            sample_rate=TARGET_SR,
            threshold=args.threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            max_chunk_duration_s=args.max_span_s,
            n_threads=args.threads,
            use_gpu=use_gpu,
            batch_size=args.batch_size,
        )
        vad_s = time.perf_counter() - t0
        rows = [(s.start, s.end, "") for s in spans]
        logger.debug(f"{audio.name}: vad on {vad_model} ({args.device}, batch={args.batch_size}) "
                     f"-> {len(rows)} spans in {vad_s:.2f}s")
        write_spans_csv(csv_path, rows)
        write_jsonl(jsonl_path, audio, rows)
        logger.info(f"{audio.name}: VAD {len(rows)} spans / {duration_s:.1f}s audio "
                    f"in {vad_s:.2f}s")
    else:
        logger.debug(f"{audio.name}: reusing {len(rows)} spans from {csv_path.name}")

    if args.stage in ("asr", "both"):
        if sess is None:
            raise SystemExit("ASR session unavailable — check --model")
        language = None if args.language in ("", "auto") else args.language
        t0 = time.perf_counter()
        filled_rows: list[tuple[float, float, str]] = []
        for i, (start, end, text) in enumerate(rows, 1):
            if not text:
                text = transcribe_span(sess, pcm, start, end, language)
                logger.debug(f"{audio.name}: span {i}/{len(rows)} "
                             f"[{start:.2f}-{end:.2f}] -> {text[:60]}")
            filled_rows.append((start, end, text))
        asr_s = time.perf_counter() - t0
        rows = filled_rows
        written = sum(1 for _, _, t in rows if t)
        write_spans_csv(csv_path, rows)
        write_jsonl(jsonl_path, audio, rows)
        write_srt(srt_path, rows)
        logger.info(f"{audio.name}: ASR {written}/{len(rows)} lines in "
                    f"{asr_s:.2f}s ({duration_s / max(asr_s, 1e-6):.1f}x realtime)")

    logger.debug(f"{audio.name}: total {time.perf_counter() - t_start:.2f}s")
    return len(rows), sum(1 for _, _, t in rows if t)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VAD + ASR transcription through the CrispASR Python (ctypes) binding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=False, help="Audio file, or a directory of audio.")
    p.add_argument("-o", "--output", default="transcripts",
                   help="Output directory; per-audio <stem>.csv/.jsonl/.srt are written here.")
    p.add_argument("-m", "--model", default=str(DEFAULT_ASR_MODEL), help="ASR model (.gguf/.bin).")
    p.add_argument("-v", "--vad-model", default=DEFAULT_VAD_MODEL,
                   help="VAD model path or a filename inside the CrispASR cache.")
    p.add_argument("-l", "--language", default="ja", help="ISO language hint ('auto' to omit).")
    p.add_argument("-d", "--device", choices=("cuda", "cpu", "auto"), default="cuda",
                   help="Compute device for VAD; 'cpu' also hides CUDA from the ASR backend.")
    p.add_argument("-w", "--overwrite", action="store_true",
                   help="Redo work even when the output csv already exists.")
    p.add_argument("--stage", choices=("both", "vad", "asr"), default="both",
                   help="vad: spans only (empty text); asr: fill text from the existing csv.")
    p.add_argument("--lib", default=None, help="Explicit libcrispasr shared library.")
    p.add_argument("--threads", type=int, default=4, help="CPU threads per inference.")
    p.add_argument("--batch-size", type=int, default=0,
                   help="VAD windows per graph batch; 0 = auto (1 CPU / 32 GPU). "
                        "Silero crashes above 64.")
    p.add_argument("--threshold", type=float, default=0.5, help="VAD speech threshold (<=0: model default).")
    p.add_argument("--min-speech-ms", type=int, default=250)
    p.add_argument("--min-silence-ms", type=int, default=100)
    p.add_argument("--speech-pad-ms", type=int, default=30)
    p.add_argument("--max-span-s", type=float, default=30.0, help="Split speech longer than this.")
    p.add_argument("--list-backends", action="store_true",
                   help="Print ASR backends compiled into the library and exit.")
    p.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    p.add_argument("--debug", action="store_true", help="Verbose per-span debug logging.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logger.remove()
    level = "WARNING" if args.quiet else ("DEBUG" if args.debug else "INFO")
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")

    lib_path = resolve_library_path(args.lib)
    logger.debug(f"CrispASR library: {lib_path}")
    cb = load_binding(lib_path, args.device)

    if args.list_backends:
        backends = cb.Session.available_backends()
        logger.info(f"{len(backends)} backends compiled in: {', '.join(backends)}")
        return 0

    if not args.input:
        build_parser().error("-i/--input is required (unless --list-backends)")

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    asr_model = resolve_model(args.model)
    files = collect_inputs(Path(args.input).expanduser())

    sess = None
    if args.stage in ("asr", "both"):
        t0 = time.perf_counter()
        sess = cb.Session(asr_model, n_threads=args.threads)
        logger.info(f"ASR session open: {Path(asr_model).name} "
                    f"(backend={sess.backend}) in {time.perf_counter() - t0:.2f}s")

    total_spans = total_lines = 0
    try:
        for n, audio in enumerate(files, 1):
            logger.info(f"[{n}/{len(files)}] {audio.name}")
            spans, lines = process_file(cb, audio, args, out_dir, sess)
            total_spans += spans
            total_lines += lines
    finally:
        if sess is not None:
            sess.close()

    logger.info(f"Done: {len(files)} file(s), {total_spans} span(s), "
                f"{total_lines} transcribed line(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
