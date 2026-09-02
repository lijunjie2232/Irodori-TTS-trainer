#!/usr/bin/env python
"""Convert speaker_inversion_example/data/speaker-inversion-dataset into the
manifest + DACVAE latents that 02_train.sh / train.py expect.

Source layout::

    <root>/<speaker>/<name>.mp3               long recording
    <root>/results/<speaker>/<name>.jsonl     VAD+ASR segments

Each jsonl line is {"audio_path", "text", "start_ms", "end_ms"}. The
`audio_path` prefix ("data/koyori/...") is stale, so the mp3 is resolved by
basename. Segments are cut with one ffmpeg seek per segment and encoded with
the same DACVAECodec prepare_manifest.py uses, so the output is field-for-field
identical to that script's::

    {"text": ..., "latent_path": "latents/00000000.pt", "num_frames": 140}

The ASR text is a sample's identity: the existing manifest is read back as a
text -> latent index, so extending or re-transcribing the dataset only encodes
what is genuinely new, and a text that appears twice in the source is written
once. `--overwrite` throws that index away and re-encodes everything.

Usage (from the repo root, or anywhere)::

    ./.venv.win/Scripts/python.exe speaker_inversion_example/scripts/convert_dataset.py \\
        --audio 閃光のハサウェイ --limit 8

`--limit` caps the rows written to the manifest (reused + newly encoded), not the
segments scanned, so a few oversize or empty leading segments do not eat your budget.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import DACVAECodec  # noqa: E402

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def probe_sample_rate(path: Path) -> int:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, check=True,
    ).stdout.decode().strip()
    return int(out.split(",")[0])


def cut_waveform(path: Path, start_ms: int, end_ms: int, sample_rate: int) -> torch.Tensor:
    """Decode [start_ms, end_ms) as mono float32 -> (1, 1, T). Native rate kept;
    the codec does the resample, exactly like prepare_manifest.py."""
    duration = (end_ms - start_ms) / 1000.0
    raw = subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error",
         "-ss", f"{start_ms / 1000.0:.3f}", "-t", f"{duration:.3f}",
         "-i", str(path), "-f", "f32le", "-ac", "1", "-"],
        capture_output=True, check=True,
    ).stdout
    wav = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
    # Guards the classic ffmpeg flag-order bug (silently empty / full-length output).
    assert abs(wav.numel() - duration * sample_rate) <= sample_rate, (
        f"ffmpeg cut mismatch: got {wav.numel()} samples, expected ~{duration * sample_rate}"
    )
    return wav.reshape(1, 1, -1)


def collect_segments(root: Path, speaker: str, audio_filter: str | None) -> list[dict]:
    result_dir = root / "results" / speaker
    files = sorted(result_dir.glob("*.jsonl"))
    if audio_filter:
        files = [f for f in files if audio_filter in f.name]
    if not files:
        raise SystemExit(f"no segments under {result_dir} matching {audio_filter!r}")

    audio_dir = root / speaker
    segments: list[dict] = []
    for f in files:
        mp3 = audio_dir / f"{f.stem}.mp3"
        if not mp3.is_file():
            raise SystemExit(f"audio not found for {f.name}: {mp3}")
        n = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["audio"] = mp3
            segments.append(item)
            n += 1
        print(f"[scan] {f.name}: {n} segments")
    return segments


def load_index(manifest: Path) -> dict[str, dict]:
    """text -> manifest row of a previous run, so its latent can be reused."""
    if not manifest.is_file():
        return {}
    index: dict[str, dict] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            index[row["text"]] = row
    return index


def next_latent_index(latent_dir: Path, index: dict[str, dict]) -> int:
    """Lowest latents/NNNNNNNN.pt number free both on disk and in the manifest.

    Above everything already on disk, not just above the manifest, so a manifest that
    was never written (or a stale one) can never make us clobber an existing latent.
    """
    used = {int(p.stem) for p in latent_dir.glob("*.pt") if p.stem.isdigit()}
    used.update(int(Path(row["latent_path"]).stem) for row in index.values()
                if Path(row["latent_path"]).stem.isdigit())
    return max(used, default=-1) + 1


def select_segments(segments, *, index, manifest_dir, min_seconds, max_seconds, stats):
    """Yield (item, text, reuse_row|None) in source order, one row per distinct ASR text.

    `reuse_row` is the manifest row of an already-encoded latent for that text; None means
    the item still needs encoding. Duplicate texts and out-of-range clips are counted in
    `stats` and dropped.
    """
    seen: set[str] = set()
    for item in segments:
        text = (item.get("text") or "").strip()
        seconds = (item["end_ms"] - item["start_ms"]) / 1000.0
        if not text or not (min_seconds <= seconds <= max_seconds):
            stats["filtered"] += 1
            continue
        if text in seen:
            stats["duplicate"] += 1
            continue
        seen.add(text)
        row = index.get(text)
        if row is not None and (manifest_dir / row["latent_path"]).is_file():
            yield item, text, row
        else:
            yield item, text, None


def prefetched(items, cut, workers):
    """Yield (item, text, reuse_row, waveform) in source order, `workers` ffmpeg cuts ahead.

    Exactly one of reuse_row / waveform is set: a reused row needs no waveform, a new one
    has already been cut by the time it is yielded.

    Cutting and encoding are serialized by a plain loop, and a good part of the encode's wall
    time is CPU anyway (ffmpeg seek, resample, loudness normalization), so the serial loop
    leaves the card half idle. Measured on a GTX 1050 Ti over 30 clips: 4.2-4.6x realtime at
    56-65% GPU utilisation serial, 7.1x realtime at 93% with 4 cuts in flight (1.6x). A T4
    encodes faster and so waits on ffmpeg even more.
    Batching clips into one encode call was measured at 1.07x *and* zero-padding a batch
    shifts the tail of every latent, so it is deliberately not done.
    """
    if workers <= 1:
        for item, text, row in items:
            yield item, text, row, None if row is not None else cut(item)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue: deque[tuple[dict, str, dict | None, Future | None]] = deque()
        source = iter(items)

        def fill() -> bool:
            while len(queue) < workers:
                try:
                    item, text, row = next(source)
                except StopIteration:
                    return False
                queue.append((item, text, row,
                              None if row is not None else pool.submit(cut, item)))
            return True

        # fill() first, so that a fully-reused run still yields every row.
        while fill() or queue:
            item, text, row, pending = queue.popleft()
            yield item, text, row, None if pending is None else pending.result()


def selftest() -> None:
    """Checks the two things that silently corrupt data: index allocation and text reuse."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "latents").mkdir()
        (root / "latents" / "00000000.pt").write_bytes(b"")
        (root / "latents" / "00000007.pt").write_bytes(b"")
        assert next_latent_index(root / "latents", {}) == 8, "must clear the files on disk"
        assert next_latent_index(root / "latents", {
            "a": {"latent_path": "latents/00000003.pt"}}) == 8, "must clear the manifest"
        assert next_latent_index(root / "latents", {
            "a": {"latent_path": "latents/00000009.pt"}}) == 10, "must clear both"

        segments = [
            {"text": "old", "start_ms": 0, "end_ms": 2000, "audio": root / "a.mp3"},
            {"text": "old", "start_ms": 2000, "end_ms": 4000, "audio": root / "a.mp3"},
            {"text": "new", "start_ms": 4000, "end_ms": 6000, "audio": root / "a.mp3"},
            {"text": "  ", "start_ms": 6000, "end_ms": 8000, "audio": root / "a.mp3"},
            {"text": "short", "start_ms": 8000, "end_ms": 8100, "audio": root / "a.mp3"},
        ]
        (root / "latents" / "00000000.pt").write_bytes(b"x")
        index = {"old": {"text": "old", "latent_path": "latents/00000000.pt", "num_frames": 50}}
        stats = {"filtered": 0, "duplicate": 0}
        got = [(t, r is not None) for _, t, r in select_segments(
            segments, index=index, manifest_dir=root, min_seconds=1.0, max_seconds=30.0,
            stats=stats)]
        assert got == [("old", True), ("new", False)], got
        assert stats == {"filtered": 2, "duplicate": 1}, stats

        # a text whose latent vanished from disk must be re-encoded, not reused
        (root / "latents" / "00000000.pt").unlink()
        assert [r is not None for _, _, r in select_segments(
            segments, index=index, manifest_dir=root, min_seconds=1.0, max_seconds=30.0,
            stats={"filtered": 0, "duplicate": 0})] == [False, False]

        # order is preserved with the prefetch pipeline, reused rows included
        items = [("i0", "t0", {"n": 0}), ("i1", "t1", None), ("i2", "t2", {"n": 2})]
        out = list(prefetched(items, lambda i: i, 2))
        assert [t for _, t, _, _ in out] == ["t0", "t1", "t2"], out
        assert out[0][2] == {"n": 0} and out[0][3] is None, out
        assert out[1][2] is None and out[1][3] == "i1", out
    print("[selftest] ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--dataset-root", type=Path,
                    default=here.parent / "data" / "speaker-inversion-dataset")
    ap.add_argument("--speaker", default="koyori", help="subdirectory name, also the results/ name")
    ap.add_argument("--audio", default=None, help="only use source files whose name contains this")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N manifest rows, reused or encoded (0 = all)")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=30.0, help="align with max_latent_steps")
    ap.add_argument("--manifest", type=Path, default=here.parent / "data" / "target_speaker_manifest.jsonl")
    ap.add_argument("--latent-dir", type=Path, default=here.parent / "data" / "latents")
    local_codec = REPO_ROOT / "models" / "Semantic-DACVAE-Japanese-32dim" / "weights.pth"
    ap.add_argument("--codec-repo", default=str(local_codec) if local_codec.is_file()
                    else "Aratako/Semantic-DACVAE-Japanese-32dim")
    ap.add_argument("--normalize-db", type=float, default=-16.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="ffmpeg cuts to run ahead of the GPU encode (1 = serial)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-encode every segment instead of reusing latents by ASR text")
    ap.add_argument("--selftest", action="store_true", help="check the reuse logic and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    # Latents cost minutes of GPU time to rebuild; never clobber them by accident.
    index = {} if args.overwrite else load_index(args.manifest)
    on_disk = list(args.latent_dir.glob("*.pt")) if args.latent_dir.is_dir() else []
    if on_disk and not index and not args.overwrite:
        raise SystemExit(
            f"{args.latent_dir} already holds {len(on_disk)} latents but {args.manifest} has no "
            "text -> latent index to reuse them by. Move them aside, or pass --overwrite."
        )

    segments = collect_segments(args.dataset_root, args.speaker, args.audio)
    if args.audio or args.limit:
        print(f"[warn] partial selection: {args.manifest.name} will list only the selected "
              "segments, so the next run without --audio/--limit re-encodes what this one "
              "leaves out. Fine for a smoke test, not for the manifest you upload.")

    args.latent_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    codec = DACVAECodec.load(repo_id=args.codec_repo, device=args.device,
                             normalize_db=args.normalize_db)
    print(f"[codec] sr={codec.sample_rate} latent_dim={codec.latent_dim} device={args.device}")

    rates: dict[Path, int] = {}

    def cut(item: dict) -> torch.Tensor:
        mp3 = item["audio"]
        if mp3 not in rates:  # ponytail: racy across cut threads, worst case one extra ffprobe
            rates[mp3] = probe_sample_rate(mp3)
        return cut_waveform(mp3, item["start_ms"], item["end_ms"], rates[mp3])

    stats = {"filtered": 0, "duplicate": 0}
    selected = select_segments(segments, index=index, manifest_dir=args.manifest.parent,
                               min_seconds=args.min_seconds, max_seconds=args.max_seconds,
                               stats=stats)
    next_free = next_latent_index(args.latent_dir, index)
    encoded = reused = 0
    rows = prefetched(selected, cut, args.prefetch)
    try:
        with args.manifest.open("w", encoding="utf-8") as out:
            for item, text, row, wav in rows:
                if args.limit and encoded + reused >= args.limit:
                    break
                if row is None:  # needs a fresh encode
                    latent = codec.encode_waveform(
                        wav, sample_rate=rates[item["audio"]])[0].cpu()
                    latent_path = args.latent_dir / f"{next_free:08d}.pt"
                    torch.save(latent, latent_path)
                    next_free += 1
                    # os.path.relpath (not Path.relative_to) so a relative --latent-dir works.
                    row = {"text": text,
                           "latent_path": Path(os.path.relpath(latent_path, args.manifest.parent)).as_posix(),
                           "num_frames": int(latent.shape[0])}
                    encoded += 1
                else:
                    reused += 1  # reused verbatim: already relative to the same manifest dir
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if (encoded + reused) % 200 == 0:
                    print(f"[convert] {encoded} encoded, {reused} reused", flush=True)
    finally:
        rows.close()  # waits for the in-flight cuts (at most --prefetch of them)

    print(f"[done] {encoded} encoded + {reused} reused -> {args.latent_dir}")
    print(f"[done] manifest -> {args.manifest} "
          f"(skipped {stats['filtered']} filtered, {stats['duplicate']} duplicate text)")


if __name__ == "__main__":
    main()
