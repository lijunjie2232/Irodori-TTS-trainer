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

Usage (from the repo root, or anywhere)::

    ./.venv.win/Scripts/python.exe speaker_inversion_example/scripts/convert_dataset.py \\
        --audio 閃光のハサウェイ --limit 8

`--limit` counts written latents, not scanned segments, so a few oversize or
empty leading segments do not eat your budget.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--dataset-root", type=Path,
                    default=here.parent / "data" / "speaker-inversion-dataset")
    ap.add_argument("--speaker", default="koyori", help="subdirectory name, also the results/ name")
    ap.add_argument("--audio", default=None, help="only use source files whose name contains this")
    ap.add_argument("--limit", type=int, default=0, help="stop after N written latents (0 = all)")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=30.0, help="align with max_latent_steps")
    ap.add_argument("--manifest", type=Path, default=here.parent / "data" / "target_speaker_manifest.jsonl")
    ap.add_argument("--latent-dir", type=Path, default=here.parent / "data" / "latents")
    local_codec = REPO_ROOT / "models" / "Semantic-DACVAE-Japanese-32dim" / "weights.pth"
    ap.add_argument("--codec-repo", default=str(local_codec) if local_codec.is_file()
                    else "Aratako/Semantic-DACVAE-Japanese-32dim")
    ap.add_argument("--normalize-db", type=float, default=-16.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow reusing a latent dir / manifest that already has data")
    args = ap.parse_args()

    # Latents cost minutes of GPU time to rebuild; never clobber them by accident.
    existing = list(args.latent_dir.glob("*.pt")) if args.latent_dir.is_dir() else []
    if existing and not args.overwrite:
        raise SystemExit(
            f"{args.latent_dir} already holds {len(existing)} latents. "
            "Pass --overwrite (and move the old manifest aside first) if that is intended."
        )

    segments = collect_segments(args.dataset_root, args.speaker, args.audio)

    args.latent_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    codec = DACVAECodec.load(repo_id=args.codec_repo, device=args.device,
                             normalize_db=args.normalize_db)
    print(f"[codec] sr={codec.sample_rate} latent_dim={codec.latent_dim} device={args.device}")

    rates: dict[Path, int] = {}
    written = skipped = 0
    with args.manifest.open("w", encoding="utf-8") as out:
        for item in segments:
            if args.limit and written >= args.limit:
                break
            text = (item.get("text") or "").strip()
            seconds = (item["end_ms"] - item["start_ms"]) / 1000.0
            if not text or not (args.min_seconds <= seconds <= args.max_seconds):
                skipped += 1
                continue

            mp3 = item["audio"]
            if mp3 not in rates:
                rates[mp3] = probe_sample_rate(mp3)
            wav = cut_waveform(mp3, item["start_ms"], item["end_ms"], rates[mp3])

            latent = codec.encode_waveform(wav, sample_rate=rates[mp3])[0].cpu()
            latent_path = args.latent_dir / f"{written:08d}.pt"
            torch.save(latent, latent_path)
            # os.path.relpath (not Path.relative_to) so a relative --latent-dir still works.
            rel = Path(os.path.relpath(latent_path, args.manifest.parent)).as_posix()
            out.write(json.dumps({"text": text, "latent_path": rel,
                                  "num_frames": int(latent.shape[0])},
                                 ensure_ascii=False) + "\n")
            written += 1
            if written % 20 == 0:
                print(f"[convert] {written} written ({seconds:.1f}s this clip)", flush=True)

    print(f"[done] wrote {written} latents -> {args.latent_dir}")
    print(f"[done] manifest -> {args.manifest} (skipped {skipped} segments)")


if __name__ == "__main__":
    main()
