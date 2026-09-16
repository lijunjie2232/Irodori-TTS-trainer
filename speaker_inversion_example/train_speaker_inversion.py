#!/usr/bin/env python3
"""Speaker Inversion — end-to-end (clone → convert → train → sync)

Trains a **Speaker Inversion** embedding for one character of the
[`l2533584225/Inversion-speaker-dataset`](https://huggingface.co/datasets/l2533584225/Inversion-speaker-dataset)
against the frozen [`Aratako/Irodori-TTS-v4.1-Small`](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)
base model. The base model never changes; only a `(16, 768)` token table is learned.

What this script does, in order:

1. installs the system deps (ffmpeg) and installs the Python env with `uv pip install --system`
   (`.[cu128]`, T4-friendly)
2. clones [Irodori-TTS-trainer](https://github.com/lijunjie2232/Irodori-TTS-trainer)
3. downloads the dataset, the base checkpoint and this character's previous checkpoints
4. unpacks `latents.tar` — the Hub copy of `latents/` is one tar, because a dataset repo may
   not hold more than 10 000 files in a single directory
5. converts the VAD/ASR segments into DACVAE latents + a training manifest, written to
   `<dataset>/<character>/converted/`. **The ASR text identifies a sample**, so a re-run only
   encodes what is new since the last conversion
6. verifies the manifest against the latents on disk, then pushes `converted/` back to the
   dataset repo
7. runs Speaker Inversion training — single process on 1 GPU, DDP on 2 — uploading the
   checkpoints to `<model repo>/<character>/` on a timer while it runs
8. optionally continues from a checkpoint a previous run left in that model repo

Every knob lives in the **CONFIG** block below: paths, HF repos, convert args, train args.
Nothing else needs editing.

Intended runtime: **GPU (T4)**. Set `UPDATE_DATASET` / `SYNC_CKPT` to False for a read-only run.

Usage:
    python speaker_inversion_example/train_speaker_inversion.py        # every step, in order
    python speaker_inversion_example/train_speaker_inversion.py 4 7 8  # just those steps

Mirrors `train_speaker_inversion.ipynb`, one section per cell group, and consumes the
`results/<character>/*.jsonl` segments that `speaker-inversion-dataset/vad_asr.ipynb` produces.
"""

# ===========================================================================
# CONFIG — every knob this script uses. Nothing below needs editing.
# ===========================================================================
import json
import os
import shlex
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# ---- repo ----------------------------------------------------------------
REPO_URL    = "https://github.com/lijunjie2232/Irodori-TTS-trainer"
REPO_BRANCH = "main"
# The notebook hardcodes the Colab path; here it is overridable so the script can also run from
# an existing checkout: REPO_DIR=/path/to/repo python .../train_speaker_inversion.py
REPO_DIR    = Path(os.environ.get("REPO_DIR") or "/content/Irodori-TTS-trainer")
UV_EXTRA    = "cu128"        # T4 is sm_75, covered by the cu128 wheels

# ---- character -----------------------------------------------------------
CHARACTER = "koyori"         # subdirectory under the dataset root

# ---- HuggingFace credentials --------------------------------------------
HF_TOKEN        = ""            # paste a write token here to skip the Colab secret
HF_TOKEN_SECRET = "HF_TOKEN"    # Colab secret name, read when HF_TOKEN is ""

# ---- dataset (HuggingFace) ----------------------------------------------
HF_DATASET_REPO = "l2533584225/Inversion-speaker-dataset"
UPDATE_DATASET  = True         # True -> upload <dataset>/<character>/converted to HF
# <character>/* = source audio, results/<character>/* = the VAD+ASR segments,
# <character>/converted/* = manifest.jsonl + latents.tar of the last conversion
DATASET_INCLUDE = [f"{CHARACTER}/*", f"results/{CHARACTER}/*", f"{CHARACTER}/converted/*"]

# ---- models --------------------------------------------------------------
BASE_MODEL_REPO = "Aratako/Irodori-TTS-v4.1-Small"
CODEC_REPO      = "Aratako/Semantic-DACVAE-Japanese-32dim"   # or a local weights.pth path

# ---- checkpoints (HuggingFace model repo, one directory per character) ----
HF_MODEL_REPO = "l2533584225/irodori_model"   # "" -> neither download nor upload
SYNC_CKPT     = True          # upload outputs/<character>/ to HF_MODEL_REPO while training
SYNC_EVERY    = 120           # seconds between those uploads
CONTINUE_FROM = ""            # e.g. "checkpoint_final.speaker.safetensors", inside <character>/

# ---- convert args (speaker_inversion_example/scripts/convert_dataset.py) --
CONVERT_AUDIO_FILTER = ""       # substring of the source mp3 name; "" = every file
CONVERT_LIMIT        = 0        # 0 = all segments; >0 = first N manifest rows (smoke test)
CONVERT_MIN_SECONDS  = 1.0
CONVERT_MAX_SECONDS  = 30.0     # keep <= max_latent_steps / LATENT_FPS of the training config
CONVERT_NORMALIZE_DB = -16.0
CONVERT_DEVICE       = "cuda"
CONVERT_PREFETCH     = 4        # ffmpeg cuts to run ahead of the GPU encode (1 = serial)
CONVERT_OVERWRITE    = False    # True -> re-encode everything instead of reusing by ASR text
SKIP_CONVERT         = True     # True -> no encoding: unpack the Hub copy, verify it, train

# ---- train args (train.py; these override configs/speaker_inversion.yaml) --
TRAIN_BATCH_SIZE  = 16          # PER PROCESS, so x2 on a 2-GPU machine
TRAIN_GRAD_ACCUM  = 1
TRAIN_MAX_STEPS   = 3000
TRAIN_LR          = 0.01        # high on purpose: only 16x768 params are trainable
TRAIN_SAVE_EVERY  = 250
TRAIN_LOG_EVERY   = 20
TRAIN_NUM_WORKERS = 2
TRAIN_PRECISION   = "fp16"      # T4 (sm_75) has no bf16: train.py warns and falls back to fp32
TRAIN_NPROC       = 1           # 0 = auto (1 process per visible GPU), 1 = force single process

# ---- derived (do not edit) ----------------------------------------------
EXAMPLE     = REPO_DIR / "speaker_inversion_example"
DATASET     = EXAMPLE / "data" / "speaker-inversion-dataset"
CONVERT     = DATASET / CHARACTER / "converted"
MANIFEST    = CONVERT / "manifest.jsonl"
LATENTS     = CONVERT / "latents"
LATENTS_TAR = CONVERT / "latents.tar"    # the Hub copy: 10k+ loose files is over its limit
REPORT      = CONVERT / "convert_report.json"
CONFIG_YAML = EXAMPLE / "configs" / "speaker_inversion.yaml"
BASE_CHECKPOINT = REPO_DIR / "models" / "Irodori-TTS-v4.1-Small" / "model.safetensors"
MODEL_REPO  = REPO_DIR / "models" / HF_MODEL_REPO.split("/")[-1] if HF_MODEL_REPO else None
OUTPUT_DIR  = EXAMPLE / "outputs" / CHARACTER
LATENT_FPS  = 25.0              # DACVAE: 48 kHz / 1920-sample hop

RUN = [sys.executable]   # the interpreter running this script, which section 8 also trains in


def run(*args, cwd=None, check=True):
    """Echo and run a command, printing its output while it runs, not only when it exits.

    cwd falls back to the repo, and to the CWD before the clone. Merging stderr into one pipe
    keeps the two interleaved in the order they happened; PYTHONUNBUFFERED stops a python child
    from block-buffering its own stdout, which is what makes the streaming visible at all.
    """
    workdir = Path(cwd or REPO_DIR)
    cmd = [str(a) for a in args]
    print(f"$ {shlex.join(cmd)}" + (f"   (cwd={workdir})" if cwd else ""), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir if workdir.is_dir() else Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    returncode = proc.wait()
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, cmd)


def hf_token():
    """Write token: the HF_TOKEN variable if it is set, else the Colab secret of that name."""
    if HF_TOKEN:
        return HF_TOKEN
    from google.colab import userdata

    return userdata.get(HF_TOKEN_SECRET)


def use_hf_token():
    """Export HF_TOKEN for the `hf` CLI, which reads the environment and not this variable."""
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token()
        source = "the HF_TOKEN variable" if HF_TOKEN else f"the {HF_TOKEN_SECRET} secret"
        print(f"[hf] token from {source}")


def manifest_records():
    """Rows of manifest.jsonl, or [] when no conversion has produced one yet."""
    if not MANIFEST.is_file():
        return []
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify_converted():
    """Assert that manifest.jsonl and latents/ agree, and return the rows.

    Cheap, and the only thing standing between a half-unpacked tar and a training run that
    dies 200 steps in.
    """
    records = manifest_records()
    assert records, (f"{MANIFEST} is missing or empty — run the convert step, or let the Hub "
                     "copy be unpacked here.")
    missing = [r["latent_path"] for r in records
               if not (MANIFEST.parent / r["latent_path"]).is_file()]
    assert not missing, (f"{len(missing)} of {len(records)} latents referenced by "
                         f"{MANIFEST.name} are missing, e.g. {missing[:3]}")
    assert all(r["num_frames"] > 0 for r in records), "manifest contains an empty latent"
    assert len({r["latent_path"] for r in records}) == len(records), "duplicate latent_path in manifest"
    return records


def unpack_latents():
    """Extract latents.tar when the manifest points at files that are not on disk yet.

    Skipped when everything the manifest references is already there, so re-running is free.
    The members are `latents/NNNNNNNN.pt`, so it extracts into the converted/ directory.
    """
    if not LATENTS_TAR.is_file():
        return
    records = manifest_records()
    if records and all((MANIFEST.parent / r["latent_path"]).is_file() for r in records):
        print(f"[tar] {LATENTS} is already complete ({len(records)} latents)")
        return
    print(f"[tar] unpacking {LATENTS_TAR.name} ({LATENTS_TAR.stat().st_size / 1e9:.2f} GB)")
    run("tar", "-xf", LATENTS_TAR.name, cwd=CONVERT)


# ===========================================================================
# 1. System dependencies
#
# `ffmpeg` / `ffprobe` are what `convert_dataset.py` uses to cut the segments out of the long
# recordings. Colab already runs as root, so `apt-get` works without `sudo`. `uv` ships with Colab.
# ===========================================================================
def step_system_deps():
    run("bash", "-c", "apt-get -qq update && apt-get -qq install -y ffmpeg")
    run("uv", "--version")
    run("bash", "-c", "ffmpeg -version | head -1")


# ===========================================================================
# 2. Clone the trainer
#
# `--recursive` is deliberately **not** used: `third_part/*` (CrispASR / FunASR) is only needed by
# the VAD/ASR stage, not by Speaker Inversion training.
# ===========================================================================
def step_clone():
    if (REPO_DIR / ".git").is_dir():
        run("git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", REPO_BRANCH)
        run("git", "-C", REPO_DIR, "checkout", "FETCH_HEAD")
    else:
        run("git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, REPO_DIR)


# ===========================================================================
# 3. Python environment
#
# Installed into the **running interpreter** (`uv pip install --system`) rather than a
# `<repo>/.venv`, because section 8 trains in this process and has to see exactly what
# `train.py` sees. `torch` / `torchaudio` resolve from the `pytorch-cu128` index and `dacvae`
# from git — both declared in the repo's `pyproject.toml`, which `uv` honours when it installs
# `-e .`.
# ===========================================================================
def step_python_env():
    run("uv", "pip", "install", "--system", "--break-system-packages", "-e", ".", f"[{UV_EXTRA}]")
    run(*RUN, "-c",
        "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda,"
        " '| available', torch.cuda.is_available(), '|', torch.cuda.device_count(), 'GPU(s)')")


# ===========================================================================
# 4. Data + models
#
# The dataset and the base checkpoint are gitignored in the trainer repo (root `.gitignore`
# ignores `speaker_inversion_example/data` and `models/`), so they are fetched from HuggingFace
# separately. The `koyori/*.mp3` set alone is ~4 GB — narrow `DATASET_INCLUDE` for a quick pass.
#
# `<character>/converted/*` brings back the manifest and the `latents.tar` of the previous
# conversion (section 5 unpacks it), and `HF_MODEL_REPO` brings back this character's earlier
# checkpoints.
# ===========================================================================
def step_download():
    DATASET.mkdir(parents=True, exist_ok=True)
    include = [flag for pattern in DATASET_INCLUDE for flag in ("--include", pattern)]
    run("uvx", "hf", "download", "--type=dataset", HF_DATASET_REPO, "--local-dir", DATASET, *include)

    # 3 GB base checkpoint the embedding is trained against; ~/.cache/huggingface is reused.
    run("uvx", "hf", "download", BASE_MODEL_REPO,
        "--local-dir", REPO_DIR / "models" / "Irodori-TTS-v4.1-Small")

    # Checkpoints of earlier runs for this character. check=False: the repo may not exist yet.
    if MODEL_REPO is not None:
        if SYNC_CKPT:
            use_hf_token()
        run("uvx", "hf", "download", HF_MODEL_REPO, "--local-dir", MODEL_REPO,
            "--include", f"{CHARACTER}/*", check=False)
        if (MODEL_REPO / CHARACTER).is_dir():
            print(f"\ncheckpoints: {sorted(p.name for p in (MODEL_REPO / CHARACTER).iterdir())}")

    print("\ndataset root:", sorted(p.name for p in DATASET.iterdir()))


# ===========================================================================
# 5. Unpack the latents
#
# `latents/` holds more files than a Hub repository allows in one directory, so the converted
# directory is uploaded with `latents/` packed into `latents.tar`. Extracting is skipped when
# every latent the manifest points at is already on disk, so re-running costs nothing.
# ===========================================================================
def step_unpack():
    unpack_latents()


# ===========================================================================
# 6. Convert segments → latents + manifest
#
# `convert_dataset.py` resolves each mp3 by file name (the `audio_path` field in the jsonl carries
# a stale `data/` prefix), cuts `[start_ms, end_ms)` with one ffmpeg seek per segment, and encodes
# it with the same `DACVAECodec` as `prepare_manifest.py`. Output is field-for-field the manifest
# `train.py` expects:
#
#     <dataset>/<character>/converted/
#       manifest.jsonl          {"text", "latent_path": "latents/00000000.pt", "num_frames"}
#       latents/00000000.pt     (T, 32) float32
#       convert_report.json     written in section 7
#
# The **ASR text identifies a sample**: latents already in the manifest are reused and only new
# texts are encoded, so extending or re-transcribing the dataset costs only the new clips.
# `CONVERT_OVERWRITE = True` (or a changed `CONVERT_NORMALIZE_DB`) re-encodes everything.
#
# `CONVERT_PREFETCH` keeps ffmpeg decoding the next few clips while the GPU encodes the current
# one: a serial loop leaves the card around 55% utilisation, four cuts in flight push it to ~93%
# (1.6x, measured). `SKIP_CONVERT = True` skips the encoding entirely and trains on the Hub copy —
# section 5 still unpacks it and section 7 still verifies it.
# ===========================================================================
def convert_args():
    args = [
        "--dataset-root", str(DATASET),
        "--speaker", CHARACTER,
        "--manifest", str(MANIFEST),
        "--latent-dir", str(LATENTS),
        "--codec-repo", CODEC_REPO,
        "--min-seconds", CONVERT_MIN_SECONDS,
        "--max-seconds", CONVERT_MAX_SECONDS,
        "--normalize-db", CONVERT_NORMALIZE_DB,
        "--device", CONVERT_DEVICE,
        "--prefetch", CONVERT_PREFETCH,
    ]
    if CONVERT_AUDIO_FILTER:
        args += ["--audio", CONVERT_AUDIO_FILTER]
    if CONVERT_LIMIT:
        args += ["--limit", CONVERT_LIMIT]
    if CONVERT_OVERWRITE:
        args += ["--overwrite"]
    return args


def step_convert():
    args = convert_args()
    if SKIP_CONVERT:
        print(f"SKIP_CONVERT — keeping the {len(manifest_records())} latents already in {CONVERT}")
    else:
        run(*RUN, EXAMPLE / "scripts" / "convert_dataset.py", *args)


# ===========================================================================
# 7. Verify + metadata + dataset upload
#
# `verify_converted()` is the cheap sanity check that the manifest and the latents agree.
#
# `convert_report.json` records the conversion provenance so the `converted/` directory is
# self-describing once it lands on the Hub.
#
# `UPDATE_DATASET = True` uploads **only** `<character>/converted/` — the source audio and the
# VAD/ASR results already live on the Hub — packing `latents/` into `latents.tar` first, because
# the Hub rejects a directory holding more than 10 000 files.
# ===========================================================================
def step_verify_and_upload():
    records = verify_converted()        # runs whether the latents were just encoded or unpacked
    total_frames = sum(r["num_frames"] for r in records)
    referenced = [MANIFEST.parent / r["latent_path"] for r in records]

    # CONVERT_AUDIO_FILTER narrows which source files were used, so mirror it here.
    def used(p):
        return not CONVERT_AUDIO_FILTER or CONVERT_AUDIO_FILTER in p.name

    report = {
        "character": CHARACTER,
        "dataset_repo": HF_DATASET_REPO,
        "source_audio": sorted(p.name for p in (DATASET / CHARACTER).glob("*.mp3") if used(p)),
        "source_segments": sorted(p.name for p in (DATASET / "results" / CHARACTER).glob("*.jsonl") if used(p)),
        "manifest": MANIFEST.name,
        "latent_dir": LATENTS.name,
        "n_samples": len(records),
        "total_frames": total_frames,
        "total_seconds": round(total_frames / LATENT_FPS, 2),
        "latent_bytes": sum(p.stat().st_size for p in referenced),
        "codec_repo": CODEC_REPO,
        "codec_sample_rate": 48000,
        "latent_dim": 32,
        "latent_fps": LATENT_FPS,
        "normalize_db": CONVERT_NORMALIZE_DB,
        "min_seconds": CONVERT_MIN_SECONDS,
        "max_seconds": CONVERT_MAX_SECONDS,
        "limit": CONVERT_LIMIT,
        "audio_filter": CONVERT_AUDIO_FILTER,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not UPDATE_DATASET:
        print("\nUPDATE_DATASET is False — skipping the HuggingFace upload.")
        return
    use_hf_token()
    # One tar, not 10k+ loose files: the Hub rejects a directory that large.
    run("tar", "-cf", LATENTS_TAR.name, "latents/", cwd=CONVERT)
    run("uvx", "hf", "upload", HF_DATASET_REPO, CONVERT,
        f"{CHARACTER}/converted", "--repo-type=dataset", "--exclude=latents/")


# ===========================================================================
# 8. Train
#
# The base model is frozen; the run trains only a `(16, 768)` embedding — watch for this line in
# the log:
#
#     Speaker Inversion freeze applied: trainable=12,288 frozen=766,052,385
#
# Three things worth knowing:
#
# - **DDP.** With more than one GPU the run is forked through `accelerate`'s `notebook_launcher`,
#   which sets the `RANK` / `LOCAL_RANK` / `WORLD_SIZE` that `train.py` reads; with one GPU the
#   entry function is simply called in this process. `batch_size` is per process, so 2 GPUs
#   doubles the effective batch.
# - **Precision.** T4 (sm_75) has no bf16 support, so `train.py` prints a warning and falls back
#   to fp32 by itself. Set `TRAIN_PRECISION = "fp32"` to skip the warning.
# - **Resuming.** `CONTINUE_FROM` warm-starts the embedding from a checkpoint of an earlier run
#   (e.g. `"checkpoint_001500.speaker.safetensors"`, downloaded from `HF_MODEL_REPO`). Only the
#   embedding carries over — step counter, optimizer state and LR schedule restart.
# ===========================================================================
def train_args():
    args = [
        "--config", str(CONFIG_YAML),
        "--manifest", str(MANIFEST),
        "--init-checkpoint", str(BASE_CHECKPOINT),
        "--output-dir", str(OUTPUT_DIR),
        "--device", "cuda",
        "--batch-size", TRAIN_BATCH_SIZE,
        "--gradient-accumulation-steps", TRAIN_GRAD_ACCUM,
        "--max-steps", TRAIN_MAX_STEPS,
        "--lr", TRAIN_LR,
        "--save-every", TRAIN_SAVE_EVERY,
        "--log-every", TRAIN_LOG_EVERY,
        "--num-workers", TRAIN_NUM_WORKERS,
        "--precision", TRAIN_PRECISION,
    ]

    if CONTINUE_FROM:
        continue_path = MODEL_REPO / CHARACTER / CONTINUE_FROM
        assert continue_path.is_file(), (f"CONTINUE_FROM={CONTINUE_FROM!r} is not in "
                                         f"{MODEL_REPO / CHARACTER} — check HF_MODEL_REPO / CONTINUE_FROM")
        args += ["--speaker-inversion-init-embedding", str(continue_path)]
        print(f"continuing from {continue_path}")
    else:
        print("CONTINUE_FROM is empty — starting from a random speaker embedding")
    return args


def sync_output():
    """Push outputs/<character>/ to <HF_MODEL_REPO>/<character>/ — one directory per character."""
    if not (SYNC_CKPT and HF_MODEL_REPO and OUTPUT_DIR.is_dir()):
        return
    run("uvx", "hf", "upload", HF_MODEL_REPO, OUTPUT_DIR, CHARACTER, "--repo-type=model")


def sync_while_training(stop):
    """Upload the output directory every SYNC_EVERY seconds until asked to stop.

    train.py has no upload hook, so watch the output directory instead: the snapshots are a
    few KB each and `hf upload` skips files it already has, so re-uploading the whole
    directory on a timer is cheaper than tracking what changed — and it is what survives a
    recycled runtime mid-run. The upload runs here, off the main thread, because the launcher
    needs the main thread for its signal handlers.
    """
    while not stop.wait(SYNC_EVERY):
        sync_output()


def train_entry(argv):
    """train.py's CLI entry point, in this process: the launcher has set RANK / WORLD_SIZE.

    Module-level, not nested in step_train(), because the launcher pickles it into the child
    processes it spawns for DDP.
    """
    sys.argv = ["train.py", *[str(a) for a in argv]]
    sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)
    import train

    train.main()


def step_train():
    import torch

    argv = train_args()

    n_gpu = torch.cuda.device_count()
    nproc = TRAIN_NPROC or n_gpu
    if nproc > 1:
        print(f"DDP over {nproc} GPU(s); batch_size={TRAIN_BATCH_SIZE} is per process "
              f"-> global batch {TRAIN_BATCH_SIZE * nproc}")
    else:
        print(f"single process ({n_gpu} GPU visible)")

    # Accelerator is imported but deliberately never built here: an AcceleratorState that already
    # exists makes notebook_launcher refuse to fork, and train.py calls init_process_group itself.
    from accelerate import notebook_launcher  # noqa: F401

    stop = threading.Event()
    if SYNC_CKPT and HF_MODEL_REPO:
        threading.Thread(target=sync_while_training, args=(stop,), daemon=True).start()
    try:
        if nproc > 1:
            notebook_launcher(train_entry, args=(argv,), num_processes=nproc)
        else:
            train_entry(argv)
    finally:
        stop.set()
        sync_output()


# ===========================================================================
# 9. Outputs
# ===========================================================================
def step_outputs():
    final_embed = OUTPUT_DIR / "checkpoint_final.speaker.safetensors"
    assert final_embed.is_file(), f"training produced no embedding at {final_embed}"

    print(f"converted -> {CONVERT}")
    run("ls", "-alh", CONVERT)
    print(f"\nembeddings -> {OUTPUT_DIR}")
    run("ls", "-alh", OUTPUT_DIR)
    if SYNC_CKPT and HF_MODEL_REPO:
        print(f"\nsynced to https://huggingface.co/{HF_MODEL_REPO}/tree/main/{CHARACTER}")


STEPS = [
    ("1. system dependencies", step_system_deps),
    ("2. clone the trainer", step_clone),
    ("3. python environment", step_python_env),
    ("4. data + models", step_download),
    ("5. unpack the latents", step_unpack),
    ("6. convert -> latents", step_convert),
    ("7. verify + dataset upload", step_verify_and_upload),
    ("8. train", step_train),
    ("9. outputs", step_outputs),
]


def main():
    for label, value in [("repo", REPO_DIR), ("dataset", DATASET), ("converted", CONVERT),
                         ("base ckpt", BASE_CHECKPOINT), ("outputs", OUTPUT_DIR)]:
        print(f"{label:<10} {value}")
    print(f"{'model repo':<10} {MODEL_REPO}")

    selected = [s for s in sys.argv[1:] if s.strip().isdigit()]
    todo = [(n, f) for n, f in STEPS if not selected or n.split(".")[0] in selected]
    for label, step in todo:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
        step()


if __name__ == "__main__":
    main()
