"""
Normalise every available SER corpus into one training manifest.

    python -m ai.voice.prepare
    python -m ai.voice.prepare --corpus emodb urdu

For each sample: map to the unified 7-emotion space, resample to 16 kHz mono,
peak-normalise, write to ``ser/processed/<corpus>/<emotion>/<speaker>__<name>.wav``
and append a row to ``ser/manifest.csv``:

    path, emotion, label_id, corpus, language, speaker, split

Split is **speaker-independent** (a speaker's clips are never in two splits),
roughly 80/10/10 per corpus, deterministic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Iterator, Optional

import librosa
import soundfile as sf

from .corpora import CORPORA, PROCESSED_DIR, MANIFEST_CSV, Sample
from .labels import LABEL2ID, UNIFIED_EMOTIONS, to_unified

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
_SR = 16_000


def _split_for(speaker: str) -> str:
    bucket = int(hashlib.md5(speaker.encode()).hexdigest(), 16) % 10
    return "train" if bucket < 8 else ("val" if bucket == 8 else "test")


def _write_16k_mono(src_path: Path, dst_path: Path) -> bool:
    try:
        audio, _ = librosa.load(src_path, sr=_SR, mono=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {src_path.name}: {exc}")
        return False
    if audio.size == 0:
        return False
    peak = float(max(abs(audio.min()), abs(audio.max())))
    if peak > 0:
        audio = audio / peak
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst_path, audio, _SR)
    return True


def _iter_filetree(corpus_key: str) -> Iterator[Sample]:
    corpus = CORPORA[corpus_key]
    root = corpus.raw_root()
    if not root.exists():
        return
    parser = corpus.parser
    if parser is None:
        return
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in _AUDIO_EXTS:
            continue
        sample = parser(path)
        if sample is not None:
            yield sample


def _iter_resd() -> Iterator[tuple]:
    """RESD (Russian) comes from HuggingFace; yields (array, sr, emotion, speaker)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [skip] resd: `datasets` not installed")
        return
    try:
        ds = load_dataset(CORPORA["resd"].hf_id)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] resd: {exc}")
        return
    for split_name, split in ds.items():
        for i in range(len(split)):
            try:
                row = split[i]
            except Exception as exc:  # noqa: BLE001
                if i == 0:
                    print(f"  [skip] resd: cannot decode audio ({exc}). "
                          f"`pip install torchcodec` to enable it.")
                    return
                continue
            raw = str(row.get("emotion", "")).strip()
            try:
                unified = to_unified("resd", raw)
            except KeyError:
                continue
            if unified is None:
                continue
            audio = row["path"] if "path" in row and isinstance(row["path"], dict) else row.get("audio")
            if not isinstance(audio, dict):
                continue
            speaker = f"resd_{split_name}_{row.get('speaker', i % 20)}"
            yield audio["array"], audio["sampling_rate"], unified, speaker


def _process_corpus(corpus_key: str, rows: list[dict]) -> None:
    corpus = CORPORA[corpus_key]
    print(f"\n=== {corpus_key} ({corpus.language}) ===")
    n_ok = 0

    if corpus_key == "resd":
        for idx, (array, sr, emotion, speaker) in enumerate(_iter_resd()):
            dst = PROCESSED_DIR / "resd" / emotion / f"{speaker}__{idx:05d}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            y = librosa.resample(array.astype("float32"), orig_sr=sr, target_sr=_SR) if sr != _SR else array.astype("float32")
            peak = float(max(abs(y.min()), abs(y.max()))) if y.size else 0.0
            if peak > 0:
                y = y / peak
            sf.write(dst, y, _SR)
            rows.append(_row(dst, emotion, corpus_key, corpus.language, speaker))
            n_ok += 1
        print(f"  {n_ok} samples")
        return

    for sample in _iter_filetree(corpus_key):
        dst = (PROCESSED_DIR / corpus_key / sample.unified_emotion /
               f"{sample.speaker}__{sample.audio_path.stem}.wav")
        if _write_16k_mono(sample.audio_path, dst):
            rows.append(_row(dst, sample.unified_emotion, corpus_key,
                             corpus.language, sample.speaker))
            n_ok += 1
    if n_ok == 0:
        print(f"  no usable samples (raw dir: {corpus.raw_root()})")
    else:
        print(f"  {n_ok} samples")


def _row(path: Path, emotion: str, corpus: str, language: str, speaker: str) -> dict:
    return {
        "path": str(path.resolve()),
        "emotion": emotion,
        "label_id": LABEL2ID[emotion],
        "corpus": corpus,
        "language": language,
        "speaker": speaker,
        "split": _split_for(speaker),
    }


def _available(keys: list[str]) -> list[str]:
    out = []
    for k in keys:
        c = CORPORA[k]
        if c.hf_id or c.raw_root().exists():
            out.append(k)
        else:
            print(f"[missing] {k}: nothing at {c.raw_root()} — run download.py or add it manually")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", nargs="+", default=["all"])
    args = ap.parse_args(argv)

    keys = list(CORPORA) if args.corpus == ["all"] else args.corpus
    unknown = [k for k in keys if k not in CORPORA]
    if unknown:
        ap.error(f"unknown corpus keys: {unknown}")

    keys = _available(keys)
    if not keys:
        print("\nNo corpora available. Run: python -m ai.voice.download --corpus all")
        return 1

    rows: list[dict] = []
    for key in keys:
        before = len(rows)
        try:
            _process_corpus(key, rows)
        except Exception as exc:  # noqa: BLE001
            del rows[before:]
            print(f"  [error] {key}: {type(exc).__name__}: {exc} — skipping this corpus")

    if not rows:
        print("\nNo samples produced.")
        return 1

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _summary(rows)
    print(f"\nManifest: {MANIFEST_CSV}  ({len(rows)} samples)")
    return 0


def _summary(rows: list[dict]) -> None:
    from collections import Counter
    print("\n" + "=" * 60)
    by_split = Counter(r["split"] for r in rows)
    by_emotion = Counter(r["emotion"] for r in rows)
    by_lang = Counter(r["language"] for r in rows)
    by_corpus = Counter(r["corpus"] for r in rows)
    print("splits   :", dict(by_split))
    print("languages:", dict(by_lang))
    print("corpora  :", dict(by_corpus))
    print("emotions :")
    for e in UNIFIED_EMOTIONS:
        print(f"    {e:10s} {by_emotion.get(e, 0)}")


if __name__ == "__main__":
    sys.exit(main())
