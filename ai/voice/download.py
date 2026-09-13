"""
Fetch the auto-downloadable SER corpora into ``ai/training/ser/raw/<key>/``.

    python -m ai.voice.download --corpus all
    python -m ai.voice.download --corpus emodb urdu shemo

Corpora already present on disk (RAVDESS, CREMA-D under ai/training/datasets/)
are detected and skipped. HuggingFace dataset corpora (RESD) are pre-cached via
``datasets``. Deferred / EULA corpora print their manual instructions.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from .corpora import CORPORA, RAW_DIR, AUTO_CORPORA, DEFERRED_CORPORA, Corpus

_CHUNK = 1 << 20
_TIMEOUT = (30, 180)   # (connect, read)
_RETRIES = 3


def _get(url: str) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            print(f"    GET {url}  (attempt {attempt}/{_RETRIES})")
            resp = requests.get(url, stream=True, timeout=_TIMEOUT,
                                headers={"User-Agent": "MindAura-SER/0.1"})
            resp.raise_for_status()
            buf = io.BytesIO()
            total = int(resp.headers.get("content-length", 0))
            got = 0
            for chunk in resp.iter_content(_CHUNK):
                buf.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r    {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            return buf.getvalue()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"\n    ! {exc}")
            if attempt < _RETRIES:
                time.sleep(5 * attempt)
    raise last_exc  # type: ignore[misc]


def _flatten_github_archive(dest: Path) -> None:
    """GitHub 'master.zip' extracts to <repo>-master/ — pull its contents up one level."""
    subdirs = [d for d in dest.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name.endswith(("-master", "-main")):
        inner = subdirs[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(dest / item.name))
        inner.rmdir()


def _download_archive(corpus: Corpus) -> bool:
    dest = RAW_DIR / corpus.key
    if dest.exists() and any(dest.rglob("*.wav")):
        print(f"[skip] {corpus.key}: already extracted at {dest}")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    try:
        data = _get(corpus.url)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {corpus.key}: download error: {exc}")
        if corpus.manual:
            print(f"       manual fallback:\n       {corpus.manual}")
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile:
        # not a zip: dump raw bytes for the user to sort out
        (dest / "download.bin").write_bytes(data)
        print(f"[warn] {corpus.key}: response was not a zip archive; saved raw bytes.")
        if corpus.manual:
            print(f"       {corpus.manual}")
        return False
    _flatten_github_archive(dest)
    n = len(list(dest.rglob("*.wav")))
    print(f"[ok]   {corpus.key}: {n} wav files under {dest}")
    return n > 0


def _precache_hf(corpus: Corpus) -> bool:
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"[FAIL] {corpus.key}: `datasets` not installed")
        return False
    print(f"[hf]   {corpus.key}: caching {corpus.hf_id} ...")
    try:
        load_dataset(corpus.hf_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {corpus.key}: {exc}")
        return False
    print(f"[ok]   {corpus.key}: cached")
    return True


def fetch(keys: list[str]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for key in keys:
        corpus = CORPORA[key]
        print(f"\n=== {key} ({corpus.language}) ===")
        if corpus.local_dir and corpus.local_dir.exists():
            print(f"[skip] {key}: using existing local copy at {corpus.local_dir}")
            results[key] = True
        elif corpus.hf_id:
            results[key] = _precache_hf(corpus)
        elif corpus.auto and corpus.url:
            results[key] = _download_archive(corpus)
        else:
            print(f"[manual] {key}: {corpus.manual}")
            results[key] = False
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", nargs="+", default=["all"],
                    help="corpus keys, or 'all' for every auto-downloadable corpus")
    args = ap.parse_args(argv)

    if args.corpus == ["all"]:
        keys = AUTO_CORPORA
    else:
        keys = args.corpus

    unknown = [k for k in keys if k not in CORPORA]
    if unknown:
        ap.error(f"unknown corpus keys: {unknown}. known: {sorted(CORPORA)}")

    results = fetch(keys)

    print("\n" + "=" * 60)
    for k, ok in results.items():
        print(f"  {'OK  ' if ok else 'MISS'}  {k}")
    if DEFERRED_CORPORA:
        print("\nDeferred (need registration / EULA) — run prepare.py once you have them:")
        for k in DEFERRED_CORPORA:
            print(f"  - {k} ({CORPORA[k].language})")
    ok_count = sum(results.values())
    print(f"\n{ok_count}/{len(results)} corpora ready. Next: python -m ai.voice.prepare")
    return 0 if ok_count else 1


if __name__ == "__main__":
    sys.exit(main())
