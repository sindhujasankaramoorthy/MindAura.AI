"""
Record one speaker for the `mindaura_tamil` emotional-speech corpus.

    python -m ai.voice.tamil.record
    python -m ai.voice.tamil.record --speaker S03 --takes 2
    python -m ai.voice.tamil.record --progress          # show what's done

For each (emotion, sentence, take) it shows the Tamil sentence + an emotional
scenario, records from the default microphone until you press ENTER, plays it
back, and asks keep / redo / skip. Already-recorded clips are skipped, so you
can stop and resume any time.

Files:  ser/raw/mindaura_tamil/<emotion>/<spk>_<senID>_<EMO3>_<take>.wav  (16 kHz mono)
Roster: ser/raw/mindaura_tamil/speakers.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "raw" / "mindaura_tamil"
SPEAKERS_CSV = OUT_ROOT / "speakers.csv"
SR = 16_000
EMO3 = {"angry": "ANG", "happy": "HAP", "sad": "SAD", "fear": "FEA", "neutral": "NEU"}


def _load_tsv(name: str) -> list[dict]:
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _record_until_enter() -> np.ndarray:
    import sounddevice as sd

    frames: list[np.ndarray] = []

    def cb(indata, _frames, _time, _status):  # noqa: ANN001
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=cb):
        input("   ● RECORDING — press ENTER to stop... ")
    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


def _clean(audio: np.ndarray) -> np.ndarray:
    import librosa

    if audio.size == 0:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    if trimmed.size < int(SR * 0.3):
        trimmed = audio
    peak = float(np.max(np.abs(trimmed)))
    if peak > 0:
        trimmed = trimmed / peak * 0.95
    return trimmed.astype("float32")


def _play(audio: np.ndarray) -> None:
    import sounddevice as sd

    sd.play(audio, SR)
    sd.wait()


def _register_speaker(speaker_id: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    existing = {}
    if SPEAKERS_CSV.exists():
        with open(SPEAKERS_CSV, encoding="utf-8") as f:
            existing = {r["speaker_id"]: r for r in csv.DictReader(f)}
    if speaker_id in existing:
        print(f"Speaker {speaker_id} already registered ({existing[speaker_id]}).")
        return
    print(f"\nNew speaker {speaker_id} — a few details (stored without any name):")
    gender = input("  gender (m/f/other)      : ").strip() or "na"
    age = input("  age                     : ").strip() or "na"
    region = input("  region / native district: ").strip() or "na"
    consent = input('  consent form signed? type "YES" to confirm: ').strip()
    if consent != "YES":
        print("Consent not confirmed — aborting. See tamil/CONSENT_FORM.md")
        sys.exit(1)
    new = not SPEAKERS_CSV.exists()
    with open(SPEAKERS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["speaker_id", "gender", "age", "region", "consent"])
        w.writerow([speaker_id, gender, age, region, "yes"])
    print(f"  registered {speaker_id}.")


def _targets(sentences: list[dict], scenarios: list[dict], takes: int,
             only_emotions: list[str] | None):
    for sc in scenarios:
        emo = sc["emotion"]
        if only_emotions and emo not in only_emotions:
            continue
        for sen in sentences:
            for take in range(1, takes + 1):
                yield emo, sc, sen, take


def _path_for(spk: str, emo: str, sen_id: str, take: int) -> Path:
    return OUT_ROOT / emo / f"{spk}_{sen_id}_{EMO3[emo]}_{take}.wav"


def show_progress(sentences, scenarios, takes) -> None:
    print(f"\nmindaura_tamil progress  ({OUT_ROOT})\n")
    if not SPEAKERS_CSV.exists():
        print("  no speakers recorded yet.")
        return
    with open(SPEAKERS_CSV, encoding="utf-8") as f:
        speakers = [r["speaker_id"] for r in csv.DictReader(f)]
    per_speaker = len(list(_targets(sentences, scenarios, takes, None)))
    total = 0
    for spk in speakers:
        done = sum(1 for emo, sc, sen, tk in _targets(sentences, scenarios, takes, None)
                   if _path_for(spk, emo, sen["id"], tk).exists())
        total += done
        print(f"  {spk:8s} {done:4d}/{per_speaker}")
    print(f"\n  speakers: {len(speakers)}   clips: {total}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speaker", help="speaker id, e.g. S03 (prompted if omitted)")
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--emotions", nargs="+", choices=list(EMO3),
                    help="restrict to these emotions")
    ap.add_argument("--progress", action="store_true", help="show progress and exit")
    args = ap.parse_args(argv)

    sentences = _load_tsv("sentences.tsv")
    scenarios = _load_tsv("scenarios.tsv")

    if args.progress:
        show_progress(sentences, scenarios, args.takes)
        return 0

    try:
        import sounddevice  # noqa: F401
        import soundfile as sf
    except Exception as exc:  # noqa: BLE001
        print(f"audio libraries not available: {exc}")
        return 1

    speaker = args.speaker or input("Speaker id (e.g. S03): ").strip()
    if not speaker:
        print("no speaker id"); return 1
    _register_speaker(speaker)

    todo = [t for t in _targets(sentences, scenarios, args.takes, args.emotions)
            if not _path_for(speaker, t[0], t[2]["id"], t[3]).exists()]
    if not todo:
        print(f"\nNothing left to record for {speaker}. 🎉")
        return 0

    print(f"\n{len(todo)} clips to record for {speaker}. "
          f"Ctrl-C any time — progress is saved.\n")
    input("ENTER to begin... ")

    done = 0
    for emo, sc, sen, take in todo:
        while True:
            _clear()
            print("=" * 66)
            print(f"  speaker {speaker}   |   clip {done + 1}/{len(todo)}   |   take {take}")
            print("=" * 66)
            print(f"\n  EMOTION:  {emo.upper()}")
            print(f"  Feel:     {sc['scenario_en']}")
            print(f"            {sc['scenario_ta']}")
            print("\n  SAY THIS SENTENCE (in that emotion):\n")
            print(f"      {sen['tamil']}")
            print(f"      ({sen['transliteration']})")
            print(f"      [{sen['english_gloss']}]\n")
            input("  ENTER to start recording... ")
            audio = _clean(_record_until_enter())
            if audio.size < int(SR * 0.3):
                print("  ⚠ too short / silent — let's redo.")
                continue
            print(f"  captured {audio.size / SR:.1f}s — playing back...")
            _play(audio)
            choice = input("  [k]eep  [r]edo  [s]kip : ").strip().lower() or "k"
            if choice.startswith("s"):
                break
            if choice.startswith("r"):
                continue
            out = _path_for(speaker, emo, sen["id"], take)
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, audio, SR)
            done += 1
            break

    _clear()
    print(f"\nDone — {done} new clips for {speaker}.")
    show_progress(sentences, scenarios, args.takes)
    print("\nNext: repeat for other speakers, then\n"
          "  python -m ai.voice.prepare --corpus mindaura_tamil")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nStopped — recorded clips are saved. Re-run to continue.")
        sys.exit(0)
