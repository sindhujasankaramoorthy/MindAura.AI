"""
Registry of every emotional-speech corpus in the SER training set.

Two kinds:
  * file-tree corpora  -> a ``parser`` turns each .wav path into a Sample
  * dataset corpora     -> pulled via HuggingFace ``datasets`` in prepare.py
                           (``parser is None``, ``hf_id`` set)

Only corpora with ``auto=True`` are fetched by ``download.py``. For the rest,
``manual`` holds the instructions and the expected drop location under
``ser/raw/<key>/``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .labels import to_unified

# repo-relative anchors
_TRAIN_DIR = Path(__file__).resolve().parent.parent          # ai/training
SER_DIR = Path(__file__).resolve().parent                    # ai/training/ser
RAW_DIR = SER_DIR / "raw"
PROCESSED_DIR = SER_DIR / "processed"
MANIFEST_CSV = SER_DIR / "manifest.csv"
EXISTING_DATASETS = _TRAIN_DIR / "datasets"                  # RAVDESS/CREMA-D already here


@dataclass(frozen=True)
class Sample:
    audio_path: Path
    unified_emotion: str
    speaker: str


@dataclass(frozen=True)
class Corpus:
    key: str
    language: str                      # ISO 639-1 (or "multi")
    license: str
    auto: bool
    url: Optional[str] = None          # direct archive URL for auto download
    hf_id: Optional[str] = None        # HuggingFace datasets id (dataset corpora)
    hf_gated: bool = False             # needs an accepted usage agreement + HF_TOKEN
    local_dir: Optional[Path] = None   # already on disk -> download.py skips
    manual: str = ""                   # instructions when auto is False
    primary: bool = False              # English + Tamil: the target + eval languages
    parser: Optional[Callable[[Path], Optional[Sample]]] = field(default=None, repr=False)

    def raw_root(self) -> Path:
        return self.local_dir if self.local_dir else (RAW_DIR / self.key)


# ── per-corpus filename parsers ──────────────────────────────────────────────

_RAVDESS_CODE = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


def _parse_ravdess(p: Path) -> Optional[Sample]:
    # 03-01-06-01-02-01-12.wav : [2]=emotion code, [6]=actor id
    parts = p.stem.split("-")
    if len(parts) != 7:
        return None
    raw = _RAVDESS_CODE.get(parts[2])
    if raw is None:
        return None
    unified = to_unified("ravdess", raw)
    if unified is None:
        return None
    return Sample(p, unified, f"ravdess_actor{parts[6]}")


def _parse_crema(p: Path) -> Optional[Sample]:
    # 1001_DFA_ANG_XX.wav : [0]=speaker, [2]=emotion
    parts = p.stem.split("_")
    if len(parts) < 3:
        return None
    unified = to_unified("crema_d", parts[2])
    if unified is None:
        return None
    return Sample(p, unified, f"crema_{parts[0]}")


def _parse_tess(p: Path) -> Optional[Sample]:
    # OAF_back_angry.wav / YAF_dog_ps.wav : speaker = first token, emotion = tail
    stem = p.stem.lower()
    speaker = stem.split("_")[0]
    if stem.endswith("pleasant_surprise") or stem.endswith("_ps"):
        raw = "ps"
    else:
        raw = stem.split("_")[-1]
    try:
        unified = to_unified("tess", raw)
    except KeyError:
        return None
    if unified is None:
        return None
    return Sample(p, unified, f"tess_{speaker}")


def _parse_emodb(p: Path) -> Optional[Sample]:
    # 03a01Fa.wav : [0:2]=speaker, [5]=emotion letter
    name = p.stem
    if len(name) < 6:
        return None
    unified = to_unified("emodb", name[5])
    if unified is None:
        return None
    return Sample(p, unified, f"emodb_{name[:2]}")


def _parse_urdu(p: Path) -> Optional[Sample]:
    # <Angry|Happy|Neutral|Sad>/SM1_F10_A017.wav : emotion from parent folder
    raw = p.parent.name
    try:
        unified = to_unified("urdu", raw)
    except KeyError:
        return None
    if unified is None:
        return None
    speaker = p.stem.split("_")[0]
    return Sample(p, unified, f"urdu_{speaker}")


def _parse_emota(p: Path) -> Optional[Sample]:
    # <emotion>/<spkID>_<senID>_<emo3>.wav — emotion from parent folder, speaker from stem
    raw = p.parent.name
    try:
        unified = to_unified("emota", raw)
    except KeyError:
        # fall back to the emo3 suffix in the filename
        tail = p.stem.split("_")[-1]
        try:
            unified = to_unified("emota", tail)
        except KeyError:
            return None
    if unified is None:
        return None
    speaker = f"emota_{p.stem.split('_')[0]}"
    return Sample(p, unified, speaker)


def _parse_mindaura_tamil(p: Path) -> Optional[Sample]:
    # <emotion>/<spk>_<senID>_<EMO3>_<take>.wav
    raw = p.parent.name
    try:
        unified = to_unified("mindaura_tamil", raw)
    except KeyError:
        parts = p.stem.split("_")
        try:
            unified = to_unified("mindaura_tamil", parts[2]) if len(parts) >= 3 else None
        except KeyError:
            return None
    if unified is None:
        return None
    return Sample(p, unified, f"mtamil_{p.stem.split('_')[0]}")


def _parse_subesco(p: Path) -> Optional[Sample]:
    # F_02_MONIKA_S_1_NEUTRAL_5.wav : [1]=speaker num, [2]=name, [5]=EMOTION
    parts = p.stem.split("_")
    if len(parts) < 7:
        return None
    try:
        unified = to_unified("subesco", parts[5])
    except KeyError:
        return None
    if unified is None:
        return None
    return Sample(p, unified, f"subesco_{parts[0]}{parts[1]}")


def _parse_kannada(p: Path) -> Optional[Sample]:
    # AA-EE-SS.wav : AA=actor, EE=emotion code 01-06, SS=sentence
    parts = p.stem.split("-")
    if len(parts) != 3:
        return None
    try:
        unified = to_unified("kannada", parts[1])
    except KeyError:
        return None
    if unified is None:
        return None
    return Sample(p, unified, f"kannada_{parts[0]}")


_SHEMO_RE = re.compile(r"^([FM]\d+)([A-Za-z])\d+")


def _parse_shemo(p: Path) -> Optional[Sample]:
    # F21A11.wav : speaker = F21, emotion letter = A
    m = _SHEMO_RE.match(p.stem)
    if not m:
        return None
    unified = to_unified("shemo", m.group(2))
    if unified is None:
        return None
    return Sample(p, unified, f"shemo_{m.group(1)}")


# ── the registry ────────────────────────────────────────────────────────────

CORPORA: dict[str, Corpus] = {
    "ravdess": Corpus(
        key="ravdess", language="en", license="CC BY-NC-SA 4.0", auto=True, primary=True,
        url="https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip",
        local_dir=(EXISTING_DATASETS / "ravdess") if (EXISTING_DATASETS / "ravdess").exists() else None,
        parser=_parse_ravdess,
    ),
    "crema_d": Corpus(
        key="crema_d", language="en", license="Open Database License (ODbL)", auto=True, primary=True,
        url="https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/",
        local_dir=(EXISTING_DATASETS / "crema_d") if (EXISTING_DATASETS / "crema_d").exists() else None,
        manual="CREMA-D is large and git-LFS backed. If auto download fails, clone "
               "https://github.com/CheyneyComputerScience/CREMA-D and point "
               "ser/raw/crema_d/ at its AudioWAV/ folder.",
        parser=_parse_crema,
    ),
    "mindaura_tamil": Corpus(
        key="mindaura_tamil", language="ta", license="MindAura-owned (speaker consent)",
        auto=False, primary=True,
        manual="Record it yourself: see ai/training/ser/tamil/README.md and run "
               "`python -m ai.voice.tamil.record`. Commercial-clean, "
               "Tamil-Nadu accent. Lands in ser/raw/mindaura_tamil/.",
        parser=_parse_mindaura_tamil,
    ),
    "emota": Corpus(
        key="emota", language="ta", license="EmoTa Academic-Commercial License", auto=True,
        primary=True, hf_id="aaivu-labs/EmoTa", hf_gated=True,
        manual="Tamil (Sri Lankan). Gated on HuggingFace: sign in at "
               "https://huggingface.co/datasets/aaivu-labs/EmoTa , accept the usage "
               "agreement (instant), then `huggingface-cli login` (or set HF_TOKEN). "
               "Alternatively drop the extracted emotion folders into ser/raw/emota/.",
        parser=_parse_emota,
    ),
    "tess": Corpus(
        key="tess", language="en", license="CC BY-NC 4.0", auto=True, primary=True,
        url="https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF",
        manual="Toronto Emotional Speech Set. If the Borealis Dataverse bulk "
               "download fails, download the archive from "
               "https://tspace.library.utoronto.ca/handle/1807/24487 into ser/raw/tess/.",
        parser=_parse_tess,
    ),
    "emodb": Corpus(
        key="emodb", language="de", license="free for research (EmoDB terms)", auto=True,
        url="http://emodb.bilderbar.info/download/download.zip",
        parser=_parse_emodb,
    ),
    "urdu": Corpus(
        key="urdu", language="ur", license="MIT", auto=True,
        url="https://github.com/siddiquelatif/URDU-Dataset/archive/refs/heads/master.zip",
        parser=_parse_urdu,
    ),
    "subesco": Corpus(
        key="subesco", language="bn", license="CC BY 4.0", auto=True,
        url="https://zenodo.org/records/4526477/files/SUBESCO.zip?download=1",
        manual="SUST Bangla Emotional Speech Corpus. Direct download from Zenodo "
               "record 4526477. 7 000 utterances, 20 speakers, all 7 emotions.",
        parser=_parse_subesco,
    ),
    "kannada": Corpus(
        key="kannada", language="kn", license="CC BY 4.0", auto=True,
        url="https://zenodo.org/api/records/6345107/files-archive",
        manual="Kannada emotional speech (Dravidian family, closest to Tamil). "
               "Zenodo record 6345107. If the archive URL fails, download the "
               "individual wavs from https://zenodo.org/records/6345107 into ser/raw/kannada/.",
        parser=_parse_kannada,
    ),
    "shemo": Corpus(
        key="shemo", language="fa", license="free for research (ShEMO terms)", auto=True,
        url="https://github.com/mansourehk/ShEMO/archive/refs/heads/master.zip",
        manual="If the wav files are not inside the GitHub archive, fetch them "
               "from the download link in the ShEMO README into ser/raw/shemo/.",
        parser=_parse_shemo,
    ),
    "resd": Corpus(
        key="resd", language="ru", license="MIT", auto=True,
        hf_id="Aniemore/resd",          # handled in prepare.py (dataset corpus)
        parser=None,
    ),
    # ── deferred: need a EULA / registration (Stage 3 backfill) ──────────────
    "savee": Corpus(
        key="savee", language="en", license="free for research (registration)", auto=False,
        manual="Register at http://kahlan.eps.surrey.ac.uk/savee/ , download "
               "AudioData.zip, extract into ser/raw/savee/ (DC/ JE/ JK/ KL/ folders).",
        parser=None,
    ),
    "emovo": Corpus(
        key="emovo", language="it", license="free for research (request form)", auto=False,
        manual="Request EMOVO from http://voice.fub.it/activities/corpora/emovo/ , "
               "extract into ser/raw/emovo/.",
        parser=None,
    ),
    "aesdd": Corpus(
        key="aesdd", language="el", license="CC BY-NC-SA 4.0", auto=False,
        manual="Download AESDD from http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/ "
               "into ser/raw/aesdd/.",
        parser=None,
    ),
    "iemocap": Corpus(
        key="iemocap", language="en", license="USC EULA", auto=False, primary=True,
        manual="Request IEMOCAP at https://sail.usc.edu/iemocap/ (approval takes weeks). "
               "Extract sessions into ser/raw/iemocap/.",
        parser=None,
    ),
}

PRIMARY_CORPORA = [k for k, c in CORPORA.items() if c.primary]      # English + Tamil

AUTO_CORPORA = [k for k, c in CORPORA.items() if c.auto]
DEFERRED_CORPORA = [k for k, c in CORPORA.items() if not c.auto]
