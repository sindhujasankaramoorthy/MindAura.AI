"""
Voice modality: runtime inference (recorder, transcription, feature
extraction, emotion prediction, fusion) plus the SER training track.

SER training (this package also holds it, see labels/corpora/download/prepare
/train/evaluate/supervise):
Stage 1: acquire and normalise multi-lingual emotional-speech corpora into a
single manifest with a unified 7-emotion label space.
Stage 2: fine-tune `facebook/wav2vec2-xls-r-300m` on that manifest.

Sub-modules
-----------
labels   : the unified label space + per-corpus label maps
corpora  : registry of every corpus (language, download method, filename parser)
download : `python -m ai.voice.download --corpus all`
prepare  : `python -m ai.voice.prepare`  -> voice/processed/ + voice/manifest.csv
"""
