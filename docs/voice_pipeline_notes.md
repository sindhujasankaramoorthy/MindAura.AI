# Voice pipeline: Indian names + Tamil/code-switch fixes

Scope notes for the changes in `ai/voice/voice_analysis.py`,
`ai/voice/name_protection.py`, and `ai/video/existing_voice_adapter.py`.
Written so the reasoning and known limitations aren't lost to history.

## What was wrong

1. **No name-protection stage existed for voice at all.** The text-journal
   pipeline already had one (`ai/text/entities/protector.py`'s
   `NERProtection` + `ai/text/entities/name_dictionary.py`), but
   `voice_analysis.py` never called it -- an ASR mis-hearing of a name
   passed straight through to the final output.
2. **Faster-Whisper "small" hallucinates badly on Tamil.** Verified against
   a real recording already in the repo
   (`ai/voice/recordings/voice_260911.wav`, prior output saved at
   `ai/voice/recordings/voice_260911_output.json`): the raw transcript
   contains random Chinese/French fragments woven into the Tamil, and the
   `contextual_translation` was entirely fabricated content unrelated to
   what was said. This matches Whisper's own published per-language WER
   tables -- Tamil is a weak point for `small`, much better on `medium`/
   `large-v3`.

## What changed

- **`ai/voice/name_protection.py`** (new, dedicated module): PERSON
  detection via the existing `NERProtection`, phonetic/fuzzy matching via
  `rapidfuzz` (already a dependency) against an expanded
  `COMMON_NAMES` dictionary. Never corrects unless a candidate clears an
  85% similarity threshold; below that, the original ASR text is kept
  untouched. Applied independently to both the native transcript and the
  English translation, since Whisper's `task="translate"` decodes directly
  from audio in its own pass and doesn't consume our corrected text.
- **`ai/voice/voice_analysis.py`**: `vad_filter=True` enabled (built into
  faster-whisper already, no new dependency); default model size raised
  from `small` to `medium` via `MINDAURA_WHISPER_MODEL` env var (defaults
  to `"medium"`); `analyze_voice()` now additionally returns
  `normalized_transcription`, `english_contextual_text`, a structured
  `language` block (`primary`/`secondary`/`code_switched`/`confidence`),
  `entities`, and a `confidence` block. All previously-existing keys are
  unchanged in shape, so existing callers keep working.
- **`ai/video/existing_voice_adapter.py`**: passes the new
  `normalized_text`, `language` (structured), `entities`, and
  `confidence` fields through to its own schema (used by
  `backend/app/api/checkins.py`'s `/checkins/voice`), while deliberately
  *not* adding English-translation output there -- that adapter's own
  documented scope is native-script transcription only; translation
  belongs to `ai/text/pipeline.py` downstream.

## Known, honestly-stated limitations

- **The "medium" model upgrade could not be empirically re-verified on
  this specific Tamil clip in this environment.** `cdn-lfs.huggingface.co`
  (the CDN that actually serves model weights) fails DNS resolution from
  this sandbox, even though `huggingface.co` itself is reachable. The
  choice to default to `medium` is based on Whisper's own published
  per-language benchmarks, not a local before/after test. **Once you have
  a network that can reach the LFS CDN, re-run**
  `python -m ai.voice.voice_analysis ai/voice/recordings/voice_260911.wav multi`
  **and compare against the old output in
  `ai/voice/recordings/voice_260911_output.json`** -- that's the
  validation this change still needs.
- **Name correction only fires when NER itself flags a span as PERSON
  first.** For a name so garbled that NER doesn't recognize it as a person
  span at all (tested: "Sooresh" in isolation, without dictionary-fallback
  help, wasn't always tagged), no correction is attempted -- the ASR
  output is left as-is rather than guessed. This is the safe failure
  mode the project asked for ("if confidence is low, keep the original"),
  but it means recall on badly garbled names is inherently limited by the
  underlying NER model, not just this stage's matching logic.
- **Only Latin-script name mentions are checked against the Indian-name
  dictionary.** A name rendered entirely in Tamil/Telugu/etc. script
  within the native transcript isn't fuzzy-matched -- that would need a
  transliterated name reference set and a differently-scoped detector,
  which is out of scope for this change (no such infrastructure existed
  to reuse, and building one would be a new subsystem, not a minimal fix).
- **No measured accuracy numbers (WER, name-recognition rate, etc.) are
  reported**, because there is no labeled ground-truth dataset in this
  repo for Tamil/Malayalam/Telugu/Kannada/Hindi speech content or names
  (the existing `ai/voice/raw/*` datasets are emotion-recognition corpora
  with fixed carrier sentences, not name/content-labeled). Fabricating
  such numbers would misrepresent real accuracy; the qualitative
  comparison above (real audio, real prior output, real re-run once
  network allows) is what's honestly available without new labeled data.
