"""
Toolkit for recording MindAura's own Tamil emotional-speech corpus
(`mindaura_tamil`) — a commercial-clean, Tamil-Nadu-accent alternative to the
gated EmoTa dataset.

  python -m ai.voice.tamil.record      # record one speaker
  python -m ai.voice.tamil.validate    # listener label-check (later)

Output lands in ai/training/ser/raw/mindaura_tamil/ and is folded into training
by `python -m ai.voice.prepare --corpus mindaura_tamil`.
"""
