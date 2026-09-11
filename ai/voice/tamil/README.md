# Building the `mindaura_tamil` corpus — what YOU do

Goal: ~15–20 Tamil speakers, each reading 20 neutral sentences in 5 emotions
(angry / happy / sad / fear / neutral), 2 takes → ≈ **4,000 Tamil clips** that
MindAura owns and can legally use in the product.

You do **not** need to wait for this to train the first model — it plugs in later
and we retrain.

---

## 1. Find speakers
- 15–20 people. **Mix of men and women**, mix of ages, ideally from different
  parts of Tamil Nadu (Chennai / Madurai / Coimbatore / Tirunelveli accents).
- More speakers matters more than more sentences per speaker.

## 2. Consent
- Each speaker reads and signs `CONSENT_FORM.md` (print it, or e-sign).
- Fill in the project contact before printing.
- Keep the signed forms — store them separately from the audio.

## 3. Set up a quiet spot
- A small room with soft furnishings (bed, curtains, sofa) — kills echo.
- Laptop mic is OK; a cheap USB mic or phone earphones-with-mic is better.
- Same mic, same distance (about a hand-span) for everyone. No fan/AC noise.

## 4. Record — run this on the machine with the mic
```bash
cd MindAura.AI
.venv/bin/python -m ai.voice.tamil.record --speaker S01
```
- It asks for gender / age / region and "consent signed? YES".
- Then for each prompt it shows: the **emotion**, a **scenario to imagine**, and
  the **Tamil sentence**.
- The speaker gets into the emotion (use the scenario), you press ENTER, they say
  the sentence, press ENTER to stop. It plays back → **k**eep / **r**edo / **s**kip.
- ~20 min per speaker. Use `S01`, `S02`, … as ids.
- Stop any time with Ctrl-C — progress is saved, just re-run with the same
  `--speaker` to continue.

Check progress at any point:
```bash
.venv/bin/python -m ai.voice.tamil.record --progress
```

## 5. Tips for good data
- The **words are neutral on purpose** — all the emotion must come from the voice.
- Encourage real delivery: for "angry", actually sound irritated; for "sad",
  slower and heavier. Over-acting slightly is fine; flat reading is not.
- Neutral = newsreader voice, no feeling.
- If a speaker can't do an emotion convincingly, skip it — bad labels hurt.

## 6. When you have a batch (even 5–8 speakers)
Tell me, or run:
```bash
.venv/bin/python -m ai.voice.prepare --corpus mindaura_tamil
```
This folds the Tamil clips into the training set. Then we retrain and Tamil
emotion detection becomes measurable and much better.

## 7. (Later) Quality check
`python -m ai.voice.tamil.validate` — has other people listen to clips and
confirm the emotion, so we keep only the ones that read clearly.

---

Files produced: `ai/training/ser/raw/mindaura_tamil/<emotion>/<spk>_<sentence>_<EMO>_<take>.wav`
(16 kHz mono) and `speakers.csv`. All git-ignored.
