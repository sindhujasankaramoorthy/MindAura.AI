"""
Evaluate the fine-tuned SER model on the held-out `test` split.

    python -m ai.voice.evaluate
    python -m ai.voice.evaluate --checkpoint ser/checkpoints/xls-r-300m-ser

Reports overall accuracy + macro-F1, a per-emotion report, a confusion matrix,
and — the number that matters for MindAura — **per-language** macro-F1, so we can
see how English vs. Tamil vs. the auxiliary languages actually do.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .corpora import SER_DIR
from .dataset import Collator, SERDataset, read_manifest
from .labels import ID2LABEL, UNIFIED_EMOTIONS

DEFAULT_CKPT = SER_DIR / "checkpoints" / "xls-r-300m-ser"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)

    import torch
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint} — train first")

    rows = read_manifest(args.split)
    if not rows:
        raise SystemExit(f"empty {args.split} split")

    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.checkpoint)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    loader = DataLoader(SERDataset(rows, fe), batch_size=args.batch,
                        collate_fn=Collator(fe))
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            y_pred.extend(logits.argmax(-1).cpu().tolist())
            y_true.extend(labels.tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    names = list(UNIFIED_EMOTIONS)

    print("\n" + "=" * 60)
    print(f"SER evaluation — {args.split} split ({len(rows)} clips)")
    print("=" * 60)
    print(f"accuracy   : {(y_true == y_pred).mean():.4f}")
    print(f"macro F1   : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    print("\nper-emotion:")
    print(classification_report(y_true, y_pred, labels=list(range(len(names))),
                                target_names=names, zero_division=0))

    print("confusion matrix (rows = true):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    print("           " + " ".join(f"{n[:5]:>6}" for n in names))
    for n, r in zip(names, cm):
        print(f"    {n:8s} " + " ".join(f"{v:6d}" for v in r))

    # per-language
    by_lang_true: dict[str, list] = defaultdict(list)
    by_lang_pred: dict[str, list] = defaultdict(list)
    for r, t, p in zip(rows, y_true, y_pred):
        by_lang_true[r["language"]].append(t)
        by_lang_pred[r["language"]].append(p)
    print("\nper-language macro-F1:")
    lang_scores = {}
    for lang in sorted(by_lang_true):
        s = f1_score(by_lang_true[lang], by_lang_pred[lang], average="macro",
                     labels=list(range(len(names))), zero_division=0)
        lang_scores[lang] = float(s)
        print(f"    {lang:4s}  {s:.4f}   (n={len(by_lang_true[lang])})")

    out = args.checkpoint / f"{args.split}_metrics.json"
    out.write_text(json.dumps({
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_language_macro_f1": lang_scores,
    }, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
