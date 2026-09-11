"""
Fine-tune facebook/wav2vec2-xls-r-300m for 7-class speech emotion recognition.

    python -m ai.voice.train
    python -m ai.voice.train --epochs 8 --batch 4 --grad-accum 8

Reads ser/manifest.csv (built by prepare.py), trains on the `train` split,
early-stops on `val` macro-F1, writes the best model to
ser/checkpoints/xls-r-300m-ser/.

Tuned for a 12 GB GPU: feature encoder frozen, gradient checkpointing, fp16,
dynamic padding. Class-weighted loss handles the corpus imbalance.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .corpora import SER_DIR
from .dataset import Collator, SERDataset, class_weights, read_manifest
from .labels import ID2LABEL, LABEL2ID, UNIFIED_EMOTIONS

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
OUT_DIR = SER_DIR / "checkpoints" / "xls-r-300m-ser"


def _metrics_fn():
    from sklearn.metrics import accuracy_score, f1_score

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }

    return compute


def build_trainer(args):
    import torch
    from transformers import (
        Trainer, TrainingArguments, Wav2Vec2FeatureExtractor,
        Wav2Vec2ForSequenceClassification,
    )

    train_rows = read_manifest("train")
    val_rows = read_manifest("val")
    if not train_rows:
        raise SystemExit("empty train split — run prepare.py first")
    print(f"train={len(train_rows)}  val={len(val_rows)}")

    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(UNIFIED_EMOTIONS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    model.freeze_feature_encoder()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    weights = class_weights(train_rows)
    print("class weights:", {e: round(float(w), 2) for e, w in zip(UNIFIED_EMOTIONS, weights)})

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        fp16=args.fp16,
        gradient_checkpointing=args.grad_checkpoint,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=args.workers,
        report_to=[],
        remove_unused_columns=False,
    )

    return WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=SERDataset(train_rows, fe, args.max_seconds),
        eval_dataset=SERDataset(val_rows, fe, args.max_seconds),
        data_collator=Collator(fe),
        compute_metrics=_metrics_fn(),
    ), fe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-seconds", type=float, default=7.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    args = ap.parse_args(argv)

    trainer, fe = build_trainer(args)
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    fe.save_pretrained(str(OUT_DIR))
    metrics = trainer.evaluate()
    (OUT_DIR / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nbest val metrics:", json.dumps(metrics, indent=2))
    print(f"\nsaved to {OUT_DIR}\nnext: python -m ai.voice.evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
