"""
Accuracy benchmark for the text emotion classifier (SamLowe/roberta-base-go_emotions),
evaluated against the real, public GoEmotions test set -- the same dataset
this model was trained on. Mirrors the role ai/voice/evaluate.py plays for
the voice SER model: this repo previously had zero measured accuracy for
the text pipeline, only eyeballed output.

Benchmarks the classifier itself (EmotionAnalyzer._classify_emotions),
bypassing Tanglish/translation preprocessing -- GoEmotions examples are
already clean English text, so this isolates the model's own accuracy from
preprocessing effects (those are covered separately by the Tanglish
integration tests).

Usage:
    python -m ai.inference.evaluate_text_emotion [--limit N] [--threshold 0.5]

Writes a metrics JSON next to this file's checkpoints-equivalent location:
    ai/inference/text_eval_results.json
"""
import argparse
import json
import logging
import os
from typing import List

import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

from ai.inference.emotion_predict import EmotionAnalyzer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "text_eval_results.json")
DEFAULT_THRESHOLD = 0.5


def load_test_set(limit: int = None):
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    label_names = ds.features["labels"].feature.names
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds, label_names


def evaluate(limit: int = None, threshold: float = DEFAULT_THRESHOLD):
    analyzer = EmotionAnalyzer()
    ds, label_names = load_test_set(limit)

    assert label_names == analyzer.ALL_GOEMOTIONS, (
        "GoEmotions dataset label order doesn't match EmotionAnalyzer.ALL_GOEMOTIONS -- "
        "mapping would silently be wrong."
    )
    n_labels = len(label_names)

    y_true = np.zeros((len(ds), n_labels), dtype=int)
    y_pred_binary = np.zeros((len(ds), n_labels), dtype=int)
    top1_hits = 0

    for i, example in enumerate(ds):
        text = example["text"]
        gold_indices = set(example["labels"])
        for idx in gold_indices:
            y_true[i, idx] = 1

        result = analyzer._classify_emotions(text)
        scores = result["emotion_scores"]

        top_label = result["dominant_emotion"]
        top_idx = label_names.index(top_label) if top_label in label_names else -1
        if top_idx in gold_indices:
            top1_hits += 1

        for j, label in enumerate(label_names):
            if scores.get(label, 0.0) >= threshold:
                y_pred_binary[i, j] = 1

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(ds)} evaluated")

    top1_accuracy = top1_hits / len(ds)
    macro_f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_binary, average="micro", zero_division=0)
    subset_accuracy = accuracy_score(y_true, y_pred_binary)

    report = classification_report(
        y_true, y_pred_binary, target_names=label_names, zero_division=0, output_dict=True
    )

    summary = {
        "dataset": "google-research-datasets/go_emotions (simplified, test split)",
        "n_examples": len(ds),
        "threshold": threshold,
        "top1_accuracy": round(top1_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "exact_match_subset_accuracy": round(subset_accuracy, 4),
        "per_label_f1": {
            label: round(report[label]["f1-score"], 4) for label in label_names
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Text emotion classifier benchmark ({len(ds)} examples)")
    print("=" * 60)
    print(f"Top-1 accuracy (predicted label in gold set): {top1_accuracy:.4f}")
    print(f"Macro F1 (multi-label, threshold={threshold})   : {macro_f1:.4f}")
    print(f"Micro F1                                       : {micro_f1:.4f}")
    print(f"Exact-match subset accuracy                    : {subset_accuracy:.4f}")
    print(f"\nWritten to {RESULTS_PATH}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N test examples")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    evaluate(limit=args.limit, threshold=args.threshold)
