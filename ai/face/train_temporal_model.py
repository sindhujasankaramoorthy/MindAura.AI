"""
Training script for FacialTemporalEventModel (ai/face/temporal_model.py).

============================================================================
STATUS: NOT RUN. No checkpoint exists. This script is architecturally
complete and ready to run, but has not been executed, because none of the
three specified datasets could actually be obtained in this environment:

  - CK+     requires a manual request at https://ckplus.jeffcohn.net/request
            (individual approval, not a direct download)
  - DFEW    requires following an access form at
            https://dfew-dataset.github.io/download.html
  - FERV39k is hosted at https://github.com/wangyanckxx/FERV39k and also
            requires following its stated access process

None of these are plain wget/curl downloads -- they're license-gated,
requiring a human to submit a request and wait for approval. That step
cannot be completed by an automated session. Per this task's own
instruction ("do not automatically download every facial dataset... do not
fabricate training data"), this script has NOT been run against fake or
substitute data. ai/face/events.py's rule-based detector is what actually
produces temporal_events in the pipeline today.

Once a dataset is obtained (by a human completing the access request) and
placed under ai/face/data/<dataset_name>/raw/, this script is ready to use.
============================================================================

Phases (per spec):
  1. MediaPipe extraction + feature validation
  2. Train on labeled sequences
  3. Validate on unseen subjects (subject-level split, never frame-level)
  4. Optional cross-dataset validation (DFEW/FERV39k)

Evaluates: event precision/recall/F1, temporal localization quality,
tracking stability, face detection rate, landmark extraction success --
NOT emotion accuracy (this model does not classify emotion).
"""
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ai.face.temporal_model import FacialTemporalEventModel, NUM_BLENDSHAPE_FEATURES

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


@dataclass
class SequenceExample:
    """One labeled blendshape sequence: (seq_len, NUM_BLENDSHAPE_FEATURES)
    features, with per-timestep event-type labels and a subject_id used for
    the subject-level train/val/test split (never split by frame)."""
    subject_id: str
    features: "torch.Tensor"      # (seq_len, NUM_BLENDSHAPE_FEATURES)
    event_labels: "torch.Tensor"  # (seq_len,) int class ids
    intensity_labels: "torch.Tensor"  # (seq_len,) float 0-1


class BlendshapeSequenceDataset(Dataset):
    """
    Loads preprocessed (MediaPipe-extracted) sequence examples from a
    dataset's processed/ directory. See prepare_dataset() for how raw
    dataset video gets turned into these sequence files.
    """

    def __init__(self, examples: List[SequenceExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.features, ex.event_labels, ex.intensity_labels


def subject_level_split(examples: List[SequenceExample], train_frac=0.7, val_frac=0.15):
    """
    Splits by SUBJECT, not by frame/sequence -- the same person must never
    appear in both train and validation/test (spec section 7).
    """
    subjects = sorted({ex.subject_id for ex in examples})
    n_train = int(len(subjects) * train_frac)
    n_val = int(len(subjects) * val_frac)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])

    return (
        [ex for ex in examples if ex.subject_id in train_subjects],
        [ex for ex in examples if ex.subject_id in val_subjects],
        [ex for ex in examples if ex.subject_id in test_subjects],
    )


def prepare_dataset(dataset_name: str, raw_dir: str, processed_dir: str) -> List[SequenceExample]:
    """
    Runs MediaPipe landmark/blendshape extraction over a dataset's raw
    video files and produces timestamped feature sequences, labeled only
    with what that dataset's own labels actually support (spec section 7 --
    "generate only labels actually supported by the dataset").

    NOT IMPLEMENTED: no dataset has been obtained to implement this
    against (see module docstring). Raises to make that explicit rather
    than silently returning an empty/fake dataset.
    """
    raise NotImplementedError(
        f"prepare_dataset('{dataset_name}') has no data to process -- "
        f"'{raw_dir}' would need to contain that dataset's raw videos, "
        "obtained via that dataset's own (manually gated) access process. "
        "See this file's module docstring."
    )


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Event precision/recall/F1 (per-timestep, treating "which event type is
    active" as the classification target) -- not emotion accuracy.
    """
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for features, event_labels, _ in loader:
            features, event_labels = features.to(device), event_labels.to(device)
            logits, _ = model(features)
            preds = logits.argmax(dim=-1)

            positive_mask = event_labels != 0  # class 0 = "no event"
            pred_positive_mask = preds != 0

            tp += ((preds == event_labels) & positive_mask).sum().item()
            fp += (pred_positive_mask & ~positive_mask).sum().item()
            fn += (~pred_positive_mask & positive_mask).sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"event_precision": precision, "event_recall": recall, "event_f1": f1}


def train(
    train_examples: List[SequenceExample],
    val_examples: List[SequenceExample],
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    resume_from: Optional[str] = None,
) -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FacialTemporalEventModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info("Resumed training from %s at epoch %d", resume_from, start_epoch)

    train_loader = DataLoader(BlendshapeSequenceDataset(train_examples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(BlendshapeSequenceDataset(val_examples), batch_size=batch_size)

    event_criterion = nn.CrossEntropyLoss()
    intensity_criterion = nn.MSELoss()

    for epoch in range(start_epoch, epochs):
        model.train()
        for features, event_labels, intensity_labels in train_loader:
            features = features.to(device)
            event_labels = event_labels.to(device)
            intensity_labels = intensity_labels.to(device)

            optimizer.zero_grad()
            event_logits, intensity_pred = model(features)
            loss = (
                event_criterion(event_logits.reshape(-1, event_logits.size(-1)), event_labels.reshape(-1))
                + intensity_criterion(intensity_pred.squeeze(-1), intensity_labels)
            )
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_loader, device)
        logger.info("Epoch %d: val_f1=%.4f", epoch, metrics["event_f1"])

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": metrics,
        }, checkpoint_path)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ck+", "dfew", "ferv39k"], default="ck+")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.error(
        "No dataset is available to train on -- %s must be obtained via its own "
        "manual access process first. See this file's module docstring. Exiting.",
        args.dataset,
    )
