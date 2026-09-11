"""
Torch dataset over ``ser/manifest.csv`` for the wav2vec2-xls-r-300m fine-tune.

Loads each clip at 16 kHz mono, trims/pads to ``max_seconds``, and returns
``input_values`` (+ ``attention_mask``) plus the integer ``label``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .corpora import MANIFEST_CSV
from .labels import UNIFIED_EMOTIONS

SR = 16_000


def read_manifest(split: str | None = None, path: Path = MANIFEST_CSV) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m ai.voice.prepare`")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


def class_weights(rows: list[dict]) -> torch.Tensor:
    counts = np.zeros(len(UNIFIED_EMOTIONS), dtype="float64")
    for r in rows:
        counts[int(r["label_id"])] += 1
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


class SERDataset(Dataset):
    def __init__(self, rows: list[dict], feature_extractor, max_seconds: float = 7.0):
        self.rows = rows
        self.fe = feature_extractor
        self.max_len = int(max_seconds * SR)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, fpath: str) -> np.ndarray:
        import librosa

        y, _ = librosa.load(fpath, sr=SR, mono=True)
        if y.size == 0:
            y = np.zeros(SR, dtype="float32")
        if y.size > self.max_len:
            start = (y.size - self.max_len) // 2
            y = y[start:start + self.max_len]
        return y.astype("float32")

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        y = self._load(row["path"])
        feats = self.fe(y, sampling_rate=SR, return_attention_mask=True,
                        max_length=self.max_len, truncation=True)
        item = {"input_values": torch.tensor(feats["input_values"][0], dtype=torch.float32),
                "labels": int(row["label_id"])}
        if "attention_mask" in feats:
            item["attention_mask"] = torch.tensor(feats["attention_mask"][0], dtype=torch.long)
        return item


class Collator:
    """Dynamic padding to the longest clip in the batch."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, batch: list[dict]) -> dict:
        inputs = [{"input_values": b["input_values"]} for b in batch]
        padded = self.fe.pad(inputs, padding=True, return_tensors="pt")
        padded["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return padded
