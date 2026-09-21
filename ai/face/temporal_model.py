"""
Lightweight temporal model architecture for facial MOVEMENT/EVENT
segmentation (onset/apex/offset/intensity) over a MediaPipe blendshape
feature sequence -- NOT an emotion classifier.

This defines the model only. It is not trained (see train_temporal_model.py
for why: the required datasets -- CK+, DFEW, FERV39k -- all require manual,
license-gated registration this environment cannot complete). Until a
checkpoint exists, ai/face/events.py's threshold-based detector is what
actually runs in the pipeline.

Architecture: a small GRU encoder (per the spec: "small Transformer encoder
or GRU/LSTM depending on the repository's existing ML stack" -- this repo's
existing ML stack is PyTorch throughout, so this uses torch.nn), followed by
a per-timestep event head. Deliberately small: this is movement
segmentation over ~52 blendshape channels, not a large vision model.
"""
from typing import Optional

import torch
import torch.nn as nn

# MediaPipe Face Landmarker's standard blendshape category count.
NUM_BLENDSHAPE_FEATURES = 52


class FacialTemporalEventModel(nn.Module):
    """
    Input:  (batch, sequence_len, NUM_BLENDSHAPE_FEATURES) blendshape scores
    Output: per-timestep event logits + intensity, for movement segmentation
            (onset/active/offset), not emotion classification.
    """

    def __init__(
        self,
        input_dim: int = NUM_BLENDSHAPE_FEATURES,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_event_types: int = 4,  # e.g. blink, smile, eyebrow_raise, none
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        encoder_out_dim = hidden_dim * 2  # bidirectional

        # Per-timestep heads: which movement type is active, and how intense.
        self.event_type_head = nn.Linear(encoder_out_dim, num_event_types)
        self.intensity_head = nn.Sequential(
            nn.Linear(encoder_out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        x: (batch, seq_len, input_dim)
        Returns: event_type_logits (batch, seq_len, num_event_types),
                 intensity (batch, seq_len, 1)
        """
        encoded, _ = self.encoder(x)
        event_type_logits = self.event_type_head(encoded)
        intensity = self.intensity_head(encoded)
        return event_type_logits, intensity
