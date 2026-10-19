"""Output heads of ASTRA-DF.

Three lightweight MLPs:

* ``ClassificationHead`` -> clip-level real/fake logit.
* ``FrameSyncHead``      -> per-frame fake probability (used for both
  per-frame loss and for the explainability timeline).
* ``LocalizationHead``   -> (start_norm, end_norm) of the fake interval
  inside the clip, both in [0, 1]; we apply a sigmoid before loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int = 256, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, fused_video: torch.Tensor, fused_audio: torch.Tensor) -> torch.Tensor:
        # Pool both modalities (temporal mean) then concatenate-summarize.
        v = fused_video.mean(dim=1)
        a = fused_audio.mean(dim=1)
        return self.net(v + a)


class FrameSyncHead(nn.Module):
    """Maps fused video tokens to per-frame fake probabilities."""

    def __init__(self, d_model: int = 256, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused_video: torch.Tensor) -> torch.Tensor:
        return self.net(fused_video).squeeze(-1)  # (B, T_v)


class LocalizationHead(nn.Module):
    """Predicts (start, end) of the fake interval normalised to [0, 1]."""

    def __init__(self, d_model: int = 256, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, fused_video: torch.Tensor) -> torch.Tensor:
        pooled = fused_video.mean(dim=1)
        return torch.sigmoid(self.net(pooled))  # (B, 2) in [0, 1]
