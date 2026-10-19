"""Bidirectional LSTM temporal backbone (ablation baseline).

Matches the AVSFF reference architecture (Javed et al., 2025) so that the
ablation table in the paper isolates the gain from cross-modal attention
rather than from the temporal model itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBiLSTM(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        # Project back to d_model so the downstream attention bridge sees
        # the same width regardless of backbone choice.
        self.proj = nn.Linear(hidden_dim * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> (B, T, d)."""
        y, _ = self.lstm(x)
        y = self.proj(y)
        return self.norm(y)
