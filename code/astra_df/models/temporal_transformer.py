"""Lightweight Transformer encoder for the temporal stream.

Plain post-norm Transformer encoder, deliberately small (default: 4 layers,
4 heads, d=256) so the full ASTRA-DF model stays under 10M parameters and
inference stays real-time on CPU. A learned positional embedding is used —
sequences are short (≤81 tokens) so sinusoidal isn't needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> (B, T, d)."""
        t = x.shape[1]
        x = x + self.pos[:, :t]
        return self.encoder(x)
