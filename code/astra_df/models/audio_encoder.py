"""Audio encoder operating on MFCC sequences.

A stack of 1-D convolutions over the time axis preserves temporal
resolution at the encoder output, which we then feed to the temporal
backbone exactly like the visual stream. The MFCC time axis is the
"sequence" axis; coefficients are channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 13,
        hidden_channels: int = 128,
        out_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for i in range(num_layers):
            c_out = hidden_channels * (2 ** min(i, 2))  # 128, 256, 512 cap
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Conv1d(c_in, out_dim, kernel_size=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_a, n_mfcc) -> (B, T_a, out_dim)."""
        # (B, T_a, C) -> (B, C, T_a) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return self.norm(x)
