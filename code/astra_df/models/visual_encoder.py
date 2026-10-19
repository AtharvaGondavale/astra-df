"""Per-frame visual encoder for mouth-ROI clips.

Wraps ``torchvision.models.mobilenet_v3_small`` so the whole clip is
processed in a single forward pass. The temporal axis is folded into the
batch axis, then re-expanded, so the encoder stays purely 2-D while the
caller receives a ``(B, T, d)`` sequence ready for the temporal backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class VisualEncoder(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v3_small", out_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        if backbone != "mobilenet_v3_small":
            raise ValueError(f"Unsupported backbone: {backbone}")

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        # Strip the original classifier; keep features + adaptive pool.
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        backbone_dim = 576  # mobilenet_v3_small last channels
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) -> (B, T, out_dim)."""
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.pool(self.features(x)).flatten(1)  # (B*T, 576)
        feat = self.proj(feat)                          # (B*T, out_dim)
        return feat.view(b, t, self.out_dim)
