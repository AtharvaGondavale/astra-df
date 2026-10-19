"""Cross-modal attention bridge.

This module is the explainability primitive of ASTRA-DF. Given visual
tokens :math:`V \\in \\mathbb{R}^{B \\times T_v \\times d}` and audio tokens
:math:`A \\in \\mathbb{R}^{B \\times T_a \\times d}` it produces:

1. A fused video sequence :math:`\\tilde{V}` where each visual frame has
   absorbed information from the audio stream (audio-attended visual).
2. A fused audio sequence :math:`\\tilde{A}` (visual-attended audio).
3. The raw attention weights :math:`\\alpha_{v \\to a} \\in [0, 1]^{T_v \\times T_a}`.
   Because these are *the* weights the model used at inference, they
   double as the explanation: high attention on the wrong audio frame
   for a given visual frame is a sync mismatch.

The bridge is two multi-head cross-attention blocks (V←A and A←V), each
pre-normalised (LayerNorm on query and context before attention, and inside
the residual MLP) with residual connections. We expose the attention
weights on the V←A direction (the natural "what audio is this lip
movement attending to?" reading).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossModalOutput:
    fused_video: torch.Tensor  # (B, T_v, d)
    fused_audio: torch.Tensor  # (B, T_a, d)
    attn_v2a: torch.Tensor      # (B, T_v, T_a) audio-attention from each visual frame
    attn_a2v: torch.Tensor      # (B, T_a, T_v)


class _CrossAttnBlock(nn.Module):
    """One direction of cross-attention: query stream attends to context stream."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (updated_query, attention_weights).

        attention_weights has shape (B, T_q, T_kv) — averaged across heads
        which is the right granularity for our timeline visualization.
        """
        q_n, kv_n = self.q_norm(q), self.kv_norm(kv)
        attended, attn = self.attn(q_n, kv_n, kv_n, need_weights=True, average_attn_weights=True)
        q = q + attended
        q = q + self.ffn(q)
        return q, attn


class CrossModalAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.v2a = _CrossAttnBlock(d_model, n_heads, dropout)
        self.a2v = _CrossAttnBlock(d_model, n_heads, dropout)
        self.d_model = d_model

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> CrossModalOutput:
        fused_video, attn_v2a = self.v2a(video, audio)
        fused_audio, attn_a2v = self.a2v(audio, video)
        return CrossModalOutput(
            fused_video=fused_video,
            fused_audio=fused_audio,
            attn_v2a=attn_v2a,
            attn_a2v=attn_a2v,
        )


# ---------------------------------------------------------------------------
# Per-frame sync confidence derived from the attention weights.
# ---------------------------------------------------------------------------
def per_frame_sync_score(attn_v2a: torch.Tensor) -> torch.Tensor:
    """Convert (B, T_v, T_a) audio-attention into a per-visual-frame sync
    confidence in [0, 1].

    Intuition: when audio and lips are in sync, each visual frame strongly
    attends to its temporally-aligned audio window (a peaky distribution
    centred on a "diagonal" entry). When out-of-sync, the attention smears
    out. We measure peakedness with negative entropy, mapped through a
    sigmoid to [0, 1] where 1 = sharp, 0 = uniform.
    """
    eps = 1e-8
    p = attn_v2a.clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1)                # (B, T_v)
    max_entropy = torch.log(torch.tensor(p.shape[-1], dtype=p.dtype, device=p.device))
    norm_entropy = entropy / (max_entropy + eps)        # in [0, 1]
    return 1.0 - norm_entropy                            # 1 = sharp/in-sync, 0 = smeared
