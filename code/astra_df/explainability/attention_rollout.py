"""Attention rollout for the cross-modal bridge.

For a stack of attention matrices ``A_1, A_2, ..., A_L`` (one per layer),
rollout returns the product of ``A_i + I`` (with row-normalisation), giving
the *effective* attention from each query token to each key token after
information has been mixed through all layers (Abnar & Zuidema, 2020).

We only have one explicit cross-attention layer in ASTRA-DF, so rollout
here is conceptually the same as the raw ``attn_v2a`` weights but with the
residual ``+ I`` correction applied, which makes the visualisation smoother.
"""
from __future__ import annotations

import torch


def attention_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    """Compute attention rollout from a list of (B, T_q, T_kv) matrices.

    For non-square matrices we treat each layer as a transport step; the
    rollout is ``A_L @ ... @ A_2 @ A_1`` with row-normalisation. We do not
    add identity since query and key spaces differ.
    """
    rolled = attentions[0]
    rolled = rolled / rolled.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    for a in attentions[1:]:
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rolled = torch.bmm(rolled, a)
    return rolled
