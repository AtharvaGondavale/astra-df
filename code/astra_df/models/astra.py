"""ASTRA-DF: the full model.

Assembles the visual encoder, audio encoder, temporal backbone (Transformer
or BiLSTM, chosen from config), cross-modal attention bridge, and the three
output heads (clip, frame-sync, localization). The forward pass returns a
dict so downstream training/inference code can pull whichever outputs it
needs without changing signatures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from astra_df.models.audio_encoder import AudioEncoder
from astra_df.models.cross_modal_attention import CrossModalAttention, per_frame_sync_score
from astra_df.models.heads import ClassificationHead, FrameSyncHead, LocalizationHead
from astra_df.models.temporal_bilstm import TemporalBiLSTM
from astra_df.models.temporal_transformer import TemporalTransformer
from astra_df.models.visual_encoder import VisualEncoder


@dataclass
class ASTRAOutput:
    logits: torch.Tensor          # (B, 2) real/fake
    frame_sync: torch.Tensor       # (B, T_v) per-frame fake prob (logits)
    interval: torch.Tensor         # (B, 2) normalised (start, end)
    attn_v2a: torch.Tensor          # (B, T_v, T_a) cross-modal attention
    attn_a2v: torch.Tensor          # (B, T_a, T_v)
    sync_confidence: torch.Tensor   # (B, T_v) derived from attention sharpness


def _build_backbone(cfg: dict, d_model: int) -> nn.Module:
    btype = cfg.get("type", "transformer").lower()
    if btype == "transformer":
        return TemporalTransformer(
            d_model=d_model,
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 4),
            ffn_dim=cfg.get("ffn_dim", 512),
            dropout=cfg.get("dropout", 0.1),
        )
    if btype == "bilstm":
        return TemporalBiLSTM(
            d_model=d_model,
            hidden_dim=cfg.get("hidden_dim", 128),
            n_layers=cfg.get("n_layers", 2),
            dropout=cfg.get("dropout", 0.2),
        )
    raise ValueError(f"Unknown backbone type: {btype}")


class ASTRADF(nn.Module):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        d = model_cfg["d_model"]
        vcfg = model_cfg["visual_encoder"]
        acfg = model_cfg["audio_encoder"]
        bcfg = model_cfg["backbone"]
        xcfg = model_cfg["cross_modal"]
        hcfg = model_cfg["heads"]

        self.visual_encoder = VisualEncoder(
            backbone=vcfg["backbone"], out_dim=d, pretrained=vcfg.get("pretrained", True),
        )
        self.audio_encoder = AudioEncoder(
            in_channels=acfg["in_channels"],
            hidden_channels=acfg["hidden_channels"],
            out_dim=d,
            num_layers=acfg["num_layers"],
        )
        self.visual_temporal = _build_backbone(bcfg, d)
        self.audio_temporal = _build_backbone(bcfg, d)
        self.cross_modal = CrossModalAttention(
            d_model=d, n_heads=xcfg["n_heads"], dropout=xcfg["dropout"],
        )
        self.cls_head = ClassificationHead(d_model=d, **hcfg["classification"])
        self.sync_head = FrameSyncHead(d_model=d, **hcfg["sync"])
        self.loc_head = LocalizationHead(d_model=d, **hcfg["localization"])

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> ASTRAOutput:
        v = self.visual_encoder(video)         # (B, T_v, d)
        a = self.audio_encoder(audio)          # (B, T_a, d)
        v = self.visual_temporal(v)
        a = self.audio_temporal(a)
        fused = self.cross_modal(v, a)

        logits = self.cls_head(fused.fused_video, fused.fused_audio)
        frame_sync = self.sync_head(fused.fused_video)
        interval = self.loc_head(fused.fused_video)
        sync_conf = per_frame_sync_score(fused.attn_v2a)

        return ASTRAOutput(
            logits=logits,
            frame_sync=frame_sync,
            interval=interval,
            attn_v2a=fused.attn_v2a,
            attn_a2v=fused.attn_a2v,
            sync_confidence=sync_conf,
        )

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(cfg: dict) -> ASTRADF:
    """Convenience factory mirroring the YAML config layout."""
    return ASTRADF(cfg["model"])
