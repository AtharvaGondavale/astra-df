"""Multi-task loss for ASTRA-DF.

L_total = alpha * L_classification(logits, y)
        + beta  * L_sync(frame_sync_logits, frame_y) * mask_fake_clip
        + gamma * L_interval(pred_interval, gt_interval) * mask_fake_clip

The frame-sync and interval losses are masked out on real clips so the
model is not pushed to invent a fake interval where none exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    cls: torch.Tensor
    sync: torch.Tensor
    interval: torch.Tensor


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.25,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,            # (B, 2)
        frame_sync: torch.Tensor,        # (B, T)
        interval: torch.Tensor,          # (B, 2) sigmoid'd
        target_label: torch.Tensor,      # (B,)
        target_frame: torch.Tensor,      # (B, T)
        target_interval: torch.Tensor,   # (B, 2) values in [-1, 1] (-1 = real clip)
    ) -> LossOutput:
        cls = self.cls_loss(logits, target_label)

        # mask: 1 where the clip is fake (we have valid frame/interval labels)
        fake_mask = (target_label == 1).float()  # (B,)
        denom_b = fake_mask.sum().clamp_min(1.0)

        # Frame-level BCE (averaged over time then over fake clips).
        bce = self.bce(frame_sync, target_frame).mean(dim=-1)  # (B,)
        sync = (bce * fake_mask).sum() / denom_b

        # Interval L1 on the fake-only subset.
        valid_interval = (target_interval[:, 0] >= 0).float() * fake_mask  # (B,)
        l1 = F.l1_loss(interval, target_interval.clamp(min=0.0), reduction="none").sum(dim=-1)
        loc = (l1 * valid_interval).sum() / valid_interval.sum().clamp_min(1.0)

        total = self.alpha * cls + self.beta * sync + self.gamma * loc
        return LossOutput(total=total, cls=cls, sync=sync, interval=loc)


def build_loss(cfg: dict) -> MultiTaskLoss:
    lcfg = cfg["loss"]
    return MultiTaskLoss(
        alpha=lcfg["alpha"], beta=lcfg["beta"], gamma=lcfg["gamma"],
        label_smoothing=lcfg.get("label_smoothing", 0.0),
    )
