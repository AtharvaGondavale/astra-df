"""Training loop for ASTRA-DF.

Standard PyTorch loop with AMP, AdamW + cosine schedule, gradient clipping,
and a clip-level F1 early-stop on the val split. Designed to be runnable
on a single GPU / Colab T4 or to fall back to CPU for the smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from astra_df.data.lavdf_dataset import LAVDFDataset, collate_fn
from astra_df.data.transforms import build_train_transform
from astra_df.losses import build_loss
from astra_df.models import build_model
from astra_df.utils import RunningMean, binary_metrics, get_logger


log = get_logger("train")


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------
def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_path = path.parent / "base.yaml"
    if path.name != "base.yaml" and base_path.exists():
        with open(base_path) as f:
            base = yaml.safe_load(f)
        cfg = _deep_update(base, cfg)
    return cfg


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_dataloaders(cfg: dict):
    d = cfg["data"]
    train_tf = build_train_transform(d["augment"]) if d["augment"].get("enabled", True) else None
    train_ds = LAVDFDataset(d["metadata_csv"], lavdf_root=d.get("root"), split="train", transform=train_tf)
    val_ds = LAVDFDataset(d["metadata_csv"], lavdf_root=d.get("root"), split="val")
    loader_kwargs = dict(batch_size=d["batch_size"], num_workers=d["num_workers"],
                         pin_memory=d.get("pin_memory", False), collate_fn=collate_fn)
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
    )


# ---------------------------------------------------------------------------
# training step
# ---------------------------------------------------------------------------
def _train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, cfg) -> dict:
    model.train()
    metrics = RunningMean()
    grad_clip = cfg["optim"].get("grad_clip", 0.0)
    use_amp = cfg["optim"].get("amp", False) and device.type == "cuda"
    log_every = cfg["eval"].get("log_every", 50)

    for step, batch in enumerate(loader):
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        frame_labels = batch["frame_labels"].to(device, non_blocking=True)
        interval = batch["interval"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            out = model(video, audio)
            loss = loss_fn(out.logits, out.frame_sync, out.interval, labels, frame_labels, interval)

        if use_amp:
            scaler.scale(loss.total).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.total.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        metrics.add(
            loss=loss.total.item(), cls=loss.cls.item(),
            sync=loss.sync.item(), interval=loss.interval.item(),
        )
        if step % log_every == 0:
            m = metrics.mean()
            log.info("train step=%d loss=%.4f cls=%.4f sync=%.4f int=%.4f",
                     step, m["loss"], m["cls"], m["sync"], m["interval"])
    return metrics.mean()


@torch.no_grad()
def _validate(model, loader, loss_fn, device) -> dict:
    model.eval()
    metrics = RunningMean()
    all_labels, all_preds, all_scores = [], [], []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        frame_labels = batch["frame_labels"].to(device, non_blocking=True)
        interval = batch["interval"].to(device, non_blocking=True)
        out = model(video, audio)
        loss = loss_fn(out.logits, out.frame_sync, out.interval, labels, frame_labels, interval)
        probs = torch.softmax(out.logits, dim=-1)[:, 1]
        preds = (probs > 0.5).long()
        metrics.add(val_loss=loss.total.item(), val_cls=loss.cls.item())
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_scores.extend(probs.tolist())

    bm = binary_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores))
    return {**metrics.mean(), **{f"val_{k}": v for k, v in bm.items()}}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg["experiment"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    log.info("Model params: %s", model.count_parameters())

    loss_fn = build_loss(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    epochs = cfg["optim"]["epochs"]
    warmup = cfg["optim"].get("warmup_epochs", 0)
    if warmup > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = GradScaler(enabled=cfg["optim"].get("amp", False) and device.type == "cuda")
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = _train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, cfg)
        val_metrics = _validate(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0
        record = {"epoch": epoch, **train_metrics, **val_metrics, "time_sec": elapsed}
        history.append(record)
        log.info("epoch=%d val_acc=%.4f val_f1=%.4f val_auc=%.4f time=%.1fs",
                 epoch, val_metrics.get("val_accuracy", 0.0), val_metrics.get("val_f1", 0.0),
                 val_metrics.get("val_auc", 0.0), elapsed)
        if val_metrics.get("val_f1", 0.0) > best_f1:
            best_f1 = val_metrics["val_f1"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, out_dir / "best.pt")
            log.info("Saved new best (f1=%.4f)", best_f1)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
