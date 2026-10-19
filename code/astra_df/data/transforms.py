"""Lightweight augmentations operating on a sample dict from LAVDFDataset.

Each transform receives and returns the standard sample dict with at least
``video`` (T, 3, H, W) and ``audio`` (T_a, n_mfcc) tensors. Augmentations
intentionally stay simple: we don't want to introduce artifacts that *look
like* the manipulations we're trying to detect.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class HorizontalFlip:
    p: float = 0.5

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            sample["video"] = torch.flip(sample["video"], dims=[-1])
        return sample


@dataclass
class ColorJitter:
    p: float = 0.3
    brightness: float = 0.15
    contrast: float = 0.15

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample
        v = sample["video"]
        b = 1.0 + (random.random() * 2 - 1) * self.brightness
        c = 1.0 + (random.random() * 2 - 1) * self.contrast
        mean = v.mean(dim=(-1, -2, -3), keepdim=True)
        v = (v - mean) * c + mean
        v = (v * b).clamp(0.0, 1.0)
        sample["video"] = v
        return sample


@dataclass
class GaussianAudioNoise:
    p: float = 0.2
    snr_db: float = 30.0

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample
        a = sample["audio"]
        rms = a.pow(2).mean().sqrt() + 1e-8
        noise_rms = rms / (10 ** (self.snr_db / 20.0))
        sample["audio"] = a + torch.randn_like(a) * noise_rms
        return sample


@dataclass
class Compose:
    transforms: list

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_train_transform(cfg) -> Compose:
    """Builds an augmentation pipeline from the data.augment YAML block."""
    ts = []
    if cfg.get("horizontal_flip_p", 0) > 0:
        ts.append(HorizontalFlip(p=cfg["horizontal_flip_p"]))
    if cfg.get("color_jitter_p", 0) > 0:
        ts.append(ColorJitter(p=cfg["color_jitter_p"]))
    if cfg.get("gaussian_noise_p", 0) > 0:
        ts.append(GaussianAudioNoise(p=cfg["gaussian_noise_p"], snr_db=cfg.get("audio_noise_db", 30.0)))
    return Compose(ts)
