"""
ASTRA-DF :: Stage 2 -- experiment driver (run on Colab T4)
==========================================================

Consumes the cache built by prepare_subset.py and produces every number the
paper needs, appending each result to results.json as soon as it is measured
so a partial run still leaves you with usable tables.

Stages
------
    stats       parameter counts, FLOPs, CPU/GPU latency         (~1 min)
    main        train + eval Transformer and BiLSTM              (bulk of time)
    fusion      concat / V->A only / A->V only ablation
    modality    visual-only / audio-only ablation
    lossweight  (alpha, beta, gamma) sensitivity sweep
    hparam      clip-duration and mouth-ROI sweeps
    robust      audio-noise SNR degradation
    gradcam     spatial Grad-CAM mass inside the lip box
    figures     regenerate every paper figure from real data

    python run_experiments.py --cache ./astra_cache --stages stats,main
    python run_experiments.py --cache ./astra_cache --stages all --epochs 15

Fixes applied relative to the repo
----------------------------------
1. split naming: LAV-DF ships train/dev/test; train.py asks for "val".
   The cache already maps dev -> val, so the val loader is non-empty.
2. localisation AP is computed **per video and then averaged over videos**
   (evaluate.py pools all predictions globally). The LAV-DF paper (Cai et al.,
   2022) uses the same IoU thresholds but its own AP protocol, so the two are
   not directly comparable.
3. audio encoder channel widths are reported as measured, not as assumed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------
# repo import
# --------------------------------------------------------------------------
def _import_astra(repo_root: str):
    sys.path.insert(0, str(Path(repo_root).resolve()))
    from astra_df.models.audio_encoder import AudioEncoder
    from astra_df.models.cross_modal_attention import CrossModalAttention, per_frame_sync_score
    from astra_df.models.heads import ClassificationHead, FrameSyncHead, LocalizationHead
    from astra_df.models.temporal_bilstm import TemporalBiLSTM
    from astra_df.models.temporal_transformer import TemporalTransformer
    from astra_df.models.visual_encoder import VisualEncoder

    return dict(
        VisualEncoder=VisualEncoder, AudioEncoder=AudioEncoder,
        TemporalTransformer=TemporalTransformer, TemporalBiLSTM=TemporalBiLSTM,
        CrossModalAttention=CrossModalAttention,
        per_frame_sync_score=per_frame_sync_score,
        ClassificationHead=ClassificationHead, FrameSyncHead=FrameSyncHead,
        LocalizationHead=LocalizationHead,
    )


M = {}        # populated in main()
_ONLY = None  # optional whitelist of run names (--only)
_RETRAIN = set()  # run names to force-retrain (--retrain)
# Identity of the checkpoint that evaluation-only stages used. Cross-dataset
# and Grad-CAM records are meaningless without it: they say nothing about which
# model produced them, so a file can silently mix a 10-epoch zero-shot result
# with 15-epoch in-domain results.
_CKPT_INFO = None

AUDIO_SR = 16000
N_MFCC = 13
N_FFT = 512
HOP = 200


# ==========================================================================
# Dataset over the packed cache
# ==========================================================================
class PackedLAVDF(Dataset):
    """1-second (configurable) clips decoded on the fly from the packed cache.

    Every experimental knob the paper ablates is a constructor argument, so
    no re-preprocessing is ever required:
        clip_sec    0.5 / 1.0 / 2.0
        roi         64 / 96 / 128
        noise_snr   audio-noise robustness (dB, None = clean)
    """

    def __init__(self, cache: str | Path, split: str, clip_sec: float = 1.0,
                 roi: int = 96, fps: int = 30, noise_snr: float | None = None,
                 augment: bool = False, seed: int = 0, audio_shift: float = 0.0):
        import pandas as pd

        self.cache = Path(cache)
        # Delay (s > 0) or advance (s < 0) the audio relative to the video, in
        # seconds. Used only by the synchronisation test of the attention
        # alignment metrics; 0.0 everywhere else.
        self.audio_shift = audio_shift
        self.clip_sec = clip_sec
        self.roi = roi
        self.fps = fps
        self.T = int(round(clip_sec * fps))
        self.noise_snr = noise_snr
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        idx = np.load(self.cache / "index.npz")
        self.frame_off = idx["frame_offsets"]
        self.vptr = idx["video_frame_ptr"]
        self.audio_off = idx["audio_offsets"]

        df = pd.read_csv(self.cache / "videos.csv")
        df["_row"] = np.arange(len(df))
        self.videos = df[df["split"] == split].reset_index(drop=True)

        self._frames_fh = None
        self._audio_fh = None

        # Enumerate clips: (video_row, clip_index, start_sec)
        self.clips = []
        for _, v in self.videos.iterrows():
            n_clips = int(v["duration"] // clip_sec)
            for c in range(n_clips):
                self.clips.append((int(v["_row"]), c, c * clip_sec))
        self.meta = df

    # -------- lazy file handles (fork-safe for DataLoader workers) --------
    def _fh(self):
        if self._frames_fh is None:
            self._frames_fh = open(self.cache / "frames.bin", "rb")
            self._audio_fh = open(self.cache / "audio.bin", "rb")
        return self._frames_fh, self._audio_fh

    def __len__(self):
        return len(self.clips)

    # -------- labels from official fake_periods --------
    def _labels(self, periods, start, end):
        ft = np.linspace(start, end, self.T, endpoint=False)
        lab = np.zeros(self.T, np.float32)
        clip_label, lo, hi = 0, None, None
        for p in periods:
            ps, pe = float(p[0]), float(p[1])
            o_lo, o_hi = max(ps, start), min(pe, end)
            if o_hi > o_lo:
                clip_label = 1
                lab[(ft >= ps) & (ft < pe)] = 1.0
                lo = o_lo if lo is None else min(lo, o_lo)
                hi = o_hi if hi is None else max(hi, o_hi)
        if lo is None:
            return clip_label, lab, np.array([-1.0, -1.0], np.float32)
        d = end - start
        return clip_label, lab, np.array([(lo - start) / d, (hi - start) / d], np.float32)

    def __getitem__(self, i):
        import cv2
        import librosa

        fh_f, fh_a = self._fh()
        vrow, cidx, start = self.clips[i]
        v = self.meta.iloc[vrow]
        end = start + self.clip_sec

        # ---- video ----
        f0 = int(self.vptr[vrow])
        n_avail = int(self.vptr[vrow + 1]) - f0
        first = int(round(start * self.fps))
        frames = np.empty((self.T, self.roi, self.roi, 3), np.uint8)
        for t in range(self.T):
            j = min(first + t, n_avail - 1)
            a, b = int(self.frame_off[f0 + j]), int(self.frame_off[f0 + j + 1])
            fh_f.seek(a)
            buf = np.frombuffer(fh_f.read(b - a), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((self.roi, self.roi, 3), np.uint8)
            elif img.shape[0] != self.roi:
                img = cv2.resize(img, (self.roi, self.roi), interpolation=cv2.INTER_AREA)
            frames[t] = img

        # ---- audio ----
        a0 = int(self.audio_off[vrow])
        n_s = int(self.audio_off[vrow + 1]) - a0
        need = int(self.clip_sec * AUDIO_SR)
        # Audio shifted by s: the visual frame at time t is paired with the
        # audio at time t - s (s > 0 delays the audio). Out-of-range samples
        # are zero-padded.
        s0 = int(round((start - self.audio_shift) * AUDIO_SR))
        lo, hi = max(0, s0), min(s0 + need, n_s)
        fh_a.seek((a0 + lo) * 2)
        pcm = np.frombuffer(fh_a.read(max(0, hi - lo) * 2), np.int16).astype(np.float32) / 32768.0
        pcm = np.pad(pcm, (lo - s0, max(0, need - (lo - s0) - pcm.size)))[:need]

        if self.noise_snr is not None:
            rms = np.sqrt((pcm ** 2).mean()) + 1e-9
            npow = rms / (10 ** (self.noise_snr / 20.0))
            pcm = pcm + self.rng.randn(pcm.size).astype(np.float32) * npow

        mfcc = librosa.feature.mfcc(y=pcm, sr=AUDIO_SR, n_mfcc=N_MFCC,
                                    n_fft=N_FFT, hop_length=HOP).T.astype(np.float32)
        mfcc = (mfcc - mfcc.mean(0, keepdims=True)) / (mfcc.std(0, keepdims=True) + 1e-6)

        # ---- labels ----
        periods = json.loads(v["fake_periods"])
        clip_label, frame_lab, interval = self._labels(periods, start, end)

        vid = frames.astype(np.float32) / 255.0
        vid = torch.from_numpy(vid).permute(0, 3, 1, 2).contiguous()

        if self.augment:
            if self.rng.rand() < 0.5:
                vid = torch.flip(vid, dims=[-1])
            if self.rng.rand() < 0.3:
                b = 1.0 + (self.rng.rand() * 2 - 1) * 0.15
                vid = (vid * b).clamp(0, 1)

        return {
            "video": vid,
            "audio": torch.from_numpy(mfcc),
            "label": torch.tensor(clip_label, dtype=torch.long),
            "frame_labels": torch.from_numpy(frame_lab),
            "interval": torch.from_numpy(interval),
            "video_id": str(v["video_id"]),
            "start_sec": float(start),
            "end_sec": float(end),
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


# ==========================================================================
# Model variants (reuses the repo's modules)
# ==========================================================================
class _Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def _act(kind, ch):
    return {"relu": lambda: nn.ReLU(inplace=True), "prelu": lambda: nn.PReLU(ch),
            "swish": _Swish}[kind]()


class _LRBasicBlock(nn.Module):
    """BasicBlock with the attribute names of mpc001/Lipreading_using_TCN
    (conv1, bn1, relu1, conv2, bn2, relu2, downsample) so LRW checkpoints load
    without remapping, and with the same activation as the checkpoint."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act="swish"):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = _act(act, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = _act(act, planes)
        self.downsample = downsample

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + res)


class _LRResNet18(nn.Module):
    def __init__(self, act="swish"):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make(64, 2, 1, act)
        self.layer2 = self._make(128, 2, 2, act)
        self.layer3 = self._make(256, 2, 2, act)
        self.layer4 = self._make(512, 2, 2, act)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make(self, planes, blocks, stride, act):
        ds = None
        if stride != 1 or self.inplanes != planes:
            ds = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                               nn.BatchNorm2d(planes))
        layers = [_LRBasicBlock(self.inplanes, planes, stride, ds, act)]
        self.inplanes = planes
        layers += [_LRBasicBlock(planes, planes, act=act) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


class LipreadEncoder(nn.Module):
    r"""Visual front-end of the LRW lipreading model of Martinez et al. (2020),
    mpc001/Lipreading_using_Temporal_Convolutional_Networks: a 3-D convolutional
    stem over time followed by a 2-D ResNet-18, applied to grey-scale mouth crops.

    The point of this encoder is the *initialisation*: weights pretrained for
    word-level lipreading on LRW (Haliassos et al., 2021, report that such
    weights transfer far better to forgery detection than ImageNet ones). To use
    those weights faithfully, the module reproduces the checkpoint exactly:

      * parameter names frontend3D.* and trunk.layer{1..4}.*, so the published
        ``model_state_dict`` loads without remapping (the TCN back-end is ignored);
      * the checkpoint's activation (Swish for the current model zoo, PReLU or
        ReLU for older ones), detected from the checkpoint and the model zoo
        config -- a ReLU network with Swish-trained weights would be a different
        function;
      * the published input pipeline: grey-scale, centre crop 88x88, then
        (x - 0.421) / 0.165 on [0, 1] intensities. Our cache stores frames as
        OpenCV BGR, so grey = 0.114 B + 0.587 G + 0.299 R.

    Without a checkpoint the encoder is random and answers no scientific
    question; the stage refuses to run in that case. Parameter count rises to
    roughly 17 M for the full model, so it is reported as a separate
    configuration.
    """

    MEAN, STD, CROP = 0.421, 0.165, 88

    def __init__(self, out_dim=256, ckpt=None, act="auto"):
        super().__init__()
        sd = self._read(ckpt) if ckpt else None
        if act == "auto":
            act = ("prelu" if sd is not None and "frontend3D.2.weight" in sd else "swish")
        self.act = act
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                      padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64), _act(act, 64),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = _LRResNet18(act)
        self.proj = nn.Sequential(nn.Linear(512, out_dim), nn.GELU(), nn.LayerNorm(out_dim))
        self.loaded_pretrained = False
        if sd is not None:
            self._load(sd, ckpt)
        else:
            print("  ! LipreadEncoder built WITHOUT pretrained weights (plumbing "
                  "test only); pass --lipread-ckpt for the actual experiment.")

    @staticmethod
    def _read(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(sd, dict) and key in sd:
                sd = sd[key]
                break
        return {k.replace("module.", ""): v for k, v in sd.items()}

    def _load(self, sd, ckpt):
        own = self.state_dict()
        mine = [k for k in own if k.startswith(("frontend3D", "trunk"))]
        matched = {k: sd[k] for k in mine if k in sd and tuple(sd[k].shape) == tuple(own[k].shape)}
        print(f"  LipreadEncoder[{self.act}]: {len(matched)}/{len(mine)} frontend+trunk "
              f"tensors matched from {ckpt}")
        if len(matched) != len(mine):
            missing = [k for k in mine if k not in matched][:8]
            raise RuntimeError(
                f"only {len(matched)}/{len(mine)} frontend/trunk tensors matched from "
                f"{ckpt} (e.g. missing {missing}). The checkpoint does not correspond "
                "to this module; refusing to report the run as lipreading-pretrained.")
        self.load_state_dict(matched, strict=False)
        self.loaded_pretrained = True

    def forward(self, x):                      # x: (B, T, 3, H, W), BGR in [0, 1]
        B, T, _, H, W = x.shape
        g = 0.114 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.299 * x[:, :, 2]   # (B, T, H, W)
        if H > self.CROP and W > self.CROP:
            t0, l0 = (H - self.CROP) // 2, (W - self.CROP) // 2
            g = g[:, :, t0:t0 + self.CROP, l0:l0 + self.CROP]
        g = ((g - self.MEAN) / self.STD).unsqueeze(1)                      # (B, 1, T, h, w)
        f = self.frontend3D(g)                                             # (B, 64, T, h', w')
        f = f.transpose(1, 2).reshape(B * T, 64, *f.shape[-2:])
        f = self.trunk(f)                                                  # (B*T, 512)
        return self.proj(f).view(B, T, -1)


class VariantASTRA(nn.Module):
    """ASTRA-DF with switchable fusion and modalities, for the ablation tables.

    fusion    : 'cross' | 'concat' | 'v2a' | 'a2v'
    modalities: 'va' | 'v' | 'a'
    """

    def __init__(self, backbone="transformer", fusion="cross", modalities="va",
                 d=256, pretrained=True, visual_encoder="mobilenetv3",
                 lipread_ckpt=None, rgb_norm=False, lipread_act="auto"):
        super().__init__()
        # rgb_norm: the cache stores OpenCV BGR frames in [0, 1]; ImageNet
        # MobileNetV3 expects RGB normalised with ImageNet mean/std. The runs
        # reported in the submitted manuscript used neither conversion
        # (rgb_norm=False, kept as the default so their checkpoints load and
        # evaluate unchanged); new runs can opt in with --rgb-norm.
        self.rgb_norm = bool(rgb_norm) and visual_encoder == "mobilenetv3"
        if self.rgb_norm:
            self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
            self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        self.fusion = fusion
        self.modalities = modalities
        self.visual_encoder_kind = visual_encoder
        self.d = d

        if "v" in modalities:
            if visual_encoder == "mobilenetv3":
                self.visual_encoder = M["VisualEncoder"](out_dim=d, pretrained=pretrained)
            elif visual_encoder == "lipread":
                self.visual_encoder = LipreadEncoder(out_dim=d, ckpt=lipread_ckpt,
                                                     act=lipread_act)
            else:
                raise ValueError(f"unknown visual encoder: {visual_encoder}")
            self.visual_temporal = self._backbone(backbone, d)
        if "a" in modalities:
            self.audio_encoder = M["AudioEncoder"](in_channels=N_MFCC, hidden_channels=128,
                                                   out_dim=d, num_layers=3)
            self.audio_temporal = self._backbone(backbone, d)

        if modalities == "va" and fusion != "concat":
            self.cross = M["CrossModalAttention"](d_model=d, n_heads=4, dropout=0.1)

        self.config = dict(backbone=backbone, fusion=fusion, modalities=modalities,
                           visual_encoder=visual_encoder, rgb_norm=self.rgb_norm,
                           lipread_act=(self.visual_encoder.act
                                        if visual_encoder == "lipread" and "v" in modalities
                                        else None))
        self.cls_head = M["ClassificationHead"](d_model=d, hidden_dim=128, dropout=0.3)
        self.sync_head = M["FrameSyncHead"](d_model=d, hidden_dim=64)
        self.loc_head = M["LocalizationHead"](d_model=d, hidden_dim=64)

    @staticmethod
    def _backbone(kind, d):
        if kind == "transformer":
            return M["TemporalTransformer"](d_model=d, n_heads=4, n_layers=4,
                                            ffn_dim=512, dropout=0.1)
        if kind == "bilstm":
            return M["TemporalBiLSTM"](d_model=d, hidden_dim=128, n_layers=2, dropout=0.2)
        raise ValueError(kind)

    def forward(self, video, audio):
        v = a = None
        if "v" in self.modalities:
            if self.rgb_norm:
                video = (video.flip(2) - self.img_mean) / self.img_std
            v = self.visual_temporal(self.visual_encoder(video))
        if "a" in self.modalities:
            a = self.audio_temporal(self.audio_encoder(audio))

        attn_v2a = None
        if self.modalities == "va":
            if self.fusion == "cross":
                o = self.cross(v, a)
                fv, fa, attn_v2a = o.fused_video, o.fused_audio, o.attn_v2a
            elif self.fusion == "v2a":
                fv, attn_v2a = self.cross.v2a(v, a)
                fa = a
            elif self.fusion == "a2v":
                fa, _ = self.cross.a2v(a, v)
                fv = v
            else:  # concat / no cross-attention
                fv, fa = v, a
        elif self.modalities == "v":
            fv, fa = v, torch.zeros_like(v)
        else:
            # Audio-only. The frame-sync head must emit one value per *visual*
            # frame to match the labels (T=30), but the audio stream has its own
            # rate (T_a=81). Resample onto the visual grid rather than letting
            # the shapes silently disagree.
            fa = a
            n_frames = video.shape[1]
            fv = F.interpolate(a.transpose(1, 2), size=n_frames,
                               mode="linear", align_corners=False).transpose(1, 2)

        logits = self.cls_head(fv, fa)
        frame_sync = self.sync_head(fv)
        interval = self.loc_head(fv)
        sync_conf = (M["per_frame_sync_score"](attn_v2a)
                     if attn_v2a is not None else torch.zeros(fv.shape[:2], device=fv.device))
        return dict(logits=logits, frame_sync=frame_sync, interval=interval,
                    attn_v2a=attn_v2a, sync_confidence=sync_conf)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ==========================================================================
# Loss (mirrors astra_df/losses/multi_task.py)
# ==========================================================================
class MultiTask(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.25, ls=0.05):
        super().__init__()
        self.a, self.b, self.g = alpha, beta, gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=ls)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, out, y, yf, yi):
        cls = self.ce(out["logits"], y)
        fake = (y == 1).float()
        d = fake.sum().clamp_min(1.0)
        sync = ((self.bce(out["frame_sync"], yf).mean(-1)) * fake).sum() / d
        valid = (yi[:, 0] >= 0).float() * fake
        l1 = F.l1_loss(out["interval"], yi.clamp(min=0.0), reduction="none").sum(-1)
        loc = (l1 * valid).sum() / valid.sum().clamp_min(1.0)
        return self.a * cls + self.b * sync + self.g * loc, dict(
            cls=cls.item(), sync=sync.item(), loc=loc.item())


# ==========================================================================
# Metrics
# ==========================================================================
def binary_metrics(y, p, s):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    pos, neg = np.asarray(s)[y == 1], np.asarray(s)[y == 0]
    if len(pos) and len(neg):
        diff = pos[:, None] - neg[None, :]
        auc = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
    else:
        auc = float("nan")
    return dict(accuracy=(tp + tn) / max(1, len(y)), precision=prec, recall=rec,
                f1=f1, auc=auc, tp=tp, tn=tn, fp=fp, fn=fn)


def tiou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(1e-9, union)


def ap_at_iou(preds, gts, thr):
    """AP for one video. preds=[(s,e,score)], gts=[(s,e)]."""
    if not gts:
        return None            # undefined -- excluded from the average
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: -x[2])
    matched = [False] * len(gts)
    tps, fps = [], []
    for s, e, _ in preds:
        best, bi = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = tiou((s, e), g)
            if v > best:
                best, bi = v, j
        if best >= thr and bi >= 0:
            matched[bi] = True; tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)
    tps, fps = np.cumsum(tps), np.cumsum(fps)
    rec = tps / len(gts)
    prec = tps / np.maximum(1e-9, tps + fps)
    ap, prev = 0.0, 0.0
    for r, p in zip(rec, prec):
        ap += (r - prev) * p
        prev = r
    return float(ap)


def per_video_map(pred_by_vid, gt_by_vid, ious=(0.5, 0.75, 0.95)):
    """Per-video AP averaged over videos, at the IoU thresholds of LAV-DF."""
    out = {}
    for thr in ious:
        vals = []
        for vid, gts in gt_by_vid.items():
            v = ap_at_iou(pred_by_vid.get(vid, []), list(gts), thr)
            if v is not None:
                vals.append(v)
        out[f"AP@{thr}"] = float(np.mean(vals)) if vals else 0.0
    out["mAP"] = float(np.mean(list(out.values())))
    return out


def build_intervals(frame_probs, times, clip_pred, reg, tau=0.5,
                    min_dur=0.1, merge_gap=0.25, nms_iou=0.3):
    """Video-level timeline (M6): threshold -> runs -> NMS -> merge across short gaps."""
    runs, cur = [], None
    for t, p in zip(times, frame_probs):
        if p > tau:
            cur = [t, t, p] if cur is None else [cur[0], t, max(cur[2], p)]
        elif cur is not None:
            runs.append(cur); cur = None
    if cur is not None:
        runs.append(cur)
    runs = [r for r in runs if (r[1] - r[0]) >= min_dur]
    runs.extend(reg)

    runs.sort(key=lambda r: -r[2])
    keep = []
    for r in runs:
        if all(tiou((r[0], r[1]), (k[0], k[1])) < nms_iou for k in keep):
            keep.append(r)
    keep.sort(key=lambda r: r[0])

    merged = []
    for r in keep:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1] = [merged[-1][0], max(merged[-1][1], r[1]),
                          max(merged[-1][2], r[2])]
        else:
            merged.append(list(r))
    return [(m[0], m[1], m[2]) for m in merged]


# ==========================================================================
# Train / evaluate
# ==========================================================================
def evaluate(model, loader, device, tau=0.5):
    model.eval()
    ys, ps, ss = [], [], []
    vids = []
    frames_by_vid = defaultdict(list)
    reg_by_vid = defaultdict(list)
    gt_by_vid = defaultdict(set)

    with torch.no_grad():
        for b in loader:
            v, a = b["video"].to(device), b["audio"].to(device)
            out = model(v, a)
            prob = torch.softmax(out["logits"], -1)[:, 1].cpu().numpy()
            fs = torch.sigmoid(out["frame_sync"]).cpu().numpy()
            iv = out["interval"].cpu().numpy()

            ys.extend(b["label"].numpy().tolist())
            ps.extend((prob > tau).astype(int).tolist())
            ss.extend(prob.tolist())
            vids.extend(list(b["video_id"]))

            for i in range(len(prob)):
                vid = b["video_id"][i]
                s0, e0 = float(b["start_sec"][i]), float(b["end_sec"][i])
                L = e0 - s0
                T = fs.shape[1]
                times = s0 + (np.arange(T) + 0.5) * (L / T)
                frames_by_vid[vid].extend(zip(times.tolist(), fs[i].tolist()))
                if prob[i] > tau:
                    reg_by_vid[vid].append([s0 + iv[i, 0] * L, s0 + iv[i, 1] * L, float(prob[i])])
                g = b["interval"][i].numpy()
                if g[0] >= 0:
                    gt_by_vid[vid].add((s0 + float(g[0]) * L, s0 + float(g[1]) * L))

    pred_by_vid = {}
    for vid, fp in frames_by_vid.items():
        fp.sort()
        times = [t for t, _ in fp]
        probs = [p for _, p in fp]
        pred_by_vid[vid] = build_intervals(probs, times, None, reg_by_vid.get(vid, []), tau)

    # Detection rate split by manipulation family. LAV-DF flags each source
    # video as visual-only / audio-only / both, so this shows which kind of
    # manipulation the model actually catches.
    breakdown = {}
    try:
        # LAV-DF ids are numeric strings, so pandas infers an int64 index while
        # the dataset yields str -- coerce both sides or every lookup misses.
        meta = loader.dataset.meta.copy()
        meta["video_id"] = meta["video_id"].astype(str)

        # Build plain dicts rather than indexing a DataFrame: a duplicated id
        # makes .loc return a DataFrame instead of a Series, which throws and
        # (previously) silently emptied the entire breakdown.
        mv_of = dict(zip(meta["video_id"], meta["modify_video"].astype(int)))
        ma_of = dict(zip(meta["video_id"], meta["modify_audio"].astype(int)))

        n_dup = int(meta["video_id"].duplicated().sum())
        if n_dup:
            print(f"    ! {n_dup} duplicate video ids in metadata; "
                  "breakdown uses the last occurrence of each")

        buckets = defaultdict(lambda: [0, 0])
        missed = 0
        for vid, y, p in zip(vids, ys, ps):
            if y != 1:
                continue
            if vid not in mv_of:
                missed += 1
                continue
            mv, ma = mv_of[vid], ma_of[vid]
            key = ("both" if (mv and ma) else
                   "visual only" if mv else
                   "audio only" if ma else "unspecified")
            buckets[key][0] += int(p == 1)
            buckets[key][1] += 1
        if missed:
            print(f"    ! {missed} clips had no metadata match")
        breakdown = {k: v[0] / max(1, v[1]) for k, v in sorted(buckets.items())}
    except Exception as exc:  # noqa: BLE001
        # Report rather than swallow -- an empty breakdown used to look like a
        # legitimately empty result.
        print(f"    ! per-manipulation breakdown failed: {type(exc).__name__}: {exc}")
        breakdown = {}

    return dict(clip=binary_metrics(ys, ps, ss),
                localization=per_video_map(pred_by_vid, gt_by_vid),
                scores=[round(float(x), 5) for x in ss],
                labels=[int(x) for x in ys],
                per_manipulation=breakdown)


def train_run(name, cache, results, device, backbone="transformer", fusion="cross",
              modalities="va", epochs=15, bs=32, lr=1e-4, clip_sec=1.0, roi=96,
              loss_w=(1.0, 0.5, 0.25), workers=2, seed=42, save_ckpt=None,
              pretrained=True, skip_existing=False,
              visual_encoder="mobilenetv3", lipread_ckpt=None, rgb_norm=False, accum=1):
    # Colab drops runtimes without warning; resuming beats restarting.
    if _ONLY and name not in _ONLY:
        print(f"[{name}] not in --only list -- skipping")
        return None, results.data.get(f"run::{name}")
    if skip_existing and name not in _RETRAIN and f"run::{name}" in results.data:
        prev = results.data[f"run::{name}"]
        print(f"[{name}] already in results.json "
              f"(F1={prev['test']['clip']['f1']:.4f}) -- skipping")
        return None, prev

    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()

    tr = PackedLAVDF(cache, "train", clip_sec, roi, augment=True, seed=seed)
    va = PackedLAVDF(cache, "val", clip_sec, roi)
    te = PackedLAVDF(cache, "test", clip_sec, roi)
    print(f"[{name}] clips train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    if len(va) == 0:
        raise RuntimeError("validation split is empty -- check the dev->val mapping")

    # A validation split with no manipulated clips makes F1 identically zero, so
    # the best-checkpoint selection degenerates silently. Fail loudly instead.
    for split_name, ds in (("train", tr), ("val", va), ("test", te)):
        n_fake = sum(1 for c in ds.clips
                     if ds.meta.iloc[c[0]]["label"] == 1)
        if n_fake == 0 or n_fake == len(ds):
            raise RuntimeError(
                f"{split_name} split is single-class ({n_fake}/{len(ds)} fake). "
                "Increase --n-videos when building the cache.")

    # Gradient accumulation: `accum` micro-batches of bs/accum clips per optimiser
    # step, so the effective batch stays `bs` on memory-limited accelerators.
    if bs % accum:
        raise ValueError(f"batch size {bs} not divisible by accum {accum}")
    kw = dict(batch_size=bs // accum, num_workers=workers, collate_fn=collate,
              pin_memory=(device.type == "cuda"), persistent_workers=workers > 0)
    dl_tr = DataLoader(tr, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(va, shuffle=False, **kw)
    dl_te = DataLoader(te, shuffle=False, **kw)

    model = VariantASTRA(backbone, fusion, modalities, pretrained=pretrained,
                         visual_encoder=visual_encoder,
                         lipread_ckpt=lipread_ckpt, rgb_norm=rgb_norm).to(device)
    loss_fn = MultiTask(*loss_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=2)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 2))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[2])
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_f1, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        n_micro = (len(dl_tr) // accum) * accum      # drop an incomplete final step
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(dl_tr):
            if i >= n_micro:
                break
            v, a = b["video"].to(device, non_blocking=True), b["audio"].to(device, non_blocking=True)
            y, yf, yi = (b["label"].to(device), b["frame_labels"].to(device),
                         b["interval"].to(device))
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(v, a)
                loss, parts = loss_fn(out, y, yf, yi)
            scaler.scale(loss / accum).backward()
            tot += loss.item()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                n += 1
        tot /= max(1, accum)
        sched.step()

        vm = evaluate(model, dl_va, device)
        f1 = vm["clip"]["f1"]
        history.append(dict(epoch=ep, train_loss=tot / max(1, n), val_f1=f1,
                            val_acc=vm["clip"]["accuracy"]))
        print(f"  ep{ep:02d} loss={tot/max(1,n):.4f} val_f1={f1:.4f} "
              f"val_acc={vm['clip']['accuracy']:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    test = evaluate(model, dl_te, device)

    rec = dict(name=name, backbone=backbone, fusion=fusion, modalities=modalities,
               pretrained=pretrained, seed=seed, device=str(device),
               lr=lr, batch_size=bs, accum=accum, torch_version=torch.__version__,
               visual_encoder=visual_encoder, rgb_norm=model.rgb_norm,
               lipread_pretrained=(getattr(model.visual_encoder,
                                           "loaded_pretrained", None)
                                   if "v" in modalities else None),
               test_scores=test.pop("scores", None),
               test_labels=test.pop("labels", None),
               epochs=epochs, clip_sec=clip_sec, roi=roi, loss_weights=list(loss_w),
               params=model.n_params(), n_train=len(tr), n_val=len(va), n_test=len(te),
               best_val_f1=best_f1, test=test, history=history,
               minutes=round((time.time() - t0) / 60, 2))
    results.add(f"run::{name}", rec)
    print(f"[{name}] test F1={test['clip']['f1']:.4f} "
          f"AP@0.5={test['localization']['AP@0.5']:.4f} "
          f"({rec['minutes']:.1f} min)")

    if save_ckpt:
        Path(save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "name": name,
                    "config": model.config}, save_ckpt)
    return model, rec


# ==========================================================================
# Results store
# ==========================================================================
class Results:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.load(open(self.path)) if self.path.exists() else {}

    def add(self, key, value):
        self.data[key] = value
        self.flush(new_keys=(key,))

    def flush(self, new_keys=()):
        """Merge into whatever is on disk rather than overwriting it.

        A long training job holds an in-memory snapshot taken at startup. If a
        second process writes results meanwhile (a cross-dataset re-run, say),
        a plain dump of the snapshot silently destroys them -- the file simply
        reverts, with no error and nothing in the log. That happened here: a
        regenerated zero-shot evaluation was overwritten by a multi-seed job
        that had started earlier.

        Re-reading before writing makes concurrent writers last-writer-wins per
        KEY instead of per FILE, so independent stages no longer erase each
        other. The write goes through a temporary file and os.replace so a
        crash mid-write cannot truncate the results.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged = {}
        if self.path.exists():
            try:
                merged = json.load(open(self.path))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  ! could not read {self.path} to merge ({exc}); "
                      "writing in-memory state only")
                merged = {}
        # Keys this process produced win; everything else on disk is preserved.
        for k in (new_keys or self.data.keys()):
            merged[k] = self.data[k]
        for k, v in self.data.items():
            merged.setdefault(k, v)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(merged, fh, indent=2, default=str)
        os.replace(tmp, self.path)
        self.data = merged


# ==========================================================================
# Stages
# ==========================================================================
def stage_stats(results, device, cache):
    """Parameter counts, FLOPs and latency -- the cheap, exactly-measurable table."""
    out = {}
    for bk in ("transformer", "bilstm"):
        m = VariantASTRA(bk, "cross", "va", pretrained=False).eval()
        comp = {n: sum(p.numel() for p in mod.parameters())
                for n, mod in [("visual_encoder", m.visual_encoder),
                               ("audio_encoder", m.audio_encoder),
                               ("visual_temporal", m.visual_temporal),
                               ("audio_temporal", m.audio_temporal),
                               ("cross_modal", m.cross),
                               ("heads", nn.ModuleList([m.cls_head, m.sync_head, m.loc_head]))]}
        v = torch.randn(1, 30, 3, 96, 96)
        a = torch.randn(1, 81, 13)

        # CPU latency, single thread (matches the paper's stated protocol)
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.no_grad():
            for _ in range(10):
                m(v, a)
            t0 = time.perf_counter()
            for _ in range(30):
                m(v, a)
            cpu_ms = (time.perf_counter() - t0) / 30 * 1000
        torch.set_num_threads(prev)

        gpu_ms = None
        if device.type == "cuda":
            mg = m.to(device); vg, ag = v.to(device), a.to(device)
            with torch.no_grad():
                for _ in range(10):
                    mg(vg, ag)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(50):
                    mg(vg, ag)
                torch.cuda.synchronize()
                gpu_ms = (time.perf_counter() - t0) / 50 * 1000
            m = mg.cpu()

        flops = None
        try:
            from torch.utils.flop_counter import FlopCounterMode
            fc = FlopCounterMode(display=False)
            with fc, torch.no_grad():
                m(v, a)
            flops = fc.get_total_flops()
        except Exception as exc:  # noqa: BLE001
            print("  (FLOP counter unavailable:", exc, ")")

        out[bk] = dict(total_params=m.n_params(), components=comp,
                       cpu_latency_ms=round(cpu_ms, 2),
                       gpu_latency_ms=round(gpu_ms, 3) if gpu_ms else None,
                       flops=flops,
                       fps_equivalent=round(1000.0 / cpu_ms, 1))
        print(f"  {bk}: {m.n_params()/1e6:.2f}M params, CPU {cpu_ms:.1f} ms, "
              f"GPU {gpu_ms if gpu_ms is None else round(gpu_ms,2)} ms, FLOPs {flops}")

    # Audio-encoder channel widths, measured -- the paper claims {128,256,256}
    m = VariantASTRA("transformer", "cross", "va", pretrained=False)
    widths = [mod.out_channels for mod in m.audio_encoder.conv if isinstance(mod, nn.Conv1d)]
    out["audio_encoder_channels_measured"] = widths
    print("  audio encoder conv widths:", widths)

    # Record the measurement platform: latency is meaningless without it.
    import platform
    import subprocess
    hw = {"os": platform.platform(), "torch": torch.__version__}
    try:
        if sys.platform == "darwin":
            hw["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       capture_output=True, text=True).stdout.strip()
        else:
            for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines():
                if line.startswith("Model name:"):
                    hw["cpu"] = line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
    out["hardware"] = hw

    ds = PackedLAVDF(cache, "test")
    s = ds[0]
    out["tensor_shapes"] = dict(video=list(s["video"].shape), audio=list(s["audio"].shape))
    print("  shapes:", out["tensor_shapes"])

    results.add("stats", out)
    return out


def stage_gradcam(model, cache, results, device, n=50):
    """Fraction of Grad-CAM mass falling inside the central lip region."""
    ds = PackedLAVDF(cache, "test")

    # eval() is REQUIRED here, not merely tidy. A freshly constructed module is
    # in train mode, in which (a) dropout is active, making Grad-CAM
    # nondeterministic, and (b) every BatchNorm layer updates its running
    # statistics on each forward pass. This function performs ~50 forward
    # passes, so in train mode it silently *mutates the model*, and any stage
    # that runs afterwards in the same process -- cross-dataset evaluation, for
    # instance -- then scores a different network than the one on disk.
    # Gradients still flow normally in eval mode, so nothing is lost.
    was_training = model.training
    model.eval()

    feats = model.visual_encoder.features
    store = {}

    def fwd_hook(_, __, o):
        store["act"] = o

    def bwd_hook(_, __, go):
        store["grad"] = go[0]

    h1 = feats.register_forward_hook(fwd_hook)
    h2 = feats.register_full_backward_hook(bwd_hook)

    fake_mass, real_mass = [], []
    # Stratify by label. Sampling uniformly over the dataset draws in
    # proportion to the class balance (~13% manipulated here), which yielded
    # only 6 manipulated clips out of 50 -- far too few to support any claim
    # about lip-region specificity. Draw n/2 from each class instead.
    # Stratify on the CLIP label, not the video label. A manipulated video
    # contains mostly genuine seconds, so sampling 25 fake *videos* yielded
    # only 5 fake *clips* -- the same failure the stratification was meant to
    # fix. Recompute exactly as PackedLAVDF._labels does: a clip is fake iff a
    # fake period overlaps its time span.
    labels = []
    for vrow, _cidx, start in ds.clips:
        raw = ds.meta.iloc[vrow]["fake_periods"]
        periods = json.loads(raw) if isinstance(raw, str) else (raw or [])
        end = start + ds.clip_sec
        labels.append(int(any(min(float(q[1]), end) > max(float(q[0]), start)
                              for q in periods)))
    labels = np.array(labels)
    fake_idx = np.flatnonzero(labels == 1)
    real_idx = np.flatnonzero(labels == 0)
    half = max(1, n // 2)
    take = lambda pool, k: (pool[np.linspace(0, len(pool) - 1,
                                             min(k, len(pool))).astype(int)]
                            if len(pool) else pool)
    idxs = np.concatenate([take(fake_idx, half), take(real_idx, half)])
    print(f"  gradcam sampling: {min(half,len(fake_idx))} manipulated, "
          f"{min(half,len(real_idx))} genuine (stratified)")
    for i in idxs:
        s = ds[int(i)]
        v = s["video"].unsqueeze(0).to(device)
        a = s["audio"].unsqueeze(0).to(device)
        model.zero_grad()
        out = model(v, a)
        out["logits"][0, 1].backward()
        act, grad = store.get("act"), store.get("grad")
        if act is None or grad is None:
            continue
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * act).sum(1))              # (B*T, h, w)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        H, W = cam.shape[-2:]
        # central lip box: the middle 50% horizontally, lower-middle vertically
        box = cam[:, int(0.25 * H):int(0.85 * H), int(0.25 * W):int(0.75 * W)]
        frac = (box.sum() / (cam.sum() + 1e-8)).item()
        (fake_mass if s["label"].item() == 1 else real_mass).append(frac)

    h1.remove(); h2.remove()
    if was_training:
        model.train()
    res = dict(checkpoint=_CKPT_INFO,
               n_fake=len(fake_mass), n_real=len(real_mass),
               mean_mass_fake=float(np.mean(fake_mass)) if fake_mass else None,
               mean_mass_real=float(np.mean(real_mass)) if real_mass else None,
               box="central 50% width, 25-85% height of the ROI")
    results.add("gradcam", res)
    print("  gradcam:", res)
    return res


def stage_crossdata(model, eval_cache, name, results, device, bs=32, workers=2):
    """Zero-shot evaluation of a LAV-DF-trained checkpoint on another dataset.

    No training happens here -- that is the point. Cross-dataset transfer is
    the standard test of whether a detector learned the manipulation or the
    dataset.

    Note for DeepfakeTIMIT specifically: it is identity face-swap with
    unmodified audio, so lip-sync is largely preserved and a sync-based
    detector is expected to underperform. Report it as an out-of-scope probe,
    not as a failure to be explained away.
    """
    ds = PackedLAVDF(eval_cache, "test")
    if len(ds) == 0:
        print(f"  ! {name}: no clips found in {eval_cache}")
        return None

    n_fake = sum(1 for c in ds.clips if ds.meta.iloc[c[0]]["label"] == 1)
    n_real = len(ds) - n_fake
    print(f"  {name}: {len(ds)} clips ({n_fake} fake, {n_real} real)")

    dl = DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=collate)
    m = evaluate(model, dl, device)

    rec = {"dataset": name, "n_clips": len(ds), "n_fake": n_fake, "n_real": n_real,
           "checkpoint": _CKPT_INFO,
           "clip": m["clip"], "localization": m["localization"],
           # which manipulation families transfer -- the point of the exercise
           "per_manipulation": m.get("per_manipulation", {}),
           "scores": m.get("scores"), "labels": m.get("labels")}
    if n_real == 0:
        # F1/AUC are undefined without negatives; recall is the honest metric.
        rec["metric_note"] = "fakes-only: report detection rate (recall)"
        print(f"    detection rate (recall) = {m['clip']['recall']:.4f}")
    else:
        print(f"    F1 = {m['clip']['f1']:.4f}   AUC = {m['clip']['auc']:.4f}")

    results.add(f"crossdata::{name}", rec)
    return rec


def stage_robust(model, cache, results, device, snrs=(30, 20, 10), bs=32, workers=2):
    """Audio-noise degradation. NOTE: H.264 CRF robustness needs a second
    prepare_subset pass over re-encoded video and is *not* covered here."""
    out = {}
    for snr in (None,) + tuple(snrs):
        ds = PackedLAVDF(cache, "test", noise_snr=snr)
        dl = DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=collate)
        m = evaluate(model, dl, device)
        key = "clean" if snr is None else f"snr_{snr}db"
        out[key] = m["clip"]["f1"]
        print(f"  {key}: F1={m['clip']['f1']:.4f}")
    results.add("robustness_audio_noise", out)
    return out


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", default=".", help="path containing the astra_df package")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--stages", default="stats,main")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip ImageNet weights (offline smoke tests)")
    ap.add_argument("--only", default=None,
                    help="comma-separated run names to execute; everything else "
                         "is skipped (e.g. --only transformer)")
    ap.add_argument("--retrain", default=None,
                    help="comma-separated run names to force-retrain even if "
                         "already in results.json (regenerates checkpoints)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for a single-seed run")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds for the 'multiseed' stage "
                         "(e.g. 42,43,44,45,46)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip any run already present in results.json -- makes "
                         "the whole pipeline resumable after a Colab reset")
    ap.add_argument("--lipread-ckpt", default=None,
                    help="path to LRW-lipreading-pretrained ResNet-18 weights "
                         "(LipForensics / Lipreading-ResNet layout). Required "
                         "by the 'lipread' stage; without it the encoder would "
                         "be random and the experiment meaningless.")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient-accumulation steps; the effective batch stays "
                         "--batch-size (micro-batch = batch-size/accum)")
    ap.add_argument("--rgb-norm", action="store_true",
                    help="feed MobileNetV3 RGB, ImageNet-normalised frames (the cache "
                         "stores BGR in [0,1]); recorded in every run and checkpoint")
    ap.add_argument("--crossdata-ckpt", action="append", default=[],
                    metavar="TAG=MODALITIES:PATH",
                    help="evaluate an ADDITIONAL checkpoint zero-shot on every "
                         "--eval-cache, e.g. "
                         "--crossdata-ckpt audio=a:checkpoints/audio_s42.pt . "
                         "This is the decisive control for the 0.690/0.007 "
                         "analysis: it measures what an audio-only model scores "
                         "on visually-manipulated clips it has never seen.")
    ap.add_argument("--eval-cache", action="append", default=[], metavar="NAME=PATH",
                    help="extra dataset for zero-shot eval, e.g. "
                         "--eval-cache dftimit=/content/timit_cache (repeatable)")
    args = ap.parse_args()

    global M, _ONLY, _RETRAIN
    M = _import_astra(args.repo)
    _ONLY = set(x.strip() for x in args.only.split(",")) if args.only else None
    _RETRAIN = set(x.strip() for x in args.retrain.split(",")) if args.retrain else set()

    # Apple Silicon (MPS) is a viable fallback when no CUDA GPU is available --
    # markedly faster than CPU, and the whole model is MPS-compatible.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: cuda", torch.cuda.get_device_name(0))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: mps (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Device: cpu -- training will be slow")

    results = Results(args.out)
    ckpt_dir = Path(args.out).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("Checkpoints ->", ckpt_dir)
    stages = ([s.strip() for s in args.stages.split(",")]
              if args.stages != "all"
              else ["stats", "main", "fusion", "modality", "lossweight",
                    "hparam", "robust", "gradcam", "crossdata", "multiseed",
                    "lipread", "figures"])

    common = dict(cache=args.cache, results=results, device=device,
                  epochs=args.epochs, bs=args.batch_size, workers=args.workers,
                  pretrained=not args.no_pretrained,
                  skip_existing=args.skip_existing,
                  seed=args.seed, rgb_norm=args.rgb_norm, accum=args.accum)
    main_model = None

    if "stats" in stages:
        print("\n=== STAGE: stats ===")
        stage_stats(results, device, args.cache)

    if "main" in stages:
        print("\n=== STAGE: main ===")
        main_model, rec_tf = train_run("transformer", save_ckpt=str(ckpt_dir / "transformer.pt"),
                                       backbone="transformer", **common)
        if rec_tf["test"].get("per_manipulation"):
            results.add("per_manipulation", rec_tf["test"]["per_manipulation"])
        train_run("bilstm", backbone="bilstm", save_ckpt=str(ckpt_dir / "bilstm.pt"), **common)

    if "multiseed" in stages:
        print("\n=== STAGE: multi-seed (full vs audio-only) ===")
        seeds = [int(x) for x in (args.seeds or "42,43,44,45,46").split(",")]
        print(f"  seeds: {seeds}")
        ms_common = {k: v for k, v in common.items() if k != "seed"}
        for sd in seeds:
            for tag, mods in (("full", "va"), ("audio", "a")):
                nm = f"{tag}_s{sd}"
                train_run(nm, backbone="transformer", modalities=mods,
                          seed=sd, save_ckpt=str(ckpt_dir / f"{nm}.pt"),
                          **ms_common)

    if "ablseeds" in stages:
        # Every ablation variant at the same seeds and on the same platform as
        # the full/audio-only multiseed runs, so all in-domain tables can report
        # mean +- sd instead of mixing single runs with seed sweeps.
        print("\n=== STAGE: ablation variants across seeds ===")
        seeds = [int(x) for x in (args.seeds or "42,43,44,45,46").split(",")]
        ms_common = {k: v for k, v in common.items() if k != "seed"}
        variants = (("visual", dict(backbone="transformer", modalities="v")),
                    ("concat", dict(backbone="transformer", fusion="concat")),
                    ("v2a", dict(backbone="transformer", fusion="v2a")),
                    ("a2v", dict(backbone="transformer", fusion="a2v")),
                    ("bilstm", dict(backbone="bilstm")))
        for sd in seeds:
            for tag, kw in variants:
                nm = f"{tag}_s{sd}"
                train_run(nm, seed=sd, save_ckpt=str(ckpt_dir / f"{nm}.pt"),
                          **kw, **ms_common)

    if "lipreadrandom" in stages:
        # Capacity control for the lipreading experiment: the same 3-D stem +
        # ResNet-18 visual encoder, trained from RANDOM initialisation. If it
        # matches the LRW-pretrained encoder, the gain is capacity/architecture;
        # if it stays low, the gain is the lipreading pretraining.
        print("\n=== STAGE: lipreading architecture, random initialisation (control) ===")
        seeds = [int(x) for x in (args.seeds or "42,43,44").split(",")]
        ms_common = {k: v for k, v in common.items() if k != "seed"}
        for sd in seeds:
            nm = f"lipread_rand_visual_s{sd}"
            train_run(nm, backbone="transformer", modalities="v", seed=sd,
                      visual_encoder="lipread", lipread_ckpt=None,
                      save_ckpt=str(ckpt_dir / f"{nm}.pt"), **ms_common)

    if "lipreadseeds" in stages:
        print("\n=== STAGE: lipreading-pretrained encoder across seeds ===")
        if not args.lipread_ckpt:
            print("  ! --lipread-ckpt not supplied; skipping (a random encoder "
                  "would answer nothing).")
        else:
            seeds = [int(x) for x in (args.seeds or "42,43,44").split(",")]
            ms_common = {k: v for k, v in common.items() if k != "seed"}
            for sd in seeds:
                for tag, mods in (("lipread_visual", "v"), ("lipread_full", "va")):
                    nm = f"{tag}_s{sd}"
                    train_run(nm, backbone="transformer", modalities=mods, seed=sd,
                              visual_encoder="lipread", lipread_ckpt=args.lipread_ckpt,
                              save_ckpt=str(ckpt_dir / f"{nm}.pt"), **ms_common)

    if "lipread" in stages:
        print("\n=== STAGE: lipreading-pretrained visual encoder ===")
        if not args.lipread_ckpt:
            print("  ! --lipread-ckpt not supplied. Download LRW-pretrained "
                  "ResNet-18 weights from the LipForensics repository first.\n"
                  "  ! Skipping: a randomly-initialised encoder would answer "
                  "nothing.")
        else:
            lr_common = dict(common)
            # (A) visual-only: does a better initialisation rescue the visual
            #     stream at all? This is the cheap, decisive half.
            train_run("lipread_visual", backbone="transformer", modalities="v",
                      visual_encoder="lipread", lipread_ckpt=args.lipread_ckpt,
                      save_ckpt=str(ckpt_dir / "lipread_visual.pt"), **lr_common)
            # (B) full multimodal, if compute allows.
            train_run("lipread_full", backbone="transformer", modalities="va",
                      visual_encoder="lipread", lipread_ckpt=args.lipread_ckpt,
                      save_ckpt=str(ckpt_dir / "lipread_full.pt"), **lr_common)

    if "fusion" in stages:
        print("\n=== STAGE: fusion ablation ===")
        for f in ("concat", "v2a", "a2v"):
            train_run(f"fusion_{f}", backbone="transformer", fusion=f, **common)

    if "modality" in stages:
        print("\n=== STAGE: modality ablation ===")
        train_run("visual_only", backbone="transformer", modalities="v", **common)
        train_run("audio_only", backbone="transformer", modalities="a", **common)

    if "lossweight" in stages:
        print("\n=== STAGE: loss-weight sweep ===")
        for w in [(1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.5, 0.25),
                  (1.0, 1.0, 0.5), (1.0, 0.25, 0.25)]:
            train_run(f"lw_{w[0]}_{w[1]}_{w[2]}", backbone="transformer",
                      loss_w=w, **common)

    if "hparam" in stages:
        print("\n=== STAGE: hyperparameter sweep ===")
        for cs in (0.5, 2.0):
            train_run(f"clip_{cs}s", backbone="transformer", clip_sec=cs, **common)
        for r in (64, 128):
            train_run(f"roi_{r}", backbone="transformer", roi=r, **common)

    needs_ckpt = any(x in stages for x in ("robust", "gradcam", "crossdata"))
    if main_model is None and needs_ckpt:
        ck = ckpt_dir / "transformer.pt"
        if ck.exists():
            main_model = VariantASTRA("transformer", "cross", "va",
                                      pretrained=False).to(device)
            main_model.load_state_dict(torch.load(ck, map_location=device)["model"])
            import hashlib
            h = hashlib.sha256(ck.read_bytes()).hexdigest()[:12]
            global _CKPT_INFO
            _CKPT_INFO = {"path": str(ck), "sha256_12": h,
                          "mtime": time.strftime("%Y-%m-%d %H:%M",
                                                 time.localtime(ck.stat().st_mtime))}
            print(f"  loaded checkpoint {ck.name} sha={h}")
        else:
            print("! no trained checkpoint -- skipping robust/gradcam")

    if "robust" in stages and main_model is not None:
        print("\n=== STAGE: robustness ===")
        stage_robust(main_model, args.cache, results, device,
                     bs=args.batch_size, workers=args.workers)

    if "gradcam" in stages and main_model is not None:
        print("\n=== STAGE: gradcam ===")
        stage_gradcam(main_model, args.cache, results, device)

    if "crossdata" in stages and main_model is not None:
        print("\n=== STAGE: cross-dataset (zero-shot) ===")
        if not args.eval_cache:
            print("  ! no --eval-cache given; nothing to evaluate")
        for spec in args.eval_cache:
            if "=" not in spec:
                print(f"  ! bad --eval-cache '{spec}', expected NAME=PATH")
                continue
            nm, pth = spec.split("=", 1)
            stage_crossdata(main_model, pth, nm, results, device,
                            bs=args.batch_size, workers=args.workers)

        # Additional checkpoints (e.g. the audio-only model) on the same caches.
        for spec in args.crossdata_ckpt:
            try:
                tag, rest = spec.split("=", 1)
                mods, ckpt_path = rest.split(":", 1)
            except ValueError:
                print(f"  ! bad --crossdata-ckpt '{spec}', "
                      "expected TAG=MODALITIES:PATH")
                continue
            if not Path(ckpt_path).exists():
                print(f"  ! checkpoint not found: {ckpt_path}")
                continue
            m2 = VariantASTRA("transformer", "cross", mods,
                              pretrained=False).to(device)
            m2.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
            print(f"  loaded {ckpt_path} as '{tag}' (modalities={mods})")
            for spec2 in args.eval_cache:
                if "=" not in spec2:
                    continue
                nm2, pth2 = spec2.split("=", 1)
                stage_crossdata(m2, pth2, f"{nm2}_{tag}", results, device,
                                bs=args.batch_size, workers=args.workers)

    if "figures" in stages:
        print("\n=== STAGE: figures ===")
        try:
            import make_figures
            make_figures.build(args.out, "results/figures")
        except Exception as exc:  # noqa: BLE001
            print("  figures failed:", exc)

    print(f"\nAll requested stages complete -> {args.out}")


if __name__ == "__main__":
    main()
