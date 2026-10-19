"""PyTorch ``Dataset`` for the LAV-DF (Localized Audio-Visual DeepFake) benchmark.

LAV-DF (Cai et al., 2022) ships with a top-level ``metadata.min.json`` that
lists every video and, for fake clips, a ``fake_periods`` array of
``[start_sec, end_sec]`` intervals. This loader expects the dataset to have
already been preprocessed by :func:`astra_df.data.preprocessing.process_video`,
which produces per-clip ``.npy`` files in 1-second windows.

For each preprocessed clip we emit:

    - ``video``  : (T=30, 96, 96, 3) float32 normalised to [0, 1]
    - ``audio``  : (T_a, 13) float32 (mean-variance normalised per clip)
    - ``label``  : 0 = real, 1 = fake (clip overlaps any fake interval)
    - ``frame_labels`` : (T,) float32, per-frame fake probability target
    - ``interval`` : (2,) float32, normalised (start, end) of fake region in [0, 1]
                     relative to the clip; (-1, -1) when the clip is real.
    - ``video_id``, ``clip_index`` : provenance for evaluation/localization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LAVDFDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        lavdf_root: str | Path | None = None,
        split: str = "train",
        transform: Any | None = None,
        normalize_audio: bool = True,
    ) -> None:
        super().__init__()
        self.metadata = pd.read_csv(metadata_csv)
        if "split" in self.metadata.columns:
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.normalize_audio = normalize_audio
        self.video_meta = self._load_lavdf_metadata(lavdf_root)

    # ---------- LAV-DF JSON ----------
    @staticmethod
    def _load_lavdf_metadata(root: str | Path | None) -> dict[str, dict]:
        """Read the official LAV-DF ``metadata.min.json``.

        Returns a mapping from video id (filename stem) -> entry. Missing files
        are tolerated so the preprocessing CSV can be used standalone for
        smoke tests / partial datasets.
        """
        if root is None:
            return {}
        root = Path(root)
        for candidate in (root / "metadata.min.json", root / "metadata.json"):
            if candidate.exists():
                with open(candidate) as f:
                    entries = json.load(f)
                return {Path(e.get("file", "")).stem: e for e in entries}
        return {}

    # ---------- Labels ----------
    def _label_for_clip(self, video_id: str, start: float, end: float):
        """Return (clip_label, frame_labels, normalised_interval).

        clip_label = 1 if the clip overlaps any fake interval, else 0.
        frame_labels has length target_fps; entry t is 1 iff that frame's
        timestamp falls inside a fake interval.
        normalised_interval = (s_norm, e_norm) of the fake overlap within
        the clip, both in [0, 1]; (-1, -1) for real clips.
        """
        T = 30  # target_fps; matches preprocessing contract
        frame_t = np.linspace(start, end, T, endpoint=False)
        labels = np.zeros(T, dtype=np.float32)
        clip_label = 0
        s_norm, e_norm = -1.0, -1.0

        meta = self.video_meta.get(video_id)
        if meta is None:
            return clip_label, labels, np.array([s_norm, e_norm], dtype=np.float32)

        periods = meta.get("fake_periods") or meta.get("audio_fake_periods") or meta.get("visual_fake_periods") or []
        if not periods and bool(meta.get("modify_type")) is False:
            return clip_label, labels, np.array([s_norm, e_norm], dtype=np.float32)

        clip_duration = end - start
        overlap_lo, overlap_hi = None, None
        for p in periods:
            ps, pe = float(p[0]), float(p[1])
            lo, hi = max(ps, start), min(pe, end)
            if hi > lo:
                clip_label = 1
                mask = (frame_t >= ps) & (frame_t < pe)
                labels[mask] = 1.0
                if overlap_lo is None or lo < overlap_lo:
                    overlap_lo = lo
                if overlap_hi is None or hi > overlap_hi:
                    overlap_hi = hi
        if overlap_lo is not None:
            s_norm = float((overlap_lo - start) / clip_duration)
            e_norm = float((overlap_hi - start) / clip_duration)
        return clip_label, labels, np.array([s_norm, e_norm], dtype=np.float32)

    # ---------- Audio normalization ----------
    @staticmethod
    def _normalize_mfcc(mfcc: np.ndarray) -> np.ndarray:
        m = mfcc.mean(axis=0, keepdims=True)
        s = mfcc.std(axis=0, keepdims=True) + 1e-6
        return (mfcc - m) / s

    # ---------- PyTorch hooks ----------
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        video = np.load(row["lip_path"])  # (T, H, W, 3) uint8
        audio = np.load(row["audio_path"])  # (T_a, n_mfcc) float32

        video = video.astype(np.float32) / 255.0
        if self.normalize_audio:
            audio = self._normalize_mfcc(audio.astype(np.float32))

        clip_label, frame_labels, interval = self._label_for_clip(
            row["video_id"], float(row["start_sec"]), float(row["end_sec"]),
        )

        # If metadata wasn't available we still need a label — fall back to
        # an optional explicit "label" column in the CSV (used by smoke tests).
        if "label" in row.index and not pd.isna(row["label"]):
            clip_label = int(row["label"])

        sample = {
            "video": torch.from_numpy(video).permute(0, 3, 1, 2).contiguous(),  # (T, 3, H, W)
            "audio": torch.from_numpy(audio).contiguous(),                       # (T_a, n_mfcc)
            "label": torch.tensor(clip_label, dtype=torch.long),
            "frame_labels": torch.from_numpy(frame_labels),
            "interval": torch.from_numpy(interval),
            "video_id": row["video_id"],
            "clip_index": int(row["clip_index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that keeps string fields as plain lists."""
    out: dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            try:
                out[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                # variable-length audio — pad to longest
                max_t = max(v.shape[0] for v in vals)
                padded = []
                for v in vals:
                    if v.shape[0] < max_t:
                        pad = torch.zeros(max_t - v.shape[0], *v.shape[1:], dtype=v.dtype)
                        v = torch.cat([v, pad], dim=0)
                    padded.append(v)
                out[k] = torch.stack(padded, dim=0)
        else:
            out[k] = vals
    return out
