"""End-to-end inference on a single video file.

Pipeline:

    video.mp4
       │
       ▼
    preprocess  → (N_clips × (T=30, 96, 96, 3))
                  (N_clips × (T_a, 13))
       │
       ▼
    ASTRA-DF forward (batch over clips)
       │
       ▼
    per-clip: real/fake prob, frame_sync (T,), regressed interval (s, e)
       │
       ▼
    stitch frame_sync over the full video → sync_timeline
       │
       ▼
    threshold + NMS + merge → list of fake intervals
       │
       ▼
    JSON / figure / heatmap outputs

Usage:
    python -m astra_df.inference.predict --video path/to/video.mp4 \
        --ckpt runs/transformer/best.pt --config astra_df/configs/transformer.yaml \
        --out timeline.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from astra_df.data.preprocessing import (
    PreprocConfig, _extract_audio_with_ffmpeg, chunk_aligned, extract_mouth_frames,
)
from astra_df.explainability.timeline import merge_close, nms_intervals, timeline_from_frames
from astra_df.models import build_model
from astra_df.training.train import load_config
from astra_df.utils import get_logger


log = get_logger("predict")


def _preprocess_video(video_path: str, cfg: PreprocConfig) -> list[dict]:
    """Returns a list of clip dicts ready for batched inference."""
    import librosa
    frames, times, _ = extract_mouth_frames(video_path, cfg)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    _extract_audio_with_ffmpeg(video_path, wav_path, cfg.audio_sr)
    audio, sr = librosa.load(wav_path, sr=cfg.audio_sr, mono=True)
    clips = list(chunk_aligned(frames, times, audio, sr, cfg))
    return clips


def _normalize_mfcc(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + 1e-6)


_VALID_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _validate_inputs(video_path: str, config_path: str, ckpt_path: str | None,
                     threshold: float) -> None:
    """Friendly validation for the CLI and Streamlit demo.

    Raises ``FileNotFoundError`` / ``ValueError`` with human-readable messages
    so users see "video.mp4 not found" instead of an opaque traceback from
    deep inside ffmpeg or torch.
    """
    if not video_path:
        raise ValueError("--video is required")
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not vp.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")
    if vp.suffix.lower() not in _VALID_VIDEO_EXTS:
        raise ValueError(
            f"Unsupported video extension '{vp.suffix}'. "
            f"Supported: {sorted(_VALID_VIDEO_EXTS)}"
        )
    if vp.stat().st_size == 0:
        raise ValueError(f"Video file is empty: {video_path}")

    cp = Path(config_path)
    if not cp.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if ckpt_path:
        kp = Path(ckpt_path)
        if not kp.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Train one with: bash scripts/train_transformer.sh, "
                f"or omit --ckpt to use a randomly initialised model."
            )

    if not 0.0 < threshold < 1.0:
        raise ValueError(f"--threshold must be in (0, 1); got {threshold}")


@torch.no_grad()
def predict(video_path: str, config_path: str, ckpt_path: str | None,
            threshold: float = 0.5, device: str | None = None) -> dict:
    _validate_inputs(video_path, config_path, ckpt_path, threshold)

    cfg = load_config(config_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg).to(device).eval()
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        log.info("Loaded checkpoint %s", ckpt_path)
    else:
        log.warning("No checkpoint provided — using randomly initialized model. "
                    "Verdict will not be meaningful.")

    pc = PreprocConfig(
        target_fps=cfg["data"]["target_fps"],
        audio_sr=cfg["data"]["audio_sr"],
        clip_duration_sec=cfg["data"]["clip_duration_sec"],
        mouth_size=cfg["data"]["mouth_size"],
        n_mfcc=cfg["data"]["n_mfcc"],
        hop_length=cfg["data"]["hop_length"],
        n_fft=cfg["data"]["n_fft"],
    )
    clips = _preprocess_video(video_path, pc)
    if not clips:
        return {"verdict": "unknown", "reason": "no clips produced", "intervals": []}

    # Batch them.
    videos = np.stack([c["video_frames"] for c in clips], axis=0).astype(np.float32) / 255.0
    audios = np.stack([_normalize_mfcc(c["audio_mfcc"].astype(np.float32))
                       for c in clips], axis=0)
    video_t = torch.from_numpy(videos).permute(0, 1, 4, 2, 3).contiguous().to(device)
    audio_t = torch.from_numpy(audios).contiguous().to(device)

    out = model(video_t, audio_t)
    probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
    frame_sync = torch.sigmoid(out.frame_sync).cpu().numpy()  # (N, T)
    intervals = out.interval.cpu().numpy()                     # (N, 2)
    attn = out.attn_v2a.cpu().numpy()                          # (N, T_v, T_a)

    # Reconstruct the global per-frame timeline.
    global_scores, global_times = [], []
    T = frame_sync.shape[1]
    for i, c in enumerate(clips):
        s, e = c["start_sec"], c["end_sec"]
        ts = np.linspace(s, e, T, endpoint=False)
        global_scores.extend(frame_sync[i].tolist())
        global_times.extend(ts.tolist())

    # Frame-level → contiguous intervals.
    raw_intervals = timeline_from_frames(global_scores, global_times, threshold=threshold)

    # Also bring in the regressed intervals from clips that the model thinks are fake.
    regressed = []
    for i, c in enumerate(clips):
        if probs[i] > threshold:
            s, e = c["start_sec"], c["end_sec"]
            iv = intervals[i]
            regressed.append({
                "start": float(s + iv[0] * (e - s)),
                "end":   float(s + iv[1] * (e - s)),
                "score": float(probs[i]),
            })
    regressed_iv = nms_intervals([
        type(raw_intervals[0])(start=r["start"], end=r["end"], score=r["score"])  # type: ignore[index]
        for r in regressed
    ], iou_thresh=0.3) if raw_intervals else []

    final = merge_close(raw_intervals + regressed_iv, gap_sec=0.25)
    verdict_score = float(np.mean(probs))
    verdict = "fake" if verdict_score > threshold else "real"

    result = {
        "video": str(video_path),
        "verdict": verdict,
        "confidence": verdict_score,
        "clip_probs": probs.tolist(),
        "intervals": [iv.to_dict() for iv in final],
        "frame_sync": {"times": global_times, "scores": global_scores},
        "n_clips": len(clips),
    }
    # Attach a small attention sample for visualisations (not the whole tensor).
    result["attention_sample"] = attn[0].tolist() if len(attn) > 0 else None
    return result


def _cli() -> None:
    import sys
    p = argparse.ArgumentParser(
        description="Run ASTRA-DF inference on a single video file.",
    )
    p.add_argument("--video", required=True, help="Path to .mp4/.mov/.avi/.mkv/.webm input video.")
    p.add_argument("--config", default="astra_df/configs/transformer.yaml",
                   help="YAML config (default: transformer variant).")
    p.add_argument("--ckpt", default=None,
                   help="Path to a trained .pt checkpoint. Omit to use a randomly initialised model.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold in (0, 1). Default 0.5.")
    p.add_argument("--out", default="timeline.json",
                   help="Output JSON path. Default ./timeline.json.")
    args = p.parse_args()
    try:
        res = predict(args.video, args.config, args.ckpt, args.threshold)
    except FileNotFoundError as e:
        log.error("%s", e); sys.exit(2)
    except ValueError as e:
        log.error("Bad input: %s", e); sys.exit(2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log.info("Wrote %s — verdict: %s (%.3f)", args.out, res["verdict"], res["confidence"])


if __name__ == "__main__":
    _cli()
