"""Mouth-ROI + MFCC preprocessing pipeline.

Refactored from the team's earlier Colab notebook
``av_feature_extraction_colab.ipynb`` into a reusable module. The output
contract is preserved exactly: per 1-second window we save a
``(T=30, 96, 96, 3)`` uint8 lip clip and a ``(~Ta, 13)`` float32 MFCC tensor,
keeping downstream code unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# Heavy / optional deps are imported lazily inside functions so that simply
# importing this module (e.g. from a unit test) does not require ffmpeg,
# OpenCV, MediaPipe, or librosa to be installed.


from astra_df.utils.logging import get_logger

log = get_logger("preprocessing")


# MediaPipe outer + inner lip landmarks (FaceMesh refined).
MOUTH_LANDMARKS = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
)


@dataclass
class PreprocConfig:
    target_fps: int = 30
    audio_sr: int = 16000
    clip_duration_sec: float = 1.0
    mouth_size: int = 96
    n_mfcc: int = 13
    hop_length: int = 200
    n_fft: int = 512
    bbox_padding: float = 0.35
    bbox_smoothing: float = 0.75


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _extract_audio_with_ffmpeg(video_path: str, wav_path: str, sr: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sr), "-vn", wav_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _video_info(video_path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = n / fps if fps > 0 else 0.0
    return {"fps": fps, "frame_count": n, "duration": duration, "width": w, "height": h}


def _resize_with_padding(img: np.ndarray, size: int) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_off, y_off = (size - new_w) // 2, (size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _mouth_bbox(landmarks, img_w: int, img_h: int, pad: float) -> tuple[int, int, int, int]:
    xs = [int(landmarks[i].x * img_w) for i in MOUTH_LANDMARKS]
    ys = [int(landmarks[i].y * img_h) for i in MOUTH_LANDMARKS]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    )


def _smooth_box(prev, curr, alpha: float):
    if prev is None:
        return curr
    return tuple(int(alpha * p + (1 - alpha) * c) for p, c in zip(prev, curr))


# ---------------------------------------------------------------------------
# main steps
# ---------------------------------------------------------------------------
def extract_mouth_frames(video_path: str, cfg: PreprocConfig) -> tuple[list[np.ndarray], list[float], dict]:
    """Sample frames at ``target_fps`` and crop the mouth ROI from each.

    Returns
    -------
    frames : list of (mouth_size, mouth_size, 3) uint8 arrays
    timestamps : list of seconds since video start
    stats : per-video detection statistics (used for logging fault rate)
    """
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(video_path)
    info = _video_info(video_path)
    duration = info["duration"]
    target_times = np.arange(0, duration, 1.0 / cfg.target_fps)

    mp_face_mesh = mp.solutions.face_mesh
    frames: list[np.ndarray] = []
    times: list[float] = []
    prev_box = None
    n_missed = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for t in target_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                box = _smooth_box(prev_box, _mouth_bbox(lms, w, h, cfg.bbox_padding), cfg.bbox_smoothing)
                prev_box = box
                x1, y1, x2, y2 = box
                crop = _resize_with_padding(frame[y1:y2, x1:x2], cfg.mouth_size)
            else:
                crop = np.zeros((cfg.mouth_size, cfg.mouth_size, 3), dtype=np.uint8)
                n_missed += 1
            frames.append(crop)
            times.append(float(t))
    cap.release()

    stats = {
        "n_frames": len(frames),
        "n_face_missed": n_missed,
        "miss_rate": (n_missed / max(1, len(frames))),
        "duration": duration,
    }
    return frames, times, stats


def compute_mfcc(audio: np.ndarray, sr: int, cfg: PreprocConfig) -> np.ndarray:
    import librosa
    if len(audio) < cfg.n_fft:
        audio = np.pad(audio, (0, cfg.n_fft - len(audio)), mode="constant")
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )
    return mfcc.T.astype(np.float32)  # (T_a, n_mfcc)


def chunk_aligned(frames, times, audio, sr, cfg: PreprocConfig) -> Iterator[dict]:
    if not times:
        return
    total = times[-1] + (1.0 / cfg.target_fps)
    n_clips = int(total // cfg.clip_duration_sec)
    for i in range(n_clips):
        s, e = i * cfg.clip_duration_sec, (i + 1) * cfg.clip_duration_sec
        idxs = [j for j, t in enumerate(times) if s <= t < e]
        if not idxs:
            continue
        clip_frames = np.stack([frames[j] for j in idxs], axis=0)
        # pad/truncate to exactly target_fps frames so downstream shape is rigid
        clip_frames = _force_temporal_length(clip_frames, cfg.target_fps)
        audio_seg = audio[int(s * sr): int(e * sr)]
        clip_mfcc = compute_mfcc(audio_seg, sr, cfg)
        yield {
            "clip_index": i,
            "start_sec": float(s),
            "end_sec": float(e),
            "video_frames": clip_frames,
            "audio_mfcc": clip_mfcc,
        }


def _force_temporal_length(clip_frames: np.ndarray, target: int) -> np.ndarray:
    t = clip_frames.shape[0]
    if t == target:
        return clip_frames
    if t > target:
        return clip_frames[:target]
    # pad by repeating the last frame
    pad = np.repeat(clip_frames[-1:], target - t, axis=0)
    return np.concatenate([clip_frames, pad], axis=0)


def process_video(video_path: str, output_root: str, cfg: PreprocConfig | None = None) -> dict:
    """Run full preprocessing on one video and write npy + metadata."""
    import librosa
    cfg = cfg or PreprocConfig()
    base = Path(video_path).stem
    out = Path(output_root) / base
    lips_dir = out / "lips"
    audio_dir = out / "audio"
    tmp_dir = out / "tmp"
    for d in (out, lips_dir, audio_dir, tmp_dir):
        _ensure_dir(d)

    wav_path = str(tmp_dir / f"{base}.wav")
    frames, times, stats = extract_mouth_frames(video_path, cfg)
    _extract_audio_with_ffmpeg(video_path, wav_path, cfg.audio_sr)
    audio, sr = librosa.load(wav_path, sr=cfg.audio_sr, mono=True)

    rows = []
    for clip in chunk_aligned(frames, times, audio, sr, cfg):
        idx = clip["clip_index"]
        lip_path = lips_dir / f"clip_{idx:04d}.npy"
        mfcc_path = audio_dir / f"clip_{idx:04d}.npy"
        np.save(lip_path, clip["video_frames"])
        np.save(mfcc_path, clip["audio_mfcc"])
        rows.append({
            "video_id": base,
            "clip_index": idx,
            "start_sec": clip["start_sec"],
            "end_sec": clip["end_sec"],
            "lip_path": str(lip_path),
            "audio_path": str(mfcc_path),
            "video_shape": str(tuple(clip["video_frames"].shape)),
            "audio_shape": str(tuple(clip["audio_mfcc"].shape)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)

    summary = {"video": str(video_path), "n_clips": len(rows), **asdict(cfg), **stats}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Processed %s -> %d clips (miss rate %.2f%%)",
             base, len(rows), 100.0 * stats["miss_rate"])
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    p = argparse.ArgumentParser(description="Preprocess a single video into mouth-ROI clips and MFCC features.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="processed/")
    args = p.parse_args()
    process_video(args.input, args.output)


if __name__ == "__main__":
    _cli()
