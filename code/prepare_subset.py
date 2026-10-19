"""
ASTRA-DF :: Stage 1 -- local subset preparation (CPU only, run on your Mac)
===========================================================================

Builds a compact, upload-friendly cache from your local LAV-DF copy so that
Stage 2 (training, on a Colab T4) needs no video decoding at all.

Why this exists
---------------
`astra_df/data/preprocessing.py` calls ``cap.set(cv2.CAP_PROP_POS_MSEC, ...)``
before *every* frame read. Measured on train/000469.mp4: 1.00 s with seeking
vs 0.02 s with sequential decode -- a 48x penalty. This script decodes
sequentially and runs MediaPipe FaceMesh once per unique source frame.

Design
------
MediaPipe landmarking is by far the dominant cost, so we pay it exactly once
per video and cache a *raw, per-video* representation that every downstream
experiment can be derived from:

    * mouth crops at 128x128 (the largest ROI the paper ablates), JPEG-packed
      -> any ROI size (64/96/128) is a cheap resize at load time
    * the full 16 kHz mono waveform as int16
      -> any clip duration (0.5/1.0/2.0 s) and any audio-noise SNR can be
         derived at load time; MFCCs are computed on the fly
    * the official fake_periods, split and modify flags

That single cache therefore supports the clip-duration ablation, the ROI-size
ablation, and the audio-noise robustness sweep without re-decoding anything.

Output layout
-------------
    <out>/
        frames.bin          concatenated JPEG bytes, all videos
        index.npz           per-video frame offsets, waveform offsets, labels
        audio.bin           concatenated int16 PCM, all videos
        videos.csv          one row per video (id, split, n_frames, duration, ...)
        manifest.json       provenance + config

Usage
-----
    python prepare_subset.py \
        --root /path/to/LAV-DF \
        --out  "./astra_cache" \
        --n-videos 900 --workers 8

Then tar it up and drop it in Google Drive:

    tar -czf astra_cache.tgz astra_cache
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Preprocessing contract. CROP_SIZE is deliberately the *largest* ROI the
# paper ablates; smaller ROIs are produced by resizing at load time.
# --------------------------------------------------------------------------
CROP_SIZE = 128
TARGET_FPS = 30
AUDIO_SR = 16000
JPEG_QUALITY = 92
BBOX_PAD = 0.35
BBOX_SMOOTH = 0.75

# MediaPipe FaceMesh outer + inner lip contour (22 landmarks), matching
# astra_df/data/preprocessing.py exactly.
MOUTH_LANDMARKS = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
)


# ==========================================================================
# geometry helpers
# ==========================================================================
def resize_pad(img: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Aspect-preserving resize onto a black square canvas."""
    import cv2

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), np.uint8)
    s = min(size / w, size / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    canvas = np.zeros((size, size, 3), np.uint8)
    xo, yo = (size - nw) // 2, (size - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = cv2.resize(img, (nw, nh))
    return canvas


def mouth_bbox(lms, w: int, h: int, pad: float = BBOX_PAD):
    xs = [int(lms[i].x * w) for i in MOUTH_LANDMARKS]
    ys = [int(lms[i].y * h) for i in MOUTH_LANDMARKS]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def smooth_box(prev, curr, alpha: float = BBOX_SMOOTH):
    if prev is None:
        return curr
    return tuple(int(alpha * p + (1 - alpha) * c) for p, c in zip(prev, curr))


# ==========================================================================
# per-video worker
# ==========================================================================
def process_video(video_path: str):
    """Decode one video, landmark it, return JPEG-encoded 30 FPS mouth crops
    plus the 16 kHz waveform.

    Returns (jpegs, wav_int16, stats) or None on failure.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    raw = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw.append(frame)
    cap.release()
    if not raw:
        raise RuntimeError(
            "OpenCV opened the file but decoded 0 frames -- the build may "
            "lack H.264 support")

    duration = len(raw) / src_fps
    # Resample the source frame sequence onto a uniform 30 FPS grid.
    target_t = np.arange(0.0, duration, 1.0 / TARGET_FPS)
    src_idx = np.clip((target_t * src_fps).astype(int), 0, len(raw) - 1)

    # Landmark each *unique* source frame once -- with 25 FPS sources mapped to
    # a 30 FPS grid this saves ~17% of all MediaPipe calls.
    crops: dict[int, np.ndarray] = {}
    prev_box = None
    missed = 0
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh:
        for i in sorted(set(src_idx.tolist())):
            frame = raw[i]
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                h, w = frame.shape[:2]
                box = smooth_box(
                    prev_box,
                    mouth_bbox(res.multi_face_landmarks[0].landmark, w, h),
                )
                prev_box = box
                x1, y1, x2, y2 = box
                crops[i] = resize_pad(frame[y1:y2, x1:x2])
            else:
                crops[i] = np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8)
                missed += 1

    enc = [cv2.imencode(".jpg", crops[i],
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])[1].tobytes()
           for i in src_idx]

    wav = extract_wav(video_path)

    stats = {
        "n_frames": len(enc),
        "n_landmarked": len(crops),
        "miss_rate": missed / max(1, len(crops)),
        "duration": duration,
        "src_fps": src_fps,
    }
    return enc, wav, stats


def extract_wav(video_path: str):
    """16 kHz mono int16 PCM via ffmpeg.

    Raises with a specific message rather than returning None, so a missing
    ffmpeg binary is never mistaken for a corrupt video.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = tf.name
    try:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-ac", "1",
                 "-ar", str(AUDIO_SR), "-vn", "-f", "wav", tmp],
                capture_output=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg binary not found on PATH -- install it "
                "(macOS: brew install ffmpeg)") from None
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            raise RuntimeError("ffmpeg failed: " + " | ".join(tail))

        import wave

        with wave.open(tmp, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if pcm.size == 0:
            raise RuntimeError("ffmpeg produced an empty audio track")
        return pcm.copy()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _worker(args):
    path, vid = args
    try:
        enc, wav, stats = process_video(path)
        return vid, enc, wav, stats
    except Exception as exc:  # noqa: BLE001
        return vid, None, None, {"error": f"{type(exc).__name__}: {exc}"}


# ==========================================================================
# stratified sampling
# ==========================================================================
def select_videos(entries, n_videos, seed=42):
    """Pick a class- and manipulation-balanced subset, split-proportional.

    LAV-DF splits are train / dev / test; `dev` is mapped to `val` downstream
    (train.py asks for split=="val", which is the bug that silently empties
    the validation loader).
    """
    rng = random.Random(seed)

    def kind(e):
        if e["n_fakes"] == 0:
            return "real"
        mv, ma = bool(e["modify_video"]), bool(e["modify_audio"])
        if mv and ma:
            return "both"
        return "visual" if mv else "audio"

    # Proportions chosen so real ~= fake overall, with the three manipulation
    # families equally represented inside the fake half.
    quota_frac = {"real": 0.50, "visual": 0.1667, "audio": 0.1667, "both": 0.1666}
    split_frac = {"train": 0.70, "dev": 0.15, "test": 0.15}

    buckets = defaultdict(list)
    for e in entries:
        buckets[(e["split"], kind(e))].append(e)
    for v in buckets.values():
        rng.shuffle(v)

    chosen = []
    for sp, sf in split_frac.items():
        for kd, kf in quota_frac.items():
            want = int(round(n_videos * sf * kf))
            pool = buckets.get((sp, kd), [])
            if len(pool) < want:
                print(f"  ! only {len(pool)} available for {sp}/{kd} (wanted {want})")
            chosen.extend(pool[:want])
    rng.shuffle(chosen)
    return chosen


# ==========================================================================
# main
# ==========================================================================
def preflight(root: str) -> None:
    """Verify every external dependency before spending time on real work.

    Each of these has bitten us once already; failing here costs one second
    instead of one full pass over the dataset.
    """
    import shutil

    problems = []

    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg not on PATH            -> brew install ffmpeg")

    try:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            problems.append(
                f"mediapipe {mp.__version__} has no .solutions API "
                "-> pip install 'mediapipe==0.10.13' 'protobuf==4.25.3'")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"mediapipe import failed: {exc}")

    try:
        import cv2  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"cv2 import failed: {exc}")

    meta = Path(root) / "metadata.min.json"
    if not meta.exists():
        problems.append(f"{meta} not found -- is --root correct?")

    # Decode one real file end to end; catches codec issues immediately.
    if not problems:
        import cv2
        sample = next((Path(root) / "train").glob("*.mp4"), None)
        if sample is not None:
            cap = cv2.VideoCapture(str(sample))
            ok, _ = cap.read()
            cap.release()
            if not ok:
                problems.append(
                    f"OpenCV decoded 0 frames from {sample.name} "
                    "-> pip install 'opencv-python<5'")

    if problems:
        print("Preflight failed:\n")
        for p in problems:
            print("  *", p)
        sys.exit(1)
    print("Preflight OK (ffmpeg, mediapipe.solutions, cv2 decode, metadata)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="LAV-DF root (has metadata.min.json)")
    ap.add_argument("--out", required=True, help="output cache directory")
    ap.add_argument("--n-videos", type=int, default=900)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-duration", type=float, default=12.0,
                    help="skip videos longer than this (keeps the cache compact)")
    args = ap.parse_args()

    preflight(args.root)

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = root / "metadata.min.json"
    if not meta_path.exists():
        sys.exit(f"metadata.min.json not found under {root}")

    print(f"Reading {meta_path} ...")
    entries = json.load(open(meta_path))
    entries = [e for e in entries if e["duration"] <= args.max_duration]
    print(f"  {len(entries)} videos within {args.max_duration}s")

    selected = select_videos(entries, args.n_videos, args.seed)
    print(f"Selected {len(selected)} videos")
    print("  split:", Counter(e["split"] for e in selected))
    print("  fake :", Counter("real" if e["n_fakes"] == 0 else "fake" for e in selected))

    tasks = []
    for e in selected:
        p = root / e["file"]
        if p.exists():
            tasks.append((str(p), Path(e["file"]).stem))
    print(f"  {len(tasks)} files present on disk")

    by_id = {Path(e["file"]).stem: e for e in selected}

    frames_bin = open(out / "frames.bin", "wb")
    audio_bin = open(out / "audio.bin", "wb")

    frame_off = [0]          # running offset into frames.bin, per JPEG
    video_frame_ptr = [0]    # index into frame_off marking each video's start
    audio_off = [0]
    rows = []
    failures = []

    t0 = time.time()
    done = 0

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t[1] for t in tasks}
        for fut in as_completed(futures):
            vid, enc, wav, stats = fut.result()
            done += 1
            if enc is None:
                failures.append({"video_id": vid, **stats})
            else:
                for b in enc:
                    frames_bin.write(b)
                    frame_off.append(frame_off[-1] + len(b))
                video_frame_ptr.append(len(frame_off) - 1)

                audio_bin.write(wav.tobytes())
                audio_off.append(audio_off[-1] + wav.size)

                e = by_id[vid]
                rows.append({
                    "video_id": vid,
                    "split": "val" if e["split"] == "dev" else e["split"],
                    "n_frames": stats["n_frames"],
                    "duration": round(stats["duration"], 4),
                    "src_fps": stats["src_fps"],
                    "miss_rate": round(stats["miss_rate"], 4),
                    "n_samples": int(wav.size),
                    "label": 0 if e["n_fakes"] == 0 else 1,
                    "modify_video": int(bool(e["modify_video"])),
                    "modify_audio": int(bool(e["modify_audio"])),
                    "fake_periods": json.dumps(e["fake_periods"]),
                })

            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = done / max(1e-9, el)
                eta = (len(tasks) - done) / max(1e-9, rate)
                mb = frame_off[-1] / 1e6
                print(f"  [{done}/{len(tasks)}] {rate:.2f} vid/s  "
                      f"frames={mb:.0f} MB  eta={eta/60:.1f} min", flush=True)

    frames_bin.close()
    audio_bin.close()

    np.savez_compressed(
        out / "index.npz",
        frame_offsets=np.array(frame_off, dtype=np.int64),
        video_frame_ptr=np.array(video_frame_ptr, dtype=np.int64),
        audio_offsets=np.array(audio_off, dtype=np.int64),
    )

    import csv

    if rows:
        with open(out / "videos.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lavdf_root": str(root),
        "n_requested": args.n_videos,
        "n_written": len(rows),
        "n_failed": len(failures),
        "crop_size": CROP_SIZE,
        "target_fps": TARGET_FPS,
        "audio_sr": AUDIO_SR,
        "jpeg_quality": JPEG_QUALITY,
        "bbox_padding": BBOX_PAD,
        "bbox_smoothing": BBOX_SMOOTH,
        "mouth_landmarks": list(MOUTH_LANDMARKS),
        "seed": args.seed,
        "max_duration": args.max_duration,
        "elapsed_sec": round(time.time() - t0, 1),
        "mean_miss_rate": round(float(np.mean([r["miss_rate"] for r in rows])), 5) if rows else None,
        "failures": failures[:50],
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)

    size_mb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  videos written : {len(rows)}  (failed {len(failures)})")
    print(f"  mean miss rate : {manifest['mean_miss_rate']}")
    print(f"  cache size     : {size_mb:.0f} MB  -> {out}")
    print(f"\nNext:  tar -czf astra_cache.tgz -C {out.parent} {out.name}")


if __name__ == "__main__":
    main()
