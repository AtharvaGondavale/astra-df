"""
ASTRA-DF :: FakeAVCeleb cache builder (cross-dataset evaluation)
================================================================

Builds a cache in the same format as prepare_subset.py, so run_experiments.py
evaluates a LAV-DF-trained checkpoint on it with no code changes. Everything is
written with split="test" -- this set is used zero-shot.

Why this dataset is the decisive experiment
-------------------------------------------
FakeAVCeleb is partitioned by *which modality was manipulated*:

    A  RealVideo-RealAudio   real                      -> label 0
    B  RealVideo-FakeAudio   audio synthesised (SV2TTS) -> label 1, audio-only
    C  FakeVideo-RealAudio   video swapped/lip-synced   -> label 1, visual-only
    D  FakeVideo-FakeAudio   both                       -> label 1, both

Category C is the diagnostic that matters. Its audio track is genuine, so a
detector relying on audio artefacts *cannot* succeed on it, while a detector
using lip--speech synchronisation should -- several C videos are generated with
Wav2Lip, which re-renders the mouth against unchanged speech and therefore
breaks synchrony directly.

Comparing detection rate on B against C measures which modality the model
actually uses, on data entirely unseen during training. We map the categories
onto the existing modify_video / modify_audio flags so the per-manipulation
breakdown in run_experiments.py reports exactly this split.

Usage
-----
    python prepare_fakeavceleb.py \
        --root /path/to/FakeAVCeleb_v1.2 \
        --out ./favc_cache --per-category 500 --workers 8

Default sampling takes all of A and B (500 each, the full set) and an equal
stratified draw from C and D, giving a balanced ~2000-video evaluation set
rather than the natural 1:40 real-to-fake imbalance.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_subset import (  # noqa: E402
    CROP_SIZE, TARGET_FPS, AUDIO_SR, process_video,
)

# category -> (label, modify_video, modify_audio)
CATEGORIES = {
    "RealVideo-RealAudio": (0, 0, 0),
    "RealVideo-FakeAudio": (1, 0, 1),
    "FakeVideo-RealAudio": (1, 1, 0),
    "FakeVideo-FakeAudio": (1, 1, 1),
}

# short, unique per-category tag used to build video ids
CAT_TAG = {
    "RealVideo-RealAudio": "RVRA",
    "RealVideo-FakeAudio": "RVFA",
    "FakeVideo-RealAudio": "FVRA",
    "FakeVideo-FakeAudio": "FVFA",
}

# human-readable bucket names matching run_experiments.py's breakdown
BUCKET = {
    "RealVideo-RealAudio": "real",
    "RealVideo-FakeAudio": "audio only",
    "FakeVideo-RealAudio": "visual only",
    "FakeVideo-FakeAudio": "both",
}


def load_metadata(root: Path) -> dict:
    """(relative directory, filename) -> generation method.

    meta_data.csv must be indexed on the *full* path: filenames repeat across
    categories, so keying on name alone silently returns another category's
    method. Columns are positional --- the header has a trailing comma, which
    makes DictReader put the final path column under a None key.
    """
    csv_path = root / "meta_data.csv"
    out: dict = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as f:
        rd = csv.reader(f)
        next(rd, None)                       # header
        for row in rd:
            if len(row) < 10:
                continue
            method, fname, dpath = row[3].strip(), row[8].strip(), row[9].strip()
            dpath = dpath.replace("FakeAVCeleb/", "", 1).strip("/")
            if fname and dpath:
                out[(dpath, fname)] = method or "unknown"
    return out


def _method_of(path: Path, category: str, root: Path, meta: dict) -> str:
    """Generation method (faceswap / fsgan / wavtolip / rtvc)."""
    if category == "RealVideo-RealAudio":
        return "real"
    try:
        rel = str(path.parent.relative_to(root))
    except ValueError:
        rel = ""
    m = meta.get((rel, path.name))
    if m:
        return m
    # Fallback on the filename suffix. Note FakeAVCeleb spells it "wavtolip".
    stem = path.stem.lower()
    hits = [k for k in ("faceswap", "fsgan", "wavtolip", "wav2lip", "rtvc")
            if k in stem]
    return "-".join(hits) if hits else "unknown"


def discover(root: Path, per_category: int, seed: int = 42):
    """Stratified draw: all of the small categories, balanced sample of the large."""
    rng = random.Random(seed)
    meta = load_metadata(root)
    print(f"  metadata entries: {len(meta)}")
    selected = []

    for cat, (label, mv, ma) in CATEGORIES.items():
        cdir = root / cat
        if not cdir.is_dir():
            print(f"  ! missing category directory: {cat}")
            continue
        files = sorted(cdir.rglob("*.mp4"))

        # Stratify by generation method so one method cannot dominate the draw.
        by_method = defaultdict(list)
        for f in files:
            by_method[_method_of(f, cat, root, meta)].append(f)
        for v in by_method.values():
            rng.shuffle(v)

        take, methods = [], sorted(by_method)
        i = 0
        while len(take) < min(per_category, len(files)) and methods:
            m = methods[i % len(methods)]
            if by_method[m]:
                take.append(by_method[m].pop())
            else:
                methods.remove(m)
                continue
            i += 1

        print(f"  {cat:<22} {len(files):>6} available, taking {len(take):>4} "
              f"({', '.join(f'{m}:{len(by_method[m])}' for m in sorted(by_method))[:60]})")

        for f in take:
            selected.append({
                "path": str(f),
                # Full category tag, not cat[:4]: "FakeVideo-RealAudio" and
                # "FakeVideo-FakeAudio" both truncate to "Fake", and filenames
                # repeat across categories, so the short form collides.
                "video_id": f"favc_{CAT_TAG[cat]}_{f.parent.name}_{f.stem}",
                "label": label, "modify_video": mv, "modify_audio": ma,
                "category": cat, "bucket": BUCKET[cat],
                "method": _method_of(f, cat, root, meta),
            })
    rng.shuffle(selected)
    return selected


def _worker(item):
    try:
        enc, wav, stats = process_video(item["path"])
        return item, enc, wav, stats
    except Exception as exc:  # noqa: BLE001
        return item, None, None, {"error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="FakeAVCeleb_v1.2 directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", type=int, default=500,
                    help="videos per category (A and B cap out at 500 each)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Discovering FakeAVCeleb videos ...")
    items = discover(root, args.per_category, args.seed)
    print(f"\nselected {len(items)} videos")
    print("  by bucket:", dict(Counter(i["bucket"] for i in items)))
    print("  by method:", dict(Counter(i["method"] for i in items)))
    print()

    frames_bin = open(out / "frames.bin", "wb")
    audio_bin = open(out / "audio.bin", "wb")
    frame_off, video_frame_ptr, audio_off = [0], [0], [0]
    rows, failures = [], []
    t0, done = time.time(), 0

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, it) for it in items]
        for fut in as_completed(futures):
            item, enc, wav, stats = fut.result()
            done += 1
            if enc is None:
                failures.append({"video_id": item["video_id"], **stats})
            else:
                for b in enc:
                    frames_bin.write(b)
                    frame_off.append(frame_off[-1] + len(b))
                video_frame_ptr.append(len(frame_off) - 1)
                audio_bin.write(wav.tobytes())
                audio_off.append(audio_off[-1] + wav.size)

                dur = stats["duration"]
                periods = [[0.0, dur]] if item["label"] == 1 else []
                rows.append({
                    "video_id": item["video_id"],
                    "split": "test",
                    "n_frames": stats["n_frames"],
                    "duration": round(dur, 4),
                    "src_fps": stats["src_fps"],
                    "miss_rate": round(stats["miss_rate"], 4),
                    "n_samples": int(wav.size),
                    "label": item["label"],
                    "modify_video": item["modify_video"],
                    "modify_audio": item["modify_audio"],
                    "category": item["category"],
                    "method": item["method"],
                    "fake_periods": json.dumps(periods),
                })
            if done % 50 == 0 or done == len(items):
                el = time.time() - t0
                rate = done / max(1e-9, el)
                print(f"  [{done}/{len(items)}] {rate:.2f} vid/s  "
                      f"eta={(len(items)-done)/max(1e-9,rate)/60:.1f} min", flush=True)

    frames_bin.close(); audio_bin.close()

    np.savez_compressed(out / "index.npz",
                        frame_offsets=np.array(frame_off, dtype=np.int64),
                        video_frame_ptr=np.array(video_frame_ptr, dtype=np.int64),
                        audio_offsets=np.array(audio_off, dtype=np.int64))
    if rows:
        with open(out / "videos.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    by_bucket = Counter(BUCKET[r["category"]] for r in rows)
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "FakeAVCeleb v1.2",
        "note": ("category C (FakeVideo-RealAudio) has genuine audio; detection "
                 "rate on C versus B measures which modality the model uses"),
        "n_written": len(rows), "n_failed": len(failures),
        "n_fake": sum(r["label"] == 1 for r in rows),
        "n_real": sum(r["label"] == 0 for r in rows),
        "by_bucket": dict(by_bucket),
        "by_method": dict(Counter(r["method"] for r in rows)),
        "crop_size": CROP_SIZE, "target_fps": TARGET_FPS, "audio_sr": AUDIO_SR,
        "mean_miss_rate": round(float(np.mean([r["miss_rate"] for r in rows])), 5)
        if rows else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "failures": failures[:50],
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)

    size_mb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  written   : {len(rows)}  (fake {manifest['n_fake']}, "
          f"real {manifest['n_real']}, failed {len(failures)})")
    print(f"  by bucket : {dict(by_bucket)}")
    print(f"  miss rate : {manifest['mean_miss_rate']}")
    print(f"  size      : {size_mb:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
