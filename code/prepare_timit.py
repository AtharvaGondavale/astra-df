"""
ASTRA-DF :: DeepfakeTIMIT + VidTIMIT cache builder (cross-dataset evaluation)
=============================================================================

Builds a cache in exactly the same format as prepare_subset.py, so
run_experiments.py can evaluate a LAV-DF-trained checkpoint on it with no
code changes. Everything is written with split="test" because this set is
used zero-shot -- we never train on it.

Why this dataset matters, and what to expect
--------------------------------------------
DeepfakeTIMIT is *identity* face-swapping (faceswap-GAN), and its README is
explicit: "no manipulation was done to the audio channel". A face swap
transfers identity while preserving the driving video's mouth motion, so
audio-visual synchrony is largely INTACT.

ASTRA-DF detects lip-speech desynchronisation. That cue is therefore mostly
absent here, and we expect degraded recall. This is deliberate: the dataset
probes the boundary of the method's operating envelope rather than flattering
it. Report it as such.

Layout expected
---------------
DeepfakeTIMIT (fakes, label=1):
    <root>/higher_quality/<speaker>/<utt>-video-<target>.avi   (+ .wav)
    <root>/lower_quality/<speaker>/<utt>-video-<target>.avi

VidTIMIT (reals, label=0) -- JPEG frame sequences, not video files:
    <root>/<speaker>/video/<sentence>/<numbered JPEGs>
    <root>/<speaker>/audio/<sentence>[.wav]

Usage
-----
    python prepare_timit.py \
        --deepfake-timit /path/to/DeepfakeTIMIT \
        --vidtimit       /path/to/vidtimit \
        --out ./timit_cache --workers 8

VidTIMIT is optional; without it you get a fakes-only set and can report
detection rate but not F1.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_subset import (  # noqa: E402
    CROP_SIZE, TARGET_FPS, AUDIO_SR, JPEG_QUALITY,
    mouth_bbox, resize_pad, smooth_box, process_video, extract_wav,
)


# ==========================================================================
# VidTIMIT: JPEG frame sequences instead of a video container
# ==========================================================================
def _read_frame_sequence(frame_dir: Path):
    """VidTIMIT stores frames as numbered JPEGs, often with no extension."""
    import cv2

    files = sorted(p for p in frame_dir.iterdir() if p.is_file()
                   and not p.name.startswith('.'))
    frames = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def _find_vidtimit_audio(speaker_dir: Path, sentence: str):
    """Audio may be <audio>/<sentence>, <sentence>.wav or <sentence>.au."""
    audio_dir = speaker_dir / 'audio'
    for cand in (audio_dir / sentence,
                 audio_dir / f'{sentence}.wav',
                 audio_dir / f'{sentence}.au'):
        if cand.exists():
            return cand
    return None


def _decode_audio_to_pcm(path: Path):
    """Any audio container -> 16 kHz mono int16, via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
        tmp = tf.name
    try:
        proc = subprocess.run(
            ['ffmpeg', '-y', '-i', str(path), '-ac', '1',
             '-ar', str(AUDIO_SR), '-f', 'wav', tmp],
            capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError('ffmpeg failed on ' + str(path))
        import wave
        with wave.open(tmp, 'rb') as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def process_frame_sequence(frame_dir: Path, audio_path: Path, src_fps: float = 25.0):
    """VidTIMIT equivalent of prepare_subset.process_video."""
    import cv2
    import mediapipe as mp

    raw = _read_frame_sequence(frame_dir)
    if not raw:
        raise RuntimeError(f'no frames decoded from {frame_dir}')

    duration = len(raw) / src_fps
    target_t = np.arange(0.0, duration, 1.0 / TARGET_FPS)
    src_idx = np.clip((target_t * src_fps).astype(int), 0, len(raw) - 1)

    crops, prev_box, missed = {}, None, 0
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
        for i in sorted(set(src_idx.tolist())):
            frame = raw[i]
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                h, w = frame.shape[:2]
                box = smooth_box(prev_box,
                                 mouth_bbox(res.multi_face_landmarks[0].landmark, w, h))
                prev_box = box
                x1, y1, x2, y2 = box
                crops[i] = resize_pad(frame[y1:y2, x1:x2])
            else:
                crops[i] = np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8)
                missed += 1

    enc = [cv2.imencode('.jpg', crops[i],
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])[1].tobytes()
           for i in src_idx]
    wav = _decode_audio_to_pcm(audio_path)

    stats = {'n_frames': len(enc), 'n_landmarked': len(crops),
             'miss_rate': missed / max(1, len(crops)),
             'duration': duration, 'src_fps': src_fps}
    return enc, wav, stats


# ==========================================================================
# discovery
# ==========================================================================
def discover_deepfake_timit(root: Path):
    items = []
    for quality in ('higher_quality', 'lower_quality'):
        qdir = root / quality
        if not qdir.is_dir():
            continue
        for spk in sorted(p for p in qdir.iterdir() if p.is_dir()):
            for avi in sorted(spk.glob('*.avi')):
                items.append({
                    'video_id': f'dftimit_{quality[:2]}_{spk.name}_{avi.stem}',
                    'path': str(avi), 'kind': 'video',
                    'label': 1, 'speaker': spk.name, 'quality': quality,
                })
    return items


def discover_vidtimit(root: Path):
    items = []
    for spk in sorted(p for p in root.iterdir() if p.is_dir()):
        vdir = spk / 'video'
        if not vdir.is_dir():
            continue
        for sent in sorted(p for p in vdir.iterdir() if p.is_dir()):
            # skip head-rotation sequences, which contain no speech
            if sent.name.lower().startswith('head'):
                continue
            audio = _find_vidtimit_audio(spk, sent.name)
            if audio is None:
                continue
            items.append({
                'video_id': f'vidtimit_{spk.name}_{sent.name}',
                'path': str(sent), 'audio': str(audio), 'kind': 'frames',
                'label': 0, 'speaker': spk.name, 'quality': 'real',
            })
    return items


def _worker(item):
    try:
        if item['kind'] == 'video':
            enc, wav, stats = process_video(item['path'])
        else:
            enc, wav, stats = process_frame_sequence(Path(item['path']),
                                                     Path(item['audio']))
        if item.get('audio_override'):
            # Identical audio path for genuine and fake clips: both take the
            # 16 kHz original-sentence WAV that DeepfakeTIMIT ships, decoded by
            # the same function and length-matched to the clip's own audio.
            # Without this, genuine (VidTIMIT 32 kHz) and fake (AVI track)
            # audio go through different decoders/resamplers, and an
            # audio-only model can separate them on processing artefacts.
            o = _decode_audio_to_pcm(Path(item['audio_override']))
            n = wav.size
            wav = (o[:n] if o.size >= n else np.pad(o, (0, n - o.size))).astype(np.int16)
            stats = {**stats, 'audio_source': item['audio_override']}
        return item, enc, wav, stats
    except Exception as exc:  # noqa: BLE001
        return item, None, None, {'error': f'{type(exc).__name__}: {exc}'}


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deepfake-timit', required=True)
    ap.add_argument('--vidtimit', default=None,
                    help='optional; without it the set is fakes-only')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--shared-audio', action='store_true',
                    help='give genuine and fake clips of the same sentence the '
                         'identical DeepfakeTIMIT <spk>/<sentence>.wav audio')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = discover_deepfake_timit(Path(args.deepfake_timit))
    print(f'DeepfakeTIMIT fakes : {len(items)}')
    if args.vidtimit:
        reals = discover_vidtimit(Path(args.vidtimit))
        print(f'VidTIMIT reals      : {len(reals)}')
        items += reals
    else:
        print('VidTIMIT reals      : 0  (fakes-only: recall only, no F1)')

    if args.shared_audio:
        root = Path(args.deepfake_timit)
        n_set = 0
        for it in items:
            sent = (Path(it['path']).stem.split('-video-')[0] if it['kind'] == 'video'
                    else Path(it['path']).name)
            for q in ('higher_quality', 'lower_quality'):
                w = root / q / it['speaker'] / f'{sent}.wav'
                if w.exists():
                    it['audio_override'] = str(w); n_set += 1
                    break
        print(f'shared audio        : {n_set}/{len(items)} clips use the DeepfakeTIMIT sentence WAV')
        if n_set != len(items):
            raise SystemExit('shared audio missing for some clips -- refusing to build a mixed cache')

    if args.limit:
        items = items[:args.limit]
    print(f'total to process    : {len(items)}\n')

    frames_bin = open(out / 'frames.bin', 'wb')
    audio_bin = open(out / 'audio.bin', 'wb')
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
                failures.append({'video_id': item['video_id'], **stats})
            else:
                for b in enc:
                    frames_bin.write(b)
                    frame_off.append(frame_off[-1] + len(b))
                video_frame_ptr.append(len(frame_off) - 1)
                audio_bin.write(wav.tobytes())
                audio_off.append(audio_off[-1] + wav.size)

                dur = stats['duration']
                # whole-video manipulation for fakes; nothing for reals
                periods = [[0.0, dur]] if item['label'] == 1 else []
                rows.append({
                    'video_id': item['video_id'],
                    'split': 'test',            # zero-shot: never trained on
                    'n_frames': stats['n_frames'],
                    'duration': round(dur, 4),
                    'src_fps': stats['src_fps'],
                    'miss_rate': round(stats['miss_rate'], 4),
                    'n_samples': int(wav.size),
                    'label': item['label'],
                    'modify_video': item['label'],
                    'modify_audio': 0,          # audio never manipulated here
                    'speaker': item['speaker'],
                    'quality': item['quality'],
                    'fake_periods': json.dumps(periods),
                })
            if done % 25 == 0 or done == len(items):
                el = time.time() - t0
                rate = done / max(1e-9, el)
                print(f'  [{done}/{len(items)}] {rate:.2f} vid/s  '
                      f'eta={(len(items)-done)/max(1e-9,rate)/60:.1f} min', flush=True)

    frames_bin.close()
    audio_bin.close()

    np.savez_compressed(out / 'index.npz',
                        frame_offsets=np.array(frame_off, dtype=np.int64),
                        video_frame_ptr=np.array(video_frame_ptr, dtype=np.int64),
                        audio_offsets=np.array(audio_off, dtype=np.int64))

    if rows:
        with open(out / 'videos.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    manifest = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'DeepfakeTIMIT' + ('+VidTIMIT' if args.vidtimit else ''),
        'note': ('identity face-swap with UNMODIFIED audio; lip-sync is largely '
                 'preserved, so this set probes the limit of a sync-based detector'),
        'n_written': len(rows), 'n_failed': len(failures),
        'n_fake': sum(r['label'] == 1 for r in rows),
        'n_real': sum(r['label'] == 0 for r in rows),
        'crop_size': CROP_SIZE, 'target_fps': TARGET_FPS, 'audio_sr': AUDIO_SR,
        'mean_miss_rate': round(float(np.mean([r['miss_rate'] for r in rows])), 5)
        if rows else None,
        'elapsed_sec': round(time.time() - t0, 1),
        'failures': failures[:50],
    }
    json.dump(manifest, open(out / 'manifest.json', 'w'), indent=2)

    size_mb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  written   : {len(rows)}  (fake {manifest['n_fake']}, "
          f"real {manifest['n_real']}, failed {len(failures)})")
    print(f"  miss rate : {manifest['mean_miss_rate']}")
    print(f"  size      : {size_mb:.0f} MB -> {out}")
    if manifest['n_real'] == 0:
        print('\n  NOTE: no reals -> report detection rate (recall) only, not F1.')


if __name__ == '__main__':
    main()
