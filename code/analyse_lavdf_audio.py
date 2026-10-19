"""
ASTRA-DF :: what happens to the audio of LAV-DF "video-only" fakes?
===================================================================

LAV-DF labels these videos as visually manipulated with real audio, yet an
audio-only model detects most of their manipulated clips. This script compares
each fake's soundtrack with the soundtrack of the genuine video it was made
from (metadata field ``original``):

  * residual energy BEFORE the manipulated segment (identical audio -> ~0);
  * the time shift of the audio AFTER the segment, found by cross-correlating a
    0.75-s excerpt of the fake against the original.

A non-zero shift after the segment with identical audio before it means the
original audio was spliced and re-timed at the manipulated word: a local acoustic
edit that an audio-only detector can find.

Needs ffmpeg on PATH (use ../.venv-prep/bin) and the raw LAV-DF videos.

    PATH="../.venv-prep/bin:$PATH" ../.venv/bin/python analyse_lavdf_audio.py \
        --lavdf-root "../../Major Project/LAV-DF" --cache ../assets/astra_cache
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate

SR = 16000


def wav(path: Path) -> np.ndarray:
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-ac", "1", "-ar", str(SR), tmp], check=True)
        a, _ = sf.read(tmp)
        return a
    finally:
        os.remove(tmp)


def analyse(fake: np.ndarray, orig: np.ndarray, start: float, end: float) -> dict:
    S, T = int(start * SR), int(end * SR)
    out = {}
    if S > SR // 2:   # enough audio before the segment to compare sample-by-sample
        n = min(S - SR // 10, len(orig))
        d = fake[:n] - orig[:n]
        out["pre_residual"] = float(np.sum(d ** 2) / (np.sum(fake[:n] ** 2) + 1e-12))
    if len(fake) - T > SR:   # 0.75-s excerpt after the segment, located in the original
        tail = fake[T + SR // 20: T + SR // 20 + int(0.75 * SR)]
        c = correlate(orig, tail, mode="valid")
        j = int(np.argmax(c))
        out["post_shift_ms"] = 1000 * (j - (T + SR // 20)) / SR
        out["post_corr"] = float(np.corrcoef(tail, orig[j:j + len(tail)])[0, 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lavdf-root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "results" / "lavdf_audio_check.json"))
    args = ap.parse_args()

    root = Path(args.lavdf_root)
    meta = {Path(e["file"]).stem: e for e in json.loads((root / "metadata.json").read_text())}
    df = pd.read_csv(Path(args.cache) / "videos.csv", dtype={"video_id": str})
    report = {}
    for kind, (mv, ma) in {"video-only": (1, 0), "audio-only": (0, 1), "both": (1, 1)}.items():
        rows = []
        for vid in df[(df.modify_video == mv) & (df.modify_audio == ma)].video_id:
            e = meta[vid]
            o = meta[Path(e["original"]).stem]
            fake, orig = wav(root / e["file"]), wav(root / o["file"])
            s, t = e["fake_periods"][0]
            rows.append({"video_id": vid, "segment": [s, t], **analyse(fake, orig, s, t)})
        pre = np.array([r["pre_residual"] for r in rows if "pre_residual" in r])
        sh = np.array([r["post_shift_ms"] for r in rows
                       if "post_shift_ms" in r and r["post_corr"] > 0.95])
        report[kind] = {
            "n_videos": len(rows),
            "n_pre_checked": int(len(pre)),
            "n_pre_identical": int((pre < 0.01).sum()),
            "n_post_matched": int(len(sh)),
            "post_shift_ms_median": float(np.median(sh)) if len(sh) else None,
            "post_shift_ms_p10_p90": [float(np.percentile(sh, 10)), float(np.percentile(sh, 90))] if len(sh) else None,
            "n_post_shift_ge_20ms": int((np.abs(sh) >= 20).sum()),
            "rows": rows,
        }
        print(f"{kind:11s} videos={len(rows)}  pre-identical {int((pre < 0.01).sum())}/{len(pre)}  "
              f"post matched {len(sh)}  |shift|>=20ms {int((np.abs(sh) >= 20).sum())}  "
              f"median shift {report[kind]['post_shift_ms_median']} ms  p10-p90 {report[kind]['post_shift_ms_p10_p90']}")
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
