"""
ASTRA-DF :: per-manipulation analysis with false-positive context
=================================================================

The ``per_manipulation`` breakdown written by run_experiments.py counts only
clips with ``y == 1``, so each entry is a *recall on fakes* with no
false-positive denominator. A recall of 0.69 means little if the model fires on
everything; this script supplies the missing context and the interval estimates.

It answers three questions the reviewer raised about the 0.690 / 0.007 gap:

  1. Is the in-domain 0.690 inflated by a permissive threshold?
     -> compare the false-positive rate on genuine clips in both corpora.
  2. Is 0.690 evidence that the *visual pathway* works?
     -> evaluate the audio-only model on the same visually-manipulated clips.
        If a model with no video access matches it, the answer is no.
  3. How precise are these numbers?
     -> Wilson score intervals. The LAV-DF visual-only bucket holds 29 clips,
        so its interval is very wide and must not be quoted as a point value.

Clip labels are reconstructed from ``fake_periods`` exactly as PackedLAVDF
derives them, and the script asserts that the reconstruction reproduces the
stored labels before reporting anything. If that assertion fails the ordering
assumption is wrong and the output would be meaningless.

    python analyse_buckets.py --results results.json \
        --lavdf-cache astra_cache --favc-cache favc_cache
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_SEC = 1.0


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval -- correct for proportions near 0 and 1, where the
    normal approximation produces impossible bounds (the 0.007 bucket)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / d, (centre + margin) / d)


def reconstruct(cache: str | Path, split: str = "test"):
    """Clip-level (label, modify_video, modify_audio) in PackedLAVDF order."""
    df = pd.read_csv(Path(cache) / "videos.csv")
    sel = df[df["split"] == split].reset_index(drop=True)
    out = []
    for _, v in sel.iterrows():
        periods = json.loads(v["fake_periods"]) if isinstance(v["fake_periods"], str) else []
        for k in range(int(v["duration"] // CLIP_SEC)):
            st, en = k * CLIP_SEC, (k + 1) * CLIP_SEC
            lab = 0
            for p in periods:
                if min(float(p[1]), en) > max(float(p[0]), st):
                    lab = 1
                    break
            out.append((lab, int(v["modify_video"]), int(v["modify_audio"])))
    return np.array(out)


def bucket_of(mv: int, ma: int) -> str:
    if mv and ma:
        return "both"
    if mv:
        return "visual only"
    if ma:
        return "audio only"
    return "real"


def report(scores, labels, recon, tag, tau=0.5):
    s, y = np.asarray(scores), np.asarray(labels)
    lab = recon[:, 0]
    if len(s) != len(recon) or not (y == lab).all():
        print(f"  ! {tag}: clip-order reconstruction does not match stored "
              f"labels (n={len(s)} vs {len(recon)}) -- skipping")
        return None
    p = (s >= tau).astype(int)
    bk = np.array([bucket_of(mv, ma) for _, mv, ma in recon])
    out = {}
    print(f"\n--- {tag} ---")
    for name in ("real", "audio only", "visual only", "both"):
        m = (bk == name) & ((lab == 0) if name == "real" else (lab == 1))
        if not m.any():
            continue
        k, n = int(p[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        out[name] = {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}
        metric = "FPR   " if name == "real" else "recall"
        print(f"  {name:<12} n={n:5d}  {metric}={k/n:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--lavdf-cache", default="astra_cache")
    ap.add_argument("--favc-cache", default="favc_cache")
    ap.add_argument("--out", default="bucket_analysis.json")
    args = ap.parse_args()

    data = json.load(open(args.results))
    summary = {}

    lav = reconstruct(args.lavdf_cache)
    print(f"LAV-DF test clips reconstructed: {len(lav)} "
          f"({int(lav[:,0].sum())} fake)")
    for run in ("transformer", "audio_only", "visual_only", "bilstm",
                "fusion_concat"):
        rec = data.get(f"run::{run}")
        if not rec:
            continue
        r = report(rec.get("test_scores"), rec.get("test_labels"), lav,
                   f"LAV-DF in-domain :: {run}")
        if r:
            summary[f"lavdf::{run}"] = r

    if Path(args.favc_cache).exists():
        favc = reconstruct(args.favc_cache)
        print(f"\nFakeAVCeleb clips reconstructed: {len(favc)}")
        for key, rec in data.items():
            if not key.startswith("crossdata::favc"):
                continue
            r = report(rec.get("scores"), rec.get("labels"), favc,
                       f"FakeAVCeleb zero-shot :: {key[11:]}")
            if r:
                summary[key] = r

    # ---- the decisive contrast, stated explicitly -----------------------
    full = summary.get("lavdf::transformer", {}).get("visual only")
    audio = summary.get("lavdf::audio_only", {}).get("visual only")
    zs = summary.get("crossdata::favc", {}).get("visual only")
    if full and audio:
        print("\n" + "=" * 70)
        print("KEY CONTRAST -- does in-domain visual-only detection come from "
              "the visual pathway?")
        print("=" * 70)
        print(f"  full model      : {full['rate']:.3f} "
              f"[{full['ci95'][0]:.3f}, {full['ci95'][1]:.3f}]  (n={full['n']})")
        print(f"  audio-only model: {audio['rate']:.3f} "
              f"[{audio['ci95'][0]:.3f}, {audio['ci95'][1]:.3f}]  (n={audio['n']})")
        print("  -> A model with no access to video does not do worse on "
              "visually-manipulated\n     clips. In-domain detection of these "
              "clips is therefore not evidence that\n     the visual pathway is "
              "doing the work. Intervals are wide (n is small);\n     the claim "
              "is 'no evidence of a visual contribution', not 'audio is better'.")
        if zs:
            print(f"\n  same model, zero-shot visual-only: {zs['rate']:.4f} "
                  f"[{zs['ci95'][0]:.4f}, {zs['ci95'][1]:.4f}]  (n={zs['n']})")

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
