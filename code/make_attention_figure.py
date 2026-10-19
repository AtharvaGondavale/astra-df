"""
ASTRA-DF :: real cross-modal attention figure (replaces the illustrative one)
=============================================================================

Plots the head-averaged V->A attention of a TRAINED checkpoint on a REAL
genuine LAV-DF test clip, with its original audio (left) and with the audio
delayed by 8 video frames = 267 ms (right). The dashed line marks the
temporally corresponding audio frame j*(t).

To avoid cherry-picking, the clip is not chosen by eye: among all genuine test
clips we take the one whose diagonal mass D_3 (unshifted) is the MEDIAN, and
the chosen clip id is printed and written next to the figure.

    python make_attention_figure.py --ckpt ../assets/checkpoints/full_s42.pt \
        --lavdf ../assets/astra_cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as R  # noqa: E402
from eval_suite import FPS, HOP_SEC, attention_stats, genuine_subset, load_model  # noqa: E402

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8})


def attn_for(model, ds, idx, device):
    b = R.collate([ds[idx]])
    with torch.no_grad():
        out = model(b["video"].to(device), b["audio"].to(device))
    return out["attn_v2a"][0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lavdf", required=True)
    ap.add_argument("--shift-frames", type=int, default=8)
    ap.add_argument("--out", default=str(HERE.parent / "paper" / "figures" / "attention_real.pdf"))
    args = ap.parse_args()

    R.M.update(R._import_astra(str(HERE)))
    device = torch.device("cpu")
    model, cfg, info = load_model(Path(args.ckpt), device)

    base = genuine_subset(R.PackedLAVDF(args.lavdf, "test"))
    # D_3 of every genuine clip, unshifted -> pick the median clip
    Ds = []
    for i in range(len(base)):
        A = torch.from_numpy(attn_for(model, base, i, device))[None]
        Ds.append(float(attention_stats(A)[1]))
    order = np.argsort(Ds)
    pick = int(order[len(order) // 2])
    shifted = genuine_subset(R.PackedLAVDF(args.lavdf, "test",
                                           audio_shift=args.shift_frames / FPS))
    A0 = attn_for(model, base, pick, device)
    A1 = attn_for(model, shifted, pick, device)
    clip = base.dataset.clips[base.indices[pick]]
    vid = str(base.dataset.meta.iloc[clip[0]]["video_id"])

    T, Ta = A0.shape
    jstar = (np.arange(T) + 0.5) / FPS / HOP_SEC
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
    vmax = max(A0.max(), A1.max())
    for ax, A, title in ((axes[0], A0, "original audio"),
                         (axes[1], A1, f"audio delayed {1000 * args.shift_frames / FPS:.0f} ms")):
        c, D, dsig, dabs = (float(x) for x in attention_stats(torch.from_numpy(A)[None]))
        im = ax.imshow(A, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.plot(jstar, np.arange(T), "--", color="white", lw=0.8)
        ax.set_title(f"{title}\n$D_3$={D:.2f}, $\\bar\\Delta$={1000 * dsig:+.0f} ms, $\\bar c$={c:.2f}",
                     fontsize=6.5)
        ax.set_xlabel("audio frame $j$")
    axes[0].set_ylabel("visual frame $t$")
    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(Path(args.out).with_suffix(".png"), dpi=200, bbox_inches="tight")
    meta = {"checkpoint": info, "video_id": vid, "clip_index": int(clip[1]),
            "selection": "median D_3 over genuine LAV-DF test clips",
            "n_genuine_clips": len(base), "D3_median": Ds[pick]}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
