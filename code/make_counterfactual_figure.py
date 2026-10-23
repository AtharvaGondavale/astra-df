"""
ASTRA-DF :: test-time input figure (replaces the former test-time input table)
=============================================================================

For the trained full model (seeds 42-46) on the LAV-DF test clips, plots

  (a) clip-level F1 with the video altered or the audio corrupted, and
  (b) the share of clip decisions (y_hat > tau) that change relative to the
      clean input,

as mean +- s.d. over seeds.  The statistics are computed exactly as in
analyse_suite.py (same clip table, F1 and threshold), so the figure agrees
with the cf.* macros quoted in the text.

    python make_counterfactual_figure.py --lavdf ../assets/astra_cache \
        --out ../paper/figures/counterfactual.pdf
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse_suite import TAU, check, clip_table, f1_of  # noqa: E402

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8})

# (eval key, label, group); top to bottom in the figure
CONDS = [("clean", "Clean input", "clean"),
         ("cf:video-zero", "Video replaced by zeros", "video"),
         ("cf:video-swap", "Video from another clip", "video"),
         ("cf:video-shuffle", "Video frames shuffled", "video"),
         ("cf:mouth32", r"Mouth crop at $32{\times}32$, upsampled", "video"),
         ("cf:snr30", "Audio noise, 30 dB SNR", "audio"),
         ("cf:snr20", "Audio noise, 20 dB SNR", "audio"),
         ("cf:snr10", "Audio noise, 10 dB SNR", "audio")]
STYLE = {"clean": ("0.35", "D"), "video": ("#0072B2", "o"), "audio": ("#D55E00", "s")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(HERE.parent / "results" / "rev" / "eval_suite.json"))
    ap.add_argument("--lavdf", required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--out", default=str(HERE.parent / "paper" / "figures" / "counterfactual.pdf"))
    args = ap.parse_args()

    suite = json.loads(Path(args.suite).read_text())
    lav = clip_table(args.lavdf, "test")
    y = lav["clip_label"].to_numpy()
    f1 = {k: [] for k, _, _ in CONDS}
    changed = {k: [] for k, _, _ in CONDS}
    for s in (int(x) for x in args.seeds.split(",")):
        run = suite[f"full_s{s}"]
        check(lav, run["clean"]["labels"], f"full_s{s} clean")
        base = np.asarray(run["clean"]["scores"])
        for key, _, _ in CONDS:
            check(lav, run[key]["labels"], f"full_s{s} {key}")
            sc = np.asarray(run[key]["scores"])
            f1[key].append(f1_of(y, sc))
            changed[key].append(100.0 * float(((sc > TAU) != (base > TAU)).mean()))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.9), sharey=True,
                                 gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.12})
    ypos = np.arange(len(CONDS))[::-1]
    for (key, _, group), yy in zip(CONDS, ypos):
        colour, marker = STYLE[group]
        a1.errorbar(np.mean(f1[key]), yy, xerr=np.std(f1[key], ddof=1), fmt=marker, color=colour,
                    ms=3.5, capsize=1.5, elinewidth=0.8)
        a2.barh(yy, np.mean(changed[key]), xerr=np.std(changed[key], ddof=1), color=colour,
                height=0.6, error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": "0.25"})
    a1.axvline(np.mean(f1["clean"]), color="0.35", ls="--", lw=0.7)
    a1.set_yticks(ypos)
    a1.set_yticklabels([lab for _, lab, _ in CONDS])
    a1.set_xlim(0.0, 1.0)
    a1.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    a1.set_xlabel("F1")
    a2.set_xlabel("Decisions changed (%)")
    a2.set_xlim(0, None)
    for ax, tag in ((a1, "(a)"), (a2, "(b)")):
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, va="top", fontsize=7.5)
        ax.tick_params(length=2, labelsize=7)
        ax.grid(axis="x", lw=0.3, color="0.85")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    summary = {key: {"f1_mean": float(np.mean(f1[key])), "f1_sd": float(np.std(f1[key], ddof=1)),
                     "changed_pct_mean": float(np.mean(changed[key])),
                     "changed_pct_sd": float(np.std(changed[key], ddof=1))} for key, _, _ in CONDS}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    for key, lab, _ in CONDS:
        d = summary[key]
        print(f"{lab:28s} F1 {d['f1_mean']:.3f} +- {d['f1_sd']:.3f}   changed {d['changed_pct_mean']:.2f}%")
    print("wrote", out)


if __name__ == "__main__":
    main()
