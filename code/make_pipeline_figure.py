"""
ASTRA-DF :: processing-pipeline diagram (vector PDF)
====================================================

Draws the end-to-end pipeline requested by Reviewer A ("it is easier to
understand the pipeline if it is shown with a diagram"), with the tensor shape
on every edge and the notation used in the text (Table "Notation").

    python make_pipeline_figure.py            # -> ../paper/figures/pipeline.pdf (+ .png preview)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.2,
})

C_VIS = "#dbe8f5"     # visual stream
C_AUD = "#f8e3cf"     # audio stream
C_FUS = "#e4dcef"     # fusion
C_OUT = "#dcefdc"     # outputs
EDGE = "#333333"


def box(ax, x, y, w, h, title, body="", fc="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=EDGE, lw=0.7))
    if body:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontweight="bold" if bold else "normal", fontsize=7.2)
        ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=6.4,
                linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontweight="bold" if bold else "normal", fontsize=7.2)


def arrow(ax, x0, y0, x1, y1, label="", dy=0.022, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.7, color=EDGE, ls=ls, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom",
                fontsize=6.0, color="#222222")


def main() -> None:
    fig = plt.figure(figsize=(7.0, 2.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    yv, ya, h = 0.63, 0.15, 0.32          # visual row, audio row, box height
    w_in, w_pre, w_enc, w_tmp = 0.072, 0.122, 0.118, 0.082
    x_in, x_pre, x_enc, x_tmp = 0.004, 0.096, 0.300, 0.458
    x_br, w_br = 0.572, 0.150
    x_hd, w_hd = 0.748, 0.112
    x_tl, w_tl = 0.884, 0.112

    # ---- visual row ---------------------------------------------------------
    box(ax, x_in, yv, w_in, h, "Video", "1-s window\n(src. 25 fps)", fc="white")
    box(ax, x_pre, yv, w_pre, h, "Mouth ROI (M1)",
        "FaceMesh lip box,\npad 35%, EMA $\\rho$=0.75,\n30 fps, $96{\\times}96$", fc=C_VIS)
    box(ax, x_enc, yv, w_enc, h, "Visual enc. (M2)",
        "MobileNetV3-small\nper frame, 576$\\to$$d$", fc=C_VIS)
    box(ax, x_tmp, yv, w_tmp, h, "Temporal\n(M3)",
        "Transformer\nor BiLSTM", fc=C_VIS)

    # ---- audio row ----------------------------------------------------------
    box(ax, x_in, ya, w_in, h, "Audio", "16 kHz mono", fc="white")
    box(ax, x_pre, ya, w_pre, h, "MFCC (M1)",
        "13 coeffs, $n_\\mathrm{fft}$=512,\nhop 200 (12.5 ms)", fc=C_AUD)
    box(ax, x_enc, ya, w_enc, h, "Audio enc. (M2)",
        "3-layer 1-D CNN\n(128/256/512 ch), $\\to d$", fc=C_AUD)
    box(ax, x_tmp, ya, w_tmp, h, "Temporal\n(M3)",
        "Transformer\nor BiLSTM", fc=C_AUD)

    # ---- edges with shapes --------------------------------------------------
    arrow(ax, x_in + w_in, yv + h / 2, x_pre, yv + h / 2)
    arrow(ax, x_pre + w_pre, yv + h / 2, x_enc, yv + h / 2, "$T{\\times}3{\\times}96^2$")
    arrow(ax, x_enc + w_enc, yv + h / 2, x_tmp, yv + h / 2, "$T{\\times}d$")
    arrow(ax, x_in + w_in, ya + h / 2, x_pre, ya + h / 2)
    arrow(ax, x_pre + w_pre, ya + h / 2, x_enc, ya + h / 2, "$T_a{\\times}13$")
    arrow(ax, x_enc + w_enc, ya + h / 2, x_tmp, ya + h / 2, "$T_a{\\times}d$")

    # ---- bridge -------------------------------------------------------------
    yb, hb = ya, yv + h - ya
    box(ax, x_br, yb, w_br, hb, "", fc=C_FUS)
    ax.text(x_br + w_br / 2, yb + hb * 0.90, "Cross-modal bridge (M4)", ha="center",
            va="center", fontweight="bold", fontsize=7.2)
    ax.text(x_br + w_br / 2, yb + hb * 0.70,
            "$\\mathbf{H}^v$ queries $\\mathbf{H}^a$ (V$\\to$A)\n"
            "$\\mathbf{H}^a$ queries $\\mathbf{H}^v$ (A$\\to$V)\n$h$=4 heads, residual + MLP",
            ha="center", va="center", fontsize=6.4, linespacing=1.25)
    ax.text(x_br + w_br / 2, yb + hb * 0.36,
            "attention $\\mathbf{A}\\in\\mathbb{R}^{T\\times T_a}$\n$\\to$ concentration $c_t$,\n"
            "alignment $D_w$, offset $\\Delta$",
            ha="center", va="center", fontsize=6.2, linespacing=1.2, style="italic")
    arrow(ax, x_tmp + w_tmp, yv + h / 2, x_br, yv + h / 2, "$\\mathbf{H}^v$")
    arrow(ax, x_tmp + w_tmp, ya + h / 2, x_br, ya + h / 2, "$\\mathbf{H}^a$")

    # ---- heads --------------------------------------------------------------
    gap = 0.03
    hh = (yv + h - ya - 2 * gap) / 3
    ys = [yv + h - hh, yv + h - 2 * hh - gap, yv + h - 3 * hh - 2 * gap]
    heads = [("Clip (M5)", "$\\hat{y}=P(\\mathrm{fake})$\nfake iff $\\hat{y}>\\tau$=0.5"),
             ("Frame (M5)", "$\\hat{p}_t$, $t=1..T$"),
             ("Interval (M5)", "$(\\hat{s},\\hat{e})\\in[0,1]^2$")]
    for y, (t, b) in zip(ys, heads):
        box(ax, x_hd, y, w_hd, hh, t, b, fc=C_OUT)
        arrow(ax, x_br + w_br, y + hh / 2, x_hd, y + hh / 2)

    # ---- timeline -----------------------------------------------------------
    box(ax, x_tl, ys[2], w_tl, ys[0] + hh - ys[2], "", fc="white")
    ax.text(x_tl + w_tl / 2, ys[0] + hh * 0.70, "Timeline (M6)", ha="center", va="center",
            fontweight="bold", fontsize=7.2)
    ax.text(x_tl + w_tl / 2, (ys[0] + ys[2] + hh) / 2 - 0.04,
            "1-s windows\nmerged over the\nvideo: threshold,\nNMS, gap merge\n$\\to$ manipulated\nintervals",
            ha="center", va="center", fontsize=6.3, linespacing=1.2)
    for y in ys:
        arrow(ax, x_hd + w_hd, y + hh / 2, x_tl, y + hh / 2)

    # ---- notation footer ----------------------------------------------------
    ax.text(0.005, 0.05,
            "$T$=30 visual frames, $T_a$=81 audio frames per 1-s window; $d$=256. "
            "Ablations remove the visual or audio row, replace M4 by concatenation, "
            "or keep one attention direction.",
            ha="left", va="center", fontsize=6.3, color="#333333")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pipeline.pdf")
    fig.savefig(OUT / "pipeline_preview.png", dpi=220)
    print("wrote", OUT / "pipeline.pdf")


if __name__ == "__main__":
    main()
