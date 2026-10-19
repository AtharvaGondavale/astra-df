"""Visualization helpers — turn model outputs into images/figures."""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np


def plot_sync_timeline(times, scores, intervals=None, threshold: float = 0.5,
                       fake_intervals=None, title: str = "Audio-Visual Sync Timeline") -> bytes:
    """Render the frame-level sync score against time, with optional fake-interval shading."""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(times, scores, linewidth=1.4, color="#1f4e79", label="Fake probability")
    ax.axhline(threshold, linestyle="--", color="grey", alpha=0.6, label=f"Threshold = {threshold}")
    if fake_intervals:
        for iv in fake_intervals:
            ax.axvspan(iv["start"], iv["end"], color="#d6604d", alpha=0.25)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fake probability")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def plot_attention_heatmap(attn_matrix: np.ndarray, title: str = "Cross-Modal Attention") -> bytes:
    """attn_matrix : (T_v, T_a) — produce a heatmap PNG (bytes)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(attn_matrix, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Audio frame")
    ax.set_ylabel("Visual frame")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def overlay_heatmap_on_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap (H, W) in [0, 1] onto an RGB frame (H, W, 3) in [0, 1]."""
    import matplotlib.cm as cm
    h = np.clip(heatmap, 0, 1)
    hm_rgb = cm.get_cmap("jet")(h)[..., :3]
    blended = (1 - alpha) * frame + alpha * hm_rgb
    return np.clip(blended, 0, 1)
