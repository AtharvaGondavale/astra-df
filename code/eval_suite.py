"""
ASTRA-DF :: evaluation-only suite for the revision (no training)
================================================================

Runs every evaluation the revised manuscript needs over a set of trained
checkpoints, and stores raw per-clip outputs so all statistics can be
recomputed without touching a GPU again (see analyse_suite.py).

For each checkpoint:

  clean   LAV-DF test clips, unperturbed (per-clip P(fake) + labels)
  cf      test-time counterfactuals on the same clips: video replaced by
          zeros, by the video of another clip, frames shuffled in time, mouth
          crop degraded to 32x32 / 16x16 (models with both modalities), and
          additive audio noise at 30/20/10 dB SNR (models that hear audio)
  xd      zero-shot FakeAVCeleb / DeepfakeTIMIT(+VidTIMIT), per-clip scores
  attn    V->A attention statistics per clip (models with cross-attention):
          concentration c, diagonal mass D_w, signed and absolute expected
          offset, on LAV-DF test, FakeAVCeleb, and genuine LAV-DF test clips
          with the audio delayed by 0, 4 and 8 video frames

Nothing here tunes anything: the threshold is fixed at 0.5 downstream and no
zero-shot corpus feeds back into a model or a hyperparameter.

    python eval_suite.py --ckpt-dir ../assets/checkpoints \
        --lavdf ../assets/astra_cache --favc ../assets/favc_cache \
        --dftimit ../assets/timit_cache --out ../results/eval_suite.json

Resumable: (run, evaluation) pairs already in --out are skipped unless --force.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as R  # noqa: E402

FPS = 30
HOP_SEC = R.HOP / R.AUDIO_SR          # 12.5 ms
WINDOW = 3                            # w in Eq. (diag): +-3 audio frames
SHIFTS = (0, 4, 8)                    # audio delay in video frames for the sync test


# --------------------------------------------------------------------------
# run name -> model configuration
# --------------------------------------------------------------------------
def config_of(name: str) -> dict:
    """Map a checkpoint/run name to the VariantASTRA arguments that built it."""
    base = re.sub(r"_s\d+$", "", name)
    table = {
        "transformer": ("transformer", "cross", "va", "mobilenetv3"),
        "full": ("transformer", "cross", "va", "mobilenetv3"),
        "audio": ("transformer", "cross", "a", "mobilenetv3"),
        "audio_only": ("transformer", "cross", "a", "mobilenetv3"),
        "visual": ("transformer", "cross", "v", "mobilenetv3"),
        "visual_only": ("transformer", "cross", "v", "mobilenetv3"),
        "concat": ("transformer", "concat", "va", "mobilenetv3"),
        "fusion_concat": ("transformer", "concat", "va", "mobilenetv3"),
        "v2a": ("transformer", "v2a", "va", "mobilenetv3"),
        "fusion_v2a": ("transformer", "v2a", "va", "mobilenetv3"),
        "a2v": ("transformer", "a2v", "va", "mobilenetv3"),
        "fusion_a2v": ("transformer", "a2v", "va", "mobilenetv3"),
        "bilstm": ("bilstm", "cross", "va", "mobilenetv3"),
        "lipread_visual": ("transformer", "cross", "v", "lipread"),
        "lipread_rand_visual": ("transformer", "cross", "v", "lipread"),
        "lipread_full": ("transformer", "cross", "va", "lipread"),
    }
    if base not in table:
        raise KeyError(f"unknown run name '{name}' (base '{base}')")
    bk, fu, mo, ve = table[base]
    return dict(backbone=bk, fusion=fu, modalities=mo, visual_encoder=ve)


def load_model(ckpt: Path, device: torch.device):
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Checkpoints written by the revised pipeline carry their own architecture
    # config; older ones are identified by run name (and predate rgb_norm).
    cfg = {**config_of(ckpt.stem), "rgb_norm": False, "lipread_act": "auto",
           **{k: v for k, v in (raw.get("config") or {}).items() if v is not None}}
    m = R.VariantASTRA(cfg["backbone"], cfg["fusion"], cfg["modalities"],
                       pretrained=False, visual_encoder=cfg["visual_encoder"],
                       rgb_norm=cfg["rgb_norm"], lipread_act=cfg["lipread_act"])
    state = raw.get("model", raw)
    m.load_state_dict(state, strict=True)
    m = m.to(device).eval()
    info = {"path": str(ckpt), "sha256_12": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12],
            **cfg, "params": m.n_params()}
    return m, cfg, info


# --------------------------------------------------------------------------
# datasets for the counterfactuals
# --------------------------------------------------------------------------
class SwapVideo(Dataset):
    """Item i keeps its audio and label but takes the video of item perm[i].

    perm is a rotation by half the dataset, so the replacement video comes from
    a distant clip; the assertion below guarantees it is never the same video.
    """

    def __init__(self, ds: R.PackedLAVDF):
        self.ds = ds
        n = len(ds)
        self.perm = np.roll(np.arange(n), n // 2)
        vids = [ds.clips[i][0] for i in range(n)]
        same = sum(vids[i] == vids[self.perm[i]] for i in range(n))
        assert same == 0, f"{same} swapped clips kept their own video"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = dict(self.ds[i])
        item["video"] = self.ds[int(self.perm[i])]["video"]
        return item


def perturb_video(v: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
    if mode == "video-zero":
        return torch.zeros_like(v)
    if mode == "video-shuffle":
        B, T = v.shape[:2]
        idx = torch.stack([torch.randperm(T, generator=gen) for _ in range(B)])
        return v[torch.arange(B)[:, None], idx]
    if mode.startswith("mouth"):
        s = int(mode[5:])
        B, T, C, H, W = v.shape
        x = v.reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(s, s), mode="area")
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.reshape(B, T, C, H, W)
    raise ValueError(mode)


# --------------------------------------------------------------------------
# attention statistics
# --------------------------------------------------------------------------
def attention_stats(A: torch.Tensor, clip_sec: float = 1.0, w: int = WINDOW):
    """Per-clip statistics of head-averaged V->A attention A: (B, T, Ta).

    c      mean over rows of 1 - H(row)/log Ta                (concentration)
    D      mean over rows of mass in |j - j*(t)| <= w divided by the mass the
           same window receives under uniform attention        (1 = uniform)
    dsig   mean over rows of (sum_j A_tj * tau_j - t_v)  [s]  (signed offset)
    dabs   mean over rows of |sum_j A_tj * tau_j - t_v|  [s]  (alignment error)
    with t_v = (t + 1/2)/F the centre of visual frame t (0-indexed) and
    tau_j = j * hop the centre of audio frame j (librosa centred frames).
    """
    A = A.float().clamp_min(0)
    A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
    B, T, Ta = A.shape
    eps = 1e-12
    H = -(A * (A + eps).log()).sum(-1)
    c = (1 - H / np.log(Ta)).mean(-1)

    tv = (torch.arange(T, device=A.device, dtype=A.dtype) + 0.5) * (clip_sec / T)
    tau = torch.arange(Ta, device=A.device, dtype=A.dtype) * HOP_SEC
    jstar = tv / HOP_SEC                                         # (T,)
    dist = (torch.arange(Ta, device=A.device, dtype=A.dtype)[None, :] - jstar[:, None]).abs()
    win = (dist <= w + 1e-6).to(A.dtype)                         # (T, Ta)
    mass = (A * win[None]).sum(-1)                               # (B, T)
    unif = win.sum(-1) / Ta                                      # (T,)
    D = (mass / unif[None]).mean(-1)

    expected = (A * tau[None, None, :]).sum(-1)                  # (B, T)
    off = expected - tv[None, :]
    return c, D, off.mean(-1), off.abs().mean(-1)


# --------------------------------------------------------------------------
# inference loops
# --------------------------------------------------------------------------
def run_scores(model, ds, device, bs, workers, mode=None, want_attn=False):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    collate_fn=R.collate)
    gen = torch.Generator().manual_seed(0)
    scores, labels = [], []
    st = {"c": [], "D": [], "dsig": [], "dabs": []}
    with torch.no_grad():
        for b in dl:
            v = b["video"]
            if mode and mode != "video-swap":
                v = perturb_video(v, mode, gen)
            out = model(v.to(device), b["audio"].to(device))
            scores.extend(torch.softmax(out["logits"], -1)[:, 1].cpu().tolist())
            labels.extend(b["label"].tolist())
            if want_attn and out.get("attn_v2a") is not None:
                c, D, ds_, da = attention_stats(out["attn_v2a"].cpu())
                st["c"].extend(c.tolist()); st["D"].extend(D.tolist())
                st["dsig"].extend(ds_.tolist()); st["dabs"].extend(da.tolist())
    res = {"scores": [round(x, 5) for x in scores], "labels": [int(x) for x in labels]}
    if want_attn:
        res.update({k: [round(x, 5) for x in v] for k, v in st.items()})
    return res


def genuine_subset(ds: R.PackedLAVDF) -> Subset:
    keep = [i for i, c in enumerate(ds.clips) if int(ds.meta.iloc[c[0]]["label"]) == 0]
    return Subset(ds, keep)


# --------------------------------------------------------------------------
# results file (atomic, merge-on-write, resumable)
# --------------------------------------------------------------------------
class Store:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def has(self, run, key):
        return key in self.data.get(run, {})

    def put(self, run, key, value):
        disk = json.loads(self.path.read_text()) if self.path.exists() else {}
        disk.setdefault(run, {})[key] = value
        self.data = disk
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(disk))
        os.replace(tmp, self.path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--runs", default=None,
                    help="comma-separated checkpoint names (default: every *.pt in --ckpt-dir)")
    ap.add_argument("--lavdf", required=True)
    ap.add_argument("--favc", default=None)
    ap.add_argument("--dftimit", default=None)
    ap.add_argument("--evals", default="meta,clean,cf,xd,attn")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "eval_suite.json"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    R.M.update(R._import_astra(str(HERE)))
    device = (torch.device("cpu") if args.cpu else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print("device:", device)

    ckdir = Path(args.ckpt_dir)
    names = (args.runs.split(",") if args.runs else sorted(p.stem for p in ckdir.glob("*.pt")))
    evals = set(args.evals.split(","))
    store = Store(Path(args.out))

    test = R.PackedLAVDF(args.lavdf, "test")
    corpora = {k: R.PackedLAVDF(p, "test") for k, p in
               (("favc", args.favc), ("dftimit", args.dftimit)) if p}
    print(f"LAV-DF test clips: {len(test)}; " +
          ", ".join(f"{k}: {len(v)}" for k, v in corpora.items()))

    for name in names:
        ck = ckdir / f"{name}.pt"
        if not ck.exists():
            print(f"! missing checkpoint {ck}")
            continue
        try:
            model, cfg, info = load_model(ck, device)
        except KeyError as exc:
            print(f"! {exc}; skipping")
            continue
        both = cfg["modalities"] == "va"
        has_attn = both and cfg["fusion"] in ("cross", "v2a")
        print(f"\n== {name}  {info['sha256_12']}  {cfg}")
        t0 = time.time()

        def todo(key):
            return args.force or not store.has(name, key)

        if "meta" in evals:
            store.put(name, "meta", {**info, "device": str(device), "torch": torch.__version__,
                                     "time": time.strftime("%Y-%m-%d %H:%M")})
        if "clean" in evals and todo("clean"):
            store.put(name, "clean", run_scores(model, test, device, args.batch_size,
                                                args.workers, want_attn=has_attn))
            print("  clean done")
        if "cf" in evals and both:
            for mode in ("video-zero", "video-swap", "video-shuffle", "mouth32", "mouth16"):
                key = f"cf:{mode}"
                if todo(key):
                    ds = SwapVideo(test) if mode == "video-swap" else test
                    store.put(name, key, run_scores(model, ds, device, args.batch_size,
                                                    args.workers, mode=mode))
                    print(f"  {key} done")
        if "cf" in evals:
            # Audio degradation for every model that hears audio, at the same
            # SNRs as the original robustness sweep; MFCCs are recomputed from
            # the noisy waveform inside PackedLAVDF.
            if "a" in cfg["modalities"]:
                for snr in (30, 20, 10):
                    key = f"cf:snr{snr}"
                    if todo(key):
                        ds = R.PackedLAVDF(args.lavdf, "test", noise_snr=snr, seed=0)
                        store.put(name, key, run_scores(model, ds, device, args.batch_size,
                                                        args.workers))
                        print(f"  {key} done")
        if "xd" in evals:
            for cname, ds in corpora.items():
                key = f"xd:{cname}"
                if todo(key):
                    store.put(name, key, run_scores(model, ds, device, args.batch_size,
                                                    args.workers, want_attn=has_attn))
                    print(f"  {key} done")
        if "attn" in evals and has_attn:
            for s in SHIFTS:
                key = f"attn:shift{s}"
                if todo(key):
                    ds = genuine_subset(R.PackedLAVDF(args.lavdf, "test", audio_shift=s / FPS))
                    store.put(name, key, run_scores(model, ds, device, args.batch_size,
                                                    args.workers, want_attn=True))
                    print(f"  {key} done")
        print(f"  {name}: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
