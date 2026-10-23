"""
ASTRA-DF :: statistics and LaTeX macros for the revised manuscript
==================================================================

Turns the raw per-clip outputs of eval_suite.py (plus the training records in
results_final.json and the hardware benchmark in latency.json) into every
number the revised paper cites, written as \\setres{key}{value} lines to
paper/results_suite.tex (input after results_auto.tex).

Clip identities (category, generator, video label) are reconstructed from each
cache's videos.csv in exactly the order PackedLAVDF enumerates clips, and the
script ASSERTS that the reconstructed clip labels equal the labels stored by
eval_suite.py before computing anything. A mismatch means the ordering
assumption is broken and nothing downstream would be meaningful.

    python analyse_suite.py --suite ../results/rev/eval_suite.json \
        --results ../results/rev/results_rev.json --latency ../results/latency.json \
        --lavdf ../assets/astra_cache --favc ../assets/favc_cache \
        --dftimit ../assets/timit_cache
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLIP_SEC = 1.0
TAU = 0.5


# --------------------------------------------------------------------------
# small statistics helpers
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as 1/2 (P(score_pos > score_neg))."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):                       # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def f1_of(y: np.ndarray, s: np.ndarray, tau: float = TAU) -> float:
    p = s > tau
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def ms(vals, nd=3, signed=False):
    """'mean $\\pm$ sd' (sample sd) or a single value when only one is available."""
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return None
    fmt = f"{{:{'+' if signed else ''}.{nd}f}}"
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(v.mean())} $\\pm$ {v.std(ddof=1):.{nd}f}"


def mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


# --------------------------------------------------------------------------
# clip identity reconstruction (PackedLAVDF order)
# --------------------------------------------------------------------------
def clip_table(cache: str | Path, split: str = "test") -> pd.DataFrame:
    df = pd.read_csv(Path(cache) / "videos.csv")
    sel = df[df["split"] == split].reset_index(drop=True)
    rows = []
    for _, v in sel.iterrows():
        periods = json.loads(v["fake_periods"]) if isinstance(v["fake_periods"], str) else []
        for k in range(int(v["duration"] // CLIP_SEC)):
            st, en = k * CLIP_SEC, (k + 1) * CLIP_SEC
            lab = int(any(min(float(p[1]), en) > max(float(p[0]), st) for p in periods))
            mv, ma = int(v["modify_video"]), int(v["modify_audio"])
            bucket = ("real" if int(v["label"]) == 0 else
                      "both" if (mv and ma) else "visual-only" if mv else
                      "audio-only" if ma else "unspecified")
            rows.append(dict(video_id=str(v["video_id"]), clip_label=lab,
                             video_label=int(v["label"]), bucket=bucket,
                             method=str(v.get("method", "")) if "method" in v else "",
                             quality=str(v.get("quality", "")) if "quality" in v else ""))
    return pd.DataFrame(rows)


def check(tab: pd.DataFrame, labels: list, what: str) -> None:
    got = np.asarray(labels, int)
    exp = tab["clip_label"].to_numpy()
    if len(got) != len(exp) or not (got == exp).all():
        raise AssertionError(
            f"{what}: stored labels ({len(got)}) do not match reconstruction "
            f"({len(exp)}); clip ordering assumption broken -- refusing to report")


def method_group(m: str) -> str:
    m = m.lower()
    if "faceswap" in m:
        return "faceswap"
    if "fsgan" in m:
        return "fsgan"
    if "wav2lip" in m or "wavtolip" in m:
        return "wav2lip"
    return m or "unknown"


# --------------------------------------------------------------------------
# per-category rates and AUCs
# --------------------------------------------------------------------------
def category_stats(tab: pd.DataFrame, scores: np.ndarray, groups: dict) -> dict:
    """groups: name -> boolean mask over clips (positives). Genuine = real-video clips."""
    neg = scores[(tab["bucket"] == "real").to_numpy()]
    out = {"real": {"n": int(len(neg)), "k": int((neg > TAU).sum()),
                    "rate": float((neg > TAU).mean()) if len(neg) else float("nan")}}
    for g, mask in groups.items():
        pos = scores[mask]
        out[g] = {"n": int(len(pos)), "k": int((pos > TAU).sum()),
                  "rate": float((pos > TAU).mean()) if len(pos) else float("nan"),
                  "auc": auc(pos, neg)}
    return out


def lavdf_groups(tab):
    fake = tab["clip_label"].to_numpy() == 1
    return {b: fake & (tab["bucket"] == b).to_numpy() for b in ("audio-only", "visual-only", "both")}


def favc_groups(tab):
    fake = tab["video_label"].to_numpy() == 1
    g = {b: fake & (tab["bucket"] == b).to_numpy() for b in ("audio-only", "visual-only", "both")}
    vo = g["visual-only"]
    meth = tab["method"].map(method_group).to_numpy()
    for m in ("faceswap", "fsgan", "wav2lip"):
        g[f"vo-{m}"] = vo & (meth == m)
    return g


def dft_groups(tab):
    return {"fake": tab["video_label"].to_numpy() == 1}


TABLE_SD: list[float] = []   # every across-seed s.d. that enters the zero-shot table


def aggregate(per_seed: list[dict], prefix: str, M: dict) -> None:
    # only the three models shown in Table 6 enter its caption statistics
    in_table = prefix.split(".")[-1] in ("full", "audio", "visual")
    """Mean over seeds of rate/auc, pooled Wilson CI, and n (identical across seeds)."""
    if not per_seed:
        return
    for g in per_seed[0]:
        rates = [d[g]["rate"] for d in per_seed]
        k = sum(d[g]["k"] for d in per_seed); n = sum(d[g]["n"] for d in per_seed)
        lo, hi = wilson(k, n)
        M[f"{prefix}.{g}.rate"] = ms(rates)
        M[f"{prefix}.{g}.mean"] = f"{mean(rates):.3f}"
        M[f"{prefix}.{g}.ci"] = f"[{lo:.3f}, {hi:.3f}]"
        if "auc" in per_seed[0][g]:
            aucs = [d[g]["auc"] for d in per_seed]
            M[f"{prefix}.{g}.auc"] = ms(aucs)
            M[f"{prefix}.{g}.aucm"] = f"{mean(aucs):.3f}"
            if len(aucs) > 1 and in_table:
                TABLE_SD.append(float(np.std(aucs, ddof=1)))
        if len(rates) > 1 and in_table:
            TABLE_SD.append(float(np.std(rates, ddof=1)))
        M[f"{prefix}.{g}.n"] = str(per_seed[0][g]["n"])


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# analyses added after the external pre-submission review
# --------------------------------------------------------------------------
def _auc_fast(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def video_bootstrap(tab: pd.DataFrame, per_seed_scores: list, pos: np.ndarray, neg: np.ndarray,
                    B: int = 500, seed: int = 0):
    """95% CI of the MEAN-OVER-SEEDS AUC and detection rate, resampling whole
    videos (clips of one video are correlated) within the positive and the
    negative group independently."""
    rng = np.random.default_rng(seed)
    vid = tab["video_id"].to_numpy()
    def groups(mask):
        idx = np.where(mask)[0]
        u, inv = np.unique(vid[idx], return_inverse=True)
        return [idx[inv == k] for k in range(len(u))]
    gp, gn = groups(pos), groups(neg)
    S = [np.asarray(x) for x in per_seed_scores]
    aucs, rates = [], []
    for _ in range(B):
        P = np.concatenate([gp[i] for i in rng.integers(0, len(gp), len(gp))])
        N = np.concatenate([gn[i] for i in rng.integers(0, len(gn), len(gn))])
        aucs.append(np.mean([_auc_fast(sc[P], sc[N]) for sc in S]))
        rates.append(np.mean([(sc[P] > TAU).mean() for sc in S]))
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(rates, [2.5, 97.5])


def review_extras(args, suite: dict, runs: dict, seeds: list, M: dict) -> None:
    models = (("full", "full", seeds), ("audio", "audio", seeds), ("visual", "visual", seeds),
              ("lipread_visual", "lrvisual", [42, 43, 44]), ("lipread_full", "lrfull", [42, 43, 44]),
              ("lipread_rand_visual", "lrrvisual", [42, 43, 44]))
    hw = {"zs": [], "lav": []}
    for cname, cache, gfun, prefix, key in (("lavdf", args.lavdf, lavdf_groups, "lv", "clean"),
                                            ("favc", args.favc, favc_groups, "xd", "xd:favc"),
                                            ("dftimit", args.dftimit, dft_groups, "dft", "xd:dftimit")):
        if not cache:
            continue
        tab = clip_table(cache, "test")
        g = gfun(tab)
        neg = (tab["bucket"] == "real").to_numpy()
        for tag, short, sd in models:
            S = [suite[f"{tag}_s{s}"][key]["scores"] for s in sd
                 if f"{tag}_s{s}" in suite and key in suite[f"{tag}_s{s}"]]
            if len(S) < len(sd):
                continue
            for gname, pos in g.items():
                if pos.sum() == 0:
                    continue
                (alo, ahi), (rlo, rhi) = video_bootstrap(tab, S, pos, neg)
                M[f"{prefix}.{short}.{gname}.aucci"] = f"[{alo:.3f}, {ahi:.3f}]"
                M[f"{prefix}.{short}.{gname}.rateci"] = f"[{rlo:.3f}, {rhi:.3f}]"
                if short in ("full", "audio", "visual"):   # Table 6 models only
                    hw["lav" if cname == "lavdf" else "zs"].append((ahi - alo) / 2)
            # DeepfakeTIMIT: lower- vs higher-quality fakes (free resolution probe)
            if cname == "dftimit":
                q = dict(zip(pd.read_csv(Path(cache) / "videos.csv")["video_id"].astype(str),
                             pd.read_csv(Path(cache) / "videos.csv")["quality"]))
                qq = tab["video_id"].map(q).to_numpy()
                for ql, lab in (("lower_quality", "lq"), ("higher_quality", "hq")):
                    m = (qq == ql)
                    M[f"dft.{short}.{lab}.aucm"] = f"{np.mean([_auc_fast(np.asarray(x)[m], np.asarray(x)[neg]) for x in S]):.3f}"
            # LAV-DF: genuine seconds INSIDE manipulated videos (excluded from the FPR)
            if cname == "lavdf":
                for b in ("audio-only", "visual-only", "both"):
                    m = ((tab["bucket"] == b) & (tab["clip_label"] == 0)).to_numpy()
                    M[f"lv.{short}.{b}-gen.mean"] = f"{np.mean([(np.asarray(x)[m] > TAU).mean() for x in S]):.3f}"
                    M[f"lv.{b}-gen.n"] = str(int(m.sum()))
    if hw["zs"]:
        M["tab.crossdata.maxhw-zs"] = f"{max(hw['zs']):.3f}"
    if hw["lav"]:
        M["tab.crossdata.maxhw-lav"] = f"{max(hw['lav']):.3f}"

    # exact composition of the LAV-DF subset (videos)
    v = pd.read_csv(Path(args.lavdf) / "videos.csv")
    cat = np.where(v["label"] == 0, "genuine",
                   np.where((v["modify_video"] == 1) & (v["modify_audio"] == 1), "both",
                            np.where(v["modify_video"] == 1, "video-only", "audio-only")))
    for c in ("genuine", "audio-only", "video-only", "both"):
        M[f"data.nvid.{c}"] = str(int((cat == c).sum()))
    M["data.nvid.total"] = str(len(v))
    M["data.nvid.fake"] = str(int((v["label"] == 1).sum()))

    # seed range of the full model, from the values as printed in Table 4
    f = [round(runs[f"run::full_s{s}"]["test"]["clip"]["f1"], 3) for s in seeds if f"run::full_s{s}" in runs]
    if len(f) > 1:
        M["ms.full.f1.range"] = f"{(max(f) - min(f)) * 100:.1f}"

    # LAV-DF audio re-timing check (analyse_lavdf_audio.py)
    p = Path(args.lavdf_audio)
    if p.exists():
        r = json.loads(p.read_text())["video-only"]
        M["lavaudio.n"] = str(r["n_videos"])
        M["lavaudio.pre-checked"] = str(r["n_pre_checked"])
        M["lavaudio.pre-identical"] = str(r["n_pre_identical"])
        M["lavaudio.post-matched"] = str(r["n_post_matched"])
        M["lavaudio.post-shifted"] = str(r["n_post_shift_ge_20ms"])
        M["lavaudio.shift-median"] = f"{abs(r['post_shift_ms_median']):.0f}"
        lo, hi = sorted(abs(x) for x in r["post_shift_ms_p10_p90"])
        M["lavaudio.shift-p10"] = f"{lo:.0f}"
        M["lavaudio.shift-p90"] = f"{hi:.0f}"
        # Denominators and overlap of the two tests, from the per-video rows, so
        # the paper can state exactly which videos each count refers to.
        rows = r["rows"]
        pre_ok = {x["video_id"] for x in rows if x.get("pre_residual", 1.0) < 0.01}
        matched = [x for x in rows if "post_shift_ms" in x and x["post_corr"] > 0.95]
        shifted = {x["video_id"] for x in matched if abs(x["post_shift_ms"]) >= 20}
        # Negative shift: the fake's post-word audio is found earlier in the
        # source, i.e. it is delayed. The text says "delayed", so count only those.
        delayed = {x["video_id"] for x in matched if x["post_shift_ms"] <= -20}
        assert len(pre_ok) == r["n_pre_identical"] and len(shifted) == r["n_post_shift_ge_20ms"]
        M["lavaudio.post-checked"] = str(sum("post_shift_ms" in x for x in rows))
        M["lavaudio.post-delayed"] = str(len(delayed))
        M["lavaudio.both"] = str(len(pre_ok & delayed))
        M["lavaudio.both-checked"] = str(sum("pre_residual" in x and x in matched for x in rows))
        res = [x["pre_residual"] for x in rows if x["video_id"] in pre_ok]
        M["lavaudio.pre-res-median"] = f"{100 * float(np.median(res)):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default=str(HERE.parent / "results" / "rev" / "eval_suite.json"))
    ap.add_argument("--results", default=str(HERE.parent / "results" / "rev" / "results_rev.json"))
    ap.add_argument("--split-check", default=str(HERE.parent / "results" / "split_check.json"))
    ap.add_argument("--latency", default=str(HERE.parent / "results" / "latency.json"))
    ap.add_argument("--lavdf-audio", default=str(HERE.parent / "results" / "lavdf_audio_check.json"),
                    help="output of analyse_lavdf_audio.py")
    ap.add_argument("--lavdf", default=None)
    ap.add_argument("--favc", default=None)
    ap.add_argument("--dftimit", default=None)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--hw-train", default=None,
                    help="free text naming the training platform, e.g. "
                         "'an Apple M2 Pro GPU (MPS, PyTorch 2.13.0)'")
    ap.add_argument("--out-tex", default=str(HERE.parent / "paper" / "results_suite.tex"))
    ap.add_argument("--out-json", default=str(HERE.parent / "results" / "suite_summary.json"))
    args = ap.parse_args()

    suite = json.loads(Path(args.suite).read_text()) if Path(args.suite).exists() else {}
    runs = json.loads(Path(args.results).read_text()) if Path(args.results).exists() else {}
    seeds = [int(s) for s in args.seeds.split(",")]
    M: dict[str, str] = {}
    summary: dict = {}

    # ---- training-record metrics: ab.<variant>.<metric> (mean +- sd over seeds)
    variants = {"full": "full", "audio": "audio", "visual": "visual", "concat": "concat",
                "v2a": "v2a", "a2v": "a2v", "bilstm": "bilstm"}
    metric_path = {"f1": ("clip", "f1"), "auc": ("clip", "auc"),
                   "ap05": ("localization", "AP@0.5"), "ap075": ("localization", "AP@0.75"),
                   "map": ("localization", "mAP")}
    for tag, run_prefix in list(variants.items()) + [("lr-visual", "lipread_visual"), ("lr-randvisual", "lipread_rand_visual"),
                                                     ("lr-full", "lipread_full")]:
        recs = [runs.get(f"run::{run_prefix}_s{s}") for s in seeds]
        recs = [r for r in recs if r]
        if not recs:
            continue
        key = tag.replace("lr-", "lr.") if tag.startswith("lr-") else f"ab.{tag}"
        for m, (a, b) in metric_path.items():
            M[f"{key}.{m}"] = ms([r["test"][a][b] for r in recs])
        M[f"{key}.nseeds"] = str(len(recs))
        if key.startswith("lr."):
            M["lr.seeds"] = ", ".join(str(r["seed"]) for r in recs)
        M[f"{key}.params"] = f"{recs[0]['params'] / 1e6:.2f}"
        summary[key] = {"seeds": [r["seed"] for r in recs], "device": recs[0].get("device"),
                        "torch": recs[0].get("torch_version")}
    full_f1 = [runs[f"run::full_s{s}"]["test"]["clip"]["f1"] for s in seeds if f"run::full_s{s}" in runs]
    if len(full_f1) > 1:
        M["ms.full.f1.range"] = f"{(max(full_f1) - min(full_f1)) * 100:.1f}"

    # ---- per-seed full vs audio-only table and paired test (seed = block)
    from scipy import stats
    paired = [s for s in seeds if f"run::full_s{s}" in runs and f"run::audio_s{s}" in runs]
    for tag in ("full", "audio"):
        for s in paired:
            t = runs[f"run::{tag}_s{s}"]["test"]
            M[f"ms.{tag}.f1.s{s}"] = f"{t['clip']['f1']:.3f}"
            M[f"ms.{tag}.auc.s{s}"] = f"{t['clip']['auc']:.3f}"
            M[f"ms.{tag}.ap05.s{s}"] = f"{t['localization']['AP@0.5']:.3f}"
        M[f"ms.{tag}.f1"] = M.get(f"ab.{tag}.f1")
        M[f"ms.{tag}.auc"] = M.get(f"ab.{tag}.auc")
        M[f"ms.{tag}.ap05"] = M.get(f"ab.{tag}.ap05")
    M["ms.full.nseeds"] = str(len(paired))
    M["ms.full.seeds"] = ", ".join(str(s) for s in paired)
    if len(paired) >= 3:
        for m, (a, b) in (("f1", ("clip", "f1")), ("auc", ("clip", "auc")),
                          ("ap05", ("localization", "AP@0.5"))):
            d = np.array([runs[f"run::audio_s{s}"]["test"][a][b] - runs[f"run::full_s{s}"]["test"][a][b]
                          for s in paired])
            n = len(d); se = d.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            lo, hi = d.mean() - tcrit * se, d.mean() + tcrit * se
            p = stats.ttest_rel([runs[f"run::audio_s{s}"]["test"][a][b] for s in paired],
                                [runs[f"run::full_s{s}"]["test"][a][b] for s in paired]).pvalue
            M[f"ms.audio-vs-full.{m}.diff"] = f"{d.mean():+.3f}"
            M[f"ms.audio-vs-full.{m}.ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
            M[f"ms.audio-vs-full.{m}.p"] = f"{p:.3f}"
            M[f"ms.audio-vs-full.{m}.dz"] = f"{d.mean() / d.std(ddof=1):.2f}" if d.std(ddof=1) > 0 else "--"
            if m == "f1":   # plain-language bounds used in the text, in F1 points
                M["ms.full-adv.max"] = f"{max(0.0, -lo) * 100:.1f}"
                M["ms.audio-adv.max"] = f"{max(0.0, hi) * 100:.1f}"
            summary.setdefault("paired", {})[m] = dict(diff=float(d.mean()), ci=[float(lo), float(hi)],
                                                        p=float(p), per_seed=d.tolist())

    # ---- LRW-encoder full model vs audio-only on their shared seeds
    lr_paired = [s for s in seeds if f"run::lipread_full_s{s}" in runs and f"run::audio_s{s}" in runs]
    if len(lr_paired) >= 3:
        a_ = [runs[f"run::audio_s{s}"]["test"]["clip"]["f1"] for s in lr_paired]
        l_ = [runs[f"run::lipread_full_s{s}"]["test"]["clip"]["f1"] for s in lr_paired]
        d = np.array(l_) - np.array(a_)
        tcrit = stats.t.ppf(0.975, len(d) - 1); se = d.std(ddof=1) / np.sqrt(len(d))
        M["lr.full-vs-audio.f1.diff"] = f"{d.mean():+.3f}"
        M["lr.full-vs-audio.f1.ci"] = f"[{d.mean() - tcrit * se:+.3f}, {d.mean() + tcrit * se:+.3f}]"
        M["lr.full-vs-audio.f1.p"] = f"{stats.ttest_rel(l_, a_).pvalue:.2f}"
        dn = -d   # audio-only minus full: the sign convention of the five-seed comparison
        M["lr.audio-vs-full.f1.diff"] = f"{dn.mean():+.3f}"
        M["lr.audio-vs-full.f1.ci"] = f"[{dn.mean() - tcrit * se:+.3f}, {dn.mean() + tcrit * se:+.3f}]"
        M["lr.audio-vs-full.f1.p"] = M["lr.full-vs-audio.f1.p"]
        M["lr.audio-same-seeds.f1"] = ms(a_)

    # ---- training platform, from the run records themselves
    if paired:
        r0 = runs[f"run::full_s{paired[0]}"]
        devs = {runs[k].get("device") for k in runs if k.startswith("run::")}
        tvs = {runs[k].get("torch_version") for k in runs if k.startswith("run::")}
        summary["train_platform"] = {"devices": sorted(map(str, devs)), "torch": sorted(map(str, tvs)),
                                     "rgb_norm": r0.get("rgb_norm")}

    # ---- split independence
    sc = Path(args.split_check)
    if sc.exists():
        r = json.loads(sc.read_text())
        ok = r["sources_shared_across_splits"] == 0 and not r["problems"]
        M["split.check"] = (f"{sum(r['n_sources_by_split'].values())} source videos, none shared across "
                            f"partitions; all {r['n_fake_videos_with_source']} manipulated videos derive "
                            f"from a source in their own partition" if ok else "CHECK FAILED -- see split_check.json")

    # ---- LAV-DF per-bucket (in-domain), counterfactuals, attention
    per = defaultdict(list)
    cf = defaultdict(lambda: defaultdict(list))
    att = defaultdict(lambda: defaultdict(list))
    cf_lr = defaultdict(lambda: defaultdict(list))
    if args.lavdf:
        lav = clip_table(args.lavdf, "test")
        groups = lavdf_groups(lav)
        per = defaultdict(list)                       # model -> list of category stats
        cf = defaultdict(lambda: defaultdict(list))   # cond -> metric -> values
        att = defaultdict(lambda: defaultdict(list))  # group -> metric -> per-seed means
        for s in seeds:
            for tag in ("full", "audio", "visual", "lipread_full", "lipread_visual", "lipread_rand_visual"):
                run = suite.get(f"{tag}_s{s}")
                if not run or "clean" not in run:
                    continue
                check(lav, run["clean"]["labels"], f"{tag}_s{s} clean")
                sc = np.asarray(run["clean"]["scores"])
                per[tag].append(category_stats(lav, sc, groups))
                if tag == "lipread_full":   # video-removal test for the LRW-encoder model
                    yl = lav["clip_label"].to_numpy()
                    for key, val in run.items():
                        if key.startswith("cf:video"):
                            s2 = np.asarray(val["scores"])
                            cf_lr[key[3:]]["df1"].append(f1_of(yl, s2) - f1_of(yl, sc))
                            cf_lr[key[3:]]["agree"].append(float(((s2 > TAU) == (sc > TAU)).mean()))
                if tag != "full":
                    continue
                y = lav["clip_label"].to_numpy()
                base = sc > TAU
                cf["clean"]["f1"].append(f1_of(y, sc))
                for key, val in run.items():
                    if not key.startswith("cf:"):
                        continue
                    cond = key[3:]
                    check(lav, val["labels"], f"{tag}_s{s} {key}")
                    s2 = np.asarray(val["scores"])
                    cf[cond]["f1"].append(f1_of(y, s2))
                    cf[cond]["df1"].append(f1_of(y, s2) - f1_of(y, sc))
                    cf[cond]["agree"].append(float(((s2 > TAU) == base).mean()))
                # attention on clean LAV-DF test clips
                if "D" in run["clean"]:
                    A = {k: np.asarray(run["clean"][k]) for k in ("c", "D", "dsig", "dabs")}
                    real = (lav["bucket"] == "real").to_numpy()
                    fake = lav["clip_label"].to_numpy() == 1
                    for gname, mask in (("lav-real", real), ("lav-fake", fake)):
                        for k in A:
                            att[gname][k].append(float(A[k][mask].mean()))
                for sh in (0, 4, 8):
                    v = run.get(f"attn:shift{sh}")
                    if v:
                        for k in ("c", "D", "dsig", "dabs"):
                            att[f"shift{sh}"][k].append(float(np.mean(v[k])))
        for tag, prefix in (("full", "lv.full"), ("audio", "lv.audio"), ("visual", "lv.visual"),
                            ("lipread_full", "lv.lrfull"), ("lipread_visual", "lv.lrvisual"),
                            ("lipread_rand_visual", "lv.lrrvisual")):
            aggregate(per[tag], prefix, M)
        summary["lavdf"] = {k: v for k, v in per.items()}
        for cond, d in cf.items():
            M[f"cf.{cond}.f1"] = ms(d["f1"])
            if d.get("df1"):
                M[f"cf.{cond}.df1"] = ms(d["df1"], signed=True)
                M[f"cf.{cond}.agree"] = f"{mean(d['agree']):.3f}"
        for cond, d in cf_lr.items():
            M[f"cf-lr.{cond}.df1"] = ms(d["df1"], signed=True)
            M[f"cf-lr.{cond}.agree"] = f"{mean(d['agree']):.3f}"
        if all(f"snr{x}" in cf for x in (30, 20, 10)):
            M["cf.snr.f1"] = " / ".join(f"{mean(cf[f'snr{x}']['f1']):.3f}" for x in (30, 20, 10))


    # ---- zero-shot corpora
    for cname, cache, gfun, prefix in (("favc", args.favc, favc_groups, "xd"),
                                       ("dftimit", args.dftimit, dft_groups, "dft")):
        if not cache:
            continue
        tab = clip_table(cache, "test")
        g = gfun(tab)
        for name, mask in g.items():
            M[f"{prefix}.n.{name}"] = str(int(mask.sum()))
        M[f"{prefix}.n.real"] = str(int((tab["bucket"] == "real").sum()))
        per_c = defaultdict(list)
        for s in seeds:
            for tag in ("full", "audio", "visual", "lipread_full", "lipread_visual", "lipread_rand_visual"):
                run = suite.get(f"{tag}_s{s}")
                if not run or f"xd:{cname}" not in run:
                    continue
                v = run[f"xd:{cname}"]
                check(tab, v["labels"], f"{tag}_s{s} {cname}")
                sc = np.asarray(v["scores"])
                per_c[tag].append(category_stats(tab, sc, g))
                if tag == "full" and cname == "favc" and "D" in v:
                    for gname, mask in (("favc-real", (tab["bucket"] == "real").to_numpy()),
                                        ("favc-ao", g["audio-only"]), ("favc-vo", g["visual-only"]),
                                        ("favc-both", g["both"])):
                        for k in ("c", "D", "dsig", "dabs"):
                            att[gname][k].append(float(np.asarray(v[k])[mask].mean()))
        for tag, short in (("full", "full"), ("audio", "audio"), ("visual", "visual"),
                           ("lipread_full", "lrfull"), ("lipread_visual", "lrvisual"),
                           ("lipread_rand_visual", "lrrvisual")):
            aggregate(per_c[tag], f"{prefix}.{short}", M)
        summary[cname] = dict(per_c)
    if args.dftimit:
        M["dft.n-fake"] = M.get("dft.n.fake", "??")
        M["dft.n-real"] = M.get("dft.n.real", "??")
        for t in ("full", "audio"):
            if f"dft.{t}.fake.auc" in M:
                M[f"dft.{t}.auc"] = M[f"dft.{t}.fake.auc"]
    if "xd.lrfull.visual-only.auc" in M:
        M["lr.full.favc-vo.auc"] = M["xd.lrfull.visual-only.auc"]

    if TABLE_SD:
        M["tab.crossdata.maxsd"] = f"{max(TABLE_SD):.3f}"
    if args.lavdf:
        review_extras(args, suite, runs, seeds, M)

    # ---- attention macros (ms for offsets)
    for gname, d in att.items():
        M[f"at.{gname}.c"] = f"{mean(d['c']):.3f}"
        M[f"at.{gname}.D"] = f"{mean(d['D']):.2f}"
        M[f"at.{gname}.dsig"] = f"{mean(d['dsig']) * 1000 + 0.0:+.0f}".replace("+0", "0").replace("-0", "0") if abs(mean(d['dsig']) * 1000) < 0.5 else f"{mean(d['dsig']) * 1000:+.0f}"
        M[f"at.{gname}.dabs"] = f"{mean(d['dabs']) * 1000:.0f}"
    summary["attention"] = {g: {k: mean(v) for k, v in d.items()} for g, d in att.items()}
    M["at.dmax"] = f"{81 / 7:.1f}--{81 / 6:.1f}"

    # ---- hardware and latency
    lp = Path(args.latency)
    if lp.exists():
        lat = json.loads(lp.read_text())
        hw = lat["hardware"]
        M["hw.cpu"] = f"an {hw.get('cpu', '?')} ({hw.get('model', '')}, {hw.get('ram_gb', '?')}\\,GB RAM)"
        M["hw.gpu"] = hw.get("gpu", "?").replace("integrated GPU (MPS)", "integrated GPU via PyTorch MPS")
        M["hw.sw"] = f"PyTorch {hw.get('torch')}, Python {hw.get('python')}, {hw.get('os', '').split('-arm64')[0].replace('macOS-', 'macOS ')}"
        for v, r in lat["variants"].items():
            M[f"lat.{v}.cpu"] = f"{r['cpu_ms_1thread']:.0f}" if r["cpu_ms_1thread"] >= 10 else f"{r['cpu_ms_1thread']:.1f}"
            if r.get("accel_ms") is not None:
                M[f"lat.{v}.gpu"] = f"{r['accel_ms']:.1f}"
    if args.hw_train:
        M["hw.train"] = args.hw_train
    elif summary.get("train_platform", {}).get("devices") == ["mps"] and lp.exists():
        tv = summary["train_platform"]["torch"]
        # Same machine as the latency measurements; worded so that it reads
        # correctly wherever it is used (Sect. 3 precedes the Hardware paragraph).
        M["hw.train"] = (f"the integrated GPU of an {hw.get('cpu', 'Apple')} "
                         f"(PyTorch {', '.join(tv)}, MPS back-end)")

    # ---- write
    M = {k: v for k, v in M.items() if v is not None}
    # Guard against stale numbers: results_auto.tex holds values from the
    # submitted version's runs. Every key the revised paper uses from the
    # superseded families (per-seed table, paired test) must come from the
    # retrained runs; until it does, force it to a red ?? rather than let the
    # old value show through.
    # (Only applies next to the manuscript; the public code release has no paper/.)
    paper = HERE.parent / "paper" / "main_rev.tex"
    tex = paper.read_text() if paper.exists() else ""
    used = set(re.findall(r"\\R\{([^}]+)\}", tex))
    stale_prefixes = ("ms.",)
    for k in sorted(used):
        if k.startswith(stale_prefixes) and k not in M:
            M[k] = "\\textcolor{red}{\\textbf{??}}"
    # Typeset negative numbers with a true minus sign.
    M = {k: re.sub(r"(?<![\w$-])-(?=\d)", r"$-$", v) for k, v in M.items()}
    lines = ["% AUTO-GENERATED by code/analyse_suite.py -- do not edit by hand.",
             f"% Sources: {Path(args.suite).name}, {Path(args.results).name}, {lp.name}"]
    for k in sorted(M):
        lines.append(f"\\setres{{{k}}}{{{M[k]}}}")
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(lines) + "\n")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {len(M)} macros -> {args.out_tex}")


if __name__ == "__main__":
    main()
