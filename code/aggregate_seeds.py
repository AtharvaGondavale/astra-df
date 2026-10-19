"""
ASTRA-DF :: multi-seed aggregation and significance testing
===========================================================

Consumes the ``run::full_s<seed>`` / ``run::audio_s<seed>`` records written by
``run_experiments.py --stages multiseed`` and produces:

    seed_stats.json     mean/std per variant + paired and Welch tests
    experiment_log.csv  one row per run: seed, config, checkpoint, metrics,
                        wall-clock, hardware, git commit  (reviewer item 12)

Why a *paired* test is the primary one
--------------------------------------
Both variants are trained on the identical seed, hence identical data order and
identical initialisation of every shared module. The seed is therefore a
blocking factor, and pairing removes the between-seed variance that would
otherwise dominate a 5-sample comparison. Welch's unpaired test is reported
alongside as the conservative fallback; if the two disagree, say so in the
paper rather than quoting whichever is smaller.

With n=5 the power is low. A non-significant result here means "this experiment
cannot resolve a difference of this size", NOT "the variants are equivalent".
The paper must use the former phrasing.

    python aggregate_seeds.py --results results.json --out-dir .
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path

METRICS = ("f1", "auc", "ap0.5")


# --------------------------------------------------------------------------
# statistics (no scipy dependency)
# --------------------------------------------------------------------------
def _betacf(a, b, x, itmax=200, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta function I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """Two-sided p-value for Student's t."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def mean(xs):
    return sum(xs) / len(xs)


def std(xs, ddof=1):
    """Sample standard deviation. ddof=1 because these seeds are a sample of
    the seed distribution, not the whole population."""
    n = len(xs)
    if n - ddof <= 0:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def paired_t(a, b):
    """Paired t-test on a-b. Returns (t, df, p, mean_diff, ci95_lo, ci95_hi)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    md, sd = mean(d), std(d)
    if n < 2 or sd == 0 or not math.isfinite(sd):
        return dict(t=None, df=n - 1, p=None, mean_diff=md,
                    ci95=[None, None],
                    note="zero variance or n<2; test undefined")
    se = sd / math.sqrt(n)
    t = md / se
    p = t_sf_two_sided(t, n - 1)
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
             5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    return dict(t=t, df=n - 1, p=p, mean_diff=md,
                ci95=[md - tcrit * se, md + tcrit * se], note=None)


def welch_t(a, b):
    na, nb = len(a), len(b)
    va, vb = std(a) ** 2, std(b) ** 2
    if na < 2 or nb < 2:
        return dict(t=None, df=None, p=None)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return dict(t=None, df=None, p=None, note="zero variance")
    t = (mean(a) - mean(b)) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return dict(t=t, df=df, p=t_sf_two_sided(t, df))


def cohens_dz(a, b):
    """Paired effect size."""
    d = [x - y for x, y in zip(a, b)]
    s = std(d)
    return mean(d) / s if s and math.isfinite(s) and s > 0 else None


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def metrics_of(rec):
    clip = rec.get("test", {}).get("clip", {})
    loc = rec.get("test", {}).get("localization", {})
    return {"f1": clip.get("f1"), "auc": clip.get("auc"),
            "ap0.5": loc.get("AP@0.5"),
            "accuracy": clip.get("accuracy"),
            "precision": clip.get("precision"),
            "recall": clip.get("recall")}


def collect(data):
    """variant -> {seed: record}. Recognises full_s42 / audio_s42 naming."""
    out: dict[str, dict[int, dict]] = {}
    pat = re.compile(r"^(?P<variant>.+)_s(?P<seed>\d+)$")
    for k, v in data.items():
        if not k.startswith("run::"):
            continue
        m = pat.match(k[5:])
        if not m:
            continue
        out.setdefault(m.group("variant"), {})[int(m.group("seed"))] = v
    return out


def git_commit(repo="."):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
def build(results_path, out_dir, repo="."):
    data = json.load(open(results_path))
    variants = collect(data)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not variants:
        print("No multi-seed runs found (expected run::full_s42 etc.).")
        print("Run:  python run_experiments.py --cache <cache> "
              "--stages multiseed --seeds 42,43,44,45,46")
        return None

    commit = git_commit(repo)
    summary = {"source": str(results_path), "git_commit": commit, "variants": {}}

    for var, by_seed in sorted(variants.items()):
        seeds = sorted(by_seed)
        per_seed = {s: metrics_of(by_seed[s]) for s in seeds}
        stats = {}
        for m in METRICS:
            vals = [per_seed[s][m] for s in seeds if per_seed[s][m] is not None]
            if not vals:
                continue
            stats[m] = {"n": len(vals), "mean": mean(vals),
                        "std": std(vals) if len(vals) > 1 else None,
                        "min": min(vals), "max": max(vals),
                        "values": vals}
        summary["variants"][var] = {"seeds": seeds, "per_seed": per_seed,
                                    "stats": stats}

    # ---- paired comparisons between every pair of variants on shared seeds --
    summary["comparisons"] = {}
    names = sorted(variants)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(variants[a]) & set(variants[b]))
            if len(shared) < 2:
                continue
            cmp_rec = {"seeds": shared, "n": len(shared)}

            # Runs may legitimately be split across machines (Colab quota,
            # overnight local runs). That is safe for the PAIRED test only if
            # both variants of a given seed ran on the same accelerator, so the
            # device effect cancels within the pair. Verify rather than assume:
            # a mismatch here silently reintroduces the hardware confound.
            mixed = [s for s in shared
                     if variants[a][s].get("device") != variants[b][s].get("device")]
            devs = sorted({variants[a][s].get("device") for s in shared} |
                          {variants[b][s].get("device") for s in shared})
            cmp_rec["devices"] = devs
            cmp_rec["pairs_with_mixed_devices"] = mixed
            if mixed:
                print(f"\n  !! {a} vs {b}: seeds {mixed} have their two variants "
                      "on DIFFERENT devices. The paired difference for those "
                      "seeds confounds hardware with modality -- rerun them on "
                      "one machine before quoting the test.")
            elif len(devs) > 1:
                print(f"\n  note: {a} vs {b} spans devices {devs}, but each seed's "
                      "pair shares one device, so paired differences are clean. "
                      "Marginal means mix hardware; quote the paired CI, not the "
                      "raw per-variant means.")
            for m in METRICS:
                va = [metrics_of(variants[a][s])[m] for s in shared]
                vb = [metrics_of(variants[b][s])[m] for s in shared]
                if any(x is None for x in va + vb):
                    continue
                cmp_rec[m] = {
                    f"{a}_mean": mean(va), f"{a}_std": std(va),
                    f"{b}_mean": mean(vb), f"{b}_std": std(vb),
                    "paired": paired_t(va, vb),
                    "welch": welch_t(va, vb),
                    "cohens_dz": cohens_dz(va, vb),
                }
            summary["comparisons"][f"{a}_vs_{b}"] = cmp_rec

    (out_dir / "seed_stats.json").write_text(json.dumps(summary, indent=2))

    # ---- experiment log: every run in results.json, not just multi-seed ----
    stat = data.get("stats", {})
    rows = []
    for k, v in sorted(data.items()):
        if not k.startswith("run::"):
            continue
        m = metrics_of(v)
        rows.append({
            "run": v.get("name", k[5:]),
            "seed": v.get("seed", ""),
            "backbone": v.get("backbone", ""),
            "fusion": v.get("fusion", ""),
            "modalities": v.get("modalities", ""),
            "pretrained": v.get("pretrained", ""),
            "epochs": v.get("epochs", ""),
            "batch_size": v.get("batch_size", ""),
            "lr": v.get("lr", ""),
            "clip_sec": v.get("clip_sec", ""),
            "roi": v.get("roi", ""),
            "loss_weights": "|".join(str(x) for x in v.get("loss_weights", [])),
            "params_M": round(v["params"] / 1e6, 3) if v.get("params") else "",
            "n_train": v.get("n_train", ""), "n_val": v.get("n_val", ""),
            "n_test": v.get("n_test", ""),
            "best_val_f1": round(v["best_val_f1"], 5) if v.get("best_val_f1") is not None else "",
            "test_f1": round(m["f1"], 5) if m["f1"] is not None else "",
            "test_auc": round(m["auc"], 5) if m["auc"] is not None else "",
            "test_ap0.5": round(m["ap0.5"], 5) if m["ap0.5"] is not None else "",
            "minutes": v.get("minutes", ""),
            "device": v.get("device", ""),
            "torch": v.get("torch_version", ""),
            "checkpoint": f"checkpoints/{v.get('name', k[5:])}.pt",
            "git_commit": commit or "",
        })
    if rows:
        with open(out_dir / "experiment_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- console report -------------------------------------------------
    print(f"\n{'='*72}\nMULTI-SEED SUMMARY   (git {commit or 'n/a'})\n{'='*72}")
    for var, rec in summary["variants"].items():
        print(f"\n{var}   seeds={rec['seeds']}")
        print(f"  {'seed':<6}{'F1':>10}{'AUC':>10}{'AP@0.5':>10}")
        for s in rec["seeds"]:
            p = rec["per_seed"][s]
            def f(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "  n/a"
            print(f"  {s:<6}{f(p['f1']):>10}{f(p['auc']):>10}{f(p['ap0.5']):>10}")
        line = "  mean  "
        for m in METRICS:
            st = rec["stats"].get(m)
            line += (f"{st['mean']:.4f}".rjust(10) if st else "n/a".rjust(10))
        print(line)
        line = "  std   "
        for m in METRICS:
            st = rec["stats"].get(m)
            sd = st and st.get("std")
            line += (f"{sd:.4f}".rjust(10) if sd is not None else "n/a".rjust(10))
        print(line)

    for name, c in summary["comparisons"].items():
        print(f"\n--- {name}  (paired, n={c['n']}) ---")
        for m in METRICS:
            if m not in c:
                continue
            d = c[m]
            pt = d["paired"]
            p = pt["p"]
            ci = pt["ci95"]
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    if ci[0] is not None else "n/a")
            dz = d["cohens_dz"]
            p_s = "n/a" if p is None else f"{p:.4f}"
            dz_s = "n/a" if dz is None else f"{dz:.2f}"
            print(f"  {m:<8} diff={pt['mean_diff']:+.4f}  95%CI={ci_s}  "
                  f"p={p_s}  dz={dz_s}")

    print(f"\nwrote {out_dir/'seed_stats.json'}")
    print(f"wrote {out_dir/'experiment_log.csv'}  ({len(rows)} runs)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    build(args.results, args.out_dir, args.repo)


if __name__ == "__main__":
    main()
