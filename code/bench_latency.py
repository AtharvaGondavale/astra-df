"""
ASTRA-DF :: inference-cost benchmark with full hardware provenance
==================================================================

Answers the reviewer request "specific information of CPU and GPU is missing,
though processing time is shown in Table 3". Measures the same quantities as
``stage_stats`` in run_experiments.py (parameters, FLOPs, single-thread CPU
latency, accelerator latency) but needs no dataset, and records exactly which
hardware and software produced every number.

Protocol (identical to stage_stats): batch size 1, one 1-s clip
(video 30x3x96x96, MFCC 81x13), randomly initialised weights (latency does not
depend on weight values), CPU pinned to one thread with 10 warm-up and 30 timed
iterations; accelerator with 10 warm-up and 50 timed iterations and explicit
device synchronisation.

    python bench_latency.py --out ../results/latency.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as R  # noqa: E402

VARIANTS = {  # name -> (backbone, fusion, modalities)
    "transformer": ("transformer", "cross", "va"),
    "bilstm": ("bilstm", "cross", "va"),
    "audio-only": ("transformer", "cross", "a"),
    "visual-only": ("transformer", "cross", "v"),
    "fusion-concat": ("transformer", "concat", "va"),
}


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def hardware() -> dict:
    """Everything needed to state the measurement platform in the paper."""
    info = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads_default": torch.get_num_threads(),
    }
    if sys.platform == "darwin":
        info["cpu"] = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["cpu_cores_physical"] = _sh(["sysctl", "-n", "hw.physicalcpu"])
        info["cpu_cores_performance"] = _sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        info["ram_gb"] = round(int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1)
        info["model"] = _sh(["sysctl", "-n", "hw.model"])
    else:
        lscpu = _sh(["lscpu"])
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):
                info["cpu"] = line.split(":", 1)[1].strip()
        info["lscpu"] = lscpu
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    elif torch.backends.mps.is_available():
        info["gpu"] = f"Apple {info.get('cpu', '').replace('Apple ', '')} integrated GPU (MPS)"
    return info


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def bench(model: torch.nn.Module, dev: torch.device | None, warm: int, reps: int) -> float:
    v = torch.randn(1, 30, 3, 96, 96)
    a = torch.randn(1, 81, 13)
    if dev is not None:
        model, v, a = model.to(dev), v.to(dev), a.to(dev)
    with torch.no_grad():
        for _ in range(warm):
            model(v, a)
        if dev is not None:
            _sync(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(v, a)
        if dev is not None:
            _sync(dev)
    return (time.perf_counter() - t0) / reps * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "latency.json"))
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent repetitions of the whole timing loop; median reported")
    args = ap.parse_args()

    R.M.update(R._import_astra(str(HERE)))
    torch.manual_seed(0)
    acc = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if torch.backends.mps.is_available() else None)

    hw = hardware()
    print(json.dumps({k: v for k, v in hw.items() if k != "lscpu"}, indent=2))
    out = {"hardware": hw, "protocol": {
        "input": "batch 1, video 30x3x96x96, MFCC 81x13 (one 1-s clip)",
        "cpu": "torch.set_num_threads(1); 10 warm-up + 30 timed iterations",
        "accelerator": "10 warm-up + 50 timed iterations, explicit device synchronisation",
        "repeats": args.repeats, "statistic": "median over repeats"}, "variants": {}}

    for name, (bk, fu, mo) in VARIANTS.items():
        m = R.VariantASTRA(bk, fu, mo, pretrained=False).eval()
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu = sorted(bench(m, None, 10, 30) for _ in range(args.repeats))
        torch.set_num_threads(prev)
        accel = None
        if acc is not None:
            accel = sorted(bench(m, acc, 10, 50) for _ in range(args.repeats))
            m = m.cpu()
        flops = None
        try:
            from torch.utils.flop_counter import FlopCounterMode
            fc = FlopCounterMode(display=False)
            with fc, torch.no_grad():
                m(torch.randn(1, 30, 3, 96, 96), torch.randn(1, 81, 13))
            flops = fc.get_total_flops()
        except Exception as exc:  # noqa: BLE001
            print("  FLOP counter unavailable:", exc)
        rec = {"params": m.n_params(), "flops": flops,
               "cpu_ms_1thread": round(cpu[len(cpu) // 2], 2), "cpu_ms_all": [round(x, 2) for x in cpu],
               "accel_device": acc.type if acc else None,
               "accel_ms": round(accel[len(accel) // 2], 2) if accel else None,
               "accel_ms_all": [round(x, 2) for x in accel] if accel else None}
        out["variants"][name] = rec
        print(f"  {name:<14} {rec['params']/1e6:5.2f}M  {(flops or 0)/1e9:5.2f} GFLOPs  "
              f"CPU(1T) {rec['cpu_ms_1thread']:7.2f} ms  {rec['accel_device']} {rec['accel_ms']} ms")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
