"""
ASTRA-DF :: setup smoke test
============================

Run this BEFORE starting any experiment. It checks, in a few seconds, the
things that otherwise fail hours into a run:

  * every Python dependency imports
  * mediapipe exposes `solutions` (1.0.0 removed it; pin 0.10.13)
  * ffmpeg is on PATH
  * the astra_df package resolves
  * the model builds and a forward pass produces the expected shapes
  * an accelerator is visible, and which one

    python verify_setup.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ok = True


def check(label, fn):
    global ok
    try:
        detail = fn()
        print(f"  [ok]   {label}" + (f"  ({detail})" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {label}\n           {type(exc).__name__}: {exc}")


print("ASTRA-DF setup check\n")

def _torch():
    import torch
    return f"torch {torch.__version__}"

def _accel():
    import torch
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "CPU ONLY -- training will be very slow"

def _mediapipe():
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            f"mediapipe {mp.__version__} has no `solutions` module. "
            "Pin mediapipe==0.10.13 and protobuf==4.25.3.")
    return f"mediapipe {mp.__version__}"

def _ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH (macOS: brew install ffmpeg)")
    return None

def _astra():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from astra_df.models.visual_encoder import VisualEncoder  # noqa: F401
    return "astra_df package resolves"

def _forward():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_re", Path(__file__).with_name("run_experiments.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.M = m._import_astra(str(Path(__file__).resolve().parent))
    net = m.VariantASTRA("transformer", "cross", "va", pretrained=False)
    out = net(torch.randn(2, 30, 3, 96, 96), torch.randn(2, 81, 13))
    assert out["logits"].shape == (2, 2), out["logits"].shape
    assert out["frame_sync"].shape[1] == 30, out["frame_sync"].shape
    return f"forward pass OK, {net.n_params()/1e6:.2f}M params"

for label, fn in [("PyTorch", _torch), ("accelerator", _accel),
                  ("numpy/pandas", lambda: __import__("pandas") and None),
                  ("opencv", lambda: __import__("cv2").__version__),
                  ("librosa", lambda: __import__("librosa").__version__),
                  ("matplotlib", lambda: __import__("matplotlib").__version__),
                  ("mediapipe", _mediapipe), ("ffmpeg", _ffmpeg),
                  ("astra_df package", _astra), ("model forward pass", _forward)]:
    check(label, fn)

print("\n" + ("All checks passed -- safe to run reproduce.sh"
              if ok else
              "Some checks FAILED -- fix these before running experiments"))
sys.exit(0 if ok else 1)
