"""Height of the mouth band inside the letterboxed crops of a packed cache.

prepare_*.py resize the padded lip box with its aspect ratio kept onto a black
square canvas (resize_pad), so the mouth fills the canvas width but only part
of its height.  This measures that band (rows that are not black) in every
stored frame and reports it at the 96-pixel model input scale.

usage: python measure_mouth_band.py ../assets/astra_cache ../assets/favc_cache \
           --out ../results/mouth_band.json
"""
import argparse
import io
import json
import os

import numpy as np
from PIL import Image

ROI = 96            # model input size (PackedLAVDF resizes the stored crop to ROI x ROI)
BLACK = 20          # row mean below this counts as letterbox padding


def band_heights(cache: str):
    idx = np.load(os.path.join(cache, "index.npz"))
    off = idx["frame_offsets"]                  # n_frames + 1 byte offsets
    n_frames = int(idx["video_frame_ptr"][-1])
    with open(os.path.join(cache, "frames.bin"), "rb") as f:
        blob = f.read()
    heights = []
    for i in range(n_frames):
        img = np.asarray(Image.open(io.BytesIO(blob[off[i]:off[i + 1]])).convert("L"), dtype=np.float32)
        rows = np.where(img.mean(axis=1) > BLACK)[0]
        if len(rows):                       # all-black frames are missed detections
            heights.append((rows[-1] - rows[0] + 1) * ROI / img.shape[0])
    return np.asarray(heights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"roi": ROI, "black_threshold": BLACK}
    for cache in args.caches:
        h = band_heights(cache)
        out[os.path.basename(os.path.normpath(cache))] = {
            "n_frames": int(len(h)),
            "band_height_median_px": float(np.median(h)),
            "band_height_p10_px": float(np.percentile(h, 10)),
            "band_height_p90_px": float(np.percentile(h, 90)),
        }
        print(cache, out[os.path.basename(os.path.normpath(cache))])
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
