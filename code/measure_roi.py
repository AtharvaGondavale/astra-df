"""
ASTRA-DF :: native mouth-ROI resolution measurement
===================================================

Establishes, from the source videos rather than by assertion, how many real
pixels the mouth region actually occupies. This settles two questions:

  1. Would a larger ROI help?  The paper previously speculated that a
     128x128 setting might recover visual performance. If the *native* crop is
     already smaller than 96x96, then both 96 and 128 are upsampling and the
     larger setting cannot add information -- the limit is the source video,
     not the ROI hyperparameter. This is a methodological limitation, not an
     unrun experiment.

  2. Is the LAV-DF -> FakeAVCeleb collapse explained by a resolution domain
     gap? If the two corpora have comparable native mouth resolution, that
     explanation is ruled out and the collapse must be attributed elsewhere.

Measures the lip-landmark bounding box with the same MediaPipe landmarks and
the same padding factor used by prepare_subset.py, so the reported crop side is
the one the pipeline actually extracts.

    python measure_roi.py --lavdf-root ~/Downloads/LAV-DF \
                          --favc-root ~/Downloads/FakeAVCeleb_v1.2
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_subset import BBOX_PAD, MOUTH_LANDMARKS  # noqa: E402

TARGET_ROI = 96


def measure(video_paths, n_max, seed=0):
    import mediapipe as mp

    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5)
    rng = np.random.RandomState(seed)
    paths = list(video_paths)
    rng.shuffle(paths)

    lip_w, lip_h, crop_side, srcs = [], [], [], []
    for v in paths:
        if len(crop_side) >= n_max:
            break
        cap = cv2.VideoCapture(v)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if N < 5:
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, N // 2)   # mid-video frame
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        xs = np.array([lm[i].x for i in MOUTH_LANDMARKS]) * W
        ys = np.array([lm[i].y for i in MOUTH_LANDMARKS]) * H
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        lip_w.append(w)
        lip_h.append(h)
        # prepare_subset takes a padded square around the landmark box
        crop_side.append(max(w, h) * (1 + 2 * BBOX_PAD))
        srcs.append(f"{W}x{H}")
    fm.close()

    if not crop_side:
        return None
    s = np.array(crop_side)
    return {
        "n_videos": len(s),
        "source_resolutions": sorted(set(srcs)),
        "lip_box_w_median_px": float(np.median(lip_w)),
        "lip_box_h_median_px": float(np.median(lip_h)),
        "crop_side_median_px": float(np.median(s)),
        "crop_side_p10_px": float(np.percentile(s, 10)),
        "crop_side_p90_px": float(np.percentile(s, 90)),
        "frac_upsampled_at_96": float((s < 96).mean()),
        "frac_upsampled_at_128": float((s < 128).mean()),
    }


def show(tag, r):
    if r is None:
        print(f"  {tag}: no faces detected / no videos found")
        return
    print(f"\n  {tag}")
    print(f"    videos measured      : {r['n_videos']}")
    print(f"    source resolution    : {', '.join(r['source_resolutions'][:4])}")
    print(f"    lip box (median)     : {r['lip_box_w_median_px']:.0f} x "
          f"{r['lip_box_h_median_px']:.0f} px")
    print(f"    extracted crop side  : {r['crop_side_median_px']:.0f} px "
          f"(p10 {r['crop_side_p10_px']:.0f}, p90 {r['crop_side_p90_px']:.0f})")
    print(f"    upsampled at  96x96  : {r['frac_upsampled_at_96']*100:.1f}% of crops")
    print(f"    upsampled at 128x128 : {r['frac_upsampled_at_128']*100:.1f}% of crops")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lavdf-root", default=None)
    ap.add_argument("--favc-root", default=None)
    ap.add_argument("--n", type=int, default=60, help="videos per corpus")
    ap.add_argument("--out", default="roi_native_size.json")
    args = ap.parse_args()

    out = {"target_roi": TARGET_ROI, "bbox_pad": BBOX_PAD}
    print("Native mouth-ROI resolution")

    if args.lavdf_root:
        vids = glob.glob(f"{args.lavdf_root}/train/*.mp4")
        out["LAV-DF"] = measure(vids, args.n)
        show("LAV-DF (train)", out["LAV-DF"])

    if args.favc_root:
        for cat in ("RealVideo-RealAudio", "FakeVideo-RealAudio"):
            vids = glob.glob(f"{args.favc_root}/{cat}/**/*.mp4", recursive=True)
            out[f"FakeAVCeleb/{cat}"] = measure(vids, min(args.n, 40))
            show(f"FakeAVCeleb / {cat}", out[f"FakeAVCeleb/{cat}"])

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
