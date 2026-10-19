"""Evaluation script — clip-level metrics + temporal localization AP@tIoU."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from astra_df.data.lavdf_dataset import LAVDFDataset, collate_fn
from astra_df.models import build_model
from astra_df.utils import binary_metrics, mean_ap_at_ious, get_logger
from astra_df.training.train import load_config


log = get_logger("evaluate")


@torch.no_grad()
def evaluate(cfg_path: str, ckpt_path: str | None, split: str = "test") -> dict:
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LAVDFDataset(cfg["data"]["metadata_csv"], lavdf_root=cfg["data"].get("root"), split=split)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"],
                        shuffle=False, num_workers=cfg["data"]["num_workers"],
                        collate_fn=collate_fn)
    model = build_model(cfg).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    model.eval()

    labels, preds, scores = [], [], []
    # video_id -> list of (start, end, score)
    pred_intervals: dict[str, list] = defaultdict(list)
    gt_intervals: dict[str, set] = defaultdict(set)

    for batch in loader:
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        out = model(video, audio)
        prob = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        pred = (prob > 0.5).astype(int)
        labels.extend(batch["label"].numpy().tolist())
        preds.extend(pred.tolist())
        scores.extend(prob.tolist())

        intervals = out.interval.cpu().numpy()  # (B, 2) in [0, 1]
        for i in range(intervals.shape[0]):
            vid = batch["video_id"][i]
            s_sec = float(batch["start_sec"][i])
            e_sec = float(batch["end_sec"][i])
            clip_len = e_sec - s_sec
            pred_s = s_sec + intervals[i, 0] * clip_len
            pred_e = s_sec + intervals[i, 1] * clip_len
            pred_intervals[vid].append((float(pred_s), float(pred_e), float(prob[i])))
            gt_iv = batch["interval"][i].numpy()
            if gt_iv[0] >= 0:
                gt_s = s_sec + gt_iv[0] * clip_len
                gt_e = s_sec + gt_iv[1] * clip_len
                gt_intervals[vid].add((float(gt_s), float(gt_e)))

    clip_metrics = binary_metrics(np.array(labels), np.array(preds), np.array(scores))

    # Localization: aggregate per-video predictions, score-thresholded.
    all_preds, all_gts = [], []
    for vid, ivs in pred_intervals.items():
        # Keep only confidently-fake predictions for localization counting.
        for s, e, sc in ivs:
            if sc > 0.5 and e > s:
                all_preds.append((s, e, sc))
        for s, e in gt_intervals.get(vid, []):
            all_gts.append((s, e))
    loc_metrics = mean_ap_at_ious(all_preds, all_gts, ious=tuple(cfg["eval"]["localization_iou"]))

    summary = {"clip": clip_metrics, "localization": loc_metrics}
    log.info("Evaluation summary: %s", json.dumps(summary, indent=2))
    return summary


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    res = evaluate(args.config, args.ckpt, args.split)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    _cli()
