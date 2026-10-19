"""Classification + temporal-localization metrics."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Clip-level metrics
# ---------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
           "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    if y_score is not None:
        out["auc"] = _roc_auc(y_true, np.asarray(y_score))
    return out


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Plain Mann-Whitney U based AUC. Avoids the sklearn dep for portability."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


# ---------------------------------------------------------------------------
# Temporal localization (LAV-DF style)
# ---------------------------------------------------------------------------
def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(1e-9, union)


def average_precision_at_iou(
    pred_intervals: list[tuple[float, float, float]],  # (start, end, score)
    gt_intervals: list[tuple[float, float]],
    iou_thresh: float,
) -> float:
    """Standard AP at a fixed IoU threshold, video-pooled.

    Each prediction is matched greedily to the highest-IoU unmatched GT.
    Returns mean Average Precision across all predictions.
    """
    if not pred_intervals:
        return 0.0
    pred_sorted = sorted(pred_intervals, key=lambda x: -x[2])
    matched = [False] * len(gt_intervals)
    tps, fps = [], []
    for s, e, _ in pred_sorted:
        best_iou, best_idx = 0.0, -1
        for j, gt in enumerate(gt_intervals):
            if matched[j]:
                continue
            iou = temporal_iou((s, e), gt)
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_iou >= iou_thresh and best_idx >= 0:
            matched[best_idx] = True
            tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)

    tps_c = np.cumsum(tps); fps_c = np.cumsum(fps)
    n_gt = max(1, len(gt_intervals))
    recall = tps_c / n_gt
    precision = tps_c / np.maximum(1, tps_c + fps_c)
    # 11-point interpolation (Pascal VOC style)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(min(1.0, max(0.0, ap)))


def mean_ap_at_ious(predictions, ground_truth, ious=(0.5, 0.75, 0.95)) -> dict:
    """Aggregate AP across multiple IoU thresholds."""
    out = {}
    for iou in ious:
        out[f"AP@{iou}"] = average_precision_at_iou(predictions, ground_truth, iou)
    out["mAP"] = float(np.mean(list(out.values())))
    return out


# ---------------------------------------------------------------------------
# Running aggregator
# ---------------------------------------------------------------------------
class RunningMean:
    def __init__(self) -> None:
        self._sum: dict[str, float] = defaultdict(float)
        self._n: dict[str, int] = defaultdict(int)

    def add(self, **kv) -> None:
        for k, v in kv.items():
            v = float(v) if not isinstance(v, float) else v
            self._sum[k] += v
            self._n[k] += 1

    def mean(self) -> dict[str, float]:
        return {k: self._sum[k] / max(1, self._n[k]) for k in self._sum}

    def reset(self) -> None:
        self._sum.clear()
        self._n.clear()
