"""Convert per-frame model outputs into a human-readable timeline.

The output is a list of ``{"start": s, "end": e, "score": p}`` dicts in
absolute seconds — i.e. the localised fake intervals that the Streamlit
demo and the paper's qualitative figures show.

Two paths into this module:

1. ``timeline_from_frames(frame_scores, frame_times, threshold, min_len)`` —
   threshold the per-frame fake probability and merge contiguous runs.
2. ``timeline_from_intervals(per_clip_intervals, scores)`` — collapse the
   model's regressed intervals (one per 1-second clip) into a smaller set
   of merged intervals after non-maximum suppression.

Both are used by ``astra_df/inference/predict.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass
class Interval:
    start: float
    end: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def timeline_from_frames(
    frame_scores: Iterable[float],
    frame_times: Iterable[float],
    threshold: float = 0.5,
    min_len_sec: float = 0.1,
) -> list[Interval]:
    scores = list(frame_scores)
    times = list(frame_times)
    if not scores:
        return []
    intervals: list[Interval] = []
    in_run = False
    run_start, run_scores = 0.0, []
    for s, t in zip(scores, times):
        if s >= threshold:
            if not in_run:
                in_run = True
                run_start = t
                run_scores = []
            run_scores.append(s)
        elif in_run:
            run_end = t
            if run_end - run_start >= min_len_sec:
                intervals.append(Interval(run_start, run_end, sum(run_scores) / len(run_scores)))
            in_run = False
    if in_run:
        run_end = times[-1]
        if run_end - run_start >= min_len_sec:
            intervals.append(Interval(run_start, run_end, sum(run_scores) / len(run_scores)))
    return intervals


def nms_intervals(intervals: list[Interval], iou_thresh: float = 0.3) -> list[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda i: -i.score)
    kept: list[Interval] = []
    for iv in intervals:
        if any(_iou(iv, k) > iou_thresh for k in kept):
            continue
        kept.append(iv)
    return sorted(kept, key=lambda i: i.start)


def _iou(a: Interval, b: Interval) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    inter = max(0.0, hi - lo)
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / max(1e-9, union)


def merge_close(intervals: list[Interval], gap_sec: float = 0.2) -> list[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda i: i.start)
    merged = [intervals[0]]
    for iv in intervals[1:]:
        if iv.start - merged[-1].end <= gap_sec:
            prev = merged[-1]
            merged[-1] = Interval(
                start=prev.start,
                end=max(prev.end, iv.end),
                score=(prev.score + iv.score) / 2,
            )
        else:
            merged.append(iv)
    return merged
