"""
ASTRA-DF :: source-video independence of the LAV-DF subset partitions
=====================================================================

Answers Reviewer B (minor 3): can a source video, or any manipulated clip
derived from it, appear in more than one of train / val / test?

LAV-DF's metadata.json gives, for every video, its official split and -- for
manipulated videos -- the ``original`` genuine video it was generated from.
For each video in our subset cache this script:

  1. checks that the cache split equals the official split (dev -> val);
  2. maps the video to its *source* (itself if genuine, else ``original``);
  3. checks that the source's own official split equals the video's split;
  4. checks that no source is shared across partitions of the subset.

Exit status is non-zero if any check fails, so it can gate the paper build.

    python check_splits.py --metadata ../assets/lavdf_metadata.json \
        --cache ../assets/astra_cache --out ../results/split_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def stem(p: str | None) -> str | None:
    return Path(p).stem if p else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="LAV-DF metadata.json")
    ap.add_argument("--cache", required=True, help="subset cache with videos.csv")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "results" / "split_check.json"))
    args = ap.parse_args()

    meta = json.loads(Path(args.metadata).read_text())
    by_id = {stem(e["file"]): e for e in meta}
    norm = {"dev": "val"}

    df = pd.read_csv(Path(args.cache) / "videos.csv", dtype={"video_id": str})
    problems = defaultdict(list)
    sources = defaultdict(set)                 # split -> source ids
    n_fake_with_source = 0

    for _, r in df.iterrows():
        vid = str(r["video_id"]).zfill(6) if str(r["video_id"]).isdigit() else str(r["video_id"])
        e = by_id.get(vid) or by_id.get(str(r["video_id"]))
        if e is None:
            problems["not_in_metadata"].append(vid)
            continue
        official = norm.get(e["split"], e["split"])
        if official != r["split"]:
            problems["split_differs_from_official"].append(vid)
        src = stem(e.get("original")) if int(r["label"]) == 1 else vid
        if int(r["label"]) == 1:
            if not src:
                problems["fake_without_original"].append(vid)
                continue
            n_fake_with_source += 1
            se = by_id.get(src)
            if se is None:
                problems["original_not_in_metadata"].append(src)
            elif norm.get(se["split"], se["split"]) != r["split"]:
                problems["source_in_other_split"].append(f"{vid}<-{src}")
        sources[r["split"]].add(src)

    splits = sorted(sources)
    shared = {f"{a}&{b}": sorted(sources[a] & sources[b])
              for i, a in enumerate(splits) for b in splits[i + 1:]}
    n_shared = sum(len(v) for v in shared.values())

    # Whole-dataset sanity: does ANY LAV-DF fake have its original in another split?
    cross_global = sum(1 for e in meta if e.get("original")
                       and stem(e["original"]) in by_id
                       and by_id[stem(e["original"])]["split"] != e["split"])

    report = {
        "n_videos": int(len(df)),
        "by_split": df["split"].value_counts().to_dict(),
        "n_fake_videos_with_source": n_fake_with_source,
        "n_sources_by_split": {k: len(v) for k, v in sources.items()},
        "sources_shared_across_splits": n_shared,
        "shared_examples": {k: v[:10] for k, v in shared.items() if v},
        "problems": {k: {"count": len(v), "examples": v[:10]} for k, v in problems.items()},
        "lavdf_global_fakes_with_original_in_other_split": cross_global,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = n_shared == 0 and not problems.get("source_in_other_split") \
        and not problems.get("split_differs_from_official")
    print("\nRESULT:", "PASS -- partitions are source-video independent" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
