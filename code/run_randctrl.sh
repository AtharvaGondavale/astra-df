#!/usr/bin/env bash
# Capacity control for the lipreading experiment (external review): the LRW
# encoder architecture trained from random initialisation, visual-only, seeds
# 42-44, same schedule as the LRW runs (effective batch 32 = 4 x 8). Resumable.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PY="$HERE/../.venv/bin/python"; A="$HERE/../assets"
OUT="$HERE/../results/rev/results_rev.json"; CK="$HERE/../results/rev/checkpoints"
main() {
  for s in 42 43 44; do
    printf '\n##### %s  seed %s: random-init lipreading encoder (visual only)\n' "$(date '+%F %T')" "$s"
    "$PY" run_experiments.py --cache "$A/astra_cache" --repo "$HERE" --out "$OUT" \
        --epochs 15 --batch-size 32 --accum 4 --workers 2 --skip-existing \
        --stages lipreadrandom --seeds "$s"
    extra=(); [ -f "$A/timit_cache/videos.csv" ] && extra=(--dftimit "$A/timit_cache")
    "$PY" eval_suite.py --ckpt-dir "$CK" --runs "lipread_rand_visual_s$s" --lavdf "$A/astra_cache" \
        --favc "$A/favc_cache" "${extra[@]}" --out "$HERE/../results/rev/eval_suite.json" --workers 2
  done
  printf '\n##### %s  control complete\n' "$(date '+%F %T')"
}
if [ "${1:-}" = "--inner" ]; then main; else exec caffeinate -is "$0" --inner; fi
