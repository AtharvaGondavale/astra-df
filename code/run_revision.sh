#!/usr/bin/env bash
# ==========================================================================
# ASTRA-DF revision training + evaluation queue (IEICE 2026DKP0010)
# ==========================================================================
# Every model in the revised paper is (re)trained here, on ONE platform, with
# the corrected visual input (--rgb-norm: RGB, ImageNet-normalised frames).
# Order is seed-major so that complete seeds become available early:
#
#   for each seed: full, audio-only, visual-only, concat, V->A, A->V, BiLSTM
#   then        : lipreading-pretrained encoder (visual-only, full)
#   after each  : evaluation-only suite on the new checkpoints
#
# Resumable: finished runs are skipped (--skip-existing on results_rev.json,
# and eval_suite skips finished (run, evaluation) pairs). Re-run the script to
# continue after any interruption. Wrapped in `caffeinate -is` so the Mac does
# not idle-sleep while it runs (closing the lid still sleeps it).
#
#   nohup ./run_revision.sh > ../results/rev/queue.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="$HERE/../.venv/bin/python"
A="$HERE/../assets"
OUT="$HERE/../results/rev/results_rev.json"     # checkpoints -> results/rev/checkpoints
CK="$HERE/../results/rev/checkpoints"
EPOCHS="${EPOCHS:-15}"
WORKERS="${WORKERS:-2}"
# Effective batch is 32 everywhere. On an 8 GB Apple M3 a 32-clip batch needs
# ~4.7 GB of unified memory, so it is split into micro-batches by gradient
# accumulation: 2 x 16 for MobileNet/audio variants, 4 x 8 for the lipreading
# encoder (ResNet-18 per frame).
ACCUM="${ACCUM:-2}"
LR_ACCUM="${LR_ACCUM:-4}"
mkdir -p "$CK"

train() {   # train <stages> <seed> <accum> [extra args]
  "$PY" run_experiments.py --cache "$A/astra_cache" --repo "$HERE" --out "$OUT" \
      --epochs "$EPOCHS" --batch-size 32 --accum "$3" --workers "$WORKERS" --skip-existing \
      --rgb-norm --stages "$1" --seeds "$2" "${@:4}"
}

evaluate() {  # evaluate <comma-separated run names>
  local extra=()   # array, not $(...): the project path contains spaces
  [ -f "$A/timit_cache/videos.csv" ] && extra=(--dftimit "$A/timit_cache")
  "$PY" eval_suite.py --ckpt-dir "$CK" --runs "$1" --lavdf "$A/astra_cache" \
      --favc "$A/favc_cache" "${extra[@]}" \
      --out "$HERE/../results/rev/eval_suite.json" --workers 2
}

stamp() { printf '\n##### %s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

seed_block() {
  local s="$1"
  stamp "seed $s: full + audio-only"
  train multiseed "$s" "$ACCUM"
  stamp "seed $s: visual-only, concat, v2a, a2v, bilstm"
  train ablseeds "$s" "$ACCUM"
  stamp "seed $s: evaluation suite"
  evaluate "full_s$s,audio_s$s,visual_s$s,concat_s$s,v2a_s$s,a2v_s$s,bilstm_s$s"
}

lip_block() {
  local s="$1"
  stamp "seed $s: lipreading-pretrained encoder"
  train lipreadseeds "$s" "$LR_ACCUM" --lipread-ckpt "$A/lrw_resnet18.pth"
  evaluate "lipread_visual_s$s,lipread_full_s$s"
}

main() {
  # Ordered so that a complete 3-seed version of every result exists early.
  for s in 42 43 44; do seed_block "$s"; done
  lip_block 42
  for s in 45 46; do seed_block "$s"; done
  lip_block 43
  lip_block 44
  stamp "queue complete"
}

if [ "${1:-}" = "--inner" ]; then
  main
else
  exec caffeinate -is "$0" --inner
fi
