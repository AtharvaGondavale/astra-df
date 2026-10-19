#!/usr/bin/env bash
# ==========================================================================
# ASTRA-DF :: single reproducible pipeline
# ==========================================================================
# Reproduces every number in the paper from raw datasets. Each stage is
# independently resumable (--skip-existing), so an interrupted run continues
# rather than restarting.
#
#   ./reproduce.sh cache        build the LAV-DF cache        (~2-3 h, CPU)
#   ./reproduce.sh crossdata    build FakeAVCeleb + DFTIMIT   (~1 h, CPU)
#   ./reproduce.sh main         all single-seed experiments   (~6-8 h, GPU/MPS)
#   ./reproduce.sh multiseed    5-seed full vs audio-only     (~10-15 h, GPU/MPS)
#   ./reproduce.sh analyse      aggregate + figures + macros  (~2 min)
#   ./reproduce.sh all          everything, in order
#
# Edit the paths below to match your machine, then run. Hardware and wall-clock
# for the reported runs are recorded per-run in experiment_log.csv.
set -euo pipefail

# ---- paths (EDIT THESE) --------------------------------------------------
LAVDF_ROOT="${LAVDF_ROOT:-$HOME/Downloads/LAV-DF}"
FAVC_ROOT="${FAVC_ROOT:-$HOME/Downloads/FakeAVCeleb_v1.2}"
TIMIT_ROOT="${TIMIT_ROOT:-$HOME/Downloads/DeepfakeTIMIT}"
VIDTIMIT_ROOT="${VIDTIMIT_ROOT:-$HOME/Downloads/VidTIMIT}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The astra_df package ships inside this directory, so the repo root is here.
REPO="${REPO:-$HERE}"
CACHE="${CACHE:-$HERE/astra_cache}"
FAVC_CACHE="${FAVC_CACHE:-$HERE/favc_cache}"
TIMIT_CACHE="${TIMIT_CACHE:-$HERE/timit_cache}"
RESULTS="${RESULTS:-$HERE/results.json}"

N_VIDEOS="${N_VIDEOS:-5000}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"
SEEDS="${SEEDS:-42,43,44,45,46}"

COMMON=(--cache "$CACHE" --repo "$REPO" --out "$RESULTS"
        --epochs "$EPOCHS" --batch-size "$BATCH" --workers "$WORKERS"
        --skip-existing)

log() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

stage_cache() {
    log "Building LAV-DF cache -> $CACHE"
    python3 prepare_subset.py --root "$LAVDF_ROOT" --out "$CACHE" \
        --n-videos "$N_VIDEOS" --workers "$WORKERS"
}


stage_crossdata_cache() {
  log "Building FakeAVCeleb cache -> $FAVC_CACHE"
  python3 prepare_fakeavceleb.py --root "$FAVC_ROOT" --out "$FAVC_CACHE" \
      --per-category 500 --workers "$WORKERS"
  if [ -d "$TIMIT_ROOT" ]; then
    log "Building DeepfakeTIMIT cache -> $TIMIT_CACHE"
    python3 prepare_timit.py --dftimit "$TIMIT_ROOT" --vidtimit "$VIDTIMIT_ROOT" \
        --out "$TIMIT_CACHE" --workers "$WORKERS"
  else
    echo "  (DeepfakeTIMIT not found at $TIMIT_ROOT -- skipping)"
  fi
}

stage_main() {
  TIMIT_ARGS=(); [ -d "$TIMIT_CACHE" ] && TIMIT_ARGS=(--eval-cache "dftimit=$TIMIT_CACHE")
  log "Single-seed experiments (seed 42)"
  python3 run_experiments.py "${COMMON[@]}" --seed 42 \
      --stages stats,main,fusion,modality,lossweight,hparam,robust,gradcam,crossdata \
      --eval-cache "favc=$FAVC_CACHE" \
      "${TIMIT_ARGS[@]}"
}

stage_multiseed() {
  log "Multi-seed: full vs audio-only over seeds $SEEDS"
  echo "  NOTE: 10 training runs. Budget 10-15 h on Apple MPS."
  python3 run_experiments.py "${COMMON[@]}" --stages multiseed --seeds "$SEEDS"
}

stage_audiocross() {
  # Decisive control for the 0.690 / 0.007 analysis: what does a model with no
  # video access score on visually-manipulated clips? Evaluation only, no
  # training -- but it needs an audio-only checkpoint, which the multiseed
  # stage produces (checkpoints/audio_s42.pt).
  local CK="$HERE/checkpoints/audio_s42.pt"
  if [ ! -f "$CK" ]; then
    echo "  ! $CK not found -- run '$0 multiseed' first (it saves this checkpoint)."
    return 0
  fi
  log "Zero-shot: audio-only model on FakeAVCeleb"
  python3 run_experiments.py "${COMMON[@]}" --stages crossdata \
      --eval-cache "favc=$FAVC_CACHE" \
      --crossdata-ckpt "audio=a:$CK"
}

stage_analyse() {
  log "Aggregating seeds"
  python3 aggregate_seeds.py --results "$RESULTS" --out-dir "$HERE" --repo "$REPO"
  log "Per-manipulation analysis with false-positive context"
  python3 analyse_buckets.py --results "$RESULTS" \
      --lavdf-cache "$CACHE" --favc-cache "$FAVC_CACHE" \
      --out "$HERE/bucket_analysis.json"
  log "Measuring native mouth-ROI resolution"
  python3 measure_roi.py --lavdf-root "$LAVDF_ROOT" --favc-root "$FAVC_ROOT" \
      --out "$HERE/roi_native_size.json" || true
  log "Regenerating figures"
  python3 make_figures.py --results "$RESULTS" --out "$HERE/figures"
  log "Regenerating LaTeX macros"
  python3 fill_results.py --results "$RESULTS" \
      --out "$REPO/paper/results_auto.tex"
}

# ==========================================================================
# Revision (IEICE 2026DKP0010) -- assets copied into ../assets by default
# ==========================================================================
ASSETS="${ASSETS:-$HERE/../assets}"
PY="${PY:-$HERE/../.venv/bin/python}"
CKPTS="${CKPTS:-$ASSETS/checkpoints}"
RCACHE="${RCACHE:-$ASSETS/astra_cache}"
RFAVC="${RFAVC:-$ASSETS/favc_cache}"
RTIMIT="${RTIMIT:-$ASSETS/timit_cache}"
RRES="${RRES:-$HERE/../results/results_final.json}"
LRW_CKPT="${LRW_CKPT:-$ASSETS/lrw_resnet18.pth}"
RGB="${RGB:-}"            # set RGB=--rgb-norm to train with RGB/ImageNet-normalised frames

stage_ablseeds() {
  log "T1: every ablation variant over seeds $SEEDS (about 25 runs)"
  "$PY" run_experiments.py --cache "$RCACHE" --repo "$REPO" --out "$RRES" \
      --epochs "$EPOCHS" --batch-size "$BATCH" --workers "$WORKERS" --skip-existing \
      --stages ablseeds --seeds "$SEEDS" $RGB
  mv -n "$(dirname "$RRES")"/checkpoints/*.pt "$CKPTS"/ 2>/dev/null || true
}

stage_lipreadseeds() {
  log "T2: LRW-pretrained lipreading encoder, seeds ${LR_SEEDS:-42,43,44}"
  "$PY" run_experiments.py --cache "$RCACHE" --repo "$REPO" --out "$RRES" \
      --epochs "$EPOCHS" --batch-size "$BATCH" --workers "$WORKERS" --skip-existing \
      --stages lipreadseeds --seeds "${LR_SEEDS:-42,43,44}" --lipread-ckpt "$LRW_CKPT"
  mv -n "$(dirname "$RRES")"/checkpoints/*.pt "$CKPTS"/ 2>/dev/null || true
}

zs_args() {   # arrays, not $(...): paths may contain spaces
  ZS=(); [ -d "$RFAVC" ] && ZS+=(--favc "$RFAVC"); [ -d "$RTIMIT" ] && ZS+=(--dftimit "$RTIMIT")
}

stage_evalsuite() {
  zs_args
  log "Evaluation-only suite over all checkpoints (counterfactuals, zero-shot, attention)"
  "$PY" eval_suite.py --ckpt-dir "$CKPTS" --lavdf "$RCACHE" "${ZS[@]}" \
      --out "$HERE/../results/eval_suite.json"
}

stage_revise() {
  log "Split independence"
  [ -f "$ASSETS/lavdf_metadata.json" ] && "$PY" check_splits.py \
      --metadata "$ASSETS/lavdf_metadata.json" --cache "$RCACHE" || true
  log "Real attention figure"
  [ -f "$CKPTS/full_s42.pt" ] && "$PY" make_attention_figure.py \
      --ckpt "$CKPTS/full_s42.pt" --lavdf "$RCACHE" || true
  log "Revision macros -> paper/results_suite.tex"
  zs_args
  "$PY" analyse_suite.py --lavdf "$RCACHE" "${ZS[@]}" ${HW_TRAIN:+--hw-train "$HW_TRAIN"}
}

case "${1:-all}" in
  cache)      stage_cache ;;
  crossdata)  stage_crossdata_cache ;;
  main)       stage_main ;;
  multiseed)  stage_multiseed ;;
  audiocross) stage_audiocross ;;
  analyse)    stage_analyse ;;
  ablseeds)   stage_ablseeds ;;
  lipreadseeds) stage_lipreadseeds ;;
  evalsuite)  stage_evalsuite ;;
  revise)     stage_revise ;;
  all)        stage_cache; stage_crossdata_cache; stage_main
              stage_multiseed; stage_audiocross; stage_analyse ;;
  *)          echo "usage: $0 {cache|crossdata|main|multiseed|audiocross|analyse|ablseeds|lipreadseeds|evalsuite|revise|all}"
              exit 1 ;;
esac

log "Done."
